"""SO(3) features: scalars, Cartesian vectors, symmetric-traceless rank-2 tensors.

The rank-2 tensor occupies 3x3 storage but exactly five mathematical DOF. This
reference layout is not confused with cuEquivariance's five-component basis.
"""
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from ..geometry.primitives import symmetric_traceless, outer_stf
from ..data.batch import take


@dataclass(frozen=True)
class FiberDims:
    scalar: int = 64
    vector: int = 16
    tensor: int = 8

    @property
    def invariant(self):
        return self.scalar + self.vector + self.tensor


@dataclass
class Fiber:
    s: Tensor
    v: Tensor
    t: Tensor

    @classmethod
    def zeros(cls, n: int, dims: FiberDims, like: Tensor):
        return cls(like.new_zeros(n, dims.scalar), like.new_zeros(n, dims.vector, 3),
                   like.new_zeros(n, dims.tensor, 3, 3))

    def index(self, index):
        return Fiber(take(self.s, index), take(self.v, index), take(self.t, index))

    def detach(self):
        return Fiber(self.s.detach(), self.v.detach(), self.t.detach())

    def masked(self, mask: Tensor):
        return Fiber(self.s*mask[..., None], self.v*mask[..., None, None],
                     self.t*mask[..., None, None, None])

    def invariant(self):
        vn = (self.v.square().sum(-1) + 1e-8).sqrt()
        tn = (self.t.square().sum((-1, -2)) + 1e-8).sqrt()
        return torch.cat((self.s, vn, tn), dim=-1)

    def rotate(self, r: Tensor):
        return Fiber(self.s, self.v @ r.T, torch.einsum('ij,...jk,lk->...il', r, self.t, r))

    def __add__(self, other):
        return Fiber(self.s+other.s, self.v+other.v, self.t+other.t)

    def scale(self, value: float):
        return Fiber(self.s*value, self.v*value, self.t*value)


def cat_fibers(values: list[Fiber], dim=0):
    return Fiber(*(torch.cat([getattr(x, field) for x in values], dim=dim)
                   for field in ("s", "v", "t")))


def scatter_fiber(h: Fiber, index: Tensor, n: int, weights: Tensor | None = None,
                  mean: bool = True):
    count = h.s.new_zeros(n)
    w = h.s.new_ones(len(index)) if weights is None else weights
    count.index_add_(0, index, w)
    output = []
    for x in (h.s, h.v, h.t):
        out = x.new_zeros((n, *x.shape[1:]))
        out.index_add_(0, index, x*w.reshape((-1,) + (1,)*(x.ndim-1)))
        if mean:
            out = out / count.clamp_min(1e-8).reshape((-1,) + (1,)*(x.ndim-1))
        output.append(out)
    return Fiber(*output)


class FiberLinear(nn.Module):
    def __init__(self, di: FiberDims, do: FiberDims, *, scalar_bias: bool = True):
        super().__init__()
        self.s = nn.Linear(di.scalar, do.scalar, bias=scalar_bias)
        self.v = nn.Linear(di.vector, do.vector, bias=False)
        self.t = nn.Linear(di.tensor, do.tensor, bias=False)

    def forward(self, h: Fiber):
        return Fiber(self.s(h.s), torch.einsum('oi,...ij->...oj', self.v.weight, h.v),
                     torch.einsum('oi,...ijk->...ojk', self.t.weight, h.t))


class FiberActivation(nn.Module):
    def __init__(self, dims: FiberDims):
        super().__init__()
        self.norm = nn.LayerNorm(dims.scalar)
        self.invariant_mix = nn.Linear(dims.vector+dims.tensor, dims.scalar, bias=False)
        self.gates = nn.Linear(dims.scalar, dims.vector+dims.tensor)
        self.dims = dims

    def forward(self, h: Fiber):
        inv = h.invariant()[:, self.dims.scalar:]
        s = torch.nn.functional.silu(self.norm(h.s+self.invariant_mix(inv)))
        gv, gt = self.gates(s).sigmoid().split((self.dims.vector, self.dims.tensor), -1)
        # Per-sample isotropic scaling: never normalize Cartesian components independently.
        vn = (1+h.v.square().sum(-1).mean(-1)).sqrt()
        tn = (1+h.t.square().sum((-1, -2)).mean(-1)).sqrt()
        return Fiber(s, h.v*gv[..., None]/vn[:, None, None],
                     h.t*gt[..., None, None]/tn[:, None, None, None])


class DirectionSeed(nn.Module):
    def __init__(self, n_directions: int, dims: FiberDims):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dims.vector, n_directions)*0.1)
        self.t = nn.Parameter(torch.randn(dims.tensor, n_directions)*0.1)

    def forward(self, s: Tensor, directions: Tensor) -> Fiber:
        return Fiber(s, torch.einsum('cv,nvd->ncd', self.v, directions),
                     torch.einsum('cv,nvij->ncij', self.t, outer_stf(directions)))


class GlobalReadout(nn.Module):
    """Invariant learned attention; l>0 is generated ONLY from input geometry."""
    def __init__(self, dims: FiberDims, mean: bool = False):
        super().__init__()
        self.mean = mean   # uniform weights: the mean-pooling ablation of the learned slot
        # No final bias: softmax is shift-invariant, so it could never train.
        self.score = nn.Sequential(nn.Linear(dims.invariant, dims.scalar), nn.SiLU(),
                                   nn.Linear(dims.scalar, 1, bias=False))
        self.scalar_query = nn.Parameter(torch.zeros(dims.scalar))
        self.mix = FiberLinear(dims, dims)
        self.dims = dims

    def forward(self, h: Fiber, valid: Tensor, batch: Tensor | None = None, size: int = 1) -> Fiber:
        """One readout per record (`batch` = record of each node); a record
        without valid nodes reads out the bare query."""
        batch = torch.zeros(len(h.s), dtype=torch.long, device=h.s.device) if batch is None else batch
        index = torch.where(valid)[0]
        record = batch[index]
        selected = h.index(index)
        score = (selected.s.new_zeros(len(index)) if self.mean
                 else self.score(selected.invariant()).squeeze(-1))
        top = score.new_full((size,), -torch.inf).scatter_reduce(0, record, score.detach(), 'amax')
        weight = (score-top.index_select(0, record)).exp()
        weight = weight/weight.new_zeros(size).index_add(0, record, weight).index_select(0, record)
        out = scatter_fiber(selected, record, size, weights=weight)
        out.s = out.s+self.scalar_query
        out = self.mix(out)
        empty = (torch.bincount(record, minlength=size) == 0)
        out.s = torch.where(empty[:, None], self.scalar_query.expand(size, -1), out.s)
        out.v = out.v.masked_fill(empty[:, None, None], 0)
        out.t = out.t.masked_fill(empty[:, None, None, None], 0)
        return out
