"""Equivariant message passing with explicit reference and CuEq backends."""
import torch
from torch import nn, Tensor
from .fibers import Fiber, FiberDims, FiberActivation, scatter_fiber
from ..geometry.primitives import outer_stf, symmetric_traceless
from ..data.graphs import Graph


class RadialFeatures(nn.Module):
    def __init__(self, n=16, maximum=16.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, maximum, n))
        self.width = maximum/(n-1)

    def forward(self, g: Graph):
        rbf = (-((g.distance[:, None]-self.centers)/self.width).square()).exp()
        return torch.cat((rbf, g.relation), dim=-1)


def axial(m: Tensor) -> Tensor:
    """Vector dual of the antisymmetric part of a 3x3 matrix (l=1 content)."""
    return torch.stack((m[..., 1, 2]-m[..., 2, 1], m[..., 2, 0]-m[..., 0, 2],
                        m[..., 0, 1]-m[..., 1, 0]), dim=-1)


def cross(a: Tensor, b: Tensor) -> Tensor:
    """Cross product over the last axis with explicit broadcasting."""
    a, b = torch.broadcast_tensors(a, b)
    return torch.linalg.cross(a, b, dim=-1)


def mix(linear: nn.Linear, x: Tensor) -> Tensor:
    """Channel mixing with one weight for every Cartesian component."""
    return torch.einsum('ab,nb...->na...', linear.weight, x)


class CartesianMessage(nn.Module):
    """Analytic SO(3) contractions of h_src with Y_0..2(direction).

    Every Clebsch-Gordan path of (0+1+2) x (0+1+2) -> (0+1+2) is present, the
    same path set as CuEq's FullyConnectedTensorProduct, including parity-odd
    cross products. Normalization and weights are NOT identical to CuEq.
    """
    def __init__(self, dims: FiberDims):
        super().__init__()
        s, v, t = dims.scalar, dims.vector, dims.tensor
        self.ss, self.vs, self.ts = nn.Linear(s, s), nn.Linear(v, s, False), nn.Linear(t, s, False)
        self.vv, self.sv, self.tv = nn.Linear(v, v, False), nn.Linear(s, v, False), nn.Linear(t, v, False)
        self.cross = nn.Linear(v, v, False)
        self.qv, self.tqv = nn.Linear(v, v, False), nn.Linear(t, v, False)    # 1x2->1, 2x2->1
        self.tt, self.st, self.vt = nn.Linear(t, t, False), nn.Linear(s, t, False), nn.Linear(v, t, False)
        self.tct, self.vqt, self.tqt = (nn.Linear(t, t, False), nn.Linear(v, t, False),
                                        nn.Linear(t, t, False))           # 2x1, 1x2, 2x2 -> 2

    def forward(self, h: Fiber, unit: Tensor):
        u = unit[:, None, :]
        quad = outer_stf(unit)[:, None]
        s = self.ss(h.s) + self.vs((h.v*u).sum(-1)) + self.ts((h.t*quad).sum((-1, -2)))
        tq = mix(self.tqv, h.t) @ quad
        v = (mix(self.vv, h.v) + self.sv(h.s)[..., None]*u
             + torch.einsum('nbij,nj->nbi', mix(self.tv, h.t), unit)
             + cross(mix(self.cross, h.v), u)
             + (quad @ mix(self.qv, h.v)[..., None])[..., 0] + axial(tq))
        tc = mix(self.tct, h.t)
        t = (mix(self.tt, h.t) + self.st(h.s)[..., None, None]*quad
             + outer_stf(mix(self.vt, h.v), u)
             + symmetric_traceless(cross(tc, u[..., None, :]))
             + outer_stf(cross(mix(self.vqt, h.v), u), u)
             + symmetric_traceless(mix(self.tqt, h.t) @ quad))
        return Fiber(s, v, t)


class EquivariantBlock(nn.Module):
    def __init__(self, dims: FiberDims, backend="reference"):
        super().__init__()
        if backend == "reference":
            self.message = CartesianMessage(dims)
        elif backend in {"cueq-naive", "cueq-cuda"}:
            from .cueq_backend import CuEqMessage
            self.message = CuEqMessage(dims, backend)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.radial = RadialFeatures()
        self.gates = nn.Sequential(nn.Linear(19, dims.scalar), nn.SiLU(),
                                   nn.Linear(dims.scalar, dims.scalar+dims.vector+dims.tensor))
        self.activation = FiberActivation(dims)
        self.dims = dims

    def forward(self, h: Fiber, graph: Graph):
        src, dst = graph.edge_index
        if src.numel():
            msg = self.message(h.index(src), graph.direction)
            gs, gv, gt = self.gates(self.radial(graph)).sigmoid().split(
                (self.dims.scalar, self.dims.vector, self.dims.tensor), dim=-1)
            msg = Fiber(msg.s*gs, msg.v*gv[..., None], msg.t*gt[..., None, None])
            pooled = scatter_fiber(msg, dst, len(h.s))
            h = h + pooled.scale(0.5)
        return self.activation(h)
