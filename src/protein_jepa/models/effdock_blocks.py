"""EFF-Dock-inspired SO(3) interaction blocks for protein JEPA.

Independent l<=2 implementation, not an EFF-Dock checkpoint port. Reference:
eightmm/EFF-Dock@52d413d, models/equivariant.py and models/effdock.py.
CuEq backends execute an actual FullyConnectedTensorProduct. The reference
operator is an analytic alternative, NOT a weight-identical CuEq fallback.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fibers import Fiber, FiberDims, FiberLinear
from ..geometry.primitives import outer_stf, symmetric_traceless
from .equivariant import CartesianMessage
from ..data.graphs import Graph, NUM_EDGE_TYPES


def channel_scale(h: Fiber, scale: Tensor, dims: FiberDims) -> Fiber:
    """One coefficient for ALL magnetic components of each channel."""
    s, v, t = scale.split((dims.scalar, dims.vector, dims.tensor), dim=-1)
    return Fiber(h.s*s, h.v*v[..., None], h.t*t[..., None, None])


class FiberRMSNorm(nn.Module):
    """Per-degree RMS with per-channel gains; no componentwise centering."""
    def __init__(self, dims: FiberDims, eps: float = 1e-6):
        super().__init__()
        self.dims, self.eps = dims, eps
        self.gain = nn.Parameter(torch.ones(dims.invariant))

    def forward(self, h: Fiber) -> Fiber:
        sr = (h.s.square().mean(-1, keepdim=True)+self.eps).rsqrt()
        vr = (h.v.square().sum(-1).mean(-1, keepdim=True)+self.eps).rsqrt()
        tr = (h.t.square().sum((-1, -2)).mean(-1, keepdim=True)+self.eps).rsqrt()
        out = Fiber(h.s*sr, h.v*vr[..., None], h.t*tr[..., None, None])
        return channel_scale(out, self.gain, self.dims)


class FiberDropout(nn.Module):
    """Bernoulli masks shared across every m component of a channel."""
    def __init__(self, dims: FiberDims, p: float = 0.0):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError('dropout must be in [0, 1).')
        self.dims, self.p = dims, float(p)

    def forward(self, h: Fiber) -> Fiber:
        if not self.training or self.p == 0:
            return h
        mask = F.dropout(h.s.new_ones(len(h.s), self.dims.invariant), self.p, True)
        return channel_scale(h, mask, self.dims)


class NormGate(nn.Module):
    """Scalar SiLU plus invariant norm-derived non-scalar gates."""
    def __init__(self, dims: FiberDims):
        super().__init__()
        self.dims = dims
        self.gates = nn.Sequential(nn.Linear(dims.invariant, dims.scalar), nn.SiLU(),
                                   nn.Linear(dims.scalar, dims.vector+dims.tensor))
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.constant_(self.gates[-1].bias, 2.0)

    def forward(self, h: Fiber) -> Fiber:
        v, t = self.gates(h.invariant()).sigmoid().split(
            (self.dims.vector, self.dims.tensor), -1)
        return Fiber(F.silu(h.s), h.v*v[..., None], h.t*t[..., None, None])


class ConditionalFiberNorm(nn.Module):
    """AdaRMS-like modulation using visible invariant context only.

    Zero-init is identity MODULATION of the normalized state, not identity of
    the whole module. Non-scalar shifts are forbidden. All scales are bounded.
    """
    def __init__(self, dims: FiberDims, context_dim: int):
        super().__init__()
        self.dims = dims
        self.norm = FiberRMSNorm(dims)
        self.modulate = nn.Linear(context_dim, dims.invariant+dims.scalar)
        nn.init.zeros_(self.modulate.weight)
        nn.init.zeros_(self.modulate.bias)

    def forward(self, h: Fiber, context: Tensor) -> Fiber:
        scale, shift = self.modulate(context).split((self.dims.invariant, self.dims.scalar), -1)
        out = channel_scale(self.norm(h), 1+0.1*scale.tanh(), self.dims)
        return Fiber(out.s+shift, out.v, out.t)


class StableNormRescale(nn.Module):
    """Bounded degree-wise gates without dividing by near-zero vector norms.

    Zero stays zero, cancellations survive, and derivatives remain finite at
    zero. This deliberately differs from direction/predicted-magnitude splits.
    """
    def __init__(self, dims: FiberDims):
        super().__init__()
        self.v_map = nn.Linear(dims.vector, dims.vector)
        self.t_map = nn.Linear(dims.tensor, dims.tensor)
        for m in (self.v_map, self.t_map):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h: Fiber) -> Fiber:
        vn = (h.v.square().sum(-1)+1e-8).sqrt()
        tn = (h.t.square().sum((-1, -2))+1e-8).sqrt()
        return Fiber(h.s, h.v*(1+0.1*self.v_map(vn).tanh())[..., None],
                     h.t*(1+0.1*self.t_map(tn).tanh())[..., None, None])


def axis_projections(h: Fiber, unit: Tensor) -> Tensor:
    """Per-channel v.r and r^T T r: invariants that keep angular information."""
    quad = outer_stf(unit)[:, None]
    return torch.cat(((h.v*unit[:, None]).sum(-1), (h.t*quad).sum((-1, -2))), -1)


class GatedFFN(nn.Module):
    """v0.2 FFN: channel mixing and norm gates only, no cross-degree products."""
    def __init__(self, dims: FiberDims, expansion: int = 2):
        super().__init__()
        wide = FiberDims(dims.scalar*expansion, dims.vector*expansion, dims.tensor*expansion)
        self.up, self.down = FiberLinear(dims, wide), FiberLinear(wide, dims)
        self.act = NormGate(wide)

    def forward(self, h: Fiber) -> Fiber:
        return self.down(self.act(self.up(h)))


class EquivariantFFN(nn.Module):
    """Bilinear SO(3) FFN: scalars see channel dot products, l>0 can be created.

    Products between linear views of a pre-normalized input couple degrees
    (v.v', T:T' -> l=0; v x v', T v' -> l=1; v (x) v', {T T'} -> l=2). Every
    product vanishes when its l>0 inputs vanish, so a state without geometry
    never acquires a world-frame vector. Callers pre-normalize the input and
    scale the residual; products enter with a learnable 0.1 initial weight.
    """
    def __init__(self, dims: FiberDims, expansion: int = 2):
        super().__init__()
        w = FiberDims(dims.scalar*expansion, dims.vector*expansion, dims.tensor*expansion)
        self.wide = w
        self.a = FiberLinear(dims, w)
        self.b = FiberLinear(dims, w, scalar_bias=False)
        self.tv = nn.Linear(dims.vector, w.tensor, bias=False)  # partner of T v
        self.scalar = nn.Linear(w.scalar+2*w.vector+2*w.tensor, w.scalar)
        self.gates = nn.Linear(w.scalar, w.vector+w.tensor)
        self.product_scale = nn.Parameter(torch.full((4,), 0.1))
        self.down = FiberLinear(FiberDims(w.scalar, 2*w.vector+w.tensor, 2*w.tensor+w.vector), dims)

    def forward(self, h: Fiber) -> Fiber:
        a, b = self.a(h), self.b(h)
        partner = torch.einsum('ab,nbj->naj', self.tv.weight, h.v)
        invariants = (a.s, (a.v.square().sum(-1)+1e-8).sqrt(),
                      (a.t.square().sum((-1, -2))+1e-8).sqrt(),
                      (a.v*b.v).sum(-1), (a.t*b.t).sum((-1, -2)))
        s = F.silu(self.scalar(torch.cat(invariants, -1)))
        gv, gt = self.gates(s).sigmoid().split((self.wide.vector, self.wide.tensor), -1)
        c = self.product_scale
        v = torch.cat((a.v*gv[..., None], c[0]*torch.linalg.cross(a.v, b.v, dim=-1),
                       c[1]*(a.t@partner[..., None])[..., 0]), 1)
        t = torch.cat((a.t*gt[..., None, None], c[2]*outer_stf(a.v, b.v),
                       c[3]*symmetric_traceless(a.t@b.t)), 1)
        return self.down(Fiber(s, v, t))


def gated_aggregate(h: Fiber, gates: Tensor, dst: Tensor, n: int,
                    dims: FiberDims, mode: str = 'soft') -> Fiber:
    """Per-channel aggregation with explicit attention-mass semantics.

    soft: sum(w*m)/(1+sum(w)); preserves weak/isolated-edge attenuation.
    gate: sum(w*m)/(sum(w)+eps); EFF-Dock-style comparison baseline.
    degree: sum(w*m)/max(degree,1); retains attenuation, discrete denominator.
    """
    if mode not in {'soft', 'gate', 'degree'}:
        raise ValueError(f'Unknown aggregation: {mode}')
    denom = gates.new_zeros(n, dims.invariant).index_add(0, dst, gates)
    if mode == 'soft':
        denom = 1+denom
    elif mode == 'gate':
        denom = denom+1e-6
    else:
        degree = gates.new_zeros(n).index_add(0, dst, gates.new_ones(len(dst)))
        denom = degree.clamp_min(1)[:, None].expand(-1, dims.invariant)
    parts = []
    for x, g, d in zip((h.s, h.v, h.t),
                       gates.split((dims.scalar, dims.vector, dims.tensor), -1),
                       denom.split((dims.scalar, dims.vector, dims.tensor), -1)):
        extra = (None,)*(x.ndim-2)
        out = x.new_zeros((n, *x.shape[1:])).index_add(0, dst, x*g[(...,)+extra])
        parts.append(out/d[(...,)+extra])
    return Fiber(*parts)


class EffDockInteractionBlock(nn.Module):
    """Shared TP, dual radial scales, typed decay, gated residual FFN.

    No coordinates are changed. EFF-Dock O(3) parity copies/force heads are not
    imported into these SO(3) fibers. Stage IDs are static architecture labels;
    no target label, full-structure statistic or flow time conditions BB.
    """
    def __init__(self, dims: FiberDims, backend: str = 'reference', *,
                 radial_hidden: int = 96, cutoff: float = 12.0,
                 dropout: float = 0.0, expansion: int = 2, residual_scale: float = 0.1,
                 aggregation: str = 'soft', stage: int = 0, conditional: bool = True,
                 dual_radial: bool = True, distance_decay: bool = True,
                 norm_rescale: bool = True, smooth_cutoff: bool = True,
                 directional: bool = False, ffn: str = 'gate',
                 adaptive_cutoff: bool = False):
        super().__init__()
        if radial_hidden < 1 or expansion < 1 or cutoff <= 0:
            raise ValueError('Block widths/expansion/cutoff must be positive.')
        if aggregation not in {'soft', 'gate', 'degree'} or not 0 <= stage < 5:
            raise ValueError('Invalid aggregation or encoder stage.')
        if not 0 < residual_scale <= 1:
            raise ValueError('residual_scale must be in (0,1].')
        if ffn not in {'bilinear', 'gate'}:
            raise ValueError(f'Unknown effdock FFN: {ffn}')
        if adaptive_cutoff and aggregation == 'degree':
            raise ValueError('adaptive_cutoff needs soft/gate aggregation: a discrete degree '
                             'denominator jumps when a bonded neighbour leaves the top-k set.')
        self.dims, self.cutoff, self.aggregation = dims, float(cutoff), aggregation
        self.stage, self.conditional = stage, conditional
        self.dual_radial, self.distance_decay = dual_radial, distance_decay
        self.smooth_cutoff, self.directional = smooth_cutoff, directional
        self.adaptive_cutoff = adaptive_cutoff
        if backend == 'reference':
            self.message = CartesianMessage(dims)
        elif backend in {'cueq-naive', 'cueq-cuda'}:
            from .cueq_backend import CuEqMessage
            self.message = CuEqMessage(dims, backend)
        else:
            raise ValueError(f'Unknown backend: {backend}')
        self.pre_norm = FiberRMSNorm(dims)
        self.register_buffer('centers', torch.linspace(0, cutoff, 16))
        self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, 16)
        self.radial = nn.Sequential(nn.Linear(19+16, radial_hidden), nn.SiLU())
        if dual_radial:
            self.radial_in = nn.Linear(radial_hidden, dims.invariant)
            self.radial_out = nn.Linear(radial_hidden, dims.invariant)
            for m in (self.radial_in, self.radial_out):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        # Optional source/destination projections onto the edge axis (v.r, r^T T r):
        # rotation invariants that norms alone cannot express.
        directional_dim = 2*(dims.vector+dims.tensor) if directional else 0
        self.attention = nn.Sequential(
            nn.Linear(radial_hidden+2*dims.invariant+directional_dim, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, dims.invariant))
        if distance_decay:
            self.log_sigma = nn.Embedding(NUM_EDGE_TYPES, 1)
            nn.init.constant_(self.log_sigma.weight, torch.tensor(cutoff/2).log().item())
        self.message_act = NormGate(dims)
        self.rescale = StableNormRescale(dims) if norm_rescale else nn.Identity()
        # No scalar post-bias: a vanishing edge must produce a vanishing message.
        self.post = FiberLinear(dims, dims, scalar_bias=False)
        self.drop = FiberDropout(dims, dropout)
        self.message_scale = nn.Parameter(torch.full((dims.invariant,), residual_scale))
        self.ffn_scale = nn.Parameter(torch.full((dims.invariant,), residual_scale))
        # Each block owns its weights, so no stage embedding is needed: a
        # per-block constant would only duplicate the modulation bias.
        if conditional:
            self.context = nn.Sequential(nn.Linear(dims.invariant+1, dims.scalar), nn.SiLU())
            self.ffn_norm = ConditionalFiberNorm(dims, dims.scalar)
        else:
            self.ffn_norm = FiberRMSNorm(dims)
        self.ffn = EquivariantFFN(dims, expansion) if ffn == 'bilinear' else GatedFFN(dims, expansion)

    def forward(self, h: Fiber, graph: Graph) -> Fiber:
        n = len(h.s)
        if n == 0:
            return h
        src, dst = graph.edge_index
        normed = self.pre_norm(h)
        # Smooth weighted mass, NOT discrete degree, conditions the FFN.
        mass = h.s.new_zeros(n)
        if len(src):
            types = graph.edge_type
            if types is None:
                types = torch.zeros_like(src)
            radial = (-((graph.distance[:, None]-self.centers)/(self.cutoff/15)).square()).exp()
            trunk = self.radial(torch.cat((radial, graph.relation, self.edge_embedding(types)), -1))
            source = normed.index(src)
            if self.dual_radial:
                source = channel_scale(source, 1+0.5*self.radial_in(trunk).tanh(), self.dims)
            msg = self.message(source, graph.direction)
            if self.dual_radial:
                msg = channel_scale(msg, 1+0.5*self.radial_out(trunk).tanh(), self.dims)
            msg = self.message_act(msg)
            inv = normed.invariant()
            features = [trunk, inv[src], inv[dst]]
            if self.directional:
                features += [axis_projections(normed.index(src), graph.direction),
                             axis_projections(normed.index(dst), graph.direction)]
            gates = self.attention(torch.cat(features, -1)).sigmoid()
            if self.distance_decay:
                sigma = self.log_sigma(types).clamp(-3, 6).exp()
                gates = gates*(-graph.distance[:, None]/sigma).exp()
            if self.smooth_cutoff:
                cutoff = (graph.cutoff.clamp_max(self.cutoff)
                          if self.adaptive_cutoff and graph.cutoff is not None else self.cutoff)
                envelope = 0.5*(1+torch.cos(torch.pi*(graph.distance/cutoff).clamp(0, 1)))
                # Explicit bonds survive the spatial cutoff; they remain typed.
                envelope = torch.where(graph.relation[:, 2] > 0.5,
                                       torch.ones_like(envelope), envelope)
                gates = gates*envelope[:, None]
            mass = mass.index_add(0, dst, gates.mean(-1))
            msg = self.rescale(gated_aggregate(msg, gates, dst, n, self.dims, self.aggregation))
            h = h + channel_scale(self.drop(self.post(msg)), self.message_scale, self.dims)
        if self.conditional:
            context = self.context(torch.cat((normed.invariant(), mass.log1p()[:, None]), -1))
            ffn_input = self.ffn_norm(h, context)
        else:
            ffn_input = self.ffn_norm(h)
        delta = self.ffn(ffn_input)
        return h + channel_scale(self.drop(delta), self.ffn_scale, self.dims)
