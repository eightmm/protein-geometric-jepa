"""EFF-Dock-inspired interaction blocks for typed protein JEPA fibers.

Independent SO(3), l<=2 implementation; NOT a checkpoint-compatible EFF-Dock
port. Reference: eightmm/EFF-Dock@52d413d, models/equivariant.py and effdock.py.
The tensor product is the actual CuEq operation when a CuEq backend is selected.
All conditioning uses visible, invariant states; no time/sequence labels enter BB.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fibers import Fiber, FiberDims, FiberLinear
from .equivariant import CartesianMessage
from ..data.graphs import Graph, NUM_EDGE_TYPES


def channel_scale(h: Fiber, scale: Tensor, dims: FiberDims) -> Fiber:
    """Apply one coefficient to ALL magnetic components of each channel."""
    s, v, t = scale.split((dims.scalar, dims.vector, dims.tensor), dim=-1)
    return Fiber(h.s*s, h.v*v[..., None], h.t*t[..., None, None])


class FiberRMSNorm(nn.Module):
    """Per-degree RMS, with Frobenius norm for the five-DOF STF tensor.

    No batch statistics and no Cartesian-component centering. Empty inputs are
    legal. Gains are per channel; zero vector/tensor inputs remain exactly zero.
    """
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
    """Direction-preserving dropout; masks shared across m components."""
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
    """Scalar SiLU + invariant norm-derived non-scalar gates."""
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
    """AdaRMS-like modulation using ONLY a caller-supplied invariant context.

    Zero-init means identity modulation of the normalized state, not identity
    of the whole module. Non-scalar shifts would break equivariance and do not
    exist. The modulation scale is bounded for all degrees.
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
    """Bounded per-degree modulation, without dividing by a near-zero norm.

    Unlike direction/magnitude decomposition with an unconstrained predicted
    magnitude, this cannot create a finite vector out of numerical near-zero
    aggregation. It preserves cancellations and has finite derivatives at zero.
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


def gated_aggregate(h: Fiber, gates: Tensor, dst: Tensor, n: int,
                    dims: FiberDims, mode: str = 'soft') -> Fiber:
    """Per-channel aggregation with explicit attention-mass semantics.

    soft: sum(w*m)/(1+sum(w)); preserves weak/isolated-edge attenuation.
    gate: sum(w*m)/(sum(w)+eps); EFF-Dock-style comparison baseline.
    degree: sum(w*m)/max(degree,1); retains attenuation but degree is discrete.
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
        weighted = x*g[(...,)+extra]
        out = x.new_zeros((n, *x.shape[1:])).index_add(0, dst, weighted)
        parts.append(out/d[(...,)+extra])
    return Fiber(*parts)


class EffDockInteractionBlock(nn.Module):
    """Shared TP + dual radial scale + typed decay + gated residual FFN.

    All atoms/residues use the existing public Fiber interface. EFF-Dock's
    polar/axial O(3) copies are NOT conflated with this model's SO(3) channels;
    checkpoint weights are intentionally incompatible. Edge directions follow
    the src->dst convention. The stage code is static, never a masked label.
    """
    def __init__(self, dims: FiberDims, backend: str = 'reference', *,
                 radial_hidden: int = 96, edge_dim: int = 16, cutoff: float = 12.0,
                 dropout: float = 0.0, expansion: int = 2, residual_scale: float = 0.1,
                 aggregation: str = 'soft', stage: int = 0, conditional: bool = True):
        super().__init__()
        if radial_hidden < 1 or edge_dim < 1 or expansion < 1 or cutoff <= 0:
            raise ValueError('Block widths/expansion/cutoff must be positive.')
        if aggregation not in {'soft', 'gate', 'degree'} or not 0 <= stage < 5:
            raise ValueError('Invalid aggregation or encoder stage.')
        if not 0 < residual_scale <= 1:
            raise ValueError('residual_scale must be in (0,1].')
        self.dims, self.cutoff, self.aggregation = dims, float(cutoff), aggregation
        self.stage, self.conditional = stage, conditional
        if backend == 'reference':
            self.message = CartesianMessage(dims)
        elif backend in {'cueq-naive', 'cueq-cuda'}:
            from .cueq_backend import CuEqMessage
            self.message = CuEqMessage(dims, backend)
        else:
            raise ValueError(f'Unknown backend: {backend}')
        self.pre_norm = FiberRMSNorm(dims)
        self.register_buffer('centers', torch.linspace(0, cutoff, 16))
        self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, edge_dim)
        # Geometry + relation only controls the input/output radial scales.
        self.radial = nn.Sequential(nn.Linear(19+edge_dim, radial_hidden), nn.SiLU())
        self.radial_in = nn.Linear(radial_hidden, dims.invariant)
        self.radial_out = nn.Linear(radial_hidden, dims.invariant)
        for m in (self.radial_in, self.radial_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # Content-dependent gates see invariant summaries of both endpoints.
        self.attention = nn.Sequential(
            nn.Linear(radial_hidden+2*dims.invariant, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, dims.invariant))
        self.log_sigma = nn.Embedding(NUM_EDGE_TYPES, 1)
        nn.init.constant_(self.log_sigma.weight, torch.tensor(cutoff/2).log().item())
        self.message_act = NormGate(dims)
        self.rescale = StableNormRescale(dims)
        self.post = FiberLinear(dims, dims)
        self.drop = FiberDropout(dims, dropout)
        self.message_scale = nn.Parameter(torch.full((dims.invariant,), residual_scale))
        self.ffn_scale = nn.Parameter(torch.full((dims.invariant,), residual_scale))
        if conditional:
            self.stage_embedding = nn.Embedding(5, dims.scalar)
            self.context = nn.Sequential(nn.Linear(dims.invariant+1, dims.scalar), nn.SiLU())
            self.ffn_norm = ConditionalFiberNorm(dims, dims.scalar)
        else:
            self.ffn_norm = FiberRMSNorm(dims)
        wider = FiberDims(dims.scalar*expansion, dims.vector*expansion, dims.tensor*expansion)
        self.ffn_up, self.ffn_down = FiberLinear(dims, wider), FiberLinear(wider, dims)
        self.ffn_act = NormGate(wider)

    def forward(self, h: Fiber, graph: Graph) -> Fiber:
        n = len(h.s)
        if n == 0:
            return h
        src, dst = graph.edge_index
        normed = self.pre_norm(h)
        degree = h.s.new_zeros(n)
        if len(src):
            degree = degree.index_add(0, dst, degree.new_ones(len(src)))
            # Graphs created before v0.2 lack node-kind metadata. Those edges
            # intentionally use type 0; geometry/relation channels remain valid.
            types = graph.edge_type
            if types is None:
                types = torch.zeros_like(src)
            radial = (-((graph.distance[:, None]-self.centers)/(self.cutoff/15)).square()).exp()
            trunk = self.radial(torch.cat((radial, graph.relation, self.edge_embedding(types)), -1))
            scaled = channel_scale(normed.index(src), 1+0.5*self.radial_in(trunk).tanh(), self.dims)
            msg = self.message(scaled, graph.direction)
            msg = channel_scale(msg, 1+0.5*self.radial_out(trunk).tanh(), self.dims)
            msg = self.message_act(msg)
            inv = normed.invariant()
            gates = self.attention(torch.cat((trunk, inv[src], inv[dst]), -1)).sigmoid()
            sigma = self.log_sigma(types).clamp(-3, 6).exp()
            decay = (-graph.distance[:, None]/sigma).exp()
            # Covalent/polymer edges survive spatial cutoff. Non-bond spatial
            # edges smoothly vanish at cutoff (top-k membership still discrete).
            envelope = 0.5*(1+torch.cos(torch.pi*(graph.distance/self.cutoff).clamp(0, 1)))
            envelope = torch.where(graph.relation[:, 2] > 0.5, torch.ones_like(envelope), envelope)
            gates = gates*decay*envelope[:, None]
            msg = self.rescale(gated_aggregate(msg, gates, dst, n, self.dims, self.aggregation))
            delta = channel_scale(self.drop(self.post(msg)), self.message_scale, self.dims)
            # No artificial scalar post-bias message on nodes without neighbors.
            h = h + delta.masked(degree > 0)
        if self.conditional:
            context = self.context(torch.cat((normed.invariant(), degree.log1p()[:, None]), -1))
            context = context+self.stage_embedding.weight[self.stage]
            ffn_input = self.ffn_norm(h, context)
        else:
            ffn_input = self.ffn_norm(h)
        delta = self.ffn_down(self.ffn_act(self.ffn_up(ffn_input)))
        return h + channel_scale(self.drop(delta), self.ffn_scale, self.dims)
