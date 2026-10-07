"""Equivariant message passing with explicit reference and CuEq backends."""
import torch
from torch import nn, Tensor
from .fibers import Fiber, FiberDims, FiberActivation, scatter_fiber
from ..geometry.primitives import outer_stf
from ..data.graphs import Graph


class RadialFeatures(nn.Module):
    def __init__(self, n=16, maximum=16.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, maximum, n))
        self.width = maximum/(n-1)

    def forward(self, g: Graph):
        rbf = (-((g.distance[:, None]-self.centers)/self.width).square()).exp()
        return torch.cat((rbf, g.relation), dim=-1)


class CartesianMessage(nn.Module):
    """Analytic SO(3) contractions; includes parity-odd cross products.

    This is a correctness/reference architecture, NOT a weight-identical CuEq
    tensor product. It supports l=0,1,2 without e3nn or CUDA dependencies.
    """
    def __init__(self, dims: FiberDims):
        super().__init__()
        s, v, t = dims.scalar, dims.vector, dims.tensor
        self.ss, self.vs, self.ts = nn.Linear(s, s), nn.Linear(v, s, False), nn.Linear(t, s, False)
        self.vv, self.sv, self.tv = nn.Linear(v, v, False), nn.Linear(s, v, False), nn.Linear(t, v, False)
        self.cross = nn.Linear(v, v, False)
        self.tt, self.st, self.vt = nn.Linear(t, t, False), nn.Linear(s, t, False), nn.Linear(v, t, False)

    def forward(self, h: Fiber, unit: Tensor):
        u = unit[:, None, :]
        quad = outer_stf(unit)
        s = self.ss(h.s) + self.vs((h.v*u).sum(-1)) + self.ts((h.t*quad[:, None]).sum((-1, -2)))
        v = (torch.einsum('ab,nbj->naj', self.vv.weight, h.v) + self.sv(h.s)[..., None]*u
             + torch.einsum('ab,nbij,nj->nai', self.tv.weight, h.t, unit)
             + torch.linalg.cross(torch.einsum('ab,nbj->naj', self.cross.weight, h.v), u))
        mixed = torch.einsum('ab,nbj->naj', self.vt.weight, h.v)
        t = (torch.einsum('ab,nbij->naij', self.tt.weight, h.t)
             + self.st(h.s)[..., None, None]*quad[:, None] + outer_stf(mixed, u))
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


def make_geometry_block(cfg, stage: int = 0, atom: bool = False):
    """Keep v0.1 checkpoints/defaults unchanged; opt into the v0.2 architecture."""
    if cfg.geometry_block == "legacy":
        return EquivariantBlock(cfg.dims, cfg.backend)
    from .effdock_blocks import EffDockInteractionBlock
    return EffDockInteractionBlock(
        cfg.dims, cfg.backend, radial_hidden=cfg.eff_radial_hidden,
        edge_dim=cfg.eff_edge_dim, cutoff=cfg.radius_atom if atom else cfg.radius_residue,
        dropout=cfg.dropout, expansion=cfg.eff_expansion,
        residual_scale=cfg.eff_residual_scale, aggregation=cfg.eff_aggregation,
        stage=stage, conditional=cfg.eff_conditional)
