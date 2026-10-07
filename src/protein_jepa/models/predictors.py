"""Type-conditional latent prediction without hidden-target coordinates."""
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from ..config import ModelConfig
from .fibers import Fiber, FiberDims, FiberLinear, cat_fibers
from .encoders import EncodedView, position_encoding

VIEW_NAMES = ("seq", "bb", "sc", "aa", "bb_internal", "chi")
GEOMETRIC_VIEWS = {"bb", "sc", "aa"}
PERIODIC_VIEWS = {"bb_internal", "chi"}


@dataclass
class Latent:
    fiber: Fiber
    circular: Tensor  # [N,K,2], contextual circles, NOT raw physical torsion slots

    def detach(self):
        return Latent(self.fiber.detach(), self.circular.detach())

    def index(self, index):
        return Latent(self.fiber.index(index), self.circular[index])


class LatentProjector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.linear = FiberLinear(cfg.dims, cfg.latent_dims)
        self.scalar = nn.Sequential(nn.Linear(cfg.dims.invariant, cfg.scalar), nn.SiLU(),
                                    nn.Linear(cfg.scalar, cfg.latent_scalar))
        self.circle = nn.Linear(cfg.dims.invariant, 2*cfg.circular_channels)
        self.ncircles = cfg.circular_channels

    def forward(self, h: Fiber):
        inv = h.invariant()
        z = self.linear(h)
        z.s = self.scalar(inv)
        circle = self.circle(inv).reshape(-1, self.ncircles, 2)
        # Exactly unit-valued, with a defined fallback for zero numerical output.
        norm = circle.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(circle)
        fallback[..., 0] = 1
        circle = torch.where(norm > 1e-8, circle/norm.clamp_min(1e-8), fallback)
        return Latent(z, circle)


class CrossViewPredictor(nn.Module):
    """Invariant attention coefficients transport equivariant values.

    Target queries have sequence position, target-view type and level. Atom
    queries additionally identify a canonical atom slot (topology-known task).
    They NEVER have target xyz, target frames, hidden edges or sidechain states.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.type_embed = nn.Embedding(len(VIEW_NAMES), cfg.scalar)
        self.level_embed = nn.Embedding(3, cfg.scalar)  # node, global, atom
        self.atom_query = nn.Embedding(37, cfg.scalar)
        self.input_proj = nn.ModuleDict({v: nn.Linear(cfg.dims.invariant, cfg.scalar) for v in VIEW_NAMES})
        self.attn = nn.MultiheadAttention(cfg.scalar, cfg.heads, cfg.dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.scalar), nn.LayerNorm(cfg.scalar)
        self.ff = nn.Sequential(nn.Linear(cfg.scalar, 4*cfg.scalar), nn.GELU(), nn.Linear(4*cfg.scalar, cfg.scalar))
        self.value_maps = nn.ModuleDict({v: FiberLinear(cfg.dims, cfg.dims) for v in VIEW_NAMES})
        self.null = nn.Parameter(torch.zeros(1, cfg.scalar))
        self.v_gate = nn.Linear(cfg.scalar, cfg.vector)
        self.t_gate = nn.Linear(cfg.scalar, cfg.tensor)
        self.heads = nn.ModuleDict({v: LatentProjector(cfg) for v in VIEW_NAMES})

    def forward(self, context: dict[str, EncodedView], context_positions: Tensor,
                target_view: str, query_positions: Tensor,
                level: str = "node", atom_slots: Tensor | None = None) -> Latent:
        if target_view not in VIEW_NAMES or level not in {"node", "global", "atom"}:
            raise ValueError("Invalid query type.")
        device = context_positions.device
        keys, values = [], []
        for name, encoded in context.items():
            valid = encoded.node_valid
            type_id = torch.tensor(VIEW_NAMES.index(name), device=device)
            if bool(valid.any()):
                h = encoded.nodes.index(valid)
                key = self.input_proj[name](h.invariant()) + self.type_embed(type_id)
                key = key + position_encoding(context_positions[valid], self.cfg.scalar)
                keys.append(key)
                values.append(self.value_maps[name](h))
            if encoded.global_valid:
                h = encoded.global_state
                keys.append(self.input_proj[name](h.invariant()) + self.type_embed(type_id))
                values.append(self.value_maps[name](h))
        # Scalar null guarantees defined attention for an empty observation.
        keys.append(self.null)
        values.append(Fiber.zeros(1, self.cfg.dims, self.null))
        key = torch.cat(keys)
        val = cat_fibers(values)
        q = position_encoding(query_positions, self.cfg.scalar)
        q = q + self.type_embed(torch.tensor(VIEW_NAMES.index(target_view), device=device))
        q = q + self.level_embed(torch.tensor({"node": 0, "global": 1, "atom": 2}[level], device=device))
        if level == "atom":
            if atom_slots is None or len(atom_slots) != len(q):
                raise ValueError("Topology-known atom queries require aligned atom slots.")
            q = q+self.atom_query(atom_slots)
        if len(q) == 0:
            h = Fiber.zeros(0, self.cfg.dims, q)
            return self.heads[target_view](h)
        h, weights = self.attn(q[None], key[None], key[None], need_weights=True)
        s = self.norm1(q+h[0])
        s = self.norm2(s+self.ff(s))
        w = weights[0]
        v = torch.einsum('qn,ncj->qcj', w, val.v)*self.v_gate(s).sigmoid()[..., None]
        t = torch.einsum('qn,ncij->qcij', w, val.t)*self.t_gate(s).sigmoid()[..., None, None]
        return self.heads[target_view](Fiber(s, v, t))
