"""JEPA predictor over typed SO(3) tokens, without hidden-target coordinates.

Context tokens (visible node/global states of the requested views) and mask
tokens (target view, level, residue position, optional canonical atom slot)
attend jointly through several pre-norm blocks, as in the I-JEPA predictor.
Attention logits are invariant (scalars, norms and a learned relative
sequence-offset bias); values are equivariant per head; a bilinear FFN can
create directions that no single context token carries.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..config import ModelConfig
from .fibers import Fiber, FiberDims, FiberLinear, cat_fibers
from .effdock_blocks import FiberRMSNorm, EquivariantFFN, channel_scale
from .encoders import EncodedView
from ..data.constants import RESIDUE_ATOMS, AA3, ATOM_ID

VIEW_NAMES = ("seq", "bb", "sc", "aa", "bb_internal", "chi")
GEOMETRIC_VIEWS = {"bb", "sc", "aa"}
LEVELS = {"node": 0, "global": 1, "atom": 2}


def _rms(x: Tensor, dims: tuple[int, ...], dof: int) -> Tensor:
    """Per-token RMS per magnetic component (3 for l=1, 5 for l=2)."""
    return (x.square().sum(dims)/dof).mean(-1, keepdim=True).sqrt()


def reference_rms(h: Fiber, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Mean l=1 and l=2 token RMS over valid tokens (zero if none)."""
    return tuple(_rms(x, dims, dof)[valid].mean() if bool(valid.any()) else x.new_zeros(())
                 for x, dims, dof in ((h.v, (-1,), 3), (h.t, (-1, -2), 5)))


def make_target(h: Fiber, valid: Tensor | None = None, floor: float = 0.1,
                eps: float = 1e-6, reference: tuple[Tensor, Tensor] | None = None) -> Fiber:
    """Parameter-free per-token JEPA target (I-JEPA layer-norm analogue).

    Scalars: layer norm without affine. Vectors/tensors: divided by the
    soft per-token RMS sqrt(r^2 + tau^2), with tau = floor x mean token RMS
    of this sample, so near-zero states are not inflated to unit norm and
    zero stays zero. Two extra scalars, log((r+tau)/(mean r+tau)), keep the
    scale-free magnitude that the division removes. All terms are rotation
    invariant, so l>0 targets remain equivariant. A single-token target (the
    global state) must pass `reference` (its crop's node RMS); otherwise its
    own RMS would be the mean and the floor and magnitude would be void.
    """
    valid = torch.ones(len(h.s), dtype=torch.bool, device=h.s.device) if valid is None else valid
    means = reference_rms(h, valid) if reference is None else reference
    s = F.layer_norm(h.s, h.s.shape[-1:], eps=eps)
    out, magnitude = [], []
    for (x, dims, dof), mean in zip(((h.v, (-1,), 3), (h.t, (-1, -2), 5)), means):
        r = _rms(x, dims, dof)
        tau = floor*mean
        scale = (r.square()+tau.square()+eps**2).rsqrt()
        out.append(x*scale.reshape(scale.shape+(1,)*(x.ndim-2)))
        magnitude.append(((r+tau+eps)/(mean+tau+eps)).log())
    return Fiber(torch.cat([s]+magnitude, -1), *out)


@torch.no_grad()
def low_rms_fraction(h: Fiber, valid: Tensor, floor: float = 0.1) -> float:
    """Share of valid tokens whose l=1 RMS is below the soft floor (collapse watch)."""
    r = _rms(h.v, (-1,), 3)[valid]
    if not len(r) or float(r.mean()) == 0:
        return 0.0
    return float((r < floor*r.mean()).float().mean())


@dataclass
class Prediction:
    nodes: Fiber
    global_state: Fiber
    atoms: Fiber | None = None


class EquivariantSelfAttention(nn.Module):
    """Invariant per-head weights transport per-head scalar/vector/tensor values.

    `rows` restricts which tokens are updated (queries); every token is a key.
    """
    def __init__(self, dims: FiberDims, heads: int, relative_positions: int):
        super().__init__()
        if dims.scalar % heads:
            raise ValueError("scalar width must be divisible by predictor heads.")
        self.dims, self.heads, self.radius = dims, heads, relative_positions
        self.q = nn.Linear(dims.invariant, dims.scalar)
        self.k = nn.Linear(dims.invariant, dims.scalar)
        wide = FiberDims(dims.scalar, dims.vector*heads, dims.tensor*heads)
        self.value = FiberLinear(dims, wide)
        self.out = FiberLinear(wide, dims)
        # Buckets: clipped signed offset in [-R, R], plus one for global tokens.
        self.bias = nn.Parameter(torch.zeros(heads, 2*relative_positions+2))

    def forward(self, h: Fiber, position: Tensor, is_global: Tensor,
                rows: slice = slice(None)) -> Fiber:
        n, heads, d = len(h.s), self.heads, self.dims
        inv = h.invariant()
        q = self.q(inv[rows]).view(-1, heads, d.scalar//heads)
        k = self.k(inv).view(n, heads, d.scalar//heads)
        logits = torch.einsum('ihd,jhd->hij', q, k)/math.sqrt(d.scalar//heads)
        offset = (position[None, :]-position[rows, None]).clamp(-self.radius, self.radius)+self.radius
        offset = torch.where(is_global[rows, None] | is_global[None, :], 2*self.radius+1, offset)
        weights = (logits+self.bias[:, offset]).softmax(-1)
        val = self.value(h)
        m = len(q)
        s = torch.einsum('hij,jhd->ihd', weights, val.s.view(n, heads, -1)).reshape(m, d.scalar)
        v = torch.einsum('hij,jhcx->ihcx', weights, val.v.view(n, heads, d.vector, 3))
        t = torch.einsum('hij,jhcxy->ihcxy', weights, val.t.view(n, heads, d.tensor, 3, 3))
        return self.out(Fiber(s, v.reshape(m, heads*d.vector, 3),
                              t.reshape(m, heads*d.tensor, 3, 3)))


class PredictorBlock(nn.Module):
    """Pre-norm attention + bilinear FFN; only `rows` tokens are updated."""
    def __init__(self, dims: FiberDims, heads: int, relative_positions: int, expansion: int):
        super().__init__()
        self.dims = dims
        self.norm1, self.norm2 = FiberRMSNorm(dims), FiberRMSNorm(dims)
        self.attention = EquivariantSelfAttention(dims, heads, relative_positions)
        self.ffn = EquivariantFFN(dims, expansion)
        self.scale1 = nn.Parameter(torch.full((dims.invariant,), 0.1))
        self.scale2 = nn.Parameter(torch.full((dims.invariant,), 0.1))

    def forward(self, h: Fiber, position: Tensor, is_global: Tensor,
                rows: slice = slice(None)) -> Fiber:
        x = h.index(rows)
        x = x+channel_scale(self.attention(self.norm1(h), position, is_global, rows),
                            self.scale1, self.dims)
        return x+channel_scale(self.ffn(self.norm2(x)), self.scale2, self.dims)


def topology_atoms(seq: Tensor, seq_visible: Tensor, residues: Tensor,
                   sidechain_only: bool) -> tuple[Tensor, Tensor]:
    """Canonical (residue, slot) atom queries implied by VISIBLE sequence only.

    A hidden or unknown residue type yields backbone slots only. Whether an
    atom was actually observed never changes the query set.
    """
    out_r, out_a = [], []
    for i in residues.tolist():
        aa = int(seq[i])
        names = (RESIDUE_ATOMS[AA3[aa]] if aa < len(AA3) and bool(seq_visible[i])
                 else ("N", "CA", "C", "O"))
        slots = [ATOM_ID[a] for a in names]
        if sidechain_only:
            slots = [a for a in slots if a >= 4]
        out_r += [i]*len(slots)
        out_a += slots
    device = residues.device
    return (torch.tensor(out_r, dtype=torch.long, device=device),
            torch.tensor(out_a, dtype=torch.long, device=device))


class CrossViewPredictor(nn.Module):
    """Mask tokens carry target view, level, sequence position and (for atoms)
    a canonical slot. They NEVER carry target xyz, frames, edges or states.

    Stage 1 (joint): context + node + global mask tokens, all updated.
    Stage 2 (atom decoder): atom mask tokens attend to stage-1 tokens and to
    each other; stage-1 tokens never see atom tokens, so node/global
    predictions are independent of the atom query set.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dims
        self.dims = d
        self.input_maps = nn.ModuleDict({v: FiberLinear(d, d) for v in VIEW_NAMES})
        self.type_embed = nn.Embedding(len(VIEW_NAMES), d.scalar)
        self.level_embed = nn.Embedding(len(LEVELS), d.scalar)
        self.role_embed = nn.Embedding(2, d.scalar)  # context, mask
        self.mask_embedding = nn.Parameter(torch.randn(d.scalar)*0.02)
        self.atom_query = nn.Embedding(37, d.scalar)

        def stack(n):
            return nn.ModuleList(PredictorBlock(d, cfg.heads, cfg.relative_positions,
                                                cfg.predictor_expansion) for _ in range(n))
        self.blocks = stack(cfg.predictor_layers)
        self.atom_blocks = stack(cfg.atom_decoder_layers)
        self.final_norm = FiberRMSNorm(d)
        # Targets add two log-magnitude scalars (see make_target).
        out = FiberDims(d.scalar+2, d.vector, d.tensor)
        self.heads = nn.ModuleDict({v: FiberLinear(d, out) if v in GEOMETRIC_VIEWS
                                    else nn.Linear(d.scalar, d.scalar+2) for v in VIEW_NAMES})

    def _scalar(self, view: str, level: str, role: int, like: Tensor) -> Tensor:
        index = like.new_tensor
        return (self.type_embed(index(VIEW_NAMES.index(view), dtype=torch.long))
                + self.level_embed(index(LEVELS[level], dtype=torch.long))
                + self.role_embed(index(role, dtype=torch.long)))

    def _head(self, view: str, h: Fiber) -> Fiber:
        head = self.heads[view]
        if isinstance(head, FiberLinear):
            return head(h)
        return Fiber(head(h.s), torch.zeros_like(h.v), torch.zeros_like(h.t))

    def _queries(self, count: int, view: str, level: str) -> Fiber:
        h = Fiber.zeros(count, self.dims, self.mask_embedding)
        h.s = h.s+self.mask_embedding+self._scalar(view, level, 1, self.mask_embedding)
        return h

    def forward(self, context: dict[str, EncodedView], positions: Tensor, target_view: str,
                query_residues: Tensor, atom_residues: Tensor | None = None,
                atom_slots: Tensor | None = None) -> Prediction:
        if target_view not in VIEW_NAMES:
            raise ValueError(f"Invalid target view {target_view!r}.")
        if (atom_residues is None) != (atom_slots is None) or (
                atom_residues is not None and len(atom_residues) != len(atom_slots)):
            raise ValueError("Topology-known atom queries require aligned residues and slots.")
        like = self.mask_embedding
        tokens, pos, glob = [], [], []

        def add(h: Fiber, position: Tensor, is_global: bool):
            tokens.append(h)
            pos.append(position)
            glob.append(torch.full((len(h.s),), is_global, dtype=torch.bool, device=like.device))

        for name, encoded in context.items():
            valid = encoded.node_valid
            if bool(valid.any()):
                h = self.input_maps[name](encoded.nodes.index(valid))
                h.s = h.s+self._scalar(name, "node", 0, like)
                add(h, positions[valid], False)
            if encoded.global_valid:
                h = self.input_maps[name](encoded.global_state)
                h.s = h.s+self._scalar(name, "global", 0, like)
                add(h, positions.new_zeros(1), True)
        n_context = sum(len(h.s) for h in tokens)
        n_nodes = len(query_residues)
        add(self._queries(n_nodes, target_view, "node"), positions[query_residues], False)
        add(self._queries(1, target_view, "global"), positions.new_zeros(1), True)
        h, position, is_global = cat_fibers(tokens), torch.cat(pos), torch.cat(glob)
        for block in self.blocks:
            h = block(h, position, is_global)
        stage1 = self.final_norm(h)
        node = self._head(target_view, stage1.index(slice(n_context, n_context+n_nodes)))
        glob_out = self._head(target_view, stage1.index(slice(n_context+n_nodes, None)))
        atom = None
        if atom_residues is not None:
            q = self._queries(len(atom_residues), target_view, "atom")
            q.s = q.s+self.atom_query(atom_slots)
            n1 = len(h.s)
            position = torch.cat((position, positions[atom_residues]))
            is_global = torch.cat((is_global, torch.zeros(len(q.s), dtype=torch.bool,
                                                          device=like.device)))
            rows = slice(n1, None)
            if len(atom_residues):  # e.g. sc_infill on glycine-only targets: no atoms
                for block in self.atom_blocks:
                    q = block(cat_fibers([h, q]), position, is_global, rows)
            atom = self._head(target_view, self.final_norm(q))
        return Prediction(node, glob_out, atom)
