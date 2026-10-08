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
from .latents import TypedLatent, TypedLatentHead, ChannelMix, latent_spec, LatentSpec, safe_sqrt
from ..data.constants import RESIDUE_ATOMS, AA3, ATOM_ID
from ..data.batch import segment_mean, padded_layout, to_padded

VIEW_NAMES = ("seq", "bb", "sc", "aa", "bb_internal", "chi")
LEVELS = {"node": 0, "global": 1, "atom": 2}


def _rms(x: Tensor, dims: tuple[int, ...], dof: int) -> Tensor:
    """Per-token RMS per magnetic component (3 for l=1, 5 for l=2)."""
    return safe_sqrt((x.square().sum(dims)/dof).mean(-1, keepdim=True))


def reference_rms(h: Fiber, valid: Tensor, batch: Tensor | None = None,
                  size: int = 1) -> tuple[Tensor, Tensor]:
    """Mean l=1 and l=2 token RMS over valid tokens (zero if none), per record
    when `batch` is given."""
    index = torch.zeros(len(valid), dtype=torch.long, device=valid.device) if batch is None else batch
    out = tuple(segment_mean(_rms(x, dims, dof)[:, 0], index, size, valid)[0]
                for x, dims, dof in ((h.v, (-1,), 3), (h.t, (-1, -2), 5)))
    return tuple(x[0] for x in out) if batch is None else out


def make_target(h: Fiber, valid: Tensor | None = None, floor: float = 0.1,
                eps: float = 1e-6, reference: tuple[Tensor, Tensor] | None = None,
                instance: bool = False, batch: Tensor | None = None, size: int = 1) -> Fiber:
    """Parameter-free per-token JEPA target (I-JEPA layer-norm analogue).

    Scalars: layer norm without affine. Vectors/tensors: divided by the
    soft per-token RMS sqrt(r^2 + tau^2), with tau = floor x mean token RMS
    of this record, so near-zero states are not inflated to unit norm and
    zero stays zero. Two extra scalars, log((r+tau)/(mean r+tau)), keep the
    scale-free magnitude that the division removes. All terms are rotation
    invariant, so l>0 targets remain equivariant. A single-token target (the
    global state) must pass `reference` (its crop's node RMS); otherwise its
    own RMS would be the mean and the floor and magnitude would be void.

    `instance=True` (node/atom targets) standardizes each scalar channel over
    the record's valid tokens instead (data2vec instance norm). Untrained
    backbone states share one dominant component (about 1% residue-specific
    variance on real proteins), so a per-token norm let mean prediction
    remove nearly all loss without any residue-specific learning.
    Statistics are per record (`batch` = record of each token).
    """
    n = len(h.s)
    valid = torch.ones(n, dtype=torch.bool, device=h.s.device) if valid is None else valid
    batch = torch.zeros(n, dtype=torch.long, device=h.s.device) if batch is None else batch
    means = reference_rms(h, valid, batch, size) if reference is None else reference
    s = F.layer_norm(h.s, h.s.shape[-1:], eps=eps)
    if instance:
        mean, count = segment_mean(h.s, batch, size, valid)
        centred = h.s-mean[batch]
        var = segment_mean(centred.square(), batch, size, valid)[0]
        s = torch.where((count > 1)[batch, None], centred/(var+eps).sqrt()[batch], s)
    out, magnitude = [], []
    for (x, dims, dof), mean in zip(((h.v, (-1,), 3), (h.t, (-1, -2), 5)), means):
        mean = mean.reshape(-1)[batch][:, None]
        r = _rms(x, dims, dof)
        tau = floor*mean
        scale = (r.square()+tau.square()+eps**2).rsqrt()
        out.append(x*scale.reshape(scale.shape+(1,)*(x.ndim-2)))
        magnitude.append(((r+tau+eps)/(mean+tau+eps)).log())
    return Fiber(torch.cat([s]+magnitude, -1), *out)


@torch.no_grad()
def low_rms_fraction(h: Fiber, valid: Tensor, floor: float = 0.1, batch: Tensor | None = None,
                     size: int = 1):
    """Share of valid tokens whose l=1 RMS is below the soft floor (collapse
    watch); a float, or one value per record with `batch`."""
    index = torch.zeros(len(valid), dtype=torch.long, device=valid.device) if batch is None else batch
    r = _rms(h.v, (-1,), 3)[:, 0]
    mean = segment_mean(r, index, size, valid)[0]
    low = segment_mean((r < floor*mean[index]).float(), index, size, valid)[0]
    return float(low[0]) if batch is None else low


RAW_ANGLES = 8   # phi, psi, omega, CA dihedral, chi1-4


@dataclass
class Prediction:
    nodes: TypedLatent
    global_state: TypedLatent
    atoms: Fiber | None = None
    raw_angles: Tensor | None = None       # [Q, 8, 2] unit (cos, sin) per query residue
    raw_coordinate: Tensor | None = None   # [Q, 3] CA offset from the visible-CA centroid / 10


@dataclass
class ContextLatent:
    """Online typed latents of one context view: valid nodes and their
    (packed) residue indices, plus one global latent per record, used only
    where `global_valid` holds."""
    nodes: TypedLatent
    index: Tensor
    global_state: TypedLatent
    global_valid: Tensor


class LatentAdapter(nn.Module):
    """TypedLatent -> predictor Fiber. Circles enter as invariant scalars;
    directions and frame axes enter as l=1 vectors (all rotate correctly)."""
    def __init__(self, spec: LatentSpec, dims: FiberDims):
        super().__init__()
        self.spec = spec
        self.s = nn.Linear(spec.sem+2*spec.circ, dims.scalar)
        self.v = ChannelMix(spec.vector+spec.dir+3*spec.frame, dims.vector)
        self.t = ChannelMix(spec.tensor, dims.tensor)

    def forward(self, z: TypedLatent, global_only: bool = False) -> Fiber:
        n = len(z.sem)
        # Zero padding preserves the node adapter's weights and old checkpoint
        # layout while keeping global inputs strictly semantic + irreps.
        if global_only:
            circles = z.sem.new_zeros(n, 2*self.spec.circ)
            extra_vectors = z.sem.new_zeros(n, self.spec.dir+3*self.spec.frame, 3)
        else:
            circles = z.circ.flatten(1)
            axes = z.frame.transpose(-1, -2).reshape(n, -1, 3)
            extra_vectors = torch.cat((z.dir, axes), 1)
        s = self.s(torch.cat((z.sem, circles), -1))
        return Fiber(s, self.v(torch.cat((z.v, extra_vectors), 1)), self.t(z.t))


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
        # ALiBi-style init (-slope_h * |offset|): mask tokens share identical
        # content, so a zero init gave every query the same attention and the
        # same output (retrieval at chance) until the bias slowly learned.
        offsets = torch.arange(-relative_positions, relative_positions+1).abs().float()
        slopes = torch.tensor([2.0**(-8*(h+1)/heads) for h in range(heads)])
        bias = torch.cat((-slopes[:, None]*offsets, torch.zeros(heads, 1)), 1)
        self.bias = nn.Parameter(bias)

    def forward(self, h: Fiber, position: Tensor, is_global: Tensor, record: Tensor,
                size: int, rows: Tensor | None = None) -> Fiber:
        """Tokens attend only within their record (padded per-record layout).
        `rows` (flat token indices) restricts the queries; outputs follow it."""
        n, heads, d = len(h.s), self.heads, self.dims
        rows = torch.arange(n, device=h.s.device) if rows is None else rows
        m, width = len(rows), d.scalar//heads
        inv = h.invariant()
        q = self.q(inv[rows]).view(m, heads, width)
        k = self.k(inv).view(n, heads, width)
        keys, queries = padded_layout(record, size), padded_layout(record[rows], size)
        kj, qi = keys.clamp_min(0), queries.clamp_min(0)
        logits = torch.einsum('bihd,bjhd->bhij', to_padded(q, queries),
                              to_padded(k, keys))/math.sqrt(width)
        qpos, kpos = position[rows][qi], position[kj]
        offset = (kpos[:, None, :]-qpos[:, :, None]).clamp(-self.radius, self.radius)+self.radius
        glob = is_global[rows][qi][:, :, None] | is_global[kj][:, None, :]
        offset = torch.where(glob, 2*self.radius+1, offset)
        # Embedding lookup: its backward reduces the many pairs sharing one
        # offset bucket efficiently (advanced-index backward serializes them).
        bias = F.embedding(offset, self.bias.T).permute(0, 3, 1, 2)
        logits = (logits+bias).masked_fill(
            (keys < 0)[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = logits.softmax(-1)
        val = self.value(h)
        s = torch.einsum('bhij,bjhd->bihd', weights, to_padded(val.s.view(n, heads, -1), keys))
        v = torch.einsum('bhij,bjhcx->bihcx', weights,
                         to_padded(val.v.view(n, heads, d.vector, 3), keys))
        t = torch.einsum('bhij,bjhcxy->bihcxy', weights,
                         to_padded(val.t.view(n, heads, d.tensor, 3, 3), keys))
        real = queries >= 0
        slots = queries[real]

        def back(x):  # padded query rows -> flat `rows` order
            x = x[real].reshape(len(slots), -1, *x.shape[4:])
            return x.new_zeros((m, *x.shape[1:])).index_copy(0, slots, x)
        return self.out(Fiber(back(s).reshape(m, d.scalar), back(v), back(t)))


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

    def forward(self, h: Fiber, position: Tensor, is_global: Tensor, record: Tensor,
                size: int, rows: Tensor | None = None) -> Fiber:
        x = h if rows is None else h.index(rows)
        x = x+channel_scale(self.attention(self.norm1(h), position, is_global, record, size, rows),
                            self.scale1, self.dims)
        return x+channel_scale(self.ffn(self.norm2(x)), self.scale2, self.dims)


def _topology_table() -> Tensor:
    """[21, 14] canonical atom slots per residue type, -1 padded; row 20
    (hidden or unknown type) holds the backbone only."""
    rows = [[ATOM_ID[a] for a in RESIDUE_ATOMS[name]] for name in AA3]+[[0, 1, 2, 3]]
    table = torch.full((len(rows), max(map(len, rows))), -1, dtype=torch.long)
    for i, slots in enumerate(rows):
        table[i, :len(slots)] = torch.tensor(slots)
    return table


TOPOLOGY = _topology_table()


def topology_atoms(seq: Tensor, seq_visible: Tensor, residues: Tensor,
                   sidechain_only: bool) -> tuple[Tensor, Tensor]:
    """Canonical (residue, slot) atom queries implied by VISIBLE sequence only.

    A hidden or unknown residue type yields backbone slots only. Whether an
    atom was actually observed never changes the query set.
    """
    aa = seq[residues]
    kind = torch.where(seq_visible[residues] & (aa < len(AA3)), aa, len(AA3))
    table = TOPOLOGY.to(residues.device)[kind]
    keep = table >= (4 if sidechain_only else 0)
    return residues[:, None].expand_as(table)[keep], table[keep]


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
        self.global_only = cfg.global_latent_types == "sem_eq"
        d = cfg.dims
        self.dims = d
        typed = cfg.latent_typing == "typed"
        self.input_maps = nn.ModuleDict({v: LatentAdapter(latent_spec(cfg, v), d)
                                         for v in VIEW_NAMES})
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
        # Predictor-owned (non-EMA) typed output heads; semantic targets carry
        # two extra log-magnitude scalars (see make_typed_target).
        self.heads = nn.ModuleDict({v: TypedLatentHead(d, latent_spec(cfg, v, extra_sem=2), typed,
                                                       cfg.gram_channels) for v in VIEW_NAMES})
        # Atom targets are normalized hidden states (no head): see task_loss.
        self.atom_head = FiberLinear(d, FiberDims(d.scalar+2, d.vector, d.tensor))
        # Raw-geometry reconstruction baseline heads (used only when requested).
        self.raw_heads = cfg.raw_reconstruction_heads
        if self.raw_heads:
            self.raw_angle_head = nn.Linear(d.invariant, 2*RAW_ANGLES)
            self.raw_coordinate_head = ChannelMix(d.vector, 1)

    def _scalar(self, view: str, level: str, role: int, like: Tensor) -> Tensor:
        index = like.new_tensor
        return (self.type_embed(index(VIEW_NAMES.index(view), dtype=torch.long))
                + self.level_embed(index(LEVELS[level], dtype=torch.long))
                + self.role_embed(index(role, dtype=torch.long)))

    def _queries(self, count: int, view: str, level: str) -> Fiber:
        h = Fiber.zeros(count, self.dims, self.mask_embedding)
        h.s = h.s+self.mask_embedding+self._scalar(view, level, 1, self.mask_embedding)
        return h

    def forward(self, context: dict[str, ContextLatent], positions: Tensor, target_view: str,
                query_residues: Tensor, atom_residues: Tensor | None = None,
                atom_slots: Tensor | None = None, batch: Tensor | None = None,
                size: int = 1, raw: bool = False) -> Prediction:
        """`positions`/`batch` are per packed residue; the global prediction has
        one row per record. Tokens of different records never interact."""
        if target_view not in VIEW_NAMES:
            raise ValueError(f"Invalid target view {target_view!r}.")
        if (atom_residues is None) != (atom_slots is None) or (
                atom_residues is not None and len(atom_residues) != len(atom_slots)):
            raise ValueError("Topology-known atom queries require aligned residues and slots.")
        like = self.mask_embedding
        device = like.device
        batch = torch.zeros(len(positions), dtype=torch.long, device=device) if batch is None else batch
        tokens, pos, glob, owner = [], [], [], []

        def add(h: Fiber, position: Tensor, is_global: bool, record: Tensor):
            tokens.append(h)
            pos.append(position)
            glob.append(torch.full((len(h.s),), is_global, dtype=torch.bool, device=device))
            owner.append(record)

        for name, latent in context.items():
            if len(latent.index):
                h = self.input_maps[name](latent.nodes)
                h.s = h.s+self._scalar(name, "node", 0, like)
                add(h, positions[latent.index], False, batch[latent.index])
            records = torch.where(latent.global_valid)[0]
            if len(records):
                h = self.input_maps[name](latent.global_state.index(records), self.global_only)
                h.s = h.s+self._scalar(name, "global", 0, like)
                add(h, positions.new_zeros(len(records)), True, records)
        n_context = sum(len(h.s) for h in tokens)
        n_nodes = len(query_residues)
        add(self._queries(n_nodes, target_view, "node"), positions[query_residues], False,
            batch[query_residues])
        every = torch.arange(size, device=device)
        add(self._queries(size, target_view, "global"), positions.new_zeros(size), True, every)
        h, position, is_global = cat_fibers(tokens), torch.cat(pos), torch.cat(glob)
        record = torch.cat(owner)
        for block in self.blocks:
            h = block(h, position, is_global, record, size)
        stage1 = self.final_norm(h)
        head = self.heads[target_view]
        queries = stage1.index(slice(n_context, n_context+n_nodes))
        node = head(queries)
        glob_out = head(stage1.index(slice(n_context+n_nodes, None)), self.global_only)
        raw_angles = raw_coordinate = None
        if raw and not self.raw_heads:
            raise ValueError("Raw reconstruction losses need model.raw_reconstruction_heads.")
        if raw:
            pairs = self.raw_angle_head(queries.invariant()).view(n_nodes, RAW_ANGLES, 2)
            raw_angles = F.normalize(pairs, dim=-1, eps=1e-6)
            raw_coordinate = self.raw_coordinate_head(queries.v)[:, 0]
        atom = None
        if atom_residues is not None:
            q = self._queries(len(atom_residues), target_view, "atom")
            q.s = q.s+self.atom_query(atom_slots)
            n1 = len(h.s)
            position = torch.cat((position, positions[atom_residues]))
            is_global = torch.cat((is_global, torch.zeros(len(q.s), dtype=torch.bool, device=device)))
            record = torch.cat((record, batch[atom_residues]))
            rows = torch.arange(n1, n1+len(q.s), device=device)
            if len(atom_residues):  # e.g. sc_infill on glycine-only targets: no atoms
                for block in self.atom_blocks:
                    q = block(cat_fibers([h, q]), position, is_global, record, size, rows)
            atom = self.atom_head(self.final_norm(q))
        return Prediction(node, glob_out, atom, raw_angles, raw_coordinate)
