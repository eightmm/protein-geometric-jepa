"""Separate sequence/geometry observation paths and BB -> AA activation reuse."""
from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
from ..config import ModelConfig
from ..data.constants import MASK, CLS, ATOM_ELEMENT, ATOM_ID
from ..data.records import ProteinRecord
from ..data.batch import RecordBatch, pack, padded_layout, to_padded
from ..data.graphs import (atom_bonds, make_graph, residue_graph, frame_pair_features,
                           BB_ATOM, SC_ATOM, PAIR_FEATURES)
from ..geometry.features import backbone_features, chi_features, sidechain_shape_features
from ..geometry.primitives import normalize, local_frame, symmetric_traceless
from .fibers import (Fiber, FiberDims, FiberLinear, DirectionSeed,
                     GlobalReadout, GlobalTransport, scatter_fiber, cat_fibers)
from .equivariant import EquivariantBlock


def interaction_block(cfg: ModelConfig, stage: int, cutoff: float):
    """Stage: BB atoms=0, SC atoms=1, BB residues=2, AA atoms=3, AA residues=4."""
    pair = PAIR_FEATURES if cfg.pair_frame_features and stage in (2, 4) else 0
    if cfg.interaction == "baseline":
        return EquivariantBlock(cfg.dims, cfg.backend, pair)
    from .effdock_blocks import EffDockInteractionBlock
    return EffDockInteractionBlock(
        cfg.dims, cfg.backend, stage=stage, cutoff=cutoff, dropout=cfg.dropout,
        radial_hidden=cfg.effdock_radial_hidden, expansion=cfg.effdock_expansion,
        residual_scale=cfg.effdock_residual_scale, aggregation=cfg.effdock_aggregation,
        conditional=cfg.effdock_conditioning, dual_radial=cfg.effdock_dual_radial,
        distance_decay=cfg.effdock_distance_decay, norm_rescale=cfg.effdock_norm_rescale,
        smooth_cutoff=cfg.effdock_smooth_cutoff, directional=cfg.effdock_directional,
        ffn=cfg.effdock_ffn, adaptive_cutoff=cfg.effdock_adaptive_cutoff, pair_features=pair,
    )


def residue_pair_graph(cfg: ModelConfig, record, valid: Tensor, visible: Tensor, backbone=None):
    """Residue graph over valid residues, with frame pair features when enabled."""
    ids, graph = residue_graph(record, valid, cfg.radius_residue, cfg.max_neighbors)
    if cfg.pair_frame_features:
        feat = backbone_features(record, visible) if backbone is None else backbone
        graph.pair = frame_pair_features(feat.frames, feat.frame_valid, record.xyz[:, 1].float(),
                                         ids, graph)
    return ids, graph


def position_encoding(position: Tensor, width: int):
    n = (width+1)//2
    freqs = torch.exp(-math.log(10000)*torch.arange(n, device=position.device)/max(n-1, 1))
    phase = position.float()[:, None]*freqs[None]
    return torch.cat((phase.sin(), phase.cos()), -1)[:, :width]


def any_per_record(valid: Tensor, batch: Tensor, size: int) -> Tensor:
    return torch.zeros(size, dtype=torch.long, device=valid.device).index_add_(
        0, batch, valid.long()) > 0


def run_transformer(transformer: nn.Module, x: Tensor, cls: Tensor, record: RecordBatch,
                    key_padding: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """One CLS + residue sequence per record, padded across the batch.

    Padded positions are excluded as keys, so each record sees only itself.
    Returns per-residue outputs [N, D] and per-record CLS outputs [B, D].
    """
    layout = padded_layout(record.batch, record.size)
    real = layout >= 0
    seqs = torch.cat((cls.expand(record.size, 1, -1), to_padded(x, layout)), 1)
    pad = torch.cat((real.new_zeros(record.size, 1), ~real), 1)
    if key_padding is not None:
        pad = pad | torch.cat((key_padding.new_zeros(record.size, 1),
                               key_padding[layout.clamp_min(0)] & real), 1)
    h = transformer(seqs, src_key_padding_mask=pad if bool(pad.any()) else None)
    nodes = h.new_zeros(len(x), h.shape[-1]).index_copy(0, layout[real], h[:, 1:][real])
    return nodes, h[:, 0]


@dataclass
class EncodedView:
    nodes: Fiber
    node_valid: Tensor
    global_state: Fiber          # one state per record of the batch
    global_valid: Tensor         # [B] bool
    atoms: Fiber | None = None
    atom_residue: Tensor | None = None
    atom_slot: Tensor | None = None


class SequenceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(23, cfg.sequence_width)
        layer = nn.TransformerEncoderLayer(cfg.sequence_width, cfg.heads,
                                           cfg.sequence_width*4, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.sequence_layers, enable_nested_tensor=False)
        self.project = nn.Linear(cfg.sequence_width, cfg.scalar)

    def forward(self, record: RecordBatch, visible: Tensor):
        width = self.cfg.sequence_width
        symbols = torch.where(visible, record.seq, record.seq.new_full(record.seq.shape, MASK))
        x = self.embedding(symbols) + position_encoding(record.seq_pos, width)
        cls = self.embedding(symbols.new_tensor([CLS])) + position_encoding(symbols.new_zeros(1), width)
        h, top = run_transformer(self.transformer, x, cls, record)
        nodes = Fiber.zeros(len(record), self.cfg.dims, h)
        nodes.s = self.project(h)
        global_h = Fiber.zeros(record.size, self.cfg.dims, h)
        global_h.s = self.project(top)
        return EncodedView(nodes, visible.clone(), global_h,
                           any_per_record(visible, record.batch, record.size))


class InternalEncoder(nn.Module):
    """A true independent internal-coordinate view, with NO Cartesian kNN graph."""
    def __init__(self, cfg: ModelConfig, sidechain=False):
        super().__init__()
        self.cfg, self.sidechain = cfg, sidechain
        k = 5 if cfg.include_chi5 else 4
        self.stem = nn.Linear(k*4 if sidechain else 18, cfg.sequence_width)
        self.cls = nn.Parameter(torch.zeros(1, cfg.sequence_width))
        layer = nn.TransformerEncoderLayer(cfg.sequence_width, cfg.heads,
                                           cfg.sequence_width*4, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.internal_layers, enable_nested_tensor=False)
        self.out = nn.Linear(cfg.sequence_width, cfg.scalar)

    def forward(self, record, atom_visible, node_visible=None, backbone=None):
        if self.sidechain:
            feat = chi_features(record, atom_visible, self.cfg.include_chi5)
            x = torch.cat((feat.encoded.flatten(1), feat.defined.float(), feat.valid.float()), -1)
            valid = feat.valid.any(-1)
        else:
            feat = backbone_features(record, atom_visible) if backbone is None else backbone
            x = feat.internal
            valid = feat.periodic_valid.any(-1) | feat.ca_angle_valid
        if node_visible is not None:
            valid &= node_visible
            x = torch.where(node_visible[:, None], x, torch.zeros_like(x))
        x = self.stem(x) + position_encoding(record.seq_pos, self.cfg.sequence_width)
        # Invalid rows may receive context, but may not leak through keys or readout.
        h, top = run_transformer(self.transformer, x, self.cls[None], record, key_padding=~valid)
        nodes = Fiber.zeros(len(record), self.cfg.dims, h)
        nodes.s = self.out(h)
        global_h = Fiber.zeros(record.size, self.cfg.dims, h)
        global_h.s = self.out(top)
        return EncodedView(nodes, valid, global_h, any_per_record(valid, record.batch, record.size))


class AtomStem(nn.Module):
    def __init__(self, cfg: ModelConfig, backbone_only=False):
        super().__init__()
        self.cfg, self.backbone_only = cfg, backbone_only
        self.embedding = nn.Embedding(4 if backbone_only else 6, cfg.scalar)
        self.distance_embed = nn.Sequential(nn.Linear(1, cfg.scalar), nn.SiLU(), nn.Linear(cfg.scalar, cfg.scalar))
        self.seed = DirectionSeed(1, cfg.dims)
        self.layers = nn.ModuleList(interaction_block(cfg, 0 if backbone_only else 1, cfg.radius_atom)
                                    for _ in range(cfg.atom_layers))
        self.register_buffer("elements", torch.tensor(ATOM_ELEMENT))
        self.pool_score = nn.Linear(cfg.scalar, 1)
        self.local_frame = not backbone_only and cfg.sc_local_frame
        if self.local_frame:
            self.local_embed = nn.Sequential(nn.Linear(4, cfg.scalar), nn.SiLU(),
                                             nn.Linear(cfg.scalar, cfg.scalar))

    def forward(self, record: RecordBatch, visible: Tensor, backbone=None):
        mask = visible & record.present
        selector = torch.zeros_like(mask)
        if self.backbone_only:
            selector[:, :4] = True
        else:
            selector[:, 4:] = True
            selector[:, ATOM_ID['OXT']] = False
        mask &= selector
        ri, ai = torch.where(mask)
        x = record.xyz[ri, ai]
        atom = Fiber.zeros(len(ri), self.cfg.dims, record.xyz)
        if len(ri):
            token = ai if self.backbone_only else self.elements[ai]
            # CA anchor may be absent; in that case no anchor-derived values enter.
            anchor_seen = (visible & record.present)[ri, 1]
            rel = x-record.xyz[ri, 1]
            rel = torch.where(anchor_seen[:, None], rel, torch.zeros_like(rel))
            unit, _ = normalize(rel)
            s = self.embedding(token)+self.distance_embed(rel.norm(dim=-1, keepdim=True)/4)
            if self.local_frame:
                # Position in the residue's visible N-CA-C frame (zero without one).
                if backbone is None:
                    seen = (visible & record.present)[:, :3]
                    xyz = torch.where(seen[..., None], record.xyz[:, :3], 0)
                    frames, frame_valid = local_frame(*xyz.unbind(1))
                    frame_valid &= seen.all(-1)
                else:
                    frames, frame_valid = backbone.frames, backbone.frame_valid
                ok = frame_valid[ri]
                local = (frames[ri].transpose(-1, -2) @ rel[..., None])[..., 0]/4
                s = s+self.local_embed(torch.cat((local, torch.ones_like(local[:, :1])), -1)
                                       * ok[:, None].to(local.dtype))
            atom = self.seed(s, unit[:, None])
            bonds = atom_bonds(record, ri, ai, self.backbone_only)
            # BB atoms stay residue-local (the residue trunk adds context). SC
            # atoms may see other SC atoms: packing is SC-only information.
            local = self.backbone_only or self.cfg.sc_context == "local"
            graph = make_graph(x, ri, record.seq_pos, bonds, self.cfg.radius_atom,
                               self.cfg.max_neighbors, local_only=local,
                               node_kind=torch.full_like(ri, BB_ATOM if self.backbone_only else SC_ATOM),
                               batch=record.batch[ri])
            for layer in self.layers:
                atom = layer(atom, graph)
        weight = self.pool_score(atom.s).squeeze(-1).sigmoid()
        nodes = scatter_fiber(atom, ri, len(record), weights=weight)
        valid = mask.any(-1)
        return atom, nodes, valid, ri, ai


class BackboneEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.atom_stem = AtomStem(cfg, backbone_only=True)
        self.internal_stem = nn.Sequential(nn.Linear(18, cfg.scalar), nn.SiLU(), nn.Linear(cfg.scalar, cfg.scalar))
        self.seed = DirectionSeed(9, cfg.dims)
        self.layers = nn.ModuleList(interaction_block(cfg, 2, cfg.radius_residue)
                                    for _ in range(cfg.backbone_layers))
        self.global_layers = nn.ModuleList(
            GlobalTransport(cfg.dims, cfg.encoder_global_transport)
            for _ in self.layers if cfg.encoder_global_transport != 'none')
        self.atom_feedback = FiberLinear(cfg.dims, cfg.dims)
        self.readout = GlobalReadout(cfg.dims)

    def forward(self, record, visible, backbone=None):
        feat = backbone_features(record, visible) if backbone is None else backbone
        atoms, nodes, valid, ri, ai = self.atom_stem(record, visible)
        nodes = nodes + self.seed(self.internal_stem(feat.internal), feat.directions)
        valid &= feat.visible[:, 1]
        ids, graph = residue_pair_graph(self.cfg, record, valid, visible, feat)
        h = nodes.index(ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, graph)
            if self.global_layers:
                h = self.global_layers[i](h, record.batch[ids], record.size)
        nodes = scatter_fiber(h, ids, len(record), mean=False)
        atoms = atoms + self.atom_feedback(nodes.index(ri))
        return EncodedView(nodes, valid, self.readout(nodes, valid, record.batch, record.size),
                           any_per_record(valid, record.batch, record.size), atoms, ri, ai)


class SidechainEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stem = AtomStem(cfg, backbone_only=False)
        self.shape_map = (FiberLinear(FiberDims(4, 1, 1), cfg.dims, scalar_bias=False)
                          if cfg.sc_shape_features else None)
        self.readout = GlobalReadout(cfg.dims)
        self.global_transport = (GlobalTransport(cfg.dims, cfg.encoder_global_transport)
                                 if cfg.encoder_global_transport != 'none' else None)

    def forward(self, record, visible, backbone=None):
        atoms, nodes, valid, ri, ai = self.stem(record, visible, backbone)
        if self.shape_map is not None:
            feat = sidechain_shape_features(record, visible)
            radius2 = feat.covariance.diagonal(dim1=-2, dim2=-1).sum(-1)/16
            scalar = torch.stack((feat.offset.norm(dim=-1)/4, radius2,
                                  feat.valid.float(), feat.anchor_valid.float()), -1)
            shape = Fiber(scalar, feat.offset[:, None]/4,
                          symmetric_traceless(feat.covariance)[:, None]/16)
            nodes = nodes + self.shape_map(shape).scale(0.1)
        if self.global_transport is not None:
            ids = torch.where(valid)[0]
            h = self.global_transport(nodes.index(ids), record.batch[ids], record.size)
            nodes = scatter_fiber(h, ids, len(record), mean=False)
        return EncodedView(nodes, valid, self.readout(nodes, valid, record.batch, record.size),
                           any_per_record(valid, record.batch, record.size), atoms, ri, ai)


class AllAtomFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bb_map, self.sc_map = FiberLinear(cfg.dims, cfg.dims), FiberLinear(cfg.dims, cfg.dims)
        self.atom_layers = nn.ModuleList(interaction_block(cfg, 3, cfg.radius_atom) for _ in range(cfg.aa_layers))
        self.res_layer = interaction_block(cfg, 4, cfg.radius_residue)
        self.global_transport = (GlobalTransport(cfg.dims, cfg.encoder_global_transport)
                                 if cfg.encoder_global_transport != 'none' else None)
        self.feedback = FiberLinear(cfg.dims, cfg.dims)
        self.readout = GlobalReadout(cfg.dims)

    def forward(self, record, visible, bb: EncodedView, sc: EncodedView, backbone=None):
        # No writes to bb/sc tensors: the clean BB output survives fusion unchanged.
        initial = self.bb_map(bb.nodes) + self.sc_map(sc.nodes)
        ri = torch.cat((bb.atom_residue, sc.atom_residue))
        ai = torch.cat((bb.atom_slot, sc.atom_slot))
        atom = cat_fibers([bb.atoms, sc.atoms])
        atom = atom + self.feedback(initial.index(ri))
        graph = make_graph(record.xyz[ri, ai], ri, record.seq_pos,
                           atom_bonds(record, ri, ai, False), self.cfg.radius_atom, self.cfg.max_neighbors,
                           node_kind=torch.where(ai < 4, BB_ATOM, SC_ATOM), batch=record.batch[ri])
        for layer in self.atom_layers:
            atom = layer(atom, graph)
        nodes = scatter_fiber(atom, ri, len(record)) + initial
        valid = (bb.node_valid | sc.node_valid) & visible[:, 1] & record.present[:, 1]
        ids, graph = residue_pair_graph(self.cfg, record, valid, visible, backbone)
        h = self.res_layer(nodes.index(ids), graph)
        if self.global_transport is not None:
            h = self.global_transport(h, record.batch[ids], record.size)
        nodes = scatter_fiber(h, ids, len(record), mean=False)
        # Atom output = state after the inter-residue atom layers, which also
        # feeds the residue output. A post-hoc residue->atom map would exist
        # only on the teacher target path and therefore never be trained.
        return EncodedView(nodes, valid, self.readout(nodes, valid, record.batch, record.size),
                           any_per_record(valid, record.batch, record.size), atom, ri, ai)


class MultiViewEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.seq = SequenceEncoder(cfg)
        self.bb = BackboneEncoder(cfg)
        self.sc = SidechainEncoder(cfg)
        self.aa = AllAtomFusion(cfg)
        self.bb_internal = InternalEncoder(cfg, sidechain=False)
        self.chi = InternalEncoder(cfg, sidechain=True)

    def forward(self, record: ProteinRecord | RecordBatch, views: tuple[str, ...] | list[str],
                atom_visible: Tensor | None = None, seq_visible: Tensor | None = None,
                internal_visible: Tensor | None = None) -> dict[str, EncodedView]:
        """Encode one record or a packed batch; node outputs are residue-packed
        and global outputs hold one state per record."""
        record = pack(record)
        allowed = {"seq", "bb", "sc", "aa", "bb_internal", "chi"}
        if not set(views) <= allowed:
            raise ValueError(f"Unknown views: {set(views)-allowed}")
        atom_visible = record.present.clone() if atom_visible is None else atom_visible & record.present
        seq_visible = (record.seq < 20) if seq_visible is None else seq_visible & (record.seq < 20)
        # Shared only within this call and observation mask; a later task or
        # teacher forward recomputes its own geometry.
        backbone = (backbone_features(record, atom_visible)
                    if set(views) & {'bb', 'aa', 'bb_internal'} else None)
        out = {}
        if "seq" in views:
            out['seq'] = self.seq(record, seq_visible)
        if "bb" in views or "aa" in views:
            out['bb'] = self.bb(record, atom_visible, backbone)
        if "sc" in views or "aa" in views:
            out['sc'] = self.sc(record, atom_visible, backbone)
        if "aa" in views:
            out['aa'] = self.aa(record, atom_visible, out['bb'], out['sc'], backbone)
        if "bb_internal" in views:
            out['bb_internal'] = self.bb_internal(record, atom_visible, internal_visible, backbone)
        if "chi" in views:
            out['chi'] = self.chi(record, atom_visible, internal_visible)
        # Only explicitly allowed observation paths are returned to the predictor.
        return {view: out[view] for view in views}
