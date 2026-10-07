"""Separate sequence/geometry observation paths and BB -> AA activation reuse."""
from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
from ..config import ModelConfig
from ..data.constants import MASK, CLS, ATOM_ELEMENT, ATOM_ID
from ..data.records import ProteinRecord
from ..data.graphs import atom_bonds, make_graph, residue_graph
from ..geometry.features import backbone_features, chi_features
from ..geometry.primitives import normalize
from .fibers import (Fiber, FiberDims, FiberLinear, FiberActivation, DirectionSeed,
                     GlobalReadout, scatter_fiber, cat_fibers)
from .equivariant import make_geometry_block


def position_encoding(position: Tensor, width: int):
    n = (width+1)//2
    freqs = torch.exp(-math.log(10000)*torch.arange(n, device=position.device)/max(n-1, 1))
    phase = position.float()[:, None]*freqs[None]
    return torch.cat((phase.sin(), phase.cos()), -1)[:, :width]


@dataclass
class EncodedView:
    nodes: Fiber
    node_valid: Tensor
    global_state: Fiber
    global_valid: bool
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

    def forward(self, record: ProteinRecord, visible: Tensor):
        token = torch.where(visible, record.seq, record.seq.new_full(record.seq.shape, MASK))
        token = torch.cat((token.new_tensor([CLS]), token))
        x = self.embedding(token)
        pos = torch.cat((record.seq_pos.new_zeros(1), record.seq_pos))
        x = x + position_encoding(pos, self.cfg.sequence_width)
        h = self.project(self.transformer(x[None])[0])
        nodes = Fiber.zeros(len(record), self.cfg.dims, h)
        nodes.s = h[1:]
        global_h = Fiber.zeros(1, self.cfg.dims, h)
        global_h.s = h[:1]
        return EncodedView(nodes, visible.clone(), global_h, bool(visible.any()))


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

    def forward(self, record, atom_visible, node_visible=None):
        if self.sidechain:
            feat = chi_features(record, atom_visible, self.cfg.include_chi5)
            x = torch.cat((feat.encoded.flatten(1), feat.defined.float(), feat.valid.float()), -1)
            valid = feat.valid.any(-1)
        else:
            feat = backbone_features(record, atom_visible)
            x = feat.internal
            valid = feat.periodic_valid.any(-1) | feat.ca_angle_valid
        if node_visible is not None:
            valid &= node_visible
            x = torch.where(node_visible[:, None], x, torch.zeros_like(x))
        x = self.stem(x) + position_encoding(record.seq_pos, self.cfg.sequence_width)
        x = torch.cat((self.cls, x), 0)[None]
        # Invalid rows may receive context, but may not leak through keys or readout.
        padding = torch.cat((valid.new_zeros(1), ~valid))[None]
        h = self.out(self.transformer(x, src_key_padding_mask=padding)[0])
        nodes = Fiber.zeros(len(record), self.cfg.dims, h)
        nodes.s = h[1:]
        global_h = Fiber.zeros(1, self.cfg.dims, h)
        global_h.s = h[:1]
        return EncodedView(nodes, valid, global_h, bool(valid.any()))


class AtomStem(nn.Module):
    def __init__(self, cfg: ModelConfig, backbone_only=False):
        super().__init__()
        self.cfg, self.backbone_only = cfg, backbone_only
        self.embedding = nn.Embedding(4 if backbone_only else 6, cfg.scalar)
        self.distance_embed = nn.Sequential(nn.Linear(1, cfg.scalar), nn.SiLU(), nn.Linear(cfg.scalar, cfg.scalar))
        self.seed = DirectionSeed(1, cfg.dims)
        self.layers = nn.ModuleList(make_geometry_block(cfg, 0 if backbone_only else 1, atom=True) for _ in range(cfg.atom_layers))
        self.register_buffer("elements", torch.tensor(ATOM_ELEMENT))
        self.pool_score = nn.Linear(cfg.scalar, 1)

    def forward(self, record: ProteinRecord, visible: Tensor):
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
            atom = self.seed(s, unit[:, None])
            bonds = atom_bonds(record, ri, ai, self.backbone_only)
            graph = make_graph(x, ri, record.seq_pos, bonds, self.cfg.radius_atom,
                               self.cfg.max_neighbors, local_only=True,
                               node_kind=torch.full_like(ri, 0 if self.backbone_only else 1))
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
        self.layers = nn.ModuleList(make_geometry_block(cfg, 2) for _ in range(cfg.backbone_layers))
        self.atom_feedback = FiberLinear(cfg.dims, cfg.dims)
        self.readout = GlobalReadout(cfg.dims)

    def forward(self, record, visible):
        feat = backbone_features(record, visible)
        atoms, nodes, valid, ri, ai = self.atom_stem(record, visible)
        nodes = nodes + self.seed(self.internal_stem(feat.internal), feat.directions)
        valid &= feat.visible[:, 1]
        ids, graph = residue_graph(record, valid, self.cfg.radius_residue, self.cfg.max_neighbors)
        h = nodes.index(ids)
        for layer in self.layers:
            h = layer(h, graph)
        nodes = scatter_fiber(h, ids, len(record), mean=False)
        atoms = atoms + self.atom_feedback(nodes.index(ri))
        return EncodedView(nodes, valid, self.readout(nodes, valid), bool(valid.any()), atoms, ri, ai)


class SidechainEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stem = AtomStem(cfg, backbone_only=False)
        self.readout = GlobalReadout(cfg.dims)

    def forward(self, record, visible):
        atoms, nodes, valid, ri, ai = self.stem(record, visible)
        return EncodedView(nodes, valid, self.readout(nodes, valid), bool(valid.any()), atoms, ri, ai)


class AllAtomFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bb_map, self.sc_map = FiberLinear(cfg.dims, cfg.dims), FiberLinear(cfg.dims, cfg.dims)
        self.atom_layers = nn.ModuleList(make_geometry_block(cfg, 3, atom=True) for _ in range(cfg.aa_layers))
        self.res_layer = make_geometry_block(cfg, 4)
        self.feedback = FiberLinear(cfg.dims, cfg.dims)
        self.activation = FiberActivation(cfg.dims)
        self.readout = GlobalReadout(cfg.dims)

    def forward(self, record, visible, bb: EncodedView, sc: EncodedView):
        # No writes to bb/sc tensors: the clean BB output survives fusion unchanged.
        initial = self.bb_map(bb.nodes) + self.sc_map(sc.nodes)
        ri = torch.cat((bb.atom_residue, sc.atom_residue))
        ai = torch.cat((bb.atom_slot, sc.atom_slot))
        atom = cat_fibers([bb.atoms, sc.atoms])
        atom = atom + self.feedback(initial.index(ri))
        graph = make_graph(record.xyz[ri, ai], ri, record.seq_pos,
                           atom_bonds(record, ri, ai, False), self.cfg.radius_atom, self.cfg.max_neighbors,
                           node_kind=(ai >= 4).long())
        for layer in self.atom_layers:
            atom = layer(atom, graph)
        nodes = scatter_fiber(atom, ri, len(record)) + initial
        valid = (bb.node_valid | sc.node_valid) & visible[:, 1] & record.present[:, 1]
        ids, graph = residue_graph(record, valid, self.cfg.radius_residue, self.cfg.max_neighbors)
        h = self.res_layer(nodes.index(ids), graph)
        nodes = scatter_fiber(h, ids, len(record), mean=False)
        atom = self.activation(atom+self.feedback(nodes.index(ri)))
        return EncodedView(nodes, valid, self.readout(nodes, valid), bool(valid.any()), atom, ri, ai)


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

    def forward(self, record: ProteinRecord, views: tuple[str, ...] | list[str],
                atom_visible: Tensor | None = None, seq_visible: Tensor | None = None,
                internal_visible: Tensor | None = None) -> dict[str, EncodedView]:
        allowed = {"seq", "bb", "sc", "aa", "bb_internal", "chi"}
        if not set(views) <= allowed:
            raise ValueError(f"Unknown views: {set(views)-allowed}")
        atom_visible = record.present.clone() if atom_visible is None else atom_visible & record.present
        seq_visible = (record.seq < 20) if seq_visible is None else seq_visible & (record.seq < 20)
        out = {}
        if "seq" in views:
            out['seq'] = self.seq(record, seq_visible)
        if "bb" in views or "aa" in views:
            out['bb'] = self.bb(record, atom_visible)
        if "sc" in views or "aa" in views:
            out['sc'] = self.sc(record, atom_visible)
        if "aa" in views:
            out['aa'] = self.aa(record, atom_visible, out['bb'], out['sc'])
        if "bb_internal" in views:
            out['bb_internal'] = self.bb_internal(record, atom_visible, internal_visible)
        if "chi" in views:
            out['chi'] = self.chi(record, atom_visible, internal_visible)
        # Only explicitly allowed observation paths are returned to the predictor.
        return {view: out[view] for view in views}
