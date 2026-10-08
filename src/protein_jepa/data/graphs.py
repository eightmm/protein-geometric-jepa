"""Visible-only sparse graphs with explicit covalent/polymer edges."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .batch import padded_layout
from .constants import AA3, ATOM_ID, N_ATOMS, SC_BONDS
from ..geometry.primitives import normalize


# Directed geometry roles, NOT amino-acid identity or hidden-node metadata.
BB_ATOM, SC_ATOM, RESIDUE = 0, 1, 2
NUM_NODE_KINDS = 3
NUM_EDGE_TYPES = 1 + 2*NUM_NODE_KINDS**2  # 0=unspecified; 1..18=(src,dst,bond)


@dataclass
class Graph:
    edge_index: Tensor   # [2,E], src -> dst
    distance: Tensor
    direction: Tensor
    relation: Tensor     # [E,3], same_residue, signed_seq_sep/32, bond
    edge_type: Tensor | None = None  # integer directed role/bond type; legacy=0
    # Per-edge smooth-envelope radius: the destination's (k+1)-th candidate
    # distance when top-k truncates, else the graph radius. None = block cutoff.
    cutoff: Tensor | None = None
    # Optional invariant pair features (residue graphs): local position of the
    # source in the destination frame, relative orientation, validity.
    pair: Tensor | None = None


PAIR_FEATURES = 13


def frame_pair_features(frames: Tensor, frame_valid: Tensor, ca: Tensor, residues: Tensor,
                        graph: Graph) -> Tensor:
    """[E, 13] per edge j -> i: R_i^T (x_j - x_i)/10, R_i^T R_j, both-frames-valid flag.

    Frames are column bases (R x_local = x_world), so these are rotation and
    translation invariant; any pair with an invalid frame is all zero.
    """
    src, dst = graph.edge_index
    i, j = residues[dst], residues[src]
    ri = frames[i].transpose(-1, -2)
    local = (ri @ (ca[j]-ca[i])[..., None])[..., 0]/10
    orientation = (ri @ frames[j]).flatten(1)
    ok = (frame_valid[i] & frame_valid[j]).to(ca.dtype)[:, None]
    return torch.cat((local, orientation, ok.new_ones(len(ok), 1)), -1)*ok


def _bond_table() -> Tensor:
    """[21, B, 2] intra-residue bonded slot pairs (-1 padded); the first three
    columns are N-CA, CA-C, C-O, so backbone-only graphs slice them."""
    rows = []
    for aa in range(len(AA3)+1):
        specs = ["N-CA", "CA-C", "C-O"]
        if aa < len(AA3):
            specs += (["CA-CB"] if AA3[aa] != "GLY" else []) + SC_BONDS[AA3[aa]]
        rows.append([[ATOM_ID[x] for x in spec.split("-")] for spec in specs])
    table = torch.full((len(rows), max(map(len, rows)), 2), -1, dtype=torch.long)
    for aa, pairs in enumerate(rows):
        table[aa, :len(pairs)] = torch.tensor(pairs)
    return table


BOND_TABLE = _bond_table()


def atom_bonds(record, residues: Tensor, slots: Tensor, backbone_only=False) -> Tensor:
    """Directed covalent edges among the given atoms (both endpoints present)."""
    device = record.xyz.device
    n = len(record)
    index = torch.full((n, N_ATOMS), -1, dtype=torch.long, device=device)
    index[residues, slots] = torch.arange(len(residues), device=device)
    table = BOND_TABLE.to(device)[record.seq]
    if backbone_only:
        table = table[:, :3]
    rows = torch.arange(n, device=device)[:, None]
    u, v = index[rows, table[..., 0].clamp_min(0)], index[rows, table[..., 1].clamp_min(0)]
    keep = (table[..., 0] >= 0) & (u >= 0) & (v >= 0)
    u, v = u[keep], v[keep]
    if n > 1:
        c, nxt = index[:-1, ATOM_ID['C']], index[1:, ATOM_ID['N']]
        peptide = record.peptide & (c >= 0) & (nxt >= 0)
        u, v = torch.cat((u, c[peptide])), torch.cat((v, nxt[peptide]))
    return torch.stack((torch.cat((u, v)), torch.cat((v, u))))


def make_graph(x: Tensor, residue_index: Tensor, seq_pos: Tensor, bonds: Tensor,
               radius: float = 8.0, max_neighbors: int = 24,
               local_only: bool = False, chunk_size: int = 256,
               node_kind: Tensor | None = None, batch: Tensor | None = None) -> Graph:
    """At most max_neighbors spatial incoming edges/node, plus explicit bonds.

    Edge discovery is deliberately outside autograd. Only visible coordinates
    may be passed. Empty graphs and coincident non-bonded points are safe.
    `batch` (graph id per node) keeps spatial edges inside each graph of a
    disjoint union; distances are computed per graph on a padded layout.
    """
    if radius <= 0 or max_neighbors < 1:
        raise ValueError("radius and max_neighbors must be positive.")
    n = len(x)
    device = x.device
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if node_kind is not None:
        if node_kind.shape != (n,) or node_kind.dtype != torch.long or node_kind.device != device:
            raise ValueError("node_kind must be an on-device long tensor of shape [N].")
        if bool(((node_kind < 0) | (node_kind >= NUM_NODE_KINDS)).any()):
            raise ValueError("Invalid node_kind.")
    edges = []
    node_cutoff = x.new_full((n,), float(radius))
    with torch.no_grad():
        graph_id = torch.zeros(n, dtype=torch.long, device=device) if batch is None else batch
        layout = padded_layout(graph_id, int(graph_id.max())+1 if n else 1)   # [B, L]
        width = layout.shape[1]
        pad = layout < 0
        points = x.float()[layout.clamp_min(0)]
        residues = residue_index[layout.clamp_min(0)]
        graphs = torch.arange(len(layout), device=device)[:, None, None]
        for start in range(0, width, chunk_size):
            stop = min(start+chunk_size, width)
            d = torch.cdist(points[:, start:stop], points, compute_mode='donot_use_mm_for_euclid_dist')
            keep = (d > 1e-6) & (d <= radius) & ~pad[:, None, :] & ~pad[:, start:stop, None]
            # Do not infer self edges from numerically nonzero cdist diagonals.
            keep &= torch.arange(start, stop, device=device)[:, None] != torch.arange(width, device=device)
            if local_only:
                keep &= residues[:, start:stop, None] == residues[:, None, :]
            d = d.masked_fill(~keep, torch.inf)
            rows = layout[:, start:stop]
            k = min(max_neighbors, width)
            # Stable sort: equal distances resolve by position inside the graph,
            # so the neighbour set does not depend on other graphs' padding.
            ranked, order = d.sort(dim=-1, stable=True)
            if max_neighbors < width:
                # The first excluded candidate defines where the envelope must
                # vanish, so entering/leaving the top-k set is continuous.
                excluded = ranked[..., max_neighbors]
                real = rows >= 0
                node_cutoff[rows[real]] = torch.where(torch.isfinite(excluded), excluded,
                                                      node_cutoff.new_tensor(float(radius)))[real]
            vals, src = ranked[..., :k], order[..., :k]
            ok = torch.isfinite(vals)
            src = layout[graphs.expand_as(src), src]
            dst = rows[..., None].expand_as(src)
            edges.append(torch.stack((src[ok], dst[ok])))
        if bonds.numel():
            edges.append(bonds)
        index = torch.cat(edges, dim=1) if edges else torch.empty(2, 0, dtype=torch.long, device=device)
        if index.numel():
            key = torch.unique(index[0]*max(n, 1)+index[1], sorted=True)
            index = torch.stack((key//n, key % n))
    src, dst = index
    delta = x[dst]-x[src]
    unit, _ = normalize(delta)
    dist = delta.norm(dim=-1)
    # Device-resident membership: avoid a CPU set/list round trip for every edge.
    bonded_keys = torch.unique(bonds[0]*max(n, 1)+bonds[1], sorted=True)
    keys = src*max(n, 1)+dst
    is_bond = torch.zeros(len(src), device=device, dtype=torch.bool)
    if len(bonded_keys):
        where = torch.searchsorted(bonded_keys, keys)
        is_bond = (where < len(bonded_keys)) & (bonded_keys[where.clamp_max(len(bonded_keys)-1)] == keys)
    types = torch.zeros_like(src)
    if node_kind is not None:
        types = 1+2*(node_kind[src]*NUM_NODE_KINDS+node_kind[dst])+is_bond.long()
    is_bond = is_bond.to(x.dtype)
    same = (residue_index[src] == residue_index[dst]).to(x.dtype)
    sep = (seq_pos[residue_index[dst]]-seq_pos[residue_index[src]]).clamp(-32, 32).to(x.dtype)/32
    return Graph(index, dist, unit, torch.stack((same, sep, is_bond), dim=-1), types,
                 node_cutoff[dst])


def residue_graph(record, valid: Tensor, radius=12.0, max_neighbors=24):
    ids = torch.where(valid)[0]
    inverse = torch.full((len(record),), -1, dtype=torch.long, device=ids.device)
    inverse[ids] = torch.arange(len(ids), device=ids.device)
    link = torch.where(record.peptide & valid[:-1] & valid[1:])[0]
    a, b = inverse[link], inverse[link+1]
    bonds = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
    batch = getattr(record, "batch", None)
    graph = make_graph(record.xyz[ids, 1], ids, record.seq_pos, bonds, radius, max_neighbors,
                       node_kind=torch.full_like(ids, RESIDUE),
                       batch=None if batch is None else batch[ids])
    return ids, graph
