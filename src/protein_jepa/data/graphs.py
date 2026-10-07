"""Visible-only sparse graphs with explicit covalent/polymer edges."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .records import ProteinRecord
from .constants import AA3, ATOM_ID, SC_BONDS
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


def atom_bonds(record: ProteinRecord, residues: Tensor, slots: Tensor, backbone_only=False) -> Tensor:
    lookup = {(int(i), int(a)): j for j, (i, a) in enumerate(zip(residues.tolist(), slots.tolist()))}
    edges = []
    for i in range(len(record)):
        specs = ["N-CA", "CA-C", "C-O"]
        if not backbone_only:
            aa = int(record.seq[i])
            if aa < len(AA3):
                specs += (["CA-CB"] if AA3[aa] != "GLY" else []) + SC_BONDS[AA3[aa]]
        for spec in specs:
            a, b = spec.split("-")
            if (i, ATOM_ID[a]) in lookup and (i, ATOM_ID[b]) in lookup:
                u, v = lookup[i, ATOM_ID[a]], lookup[i, ATOM_ID[b]]
                edges.extend(((u, v), (v, u)))
        if i < len(record)-1 and bool(record.peptide[i]):
            if (i, ATOM_ID['C']) in lookup and (i+1, ATOM_ID['N']) in lookup:
                u, v = lookup[i, ATOM_ID['C']], lookup[i+1, ATOM_ID['N']]
                edges.extend(((u, v), (v, u)))
    return torch.tensor(edges, dtype=torch.long, device=record.xyz.device).reshape(-1, 2).T


def make_graph(x: Tensor, residue_index: Tensor, seq_pos: Tensor, bonds: Tensor,
               radius: float = 8.0, max_neighbors: int = 24,
               local_only: bool = False, chunk_size: int = 256,
               node_kind: Tensor | None = None) -> Graph:
    """At most max_neighbors spatial incoming edges/node, plus explicit bonds.

    Edge discovery is deliberately outside autograd. Only visible coordinates
    may be passed. Empty graphs and coincident non-bonded points are safe.
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
        for start in range(0, n, chunk_size):
            stop = min(start+chunk_size, n)
            d = torch.cdist(x[start:stop].float(), x.float(), compute_mode='donot_use_mm_for_euclid_dist')
            keep = (d > 1e-6) & (d <= radius)
            # Do not infer self edges from numerically nonzero cdist diagonals.
            keep &= torch.arange(start, stop, device=device)[:, None] != torch.arange(n, device=device)[None, :]
            if local_only:
                keep &= residue_index[start:stop, None] == residue_index[None, :]
            d = d.masked_fill(~keep, torch.inf)
            k = min(max_neighbors, n)
            if max_neighbors < n:
                # The first excluded candidate defines where the envelope must
                # vanish, so entering/leaving the top-k set is continuous.
                ranked, _ = d.topk(max_neighbors+1, largest=False, sorted=True)
                excluded = ranked[:, -1]
                node_cutoff[start:stop] = torch.where(torch.isfinite(excluded), excluded,
                                                      node_cutoff[start:stop])
            vals, src = d.topk(k, largest=False, sorted=False)
            dst = torch.arange(start, stop, device=device)[:, None].expand_as(src)
            ok = torch.isfinite(vals)
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


def residue_graph(record: ProteinRecord, valid: Tensor, radius=12.0, max_neighbors=24):
    ids = torch.where(valid)[0]
    inverse = torch.full((len(record),), -1, dtype=torch.long, device=ids.device)
    inverse[ids] = torch.arange(len(ids), device=ids.device)
    bonds = []
    for i in torch.where(record.peptide)[0].tolist():
        if bool(valid[i]) and bool(valid[i+1]):
            a, b = int(inverse[i]), int(inverse[i+1])
            bonds.extend(((a, b), (b, a)))
    bonds = torch.tensor(bonds, device=ids.device, dtype=torch.long).reshape(-1, 2).T
    graph = make_graph(record.xyz[ids, 1], ids, record.seq_pos, bonds, radius, max_neighbors,
                       node_kind=torch.full_like(ids, RESIDUE))
    return ids, graph
