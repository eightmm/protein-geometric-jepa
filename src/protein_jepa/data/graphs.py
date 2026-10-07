"""Visible-only sparse graphs with explicit covalent/polymer edges."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .records import ProteinRecord
from .constants import AA3, ATOM_ID, SC_BONDS
from ..geometry.primitives import normalize


@dataclass
class Graph:
    edge_index: Tensor   # [2,E], src -> dst
    distance: Tensor
    direction: Tensor
    relation: Tensor     # [E,3], same_residue, signed_seq_sep/32, bond


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
               local_only: bool = False, chunk_size: int = 256) -> Graph:
    """At most max_neighbors spatial incoming edges/node, plus explicit bonds.

    Edge discovery is deliberately outside autograd. Only visible coordinates
    may be passed. Empty graphs and coincident non-bonded points are safe.
    """
    if radius <= 0 or max_neighbors < 1:
        raise ValueError("radius and max_neighbors must be positive.")
    n = len(x)
    device = x.device
    edges = []
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
    bonded_keys = set((bonds[0]*max(n, 1)+bonds[1]).detach().cpu().tolist())
    is_bond = torch.tensor([int(a)*max(n, 1)+int(b) in bonded_keys
                            for a, b in index.T.detach().cpu().tolist()], device=device, dtype=x.dtype)
    same = (residue_index[src] == residue_index[dst]).to(x.dtype)
    sep = (seq_pos[residue_index[dst]]-seq_pos[residue_index[src]]).clamp(-32, 32).to(x.dtype)/32
    return Graph(index, dist, unit, torch.stack((same, sep, is_bond), dim=-1))


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
    graph = make_graph(record.xyz[ids, 1], ids, record.seq_pos, bonds, radius, max_neighbors)
    return ids, graph
