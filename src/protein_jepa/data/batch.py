"""Disjoint-union batching of single-chain records (PyG/plmol `collate` layout).

Residues of every record are concatenated; `batch` maps each residue to its
record and `ptr` holds the record offsets. Peptide links across a record
boundary are False, so torsions, bonds and graphs never connect two records.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from .records import ProteinRecord


@dataclass
class RecordBatch:
    xyz: Tensor        # [N, 37, 3]
    present: Tensor    # [N, 37]
    seq: Tensor        # [N]
    seq_pos: Tensor    # [N], positions within each record
    peptide: Tensor    # [N-1], False across record boundaries
    batch: Tensor      # [N], record of each residue
    ptr: Tensor        # [B+1]
    records: tuple[ProteinRecord, ...]

    def __len__(self):
        return len(self.seq)

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def lengths(self) -> list[int]:
        return [len(r) for r in self.records]


def pack(records) -> RecordBatch:
    """One RecordBatch from a ProteinRecord, a RecordBatch or a list of records."""
    if isinstance(records, RecordBatch):
        return records
    if isinstance(records, ProteinRecord):
        records = [records]
    records = tuple(records)
    if not records:
        raise ValueError("Cannot pack an empty list of records.")
    device = records[0].xyz.device
    lengths = torch.tensor([len(r) for r in records], device=device)
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    links = []
    for i, record in enumerate(records):
        links.append(record.peptide)
        if i < len(records)-1:
            links.append(record.peptide.new_zeros(1))
    return RecordBatch(
        torch.cat([r.xyz for r in records]), torch.cat([r.present for r in records]),
        torch.cat([r.seq for r in records]), torch.cat([r.seq_pos for r in records]),
        torch.cat(links), torch.repeat_interleave(torch.arange(len(records), device=device), lengths),
        ptr, records)


def segment_mean(x: Tensor, index: Tensor, size: int, weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Per-segment (weighted) mean over dim 0 and the per-segment weight sum."""
    w = x.new_ones(len(index)) if weight is None else weight.to(x.dtype)
    count = x.new_zeros(size).index_add_(0, index, w)
    total = x.new_zeros((size, *x.shape[1:])).index_add_(0, index, x*w.reshape((-1,)+(1,)*(x.ndim-1)))
    return total/count.clamp_min(1e-12).reshape((-1,)+(1,)*(x.ndim-1)), count


def take(x: Tensor, index):
    """x[index]; a long index uses index_select, whose backward (index_add)
    is far cheaper than advanced indexing's for heavily repeated indices."""
    if isinstance(index, Tensor) and index.dtype == torch.long and index.ndim == 1:
        return x.index_select(0, index)
    return x[index]


def to_padded(x: Tensor, layout: Tensor) -> Tensor:
    """[N, ...] -> [B, L, ...] along a padded layout; padding stays zero.
    Each item lands in exactly one slot, so the backward is a plain gather."""
    real = layout >= 0
    out = x.new_zeros((*layout.shape, *x.shape[1:]))
    out[real] = x.index_select(0, layout[real])
    return out


def padded_layout(segment: Tensor, size: int) -> Tensor:
    """[size, L] flat indices of each segment's items in order, -1 padded."""
    count = torch.bincount(segment, minlength=size)
    order = torch.argsort(segment, stable=True)
    start = torch.cat((count.new_zeros(1), count.cumsum(0)[:-1]))
    rank = torch.arange(len(segment), device=segment.device)-start[segment[order]]
    width = int(count.max()) if len(segment) else 0
    layout = torch.full((size, width), -1, dtype=torch.long, device=segment.device)
    layout[segment[order], rank] = order
    return layout
