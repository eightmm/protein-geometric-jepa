"""Auditable observation policies: conversion != infilling."""
from dataclasses import dataclass
import torch
from torch import Tensor
from ..data.records import ProteinRecord


@dataclass(frozen=True)
class TaskSpec:
    context: tuple[str, ...]
    target: str
    equivariant: bool
    atom_loss: bool = False


TASKS = {
    "seq_to_bb": TaskSpec(("seq",), "bb", False),
    "bb_to_seq": TaskSpec(("bb",), "seq", False),
    "cart_to_internal": TaskSpec(("bb",), "bb_internal", False),
    "internal_to_bb": TaskSpec(("bb_internal",), "bb", False),
    "sc_to_chi": TaskSpec(("sc",), "chi", False),
    "chi_to_sc": TaskSpec(("chi", "bb"), "sc", True),
    "sc_infill": TaskSpec(("seq", "bb", "aa"), "aa", True, True),
    "bb_infill": TaskSpec(("bb",), "bb", True),
    "aa_infill": TaskSpec(("seq", "aa"), "aa", True, True),
    # Sequence-only JEPA (masked sequence -> sequence latents): the baseline for
    # whether structure prediction adds anything to sequence representations.
    # Opt-in; not in the default task list.
    "seq_infill": TaskSpec(("seq",), "seq", False),
}


@dataclass
class Observation:
    task_name: str
    spec: TaskSpec
    atom_visible: Tensor
    seq_visible: Tensor
    target_residues: Tensor


def span_mask(n: int, k: int, blocks: int, generator: torch.Generator,
              min_span: int = 1) -> Tensor:
    """Exactly k residues as up to `blocks` NON-overlapping contiguous spans.

    Each span has at least min_span residues (when k allows), so more blocks
    never shrink gaps into trivially interpolated one-residue holes. Adjacent
    spans may touch; the union size is always k.
    """
    blocks = max(1, min(blocks, k//max(min_span, 1)))
    lengths = [k//blocks + (1 if i < k % blocks else 0) for i in range(blocks)]
    order = torch.randperm(blocks, generator=generator).tolist()
    lengths = [lengths[i] for i in order]
    # Distribute the n-k unmasked residues into blocks+1 gaps.
    cuts = torch.randint(n-k+1, (blocks,), generator=generator).sort().values.tolist()
    target = torch.zeros(n, dtype=torch.bool)
    offset = 0
    for cut, length in zip(cuts, lengths):
        target[cut+offset:cut+offset+length] = True
        offset += length
    return target


def spatial_mask(record: ProteinRecord, k: int, generator: torch.Generator) -> Tensor:
    """The k residues nearest (CA) to a random residue with an observed CA.

    Opt-in only: the query-position set then encodes hidden 3D proximity.
    """
    n = len(record)
    has_ca = record.present[:, 1].cpu()
    if not bool(has_ca.any()):
        return span_mask(n, k, 1, generator)
    candidates = torch.where(has_ca)[0]
    center = candidates[int(torch.randint(len(candidates), (), generator=generator))]
    ca = record.xyz[:, 1].detach().float().cpu()
    distance = (ca-ca[center]).norm(dim=-1).masked_fill(~has_ca, torch.inf)
    order = distance.argsort(stable=True)[:k]
    target = torch.zeros(n, dtype=torch.bool)
    target[order] = True
    return target


def make_observation(record: ProteinRecord, name: str, fraction: float,
                     generator: torch.Generator, blocks: int = 1,
                     mode: str = "span", min_span: int = 1) -> Observation:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    if mode not in {"span", "spatial", "mixed"}:
        raise ValueError(f"Unknown mask mode {mode!r}")
    n = len(record)
    k = min(n, max(1, int(round(n*fraction))))
    if mode == "mixed":
        mode = "spatial" if bool(torch.rand((), generator=generator) < 0.5) else "span"
    target = (spatial_mask(record, k, generator) if mode == "spatial"
              else span_mask(n, k, blocks, generator, min_span)).to(record.xyz.device)
    atoms = record.present.clone()
    seq = record.seq < 20
    if name == "sc_infill":
        atoms[target, 4:] = False
        # Backbone at the same residue is intentionally available.
    elif name in {"bb_infill", "aa_infill"}:
        atoms[target] = False  # closure: no alternative geometric view survives
    elif name == "seq_infill":
        seq = seq & ~target
    elif name == "seq_to_bb":
        seq = seq & ~target   # partial sequence -> full BB parent-crop semantics
        target[:] = True
    else:
        # Entire modality hidden from predictor; other views may deterministically
        # contain the same geometry. This is intentionally a conversion task.
        target[:] = True
    return Observation(name, TASKS[name], atoms, seq, target)
