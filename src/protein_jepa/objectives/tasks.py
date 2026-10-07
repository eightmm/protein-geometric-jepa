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
}


@dataclass
class Observation:
    task_name: str
    spec: TaskSpec
    atom_visible: Tensor
    seq_visible: Tensor
    target_residues: Tensor


def make_observation(record: ProteinRecord, name: str, fraction: float,
                     generator: torch.Generator) -> Observation:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    n = len(record)
    k = min(n, max(1, int(round(n*fraction))))
    start = int(torch.randint(n-k+1, (), generator=generator))
    target = torch.zeros(n, dtype=torch.bool, device=record.xyz.device)
    target[start:start+k] = True
    atoms = record.present.clone()
    seq = record.seq < 20
    if name == "sc_infill":
        atoms[target, 4:] = False
        # Backbone at the same residue is intentionally available.
    elif name in {"bb_infill", "aa_infill"}:
        atoms[target] = False  # closure: no alternative geometric view survives
    elif name == "seq_to_bb":
        seq = seq & ~target   # partial sequence -> full BB parent-crop semantics
        target[:] = True
    else:
        # Entire modality hidden from predictor; other views may deterministically
        # contain the same geometry. This is intentionally a conversion task.
        target[:] = True
    return Observation(name, TASKS[name], atoms, seq, target)
