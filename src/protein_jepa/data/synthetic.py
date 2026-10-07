"""Synthetic geometry fixtures. Sidechains are NOT physically validated structures.

These fixtures are for leakage/equivariance/gradient tests, not scientific
benchmarks, force-field tests, or pretraining a useful protein model.
"""
import math
import torch
from .records import ProteinRecord
from .constants import AA3, AA1, AA_TO_ID, ATOM_ID, N_ATOMS, SC_BONDS, UNK
from ..geometry.primitives import normalize, local_frame


def _place(a, b, c, length, theta, torsion):
    e1, _ = normalize(c-b)
    normal, _ = normalize(torch.linalg.cross(b-a, c-b))
    e2 = torch.linalg.cross(normal, e1)
    return c + length*(-math.cos(theta)*e1 + math.sin(theta)*(math.cos(torsion)*e2+math.sin(torsion)*normal))


def synthetic_record(length=32, seed=0) -> ProteinRecord:
    if length < 1:
        raise ValueError("length must be positive")
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(length, N_ATOMS, 3)
    present = torch.zeros(length, N_ATOMS, dtype=torch.bool)
    seq = torch.randint(20, (length,), generator=gen)
    # N at the origin is intentional: tests must not treat it as a missing atom.
    x[0, 0] = torch.tensor([0., 0., 0.])
    x[0, 1] = torch.tensor([1.46, 0., 0.])
    x[0, 2] = x[0, 1]+torch.tensor([0.54, 1.42, 0.])
    for i in range(length):
        if i > 0:
            psi = math.radians(-45 + 25*float(torch.randn((), generator=gen)))
            phi = math.radians(-65 + 20*float(torch.randn((), generator=gen)))
            x[i, 0] = _place(*x[i-1, :3], 1.33, math.radians(116), psi)
            x[i, 1] = _place(x[i-1, 1], x[i-1, 2], x[i, 0], 1.46, math.radians(122), math.pi)
            x[i, 2] = _place(x[i-1, 2], x[i, 0], x[i, 1], 1.52, math.radians(111), phi)
        x[i, 3] = _place(*x[i, :3], 1.23, math.radians(120), 0)
        present[i, :4] = True
        aa = AA3[int(seq[i])]
        if aa == "GLY":
            continue
        frame, _ = local_frame(*[x[i, j] for j in (0, 1, 2)])
        x[i, 4] = x[i, 1]+frame @ torch.tensor([-0.5, -0.8, 1.2])
        present[i, 4] = True
        for edge in SC_BONDS[aa]:
            a, b = [ATOM_ID[k] for k in edge.split('-')]
            if present[i, b] or not present[i, a]:
                continue
            direction, _ = normalize(torch.randn(3, generator=gen))
            x[i, b] = x[i, a]+1.5*(frame @ direction)
            present[i, b] = True
    return ProteinRecord(x, present, seq, torch.arange(length),
                         torch.ones(max(length-1, 0), dtype=torch.bool),
                         tuple(f"A:{i+1}:" for i in range(length)), f"synthetic-{seed}")


def sequence_record(sequence: str) -> ProteinRecord:
    sequence = ''.join(sequence.split()).upper()
    if not sequence or any(a not in AA1+'X' for a in sequence):
        raise ValueError("Use a nonempty standard amino-acid sequence (X allowed).")
    seq = torch.tensor([AA1.index(a) if a in AA1 else UNK for a in sequence])
    n = len(seq)
    return ProteinRecord(torch.zeros(n, N_ATOMS, 3), torch.zeros(n, N_ATOMS, dtype=torch.bool),
                         seq, torch.arange(n), torch.zeros(max(n-1, 0), dtype=torch.bool),
                         tuple(f"A:{i+1}:" for i in range(n)), "sequence-only")
