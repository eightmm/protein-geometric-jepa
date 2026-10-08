"""Derived features with dependency-closed masking.

All geometry is recomputed from the visible coordinates. No hidden coordinates
are used in the context graph or in local frames, directions, or torsions.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from ..data.records import ProteinRecord
from ..data.constants import AA3, CHI, PI_PERIODIC, ATOM_ID
from .primitives import angle, dihedral, normalize, local_frame, circular_encode


@dataclass
class BackboneFeatures:
    internal: Tensor        # [L, 18]: cos/sin 4 periodic; angle cos/sin; 6 masks; 2 lengths
    periodic: Tensor        # [L,4,2] phi,psi,omega_i,CA tau_i
    periodic_valid: Tensor  # [L,4]
    ca_angle: Tensor
    ca_angle_valid: Tensor
    directions: Tensor     # [L,9,3] prev/next, N-CA, CA-C, C-O, vCB, frame columns
    direction_valid: Tensor
    frames: Tensor
    frame_valid: Tensor
    visible: Tensor        # [L,4]


def backbone_features(record: ProteinRecord, visible: Tensor | None = None) -> BackboneFeatures:
    x = record.xyz[:, :4].float()
    seen = record.present[:, :4].clone()
    if visible is not None:
        seen &= visible[:, :4]
    x = torch.where(seen[..., None], x, torch.zeros_like(x))
    nres = len(record)
    device = x.device
    tors = x.new_zeros(nres, 4)
    tv = torch.zeros(nres, 4, dtype=torch.bool, device=device)
    ca_ang = x.new_zeros(nres)
    av = torch.zeros(nres, dtype=torch.bool, device=device)
    dirs = x.new_zeros(nres, 9, 3)
    dv = torch.zeros(nres, 9, dtype=torch.bool, device=device)
    lengths = x.new_zeros(nres, 2)
    ca = x[:, 1]

    def set_tors(slot, rows, points, dependency):
        value, valid = dihedral(*points)
        valid &= dependency
        tors[rows, slot] = torch.where(valid, value, torch.zeros_like(value))
        tv[rows, slot] = valid

    if nres > 1:
        bond = record.peptide
        set_tors(0, slice(1, None), (x[:-1, 2], x[1:, 0], ca[1:], x[1:, 2]),
                 bond & seen[:-1, 2] & seen[1:, :3].all(-1))
        set_tors(1, slice(None, -1), (x[:-1, 0], ca[:-1], x[:-1, 2], x[1:, 0]),
                 bond & seen[:-1, :3].all(-1) & seen[1:, 0])
        set_tors(2, slice(1, None), (ca[:-1], x[:-1, 2], x[1:, 0], ca[1:]),
                 bond & seen[:-1, 1:3].all(-1) & seen[1:, :2].all(-1))
        direction, valid = normalize(ca[1:] - ca[:-1])
        valid &= bond & seen[:-1, 1] & seen[1:, 1]
        direction = torch.where(valid[:, None], direction, torch.zeros_like(direction))
        dirs[:-1, 1], dirs[1:, 0] = direction, -direction
        dv[:-1, 1], dv[1:, 0] = valid, valid
        d = (ca[1:]-ca[:-1]).norm(dim=-1) * valid
        lengths[:-1, 1], lengths[1:, 0] = d, d
    if nres > 2:
        values, valid = angle(ca[:-2], ca[1:-1], ca[2:])
        valid &= (seen[:-2, 1] & seen[1:-1, 1] & seen[2:, 1]
                  & record.peptide[:-1] & record.peptide[1:])
        ca_ang[1:-1] = torch.where(valid, values, torch.zeros_like(values))
        av[1:-1] = valid
    if nres > 3:
        set_tors(3, slice(1, -2), (ca[:-3], ca[1:-2], ca[2:-1], ca[3:]),
                 seen[:-3, 1] & seen[1:-2, 1] & seen[2:-1, 1] & seen[3:, 1]
                 & record.peptide[:-2] & record.peptide[1:-1] & record.peptide[2:])
    for slot, (a, b) in enumerate(((0, 1), (1, 2), (2, 3)), start=2):
        v, valid = normalize(x[:, b]-x[:, a])
        valid &= seen[:, a] & seen[:, b]
        dirs[:, slot] = torch.where(valid[:, None], v, torch.zeros_like(v))
        dv[:, slot] = valid
    # One virtual-CB rule for every residue, including glycine; no identity lookup.
    b, c = ca-x[:, 0], x[:, 2]-ca
    vcb = -0.58273431*torch.linalg.cross(b, c) + 0.56802827*b - 0.54067466*c
    v, valid = normalize(vcb)
    valid &= seen[:, :3].all(-1)
    dirs[:, 5] = torch.where(valid[:, None], v, torch.zeros_like(v))
    dv[:, 5] = valid
    frames, fm = local_frame(x[:, 0], ca, x[:, 2])
    fm &= seen[:, :3].all(-1)
    frames = torch.where(fm[:, None, None], frames, torch.zeros_like(frames))
    dirs[:, 6:] = frames.transpose(-1, -2)  # actual frame column vectors
    dv[:, 6:] = fm[:, None]
    enc = circular_encode(tors, tv)
    caenc = circular_encode(ca_ang, av)  # nonperiodic angle; no circular target claim
    internal = torch.cat((enc.flatten(1), caenc, tv.float(), av[:, None].float(),
                          seen[:, 1:2].float(), lengths / 4.0), dim=-1)
    return BackboneFeatures(internal, enc, tv, ca_ang, av, dirs, dv, frames, fm, seen)


@dataclass
class ChiFeatures:
    value: Tensor
    defined: Tensor
    valid: Tensor
    periodicity: Tensor
    encoded: Tensor


def _chi_tables():
    """Per residue type (row 20 = UNK): chi atom slots, defined mask, period."""
    atoms = torch.zeros(len(AA3)+1, 5, 4, dtype=torch.long)
    defined = torch.zeros(len(AA3)+1, 5, dtype=torch.bool)
    period = torch.ones(len(AA3)+1, 5)
    for aa, name in enumerate(AA3):
        for j, names in enumerate(CHI[name]):
            atoms[aa, j] = torch.tensor([ATOM_ID[n] for n in names])
            defined[aa, j] = True
            period[aa, j] = 2 if j in PI_PERIODIC.get(name, set()) else 1
    return atoms, defined, period


CHI_TABLES = _chi_tables()


def chi_features(record: ProteinRecord, visible: Tensor | None = None,
                 include_chi5: bool = False) -> ChiFeatures:
    slots = 5 if include_chi5 else 4
    x = record.xyz.float()
    seen = record.present.clone()
    if visible is not None:
        seen &= visible
    x = torch.where(seen[..., None], x, torch.zeros_like(x))
    atoms, defined, period = (t.to(x.device)[record.seq, :slots] for t in CHI_TABLES)
    rows = torch.arange(len(record), device=x.device)[:, None, None]
    points = x[rows, atoms]                                   # [L, slots, 4, 3]
    values, valid = dihedral(*points.unbind(-2))
    valid &= defined & seen[rows, atoms].all(-1)
    values = torch.where(valid, values, torch.zeros_like(values))
    return ChiFeatures(values, defined, valid, period, circular_encode(values, valid, period))
