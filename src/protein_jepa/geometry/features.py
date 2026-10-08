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


# Named atom windows for phi, psi, incoming omega and CA pseudo-dihedral.
_BB_OFFSETS = torch.tensor([[-1, 0, 0, 0], [0, 0, 0, 1], [-1, -1, 0, 0], [-1, 0, 1, 2]])
_BB_SLOTS = torch.tensor([[2, 0, 1, 2], [0, 1, 2, 0], [1, 2, 0, 1], [1, 1, 1, 1]])


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
    rows = torch.arange(nres, device=device)[:, None, None] + _BB_OFFSETS.to(device)
    in_bounds = (rows >= 0) & (rows < nres)
    rows = rows.clamp(0, nres-1)
    slots = _BB_SLOTS.to(device)
    points = x[rows, slots]
    tors, tv = dihedral(*points.unbind(-2))
    links = torch.nn.functional.pad(record.peptide, (1, 2))
    before, after, following = links[:nres], links[1:nres+1], links[2:nres+2]
    connected = torch.stack((before, after, before, before & after & following), -1)
    tv &= (in_bounds & seen[rows, slots]).all(-1) & connected
    tors = torch.where(tv, tors, torch.zeros_like(tors))
    ca_ang = x.new_zeros(nres)
    av = torch.zeros(nres, dtype=torch.bool, device=device)
    dirs = x.new_zeros(nres, 9, 3)
    dv = torch.zeros(nres, 9, dtype=torch.bool, device=device)
    lengths = x.new_zeros(nres, 2)
    ca = x[:, 1]

    if nres > 1:
        bond = record.peptide
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
    v, valid = normalize(x[:, 1:4]-x[:, :3])
    valid &= seen[:, :3] & seen[:, 1:4]
    dirs[:, 2:5] = torch.where(valid[..., None], v, torch.zeros_like(v))
    dv[:, 2:5] = valid
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
class SidechainShapeFeatures:
    offset: Tensor           # [L,3] observed SC centroid minus observed CA
    covariance: Tensor       # [L,3,3] population covariance about the SC centroid
    valid: Tensor            # [L] at least one observed SC atom
    anchor_valid: Tensor     # [L] both SC centroid and CA observed


def sidechain_shape_features(record: ProteinRecord, visible: Tensor | None = None) -> SidechainShapeFeatures:
    """Unweighted visible SC moments, independent of atom naming and identity.

    No CA fallback stands in for an unobserved SC. Covariance needs only SC;
    centroid offset additionally needs CA. A nearby origin limits cancellation.
    """
    seen = record.present.clone()
    if visible is not None:
        seen &= visible
    sc = seen[:, 4:].clone()
    sc[:, ATOM_ID['OXT']-4] = False
    count = sc.sum(-1)
    valid = count > 0
    anchor_valid = valid & seen[:, 1]
    x = record.xyz[:, 4:].float()
    first = x[torch.arange(len(record), device=x.device), sc.long().argmax(-1)]
    origin = torch.where(seen[:, 1, None], record.xyz[:, 1], first)
    origin = torch.where(valid[:, None], origin, torch.zeros_like(origin))
    relative = torch.where(sc[..., None], x-origin[:, None], torch.zeros_like(x))
    mean = relative.sum(1)/count.clamp_min(1)[:, None]
    centered = torch.where(sc[..., None], relative-mean[:, None], torch.zeros_like(relative))
    covariance = (centered.transpose(1, 2) @ centered)/count.clamp_min(1)[:, None, None]
    offset = torch.where(anchor_valid[:, None], mean, torch.zeros_like(mean))
    return SidechainShapeFeatures(offset, covariance, valid, anchor_valid)


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
