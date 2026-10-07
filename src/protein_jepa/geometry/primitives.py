"""FP32 geometry, explicit validity, column-basis local frames."""
import torch
from torch import Tensor


def normalize(v: Tensor, eps: float = 1e-7) -> tuple[Tensor, Tensor]:
    v = v.float()
    finite = torch.isfinite(v).all(dim=-1)
    clean = torch.where(finite[..., None], v, torch.zeros_like(v))
    length = clean.norm(dim=-1)
    valid = finite & (length > eps)
    out = clean / length.clamp_min(eps)[..., None]
    return torch.where(valid[..., None], out, torch.zeros_like(out)), valid


def angle(a: Tensor, b: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    u, mu = normalize(a-b)
    v, mv = normalize(c-b)
    value = torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u*v).sum(-1))
    valid = mu & mv
    return torch.where(valid, value, torch.zeros_like(value)), valid


def dihedral(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    axis, ma = normalize(c-b)
    b0, b2 = a-b, d-c
    v = b0 - (b0*axis).sum(-1, keepdim=True)*axis
    w = b2 - (b2*axis).sum(-1, keepdim=True)*axis
    v, mv = normalize(v)
    w, mw = normalize(w)
    value = torch.atan2((torch.linalg.cross(axis, v)*w).sum(-1), (v*w).sum(-1))
    valid = ma & mv & mw
    return torch.where(valid, value, torch.zeros_like(value)), valid


def local_frame(n: Tensor, ca: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    e1, m1 = normalize(n-ca)
    e3, m3 = normalize(torch.linalg.cross(n-ca, c-ca))
    e2 = torch.linalg.cross(e3, e1)
    frame = torch.stack((e1, e2, e3), dim=-1)  # columns, NOT vector channel rows
    valid = m1 & m3
    return torch.where(valid[..., None, None], frame, torch.zeros_like(frame)), valid


def circular_encode(value: Tensor, valid: Tensor, periodicity: Tensor | float = 1.0) -> Tensor:
    phase = value * periodicity
    out = torch.stack((phase.cos(), phase.sin()), dim=-1)
    return torch.where(valid[..., None], out, torch.zeros_like(out))


def symmetric_traceless(t: Tensor) -> Tensor:
    t = (t + t.transpose(-1, -2)) * 0.5
    trace = t.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(3, device=t.device, dtype=t.dtype)
    return t - trace[..., None, None] * eye / 3


def outer_stf(a: Tensor, b: Tensor | None = None) -> Tensor:
    b = a if b is None else b
    return symmetric_traceless(a[..., :, None] * b[..., None, :])


def random_rotation(generator: torch.Generator | None = None) -> Tensor:
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    signs = torch.where(r.diagonal() < 0, -1.0, 1.0)
    q = q * signs[None, :]
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q
