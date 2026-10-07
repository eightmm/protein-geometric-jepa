"""Typed latents: each geometric kind lives on its own space.

A view's latent bundle Z = (sem, l1, l2, circ, dir, frame):
  sem   [N, D]        invariant semantic scalars (R^D)
  v, t  [N, C1, 3], [N, C2, 3, 3]   SO(3)-equivariant irreps (l=1, STF l=2)
  circ  [N, K, 2]     learned circles S^1 (NOT physical angles)
  dir   [N, Ku, 3]    unit directions S^2 (equivariant)
  frame [N, Kr, 3, 3] rotations SO(3), columns = axes (equivariant)
  strength [N, Ku+Kr] raw dir norm / min(|a|, |b_perp|) of each frame (no grad);
                      targets turn it into validity over the sample's VALID tokens
Missing kinds have zero channels. Heads sit after each view encoder and are
EMA-tracked with it (design B): the context side feeds online Z to the
predictor, so every head weight that shapes a teacher target is trained by
the prediction loss itself, not only by a regularizer (the v0.2 defect).
"""
from dataclasses import dataclass, fields, replace
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .fibers import Fiber, FiberDims

GEOMETRIC_VIEWS = {"bb", "sc", "aa"}
# Absolute raw-norm floor for dir/frame targets: 100x the unit-projection eps,
# so a valid target is an exact unit vector / proper rotation.
STRENGTH_FLOOR = 1e-4
CIRCULAR_VIEWS = {"bb", "sc", "aa", "bb_internal", "chi"}
FRAME_VIEWS = {"bb", "aa"}


@dataclass
class TypedLatent:
    sem: Tensor
    v: Tensor
    t: Tensor
    circ: Tensor
    dir: Tensor
    frame: Tensor
    strength: Tensor

    def index(self, index) -> "TypedLatent":
        return TypedLatent(*(getattr(self, f.name)[index] for f in fields(self)))

    def rotate(self, r: Tensor) -> "TypedLatent":
        rot = lambda m: torch.einsum('ij,...jk,lk->...il', r, m, r)  # noqa: E731
        return replace(self, v=self.v @ r.T, t=rot(self.t), dir=self.dir @ r.T,
                       frame=torch.einsum('ij,...jk->...ik', r, self.frame))

    def tensors(self):
        return [getattr(self, f.name) for f in fields(self)]


class ChannelMix(nn.Module):
    """Bias-free channel mixing of [N, C_in, ...] irreps; empty sizes allowed."""
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.outputs = outputs
        self.linear = nn.Linear(inputs, outputs, bias=False) if inputs and outputs else None

    def forward(self, x: Tensor) -> Tensor:
        if self.linear is None:
            return x.new_zeros((len(x), self.outputs)+tuple(x.shape[2:]))
        return torch.einsum('ab,nb...->na...', self.linear.weight, x)


def _unit(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x/x.norm(dim=-1, keepdim=True).clamp_min(eps)


def gram_schmidt(a: Tensor, b: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Proper rotation (columns e1, e2, e3=e1 x e2) from two vectors: the 6D
    continuous parameterization. Valid only away from zero/parallel inputs."""
    e1 = _unit(a, eps)
    b_perp = b-(b*e1).sum(-1, keepdim=True)*e1
    e2 = _unit(b_perp, eps)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    valid = (a.norm(dim=-1) > eps) & (b_perp.norm(dim=-1) > eps)
    return torch.stack((e1, e2, e3), -1), valid


def gram_invariants(v: Tensor, channels: int) -> Tensor:
    """Upper-triangular Gram v_c . v_c' of the first channels: rotation
    invariants that keep relative angles (norms alone lose them)."""
    k = min(channels, v.shape[1])
    if k == 0:
        return v.new_zeros(len(v), 0)
    g = v[:, :k] @ v[:, :k].transpose(1, 2)
    i, j = torch.triu_indices(k, k, device=v.device)
    return g[:, i, j]


@dataclass(frozen=True)
class LatentSpec:
    sem: int
    vector: int
    tensor: int
    circ: int
    dir: int
    frame: int


def latent_spec(cfg, view: str, extra_sem: int = 0) -> LatentSpec:
    geometric = view in GEOMETRIC_VIEWS
    return LatentSpec(cfg.latent_scalar+extra_sem,
                      cfg.latent_vector if geometric else 0,
                      cfg.latent_tensor if geometric else 0,
                      cfg.circular_channels if view in CIRCULAR_VIEWS else 0,
                      cfg.direction_channels if geometric else 0,
                      cfg.frame_channels if view in FRAME_VIEWS else 0)


class TypedLatentHead(nn.Module):
    """Fiber -> TypedLatent with each kind projected onto its manifold.

    `typed=False` is the all-Euclidean ablation: same heads and capacity,
    no manifold projection (circles/directions stay raw; frames unsupported).
    """
    def __init__(self, dims: FiberDims, spec: LatentSpec, typed: bool = True,
                 gram_channels: int = 4, hidden: int | None = None):
        super().__init__()
        if spec.frame and not typed:
            raise ValueError("Frame latents need typed (manifold) projection.")
        self.dims, self.spec, self.typed = dims, spec, typed
        self.gram = min(gram_channels, dims.vector)
        inv = dims.invariant+self.gram*(self.gram+1)//2
        hidden = hidden or dims.scalar
        self.sem = nn.Sequential(nn.Linear(inv, hidden), nn.SiLU(), nn.Linear(hidden, spec.sem))
        self.eq_v, self.eq_t = ChannelMix(dims.vector, spec.vector), ChannelMix(dims.tensor, spec.tensor)
        self.circ = nn.Linear(inv, 2*spec.circ) if spec.circ else None
        self.vec = ChannelMix(dims.vector, spec.dir+2*spec.frame)

    def invariants(self, h: Fiber) -> Tensor:
        return torch.cat((h.invariant(), gram_invariants(h.v, self.gram)), -1)

    def forward(self, h: Fiber) -> TypedLatent:
        n, s = len(h.s), self.spec
        inv = self.invariants(h)
        circ = (self.circ(inv).view(n, s.circ, 2) if self.circ is not None
                else h.s.new_zeros(n, 0, 2))
        vec = self.vec(h.v)
        raw_direction, pairs = vec[:, :s.dir], vec[:, s.dir:]
        direction = raw_direction
        if self.typed:
            circ = _unit(circ)
            direction = _unit(raw_direction)
        a, b = pairs[:, 0::2], pairs[:, 1::2]
        frame = gram_schmidt(a, b)[0] if s.frame else h.s.new_zeros(n, 0, 3, 3)
        with torch.no_grad():
            b_perp = b-(b*_unit(a)).sum(-1, keepdim=True)*_unit(a)
            strength = torch.cat((raw_direction.norm(dim=-1),
                                  torch.minimum(a.norm(dim=-1), b_perp.norm(dim=-1))), 1)
        return TypedLatent(self.sem(inv), self.eq_v(h.v), self.eq_t(h.t), circ, direction,
                           frame, strength)


def _rms(x: Tensor, dims: tuple[int, ...], dof: int) -> Tensor:
    """Per-token RMS per magnetic component (3 for l=1, 5 for l=2)."""
    if x.shape[1] == 0:
        return x.new_zeros(len(x), 1)
    return (x.square().sum(dims)/dof).mean(-1, keepdim=True).sqrt()


def eq_reference(z: TypedLatent, valid: Tensor) -> tuple[Tensor, Tensor]:
    return tuple(_rms(x, dims, dof)[valid].mean() if bool(valid.any()) else x.new_zeros(())
                 for x, dims, dof in ((z.v, (-1,), 3), (z.t, (-1, -2), 5)))


def make_typed_target(z: TypedLatent, valid: Tensor | None, floor: float = 0.1,
                      instance: bool = True, reference: tuple[Tensor, Tensor] | None = None,
                      eps: float = 1e-6) -> tuple[TypedLatent, dict[str, Tensor]]:
    """Parameter-free normalization of teacher latents (+ per-kind validity).

    sem: instance norm over the sample's valid tokens with a soft variance
    floor (0.01 x mean channel variance) so dead channels are not inflated;
    single tokens (global) use layer norm. l>0: soft per-token RMS with
    tau = floor x mean RMS, plus two log-magnitude scalars appended to sem.
    circ/dir/frame are already on their manifolds. A dir/frame target is
    valid only if its raw strength is above an absolute floor (so the unit
    projection is exact) AND >= 0.1 x the mean strength over VALID tokens
    (scale-free; missing rows must not change which observed tokens count).
    """
    n = len(z.sem)
    valid = torch.ones(n, dtype=torch.bool, device=z.sem.device) if valid is None else valid
    if instance and int(valid.sum()) > 1:
        ref = z.sem[valid]
        var = ref.var(0, unbiased=False)
        sem = (z.sem-ref.mean(0))/(var+0.01*var.mean()+eps).sqrt()
    else:
        sem = F.layer_norm(z.sem, z.sem.shape[-1:], eps=eps)
    means = eq_reference(z, valid) if reference is None else reference
    out, magnitude = [], []
    for (x, dims, dof), mean in zip(((z.v, (-1,), 3), (z.t, (-1, -2), 5)), means):
        r = _rms(x, dims, dof)
        tau = floor*mean
        scale = (r.square()+tau.square()+eps**2).rsqrt()
        out.append(x*scale.reshape(scale.shape+(1,)*(x.ndim-2)))
        magnitude.append(((r+tau+eps)/(mean+tau+eps)).log())
    ku = z.dir.shape[1]
    strength = z.strength
    reference = (strength[valid].mean(0, keepdim=True) if bool(valid.any())
                 else strength.new_zeros(1, strength.shape[1]))
    ok = (strength > STRENGTH_FLOOR) & (strength >= 0.1*reference)
    masks = {'dir': ok[:, :ku], 'frame': ok[:, ku:]}
    return TypedLatent(torch.cat([sem]+magnitude, -1), out[0], out[1], z.circ, z.dir,
                       z.frame, z.strength), masks


def typed_distance(pred: TypedLatent, target: TypedLatent, mask: Tensor,
                   equivariant: bool, typed: bool = True, masks: dict | None = None,
                   kinds: tuple[str, ...] = ('sem', 'eq', 'circ', 'dir', 'frame')):
    """Per-kind distances, each normalized to O(1) and averaged over valid
    tokens; kinds that rotate with the world (eq, dir, frame) only when the
    context provides a frame."""
    if not bool(mask.any()):
        return pred.sem.sum()*0, {"valid_targets": 0}
    p, t = pred.index(mask), target.index(mask)
    masks = {k: m[mask] for k, m in (masks or {}).items()}
    terms = {}
    loss = (p.sem-t.sem).square().mean()
    terms['sem'] = loss
    if equivariant and 'eq' in kinds:
        if p.v.shape[1]:
            terms['vector'] = (p.v-t.v).square().sum(-1).mean()/3
        if p.t.shape[1]:
            terms['tensor'] = (p.t-t.t).square().sum((-1, -2)).mean()/5
    if 'circ' in kinds and p.circ.shape[1]:
        terms['circ'] = (1-(p.circ*t.circ).sum(-1)).mean() if typed else \
            (p.circ-t.circ).square().sum(-1).mean()/2
    if equivariant and 'dir' in kinds and p.dir.shape[1]:
        m = masks.get('dir', torch.ones(p.dir.shape[:2], dtype=torch.bool, device=p.dir.device))
        per = (1-(p.dir*t.dir).sum(-1)) if typed else (p.dir-t.dir).square().sum(-1)/3
        if bool(m.any()):
            terms['dir'] = per[m].mean()
    if equivariant and 'frame' in kinds and p.frame.shape[1]:
        m = masks.get('frame', torch.ones(p.frame.shape[:2], dtype=torch.bool,
                                          device=p.frame.device))
        trace = (p.frame*t.frame).sum((-1, -2))  # tr(R_hat^T R)
        if bool(m.any()):
            terms['frame'] = ((3-trace)/4)[m].mean()
    total = sum(terms.values())
    info = {k: float(v.detach()) for k, v in terms.items()}
    info['valid_targets'] = int(mask.sum())
    return total, info


def circular_floor(z: Tensor, minimum: float = 0.1) -> Tensor:
    """Per-channel relu(v_min - V), V = 1 - |E z|^2 on unit circles: zero
    unless a learned circle concentrates on one point (or two antipodes
    averaged per channel). Not a uniform-torus prior on physical angles."""
    if z.shape[1] == 0 or len(z) < 2:
        return z.sum()*0
    v = 1-z.mean(0).square().sum(-1)
    return F.relu(minimum-v).mean()


def _circle_kernel(delta: Tensor, t: float, terms: int = 8) -> Tensor:
    n = torch.arange(1, terms+1, device=delta.device, dtype=delta.dtype)
    return 1+2*(torch.exp(-t*n**2)*torch.cos(delta[..., None]*n)).sum(-1)


def torus_mmd(z: Tensor, t: float = 1.0) -> Tensor:
    """Normalized heat-kernel MMD to the uniform torus (product of circle
    Fourier heat kernels; arXiv:2609.21656 App. B), divided by the value of
    a fully collapsed representation. Ablation option, not the default."""
    if z.shape[1] == 0 or len(z) < 2:
        return z.sum()*0
    theta = torch.atan2(z[..., 1], z[..., 0]).double()
    delta = theta[:, None]-theta[None]
    k = _circle_kernel(delta, t).prod(-1)
    collapsed = _circle_kernel(theta.new_zeros(()), t)**z.shape[1]
    return ((k.mean()-1)/(collapsed-1)).float()


def sphere_mmd(x: Tensor, t: float = 0.125, degree: int = 6) -> Tensor:
    """Normalized Gegenbauer heat-kernel MMD of unit-normalized latents to the
    uniform sphere S^{D-1}, D >= 3 (ablation option).

    Zero vectors are mapped to one fixed pole, so an all-zero collapse scores
    the maximum (1) instead of looking spread out. The nonconstant kernel
    terms are weighted by normalized log-weights, avoiding (k-1)/(k(0)-1)
    cancellation in high dimension.
    """
    d = x.shape[-1]
    if d < 3:
        raise ValueError("sphere_mmd needs a semantic dimension >= 3.")
    if len(x) < 2:
        return x.sum()*0
    alpha = d/2-1
    norm = x.norm(dim=-1, keepdim=True)
    pole = torch.zeros_like(x)
    pole[:, 0] = 1
    u = torch.where(norm > 1e-6, x/norm.clamp_min(1e-6), pole).double()
    c = (u @ u.T).clamp(-1, 1)
    log_w = torch.tensor([-t*l*(l+d-2)+math.lgamma(l+d-2)-math.lgamma(l+1)-math.lgamma(d-2)
                          + math.log((2*l+d-2)/(d-2)) for l in range(1, degree+1)],
                         dtype=torch.float64, device=x.device)
    weights = torch.softmax(log_w, 0)
    prev_c, cur_c = torch.ones_like(c), 2*alpha*c
    prev_1, cur_1 = 1.0, 2*alpha
    total = weights[0]*(cur_c/cur_1).mean()
    for l in range(2, degree+1):
        prev_c, cur_c = cur_c, (2*c*(l+alpha-1)*cur_c-(l+2*alpha-2)*prev_c)/l
        prev_1, cur_1 = cur_1, (2*(l+alpha-1)*cur_1-(l+2*alpha-2)*prev_1)/l
        total = total+weights[l-1]*(cur_c/cur_1).mean()
    return total.float()
