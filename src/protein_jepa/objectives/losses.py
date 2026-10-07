"""Per-sample/per-type normalized JEPA distances and online regularization."""
import torch
from torch import Tensor
from ..models.fibers import Fiber


def latent_distance(pred: Fiber, target: Fiber, mask: Tensor, equivariant: bool = False):
    if not bool(mask.any()):
        return pred.s.sum()*0, {"valid_targets": 0}
    p, t = pred.index(mask), target.index(mask)
    loss = (p.s-t.s).square().mean()
    terms = {"scalar": float(loss.detach()), "valid_targets": int(mask.sum())}
    if equivariant:
        v = (p.v-t.v).square().sum(-1).mean()/3
        # Cartesian STF Frobenius norm corresponds to five independent components.
        q = (p.t-t.t).square().sum((-1, -2)).mean()/5
        loss = loss+v+q
        terms.update(vector=float(v.detach()), tensor=float(q.detach()))
    return loss, terms


def regularize_latents(latents: list[Tensor], max_per_sample=32):
    """Variance floor and decorrelation on online, normalized context scalars.

    Balanced node subsampling; the tensors must retain online gradients. This
    is a heuristic safeguard next to EMA + predictor asymmetry, not a proof
    of non-collapse. l>0 states are not Gaussian-regularized componentwise.
    """
    if not latents:
        raise ValueError("Provide at least one online latent.")
    selected = []
    for z in latents:
        n = len(z)
        if n == 0:
            continue
        index = torch.linspace(0, n-1, min(n, max_per_sample), device=z.device).round().long()
        selected.append(z[index])
    if not selected:
        zero = latents[0].sum()*0
        return zero, zero, {"regularizer_samples": 0}
    x = torch.cat(selected)
    var = torch.relu(0.5-(x.var(0, unbiased=False)+1e-4).sqrt()).mean()
    centered = x-x.mean(0)
    cov = centered.T @ centered/max(len(x)-1, 1)
    off = cov-torch.diag_embed(cov.diag())
    covariance = off.square().sum()/max(x.shape[-1], 1)
    info = {"regularizer_samples": len(x), "scalar_std": float(x.std(0, unbiased=False).mean().detach()),
            "effective_rank": effective_rank(x.detach())}
    return var, covariance, info


def effective_rank(x: Tensor) -> float:
    """RankMe: exp(entropy) of normalized singular values of centred features."""
    if len(x) < 2:
        return float(len(x))
    sv = torch.linalg.svdvals((x-x.mean(0)).float())
    p = sv/sv.sum().clamp_min(1e-12)
    return float(torch.exp(-(p*(p+1e-12).log()).sum()))
