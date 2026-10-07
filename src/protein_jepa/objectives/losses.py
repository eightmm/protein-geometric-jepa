"""Per-sample/per-type normalized JEPA distances and online regularization."""
import torch
from torch import Tensor
from ..models.predictors import Latent


def latent_distance(pred: Latent, target: Latent, mask: Tensor,
                    equivariant: bool = False, circular: bool = False):
    if not bool(mask.any()):
        return pred.fiber.s.sum()*0, {"valid_targets": 0}
    p, t = pred.index(mask), target.index(mask)
    loss = (p.fiber.s-t.fiber.s).square().mean()
    terms = {"scalar": float(loss.detach()), "valid_targets": int(mask.sum())}
    if equivariant:
        v = (p.fiber.v-t.fiber.v).square().sum(-1).mean()/3
        # Cartesian STF Frobenius norm corresponds to five independent components.
        q = (p.fiber.t-t.fiber.t).square().sum((-1, -2)).mean()/5
        loss = loss+v+q
        terms.update(vector=float(v.detach()), tensor=float(q.detach()))
    if circular:
        c = (1-(p.circular*t.circular).sum(-1)).clamp_min(0).mean()
        loss = loss+c
        terms["circular"] = float(c.detach())
    return loss, terms


def regularize_latents(latents: list[Latent], max_per_sample=32):
    """Balanced node subsampling; the tensors must retain online gradients.

    Variance/covariance apply only to scalar semantic latents. The circular
    variance floor is a heuristic, not a uniform physical-angle prior. No
    Cartesian component-wise Gaussian regularization is applied to l>0.
    """
    if not latents:
        raise ValueError("Provide at least one online latent.")
    selected = []
    circles = []
    for z in latents:
        n = len(z.fiber.s)
        if n == 0:
            continue
        index = torch.linspace(0, n-1, min(n, max_per_sample), device=z.fiber.s.device).round().long()
        selected.append(z.fiber.s[index])
        circles.append(z.circular[index])
    if not selected:
        zero = latents[0].fiber.s.sum()*0
        return zero, zero, zero, {"regularizer_samples": 0}
    x = torch.cat(selected)
    c = torch.cat(circles)
    var = torch.relu(0.5-(x.var(0, unbiased=False)+1e-4).sqrt()).mean()
    centered = x-x.mean(0)
    cov = centered.T @ centered/max(len(x)-1, 1)
    off = cov-torch.diag_embed(cov.diag())
    covariance = off.square().sum()/max(x.shape[-1], 1)
    circular_var = 1-c.mean(0).square().sum(-1)
    circular_floor = torch.relu(0.1-circular_var).mean()
    info = {"regularizer_samples": len(x), "scalar_std": float(x.std(0, unbiased=False).mean().detach()),
            "circular_variance": float(circular_var.mean().detach())}
    return var, covariance, circular_floor, info
