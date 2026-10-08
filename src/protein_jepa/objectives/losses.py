"""Per-sample/per-type normalized JEPA distances and online regularization."""
import torch
from torch import Tensor
from ..models.fibers import Fiber
from ..data.batch import segment_mean


def latent_distance(pred: Fiber, target: Fiber, mask: Tensor, equivariant: bool = False,
                    segment: Tensor | None = None, size: int = 1):
    """Fiber distance averaged over valid tokens: one scalar with float
    diagnostics, or one value per record with `segment` (record of each token)."""
    single = segment is None
    segment = torch.zeros(len(mask), dtype=torch.long, device=mask.device) if single else segment
    seg = segment[mask]
    p, t = pred.index(mask), target.index(mask)
    scalar, count = segment_mean((p.s-t.s).square().mean(-1), seg, size)
    terms = {"scalar": scalar}
    if equivariant:
        terms["vector"] = segment_mean((p.v-t.v).square().sum(-1).mean(-1)/3, seg, size)[0]
        # Cartesian STF Frobenius norm corresponds to five independent components.
        terms["tensor"] = segment_mean((p.t-t.t).square().sum((-1, -2)).mean(-1)/5, seg, size)[0]
    loss = sum(terms.values())+pred.s.sum()*0
    if not single:
        return loss, {**terms, "valid_targets": count}
    if not bool(count[0] > 0):
        return loss[0], {"valid_targets": 0}
    return loss[0], {**{k: float(v[0].detach()) for k, v in terms.items()},
                     "valid_targets": int(count[0])}


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
