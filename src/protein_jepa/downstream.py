"""Task-head building blocks, not trained downstream models."""
import math
import torch
from torch import nn
from .models.fibers import Fiber, FiberDims, GlobalReadout


class InvariantNodeHead(nn.Module):
    """Residue or atom regression/classification logits; caller chooses task loss."""
    def __init__(self, dims: FiberDims, outputs: int, hidden: int = 128):
        super().__init__()
        if outputs < 1:
            raise ValueError('outputs must be positive')
        self.mlp = nn.Sequential(nn.Linear(dims.invariant, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

    def forward(self, states: Fiber):
        return self.mlp(states.invariant())


class InvariantProteinHead(nn.Module):
    def __init__(self, dims: FiberDims, outputs: int, hidden: int = 128):
        super().__init__()
        self.pool = GlobalReadout(dims)
        self.head = InvariantNodeHead(dims, outputs, hidden)

    def forward(self, nodes: Fiber, valid: torch.Tensor):
        if not bool(valid.any()):
            raise ValueError('Protein head needs at least one valid node.')
        return self.head(self.pool(nodes, valid))


@torch.no_grad()
def frozen_linear_probe(train_features, train_targets, eval_features, eval_targets, *,
                        train_clusters, eval_clusters, task='regression', ridge=1e-3):
    """CPU ridge probe on frozen invariant features, with train-only scaling.

    Cluster IDs must come from the dataset's independent homology audit;
    checking disjoint IDs here does not verify the clustering itself. Repeated
    IDs support residue/atom samples from one protein cluster.
    """
    if task not in {'regression', 'classification'} or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Use regression/classification and a positive ridge.')
    x, z = train_features.detach().double().cpu(), eval_features.detach().double().cpu()
    y, truth = train_targets.detach().cpu(), eval_targets.detach().cpu()
    if (x.ndim != 2 or z.ndim != 2 or y.ndim == 0 or truth.ndim == 0
            or x.shape[1] != z.shape[1] or not x.shape[1]
            or not len(x) or not len(z) or len(x) != len(y) or len(z) != len(truth)):
        raise ValueError('Features/targets must be nonempty and aligned.')
    if len(train_clusters) != len(x) or len(eval_clusters) != len(z):
        raise ValueError('Provide one audited cluster ID per sample.')
    if any(not isinstance(c, str) or not c for c in list(train_clusters)+list(eval_clusters)):
        raise ValueError('Cluster IDs must be nonempty strings.')
    if set(train_clusters) & set(eval_clusters):
        raise ValueError('Probe clusters overlap between fitting and evaluation.')
    if not all(torch.isfinite(a).all() for a in (x, z, y, truth)):
        raise ValueError('Probe features/targets must be finite.')
    if task == 'classification':
        if y.ndim != 1 or truth.ndim != 1 or y.dtype != torch.long or truth.dtype != torch.long:
            raise ValueError('Classification targets must be int64 vectors.')
        classes, index, counts = y.unique(sorted=True, return_inverse=True, return_counts=True)
        target = torch.nn.functional.one_hot(index, len(classes)).double()
    else:
        if (y.ndim not in (1, 2) or truth.ndim != y.ndim or y.shape[1:] != truth.shape[1:]
                or (y.ndim == 2 and not y.shape[1])):
            raise ValueError('Regression target shapes differ.')
        target = y.double().reshape(len(y), -1)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-8)
    x, z = (x-mean)/scale, (z-mean)/scale
    target_mean = target.mean(0)
    system = x.T@x+ridge*torch.eye(x.shape[1], dtype=x.dtype)
    weight = torch.linalg.solve(system, x.T@(target-target_mean))
    prediction = z@weight+target_mean
    if task == 'classification':
        prediction = classes[prediction.argmax(-1)]
        metrics = {'accuracy': float((prediction == truth).double().mean()),
                   'majority_accuracy': float((truth == classes[counts.argmax()]).double().mean())}
    else:
        prediction = prediction.reshape(truth.shape)
        metrics = {'mse': float((prediction-truth).square().mean()),
                   'mean_baseline_mse': float((target_mean-truth.reshape(len(truth), -1)).square().mean())}
    return {'prediction': prediction, 'weight': weight, 'feature_mean': mean,
            'feature_scale': scale, 'target_mean': target_mean, 'metrics': metrics,
            'classes': classes if task == 'classification' else None,
            'train_samples': len(x), 'eval_samples': len(z)}
