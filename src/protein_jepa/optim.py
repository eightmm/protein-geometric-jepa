"""Optimizer construction: AdamW, or Muon for hidden matrices + AdamW for the rest.

Muon orthogonalizes the update of 2D weight matrices (Newton-Schulz). It is
used only for hidden-layer `nn.Linear` weights and the Transformer QKV
projections. Embeddings, direction seeds, cuEquivariance tensor-product
weights, the relative-position bias table, CLS/query vectors, all 1D
parameters and the latent output heads stay on AdamW, as in the Muon recipe
(inputs, outputs and non-matrix parameters use AdamW). Channel-mixing
weights act identically on every m component, so orthogonalizing their
update keeps SO(3) equivariance.
"""
import math
import torch
from torch import nn


def _is_head(name: str) -> bool:
    return ".heads." in name or "_head" in name


def parameter_groups(model: nn.Module, muon: bool, decay_exclusions: bool):
    """(muon_params, adamw_decay, adamw_no_decay) over trainable parameters.

    With decay_exclusions, 1D parameters (biases, norm gains, residual
    scales), embeddings and the relative-position bias table get no weight
    decay; otherwise every AdamW parameter is decayed.
    """
    linear = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    embedding = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    to_muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        matrix = p.ndim == 2 and min(p.shape) >= 2
        if muon and matrix and not _is_head(name) and (id(p) in linear or name.endswith("in_proj_weight")):
            to_muon.append(p)
        elif decay_exclusions and (p.ndim < 2 or id(p) in embedding or name.endswith("attention.bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    return to_muon, decay, no_decay


class OptimizerSet:
    """Several optimizers stepped as one (state dict is a list)."""
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        if len(states) != len(self.optimizers):
            raise ValueError("Checkpoint optimizer set does not match the configured optimizer.")
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


class SchedulerSet:
    """One LambdaLR per optimizer with the same factor."""
    def __init__(self, optimizers: OptimizerSet, factor):
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in optimizers.optimizers]

    def step(self):
        for s in self.schedulers:
            s.step()

    def get_last_lr(self):
        return self.schedulers[0].get_last_lr()

    def state_dict(self):
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, states):
        for s, state in zip(self.schedulers, states):
            s.load_state_dict(state)


def build_optimizer(model: nn.Module, cfg, device: torch.device, learning_rate: float | None = None,
                    weight_decay: float | None = None) -> OptimizerSet:
    lr = cfg.learning_rate if learning_rate is None else learning_rate
    wd = cfg.weight_decay if weight_decay is None else weight_decay
    muon = cfg.optimizer == "muon"
    if muon and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("optimizer: muon needs torch.optim.Muon (PyTorch >= 2.9); "
                           "upgrade torch or set training.optimizer: adamw.")
    to_muon, decay, no_decay = parameter_groups(model, muon, cfg.decay_exclusions)
    groups = [g for g in ({"params": decay, "weight_decay": wd},
                          {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
    optimizers = [torch.optim.AdamW(groups, lr=lr, fused=device.type == "cuda")]
    if to_muon:
        # match_rms_adamw scales Muon's orthogonal update to AdamW's RMS, so
        # one learning rate (and schedule) serves both optimizers.
        optimizers.append(torch.optim.Muon(to_muon, lr=lr, weight_decay=wd,
                                           momentum=cfg.muon_momentum,
                                           adjust_lr_fn="match_rms_adamw"))
    return OptimizerSet(optimizers)


def warmup_cosine(steps: int, warmup: int, floor: float = 0.1):
    """Linear warmup, then cosine decay to `floor` of the peak."""
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        phase = (step-warmup)/max(steps-warmup, 1)
        return floor+(1-floor)*0.5*(1+math.cos(math.pi*min(phase, 1)))
    return factor
