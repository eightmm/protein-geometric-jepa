"""Overfit diagnostic: can the full JEPA fit a tiny fixed set? NOT a benchmark.

Every (record, task) pair gets ONE fixed observation (mask) so the problem is
stationary apart from the EMA teacher. Success means the pretext loss falls
AND each masked residue's prediction retrieves its own target far above
chance; loss alone cannot separate learning from collapse.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import torch
from .config import ModelConfig, TrainConfig
from .models.jepa import ProteinJEPA
from .objectives.tasks import make_observation
from .objectives.losses import effective_rank
from .train import ema_momentum
from .data.records import random_crop
from .geometry.primitives import random_rotation


def fixed_observations(records, train_cfg: TrainConfig, seed: int, crop: bool = False):
    """One fixed (crop, mask) per (record, task); crops only in stochastic mode."""
    pairs = []
    for i, record in enumerate(records):
        if crop:
            record = random_crop(record, train_cfg.crop_lengths,
                                 torch.Generator().manual_seed(seed+77*i),
                                 train_cfg.crop_min_observed)
        for j, task in enumerate(train_cfg.tasks):
            generator = torch.Generator().manual_seed(seed+1000*i+j)
            pairs.append((record, make_observation(record, task, train_cfg.mask_fraction, generator,
                                                   train_cfg.mask_blocks, train_cfg.mask_mode,
                                                   train_cfg.mask_min_span)))
    return pairs


@torch.no_grad()
def evaluate_pairs(model: ProteinJEPA, pairs, train_cfg: TrainConfig) -> dict:
    was_training = model.training
    model.eval()
    by_task, top1, chance, scalars = defaultdict(list), defaultdict(list), defaultdict(list), []
    by_view = defaultdict(list)
    for (_, observation), (loss, info, inputs) in zip(pairs, model.sample_losses(pairs, train_cfg, 16)):
        by_task[observation.task_name].append(float(loss))
        if 'node_top1' in info:
            top1[observation.task_name].append(info['node_top1'])
            chance[observation.task_name].append(info['node_chance'])
        scalars += [z[0] for z in inputs.values()]
        for name, z in inputs.items():
            by_view[name].append(z[0])
    model.train(was_training)
    tasks = {t: {'loss': sum(v)/len(v),
                 'node_top1': sum(top1[t])/len(top1[t]) if top1[t] else None,
                 'chance': sum(chance[t])/len(chance[t]) if chance[t] else None}
             for t, v in by_task.items()}
    # Pooled rank mixes views of different scale (one large view dominates);
    # the per-view ranks show which view, if any, actually collapsed.
    rank = effective_rank(torch.cat(scalars)) if scalars else 0.0
    view_rank = {name: effective_rank(torch.cat(v)) for name, v in sorted(by_view.items())}
    return {'loss': sum(x['loss'] for x in tasks.values())/len(tasks),
            'node_top1': _mean([x['node_top1'] for x in tasks.values()]),
            'chance': _mean([x['chance'] for x in tasks.values()]),
            'effective_rank': rank, 'view_effective_rank': view_rank, 'tasks': tasks}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def sample_batch(records, train_cfg: TrainConfig, step: int, batch_size: int,
                 generator: torch.Generator):
    """The training loop's sampling: random record, random parent crop,
    optional rigid transform, per-sample round-robin task, fresh mask."""
    batch = []
    for b in range(batch_size):
        task = train_cfg.tasks[(step*batch_size+b) % len(train_cfg.tasks)]
        record = records[int(torch.randint(len(records), (), generator=generator))]
        record = random_crop(record, train_cfg.crop_lengths, generator, train_cfg.crop_min_observed)
        if train_cfg.rigid_augmentation:
            rotation = random_rotation(generator).to(record.xyz.device)
            shift = (torch.randn(3, generator=generator)*train_cfg.translation_std).to(record.xyz.device)
            record = record.rigid_transform(rotation, shift)
        batch.append((record, make_observation(record, task, train_cfg.mask_fraction, generator,
                                               train_cfg.mask_blocks, train_cfg.mask_mode,
                                               train_cfg.mask_min_span)))
    return batch


def overfit(model_cfg: ModelConfig, train_cfg: TrainConfig, records, steps: int,
            learning_rate: float = 1e-3, eval_every: int = 50, device: str = 'cpu',
            seed: int = 0, log=print, stochastic: bool = False, batch_size: int = 4,
            patience: int = 0, lr_schedule: str = 'constant') -> dict:
    """AdamW on a tiny set; returns the evaluation history.

    Fixed mode trains on the fixed evaluation pairs (full batch). Stochastic
    mode trains exactly like `train` (random crops/masks/tasks, EMA schedule)
    and evaluates on fixed crops+masks of the same records.

    patience > 0 stops once `patience` consecutive evaluations improve neither
    retrieval (by > 0.002) nor loss (by > 0.5%): the run has converged.
    lr_schedule='cosine' warms up, then decays to 10% like `train`.
    """
    if steps < 1 or eval_every < 1 or patience < 0:
        raise ValueError('steps and eval_every must be positive; patience nonnegative.')
    if lr_schedule not in {'constant', 'cosine'}:
        raise ValueError(f'Unknown lr_schedule {lr_schedule!r}.')
    torch.manual_seed(seed)
    torch.set_num_threads(train_cfg.threads)  # concurrent runs otherwise oversubscribe CPUs
    records = [r.to(device) for r in records]
    pairs = fixed_observations(records, train_cfg, seed, crop=stochastic)
    sampler = torch.Generator().manual_seed(seed+1)
    model = ProteinJEPA(model_cfg).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=0.0)
    schedule = TrainConfig(**{**train_cfg.__dict__, 'steps': steps})
    warmup = max(1, min(500, steps//20))

    def lr_factor(step):
        if lr_schedule == 'constant':
            return 1.0
        if step < warmup:
            return (step+1)/warmup
        return 0.1+0.9*0.5*(1+math.cos(math.pi*min((step-warmup)/max(steps-warmup, 1), 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history, start = [], time.perf_counter()
    best_top1, best_loss, stale, converged = -1.0, math.inf, 0, None
    for step in range(steps+1):
        if step % eval_every == 0 or step == steps:
            row = {'step': step, 'seconds': time.perf_counter()-start,
                   'lr': optimizer.param_groups[0]['lr'],
                   **evaluate_pairs(model, pairs, train_cfg)}
            history.append(row)
            log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in row.items() if k != 'tasks'}))
            top1 = row['node_top1'] or 0.0
            if top1 > best_top1+0.002 or row['loss'] < best_loss*(1-0.005):
                stale = 0
            elif step:
                stale += 1
            best_top1, best_loss = max(best_top1, top1), min(best_loss, row['loss'])
            if patience and stale >= patience:
                converged = step
                break
        if step == steps:
            break
        batch = sample_batch(records, train_cfg, step, batch_size, sampler) if stochastic else pairs
        loss, _ = model(batch, train_cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if train_cfg.target_encoder == "ema":
            model.update_teacher(ema_momentum(schedule, step))
    first, last = history[0], history[-1]
    return {'records': len(records), 'pairs': len(pairs), 'steps': steps,
            'converged_at_step': converged, 'patience': patience, 'lr_schedule': lr_schedule,
            'mode': 'stochastic' if stochastic else 'fixed', 'batch_size': batch_size,
            'device': str(device), 'backend': model_cfg.backend,
            'interaction': model_cfg.interaction,
            'loss_ratio': last['loss']/first['loss'] if first['loss'] else math.nan,
            'history': history,
            'interpretation': 'Fit of a tiny fixed set; demonstrates trainability, not generalization.'}


def write(result: dict, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
