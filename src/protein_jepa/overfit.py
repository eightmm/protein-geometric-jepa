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
                                 torch.Generator().manual_seed(seed+77*i))
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
    for record, observation in pairs:
        loss, info, inputs = model.task_loss(record, observation, train_cfg)
        by_task[observation.task_name].append(float(loss))
        if 'node_top1' in info:
            top1[observation.task_name].append(info['node_top1'])
            chance[observation.task_name].append(info['node_chance'])
        scalars += [sem for sem, _ in inputs.values()]
    model.train(was_training)
    tasks = {t: {'loss': sum(v)/len(v),
                 'node_top1': sum(top1[t])/len(top1[t]) if top1[t] else None,
                 'chance': sum(chance[t])/len(chance[t]) if chance[t] else None}
             for t, v in by_task.items()}
    rank = effective_rank(torch.cat(scalars)) if scalars else 0.0
    return {'loss': sum(x['loss'] for x in tasks.values())/len(tasks),
            'node_top1': _mean([x['node_top1'] for x in tasks.values()]),
            'chance': _mean([x['chance'] for x in tasks.values()]),
            'effective_rank': rank, 'tasks': tasks}


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
        record = random_crop(record, train_cfg.crop_lengths, generator)
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
            seed: int = 0, log=print, stochastic: bool = False, batch_size: int = 4) -> dict:
    """AdamW on a tiny set; returns the evaluation history.

    Fixed mode trains on the fixed evaluation pairs (full batch). Stochastic
    mode trains exactly like `train` (random crops/masks/tasks, EMA schedule)
    and evaluates on fixed crops+masks of the same records.
    """
    if steps < 1 or eval_every < 1:
        raise ValueError('steps and eval_every must be positive.')
    torch.manual_seed(seed)
    torch.set_num_threads(train_cfg.threads)  # concurrent runs otherwise oversubscribe CPUs
    records = [r.to(device) for r in records]
    pairs = fixed_observations(records, train_cfg, seed, crop=stochastic)
    sampler = torch.Generator().manual_seed(seed+1)
    model = ProteinJEPA(model_cfg).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=0.0)
    schedule = TrainConfig(**{**train_cfg.__dict__, 'steps': steps})
    history, start = [], time.perf_counter()
    for step in range(steps+1):
        if step % eval_every == 0 or step == steps:
            row = {'step': step, 'seconds': time.perf_counter()-start,
                   **evaluate_pairs(model, pairs, train_cfg)}
            history.append(row)
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in row.items() if k != 'tasks'}))
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
        model.update_teacher(ema_momentum(schedule, step))
    first, last = history[0], history[-1]
    return {'records': len(records), 'pairs': len(pairs), 'steps': steps,
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
