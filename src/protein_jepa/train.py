"""Single-process or torchrun DDP training with exact same-world-size resume.

Distributed regularizer statistics are rank-local in v0.1, explicitly. Each
rank averages its own balanced samples; DDP averages gradients, not covariance
statistics. This is not a global-batch VICReg/SIGReg implementation.
"""
import json
import math
import os
import time
from pathlib import Path
from dataclasses import asdict
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from .models.jepa import ProteinJEPA
from .config import model_config, train_config
from .data.records import random_crop
from .geometry.primitives import random_rotation
from .objectives.tasks import TASKS, make_observation
from .checkpoint import save_checkpoint, load_checkpoint, rng_state, restore_rng


def ema_momentum(train_cfg, step):
    """I-JEPA-style linear teacher momentum schedule from ema to ema_end."""
    progress = step/max(train_cfg.steps-1, 1)
    return train_cfg.ema+(train_cfg.ema_end-train_cfg.ema)*min(progress, 1.0)


def train(model_cfg, train_cfg, dataset, output, resume=None, stop_after=None):
    if set(train_cfg.tasks)-set(TASKS) or not train_cfg.tasks:
        raise ValueError("Unknown or empty task schedule.")
    torch.set_num_threads(train_cfg.threads)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(train_cfg.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        device = torch.device('cuda', local_rank if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    out = Path(output)
    setup_error = None
    if rank == 0:
        try:
            out.mkdir(parents=True, exist_ok=True)
            if (out/'last.pt').exists() and resume is None:
                raise FileExistsError("Output contains a checkpoint; use --resume or a new output directory.")
            (out/'resolved_config.json').write_text(json.dumps(
                {'model': asdict(model_cfg), 'training': asdict(train_cfg)}, indent=2))
        except OSError as exc:
            setup_error = str(exc)
    if world > 1:
        messages = [setup_error]
        dist.broadcast_object_list(messages, src=0)
        setup_error = messages[0]
    if setup_error is not None:
        if world > 1:
            dist.destroy_process_group()
        raise OSError(setup_error)
    torch.manual_seed(train_cfg.seed)
    model = ProteinJEPA(model_cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    warmup = max(1, min(50, train_cfg.steps//10))
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        phase = (step-warmup)/max(train_cfg.steps-warmup, 1)
        return 0.1+0.9*0.5*(1+math.cos(math.pi*min(phase, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    sampler = torch.Generator().manual_seed(train_cfg.seed+rank*100003)
    first = 0
    payload = None
    if resume:
        payload = load_checkpoint(resume)
        if asdict(model_config(payload['model_config'])) != asdict(model_cfg):
            raise ValueError("Resume model configuration differs (including backend).")
        if payload['fingerprint'] != dataset.fingerprint or payload['world_size'] != world:
            raise ValueError("Resume requires the same dataset manifest and world size.")
        ignore = {'device', 'threads', 'save_every', 'log_every'}
        # Checked load: removed (v0.2) or unknown keys fail with an explanation.
        current, previous = asdict(train_cfg), asdict(train_config(payload['train_config']))
        if any(current[k] != previous.get(k) for k in current.keys()-ignore):
            raise ValueError("Resume training configuration changed. Keep planned steps/tasks/seed unchanged.")
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        first = payload['step']
    wrapped = (DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                       find_unused_parameters=True) if world > 1 else model)
    if payload:
        restore_rng(payload['rng'][rank], sampler)
    else:
        # Stochastic dropout can differ by rank after parameter initialization.
        torch.manual_seed(train_cfg.seed+rank*100003)
    model.train()
    end = min(train_cfg.steps, stop_after) if stop_after is not None else train_cfg.steps
    if end <= first:
        raise ValueError("No remaining steps in the requested run.")
    start_time = time.perf_counter()
    last_loss = None
    for step in range(first, end):
        batch = []
        # Cycle tasks per SAMPLE so every target encoder also receives
        # online/context updates and one step mixes objectives instead of
        # alternating them. Samples, masks and crops are randomized independently.
        tasks = []
        for b in range(train_cfg.batch_size):
            task = train_cfg.tasks[(step*train_cfg.batch_size+b) % len(train_cfg.tasks)]
            tasks.append(task)
            idx = int(torch.randint(len(dataset), (), generator=sampler))
            record = random_crop(dataset[idx], train_cfg.crop_lengths, sampler).to(device)
            # Apply the SAME rigid transform to the source record from which
            # teacher and student observations are constructed. No frame mismatch.
            if train_cfg.rigid_augmentation:
                rotation = random_rotation(sampler).to(device)
                translation = (torch.randn(3, generator=sampler)*train_cfg.translation_std).to(device)
                record = record.rigid_transform(rotation, translation)
            observation = make_observation(record, task, train_cfg.mask_fraction, sampler,
                                           train_cfg.mask_blocks, train_cfg.mask_mode,
                                           train_cfg.mask_min_span)
            batch.append((record, observation))
        optimizer.zero_grad(set_to_none=True)
        loss, details = wrapped(batch, train_cfg)
        finite = torch.tensor(int(torch.isfinite(loss)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"Nonfinite loss at step {step}; refusing optimizer/EMA update.")
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        model.update_teacher(ema_momentum(train_cfg, step))
        scalar_loss = loss.detach().clone()
        if world > 1:
            dist.all_reduce(scalar_loss)
            scalar_loss /= world
        last_loss = float(scalar_loss)
        if rank == 0 and (step % train_cfg.log_every == 0 or step == end-1):
            row = {'step': step+1, 'loss': last_loss, 'tasks': tasks,
                   'ema': ema_momentum(train_cfg, step),
                   'grad_norm': float(gradnorm), 'lr': scheduler.get_last_lr()[0],
                   'elapsed_seconds': time.perf_counter()-start_time, **details}
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps({k: row[k] for k in ('step', 'tasks', 'loss', 'grad_norm')}), flush=True)
        if (step+1) % train_cfg.save_every == 0 or step == end-1:
            local_state = rng_state(sampler)
            rank_states = [None]*world
            if world > 1:
                dist.all_gather_object(rank_states, local_state)
            else:
                rank_states[0] = local_state
            if rank == 0:
                save_checkpoint(out/'last.pt', model, optimizer, scheduler, step+1, model_cfg,
                                train_cfg, rank_states, dataset.fingerprint, world)
    result = {'steps': end, 'loss': last_loss, 'seconds': time.perf_counter()-start_time,
              'backend': model_cfg.backend, 'interaction': model_cfg.interaction, 'device': str(device), 'world_size': world,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if rank == 0:
        (out/'summary.json').write_text(json.dumps(result, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result
