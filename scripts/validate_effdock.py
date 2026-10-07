#!/usr/bin/env python3
"""Compare functional/backward cost of baseline and EFF-Dock-inspired JEPA.

Synthetic, small-model diagnostics only. Neither loss values nor timings are a
protein benchmark or evidence that the new architecture improves downstream.
"""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch
from protein_jepa.config import load_config
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.objectives.tasks import TASKS, make_observation


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/effdock_smoke.yaml')
    parser.add_argument('--backend', choices=['reference', 'cueq-naive', 'cueq-cuda'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'effdock', 'effdock-full'],
                        default=['baseline', 'effdock'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg, training = load_config(args.config)
    if args.backend:
        cfg = replace(cfg, backend=args.backend)
    device = 'cuda' if cfg.backend == 'cueq-cuda' else training.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no implicit fallback.')
    torch.set_num_threads(training.threads)
    versions = {'torch': str(torch.__version__), 'python': platform.python_version()}
    for package in ('cuequivariance', 'cuequivariance-torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    results = []
    for variant in args.variants:
        torch.manual_seed(training.seed)
        full = dict(effdock_ffn='bilinear', effdock_directional=True,
                    effdock_adaptive_cutoff=True, sc_context='spatial')
        effective = replace(cfg, interaction=variant.split('-')[0],
                            **(full if variant == 'effdock-full' else {}))
        model = ProteinJEPA(effective).to(device).train()
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows = []
        for length in args.lengths:
            if length < 4:
                raise ValueError('Validation length must be at least four.')
            record = synthetic_record(length, 71).to(device)
            for task in TASKS:
                model.zero_grad(set_to_none=True)
                obs = make_observation(record, task, training.mask_fraction,
                                       torch.Generator().manual_seed(2), training.mask_blocks,
                                       training.mask_mode, training.mask_min_span)
                if torch.device(device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize(device); start = time.perf_counter()
                loss, _ = model([(record, obs)], training)
                loss.backward()
                synchronize(device); duration = time.perf_counter()-start
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                finite = bool(torch.isfinite(loss)) and bool(gradients) and all(
                    bool(torch.isfinite(g).all()) for g in gradients)
                if not finite:
                    raise FloatingPointError((variant, length, task))
                row = dict(length=length, task=task, loss=float(loss.detach()),
                           finite_backward=finite, seconds=duration)
                if torch.device(device).type == 'cuda':
                    row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                rows.append(row)
                print(json.dumps({'variant': variant, **row}), flush=True)
        results.append(dict(model=asdict(effective), trainable_parameters=parameters, runs=rows))
        del model
    result = dict(interpretation='Synthetic functional smoke and one-shot timing, NOT a quality benchmark.',
                  device=device, threads=training.threads, versions=versions, results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
