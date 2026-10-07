#!/usr/bin/env python3
"""Matched-width legacy/EFF-Dock-block pretext cost comparison, NOT task accuracy."""
import argparse
from dataclasses import replace, asdict
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from protein_jepa.config import ModelConfig, TrainConfig, load_config
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.models.effdock_blocks import EffDockInteractionBlock
from protein_jepa.models.equivariant import EquivariantBlock
from protein_jepa.objectives.tasks import make_observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['reference', 'cueq-naive', 'cueq-cuda'], default='reference')
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 256])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--task', default='aa_infill')
    p.add_argument('--config', help='Optional model width config; defaults to a small correctness profile')
    p.add_argument('--output', default='reports/effdock_cost.json')
    args = p.parse_args()
    if args.repeats < 1 or min(args.lengths) < 4:
        p.error('positive repeats and lengths >=4 are required')
    torch.set_num_threads(1)
    device = 'cuda' if args.backend == 'cueq-cuda' else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA benchmark requested without a GPU.')
    base = load_config(args.config)[0] if args.config else ModelConfig(
        scalar=16, vector=4, tensor=2, sequence_width=32, sequence_layers=1,
        atom_layers=1, backbone_layers=1, aa_layers=1, internal_layers=1,
        latent_scalar=16, latent_vector=4, latent_tensor=2, circular_channels=4,
        eff_radial_hidden=24, eff_edge_dim=8)
    rows = []
    for length in args.lengths:
        for block in ('legacy', 'effdock'):
            torch.manual_seed(23)
            cfg = replace(base, backend=args.backend, geometry_block=block, dropout=0.)
            model = ProteinJEPA(cfg).to(device).train()
            rec = synthetic_record(length, 11).to(device)
            obs = make_observation(rec, args.task, .35, torch.Generator().manual_seed(8))
            graphs = []
            def capture(module, inputs):
                graphs.append([len(inputs[0].s), inputs[1].edge_index.shape[1]])
            hooks = [m.register_forward_pre_hook(capture) for m in model.modules()
                     if isinstance(m, (EquivariantBlock, EffDockInteractionBlock))]
            def step():
                model.zero_grad(set_to_none=True)
                loss, _ = model([(rec, obs)], TrainConfig())
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all()
                        for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite loss/gradient in benchmark')
            step()  # Warmup, including optional kernel compilation.
            topology = graphs.copy()
            for hook in hooks:
                hook.remove()
            times = []
            if device == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter(); step()
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            row = dict(length=length, geometry_block=block, backend=args.backend,
                task=args.task, median_seconds=statistics.median(times), samples_seconds=times,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                encoder_parameters=sum(p.numel() for p in model.online.encoder.parameters()),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                graph_calls_online_and_teacher=topology, finite_backward=True)
            rows.append(row); print(json.dumps(row), flush=True)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
    versions = {}
    for name in ('torch', 'numpy', 'cuequivariance', 'cuequivariance-torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(protocol='one record, fixed mask, FP32, warmup then forward+loss+backward; no optimizer',
                  limitation='matched widths, NOT matched parameter counts; synthetic geometry; no accuracy claim',
                  versions=versions, python=platform.python_version(), device=device,
                  threads=1, base_config=asdict(base), rows=rows)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
