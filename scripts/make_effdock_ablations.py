#!/usr/bin/env python3
"""Write explicit, runnable ablation configurations. Never starts training."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import yaml
from protein_jepa.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='configs/effdock_cueq_gpu.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    args = parser.parse_args()
    model, training = load_config(args.base)
    changes = {
        'baseline': {'interaction': 'baseline'},
        'effdock_soft': {'interaction': 'effdock', 'effdock_aggregation': 'soft'},
        'effdock_gate': {'interaction': 'effdock', 'effdock_aggregation': 'gate'},
        'effdock_degree': {'interaction': 'effdock', 'effdock_aggregation': 'degree'},
        'no_dual_radial': {'interaction': 'effdock', 'effdock_dual_radial': False},
        'no_conditioning': {'interaction': 'effdock', 'effdock_conditioning': False},
        'no_norm_rescale': {'interaction': 'effdock', 'effdock_norm_rescale': False},
        'no_distance_decay': {'interaction': 'effdock', 'effdock_distance_decay': False},
        'no_smooth_cutoff': {'interaction': 'effdock', 'effdock_smooth_cutoff': False},
        'depth6': {'interaction': 'effdock', 'backbone_layers': 6},
    }
    files = {}
    for name, update in changes.items():
        for seed in dict.fromkeys(args.seeds):
            path = Path(args.output)/f'{name}_seed{seed}.yaml'
            files[path] = yaml.safe_dump({'model': asdict(replace(model, **update)),
                                         'training': asdict(replace(training, seed=seed))},
                                        sort_keys=False)
    existing = [str(p) for p in files if p.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite: {existing}')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.write_text(content)
    print(f'Wrote {len(files)} configurations; no training was started.')


if __name__ == '__main__':
    main()
