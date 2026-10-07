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
        # Degree aggregation is incompatible with the adaptive envelope (SPEC 12).
        'effdock_degree': {'interaction': 'effdock', 'effdock_aggregation': 'degree',
                           'effdock_adaptive_cutoff': False},
        'no_dual_radial': {'interaction': 'effdock', 'effdock_dual_radial': False},
        'no_conditioning': {'interaction': 'effdock', 'effdock_conditioning': False},
        'no_norm_rescale': {'interaction': 'effdock', 'effdock_norm_rescale': False},
        'no_distance_decay': {'interaction': 'effdock', 'effdock_distance_decay': False},
        'no_smooth_cutoff': {'interaction': 'effdock', 'effdock_smooth_cutoff': False},
        'depth6': {'interaction': 'effdock', 'backbone_layers': 6},
        # v0.3 opt-in encoder experiments, one at a time and all together.
        'bilinear_ffn': {'interaction': 'effdock', 'effdock_ffn': 'bilinear'},
        'directional_gates': {'interaction': 'effdock', 'effdock_directional': True},
        'adaptive_cutoff': {'interaction': 'effdock', 'effdock_adaptive_cutoff': True},
        'sc_spatial': {'interaction': 'effdock', 'sc_context': 'spatial'},
        'all_extensions': {'interaction': 'effdock', 'effdock_ffn': 'bilinear',
                           'effdock_directional': True, 'effdock_adaptive_cutoff': True,
                           'sc_context': 'spatial'},
        # v0.3 JEPA objective/predictor axes.
        'predictor_depth1': {'interaction': 'effdock', 'predictor_layers': 1},
        # v0.4 typed latents: the all-Euclidean baseline and the S^2/SO(3) heads.
        'euclidean_latents': {'interaction': 'effdock', 'latent_typing': 'euclidean'},
        'direction_frame_heads': {'interaction': 'effdock', 'direction_channels': 4,
                                  'frame_channels': 2},
    }
    training_changes = {
        'single_span_mask': {'mask_blocks': 1},
        'constant_ema': {'ema_end': training.ema},
        'no_covariance': {'covariance_weight': 0.0},
        'torus_mmd': {'circular_regularizer': 'torus_mmd'},
        'sphere_mmd': {'semantic_regularizer': 'sphere_mmd'},
    }
    files = {}
    variants = [(name, update, {}) for name, update in changes.items()]
    variants += [(name, {'interaction': 'effdock'}, update) for name, update in training_changes.items()]
    for name, update, train_update in variants:
        for seed in dict.fromkeys(args.seeds):
            path = Path(args.output)/f'{name}_seed{seed}.yaml'
            files[path] = yaml.safe_dump({'model': asdict(replace(model, **update)),
                                         'training': asdict(replace(training, seed=seed,
                                                                    **train_update))},
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
