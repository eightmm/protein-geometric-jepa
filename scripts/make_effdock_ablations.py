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
        # Opt-in encoder experiments, one at a time and all together.
        'bilinear_ffn': {'interaction': 'effdock', 'effdock_ffn': 'bilinear'},
        'directional_gates': {'interaction': 'effdock', 'effdock_directional': True},
        'adaptive_cutoff': {'interaction': 'effdock', 'effdock_adaptive_cutoff': True},
        'sc_spatial': {'interaction': 'effdock', 'sc_context': 'spatial'},
        'all_extensions': {'interaction': 'effdock', 'effdock_ffn': 'bilinear',
                           'effdock_directional': True, 'effdock_adaptive_cutoff': True,
                           'sc_context': 'spatial'},
        # JEPA objective/predictor axes.
        'predictor_depth1': {'interaction': 'effdock', 'predictor_layers': 1},
        # Typed latents: the all-Euclidean baseline and the S^2/SO(3) heads.
        'euclidean_latents': {'interaction': 'effdock', 'latent_typing': 'euclidean'},
        'direction_frame_heads': {'interaction': 'effdock', 'direction_channels': 4,
                                  'frame_channels': 2},
        'legacy_global_latents': {'global_latent_types': 'legacy'},
        'global_mean': {'encoder_global_transport': 'mean'},
        'global_learned': {'encoder_global_transport': 'learned'},
        'no_sc_shape': {'sc_shape_features': False},
        # Spec-level geometry inputs and the global slot (spec 6.3, 7.1, 13.3).
        'no_pair_frames': {'interaction': 'effdock', 'pair_frame_features': False},
        'sc_local_frame': {'interaction': 'effdock', 'sc_local_frame': True},
        'mean_readout': {'interaction': 'effdock', 'global_readout': 'mean'},
    }
    training_changes = {
        'single_span_mask': {'mask_blocks': 1},
        'constant_ema': {'ema_end': training.ema},
        'no_covariance': {'covariance_weight': 0.0},
        'torus_mmd': {'circular_regularizer': 'torus_mmd'},
        'sphere_mmd': {'semantic_regularizer': 'sphere_mmd'},
        # The comparisons the design spec requires (spec 27).
        'teacher_free': {'target_encoder': 'online'},
        'cosine_semantic': {'semantic_distance': 'cosine'},
        'raw_reconstruction': {'node_weight': 0.0, 'global_weight': 0.0, 'atom_weight': 0.0,
                               'raw_angle_weight': 1.0, 'raw_coordinate_weight': 1.0},
        'node_only': {'global_weight': 0.0},
        'seq_only': {'tasks': ['seq_infill']},
        'seq_bb_only': {'tasks': ['seq_to_bb', 'bb_to_seq']},
        'bb_only': {'tasks': ['bb_infill', 'cart_to_internal', 'internal_to_bb']},
        'representation_tasks': {'tasks': ['seq_to_bb', 'bb_to_seq', 'cart_to_internal',
                                           'internal_to_bb', 'sc_to_chi', 'chi_to_sc']},
        'infilling_tasks': {'tasks': ['bb_infill', 'sc_infill', 'aa_infill']},
        'crop_128': {'crop_lengths': [128]},
        'crop_256': {'crop_lengths': [256]},
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
