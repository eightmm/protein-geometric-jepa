"""Held-out pretext evaluation; this is NOT a downstream accuracy benchmark."""
import json
from pathlib import Path
from collections import defaultdict
import torch
from .checkpoint import load_checkpoint
from .config import model_config, train_config
from .models.jepa import ProteinJEPA
from .data.dataset import ManifestDataset
from .data.records import random_crop
from .objectives.tasks import make_observation
from .objectives.losses import effective_rank


def summarize_pairs(pairs, results):
    grouped = defaultdict(list)
    for (_, observation), (loss, info, _) in zip(pairs, results):
        grouped[observation.task_name].append({'pretext_loss': float(loss), **info})
    summary = {}
    for task, infos in grouped.items():
        # A record counts when any objective term has supervision: node,
        # global or atom latent targets, or (raw baseline) its own raw targets.
        valid = [r for r in infos if r['valid_targets'] or r.get('valid_global_target')
                 or r.get('valid_atom_targets') or r.get('raw_angle_targets')
                 or r.get('raw_coordinate_targets')]
        retrieved = [r for r in valid if 'node_top1' in r]
        def mean(key, rows=valid):
            return sum(r[key] for r in rows)/len(rows) if rows else None
        raw_angle = [r for r in infos if r.get('raw_angle_targets')]
        raw_coordinate = [r for r in infos if r.get('raw_coordinate_targets')]
        counts = defaultdict(int)
        for row in infos:
            for kind, count in row['valid_targets_by_kind'].items():
                counts[kind] += count
        top1, chance = mean('node_top1', retrieved), mean('node_chance', retrieved)
        summary[task] = {'mean_pretext_loss': mean('pretext_loss'), 'records': len(valid),
                         'pairs': len(infos), 'retrieval_records': len(retrieved),
                         'node_top1': top1, 'node_chance': chance,
                         'retrieval_lift': top1/chance if chance else None,
                         'valid_targets_by_kind': dict(counts),
                         'valid_atom_targets': sum(r['valid_atom_targets'] for r in infos),
                         'mean_node_loss': mean('node_loss'),
                         'mean_global_loss': mean('global_loss'), 'mean_atom_loss': mean('atom_loss'),
                         'mean_raw_angle_loss': mean('raw_angle_loss', raw_angle),
                         'mean_raw_coordinate_loss': mean('raw_coordinate_loss', raw_coordinate),
                         'raw_angle_records': len(raw_angle), 'raw_coordinate_records': len(raw_coordinate)}
    return summary


@torch.no_grad()
def global_diagnostics(model, records):
    """Full observed crop globals, one sample per record and view. Irrep
    diversity uses invariant Gram features, so rotations cannot inflate it.
    """
    outputs = defaultdict(list)
    for record in records:
        for name, latent in model.online.context(model.online.encoder(record, tuple(model.online.heads))).items():
            if bool(latent.global_valid[0]):
                outputs[name].append(latent.global_state)
    report = {}
    def stats(x):
        n = len(x)
        std = float(x.std(0, unbiased=False).mean())
        # Like centred retrieval, avoid treating float32 rotation/centering
        # noise around a shared state as a diverse representation.
        scale = float(x.square().mean().sqrt())
        rank = None if n < 2 else (effective_rank(x) if std > 1e-4*scale else 0.0)
        return {'samples': n, 'mean_std': std, 'effective_rank': rank}
    for name in model.online.heads:
        states = outputs[name]
        if not states:
            report[name] = {'samples': 0}
            continue
        sem = torch.cat([z.sem for z in states])
        info = {'samples': len(states), 'sem': stats(sem)}
        for kind in ('v', 't'):
            x = torch.cat([getattr(z, kind) for z in states])
            if not x.shape[1]:
                continue
            flat = x.flatten(2)
            gram = flat @ flat.transpose(1, 2)
            i, j = torch.triu_indices(x.shape[1], x.shape[1], device=x.device)
            info[kind] = {'mean_norm': float(flat.norm(dim=-1).mean()),
                          'zero_fraction': float((flat.norm(dim=-1) <= 1e-6).float().mean()),
                          'gram': stats(gram[:, i, j])}
        report[name] = info
    return report


def task_gradient_diagnostics(model, pairs, training, per_task=1):
    """Bounded, eval-mode task probes; not the actual accumulated/DDP gradients.
    autograd.grad leaves existing parameter gradients and optimizer state alone.
    """
    if per_task < 1:
        raise ValueError('per_task must be positive.')
    grouped = defaultdict(list)
    for pair in pairs:
        task = pair[1].task_name
        if len(grouped[task]) < per_task:
            grouped[task].append(pair)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or not grouped:
        raise ValueError('Provide trainable parameters and nonempty task pairs.')
    was_training = model.training
    model.eval()
    vectors, norms = [], {}
    try:
        with torch.enable_grad():
            for task, batch in grouped.items():
                loss, _ = model(batch, training)
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                vector = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten().cpu()
                                    for p, g in zip(parameters, gradients)])
                if not torch.isfinite(vector).all():
                    raise FloatingPointError(f'Nonfinite diagnostic gradient for {task}.')
                vectors.append(vector)
                norms[task] = float(vector.norm())
    finally:
        model.train(was_training)
    matrix = torch.stack(vectors)
    unit = matrix/matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return {'tasks': list(grouped), 'pairs': {t: len(b) for t, b in grouped.items()},
            'gradient_norm': norms, 'gradient_cosine': (unit@unit.T).tolist(),
            'scope': 'Eval-mode per-task probe batches; includes rank-local regularizers. '
                     'Not actual DDP or accumulated training-step contributions.'}


def evaluate(checkpoint_path, manifest, output, split='val', max_records=32, device='cpu',
             controls=False, audit_content=False, gradients=False):
    if max_records < 1:
        raise ValueError('max_records must be positive.')
    if split == 'train':
        raise ValueError('Use a held-out split for evaluation.')
    torch.set_num_threads(2)
    payload=load_checkpoint(checkpoint_path)
    cfg=model_config(payload['model_config'])
    training=train_config(payload['train_config'])
    data=ManifestDataset(manifest,split=split,allow_observed_order=training.allow_observed_order)
    content = data.audit_content() if audit_content else None
    if content is not None and not content['passed']:
        raise ValueError('Cross-split content audit failed.')
    model=ProteinJEPA(cfg).to(device).eval()
    model.load_state_dict(payload['model'])
    gen=torch.Generator().manual_seed(training.seed+90001)
    pairs, records = [], []
    for i in range(min(max_records,len(data))):
        record=random_crop(data[i],training.crop_lengths,gen,training.crop_min_observed).to(device)
        records.append(record)
        for task in training.tasks:
            pairs.append((record,make_observation(record,task,training.mask_fraction,gen,
                                                  training.mask_blocks,training.mask_mode,
                                                  training.mask_min_span)))
    with torch.no_grad():
        results=model.sample_losses(pairs,training,16)
        summary=summarize_pairs(pairs,results)
        globals_report=global_diagnostics(model,records)
        control=(summarize_pairs(pairs,model.sample_losses(pairs,training,16,'position_mask_only'))
                 if controls else None)
    result={'split':split,'checkpoint_step':payload['step'],
            'manifest_fingerprint':data.fingerprint,'metrics':summary,
            'global_diagnostics':globals_report, 'content_audit':content,
            'training_manifest_matches':payload['fingerprint']==data.fingerprint,
            'position_mask_only':control,
            'task_gradients':task_gradient_diagnostics(model,pairs,training) if gradients else None,
            'interpretation':'Frozen crop pretext diagnostics; not proof of downstream benefit or '
                             'actual homology separation. Context removal retains position/mask metadata '
                             'and is an intervention, not a separately trained position-only baseline.'}
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2))
    return result
