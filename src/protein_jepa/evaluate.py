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


def evaluate(checkpoint_path, manifest, output, split='val', max_records=32, device='cpu'):
    if max_records < 1:
        raise ValueError('max_records must be positive.')
    if split == 'train':
        raise ValueError('Use a held-out split for evaluation.')
    torch.set_num_threads(2)
    payload=load_checkpoint(checkpoint_path)
    cfg=model_config(payload['model_config'])
    training=train_config(payload['train_config'])
    data=ManifestDataset(manifest,split=split,allow_observed_order=training.allow_observed_order)
    model=ProteinJEPA(cfg).to(device).eval()
    model.load_state_dict(payload['model'])
    gen=torch.Generator().manual_seed(training.seed+90001)
    metrics=defaultdict(list)
    pairs=[]
    for i in range(min(max_records,len(data))):
        record=random_crop(data[i],training.crop_lengths,gen,training.crop_min_observed).to(device)
        for task in training.tasks:
            pairs.append((record,make_observation(record,task,training.mask_fraction,gen,
                                                  training.mask_blocks,training.mask_mode,
                                                  training.mask_min_span)))
    with torch.no_grad():
        results=model.sample_losses(pairs,training,16)
    for (_,observation),(loss,info,_) in zip(pairs,results):
        if info['valid_targets']:
            metrics[observation.task_name].append(float(loss))
    summary={name:{'mean_pretext_loss':sum(vals)/len(vals),'records':len(vals)}
             for name,vals in metrics.items()}
    result={'split':split,'checkpoint_step':payload['step'],
            'manifest_fingerprint':data.fingerprint,'metrics':summary,
            'interpretation':'Held-out latent prediction losses; not proof of non-collapse or downstream benefit.'}
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2))
    return result
