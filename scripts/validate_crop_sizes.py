#!/usr/bin/env python3
"""Run all tasks at the requested canonical crop lengths on synthetic data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from protein_jepa.config import load_config
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.objectives.tasks import TASKS, make_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/smoke.yaml')
    p.add_argument('--lengths',type=int,nargs='+',default=[128,256])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    cfg,training=load_config(args.config)
    torch.set_num_threads(training.threads)
    torch.manual_seed(training.seed)
    model=ProteinJEPA(cfg).to(training.device)
    rows=[]
    for length in args.lengths:
        if length < 1:
            raise ValueError('Lengths must be positive.')
        record=synthetic_record(length,71).to(training.device)
        for task in TASKS:
            start=time.perf_counter()
            model.zero_grad(set_to_none=True)
            obs=make_observation(record,task,training.mask_fraction,torch.Generator().manual_seed(2))
            loss,_=model([(record,obs)],training)
            loss.backward()
            finite=bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not finite:
                raise FloatingPointError((length,task))
            row={'length':length,'task':task,'loss':float(loss.detach()),'finite_backward':finite,
                 'seconds':time.perf_counter()-start}
            print(json.dumps(row),flush=True); rows.append(row)
    result={'interpretation':'Shape/backward smoke only; synthetic data, small model, NOT a benchmark.',
            'model':asdict(cfg),'device':training.device,'runs':rows}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
