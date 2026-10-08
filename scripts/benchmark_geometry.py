#!/usr/bin/env python3
"""Bounded CPU synthetic feature/encoder timings; no training or GPU work."""
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import pstats
import torch
from torch.utils.benchmark import Timer
import protein_jepa
from protein_jepa.config import ModelConfig
from protein_jepa.data.synthetic import synthetic_record
from protein_jepa.geometry.features import backbone_features, chi_features
from protein_jepa.models.jepa import ProteinJEPA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths',type=int,nargs='+',default=[128,256])
    parser.add_argument('--min-run-time',type=float,default=.5)
    parser.add_argument('--sc-shape',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if any(n<1 for n in args.lengths) or not math.isfinite(args.min_run_time) or args.min_run_time<=0:
        parser.error('Lengths and min-run-time must be positive.')
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(1);torch.manual_seed(23)
    options=dict(scalar=16,vector=4,tensor=2,sequence_width=32,sequence_layers=1,
                 internal_layers=1,atom_layers=1,backbone_layers=1,aa_layers=1,
                 predictor_layers=2,latent_scalar=16,latent_vector=4,latent_tensor=2,circular_channels=4)
    # The same benchmark script can run against the immutable pre-change source.
    if 'sc_shape_features' in ModelConfig.__dataclass_fields__:
        options['sc_shape_features']=args.sc_shape
    cfg=ModelConfig(**options)
    model=ProteinJEPA(cfg).eval()
    package=Path(protein_jepa.__file__).parent
    fingerprint=hashlib.sha256()
    for source in sorted(package.rglob('*.py')):
        fingerprint.update(str(source.relative_to(package)).encode()+b'\0'+source.read_bytes())
    timings={}
    for n in args.lengths:
        record=synthetic_record(n,17)
        def encode(views):
            with torch.no_grad():
                return model.online.encoder(record,views)
        cases={'backbone_features':lambda:backbone_features(record),
               'chi_features':lambda:chi_features(record),
               'sc_encoder':lambda:encode(('sc',)),
               'all_views':lambda:encode(('seq','bb','sc','aa','bb_internal','chi'))}
        results={}
        for name,fn in cases.items():
            measurement=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(
                min_run_time=args.min_run_time)
            results[name]={'median_ms':measurement.median*1000,'iqr_ms':measurement.iqr*1000}
        profile=cProfile.Profile();profile.runcall(cases['all_views'])
        results['all_views']['backbone_feature_calls']=sum(
            stat[0] for (_,_,name),stat in pstats.Stats(profile).stats.items() if name=='backbone_features')
        timings[str(n)]=results
    result={'scope':'Synthetic CPU, one thread, seed17. No training/GPU/scientific quality evidence.',
            'source_sha256':fingerprint.hexdigest(),'model':asdict(cfg),'timings':timings}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
