"""Commands: prepare, demo, train, overfit, encode, evaluate, audit-manifest."""
import argparse
import json
from pathlib import Path
import torch
from .config import load_config, model_config
from .data.io import read_structure
from .data.records import ProteinRecord
from .data.synthetic import sequence_record
from .data.dataset import ManifestDataset, SyntheticDataset
from .checkpoint import load_checkpoint
from .models.jepa import ProteinJEPA


def _chain(path: str, default: str | None):
    """'file.cif:B' selects chain B for that file; otherwise the --chain default."""
    head, sep, tail = path.rpartition(':')
    if sep and tail and '/' not in tail and not tail.endswith(('.cif', '.pdb', '.mmcif')):
        return head, tail
    return path, default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Read a one-chain PDB/mmCIF and write a numeric NPZ.')
    prepare.add_argument('input')
    prepare.add_argument('--chain')
    prepare.add_argument('--sequence-map', help='JSON list mapping canonical positions to author residue IDs.')
    prepare.add_argument('--output', required=True)
    for name in ('demo', 'train'):
        p = commands.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--resume')
        p.add_argument('--stop-after', type=int, help='Stop early without changing the planned LR schedule.')
        if name == 'train':
            p.add_argument('--manifest', required=True)
        else:
            p.add_argument('--length', type=int, default=32)
            p.add_argument('--count', type=int, default=8)
    fit = commands.add_parser('overfit', help='Fit a tiny fixed set; trainability check only.')
    fit.add_argument('--config', required=True)
    fit.add_argument('--output', required=True)
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument('--records', nargs='+', help='.npz records or .pdb/.cif structures')
    source.add_argument('--synthetic', type=int, help='number of synthetic records')
    fit.add_argument('--length', type=int, default=32, help='synthetic length')
    fit.add_argument('--chain', help='chain for .pdb/.cif inputs (auto if omitted)')
    fit.add_argument('--steps', type=int, default=300)
    fit.add_argument('--lr', type=float, default=1e-3)
    fit.add_argument('--eval-every', type=int, default=50)
    fit.add_argument('--device', default='cpu')
    fit.add_argument('--backend', choices=['reference', 'cueq-naive', 'cueq-cuda'])
    fit.add_argument('--seed', type=int, default=0)
    fit.add_argument('--stochastic', action='store_true',
                     help='train like `train` (random crops/masks/tasks); evaluate on fixed ones')
    fit.add_argument('--batch-size', type=int, default=4)
    fit.add_argument('--patience', type=int, default=0,
                     help='stop after this many evaluations without improvement (0 = off)')
    fit.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    encode = commands.add_parser('encode')
    encode.add_argument('--checkpoint', required=True)
    group = encode.add_mutually_exclusive_group(required=True)
    group.add_argument('--record')
    group.add_argument('--sequence')
    encode.add_argument('--mode', choices=['sequence','backbone','all_atom','multimodal'], default='sequence')
    encode.add_argument('--device', default='cpu')
    encode.add_argument('--output', required=True)
    audit = commands.add_parser('audit-manifest')
    audit.add_argument('manifest')
    audit.add_argument('--content', action='store_true', help='Read records and check cross-split duplicates.')
    audit.add_argument('--min-identity', type=float, help='Optional bounded global sequence-identity screen.')
    audit.add_argument('--max-pairs', type=int, default=10000)
    ev = commands.add_parser('evaluate')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--split', default='val')
    ev.add_argument('--max-records', type=int, default=32)
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--output', required=True)
    ev.add_argument('--controls', action='store_true', help='Also evaluate the position/mask-only control.')
    ev.add_argument('--audit-content', action='store_true', help='Audit all manifest records before evaluation.')
    ev.add_argument('--gradients', action='store_true', help='One eval-mode gradient probe pair per task.')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        record = read_structure(args.input, args.chain, args.sequence_map)
        record.save(path)
        print(json.dumps({'record': str(path), 'residues': len(record),
                          'position_source': record.position_source}))
    elif args.command in {'demo','train'}:
        from .train import train
        cfg, training = load_config(args.config)
        if args.command == 'demo':
            print('SYNTHETIC SMOKE DATA: not physically validated and not a scientific benchmark.')
            dataset = SyntheticDataset(args.count, args.length, training.seed)
        else:
            dataset = ManifestDataset(args.manifest, allow_observed_order=training.allow_observed_order)
        train(cfg, training, dataset, args.output, args.resume, args.stop_after)
    elif args.command == 'overfit':
        from dataclasses import replace
        from .overfit import overfit, write
        cfg, training = load_config(args.config)
        if args.backend:
            cfg = replace(cfg, backend=args.backend)
        if args.synthetic:
            from .data.synthetic import synthetic_record
            records = [synthetic_record(args.length, args.seed+i) for i in range(args.synthetic)]
        else:
            records = [ProteinRecord.load(p) if p.endswith('.npz') else read_structure(*_chain(p, args.chain))
                       for p in args.records]
        result = overfit(cfg, training, records, args.steps, args.lr, args.eval_every,
                         args.device, args.seed, stochastic=args.stochastic,
                         batch_size=args.batch_size, patience=args.patience,
                         lr_schedule=args.lr_schedule)
        write(result, args.output)
        print(json.dumps({k: result[k] for k in ('records', 'pairs', 'steps', 'converged_at_step',
                                                 'loss_ratio')}))
    elif args.command == 'audit-manifest':
        dataset = ManifestDataset(args.manifest)
        if args.content or args.min_identity is not None:
            report = dataset.audit_content(args.min_identity, args.max_pairs)
            print(json.dumps(report))
            if not report['passed']:
                raise ValueError('Cross-split content audit failed or identity screen is incomplete.')
            return
        print(json.dumps({'train_records': len(dataset), 'fingerprint': dataset.fingerprint,
                          'audit': 'declared cluster/path split consistency; NOT sequence clustering'}))
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        result = evaluate(args.checkpoint, args.manifest, args.output, args.split, args.max_records,
                          args.device, args.controls, args.audit_content, args.gradients)
        print(json.dumps(result))
    elif args.command == 'encode':
        torch.set_num_threads(2)
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = model_config(checkpoint['model_config'])
        model = ProteinJEPA(cfg)
        model.load_state_dict(checkpoint['model'])
        record = ProteinRecord.load(args.record) if args.record else sequence_record(args.sequence)
        if args.sequence and args.mode != 'sequence':
            raise ValueError('A sequence-only input does not provide geometry for structural inference.')
        model = model.to(args.device)
        result = model.encode(record.to(args.device), args.mode)
        exported = {'record_id': record.record_id, 'residue_ids': list(record.residue_ids),
                    'seq_pos': record.seq_pos.cpu(), 'layout': 'Cartesian STF l=2 [C,3,3], five DOF', 'views': {}}
        for name, encoded in result.items():
            def pack(h):
                return {k: getattr(h, k).detach().cpu() for k in ('s','v','t')}
            entry = {'nodes': pack(encoded.nodes), 'node_valid': encoded.node_valid.cpu(),
                     'global': pack(encoded.global_state)}
            if encoded.atoms is not None:
                entry.update(atoms=pack(encoded.atoms), atom_residue=encoded.atom_residue.cpu(),
                             atom_slot=encoded.atom_slot.cpu())
            exported['views'][name] = entry
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        torch.save(exported, path)
        print(json.dumps({'output': str(path), 'views': list(result)}))


if __name__ == '__main__':
    main()
