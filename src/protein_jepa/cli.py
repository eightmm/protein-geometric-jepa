"""Commands: prepare, demo, train, encode, evaluate, audit-manifest."""
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
    ev = commands.add_parser('evaluate')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--split', default='val')
    ev.add_argument('--max-records', type=int, default=32)
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--output', required=True)
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
    elif args.command == 'audit-manifest':
        dataset = ManifestDataset(args.manifest)
        print(json.dumps({'train_records': len(dataset), 'fingerprint': dataset.fingerprint,
                          'audit': 'declared cluster/path split consistency; NOT sequence clustering'}))
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        result = evaluate(args.checkpoint, args.manifest, args.output, args.split, args.max_records, args.device)
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
