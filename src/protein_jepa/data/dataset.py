"""Manifest-based sampling; leakage audit precedes crop generation."""
from pathlib import Path
from collections import defaultdict
from itertools import combinations, islice
import hashlib
import json
from .records import ProteinRecord
from .synthetic import synthetic_record


class ManifestDataset:
    def __init__(self, path, split="train", allow_observed_order=False):
        self.path = Path(path).resolve()
        raw = self.path.read_text()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows:
            raise ValueError("Empty manifest.")
        groups, seen_paths = {}, {}
        for row in rows:
            if not {"path", "split", "cluster_id"} <= row.keys():
                raise ValueError("Each manifest row requires path, split, cluster_id.")
            if not isinstance(row['cluster_id'], str) or not row['cluster_id']:
                raise ValueError("Use a nonempty homology cluster ID; no silent fallback.")
            group = row['cluster_id']
            if group in groups and groups[group] != row['split']:
                raise ValueError(f"Cluster {group} appears in multiple splits.")
            groups[group] = row['split']
            resolved = (self.path.parent / row['path']).resolve()
            if resolved in seen_paths:
                raise ValueError(f"Repeated record path in manifest: {resolved}")
            seen_paths[resolved] = row['split']
            row['_resolved'] = resolved
        self.rows = [r for r in rows if r['split'] == split]
        self.all_rows = rows
        if not self.rows:
            raise ValueError(f"No records for split {split!r}.")
        self.allow_observed_order = allow_observed_order
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        record = ProteinRecord.load(self.rows[i]['_resolved'])
        if record.position_source != "canonical" and not self.allow_observed_order:
            raise ValueError("Canonical sequence mapping is required. Supply a sequence_map during "
                             "prepare, or explicitly enable allow_observed_order for a pilot.")
        return record

    def audit_content(self, min_identity: float | None = None, max_pairs: int = 10000):
        """Exact cross-split sequence/content duplicates and an optional bounded
        global sequence-identity screen. This does not establish homology clusters.
        """
        if min_identity is not None and not 0 < min_identity <= 1:
            raise ValueError('min_identity must be in (0, 1].')
        if max_pairs < 1:
            raise ValueError('max_pairs must be positive.')
        sequence_groups, content_groups, sequences, issues = {}, {}, [], []
        for i, row in enumerate(self.all_rows):
            record = ProteinRecord.load(row['_resolved'])
            tokens = record.seq.cpu().numpy().tobytes()
            sequence_key = hashlib.sha256(tokens).hexdigest()
            content = hashlib.sha256(tokens)
            for x in (record.seq_pos, record.present, record.xyz[record.present]):
                content.update(x.cpu().numpy().tobytes())
            sequences.append(record.seq.tolist())
            for name, key, groups in (('sequence', sequence_key, sequence_groups),
                                       ('content', content.hexdigest(), content_groups)):
                previous = groups.setdefault(key, [])
                for j in previous:
                    if row['split'] != self.all_rows[j]['split']:
                        issues.append({'kind': name, 'rows': [j, i]})
                previous.append(i)
        checked, eligible = 0, 0
        if min_identity is not None:
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                                      open_gap_score=-10, extend_gap_score=-0.5)
            from .constants import AA1
            strings = [''.join(AA1[a] if a < 20 else 'X' for a in seq) for seq in sequences]
            splits = defaultdict(list)
            for i, row in enumerate(self.all_rows):
                splits[row['split']].append(i)
            groups = list(combinations(splits.values(), 2))
            eligible = sum(len(a)*len(b) for a, b in groups)
            pairs = ((i, j) for a, b in groups for i in a for j in b)
            for i, j in islice(pairs, max_pairs):
                checked += 1
                left, right = strings[i], strings[j]
                alignment = aligner.align(left, right)[0]
                matches = 0
                for (a, b), (c, d) in zip(alignment.aligned[0], alignment.aligned[1]):
                    matches += sum(x == y and x != 'X' for x, y in zip(left[a:b], right[c:d]))
                identity = matches/max(len(left), len(right))
                if identity >= min_identity:
                    issues.append({'kind': 'sequence_identity', 'rows': [i, j],
                                   'identity': identity})
        complete = checked == eligible
        return {'passed': not issues and complete, 'records': len(sequences), 'issues': issues,
                'identity_threshold': min_identity, 'pairs_checked': checked,
                'pairs_eligible': eligible, 'complete': complete,
                'scope': 'Exact sequence/content duplicates; optional global identity screen. '
                         'Declared homology clusters still need external provenance.'}


class SyntheticDataset:
    def __init__(self, count=8, length=32, seed=17):
        self.count, self.length, self.seed = count, length, seed
        if count < 1 or length < 1:
            raise ValueError("Invalid synthetic dataset size.")
        self.fingerprint = f"synthetic:{count}:{length}:{seed}"

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return synthetic_record(self.length, self.seed+int(i))
