"""Manifest-based sampling; leakage audit precedes crop generation."""
from pathlib import Path
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
