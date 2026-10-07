"""Canonical one-chain record. Metadata never enters the geometry encoder implicitly."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import torch
from torch import Tensor
from .constants import N_ATOMS


@dataclass
class ProteinRecord:
    xyz: Tensor                       # [L, 37, 3], Angstrom
    present: Tensor                   # [L, 37], bool (never inferred from xyz)
    seq: Tensor                       # [L], 0..20; unknown = 20
    seq_pos: Tensor                   # [L], canonical sequence position
    peptide: Tensor                   # [L-1], observed/declared connectivity
    residue_ids: tuple[str, ...]       # author chain:number:icode
    record_id: str = "protein"
    position_source: str = "canonical"

    def __post_init__(self):
        n = len(self.seq)
        if self.xyz.shape != (n, N_ATOMS, 3) or self.present.shape != (n, N_ATOMS):
            raise ValueError("Expected xyz[L,37,3] and present[L,37].")
        if self.seq_pos.shape != (n,) or self.peptide.shape != (max(n-1, 0),):
            raise ValueError("Invalid sequence positions or peptide connectivity.")
        if len(self.residue_ids) != n or len(set(self.residue_ids)) != n:
            raise ValueError("Residue keys must be unique and aligned.")
        if n == 0:
            raise ValueError("Empty records are not trainable.")
        if self.xyz.dtype != torch.float32:
            raise TypeError("Record coordinates must be float32; mixed precision belongs inside the model.")
        if self.seq.dtype != torch.int64 or self.seq_pos.dtype != torch.int64:
            raise TypeError("Sequence tokens and positions must be int64.")
        if self.position_source not in {"canonical", "observed_order_unverified"}:
            raise ValueError("Unknown position_source.")
        if self.present.dtype != torch.bool or self.peptide.dtype != torch.bool:
            raise TypeError("Presence/connectivity must be boolean.")
        if not torch.isfinite(self.xyz[self.present]).all():
            raise ValueError("Observed coordinates must be finite.")
        if n > 1 and not (self.seq_pos[1:] > self.seq_pos[:-1]).all():
            raise ValueError("One-chain sequence positions must be strictly increasing.")
        if self.position_source == "canonical" and n > 1 and not (self.seq_pos[1:] == self.seq_pos[:-1]+1).all():
            raise ValueError("Canonical records must enumerate missing sequence positions explicitly.")
        if not ((self.seq >= 0) & (self.seq <= 20)).all():
            raise ValueError("Sequence tokens must be in [0,20].")

    def __len__(self):
        return len(self.seq)

    def to(self, device) -> "ProteinRecord":
        return replace(self, **{k: getattr(self, k).to(device) for k in
                               ("xyz", "present", "seq", "seq_pos", "peptide")})

    def crop(self, start: int, length: int) -> "ProteinRecord":
        """Contiguous row slice; corpus CLI defaults to verified canonical positions."""
        if start < 0 or length < 1 or start >= len(self):
            raise ValueError("Invalid crop.")
        stop = min(start + length, len(self))
        return replace(self, xyz=self.xyz[start:stop].clone(),
                       present=self.present[start:stop].clone(),
                       seq=self.seq[start:stop].clone(), seq_pos=self.seq_pos[start:stop].clone(),
                       peptide=self.peptide[start:max(start, stop-1)].clone(),
                       residue_ids=self.residue_ids[start:stop])

    def rigid_transform(self, rotation: Tensor, translation: Tensor) -> "ProteinRecord":
        x = self.xyz @ rotation.T + translation
        x = torch.where(self.present[..., None], x, torch.zeros_like(x))
        return replace(self, xyz=x)

    def save(self, path: str | Path):
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Record output must use the .npz suffix.")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k).detach().cpu().numpy() for k in
                  ("xyz", "present", "seq", "seq_pos", "peptide")}
        meta = dict(residue_ids=self.residue_ids, record_id=self.record_id,
                    position_source=self.position_source, schema_version=1)
        np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path) -> "ProteinRecord":
        # NPZ contains numeric arrays and a Unicode JSON string; never use pickle.
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            if meta.pop("schema_version") != 1:
                raise ValueError("Unsupported record schema.")
            meta["residue_ids"] = tuple(meta["residue_ids"])
            return cls(**{k: torch.from_numpy(a[k].copy()) for k in
                          ("xyz", "present", "seq", "seq_pos", "peptide")}, **meta)


def random_crop(record: ProteinRecord, lengths: list[int], generator: torch.Generator):
    k = lengths[int(torch.randint(len(lengths), (), generator=generator))]
    # Rows can include explicit missing-residue positions from sequence_map.
    # Without canonical mapping no assertion of fixed sequence span is made.
    k = min(k, len(record))
    start = int(torch.randint(len(record)-k+1, (), generator=generator))
    return record.crop(start, k)
