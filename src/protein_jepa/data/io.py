"""PDB/mmCIF and PLMol raw-parser adapters; no SASA or annotation calls."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from .constants import AA3, AA1, AA_TO_ID, ATOM_ID, N_ATOMS, UNK
from .records import ProteinRecord


def from_atoms(atoms, chain: str | None = None, record_id: str = "protein",
               sequence_map: list[dict] | None = None) -> ProteinRecord:
    """Accept objects with PLMol ParsedAtom fields.

    sequence_map = [{"position": 0, "residue_id": "A:1:", "aa": "M"}, ...].
    residue_id=null inserts a sequence position with no observed coordinates.
    The map must cover all observed residues selected from this chain.
    """
    atoms = [a for a in atoms if a.res_name.strip().upper() in AA_TO_ID or a.res_name.strip().upper() == "MSE"]
    chains = list(dict.fromkeys(a.chain_id for a in atoms))
    if chain is None:
        if len(chains) != 1:
            raise ValueError(f"Select exactly one chain from {chains}.")
        chain = chains[0]
    selected = [a for a in atoms if a.chain_id == chain]
    groups = {}
    for a in selected:
        name = a.res_name.strip().upper()
        if name not in AA_TO_ID and name != "MSE":
            continue
        key = f"{a.chain_id}:{int(a.res_num)}:{str(a.insertion_code).strip()}"
        groups.setdefault(key, []).append(a)
    if not groups:
        raise ValueError("No supported protein residues in the selected chain.")
    keys = sorted(groups, key=lambda k: (int(k.split(":")[-2]), k.split(":")[-1]))
    names = {k: ("MET" if groups[k][0].res_name.strip().upper() == "MSE" else groups[k][0].res_name.strip().upper())
             for k in keys}
    if sequence_map is None:
        entries = [{"position": i, "residue_id": k,
                    "aa": AA1[AA_TO_ID[names[k]]]} for i, k in enumerate(keys)]
        source = "observed_order_unverified"
    else:
        entries = sequence_map
        mapped = [e["residue_id"] for e in entries if e.get("residue_id") is not None]
        if len(mapped) != len(set(mapped)) or set(mapped) != set(keys):
            raise ValueError("sequence_map must map each observed residue exactly once.")
        source = "canonical"
    n = len(entries)
    xyz = torch.zeros(n, N_ATOMS, 3, dtype=torch.float32)
    present = torch.zeros(n, N_ATOMS, dtype=torch.bool)
    seq = torch.full((n,), UNK, dtype=torch.long)
    pos = torch.tensor([int(e["position"]) for e in entries], dtype=torch.long)
    ids = []
    for i, e in enumerate(entries):
        aa = e["aa"].upper()
        if len(aa) != 1 or aa not in AA1+'X':
            raise ValueError('sequence_map aa must be one standard one-letter code or X.')
        seq[i] = AA1.index(aa) if aa in AA1 else UNK
        key = e.get("residue_id")
        ids.append(key if key is not None else f"{chain}:missing:{int(pos[i])}")
        if key is None:
            continue
        if seq[i] != AA_TO_ID[names[key]] and seq[i] != UNK:
            raise ValueError(f"Sequence-map mismatch at {key}.")
        # Choose one residue-level alternate label; blank atoms are shared.
        alt_scores = {}
        for a in groups[key]:
            alt = str(getattr(a, "alt_loc", "")).strip()
            if alt:
                alt_scores[alt] = alt_scores.get(alt, 0.0) + float(a.occupancy or 0.0)
        chosen = min(alt_scores, key=lambda s: (-alt_scores[s], s)) if alt_scores else ""
        for a in sorted(groups[key], key=lambda a: float(a.occupancy or 0.0)):
            alt = str(getattr(a, "alt_loc", "")).strip()
            if alt and alt != chosen:
                continue
            name = a.atom_name.strip()
            if a.res_name.strip().upper() == "MSE" and name == "SE":
                name = "SD"  # Explicit normalized MSE -> MET geometry convention.
            if name not in ATOM_ID or float(a.occupancy or 0.0) <= 0:
                continue
            j = ATOM_ID[name]
            xyz[i, j] = torch.as_tensor(a.coords, dtype=torch.float32)
            present[i, j] = True
    # Conservative evidence for a peptide bond. Missing endpoints are NOT bonds.
    distance = (xyz[:-1, 2] - xyz[1:, 0]).norm(dim=-1)
    peptide = present[:-1, 2] & present[1:, 0] & (distance > 0.8) & (distance < 1.9)
    peptide &= pos[1:] == pos[:-1] + 1
    return ProteinRecord(xyz, present, seq, pos, peptide, tuple(ids), record_id, source)


def read_structure(path: str | Path, chain: str | None = None,
                   sequence_map: str | Path | list[dict] | None = None) -> ProteinRecord:
    """Read the first structural model; caller explicitly chooses the chain.

    Unmapped structures retain observed-order positions, not invented canonical
    residue indices. Training must opt in to this less strict indexing mode.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    atoms = []
    for ch in model:
        if chain is not None and ch.id != chain:
            continue
        for residue in ch:
            for atom in residue.get_unpacked_list():
                atoms.append(SimpleNamespace(
                    chain_id=ch.id, res_num=residue.id[1], insertion_code=residue.id[2].strip(),
                    res_name=residue.resname, atom_name=atom.name, coords=atom.coord,
                    element=atom.element, occupancy=atom.occupancy,
                    alt_loc=atom.altloc.strip(),
                ))
    if isinstance(sequence_map, (str, Path)):
        sequence_map = json.loads(Path(sequence_map).read_text())
    return from_atoms(atoms, chain, path.stem, sequence_map)


def from_plmol_parser(parser, chain: str | None = None, sequence_map=None) -> ProteinRecord:
    """Use PLMol PDBParser.protein_atoms directly, not its mixed feature bundle.

    Verified against the public ParsedAtom field contract; no PLMol dependency
    is required merely to import this module.
    """
    if not hasattr(parser, "protein_atoms"):
        raise TypeError("Expected a PLMol raw parser exposing protein_atoms.")
    return from_atoms(parser.protein_atoms, chain, "plmol", sequence_map)
