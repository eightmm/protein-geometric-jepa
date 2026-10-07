"""Standard heavy-atom names and explicit torsion dependencies.

No residue properties, SASA, or secondary-structure annotations are included.
Tables describe chemistry, not learned targets. Noncanonical chemistry requires
an explicit extension rather than a fabricated atom layout.
"""
AA3 = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
AA1 = "ARNDCQEGHILKMFPSTWYV"
AA_TO_ID = {a: i for i, a in enumerate(AA3)}
UNK, MASK, CLS = 20, 21, 22
ATOMS = ("N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
         "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1", "ND2",
         "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2",
         "OG", "OG1", "OH", "SD", "SG", "OXT")
ATOM_ID = {a: i for i, a in enumerate(ATOMS)}
N_ATOMS = len(ATOMS)
ELEMENTS = {"C": 0, "N": 1, "O": 2, "S": 3, "SE": 4}
ATOM_ELEMENT = [ELEMENTS.get(name[0], 5) for name in ATOMS]
BB_IDS = (0, 1, 2, 3)

# Explicit quadruples; do not infer chi by slicing an atom array.
CHI = {
    "ALA": [], "GLY": [],
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ"),
            ("CD", "NE", "CZ", "NH1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}
# Zero-based chi slot indices whose terminal naming admits a pi shift.
PI_PERIODIC = {"ASP": {1}, "GLU": {2}, "PHE": {1}, "TYR": {1}, "ARG": {4}}

SC_BONDS = {
    "ALA": [], "GLY": [],
    "ARG": ["CB-CG", "CG-CD", "CD-NE", "NE-CZ", "CZ-NH1", "CZ-NH2"],
    "ASN": ["CB-CG", "CG-OD1", "CG-ND2"],
    "ASP": ["CB-CG", "CG-OD1", "CG-OD2"],
    "CYS": ["CB-SG"],
    "GLN": ["CB-CG", "CG-CD", "CD-OE1", "CD-NE2"],
    "GLU": ["CB-CG", "CG-CD", "CD-OE1", "CD-OE2"],
    "HIS": ["CB-CG", "CG-ND1", "CG-CD2", "ND1-CE1", "CD2-NE2", "CE1-NE2"],
    "ILE": ["CB-CG1", "CB-CG2", "CG1-CD1"],
    "LEU": ["CB-CG", "CG-CD1", "CG-CD2"],
    "LYS": ["CB-CG", "CG-CD", "CD-CE", "CE-NZ"],
    "MET": ["CB-CG", "CG-SD", "SD-CE"],
    "PHE": ["CB-CG", "CG-CD1", "CG-CD2", "CD1-CE1", "CD2-CE2", "CE1-CZ", "CE2-CZ"],
    "PRO": ["CB-CG", "CG-CD", "CD-N"],
    "SER": ["CB-OG"], "THR": ["CB-OG1", "CB-CG2"],
    "TRP": ["CB-CG", "CG-CD1", "CG-CD2", "CD1-NE1", "NE1-CE2", "CD2-CE2",
            "CD2-CE3", "CE2-CZ2", "CE3-CZ3", "CZ2-CH2", "CZ3-CH2"],
    "TYR": ["CB-CG", "CG-CD1", "CG-CD2", "CD1-CE1", "CD2-CE2", "CE1-CZ", "CE2-CZ", "CZ-OH"],
    "VAL": ["CB-CG1", "CB-CG2"],
}
