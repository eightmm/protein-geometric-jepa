# References and provenance

Primary sources below explain API contracts and research motivation. This repository is an original implementation of the conversation's design, **not** an exact reproduction of Mol-JEPA, I-JEPA, AlphaFold, or a latent-geometry theorem.

## cuEquivariance — consulted official API documentation

- NVIDIA: [FullyConnectedTensorProduct](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.FullyConnectedTensorProduct.html). Used to specify shared/internal weights, layout, and naive/fused tensor-product methods.
- NVIDIA: [SphericalHarmonics](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SphericalHarmonics.html). Used to define l=0/1/2 directional features and calibrate the Cartesian basis bridge.
- NVIDIA: [Irreps API](https://docs.nvidia.com/cuda/cuequivariance/api/cuequivariance.html). Used for SO(3) multiplicities/layout concepts.

The documentation was consulted, but the optional backend could not be executed in the build environment. API verification is not a substitute for running the included CuEq tests on the installed package version.

## Research motivation

- [Mol-JEPA, arXiv:2608.22642](https://arxiv.org/abs/2608.22642). The shared-entity, masked-modality representation-prediction idea motivated the view/task organization. Our residue/atom granularity, equivariant geometry, masking policies, EMA target encoder and circular latent heads are project design choices; do not attribute all of them to Mol-JEPA.
- User-supplied [arXiv:2609.21656](https://arxiv.org/abs/2609.21656), discussed under the title *Beyond Gaussian Worlds: Latent Geometry Matters for JEPAs*. The article could not be fetched during this implementation turn. It is recorded as motivation, **not as a source independently verified for a theorem reproduced here**. This code does not implement its heat-kernel MMD or claim its guarantees for proteins.
- [VICReg, arXiv:2105.04906](https://arxiv.org/abs/2105.04906). Background for variance/covariance regularization. The implemented small-batch, balanced-node, rank-local variant is a heuristic project baseline rather than a full experimental reproduction.

Physical periodicity, learned circular channel geometry and an assumed latent probability distribution are separate concepts. The code does not force actual protein torsion distributions to be uniform or identify each learned circle with one specific physical angle.

## PLMol integration provenance

- [eightmm/plmol](https://github.com/eightmm/plmol): the connected repository was read to confirm the raw parser interface; it was not changed.
- [ParsedAtom/PDBParser implementation](https://github.com/eightmm/plmol/blob/9e528651369ae1718022de4691009d3db903c17c/plmol/parsers/pdb_parser.py). Read contract fields: chain_id, res_num, insertion_code, res_name, atom_name, coords, element, alt_loc, occupancy. The fetched parser blob SHA was `eab57d1613b95bf80197ee5f52f5a65d468504d7`.
- Our adapter consumes `parser.protein_atoms`; it intentionally does not invoke PLMol's mixed featurizer or SASA pipeline. Tests emulate the field contract; an installed-PLMol integration run is still required in deployment.

## Publication

- GitHub CLI: [gh repo create](https://cli.github.com/manual/gh_repo_create). Used for the user-run `--private --source --remote --push` publishing helper.

No credentials, private datasets, model weights or external font files are included in this source distribution.
