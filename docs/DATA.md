# 데이터 준비 및 계약

## 지원 단위

Record는 **하나의 protein chain**이다. v0.4부터는 생성 시점에 강제한다. 모든 residue ID(`chain:number:icode`)의 chain이 같아야 하고, 다르면 거부한다. 학습 crop은 이 chain 안에서만 잘린다. 여러 chain이 있는 구조는 chain마다 따로 `prepare --chain X`로 record를 만든다. PDB/mmCIF의 첫 model을 읽고 표준 residue와 MSE→MET 정규화를 지원한다. 여러 model/conformer, multichain complex, ligand/ion, noncanonical chemistry를 자동으로 하나의 record에 섞지 않는다.

입력 구조는 이미 선택한 biological state여야 한다. 이 프로젝트는 protonation, missing sidechain reconstruction, force-field relaxation, structure prediction을 실행하지 않는다. Hydrogen은 feature slot에 포함하지 않는다.

## Canonical sequence map

PDB author number와 insertion code는 residue key이며 서열 index가 아니다. 다음처럼 원래 sequence의 모든 위치를 열거한다.

```json
[
  {"position": 0, "residue_id": "A:100:", "aa": "M"},
  {"position": 1, "residue_id": "A:100:A", "aa": "K"},
  {"position": 2, "residue_id": null, "aa": "G"},
  {"position": 3, "residue_id": "A:102:", "aa": "L"}
]
```

`null`은 sequence는 알지만 해당 residue의 구조가 관측되지 않은 위치다. `position`은 연속해야 하며 `aa`는 한 글자여야 한다. 관측된 모든 residue는 정확히 한 번 mapping되어야 한다. 기록된 sequence와 구조 residue identity가 다르면 오류를 낸다.

```bash
protein-jepa prepare protein.pdb --chain A \
  --sequence-map sequence_map.json --output data/protein_A.npz
```

Sequence map을 주지 않으면 observed residue 순서를 유지하되 `position_source=observed_order_unverified`로 표시한다. **원래 서열의 missing-loop 길이를 알고 있다고 주장하지 않는다.** 실제 pretraining config는 기본적으로 이 record를 거부한다. 먼저 소규모 구조-only 파일을 확인하는 pilot에서는 명시적으로 `allow_observed_order: true`를 사용할 수 있다.

자동 SEQRES/entity_poly_seq 정렬, 외부 UniProt mapping 및 homology clustering은 구현 범위 밖이다. 잘못 추정한 map을 자동 생성하는 것보다 명시적 map을 요구한다.

## 원자 및 masking

`present`는 원본 관측이다. Crop/JEPA masking은 별도의 `visible` mask다. `(0,0,0)`은 유효한 좌표다. 부재 원자 값을 바꿔도 context 결과가 바뀌면 안 된다.

PDB alternate location은 residue별 합산 occupancy가 가장 높은 label 하나를 선택하고 blank atoms를 함께 사용한다. Atom별로 서로 다른 conformer를 제멋대로 고르는 방식은 피한다. Occupancy가 0인 원자는 관측으로 포함하지 않는다.

Peptide edge는 consecutive canonical positions, C/N presence, 0.8–1.9 Å C–N 거리의 보수적 조건으로 생성한다. 이는 ingestion 시 결정한 topology이며, context mask를 위해 감춘 좌표에서 다시 유추하지 않는다. Missing endpoints를 연결된 chain으로 가정하지 않는다.

χ `defined`는 residue chemistry에서, `valid`는 필요한 네 원자의 presence 및 nondegeneracy에서 얻는다. Backbone φ/ψ/ω와 Cα pseudo-angle/dihedral은 chain break 및 crop 밖의 원자를 참조하면 invalid다.

## PLMol 연결

기존 `Protein.featurize(mode="all")`나 82-dim residue feature bundle을 그대로 가져오지 않는다. 그 경로는 이 모델이 원하지 않는 annotation/identity feature를 함께 포함할 수 있다.

```python
from plmol.parsers import PDBParser
from protein_jepa.data.io import from_plmol_parser

parser = PDBParser("protein.pdb")
record = from_plmol_parser(parser, chain="A", sequence_map=sequence_map)
record.save("data/protein_A.npz")
```

이 adapter는 `protein_atoms`의 `atom_name, res_name, res_num, chain_id, coords, insertion_code, occupancy, alt_loc` 필드를 읽는다. 전체 feature extraction이나 SASA를 호출하지 않는다. Optional PLMol runtime 자체는 제작 환경에 설치되어 있지 않아, 필드 계약 fixture를 통한 테스트만 수행했다. `eightmm/plmol` 원본 코드는 수정하지 않았다.

## Manifest와 split

```jsonl
{"path":"train/p1.npz","split":"train","cluster_id":"seq-cluster-0001"}
{"path":"train/p2.npz","split":"train","cluster_id":"seq-cluster-0002"}
{"path":"val/p3.npz","split":"val","cluster_id":"seq-cluster-0100"}
{"path":"test/p4.npz","split":"test","cluster_id":"seq-cluster-0200"}
```

Path는 manifest 파일 기준 상대 경로 또는 절대 경로다. 같은 cluster가 여러 split에 있으면 거부한다. 같은 record path가 반복되어도 거부한다. Cluster ID가 없으면 임의로 protein ID를 substitute하지 않는다.

```bash
protein-jepa audit-manifest data/manifest.jsonl
```

이 명령은 **사용자가 선언한 cluster/split consistency**를 검사한다. 서열 유사도를 계산하거나 잘못 부여한 cluster ID를 검증하는 clustering tool이 아니다. Sequence clustering은 사전에 수행하고 provenance를 별도로 기록해야 한다. Crop은 split 이후에 만든다.

## 데이터 선택 기준

실제 dataset acquisition은 자동 실행하지 않는다. 먼저 허가된 protein coordinates와 corresponding sequence를 준비한다. Observed atomic completeness, chain length, resolution/confidence 등의 필터 기준은 corpus manifest 제작 단계에 기록한다. Predicted structure를 섞으면 모델의 target이 실험값과 예측값의 혼합임을 명시한다.

배포 archive에는 실제 연구 데이터나 사전학습 checkpoint를 포함하지 않는다. Synthetic fixtures는 순수 코드로 생성하며 chemical/energetic validity가 보장된 단백질이 아니다.
