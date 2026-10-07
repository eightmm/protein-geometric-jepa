# Protein Geometric JEPA

**Sequence ↔ backbone ↔ sidechain/all-atom, with node-wise typed latent prediction.**

연구 구현 v0.1. 단백질을 global vector 하나로만 압축하지 않고 **atom / residue / global 표현**을 함께 유지합니다. Sequence Transformer, backbone-only SO(3) GNN, sidechain atom branch, all-atom fusion, backbone internal-coordinate encoder, chi encoder가 포함됩니다. **Sidechain과 χ는 제외하지 않았습니다.**

## 현재 상태

CPU reference backend의 학습·역전파·equivariance·mask leakage·checkpoint resume를 실제 테스트했습니다. `reports/`에는 실행 결과가 있습니다. CuEquivariance용 실제 연산 경로도 구현했지만, 제작 환경에 패키지 설치용 네트워크와 GPU가 없어 **CuEq CPU/GPU 경로는 실행 미검증**입니다. 선택한 backend가 없으면 오류를 내며 reference로 몰래 바꾸지 않습니다.

**사전학습된 가중치나 downstream 성능 결과는 제공하지 않습니다.** Synthetic demo는 동작 검증용이며, 실제 단백질 모델의 품질을 입증하지 않습니다. 상세 범위는 [구현 상태](docs/STATUS.md)를 확인하세요.

## 빠른 실행

Python 3.11 이상 환경에서:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
protein-jepa demo --config configs/smoke.yaml --output runs/smoke
```

PowerShell 활성화는 `.venv\Scripts\Activate.ps1`입니다. 설치 없이 기존 환경에서 실행하려면 Linux에서는 `PYTHONPATH=src python -m protein_jepa.cli ...`를 사용할 수 있습니다.

128/256 crop은 `configs/reference.yaml`, `configs/cueq_gpu.yaml`에서 사용합니다. `smoke.yaml`만 실행 시간을 줄이기 위해 16/24를 사용합니다.

## 모델 구성

```text
Sequence ── Transformer ────────────────────────────── H_seq + CLS_seq

N/CA/C/O ── BB atom stem ── BB residue GNN ─────────── H_BB + G_BB(0,1,2)
                                   │
SC atoms ── SC atom stem ───────────┼── AA fusion ───── H_AA_atom + H_AA + G_AA
                                   │
BB angles/torsions ── internal-coordinate Transformer ─ H_BB_internal
Chi angles + symmetry/validity ── chi Transformer ───── H_chi

Visible view states ── conditional predictor ── masked learned target latents
                                          └── node / atom / parent-crop global
```

BB output을 SC/AA feature로 덮어쓰지 않습니다. Cartesian GNN은 scalar, Cartesian vector, symmetric-traceless tensor를 유지합니다. **l=2는 수학적으로 5차원이며 reference 구현에서는 3×3 STF tensor로 저장**합니다. CuEq의 5-component irrep basis와는 명시적으로 변환합니다.

## 입력과 target

- 구조 입력: 좌표, 연결성, backbone direction, local frame, φ/ψ/ω, Cα pseudo-angle/pseudo-dihedral, sidechain geometry 및 χ.
- 제외: SASA, DSSP, secondary-structure rule label, residue physicochemical lookup, pretrained PLM embedding의 구조 경로 유입.
- Prediction target: learned semantic/equivariant/circular representations. Raw xyz/angle reconstruction loss로 JEPA를 대체하지 않습니다.

Sequence-only context에는 임의의 world-frame l>0 target을 강제하지 않습니다. Geometry context가 있을 때만 equivariant latent loss를 활성화합니다.

## 실제 구조 데이터

```bash
protein-jepa prepare protein.cif --chain A \
  --sequence-map sequence_map.json --output data/protein_A.npz
protein-jepa audit-manifest data/manifest.jsonl
protein-jepa train --config configs/reference.yaml \
  --manifest data/manifest.jsonl --output runs/pretrain
```

`sequence_map`은 원래 서열의 결손 위치까지 명시합니다. 이것이 없으면 `observed_order_unverified`로 기록하며, 학습기는 기본적으로 거부합니다. Pilot에서만 `allow_observed_order: true`를 명시할 수 있습니다. [데이터 명세](docs/DATA.md)에 형식과 전처리 조건이 있습니다.

## CuEquivariance

```bash
python -m pip install -e '.[dev,cueq]'
pytest -q -m cueq
protein-jepa demo --config configs/cueq_naive.yaml --output runs/cueq_cpu
```

GPU에서는 CUDA 버전에 맞는 NVIDIA CuEq ops wheel도 설치하고 다음을 먼저 통과시킵니다.

```bash
pytest -q -m cuda
protein-jepa train --config configs/cueq_gpu.yaml \
  --manifest data/manifest.jsonl --output runs/cueq_gpu
```

`reference`와 `cueq-*`는 동일한 SO(3) 입출력 계약을 공유하지만 **동일 weight layout/동일한 parameterization의 두 kernel이 아닙니다.** Backend를 바꾸어 기존 checkpoint를 이어 학습하지 않습니다. [CuEq 계약](docs/CUEQ.md) 참조.

## 체크포인트 및 downstream

```bash
protein-jepa demo --config configs/smoke.yaml --output runs/resume --stop-after 9
protein-jepa demo --config configs/smoke.yaml --output runs/resume --resume runs/resume/last.pt

protein-jepa encode --checkpoint runs/smoke/last.pt \
  --sequence MARGKKIGYS --mode sequence --output runs/sequence_features.pt
protein-jepa encode --checkpoint runs/smoke/last.pt \
  --record data/protein_A.npz --mode all_atom --output runs/atom_features.pt
```

`last.pt`에는 online encoder, EMA teacher, predictor, optimizer, scheduler, rank별 RNG, task step과 configuration이 들어갑니다. 동일 world size/configuration에서 resume합니다. Encoder export와 invariant task-head 예제는 [API](docs/API.md)를 참고하세요.

## 문서

| 문서 | 내용 |
|---|---|
| [SPEC_KO.md](docs/SPEC_KO.md) | 전체 설계, 정보 접근 경계, 수학적 계약, task와 loss |
| [DATA.md](docs/DATA.md) | Canonical indexing, PDB/mmCIF, PLMol adapter, manifest |
| [CUEQ.md](docs/CUEQ.md) | Backend 구현과 basis bridge, GPU 검증 gate |
| [TRAINING.md](docs/TRAINING.md) | 단일 장치/DDP/Slurm, resume, 평가 |
| [API.md](docs/API.md) | Encoder/downstream 출력과 예제 |
| [VALIDATION.md](docs/VALIDATION.md) | 실제 테스트 결과와 검증 범위 |
| [PUBLICATION.md](docs/PUBLICATION.md) | 새 private GitHub 게시 절차 |
| [STATUS.md](docs/STATUS.md) | 구현됨·검증됨·미검증·후속 연구 구분 |
| [REFERENCES.md](docs/REFERENCES.md) | 1차 출처와 확인 범위 |

## GitHub 게시

이 패키지는 원격 저장소 생성과 별개로 완성된 로컬 Git 프로젝트입니다. GitHub CLI가 인증된 컴퓨터에서 다음을 실행하면 **새 private 저장소를 생성하고** 소스를 게시합니다. 기존 저장소가 있으면 멈추며 force-push하지 않습니다.

```bash
python scripts/publish_github.py --repo eightmm/protein-geometric-jepa
```

원격 생성/게시를 수행하지 않은 상태에서는 GitHub URL을 완료된 결과처럼 표시하지 않습니다. 원본 `eightmm/plmol`은 수정하지 않습니다. 공개 라이선스는 소유자의 결정을 위해 설정하지 않았으며 게시 기본값은 private입니다.
