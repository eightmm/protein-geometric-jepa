# Protein Geometric JEPA

**Sequence ↔ backbone ↔ sidechain/all-atom, with node-wise typed latent prediction.**

연구 구현 **v0.3.0**. Atom / residue / global 표현을 함께 유지하며, Sequence Transformer, backbone-only SO(3) GNN, SC atom branch, AA fusion, backbone internal-coordinate encoder와 χ encoder를 포함합니다. **Sidechain과 χ를 제외하지 않았습니다.**

## v0.3: JEPA target·predictor 결함 수정

v0.2를 실행해 확인한 결함을 고쳤습니다.

- Target projector가 학습되지 않아 l>0 target이 고정된 무작위 사영이었습니다. 이제 target은 teacher encoder 상태를 parameter 없이 정규화한 값입니다(soft RMS floor와 log 크기 scalar).
- Predictor는 cross-attention 1층에서 2단계 equivariant transformer로 바뀌었습니다. Joint stack 뒤에 atom decoder가 있고, relative position bias와 bilinear FFN을 씁니다.
- Atom query는 관측된 원자 목록이 아니라 visible sequence의 topology로 만듭니다.
- Mask는 겹치지 않는 multi-block span입니다. Task는 sample마다 섞이고, EMA는 schedule을 따릅니다.
- Reference message가 CG 경로 전체를 갖습니다.

Encoder 확장 네 가지(`sc_context: spatial`, `effdock_directional`, `effdock_ffn: bilinear`, `effdock_adaptive_cutoff`)는 opt-in 실험입니다. 네 가지를 모두 켠 설정은 `configs/jepa_full_cueq_gpu.yaml`입니다. **v0.2 checkpoint는 읽지 않습니다.** 계획 단계에서 Codex, Claude, Antigravity council의 검토를 거쳤습니다. 근거와 검증 범위는 [JEPA_V030_KO.md](docs/JEPA_V030_KO.md)에 있습니다. CPU에서 171 passed / 2 CUDA skips이고, 실제 CuEq 0.9.0 CPU 테스트도 실행되었습니다. GPU 학습과 품질 향상은 아직 검증하지 않았습니다.

## v0.2: EFF-Dock-inspired CuEq interaction

`model.interaction: effdock`로 BB atom, SC atom, BB residue, AA atom, AA residue의 다섯 단계에서 새 블록을 사용할 수 있습니다. Shared tensor product, dual radial scaling, directed edge-type decay, degree-wise RMSNorm, invariant conditioning, gated aggregation 및 residual equivariant FFN을 통합했습니다. 기존 설정의 기본값은 `baseline`이라 과거 모델을 조용히 바꾸지 않습니다.

**실제 CuEq 0.9.0 CPU 연산을 GitHub CI에서 검증했습니다.** 전용 테스트 7개, 18-step 학습, 9개 task 역전파가 통과했습니다. 별도 reference CI는 Python 3.11/3.12/3.13에서 통과했고, 로컬 full suite는 123 passed / 9 optional skips입니다. **CuEq CUDA fused kernel과 GPU 학습은 아직 실행 미검증**입니다. [정확한 검증 범위](docs/V020_VALIDATION.md)를 확인하세요.

사전학습된 가중치나 downstream 성능 향상 주장은 제공하지 않습니다. Synthetic demo는 동작 검증이며, EFF-Dock의 docking checkpoint와 호환되는 모델이 아닙니다. [강화 분석](docs/EFFDOCK_UPGRADE_KO.md)에 원본과의 차이, tradeoff, ablation 및 후속 우선순위를 기록했습니다.

## 빠른 실행

Python 3.11 이상 환경에서:

```bash
git clone https://github.com/eightmm/protein-geometric-jepa.git
cd protein-geometric-jepa
python -m pip install -e '.[dev]'
pytest -q
protein-jepa demo --config configs/effdock_smoke.yaml --output runs/effdock-smoke
```

기존 baseline smoke는 `configs/smoke.yaml`입니다. Small smoke만 16/24 crop을 쓰며, `configs/effdock_reference.yaml`과 `configs/effdock_cueq_gpu.yaml`은 **128/256 parent crop**을 사용합니다. Output directory는 새 경로를 사용하거나 명시적으로 resume해야 합니다.

## 모델과 정보 경계

```text
Sequence ─ Transformer ─────────────────────────── H_seq + CLS_seq
N/CA/C/O ─ BB atom stem ─ BB residue GNN ───────── H_BB + G_BB(0,1,2)
                                   │
SC atoms ─ SC atom stem ───────────┼─ AA fusion ── H_AA_atom + H_AA + G_AA
BB internal coordinates ─ independent Transformer ─ H_BB_internal
Chi + symmetry/validity ─ independent Transformer ─ H_chi
Visible view states + mask tokens ─ joint equivariant predictor ─ atom decoder
                                   └─ normalized EMA-teacher node/atom/global states
```

BB output은 SC/AA 정보로 덮어쓰지 않습니다. 좌표, polymer connectivity, backbone direction, frame, φ/ψ/ω, Cα pseudo-angle/dihedral, sidechain geometry와 χ를 사용합니다. SASA, DSSP, secondary-structure rule label이나 residue physicochemical lookup은 사용하지 않습니다.

구조 hidden state는 scalar, Cartesian vector, STF rank-2 tensor입니다. **l=2는 수학적으로 5차원**이며 3×3 reference 저장 형식과 CuEq의 5-component basis는 명시적으로 변환합니다. Sequence-only context에는 arbitrary world-frame l>0 target을 강제하지 않습니다. 기본 global target은 **선택한 parent crop 전체**이지 원래 단백질 전체가 아닙니다.

## 실제 CuEq 사용

```bash
python -m pip install 'cuequivariance==0.9.0' 'cuequivariance-torch==0.9.0'
python -m pip install -e '.[dev,cueq]'
python -c 'import cuequivariance, cuequivariance_torch'
pytest tests/test_cueq.py -m 'not cuda' -q
protein-jepa demo --config configs/effdock_cueq_naive.yaml --output runs/effdock-cueq-cpu
```

GPU는 호환되는 CUDA PyTorch와 matching CuEq ops wheel을 설치하고 [GPU 실행 gate](docs/UPGRADE_RUNBOOK.md)를 먼저 통과시킵니다. `reference`는 별도 analytic operator이며 CuEq가 없는 경우 몰래 대체되지 않습니다. Architecture와 backend 변경은 exact resume에서 거부합니다.

## 실제 데이터와 학습

```bash
protein-jepa prepare protein.cif --chain A \
  --sequence-map sequence_map.json --output data/protein_A.npz
protein-jepa audit-manifest data/train.jsonl
protein-jepa train --config configs/effdock_cueq_gpu.yaml \
  --manifest data/train.jsonl --output runs/effdock-pretrain
```

`sequence_map`은 원래 서열의 결손 위치까지 명시합니다. 없으면 `observed_order_unverified`이며 학습은 기본적으로 거부합니다. [DATA.md](docs/DATA.md)의 canonical indexing과 cluster-disjoint split 계약을 따르세요. Manifest 작성, 대규모 학습 및 downstream 검증을 자동 완료한 상태는 아닙니다.

## 검증과 ablation

```bash
python scripts/validate_effdock.py --config configs/effdock_smoke.yaml \
  --variants baseline effdock --lengths 128 256 --output runs/effdock-check.json
python scripts/make_effdock_ablations.py --base configs/effdock_cueq_gpu.yaml \
  --output runs/ablation-configs --seeds 17 29 43
```

두 번째 명령은 19개 variant(v0.2 interaction 10개, v0.3 encoder·predictor 실험 6개, masking·EMA·covariance 3개) × 3개 seed의 완전한 설정을 생성하며 **학습/job 제출을 시작하지 않습니다.** Equal-step/equal-compute 및 parameter-matched 비교를 구분하세요. 작은 smoke loss로 품질 순위를 판단하지 않습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [JEPA_V030_KO.md](docs/JEPA_V030_KO.md) | v0.3 결함 수정, target·predictor 정의, council 기록, 검증 범위 |
| [EFFDOCK_UPGRADE_KO.md](docs/EFFDOCK_UPGRADE_KO.md) | 원본 EFF-Dock 비교, 수식, 강화 tradeoff, 후속 분석 |
| [UPGRADE_RUNBOOK.md](docs/UPGRADE_RUNBOOK.md) | CPU/CuEq/GPU gate, 학습·resume·DDP·ablation |
| [V020_VALIDATION.md](docs/V020_VALIDATION.md) | 이번 릴리스의 실행 증거·CI·미검증 범위 |
| [SPEC_KO.md](docs/SPEC_KO.md) | 기본 JEPA 설계, view·mask·latent·loss 계약 |
| [DATA.md](docs/DATA.md) | PDB/mmCIF, PLMol adapter, sequence alignment, manifest |
| [CUEQ.md](docs/CUEQ.md) | Backend와 irrep basis 변환, compatibility |
| [TRAINING.md](docs/TRAINING.md), [API.md](docs/API.md) | 기본 학습·추론·downstream API |
| [STATUS.md](docs/STATUS.md) | 구현·검증·후속 연구의 구분 |
| [PUBLICATION.md](docs/PUBLICATION.md), [VALIDATION.md](docs/VALIDATION.md) | 역사적 v0.1 게시·검증 기록 |

소유자가 만든 **public** 저장소 설정을 유지합니다. `eightmm/EFF-Dock`와 `eightmm/plmol`은 변경하지 않았습니다. 공개 라이선스 선택은 소유자 결정 사항으로 남아 있습니다. 원본 v0.1의 76개 파일은 `2b2ee750edd53c3d1e5b836b77d71d63b0b370e1` 커밋에 보존되어 있습니다. 새 릴리스의 실험 증거는 `reports/v020/`에 별도로 기록합니다.
