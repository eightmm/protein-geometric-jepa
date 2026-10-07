# 실행 검증 보고서

## 판정

**CPU reference 구현의 기능 테스트는 통과했습니다. CuEquivariance 실행 및 실제 단백질 downstream 성능은 아직 검증하지 않았습니다.**

| 검증 | 결과 | 증거 |
|---|---|---|
| 전체 pytest | 72 passed / 3 skipped | [pytest.txt](../reports/pytest.txt), [JUnit](../reports/pytest.xml) |
| 9개 task forward/backward | 통과 | `tests/test_training.py` |
| 모든 online encoder에 gradient 도달 | 통과 | `test_all_encoder_paths_receive_gradients_across_tasks` |
| BB / SC / AA / predictor equivariance | 통과 | `tests/test_models.py` |
| Hidden coordinate mutation에 대한 context 불변 | 통과 | `tests/test_geometry.py`, `tests/test_models.py` |
| BB-only 경로의 SC/sequence isolation | 통과 | `tests/test_models.py` |
| Frame / origin / chain break / short chain | 통과 | `tests/test_geometry.py` |
| χ validity / dependency / π naming symmetry | 통과 | `tests/test_geometry.py` |
| Numeric NPZ / canonical indexing / insertion codes | 통과 | `tests/test_data.py` |
| 실제 PDB/mmCIF reader roundtrip | 합성 fixture에서 통과 | `tests/test_cli.py` |
| CLI prepare → train → encode → evaluate | 합성 fixture에서 통과 | `tests/test_cli.py` |
| Checkpoint exact resume | Single CPU에서 bitwise 일치 | `test_resume_is_exact`, dropout=0.1 |
| 18-step optimizer/EMA 학습 | 통과 | [reference log](../reports/reference_smoke.log) |
| 128/256 길이 × 9 tasks | 18/18 finite backward | [crop report](../reports/crop_sizes.json) |
| 2-rank CPU/Gloo, 9 tasks | 통과 | [DDP log](../reports/ddp_cpu_all_tasks.log) |
| Editable installation / CLI entry point | 통과 | [install log](../reports/package_install.log), [help](../reports/cli_help.txt) |
| CuEq naive / CUDA | 실행하지 못함; 3 tests skipped | package/GPU 부재 |
| Remote source import | 사용자 생성 저장소에 게시 완료, 원본 76파일 tree 일치 | [import commit](https://github.com/eightmm/protein-geometric-jepa/commit/2b2ee750edd53c3d1e5b836b77d71d63b0b370e1) |
| 게시 전 CPU 재검증 | 72 passed / 3 skipped | [publication_pytest.txt](../reports/publication_pytest.txt) |

## 환경

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- Biopython 1.86
- Pytest 9.0.2
- CUDA GPU 및 CuEq package 없음

정확한 환경은 [environment.json](../reports/environment.json)에 기록했습니다. 이 목록은 CPU 검증 환경이며 CUDA용 권장 dependency pinning 목록이 아닙니다.

## 테스트 중 수정한 오류

초기 equivariance 검사에서 graph의 일부 self-neighbor가 rotation에 따라 달라지는 문제가 검출되었습니다. 원인은 distance 계산의 matrix-multiplication 경로에서 자기 거리의 부동소수점 반올림값이 0이 아닌 값으로 나타나고, 이를 거리 epsilon만으로 필터링하던 점이었습니다.

수정은 다음 두 가지입니다.

1. Chunked distance 계산에서 직접 거리 계산 경로를 사용합니다.
2. Self-edge 제거는 좌표 거리 epsilon이 아니라 source/destination index equality를 기준으로 수행합니다.

허용 오차만 완화한 것이 아니라 edge construction을 수정했고, self-edge 회귀 테스트를 추가했습니다. Chain/covalent edge는 별도 관계로 유지합니다.

## 재현 명령

```bash
python -m pip install -e '.[dev]'
pytest -q --junitxml=reports/pytest-local.xml
protein-jepa demo --config configs/smoke.yaml --output runs/local_smoke --length 32 --count 8
python scripts/validate_crop_sizes.py --output runs/local_crop_validation.json
```

2-rank CPU 검증에는 `configs/smoke.yaml`을 복사해 `steps: 9`, `batch_size: 1`, `crop_lengths: [8]`, `threads: 1`, `device: cpu`로 설정한 뒤 다음을 실행합니다.

```bash
torchrun --standalone --nproc_per_node=2 -m protein_jepa.cli demo \
  --config configs/your_ddp_smoke.yaml --output runs/local_ddp --length 10 --count 4
```

원격에 게시한 뒤 CPU Actions workflow가 실행되기 시작했습니다. 실제 커밋별 결과는 [Actions](https://github.com/eightmm/protein-geometric-jepa/actions)에서 확인합니다. 이 문서의 로컬 결과를 GitHub CI의 결과로 대신 해석하지 마세요.

## 이 결과가 보장하지 않는 것

Finite loss는 representation이 유용하거나 collapse가 없다는 증명이 아닙니다. 합성 sidechain은 physical-quality 검사 대상이 아닙니다. 128/256 결과는 작은 검증 모델의 shape/backward 안정성이며, 대형 모델 throughput/memory 추정치가 아닙니다.

실제 corpus 사전학습, homology-separated downstream, GPU/CuEq correctness·성능, production-level scale-out는 별도 실행이 필요합니다. 공개 checkpoint는 포함하지 않았습니다.
