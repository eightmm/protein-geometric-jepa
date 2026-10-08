# 설계 명세 대응표

기준 문서는 소유자의 설계 대화에 있는 구현 설계안(메시지 [66], "Protein Multi-View Geometric JEPA 구현 설계안")이다. 절마다 무엇이 어디에 구현되어 있고 어떤 테스트가 그것을 확인하는지 적는다. "보류"는 명세가 downstream 평가 이후 단계로 둔 항목이다.

## 데이터와 기하 (5–7, 14, 16절)

| 절 | 요구 | 구현 | 확인 |
|---|---|---|---|
| 5 | residue key, 원자–residue mapping, 좌표와 mask 분리 | `data/records.py` (`present`는 좌표에서 추론하지 않음), `(atom_residue, atom_slot)` | `test_data.py` |
| 6.1–6.2 | backbone 좌표·방향·결합각·φψω·frame·virtual Cβ, Cα 각/이면각, 순방향 방향 | `geometry/features.py::backbone_features` | `test_geometry.py` |
| 6.3 | residue pair 기하 `R_iᵀ(x_j−x_i)`, `R_iᵀR_j` (불변) | `data/graphs.py::frame_pair_features`, residue graph edge 입력 (`pair_frame_features`) | `test_frame_pair_features_convention_and_invariance` |
| 7.1 | SC 원자 좌표, 원소, 결합, backbone anchor, (선택) `R_iᵀ(x_a−x_CA)` | `AtomStem`; local frame 좌표는 `sc_local_frame` (기본 off: 22개 단백질에서 retrieval 변화 없이 SC latent 크기를 키워 전체 effective rank를 49→3으로 낮춤, [기록](../reports/spec_completion/README.md)) | 회전 등변·숨긴 좌표 변경 테스트 |
| 7.2 | χ1–χ4, defined/observed/periodicity, χ5 확장 슬롯 | `chi_features` (lookup table) | `test_geometry.py` |
| 7.3 | 대칭 원자 이름 | χ는 `p=2` 인코딩, SC/AA encoder는 원소 token만 사용, atom loss는 AlphaFold renaming에 대해 residue별 최소값 | `test_symmetric_atom_naming_is_not_a_learnable_difference` |
| 14.1–14.2 | 128/256 연속 parent crop, target = crop 전체 | `random_crop` (chain 내부, CA 관측 ≥ 50%) | `test_data.py` |
| 14.3 | crop → 더 큰 parent / 전체 단백질 global | **보류**: 명세가 downstream 평가 뒤 확장 실험으로 둔다. teacher와 student가 서로 다른 record를 보는 구조라 현재 packed batch 계약 밖이다 | — |
| 14.4, 16 | crop 경계·결손·chain break·mask의 의존성 닫힘, visible 좌표로 graph 재구성 | feature별 의존 원자 mask, `make_graph`는 visible 좌표만 | 경계·chain break·숨긴 좌표 변경 테스트 |

## 인코더와 표현 (8–13절)

| 절 | 요구 | 구현 | 확인 |
|---|---|---|---|
| 8 | atom → residue → atom 계층, residue 간 sparse atom 상호작용, 원자 상태 유지 | `BackboneEncoder`, `SidechainEncoder`, `AllAtomFusion` | `test_models.py` |
| 9 | BB 출력 보존, AA는 읽기만 | `AllAtomFusion` | `test_aa_fusion_does_not_mutate_bb` |
| 10 | 독립 입력 경로(BB Cartesian/internal, SC Cartesian, χ), internal encoder는 Cartesian graph 없음 | `InternalEncoder` | `residue_graph` 호출 금지 테스트 |
| 11 | sem, l=1/l=2, learned circle, 선택적 S²/SO(3) | `models/latents.py` | `test_latents.py` |
| 12 | SO(3) 등변, irrep별 channel mixing, invariant gate, 같은 irrep 공통 dropout | `fibers.py`, `effdock_blocks.py` | 회전·이동 테스트 |
| 13 | sequence CLS, BB/AA global irreps, l>0 초기값 0, learned readout vs mean | `GlobalReadout` (`global_readout: attention\|mean`) | `test_semantic_cosine_and_mean_readout_options` |

## 목적함수 (15, 17–21절)

| 절 | 요구 | 구현 | 확인 |
|---|---|---|---|
| 15 | task별 masking 정책(표현 변환, geometry infilling, SC 예측, partial → global) | `objectives/tasks.py` 9개 task + 선택형 `seq_infill` | `test_every_task_finite_backward` |
| 17 | invariant/equivariant predictor, query에 숨긴 좌표 없음, 서열만 있을 때 world-frame l>0 미사용 | `CrossViewPredictor`, `TaskSpec.equivariant` | 숨긴 좌표 변경 테스트 |
| 18.1–18.2 | EMA teacher (encoder + head), predictor는 EMA 아님 | `TargetStack` | `test_every_teacher_target_parameter_is_trained_online` |
| 18.3 | teacher-free baseline (target에 gradient, regularization으로 붕괴 제어) | `target_encoder: online` | `test_teacher_free_routes_gradient_through_targets` |
| 19.1 | sample → target type → valid node 정규화 | `typed_distance` (record별 segment 평균), task별 평균 | packed == per-sample 테스트 |
| 19.2 | scalar MSE 고정, cosine은 ablation | `semantic_distance: mse\|cosine` | 위와 같음 |
| 19.3–19.5 | `‖Δ‖²/(C(2l+1))`, circle `1−cos`, global loss | `typed_distance` | `test_latents.py` |
| 19.6 | raw 좌표·각도 재구성 기본 0 | `raw_angle_weight`, `raw_coordinate_weight` (질의 residue의 torsion과 visible Cα 중심 기준 Cα 변위; 원자 단위 raw 좌표는 없음), `node_weight` | `test_raw_reconstruction_baseline_is_equivariant_and_trains` |
| 20.1–20.2 | sem variance/covariance, circle floor | `regularize_latents`, `circular_floor` | `test_latents.py` |
| 20.3 | l>0 진단: degree별 channel norm, 0 channel 비율, invariant contraction 다양성 | `equivariant_diagnostics` (loss 없음) | `test_equivariant_collapse_diagnostics_are_logged` |
| 20.4 | heat-kernel MMD는 확장 regularizer | `torus_mmd`, `sphere_mmd` | `test_latents.py` |
| 21 | 모든 view encoder가 context로 등장 | task 구성 | `test_all_encoder_paths_receive_gradients_across_tasks` |

## 사용과 검증 (24–27절)

| 절 | 요구 | 구현 | 확인 |
|---|---|---|---|
| 24 | 기하 계산 FP32 | 기하는 float32, matmul 정밀도 고정(TF32는 회전 오차 1700배) | `reports/training_env` |
| 25 | `encode_sequence/backbone/all_atom/multimodal`, BB-only는 SC 경로 미실행, 긴 단백질 window 추론 | `ProteinJEPA.encode_*`, `encode_windows` | `test_encode_windows_covers_long_records_and_stays_equivariant` |
| 26 | 필수 테스트 12종 | rigid·translation·frame convention·missing atom·chain break·crop 경계·숨긴 좌표 변경·BB 격리·대칭 이름·빈 view·EMA/resume | `tests/` (atom 배열 순서는 slot 기반 dense 저장이라 해당 없음) |
| 27 | 비교 실험 9종 | `scripts/make_effdock_ablations.py`: `seq_only`/`seq_bb_only`, `bb_only`, `euclidean_latents`, `node_only`, `mean_readout`, `representation_tasks`/`infilling_tasks`, `crop_128`/`crop_256`, `teacher_free`, `raw_reconstruction` | `test_ablation_writer_accepts_every_shipped_gpu_preset` |

## 남은 것

- **14.3 crop → 더 큰 parent, 전체 단백질 global:** 보류. 명세 28절 5단계의 확장 실험이다.
- **27절 비교의 실제 결과:** 설정만 있고 학습·평가는 하지 않았다. 대규모 corpus 학습과 downstream probe가 필요하다.
- **NCCL 다중 GPU, 다중 node:** GPU가 한 장인 환경이라 실행하지 못했다([기록](../reports/training_env/README.md)).
