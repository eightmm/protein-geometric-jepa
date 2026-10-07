# 구현 및 검증 상태 — v0.1.0

이 문서는 구현된 기능, 이 환경에서 실행한 검증, 아직 검증하지 않은 사항을 구분합니다. 아래의 통과 수는 [실행 로그](../reports/pytest.txt)와 [JUnit 결과](../reports/pytest.xml)에 근거합니다.

## 구현되어 실제 CPU 테스트한 기능

| 영역 | 구현 / 검증 |
|---|---|
| Sequence | Transformer + CLS, sequence-only inference |
| Backbone | 원자 stem, residue graph, 방향/frame/torsion, l=0/1/2 |
| SC/AA | Sidechain atom stem, χ view, BB/SC fusion, atom↔residue context |
| Internal-coordinate views | 별도의 BB internal/χ Transformer, Cartesian graph 비사용 |
| Global | Invariant attention readout + input-derived equivariant global bundle |
| JEPA | 9개 task, node/global/AA atom loss, EMA teacher |
| Typed latent | Semantic, Cartesian equivariant, learned circular channels |
| Data | Numeric NPZ, explicit presence, canonical sequence map, PDB/mmCIF |
| 안전한 masking | Dependency closure, graph 재생성, BB/SC 정보 경계 |
| 실행 | CLI prepare/demo/train/encode/evaluate, config 검증 |
| Checkpoint | Online/teacher/predictor/optimizer/scheduler/RNG, 원자적 저장 |
| Distributed | 2-rank CPU/Gloo에서 9개 task 순환 학습 |
| Downstream | Frozen feature export, invariant atom/residue/protein task-head 클래스 |

## 실제 실행 결과

- **72 passed, 3 skipped**: 로컬 pytest. Skipped 3개는 실제 CuEq package/GPU 필요 테스트입니다.
- **9개 task 모두 finite forward/backward**, 모든 online encoder 경로의 gradient 도달을 검사했습니다.
- **128/256 residue 길이 × 9 task = 18개 forward/backward 조합**을 작은 reference 모델에서 실행했습니다. [crop_sizes.json](../reports/crop_sizes.json)
- Synthetic demo **18 optimizer steps**를 실행했습니다. [reference_smoke.log](../reports/reference_smoke.log)
- 2-rank CPU/Gloo **9 steps / 전체 task 1회 순환**을 실행했습니다. [ddp_cpu_all_tasks.log](../reports/ddp_cpu_all_tasks.log)
- Dropout이 켜진 동일 설정에서 중단/재시작 후 uninterrupted run과 **모든 model tensor의 bitwise equality**를 단일 CPU 테스트로 확인했습니다.
- Synthetic fixture를 PDB/mmCIF로 쓰고 읽는 integration test를 실행했습니다. 실제 PDB corpus 품질 평가와 다릅니다.
- Manifest 기반 train, held-out pretext evaluation, sequence-only 및 AA feature export를 CLI integration test로 실행했습니다.
- `pip install --no-deps --no-build-isolation -e .`와 console entry point를 확인했습니다. 새 인터넷 환경의 dependency resolver 테스트는 아닙니다.

실행 시간은 작은 합성 데이터/작은 모델/현재 CPU 환경에 대한 참고 로그입니다. GPU throughput, 과학적 수렴 또는 downstream 성능으로 해석하면 안 됩니다.

## 코드가 있지만 이 환경에서 실행하지 못한 것

| 항목 | 상태 |
|---|---|
| CuEq naive backend | 실제 NVIDIA FCTP/SH API 호출 구현, package 부재로 미실행 |
| CuEq CUDA backend | 실제 fused_tp/uniform_1d 선택 구현, GPU 및 ops package 부재로 미실행 |
| GPU/NCCL DDP | CLI/torchrun/Slurm 경로 준비, 미실행 |
| Slurm script | 실행 예제 작성, 실제 cluster에 제출하지 않음 |
| GitHub Actions | 게시 후 CPU workflow 실행 시작; 커밋별 결과는 [Actions](https://github.com/eightmm/protein-geometric-jepa/actions)에서 확인 |
| PLMol 실물 package adapter | ParsedAtom contract와 fake parser 테스트, 설치된 PLMol 통합은 미실행 |
| GitHub publication script | Dry run/입력 검증만 실행, 실제 생성·push는 하지 않음 |
| Python 3.11/3.12 | 지원 대상으로 CI matrix 작성; 현재 로컬 실행은 Python 3.13 |

CuEq를 선택했는데 package/GPU가 없으면 예외가 발생합니다. 검증되지 않은 경로를 reference로 바꿔 성공한 것처럼 보고하지 않습니다.

## 이번 릴리스에 포함하지 않은 것

**사전학습된 weights, 실제 protein corpus, downstream 성능 수치, ligand/NA 모델**은 없습니다. 사용자가 논의한 protein 부분을 우선 구현한 연구용 모델입니다.

자동 RCSB/UniProt 다운로드, canonical sequence alignment 자동 복구, homology clustering 실행, all-Euclidean baseline runner, teacher-free training, heat-kernel MMD, crop→원래 단백질 전체 global target, spatial crop, explicit manifold-valued frame predictor, O(3) reflection training, GPU AMP/torch.compile/커널 profiling은 후속 확장입니다.

현재 manifest는 선언된 cluster ID의 split 일관성과 경로 중복을 검사합니다. 그것 자체가 실제 sequence homology 검증은 아닙니다. 기본 fingerprint는 manifest 내용이며, 동일 경로의 NPZ 파일을 사후 수정하는 것을 탐지하지 않습니다. 재현 가능한 학습에서는 전처리 산출물을 immutable하게 저장해야 합니다.

DDP regularizer 통계는 rank-local입니다. 전역 batch covariance/분산과 동등하지 않습니다. 연산 graph는 sparse지만 edge discovery는 chunked pairwise distance이므로 전체 시간복잡도는 여전히 quadratic입니다. Transformer도 record별로 처리하므로 대규모 throughput용 packed batching 구현이 아닙니다.

## 과학적 검증 범위

EMA, variance/covariance penalty, circular variance floor는 이번 구현의 baseline 안정화 구성입니다. 모든 latent에서 non-collapse를 수학적으로 보장하지 않습니다. 짧은 실행에서 finite loss가 나왔다는 것은 useful representation이나 novel architecture의 우수성을 입증하지 않습니다.

Geometry-aware latent가 실제로 이득이 있는지, SC/AA 공동학습 이득이 sequence-only downstream에도 남는지는 별도 대조 실험이 필요합니다. [설계 명세 §25](SPEC_KO.md)를 연구 평가 계획으로 사용하세요.

## 원격 상태

[eightmm/protein-geometric-jepa](https://github.com/eightmm/protein-geometric-jepa)의 `main`에 게시했습니다. 원본 76개 파일의 source import commit은 `2b2ee75`이며 Git tree hash가 원본 bundle과 일치합니다. 게시 전 [CPU 재검증](../reports/publication_pytest.txt)도 **72 passed, 3 skipped**로 재현했습니다.

원격 Actions 실행과 로컬 검증 로그는 구분합니다. 이 문서에 기재한 CuEq/GPU 및 downstream 미검증 범위는 원격에 파일을 올렸다는 이유만으로 달라지지 않습니다. 실제 CI 결과는 [Actions](https://github.com/eightmm/protein-geometric-jepa/actions)에서 확인합니다.

사용자가 만든 public 설정을 유지했으며 원본 `eightmm/plmol`은 수정하지 않았습니다. 공개 라이선스는 소유자의 결정을 위해 설정하지 않았습니다. [게시 이력](PUBLICATION.md)에 import 검증과 사용 방법을 기록했습니다.
