# EFF-Dock-inspired CuEq JEPA v0.2: 구현·비교·강화 분석

## 1. 무엇을 완성했는가

`model.interaction: effdock`를 선택하면 BB atom stem, SC atom stem, BB residue trunk, AA atom fusion, AA residue fusion의 **다섯 단계가 모두 새 interaction block을 사용**한다. Sequence Transformer와 독립적인 backbone internal-coordinate/χ encoder, node/atom/global JEPA predictor, EMA teacher는 유지한다. 단순히 사용되지 않는 클래스 하나를 추가한 변경이 아니다. 연결 위치는 `models/encoders.py::interaction_block`, 구현은 `models/effdock_blocks.py`, graph schema는 `data/graphs.py`다.

`backend: cueq-naive`와 `backend: cueq-cuda`는 각각 실제 NVIDIA FCTP/SH의 PyTorch naive 방식과 CUDA 방식을 선택한다. `reference`는 별도의 analytic SO(3) operator이며 실제 CuEq 실행으로 취급하지 않는다. 정확한 실행 결과는 [v0.2 검증](V020_VALIDATION.md), 실행법은 [업그레이드 실행](UPGRADE_RUNBOOK.md)을 기준으로 한다. CUDA 실측과 downstream 품질 향상은 이 릴리스의 완료 주장에 포함하지 않는다.

원본 비교 대상은 [EFF-Dock@52d413d](https://github.com/eightmm/EFF-Dock/tree/52d413dca5f5d8bcc97185a6c5f3aefb1803d5d7)다. EFF-Dock 원본 저장소나 가중치를 수정하지 않았고, 기존 JEPA v0.1 checkpoint의 기본 architecture는 `baseline`으로 유지한다.

## 2. EFF-Dock에서 옮긴 것과 옮기지 않은 것

| 원본 EFF-Dock | JEPA v0.2 적용 | 차이 / 이유 |
|---|---|---|
| 하나의 shared-weight tensor product로 edge message 구성 | 각 interaction block 안에서 edge들이 하나의 FCTP를 공유 | 서로 다른 stage/layer의 파라미터까지 공유하는 것은 아님 |
| Input/output dual radial scaling | 둘 다 사용, 1+0.5*tanh 형태로 bounded | 초기 scale=1, 큰 radial 출력에 대한 증폭 제한 |
| Edge-type embedding, distance decay | Directed BB/SC/residue role + bond 여부, type별 decay | Ligand pharmacophore나 residue label은 가져오지 않음 |
| Gate-normalized aggregation | 비교용 `gate` 옵션으로 유지 | 새 기본값은 attention mass를 보존하는 `soft` |
| Mixed-irrep RMSNorm, norm-gated activation | Degree별 RMSNorm과 invariant norm gate | Public Fiber는 SO(3) scalar/vector/STF 계약 유지 |
| Time-conditioned AdaLN | Visible invariant context로 조건화 (v0.2의 static stage ID는 블록별 상수 bias에 불과해 v0.3에서 제거) | JEPA에 없는 flow time을 만들어 넣지 않음 |
| Equivariant post block | Equivariant post map + 별도 residual FFN | Residual scale은 0.1로 시작, 학습 가능 |
| 원자 force → Newton–Euler fragment motion | 옮기지 않음 | 이번 모델은 coordinate refinement가 아닌 latent pretraining |
| Ligand fragment / protein을 하나의 heterogeneous graph로 처리 | BB/SC/AA hierarchy 유지 | Pure BB output의 SC 격리와 단독 inference를 보존 |
| 0e/1o/1e/2e 등의 O(3) 표현 | 하나의 SO(3) l=0/1/2 family | Reflection parity나 EFF-Dock checkpoint 호환성을 주장하지 않음 |

이것은 EFF-Dock 복제 또는 pretrained weight 이식이 아니라 **interaction 원리를 protein representation 학습에 맞게 재구성한 것**이다. 전체 node layout·output head·학습 목표가 달라 원본 docking checkpoint를 그대로 읽을 수 없다.

## 3. 블록의 수학적 계약

입력 node는 h_i=(s_i,v_i,T_i), 여기서 T는 symmetric-traceless 3×3 tensor다. T의 저장 성분은 9개지만 수학적 자유도는 5개다. CuEq로 전달할 때는 설치된 SH basis에서 calibration한 invertible bridge를 통해 C0+3C1+5C2 성분으로 변환한다. XYZ 축 성분을 scalar처럼 취급하거나 l=2를 단순 reshape하지 않는다.

전처리된 edge에는 distance, unit direction, same-residue, signed sequence separation, explicit bond flag 및 directed edge type이 있다. 각 block은 다음 순서로 계산한다.

1. Degree별 RMSNorm으로 h를 정규화한다. Vector norm과 tensor Frobenius norm만 사용하므로 회전에 따라 scale이 달라지지 않는다.
2. RBF(distance), relation, edge type을 radial trunk에 넣는다.
3. Source feature를 per-channel input radial scale로 곱하고 TP(h_src, Y_0:2(direction))를 수행한다.
4. Output radial scale과 norm-gated activation을 적용한다.
5. Source/destination의 invariant summary와 radial context로 channel별 attention gate를 계산한다.
6. Type별 exp(-distance/sigma)와 nonbond spatial cutoff envelope를 곱한다.
7. 목적 node로 aggregate하고 bounded norm rescale을 적용한다.
8. Bias-free equivariant post map, channel-wise dropout, LayerScale residual을 적용한다.
9. Visible invariant state와 incoming attention mass로 조건화한 RMSNorm 뒤에 expanded equivariant FFN을 적용한다.

h_out = h + gamma_message * Message(h, graph) + gamma_ffn * FFN(h_after_message, context)

모든 gamma, radial coefficient, attention coefficient는 scalar다. 같은 irrep channel의 모든 magnetic component에 동일한 값을 곱한다. Non-scalar bias나 component별 dropout은 없다. Learned constant world vector도 도입하지 않는다.

## 4. 강화점 A: attention 정규화의 의미를 명시

원본과 비슷한 gate-normalized mean은 다음 형태다.

    A_gate = sum_j(w_ji m_ji) / (sum_j w_ji + eps)

이웃이 하나이고 w가 eps보다 충분히 크면 A_gate는 거의 m이 된다. 따라서 distance decay가 작아져도 정규화에서 대부분 상쇄될 수 있다. 이것은 원본이 틀렸다는 뜻이 아니라, **gate를 상대적 이웃 선택으로 쓸지 절대적인 상호작용 세기로도 쓸지**의 선택이다.

새 기본값은 다음과 같다.

    A_soft = sum_j(w_ji m_ji) / (1 + sum_j w_ji)

분모의 1은 zero-message를 보내는 null neighbor 하나로 해석할 수 있다. 약한 edge의 총 질량이 작으면 출력도 작아지고, 많은 강한 이웃이 있으면 normalized mean에 가까워진다. 비교를 위해 `gate`, `degree`도 실제 구현했다. `soft`가 항상 더 좋은 것은 아니다. 희박한 crop이나 작은 sidechain에서는 전달 신호를 과도하게 약화시킬 수 있으므로 crop 크기·degree별 downstream을 함께 비교해야 한다.

`test_soft_aggregation_retains_weak_edge_attenuation`은 단일 edge의 w=0.001에서 `soft`와 `gate`가 의도대로 다른 출력을 내는지 검사한다. 이 검사는 학습 성능 비교가 아니라 operator 계약 검사다.

## 5. 강화점 B: 0 근처 vector와 cutoff 경계

Direction/magnitude split에서 매우 작은 vector를 그 norm으로 나누고 새 magnitude를 예측하면 작은 수치 잡음에 큰 배율이 곱해질 수 있다. 새 rescale은 다음처럼 bounded multiplicative form을 사용한다.

    v_new = v * (1 + 0.1*tanh(MLP(sqrt(||v||²+eps))))

Tensor에도 동일한 원칙을 적용한다. 0은 정확히 0으로 남고 derivative가 유한하며, 실제 대칭 환경에서 일어난 vector cancellation을 억지로 복원하지 않는다. 그 대신 자유로운 magnitude 재구성 능력이 줄 수 있으므로 `effdock_norm_rescale: false` 대조군을 제공한다.

Spatial envelope는 cutoff에서 0으로 간다. 하지만 envelope만 곱해도 전체 block이 연속적이라고 할 수 없다. Post map의 scalar bias, discrete degree로 만든 conditioning, gate denominator가 다시 discontinuity를 만들 수 있다. 이번 구현은 **message post map의 scalar bias를 제거하고 conditional context에 discrete degree 대신 weighted attention mass를 사용**한다. `soft` 설정에서 cutoff에 있는 nonbond edge와 edge 자체가 없는 경우가 일치하는 regression test를 추가했다.

여전히 radius/top-k neighbor membership, covalent topology와 presence mask는 이산적이다. 따라서 전체 GNN을 전역적으로 매끄러운 potential이나 MD force field라고 주장하지 않는다. Bond edge는 spatial cutoff와 별개로 유지한다.

## 6. 강화점 C: edge type과 정보 경계

Node kind는 `BB_ATOM=0`, `SC_ATOM=1`, `RESIDUE=2`다. Edge type은

    type = 1 + 2*(src_kind*3 + dst_kind) + is_bond

이며 0은 legacy/unspecified다. 예약된 19개 ID 중 현재 계층 graph에서 실제 사용되는 조합만 학습된다. BB→SC와 SC→BB를 구분한다. 이는 residue amino-acid identity나 hidden-target 종류가 아니라 **visible node의 구조적 역할**이다.

BB graph에는 BB atom role과 polymer connectivity만 들어간다. SC atom 수·χ 개수·residue name을 graph type이나 conditional stage code에 넣지 않는다. SC/AA는 의도적으로 chemical identity 정보를 가질 수 있다. 그러므로 `Seq↔BB`와 `(Seq,BB)→SC`의 접근 정책은 이전 버전 그대로 유지한다.

Bond membership 계산은 per-edge Python set/CPU round trip에서 device-resident sorted-key lookup으로 바꿨다. 하지만 atom bond template 구성, record별 Python loop, geometry preprocessing의 CPU 동기화는 남아 있다. 이것만으로 GPU pipeline 전체가 최적화되었다고 볼 수 없다.

## 7. 모델 선택과 checkpoint

`interaction`과 `backend`는 서로 독립이다.

| interaction | backend | 의미 |
|---|---|---|
| baseline | reference | 기존 v0.1 analytic architecture |
| effdock | reference | 새 블록의 analytic 연산 검증 |
| baseline | cueq-naive/cueq-cuda | 기존 블록 + 실제 CuEq |
| effdock | cueq-naive/cueq-cuda | 새 블록 + 실제 CuEq |

기존 configuration에 interaction이 없으면 baseline이다. 과거 v0.1 configuration을 현재 dataclass의 default로 정규화한 뒤 비교하므로 같은 baseline checkpoint의 resume를 유지한다. 새 architecture로 바꾸고 old optimizer/state를 몰래 이어 읽지는 않는다. Backend, width, aggregation, ablation flag가 달라지면 resume가 거부된다. EFF-Dock weights와의 호환 경로도 없다.

## 8. 측정한 것과 측정하지 않은 것

[검증 보고서](V020_VALIDATION.md)는 현재 릴리스 결과를 기록한다. `reports/v020/`는 이번 실행 증거이고, 기존 `reports/`의 v0.1 결과와 혼동하지 않는다.

작은 reference 모델에서 baseline의 trainable parameter는 75,255개, effdock variant는 117,890개였다. 이는 teacher를 제외한 수이며, production preset의 크기나 CuEq parameter count가 아니다. 동일한 node width라고 해서 동일 parameter budget은 아니다.

128/256 residue 각각 9개 task, 두 architecture에서 총 36번 forward/backward를 수행했다. 각 task를 한 번씩 잰 시간은 warmup·cache·task 차이가 섞인 diagnostic이고, latency leaderboard가 아니다. 예를 들어 이 환경의 256-residue AA infill은 baseline 약 0.359초, 새 블록 약 0.511초였다. **새 구조가 공짜로 더 강한 것이 아니라 추가 계산을 사용한다**는 사실을 보여주는 참고치다. GPU speedup이나 protein accuracy 향상을 뜻하지 않는다.

Loss가 더 작다는 사실만으로 representation이 좋아졌다고 판단하지 않는다. Teacher latent 자체가 architecture에 따라 달라지고 collapse/scale 변화도 loss를 줄일 수 있다. Frozen node/protein probe, sequence-only transfer, geometry perturbation sensitivity, cluster-disjoint split이 필요하다.

## 9. 바로 실행 가능한 ablation

`scripts/make_effdock_ablations.py`는 현재 loader가 직접 읽을 수 있는 완전한 YAML 파일을 생성한다. Training을 자동 시작하거나 cluster job을 제출하지 않는다.

    python scripts/make_effdock_ablations.py \
      --base configs/effdock_cueq_gpu.yaml \
      --output runs/ablation-configs --seeds 17 29 43

생성되는 대조군은 baseline, effdock soft/gate/degree, no-dual-radial, no-conditioning, no-norm-rescale, no-distance-decay, no-smooth-cutoff, backbone-depth6이다. 원본 EFF-Dock 완전 재현 실험이 아니라 **이 JEPA에서 새 구성의 기여를 분리하는 실험**이다.

Data, canonical mapping, sequence-cluster split, crop/mask, 총 관측 residue 수와 optimizer schedule을 고정한다. Equal-steps와 equal-wall-time 결과를 둘 다 보고한다. Parameter-matched baseline도 별도로 width/depth를 조정해 비교한다. Validation으로 checkpoint를 고르고 test는 마지막에 한 번 평가한다.

핵심 평가는 (a) sequence-only representation에 구조 학습의 이득이 남는가, (b) BB-only에서도 SC/AA 공동학습 이득이 남는가, (c) pocket/interface residue 및 atom-level task에서 추가 비용 대비 이득이 있는가다. Useful representation을 확인하기 전에는 l_max나 crop/global objective를 더 늘리는 것을 우선하지 않는다.

## 10. 다음 강화 우선순위

> v0.3 상태: P2의 directional invariant(`effdock_directional`), FFN의 degree 간 결합(`effdock_ffn: bilinear`), top-k 경계의 연속성(`effdock_adaptive_cutoff`), SC inter-residue context(`sc_context: spatial`)는 opt-in으로 구현되었고 효과는 미검증이다. Predictor/target 결함 수정은 [JEPA_V030_KO.md](JEPA_V030_KO.md)를 본다.

**P0 — 실제 CuEq GPU 실행 gate.** FCTP와 basis bridge의 forward/backward, 깊은 stack의 equivariance, hidden-target mutation, optimizer/EMA/resume, CUDA/NCCL 2-rank를 검사한다. FP32부터 시작한다. CPU naive 성공만으로 fused kernel이나 NCCL을 통과했다고 간주하지 않는다.

**P1 — 같은 연산 예산에서 backbone 품질 확인.** Task별 frozen probe와 linear separability, per-l norms, representation effective rank, circular diversity를 기록한다. 학습 loss와 embedding norm만 비교하지 않는다. 특히 norm-only invariant summary가 기하의 미세한 차이를 얼마나 잃는지 점검한다.

**P1 — GPU 입력 pipeline.** Visible-only graph를 유지하면서 bond template를 사전 인덱싱하고, packed/disjoint batching, bucketed residue/atom budget, radius-search backend를 추가한다. 현재 chunked pairwise distance는 memory peak를 줄일 뿐 계산량은 quadratic이다. Crop/mask 이전 full kNN을 재사용하는 최적화는 hidden-target leakage를 만들 수 있으므로 금지한다. Profiler로 병목을 확인한 뒤 SH/RBF reuse와 packed irrep 유지 범위를 확대한다.

**P2 — 더 풍부한 invariant edge conditioning.** Source/destination vector의 dot product, v·r_hat, r_hat^T T r_hat, 유효한 local frame의 상대 회전 등을 gate에 추가하는 실험이 가능하다. 이들은 global rotation에 불변이지만, 반드시 visible atom에서만 계산해야 한다. 단순 norm만 쓰는 gate보다 angular discrimination이 좋아질 가설이며 아직 구현/평가된 개선으로 세지 않는다.

**P2 — AA 안의 atom/residue joint heterogeneous graph.** EFF-Dock에 더 가까운 single-graph update를 AA 내부에서 실험할 수 있다. Atom→residue와 residue→atom type을 추가하고 shared TP를 사용한다. 다만 clean BB output은 AA로 들어오기 전에 보존하고, AA hidden state를 BB-only output으로 되돌려 쓰면 안 된다. Coordinate-free global token에 arbitrary xyz를 부여하는 방식은 피한다.

**P3 — 깊이, 반복, global readout 확장.** Recycled block, multi-slot global bundle, invariant attention 개선을 비교한다. Slot 간 비중복성과 l>0의 입력 유래 조건을 유지한다. Global slot 수를 늘렸다고 structure 전체를 완전하게 표현한다고 해석하지 않는다.

**P3 — 학습 목표의 강화.** Typed latent의 anti-collapse와 teacher-free 설정, geometry-compatible distribution matching은 별도 축이다. 구조 encoder를 바꾼 실험과 regularizer를 동시에 바꾸지 않는다. 실제 torsion의 주기성만을 근거로 physical angle 분포를 uniform torus에 강제하지 않는다.

## 11. 여전히 남는 범위 제한

이 릴리스에는 pretrained weight, 실제 corpus 학습 결과, ligand/NA 모델, force/docking head, GPU benchmark가 없다. Canonical sequence alignment와 homology clustering을 자동 해결하지 않는다. DDP의 representation regularizer 통계는 rank-local이며 global covariance와 다르다. Global target은 기본적으로 같은 parent crop 전체이지 원본 full protein 전체가 아니다. 이 한계들은 이번 interaction upgrade로 사라지지 않는다.

## 12. 1차 출처와 확인 범위

- [EFF-Dock equivariant operators](https://github.com/eightmm/EFF-Dock/blob/52d413dca5f5d8bcc97185a6c5f3aefb1803d5d7/src/effdock/models/equivariant.py): GatedEquivariantConv, RMSNorm, AdaLN, activation/dropout의 코드 비교.
- [EFF-Dock interaction layer](https://github.com/eightmm/EFF-Dock/blob/52d413dca5f5d8bcc97185a6c5f3aefb1803d5d7/src/effdock/models/effdock.py): node/edge schema, pre-norm, time conditioning, orientation convention.
- [NVIDIA FCTP](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.FullyConnectedTensorProduct.html): shared/internal weights, layout 및 method 계약.
- [NVIDIA SphericalHarmonics](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SphericalHarmonics.html): naive/uniform_1d 및 normalization 계약.

원본 source pin은 재현을 위한 기준이다. 코드 유사성이나 짧은 smoke test를 근거로 EFF-Dock의 docking 성능이 이 JEPA로 이전되었다고 주장하지 않는다.
