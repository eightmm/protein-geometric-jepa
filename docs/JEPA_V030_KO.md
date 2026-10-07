# JEPA v0.3: target·predictor 결함 수정과 opt-in encoder 실험

v0.3은 v0.2를 실행해 확인한 개념 결함을 고치는 릴리스입니다. JEPA의 기본 계약도 다시 맞췄습니다. 그 계약은 다음 셋입니다.

- Teacher는 online encoder의 EMA입니다.
- Target은 teacher encoder 상태를 정규화한 값입니다.
- Predictor는 context와 mask token으로 target을 예측합니다.

**v0.2 checkpoint와는 호환되지 않습니다.** Checkpoint `format_version`은 2입니다. 공개된 가중치가 없으므로 버전을 명시하고 호환성을 끊었습니다.

## 1. 확인된 결함과 수정

아래 표의 "근거"는 v0.2 HEAD에서 코드를 직접 실행해 얻은 결과입니다. 회귀 테스트는 v0.3 suite에 들어 있습니다.

| ID | v0.2 결함 | 근거 | v0.3 수정 | 회귀 테스트 |
|---|---|---|---|---|
| F1 | Online projector의 `linear.s/v/t`에 gradient가 오지 않음. Teacher projector가 초기화 값에 고정되어, l>0 target이 무작위 선형 사영이었음 | 90 step 학습 후 teacher `linear.v.weight`가 초기값과 bit 단위로 같음 | Projector와 circular latent를 제거함. Target은 teacher encoder 상태를 parameter 없이 정규화한 값임(§3) | `test_every_teacher_target_parameter_is_trained_online` |
| F2 | `AllAtomFusion.activation`이 학습되지 않는데도 atom target 경로에 있었음 | Online AA atom 출력을 쓰는 loss가 없음 | 마지막 residue→atom feedback과 activation을 제거함. Atom 출력은 atom layer 직후 상태임 | 같은 테스트 |
| F3 | Predictor가 cross-attention 1층뿐임. Query끼리 보지 못함. l>0 출력이 context vector들의 선형 span 안에 갇힘 | 구조 확인. `average_attn_weights=True`로 l>0 경로가 head 평균만 씀 | 2단계 predictor로 바꿈(§4). Head별 equivariant value, relative position bias, bilinear FFN을 둠 | `test_bilinear_ffn_synthesizes_out_of_span_direction`, predictor 등변성 테스트 |
| F4 | Atom query를 teacher가 관측한 atom 목록에서 만듦. 숨겨진 residue의 결측 패턴이 새어 나갈 수 있었음 | `jepa.py`의 query 구성 경로. Council 세 seat가 모두 지적함 | Visible sequence의 canonical topology로 query를 만듦. 관측 여부는 loss mask에만 씀 | `test_hidden_atom_presence_never_shapes_queries_or_predictions` |
| F5 | Reference message에 CG 경로 5개가 빠져 있었음 | 코드 확인 | 다섯 경로를 추가함. 경로 집합이 CuEq FCTP와 같아짐 | 경로별 등변성과 nonzero 확인, `test_reference_message_equivariance` |
| F6 | Step마다 batch 전체가 한 task였음. Mask는 연속 구간 하나. EMA는 0.99 상수 | `train.py`, `tasks.py` | Task를 sample별 round-robin으로 섞음. Loss는 task 안에서 평균한 뒤 task 사이에서 평균함. 겹치지 않는 multi-block mask를 씀(합집합 정확히 k, 최소 span 8). EMA는 0.996→1.0 선형 schedule | Resume·DDP·mask 테스트 |
| F7 | `stage_embedding`이 블록마다 사실상 bias 하나였음(4행은 죽음) | 코드 확인 | 제거함 | `test_stage_wiring_and_all_ablation_switches` |
| F8 | `GlobalReadout` score의 마지막 bias가 softmax shift 불변성 때문에 학습될 수 없었음 | v0.3 불변식 테스트가 새로 발견함 | Bias를 제거함 | 불변식 테스트 |

## 2. Opt-in encoder 실험 (기본값 꺼짐)

아래 항목은 표현력을 넓히려는 가설입니다. 결함 수정과 효과를 섞지 않으려고 기본값은 꺼 두었습니다. [`configs/jepa_full_cueq_gpu.yaml`](../configs/jepa_full_cueq_gpu.yaml)은 네 항목을 모두 켭니다. `scripts/make_effdock_ablations.py`는 항목별 variant와 모두 켠 variant를 만듭니다.

| 옵션 | 내용 | 기본값을 끈 이유 |
|---|---|---|
| `sc_context: spatial` | SC view가 다른 residue의 SC atom과도 edge를 만듭니다(BB atom은 여전히 제외). v0.2에서는 한 residue의 sidechain을 움직여도 다른 residue의 SC 출력이 정확히 0만큼 변했습니다 | SPEC §11의 per-residue SC 계약을 바꾸는 표현 확장입니다 |
| `effdock_directional: true` | Attention gate에 채널별 v·r̂와 r̂ᵀTr̂를 넣습니다 | Gate가 norm만 보던 것은 결함이 아니라 표현력 한계입니다 |
| `effdock_ffn: bilinear` | Encoder FFN에 degree 사이의 곱을 넣습니다 | 깊은 encoder에서의 안정성은 미검증입니다 |
| `effdock_adaptive_cutoff: true` | 목적 node의 첫 번째 제외 후보 거리에서 envelope가 0이 됩니다. Top-k 교체가 연속적이 됩니다 | 밀도에 따라 receptive field가 줄어드는 의미 변화가 있습니다 |

`test_adaptive_cutoff_makes_topk_swap_continuous`는 두 이웃의 k번째 순위가 바뀔 때 출력 차이를 비교합니다. 옵션을 켜면 차이가 1e-3 미만이고, 끄면 1e-3보다 큽니다. 결합(bond)된 이웃이 경계를 넘는 경우도 함께 검사합니다. 연속성은 `soft`/`gate` aggregation에서만 성립합니다. `degree` aggregation은 분모가 이산적인 이웃 수라서, bond로 남는 이웃이 top-k에서 빠지면 출력이 끊깁니다(Codex 리뷰 R3, 0.024 크기의 도약이 δ→0에서도 사라지지 않음). 그래서 이 조합은 config에서 거부합니다.

## 3. Target 정의

```text
scalar: LayerNorm(s) ⊕ log((r1+τ1)/(mean r1+τ1)) ⊕ log((r2+τ2)/(mean r2+τ2))
vector: v / sqrt(r1² + τ1²)          r1² = mean_c |v_c|²/3
tensor: T / sqrt(r2² + τ2²)          r2² = mean_c |T_c|²_F/5
τℓ = target_floor(0.1) × sample 안 valid token의 평균 rℓ
```

Global target은 token이 하나라서 평균이 자기 자신과 같아집니다. 그대로 두면 floor와 크기 scalar가 무력화됩니다. 그래서 global은 같은 crop의 node 평균 RMS를 기준으로 정규화합니다(Antigravity 리뷰).

정규화 방식은 세 가지를 비교했습니다. Token별 hard RMS는 거의 0인 벡터를 단위 크기로 증폭합니다. Council 세 seat가 모두 반대했습니다. Crop 단위 RMS는 token 사이의 상대 크기를 보존합니다(Antigravity 제안). v0.3은 token별 soft floor를 쓰면서 crop 평균 대비 log 크기 scalar를 추가했습니다. 그래서 두 관점을 함께 만족합니다. 두 방식의 우열은 같은 예산의 held-out 비교가 있어야 판단할 수 있습니다. 아직 측정하지 않았습니다.

## 4. Predictor

1. **Joint stack** (`predictor_layers`, 기본 4): context token과 node/global mask token이 함께 갱신됩니다.
2. **Atom decoder** (`atom_decoder_layers`, 기본 2): atom mask token만 갱신됩니다. Stage 1 token은 atom token을 보지 않습니다. 따라서 atom query 집합이 node/global 예측을 바꾸지 못합니다.

Block 구조는 다음과 같습니다.
- Degree별 RMS pre-norm을 적용합니다.
- Invariant logit에 [-R, R]로 자른 relative sequence-offset bias를 더합니다. Global token은 별도 bucket을 씁니다.
- Head별 scalar/vector/tensor value를 씁니다.
- Bilinear FFN을 씁니다.
- 각 residual에는 0.1로 시작하는 channel scale을 둡니다.

Bilinear FFN의 곱은 0.1 weight로 시작합니다. Pre-norm 뒤에서만 계산합니다.

## 5. Regularizer와 진단

Variance floor는 정규화하기 전의 online context scalar에 적용합니다. Covariance penalty는 opt-in입니다(`covariance_weight` 기본 0). Layer-normed scalar는 channel 합이 0입니다. 그래서 정규화한 뒤에 covariance를 걸면 정규화가 만든 상관까지 벌점이 됩니다(Codex 지적).

`metrics.jsonl`에는 두 지표를 추가로 기록합니다.
- View별 `effective_rank`(RankMe)
- Target token 가운데 l=1 RMS가 floor 아래인 비율(`target_low_rms`)

## 6. 다중 모델 토론 과정

계획 단계에서 `oms peer-ask` council(Codex, Claude, Antigravity)에 설계 검토를 요청했습니다. 세 seat의 1라운드 답변은 모두 완료되었습니다. 반박 라운드는 진행되지 않았습니다. Council이 도는 동안 parent가 tracked 파일을 수정했고, OMS가 서로 다른 revision을 섞지 않도록 반박 라운드를 중단했기 때문입니다.

| 쟁점 | Seat 의견 | 결정 |
|---|---|---|
| Atom query 구성 | 세 seat 모두 topology 기반을 요구함 | 채택 |
| Predictor 구조 | 세 seat 모두 2단계 구조를 권함 | 채택. Atom decoder가 stage 1 전체를 읽음 |
| Target 정규화 | Hard RMS는 세 seat 모두 반대. Soft floor 2명, crop 단위 1명 | Soft floor와 log 크기 scalar를 함께 씀 |
| var/cov | 유지 1명, variance만 1명, 모두 끔 1명 | Variance 유지(정규화 전 scalar). Covariance는 opt-in |
| Encoder 확장 | 세 seat 모두 결함 수정과 분리를 요구함 | 구현하되 기본값은 끔 |
| RoPE | 세 seat 모두 제외를 요구함 | 미구현 |

계획 단계에서 정정된 사항도 있습니다. 분석 단계에서는 v0.2 predictor의 l>0 출력을 "convex combination"이라고 설명했습니다. Codex가 이 설명이 틀렸다고 반박했습니다. 출력 앞뒤의 channel map은 부호가 자유롭기 때문입니다. 정확한 한계는 "context vector들의 선형 span을 벗어나지 못한다"입니다. 이 정정은 결론을 바꾸지 않습니다. 새로운 방향을 만들 수 없다는 한계는 그대로입니다.

### 구현 리뷰

구현을 마친 diff를 `oms peer-review --ml --gate`로 리뷰받았습니다. 작성자 계열인 Claude는 리뷰어에서 자동으로 제외됩니다.

- Codex는 gate를 **fail**로 판정했습니다. 실행으로 재현한 결함은 세 가지였습니다.
  - R1: 대상 residue가 모두 glycine인 `sc_infill`에서 atom query가 0개가 되어 크래시가 났습니다.
  - R2: `configs/cueq_naive.yaml`에 제거된 옵션이 남아 있었습니다.
  - R3: `degree` aggregation과 적응 cutoff를 함께 쓰면 출력이 불연속이었습니다.
- Antigravity는 전체 diff가 provider 전송 한도를 넘어, 코드 diff만 따로 리뷰했습니다. 받아들인 지적은 다섯 가지입니다.
  - global target의 기준 scale
  - 이름과 내용이 맞지 않던 resume 테스트
  - 저장된 config를 검사하지 않고 읽던 경로
  - spatial mask 사용 시 실행 경고
  - CuEq 테스트에 빠진 atom 격리 단언문
- 받아들이지 않은 지적은 하나입니다. 관측 atom이 없을 때 atom decoder를 건너뛰자는 제안입니다. 그렇게 하면 query 실행 여부가 teacher의 관측 여부에 좌우되어 정보 경계가 약해집니다.

각 결함에는 회귀 테스트를 붙였습니다. 수정을 되돌리면 실패하는 것도 확인했습니다. 남은 쟁점은 thread `jepa-v030-plan`에 기록했습니다.

## 7. 검증 증거

증거는 [`reports/v030/`](../reports/v030/)에 있습니다. CPU에서 PyTorch 2.14.1을 썼고, CuEq 0.9.0은 CPU에서만 실행했습니다.

- 전체 pytest: **165 passed, 2 skipped**. Skip 2개는 CUDA GPU가 필요한 테스트입니다. 실제 CuEq CPU 테스트 8개는 실행되었습니다(`-m 'not cuda'`: 8 passed).
- Smoke demo 18 step 4종이 완료되었습니다. 대상은 baseline reference, effdock reference, baseline CuEq-naive, effdock CuEq-naive입니다.
- 2-process Gloo DDP 18 step이 완료되었습니다(batch 2, 한 step 안에서 task가 섞이는 설정). 리뷰 반영 전에는 batch 1 설정도 통과했습니다.
- 128/256 residue 검증: baseline, effdock, effdock-full × 9 task, 총 54회 forward/backward가 모두 finite였습니다.
- Full 크기 모델(파라미터 약 317만/334만 개)로 256 residue를 처리할 때 task당 CPU 시간은 2.9초 이하이고 최대 RSS는 약 2.5 GB였습니다.

## 8. 검증하지 않은 것

- CuEq CUDA fused kernel, GPU/NCCL 학습, GPU 메모리와 처리량
- 실제 corpus pretraining, downstream 성능, 그리고 v0.2 대비 표현 품질이 좋아졌는지 여부. 결함은 사라졌지만, 품질이 좋아졌는지는 같은 예산의 ablation과 frozen linear probe로 확인해야 합니다.
- Opt-in encoder 실험의 효과와 깊은 stack에서의 안정성
- Target 정규화 방식(token별 soft floor vs crop 단위) 비교
- RoPE(미구현). Sequence/internal Transformer는 여전히 절대 sinusoidal position을 씁니다. Predictor는 relative bias를 씁니다.
