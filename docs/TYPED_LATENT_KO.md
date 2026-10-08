# Typed latent: 원래 설계 합의의 복원

v0.4는 원래 설계 대화에서 합의한 **typed latent**를 구현합니다. 각 기하적 성질을 그에 맞는 공간에 둡니다. v0.3에서는 학습되지 않던 projector를 고치는 대신 통째로 제거했고, 그 바람에 circular latent까지 사라졌습니다. 이것은 합의를 어긴 것이었고, v0.4가 이를 바로잡습니다.

## 1. 합의된 계약 (원래 설계 대화 기준)

| 항목 | 합의 내용 |
|---|---|
| Node latent | BB: semantic, l=1, l=2, learned circular. SC: semantic, l=1, l=2, χ-circular |
| 선택 head | direction(S², 단위 vector), frame(SO(3)). "다음 ablation"으로 둠 |
| Global | invariant semantic + 구조 irreps. torus나 SO(3)로 두지 않음 |
| Projection head | 학습됨. encoder, head, global readout을 함께 EMA로 갱신. predictor는 EMA 대상 아님 |
| Loss | semantic MSE, l>0 Frobenius/(C_l(2l+1)), circular 1 − ẑᵀz |
| Collapse 방지 | semantic var/cov, circular 약한 floor, l>0은 감시만. heat-kernel MMD는 확장 가능한 typed regularizer로 ablation |
| 서열 → 구조 | l=0 또는 invariant contraction만. world-frame l>0은 쓰지 않음 |
| 필수 비교 | all-Euclidean vs typed latent |

Learned circle K개는 물리 각도(φ, ψ, χ)와 일대일로 대응하지 않습니다. 실제 torsion 분포에 균일 torus prior를 강제하지도 않습니다.

## 2. 구현 (설계 B)

```text
online encoder H ─► online TypedLatentHead ─► context Z ─► predictor ─► predictor-owned typed head ─► Ẑ
teacher  encoder H ─► EMA   TypedLatentHead ─► target Z (+ parameter-free 정규화) ─────────────────► Z*
```

| Type | 구성 | Loss | 기본값 |
|---|---|---|---|
| semantic | invariant MLP. 입력은 s, ‖v‖, ‖T‖, l=1 Gram 상삼각 | MSE. target은 crop instance norm(soft floor) + log 크기 | 모든 view |
| l1 / l2 | channel mixing(등변) | Frobenius/(3C₁), /(5C₂). target은 soft RMS | bb, sc, aa |
| circular | Linear(invariants) → 단위 원 | 1 − cos | bb, sc, aa, bb_internal, chi (K=8) |
| direction | l=1 mixing → 단위 vector | 1 − cos. 크기가 약한 raw vector는 mask | 0 채널 (opt-in) |
| frame | 두 vector의 Gram–Schmidt(6D 표현) | (3 − tr(R̂ᵀR))/4. 퇴화 시 mask | 0 채널 (opt-in) |

설계 B를 고른 근거는 3개 모델 council의 반박 라운드입니다(thread `jepa-typed-latent`). 대안 A는 predictor 뒤에 공유 head를 두는 방식이었는데, 두 가지 문제가 지적됐습니다.
- online head가 학습 중에 encoder 출력을 한 번도 보지 못하는데, teacher는 그 EMA를 encoder 출력에 적용합니다.
- head 혼자 붕괴하는 경로가 생깁니다.

B에서는 predictor가 Z만 읽습니다. 따라서 teacher target을 만드는 head 가중치가 모두 prediction loss로 학습됩니다. regularizer를 꺼도 마찬가지입니다(`test_every_teacher_target_parameter_is_trained_online`). 일부러 adapter가 circular 입력을 무시하게 만들면, 이 테스트가 학습되지 않는 circular head 가중치를 정확히 찾아냅니다.

Atom target에는 head를 쓰지 않습니다. 정규화한 hidden 상태를 그대로 씁니다. node용으로 학습한 head를 atom 상태에 적용하지 않기 위해서입니다. Global target에는 semantic과 irreps만 씁니다.

`latent_typing: euclidean`은 all-Euclidean baseline입니다. head와 용량은 같고, manifold 사영 없이 MSE를 씁니다. 원에는 floor 대신 variance floor를 겁니다. frame은 이 모드에서 지원하지 않습니다.

## 3. 회전 augmentation 기본값 해제

구조 경로, typed head, 모든 loss와 regularizer는 정확히 SO(3) 등변이거나 불변입니다. 또 상대 기하만 씁니다. 1UBQ로 9개 task를 측정했습니다.

| backend | loss 상대 차이 | gradient 상대 차이 |
|---|---|---|
| reference (CPU) | 1.8e-7 | 1.6e-5 |
| CuEq CUDA | 1.5e-7 | 3.5e-5 |

두 값 모두 float32 반올림 수준입니다. 공통 강체 변환은 학습 신호를 바꾸지 않습니다. 그래서 `rigid_augmentation`의 기본값을 false로 바꿨습니다. 등변이 아닌 구성 요소를 실험할 때를 위해 옵션은 남겨 둡니다.

## 4. 단백질 22개 확률적 overfit

`protein-jepa overfit --stochastic`은 `train`과 같은 방식으로 학습합니다.
- 128/256 random crop
- 매 step 새로 뽑는 multi-block mask
- sample별 task 혼합
- EMA 0.99 → 0.999

평가는 같은 단백질에서 crop과 mask를 고정한 198쌍(22개 × 9 task)으로 합니다. 모델은 기본 크기(약 300만 파라미터)이고, backend는 CuEq CUDA, GPU는 RTX PRO 6000 Blackwell입니다.

데이터: 1A6M 1AKE:A 1CTF 1EMA 1HRC 1LYZ 1MBN 1PGA 1RIS 1RX2 1SHG 1STN 1TIM:A 1UBQ 2ACY 2CI2 2LZM 2RN2 2TRX:A 3CLN 4AKE:A 5PTI (56–247 residue). Canonical sequence map이 없어서 관측 순서를 position으로 썼습니다. overfit 진단 용도로는 문제가 없습니다.

### 결과 (3000 step, batch 4)

같은 seed로 두 번 돌렸습니다. 두 실행은 데이터, crop, mask가 모두 같고 covariance 항만 다릅니다. 증거는 [reports/typed_latent](../reports/typed_latent/)에 있습니다.

| step | loss (cov 끔 / 켬) | retrieval (끔 / 켬), 우연 수준 0.016 | effective rank (끔 / 켬) |
|---|---|---|---|
| 0 | 3.19 / 3.19 | 0.017 / 0.017 | 31.1 / 31.1 |
| 300 | 1.70 / 1.75 | 0.023 / 0.022 | 7.2 / 14.0 |
| 900 | 1.35 / 1.59 | 0.032 / 0.030 | 5.2 / 8.4 |
| 1800 | 1.22 / 1.35 | 0.046 / 0.049 | 7.3 / 10.3 |
| 2400 | 1.20 / 1.27 | 0.065 / 0.076 | 3.0 / 22.8 |
| 3000 | **1.14 / 1.18** | **0.076 / 0.111** | **3.1 / 44.0** |

마지막 시점 task별 retrieval(우연 수준 대비 배수, cov 끔 / 켬)입니다.

| task | cov 끔 | cov 켬 |
|---|---|---|
| `cart_to_internal` | 9.8 | 17.1 |
| `internal_to_bb` | 8.4 | 12.8 |
| `chi_to_sc` | 5.3 | 10.3 |
| `sc_to_chi` | 4.5 | 8.5 |
| `bb_to_seq` | 1.4 | 7.8 |
| `sc_infill` | 10.0 | 10.4 |
| `bb_infill` | 1.6 | 1.6 |
| `aa_infill` | 1.5 | 2.1 |
| `seq_to_bb` | 1.0 | 1.6 |

해석은 세 가지입니다.

1. 전체 확률적 파이프라인(random crop, mask, task 혼합, EMA)에서도 loss는 약 37%로 떨어집니다. retrieval은 우연 수준의 6.8배까지 오르고, 3000 step 시점에도 계속 가파르게 오르는 중입니다.
2. **Loss만으로는 잘못 판단합니다.** covariance를 끄면 effective rank가 3까지 무너집니다. 이때 일부 task는 loss가 오히려 더 낮습니다(`cart_to_internal` 0.08 vs 0.44). 그런데 retrieval은 더 나쁩니다. 표현이 몇 차원으로 몰리면 target도 단순해져서 맞히기 쉬워지기 때문입니다. 그래서 `covariance_weight`를 기본 0.04로 켰습니다. 입력이 정규화되지 않은 head 출력이므로, v0.3에서 이 항을 껐던 근거(layer norm과 충돌)는 여기에 해당하지 않습니다.
3. 숨긴 기하를 채우는 task(`bb_infill`, `aa_infill`)와 `seq_to_bb`는 3000 step으로는 아직 우연 수준에 가깝습니다. 더 긴 학습이 필요한 어려운 task로 보입니다. 성능이 부족하다는 증거로 해석하지는 않습니다.


## 5. 리뷰와 테스트

Codex review gate의 첫 판정은 **fail**이었습니다. 실제로 재현된 결함은 네 가지였습니다.
1. 크기가 1e-8인 raw vector가 유효한 direction·frame target으로 통과했습니다.
2. `sphere_mmd`가 0으로 붕괴한 상태에 음의 penalty를 줬습니다.
3. `sphere_mmd`가 D ≤ 2에서 오류를 내고, D = 512에서 0/0이 됐습니다.
4. 결측 residue 행이 관측된 token의 direction·frame 유효성을 바꿨습니다.

고친 내용은 다음과 같습니다.
- direction·frame 유효성은 절대 floor(1e-4)를 넘고, 동시에 valid token 평균 강도의 10% 이상일 때만 인정합니다.
- 0 vector는 pole 하나로 보내므로, 0으로 붕괴한 상태는 최대 penalty를 받습니다.
- D < 3이면 오류로 거부합니다. kernel 가중치는 log 공간에서 정규화해, 큰 차원에서 생기던 0/0을 없앴습니다.

각 결함에는 회귀 테스트를 붙였고, 재검토 gate는 **pass**(0.88)였습니다. Codex가 수정을 되돌린 코드로 테스트를 돌려 실패하는 것도 확인했습니다. 이 수정은 기본값에서 꺼져 있는 경로(direction·frame 채널 0개, `sphere_mmd`)만 바꿨습니다. 따라서 §4의 실험 결과는 그대로 유효합니다.

테스트 결과는 CPU 189 passed / 2 CUDA skips이고, GPU(CuEq CUDA 포함)에서는 `test_cueq` + `test_latents` 24개가 모두 통과했습니다.

## 6. 검증하지 않은 것

- 실제 corpus pretraining과 downstream 성능
- typed vs Euclidean의 품질 차이(대규모 학습 필요)
- direction·frame head와 heat-kernel MMD 옵션의 효과
- RoPE(미구현)
