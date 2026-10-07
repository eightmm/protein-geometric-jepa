# Protein Multi-View Geometric JEPA — 구현 명세 v0.1

## 0. 문서의 효력과 범위

이 문서는 논의한 연구 방향과 이번 저장소의 실제 구현을 연결한다. 핵심 아키텍처는 실행 가능한 코드로 구현되어 있다. 성능 향상·수학적 복원 보장·대규모 GPU 처리량은 검증 결과가 없는 가설로 남긴다. 구현되지 않은 선택지는 STATUS.md에서 구분한다.

모델의 목적은 **동일 단백질의 다른 관측 표현을 node 수준에서 서로 예측하는 것**이다. Protein-level retrieval embedding만 만드는 모델도, raw atom 좌표를 생성하는 folding decoder도 아니다.

기본 계층은 다음과 같다.

\[
\text{atom states}\to\text{residue states}\to\text{global states}
\]

단, pooling 이후 atom states를 버리지 않는다. AA residue 문맥을 atom states로 다시 전달한다.

## 1. 최상위 요구사항

| 요구사항 | 구현 계약 |
|---|---|
| 같은 entity의 여러 표현 | seq, bb, sc, aa, bb_internal, chi |
| 노드 수준 latent | residue 노드 주 objective, AA atom 보조 objective |
| Sequence 모델 | bidirectional Transformer, CLS 포함 |
| 구조 모델 | SO(3) scalar/vector/STF-tensor GNN |
| SC 및 χ | 반드시 보존; BB 경로에서만 차단 |
| 외부 annotation | SASA/DSSP/secondary-structure rule label을 계산하지 않음 |
| Crop | 128/256 sequence-contiguous parent crop |
| Global JEPA | parent crop의 부분 관측 → parent crop 전체 latent |
| 추론 | sequence-only, backbone-only, all-atom, multimodal |
| 정합성 | explicit masks, canonical residue IDs, atom→residue mapping |

## 2. 정보 분리: 통계적 독립성이 아니라 접근 권한

Backbone conformation과 amino-acid identity가 통계적으로 독립이라는 가정을 두지 않는다. Backbone으로 sequence preference를 추론하는 정보는 유용한 학습 신호다.

차단하려는 것은 BB encoder의 forward 입력에 residue token, sidechain topology, real sidechain coordinates, χ, ESM embedding 등이 직접 들어오는 경로다.

반대로 SC/AA는 chemistry를 포함한 관측이다. 원자 종류·sidechain topology가 residue identity를 알려주는 것은 이 경로의 허용된 정보다. SC/AA를 sequence-free라고 부르지 않는다.

정의:

\[
H^{BB}=E_{BB}(X^{BB},\mathcal E_{BB})
\]
\[
H^{AA}=F_{AA}(H^{BB},E_{SC}(X^{SC}),\mathcal E_{AA})
\]

AA loss가 BB encoder를 학습시키는 gradient는 허용한다. 하지만 AA activation을 BB output에 되돌려 쓰지 않는다. `test_backbone_isolation_from_sequence_and_sidechain` 및 `test_aa_fusion_does_not_mutate_bb`가 이 조건을 검증한다.

## 3. Canonical record

`ProteinRecord`는 하나의 chain이다. 여러 chain을 한 서열로 임의 연결하지 않는다.

```text
xyz                 float32 [L,37,3], Angstrom
present             bool [L,37]
seq                 int64 [L], standard 20 AA + UNK
seq_pos             int64 [L]
peptide             bool [L-1]
residue_ids         tuple[str], chain:author_number:insertion_code
record_id           str
position_source     canonical | observed_order_unverified
```

37개 slot은 atom-name의 공통 사전이다. 어떤 slot이 존재하는지는 `present`가 결정한다. 좌표 0은 결손 표현이 아니며, 원점의 원자는 정상 원자다.

Canonical record에서는 `seq_pos`가 1씩 증가해야 한다. 원래 서열의 unresolved position은 좌표가 없는 row로 명시한다. Crop은 이 row를 보존한다. Sequence map 없는 observed-only 구조를 canonical로 위장하지 않는다.

OXT는 저장 schema에 있지만 v0.1의 표준 BB+SC encoder는 네 backbone atom과 표준 sidechain atom을 모델링하고 OXT를 제외한다. Noncanonical chemistry, ligand, ion, modified residue의 완전한 지원은 범위 밖이다. MSE는 명시적으로 MET/SD로 정규화한다.

## 4. Mask의 의미

물리적 존재, 계산 가능성, student 관측은 서로 다르다.

| 구분 | 의미 |
|---|---|
| present | 원본에 원자 좌표가 관측되었는가 |
| defined | 해당 residue에서 물리적 torsion이 정의되는가 |
| valid | 필요한 관측이 있고 기하량이 비퇴화 상태인가 |
| visible | 이번 context가 접근할 수 있는가 |
| target | 이번 objective에서 예측할 위치인가 |

계산값이 0인 것과 invalid인 것을 같은 label로 처리하지 않는다. 기하 feature의 invalid value는 0으로 채우되 mask를 함께 전달한다.

## 5. Backbone geometric observation

기본 원자: N, Cα, C, O.

직접 사용하는 기하량:

- Backbone-relative distances/directions와 local atom geometry.
- φ, ψ, ω와 validity.
- Cα pseudo-angle α 및 pseudo-dihedral τ.
- Cα prev/next direction.
- N→Cα, Cα→C, C→O direction.
- Backbone에서 동일 규칙으로 만든 virtual-Cβ direction.
- N–Cα–C local frame의 세 basis vector.

\[
d_i=\|x_{i+1}^{CA}-x_i^{CA}\|,
\quad\alpha_i=\angle(x_{i-1}^{CA},x_i^{CA},x_{i+1}^{CA})
\]
\[
\tau_i=\operatorname{dihedral}(x_{i-1}^{CA},x_i^{CA},x_{i+1}^{CA},x_{i+2}^{CA})
\]
\[
u_i^-=\operatorname{normalize}(x_{i-1}^{CA}-x_i^{CA}),
\quad u_i^+=\operatorname{normalize}(x_{i+1}^{CA}-x_i^{CA})
\]

ω convention은 `omega_i = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i)`로 고정한다. 저장소 안에서 incoming/outgoing ω 인덱스를 혼용하지 않는다.

Pseudo-angle은 [0,π]의 비주기 값이다. cos/sin으로 수치 encoding할 수는 있지만 이를 물리적 S¹ target이라고 부르지 않는다.

## 6. Frame 및 회전 convention

Frame은 column-basis matrix다.

\[
R_i=[e_1,e_2,e_3],\quad
 e_1=\operatorname{normalize}(N_i-CA_i),\quad
 e_3=\operatorname{normalize}((N_i-CA_i)\times(C_i-CA_i))
\]
\[
e_2=e_3\times e_1
\]

Frame을 `[vector_channel,xyz]`로 넣을 때 `R_i.T`를 사용한다. `[3,3]`이라는 shape만 보고 row를 basis로 오해하지 않는다.

World vector와 local-frame coordinate는 다른 type이다. `R_i.T @ (x_j-x_i)`는 전역 회전에 불변인 local coordinate이고, `(x_j-x_i)`는 회전하는 vector다. v0.1 GNN의 기본 edge는 world displacement/direction이며, local-frame edge feature를 별도 irrep vector로 잘못 추가하지 않는다.

학습 기본값 `rigid_augmentation: true`는 parent crop에 proper rotation과 translation을 적용한다. Teacher와 student는 같은 변환을 공유한다. `translation_std`는 Å 단위이며 기본값은 1이다. Independent student/teacher rotation 및 사후 frame alignment는 v0.1 기본 task에 포함하지 않는다.

Reflection은 augmentation에 넣지 않는다. Reference block에는 cross product가 있으며 SO(3) 계약이다. O(3)로 확장하려면 polar/axial parity를 분리해야 한다.

## 7. Sidechain Cartesian 및 chi view

SC atom stem은 visible sidechain atoms와 visible Cα anchor를 사용한다. Element embedding, intra-residue connectivity, relative direction/distance를 사용한다. Residue identity가 이 경로에 포함될 수 있음을 허용한다.

Chi는 atom array의 연속 slice가 아니라 **정의된 네 atom name**으로 계산한다. 예를 들어 ILE χ2는 CA–CB–CG1–CD1이다.

기본 χ1…χ4를 사용한다. `include_chi5: true`이면 Arg terminal χ5 확장 슬롯을 제공한다. 이를 다섯 개의 독립적인 자유도를 모든 residue에 부여하는 것으로 해석하지 않는다.

\[
e(\chi)=(\cos(p\chi),\sin(p\chi))
\]

Naming symmetry가 π-periodic인 slot에는 p=2를 사용한다. 일반 slot은 p=1이다. `defined`는 residue chemistry에서 결정하고, `valid`는 실제 atom observation과 degeneracy로 결정한다.

SC/AA atom encoder는 element와 geometry를 기반으로 한다. Chi internal encoder가 naming symmetry를 정규화한 입력을 보며, learned latent에 임의의 raw-angle shift를 뒤늦게 적용하지 않는다.

## 8. View의 독립 forward 경로

단일 Cartesian encoder의 출력 head만 여러 개 붙이는 구조로 multi-view를 대신하지 않는다.

| View | 입력 | Encoder |
|---|---|---|
| seq | amino-acid tokens와 positions | SequenceEncoder |
| bb | visible backbone coordinates | BackboneEncoder |
| bb_internal | backbone angle/torsion/length values와 masks | InternalEncoder |
| sc | sidechain coordinates + visible backbone anchor | SidechainEncoder |
| chi | χ encoding, defined/valid masks | InternalEncoder(sidechain=True) |
| aa | BB states + SC states + sparse all-atom interactions | AllAtomFusion |

Internal-coordinate encoder는 raw Cartesian kNN graph를 받지 않는다. `MultiViewEncoder`는 내부적으로 AA 생성에 BB/SC를 사용할 수 있지만 predictor에는 task에서 요청한 context view만 반환한다.

## 9. Hidden fiber representation

\[
H=(S,V,T)
\]

```text
S: [N,C0]
V: [N,C1,3]
T: [N,C2,3,3], symmetric and traceless
```

T는 저장 공간상 9성분이나 5개의 독립 성분이다. 회전 Q에 대해:

\[
S'=S,\qquad V'=QV,\qquad T'=QTQ^T.
\]

Vector/tensor를 flatten해서 일반 dense layer로 섞지 않는다. Scalar invariant, vector norm, tensor Frobenius norm으로 gate/attention weight를 만든다. Channel mixing은 Cartesian 성분 전체에 같은 weight를 적용한다.

## 10. Reference와 CuEq backend

Reference는 직접적인 SO(3) contraction으로 message를 계산한다. Scalar↔vector↔tensor 정보를 결합하며, directional outer product의 STF component로 l=2를 생성한다.

CuEq는 실제 `FullyConnectedTensorProduct(hidden, spherical_harmonics, hidden)`를 호출한다. Cartesian STF와 CuEq의 설치된 spherical-harmonic basis 사이에 명시적인 change-of-basis를 둔다.

두 backend는 같은 물리적 type과 encoder interface를 공유하지만 weight-identical한 구현은 아니다. `reference` checkpoint를 `cueq-cuda`에 shape가 맞는다는 이유로 로드하지 않는다.

이번 제작 환경에서 reference만 실행되었다. CuEq source는 구현되었고 optional tests를 포함하지만, 실사용 전 설치 환경에서 검증해야 한다.

## 11. Atom → residue → atom hierarchy

Backbone atom stem은 네 backbone role만 구분하고 residue identity를 읽지 않는다. Sidechain stem은 element와 geometry를 사용한다.

각 원자의 local state를 invariant attention weight로 residue에 집계한다.

\[
H_i=\frac{\sum_{a\in i}w_a H_a}{\sum_{a\in i}w_a}
\]

BB residue trunk가 넓은 문맥을 처리한다. SC는 local atom representation을 유지한다. AA fusion은 BB/SC의 atom states를 결합하고 **inter-residue sparse atom interactions**를 처리한다. 다시 residue로 집계하고 residue 문맥을 atom에 전달한다.

최종 atom output은 최종 residue output의 단순 복사가 아니다. Local atom state와 contextual residue conditioning을 함께 포함한다.

## 12. 그래프

Spatial edges와 명시적 covalent/peptide edges를 union한다. Edge에는 distance RBF, unit direction, same-residue flag, signed sequence separation, bond flag를 둔다.

Hidden 좌표는 graph construction에 사용하지 않는다. 유효한 visible atom/CA만 graph node가 된다. Sequence separation은 canonical `seq_pos` 차이다. PDB author number나 observed-array index를 자동으로 canonical index로 대체하지 않는다.

Chunked cdist를 사용하므로 edge list는 sparse하지만 neighbor discovery는 여전히 O(N²) 거리 계산이다. Million-protein throughput을 검증한 spatial kernel이라는 주장은 하지 않는다.

Self edge는 index로 명시적으로 차단한다. 수치상 cdist diagonal이 0인지로 판단하지 않는다. 생성 과정에서 발견한 이 오류를 수정했고 regression test를 추가했다.

## 13. Global representation

Sequence에는 CLS token이 있다. 구조에는 `GlobalReadout`이 residue features를 invariant attention으로 집계한다.

\[
G^{BB}=G^{(0)}\oplus G^{(1)}\oplus G^{(2)},\quad G^{AA}=G^{AA,(0,1,2)}
\]

학습되는 상수 query는 scalar뿐이다. Nonzero world vector/tensor parameter를 global 초기값으로 두지 않는다. l>0 global features는 관측된 geometric state에서만 생성한다.

Global slot에 dummy atom coordinate를 부여하지 않는다. 하나의 global slot이 완전한 단백질 구조를 복원 가능하게 보존한다는 가정도 하지 않는다. Node representations가 주 objective다.

## 14. Crop 계약

Parent crop P는 128 또는 256의 contiguous sequence positions다. 현재 CPU smoke config만 더 작은 길이를 사용한다.

Teacher의 `full`은 기본적으로 **P 전체**다. Student는 P 안의 일부 관측을 가린다.

\[
X_{P,visible}\rightarrow Z_P^{teacher}
\]

작은 crop에서 원래 multi-domain protein 전체 embedding을 무조건 맞추는 objective는 기본 구현이 아니다. Crop-to-larger-parent, crop-to-full-protein, spatial crop은 별도 실험으로 남긴다.

Crop boundary를 가로질러 원본 전체에서 계산한 torsion/direction을 남기지 않는다. Strict crop에는 halo가 없다. Chain-break 및 missing residue를 포함하는 모든 dependency window를 invalid 처리한다.

## 15. 두 종류의 masking

**Representation conversion:** target modality를 predictor에서 제거한다. 다른 modality에 같은 기하 정보가 존재해도 허용한다. Cartesian→torsion은 이 범주다.

**Geometry infilling:** 실제 geometric observation을 숨긴다. 해당 좌표를 사용한 direction, torsion, frame, graph 등도 가린다. Full-input encoder를 돌린 뒤 target embedding만 지우는 것으로 대신하지 않는다.

두 task를 하나의 leakage 기준으로 혼동하지 않는다. 좌표 변환 학습이 성립하는 것은 이 프로젝트에서 금지된 일이 아니며, 그것만으로 장거리 의미 추론을 증명했다고 주장하지 않는 것이 중요하다.

## 16. 구현된 9개 task

| 이름 | Context | Target | l>0 loss | 추가 조건 |
|---|---|---|---|---|
| seq_to_bb | partial seq | bb | off | partial sequence→parent BB semantics |
| bb_to_seq | bb | seq | off | SC/AA 금지 |
| cart_to_internal | bb | bb_internal | off | deterministic geometry conversion 허용 |
| internal_to_bb | bb_internal | bb | off | orientation-free structural semantics |
| sc_to_chi | sc | chi | off | SC chemistry 허용 |
| chi_to_sc | chi + bb | sc | on | BB가 world orientation 제공 |
| sc_infill | seq + bb + partial aa | aa | on | target SC atoms 제거, BB 유지 |
| bb_infill | partial bb | bb | on | target geometry와 모든 파생량 제거 |
| aa_infill | seq + partial aa | aa | on | target residue의 모든 geometry 제거 |

SC/AA infilling은 target residue atom latent 예측 loss도 포함한다. Atom query는 canonical atom slot을 알 수 있는 **topology-known** task다. Atom count/name을 감춘 blind molecular completion task가 아니다.

## 17. Predictor

Context의 invariant node descriptor, type embedding, sequence position, global descriptors가 attention key/value의 scalar 경로가 된다. Node/global/atom query는 target type과 sequence position을 갖는다. Atom query만 slot identity를 추가한다.

Target xyz, target frame, hidden geometric edge는 query에 없다.

Scalar attention weight로 visible vector/tensor values를 집계한다. 따라서 target coordinates가 없는 node에도 equivariant latent query를 정의할 수 있다. 아무 structure context도 없으면 l>0는 0이며, sequence→world-vector MSE를 강요하지 않는다.

## 18. Learned target heads

각 view의 online/teacher projector는 semantic latent와 learned circular latent를 생성한다. Geometric view는 l>0 projection도 가진다.

```text
semantic: [L, Ds]
vector: [L, Dv, 3]
tensor: [L, Dt, 3, 3]
circular: [L, Kz, 2]
```

Kz는 physical χ 개수와 다르다. Circle의 첫 channel이 φ 또는 χ1이라고 주장하지 않는다. `bb_internal` 및 `chi` target에만 circular loss를 활성화한다.

Raw angle을 맞추는 보조 head가 기본 loss에 숨어 있지 않는다.

## 19. Teacher

v0.1은 EMA teacher다. Encoder, latent projector, global readout이 EMA 대상이다. Predictor는 EMA teacher에 포함하지 않는다.

\[
\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta.
\]

Teacher forward는 eval/no-grad다. Model.train()을 호출해도 teacher를 eval로 유지한다.

전체 view를 항상 혼합한 teacher 하나를 target으로 삼지 않는다. Target view에 필요한 입력만 읽는다. Task schedule은 각 view가 context에도 등장하여 online encoder가 gradient를 받도록 순환한다.

Mol-JEPA의 exact reproduction을 주장하지 않는다. EMA는 이 프로젝트의 안정화 선택이다.

## 20. Loss

Node semantic MSE, dimension-normalized equivariant Frobenius loss, unit-circle cosine distance를 사용한다.

\[
D_1=\frac{1}{3C_1}\|\hat V-V\|_F^2,
\quad D_2=\frac{1}{5C_2}\|\hat T-T\|_F^2.
\]

T는 9성분 storage이나 5 DOF이므로 denominator 5를 사용한다.

\[
D_{circle}=1-\langle\hat z,z\rangle.
\]

Loss는 valid target 평균→sample 평균으로 계산한다. 원자가 많은 sample이 전체 batch를 독점하지 않도록 atom loss도 sample 안에서 평균한다.

\[
L=L_{node}+\lambda_A L_{atom}+\lambda_G L_{global}+L_{reg}.
\]

Global target은 같은 parent crop이다. 모든 target invalid인 경우 gradient가 정의된 0 loss를 반환한다. Dummy invalid target을 실제 target처럼 MSE에 넣지 않는다.

## 21. Regularization과 연구 불확실성

EMA/stop-gradient/normalization 자체만으로 collapse 방지를 보장하지 않는다.

Online semantic latent에 variance-floor와 off-diagonal covariance penalty를 적용한다. 여러 sample에서 균형 있게 최대 32개 node를 뽑는다. Learned periodic latent에는 약한 circular variance floor를 둔다.

이는 **heuristic baseline**이며 물리적 torsion distribution을 uniform torus로 강제하지 않는다. l>0 Cartesian 각 성분에 Gaussian regularization을 적용하지 않는다. 회전 augmentation에 의한 variance를 semantic diversity로 오해하지 않는다.

이론 논문에서 제시한 manifold heat-kernel MMD나 특정 target distribution theorem을 이번 코드에서 재현했다고 주장하지 않는다. 해당 objective는 후속 확장이다.

## 22. Distributed와 checkpoint

DDP는 task별로 일부 parameter만 사용하므로 `find_unused_parameters=True`를 사용한다. 모든 rank가 같은 step의 task type을 사용하고, 각 rank의 protein/crop/mask는 독립 sampler RNG로 생성한다.

v0.1의 covariance/circular regularizer 통계는 rank-local이다. 전역 batch 통계와 동등하지 않다. DDP는 gradient를 평균한다.

Checkpoint는 online/teacher/predictor, optimizer, scheduler, global step, rank별 sampler RNG 및 torch CPU/CUDA RNG, configuration, manifest fingerprint와 world size를 저장한다. Atomic replace와 weights_only load를 사용한다.

Resume는 같은 world size, model/backend, task schedule, 학습 계획 및 dataset manifest에서 지원한다. 실행을 일찍 끊으려면 총 steps를 바꾸지 말고 `--stop-after`를 사용한다.

## 23. Downstream

Sequence-only는 sequence encoder만 실행한다. Backbone-only는 BB encoder만 실행하고 SC/AA를 요구하지 않는다. All-atom은 atom/residue/global representation을 반환한다.

Task head는 scalar prediction에 invariant descriptor를 사용한다. l>0 tensor를 flatten해서 orientation-dependent scalar classifier를 만들지 않는다.

실제 function, interface, pocket, affinity, sequence design 성능은 이 저장소의 smoke test로 입증되지 않는다. 그런 downstream dataset과 비교 실험은 사용자가 별도로 연결해야 한다.

## 24. Acceptance tests

필수 gate는 다음이다.

- 9개 task의 finite forward/backward 및 teacher gradient 차단.
- BB가 sequence/SC mutation에 영향을 받지 않는지.
- Hidden target coordinates를 바꾸어도 context와 prediction이 동일한지.
- BB/SC/AA와 predictor의 rotation/translation consistency.
- STF symmetry/trace, frame column convention, unit circle normalization.
- Origin atom, short chain, missing atom, internal chain break, crop boundary.
- Chi atom dependency와 naming symmetry.
- Atom–residue mapping 및 insertion-code 보존.
- Same-world-size checkpoint resume의 exact equality.
- 실제 CuEq가 없으면 optional test는 pass가 아니라 skip.

## 25. 연구 평가에서 필요한 비교

다음 ablation의 runner/결과를 이미 제공했다고 주장하지 않는다. 현재 config/task registry와 module 경계에서 확장할 수 있는 연구 계획이다.

Seq-only vs Seq↔BB; BB-only vs BB+SC/AA; all-Euclidean vs typed latent; node-only vs node+global; mean pooling vs learned global readout; conversion-only vs infilling; 128 vs 256 vs mixed; EMA vs teacher-free; raw reconstruction vs latent JEPA.

중요한 평가 조건은 같은 backbone 규모·데이터·학습량과 protein/cluster 단위 split이다. Crop 생성 이후 무작위 train/test 분리는 금지한다.
