# Python API와 downstream

## Geometry-only extraction

```python
from protein_jepa.data.records import ProteinRecord
from protein_jepa.geometry.features import backbone_features, chi_features

record = ProteinRecord.load("data/protein_A.npz")
bb = backbone_features(record)
chi = chi_features(record, include_chi5=False)

assert bb.internal.shape == (len(record), 18)
assert bb.directions.shape == (len(record), 9, 3)
assert chi.encoded.shape == (len(record), 4, 2)
```

`bb.periodic` slot order는 φ, ψ, incoming ω, Cα pseudo-dihedral이다. `bb.directions`는 prev/next, N→CA, CA→C, C→O, virtual-CB, frame의 세 column이다. Mask와 함께 사용한다.

## Model encoding

```python
from protein_jepa.config import model_config
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.checkpoint import load_checkpoint

checkpoint = load_checkpoint("runs/pretrain/last.pt")
# model_config(): a stored config that predates an architecture option
# rebuilds the architecture it was trained with.
from protein_jepa.config import model_config
model = ProteinJEPA(model_config(checkpoint["model_config"]))
model.load_state_dict(checkpoint["model"])

features = model.encode_all_atom(record)          # = model.encode(record, "all_atom")["aa"]
print(features.nodes.s.shape)        # [L, C0]
print(features.nodes.v.shape)        # [L, C1, 3]
print(features.nodes.t.shape)        # [L, C2, 3, 3], STF
print(features.atoms.s.shape)        # [A_visible, C0]
print(features.atom_residue.shape)   # [A_visible]
print(features.atom_slot.shape)      # [A_visible]
print(features.global_state.s.shape) # [1, C0]
```

Atom outputs는 residue-major라는 가정을 하지 말고 mapping으로 정렬한다. AA fusion은 BB atom list와 SC atom list를 결합하므로 atom row 순서는 canonical dense slot 순서와 다를 수 있다. 항상 `(atom_residue, atom_slot)`을 사용한다.

진입점은 네 가지다(설계 명세 25절). 추론에는 teacher를 쓰지 않는다.

| 함수 | 입력 | 출력 |
|---|---|---|
| `encode_sequence(seq 또는 record)` | 서열 | residue states, sequence CLS |
| `encode_backbone(record)` | backbone | backbone atom/residue states, global irreps. SC/AA 경로는 실행하지 않는다 |
| `encode_all_atom(record)` | 전체 원자 | atom/residue states, global irreps |
| `encode_multimodal(record)` | 전부 | view별 출력 dict. 하나로 합친 joint token은 없다 |

모드마다 그 view encoder의 표현이 나온다. 서열 표현과 backbone 표현은 같은 벡터 공간이 아니고, pretraining의 cross-view 예측으로만 연결된다. Typed latent(semantic, l=1/l=2, S¹, S², SO(3))가 필요하면 `model.latents(record, mode)`를 쓴다.

## 긴 단백질: 겹치는 window 추론

```python
views, starts = model.encode_windows(long_record, mode="all_atom", window=256, stride=128)
nodes = views["aa"].nodes            # 전체 길이 [L, ...], window 중앙에 가중치를 둔 평균
globals_ = views["aa"].global_state  # window마다 하나 [W, ...]; 평균하지 않는다
```

128/256 crop으로 학습한 모델이 긴 단백질에서도 같은 성능을 낸다고 가정하지 않는다. Node와 atom 상태는 그 위치를 포함하는 window들의 평균이다. window 가장자리 residue는 문맥이 적으므로 중앙 쪽 window에 더 큰 가중치를 준다. 모든 window가 같은 world frame을 쓰므로 l=1/l=2 상태를 평균해도 등변성이 유지된다. Crop global의 평균은 전체 단백질 global이 아니므로, global은 window별로 돌려준다.

## Sequence-only

```python
features = model.encode_sequence("MARGKKIGYS")
```

이 경로는 BB/SC/AA encoder를 실행하지 않는다. Unknown residue는 X로 입력할 수 있으며 known sequence mask가 이를 구분한다.

## Frozen downstream head

```python
from protein_jepa.downstream import InvariantProteinHead, InvariantNodeHead

model.eval()
features = model.encode(record_with_structure, mode="all_atom")["aa"]
protein_head = InvariantProteinHead(model.cfg.dims, outputs=1)
residue_head = InvariantNodeHead(model.cfg.dims, outputs=3)

protein_prediction = protein_head(features.nodes, features.node_valid)
residue_logits = residue_head(features.nodes)
# Apply the actual dataset's valid label mask before task loss.
```

Classification/regression loss와 train/val/test task data는 별도로 연결한다. 위 head는 생성 직후 random initialization이다.

`protein_jepa.downstream.frozen_linear_probe`는 이미 추출한 invariant feature 행렬에 CPU ridge classification/regression을 적합한다. `train_clusters`/`eval_clusters`는 sample별 독립 homology audit의 cluster ID이며 중복 cluster를 거부한다. Feature 평균·scale과 target baseline은 fitting split에서만 계산한다. 함수는 encoder gradient를 만들지 않으며 prediction, 계수와 train-only 통계, 정확도/MSE 및 단순 baseline을 반환한다. Cluster ID의 실제 생물학적 타당성이나 label provenance는 caller가 검증해야 한다. 실제 label dataset은 제공하지 않는다.

`protein_jepa.geometry.features.sidechain_shape_features(record, visible)`는 관측 SC centroid−CA offset `[L,3]`(Å), population covariance `[L,3,3]`(Å²), SC 존재 validity와 CA anchor validity를 반환한다. Visible mask를 적용한 뒤 원자별로 동일한 가중치를 사용하며 OXT는 제외한다. SC가 없으면 covariance/offset은 0, CA만 없으면 offset은 0이다. Covariance는 SC가 관측되면 유지한다. 이는 학습된 latent가 아니라 encoder 입력 geometry이다.

## Fine-tuning

`model.encode()`는 feature export를 위한 no-grad API다. Encoder까지 fine-tune하려면 online encoder를 직접 호출한다.

```python
model.online.encoder.train()
features = model.online.encoder(record_with_structure, ("aa",))["aa"]
logits = residue_head(features.nodes)
loss = task_loss(logits[label_mask], labels[label_mask])
loss.backward()
```

Teacher/predictor가 필요 없는 downstream에서는 online encoder와 task head만 optimizer에 포함한다. Structure의 scalar task head에 world vector/tensor를 flatten해서 넣지 않고 `Fiber.invariant()` 또는 invariant readout을 사용한다.

## 관측 마스킹

```python
visible = record.present.clone()
visible[20:30, 4:] = False  # SC-only infilling; BB remains visible
context = model.online.encoder(record, ("bb", "aa"), atom_visible=visible)
```

Raw coordinate masking을 모델 진입 전에 반영한다. Full encoder output을 계산한 뒤 일부 latent를 지우는 것을 strict geometry masking이라고 부르지 않는다. BB 내부 feature와 graph가 visible mask를 따라 재계산된다.
