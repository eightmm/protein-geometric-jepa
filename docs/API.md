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
from protein_jepa.config import ModelConfig
from protein_jepa.models.jepa import ProteinJEPA
from protein_jepa.checkpoint import load_checkpoint

checkpoint = load_checkpoint("runs/pretrain/last.pt")
model = ProteinJEPA(ModelConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model"])

features = model.encode(record, mode="all_atom")["aa"]
print(features.nodes.s.shape)        # [L, C0]
print(features.nodes.v.shape)        # [L, C1, 3]
print(features.nodes.t.shape)        # [L, C2, 3, 3], STF
print(features.atoms.s.shape)        # [A_visible, C0]
print(features.atom_residue.shape)   # [A_visible]
print(features.atom_slot.shape)      # [A_visible]
print(features.global_state.s.shape) # [1, C0]
```

Atom outputs는 residue-major라는 가정을 하지 말고 mapping으로 정렬한다. AA fusion은 BB atom list와 SC atom list를 결합하므로 atom row 순서는 canonical dense slot 순서와 다를 수 있다. 항상 `(atom_residue, atom_slot)`을 사용한다.

## Sequence-only

```python
from protein_jepa.data.synthetic import sequence_record
record = sequence_record("MARGKKIGYS")
features = model.encode(record, mode="sequence")["seq"]
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
