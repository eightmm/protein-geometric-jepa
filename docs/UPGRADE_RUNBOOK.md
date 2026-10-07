# Upgrade runbook (v0.2 interaction, v0.3 JEPA)

v0.3 changes the JEPA objective and predictor and rejects v0.2 checkpoints (format 2); configurations containing `latent_*`, `circular_channels` or `circular_weight` fail with an explicit message — delete those keys. See [JEPA_V030_KO.md](JEPA_V030_KO.md). `configs/jepa_full_cueq_gpu.yaml` enables every opt-in v0.3 encoder experiment; `scripts/validate_effdock.py --variants effdock-full` checks it.

v0.2 added `model.interaction: effdock`. Existing configurations keep `baseline`. `backend` independently selects `reference`, `cueq-naive`, or `cueq-cuda`. The new block is wired to BB atoms, SC atoms, BB residues, AA atom fusion and AA residue fusion; sidechains and chi are retained.

## 1. Update and run the analytic reference

```bash
git pull --ff-only
python -m pip install -e '.[dev]'
pytest -q
protein-jepa demo --config configs/effdock_smoke.yaml --output runs/effdock-smoke
```

Use a new output directory. Existing checkpoints are not overwritten without an explicit resume. The smoke uses synthetic 16/24-residue crops and 18 optimizer steps; it is not real protein pretraining.

The reference architecture comparison is reproducible with:

```bash
python scripts/validate_effdock.py --config configs/effdock_smoke.yaml \
  --variants baseline effdock --lengths 128 256 \
  --output runs/effdock-crop-check.json
```

## 2. Actual CuEq CPU verification

The explicitly tested CuEq package version is 0.9.0. Install into a suitable environment rather than changing an unrelated training environment:

```bash
python -m pip install 'cuequivariance==0.9.0' 'cuequivariance-torch==0.9.0'
python -m pip install -e '.[dev,cueq]'
python -c 'import cuequivariance, cuequivariance_torch'
pytest tests/test_cueq.py -m 'not cuda' -q
protein-jepa demo --config configs/effdock_cueq_naive.yaml --output runs/effdock-cueq-cpu
```

The import preflight is intentional: a fully skipped suite must not be mistaken for a CuEq pass. [Actual CI evidence](V020_VALIDATION.md) is available. The public dependency range does not imply that all versions in it were tested.

## 3. CuEq GPU gate — required before large training

First install a CUDA-enabled PyTorch build compatible with the host driver, then install the NVIDIA CuEq ops wheel matching that CUDA major version and the CuEq core/Torch package version. Do not copy the CUDA 13 wheel choice from EFF-Dock to a CUDA 12 environment blindly. See the [NVIDIA installation instructions](https://docs.nvidia.com/cuda/cuequivariance/#installation).

```bash
python - <<'PY'
import torch
import cuequivariance
import cuequivariance_torch
assert torch.cuda.is_available(), 'A real CUDA GPU is required; do not count skips.'
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
PY
pytest tests/test_cueq.py -m cuda -q
python scripts/validate_effdock.py --config configs/effdock_smoke.yaml \
  --backend cueq-cuda --variants effdock --lengths 32 128 256 \
  --output runs/effdock-cuda-check.json
```

The validation script chooses the real GPU for `cueq-cuda`, synchronizes timing, records peak allocated GPU memory, and fails on nonfinite loss/gradient. It never substitutes the analytic reference. Start with FP32; AMP and fused-kernel compatibility require a separate validation. The CUDA tests cover end-to-end AA-infill backward, not a complete multi-GPU throughput benchmark.

Verified combination on a Blackwell GPU (compute capability 12.0): `torch 2.11.0+cu128` with `cuequivariance`, `cuequivariance-torch` and `cuequivariance-ops-torch-cu12` 0.9.0. See [JEPA_V030_KO.md](JEPA_V030_KO.md) §6c.

### Overfit check before long runs

```bash
protein-jepa overfit --config configs/effdock_smoke.yaml --device cuda \
  --records 1CRN.cif 1UBQ.cif --chain A --steps 1200 --output runs/overfit.json
```

Pass criterion: the loss falls and `node_top1` rises well above `chance`. A falling loss with chance-level retrieval means mean prediction or collapse, not learning.

## 4. Real corpus pretraining

Prepare canonical sequence-aligned records and a cluster-disjoint manifest as described in [DATA.md](DATA.md). Do not split crops from the same protein across train and validation. Crop lengths are 128/256 in the full presets.

```bash
protein-jepa audit-manifest data/train.jsonl
protein-jepa train --config configs/effdock_cueq_gpu.yaml \
  --manifest data/train.jsonl --output runs/effdock-pretrain
```

A backbone-only model is available through its separate inference path; do not remove SC/AA training objectives to obtain that property. The BB isolation tests guard it. Full presets are starting configurations, not tuned settings.

Two-device launch after single-GPU gates:

```bash
torchrun --standalone --nproc_per_node=2 -m protein_jepa.cli train \
  --config configs/effdock_cueq_gpu.yaml \
  --manifest data/train.jsonl --output runs/effdock-ddp
```

The supplied trainer implements DDP with task-dependent unused parameters. Regularizer statistics remain rank-local. CUDA/NCCL execution has not been tested by this release's CPU CI. Do not interpret the successful CPU/Gloo smoke as an NCCL pass.

## 5. Resume and export

```bash
protein-jepa train --config configs/effdock_cueq_gpu.yaml \
  --manifest data/train.jsonl --output runs/effdock-pretrain \
  --resume runs/effdock-pretrain/last.pt

protein-jepa encode --checkpoint runs/effdock-pretrain/last.pt \
  --record data/protein_A.npz --mode all_atom --output runs/protein_A_features.pt
```

Keep architecture, backend, dimensions, tasks, planned steps, dataset fingerprint and world size unchanged. Old v0.1 configuration fields default to baseline when loading; an old baseline checkpoint is not an EFF-Dock-style warm-start. To compare architectures, initialize separate runs. See [TRAINING.md](TRAINING.md) for checkpoint contracts and device details.

## 6. Ablation configurations

```bash
python scripts/make_effdock_ablations.py --base configs/effdock_cueq_gpu.yaml \
  --output runs/effdock-ablation-configs --seeds 17 29 43
```

This writes **30 complete YAML files** for ten variants and three seeds. It never starts training or submits jobs. Existing files cause an error before any generated configuration is overwritten. For a quick CPU functionality check, use `--base configs/effdock_smoke.yaml --seeds 17` instead.

Keep data, crop/mask policy, optimization and evaluation budget controlled. Both equal-step and equal-compute comparisons matter because the larger block costs more. Read [EFFDOCK_UPGRADE_KO.md](EFFDOCK_UPGRADE_KO.md) for the operator comparison, tradeoffs and prioritized next upgrades.
