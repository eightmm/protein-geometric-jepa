# CuEquivariance backend contract — v0.2

## Architecture and execution backend are independent

`model.interaction` selects `baseline` or `effdock`. `model.backend` selects:

| Backend | Operator | Current verification |
|---|---|---|
| `reference` | Analytic Cartesian SO(3) message | CPU tested, both architectures |
| `cueq-naive` | Actual NVIDIA FCTP/SH, naive method | **CuEq 0.9.0 CPU CI passed**, both architectures |
| `cueq-cuda` | Actual NVIDIA FCTP fused TP / SH uniform_1d | Implemented; **GPU execution unverified** |

The reference is not a weight-identical replacement for CuEq. Backend selection is explicit and never silently falls back. The EFF-Dock-style block reuses `CuEqMessage` and the common Fiber interface. It does not import docking heads or original O(3) checkpoint weights.

## Representation bridge

The public Fiber contains scalar channels, Cartesian vectors and 3x3 symmetric-traceless tensors. A tensor has five mathematical degrees of freedom, even though stored in nine elements. CuEq uses a packed `mul_ir` layout with `C0+3*C1+5*C2` entries.

`models/cueq_backend.py` calibrates a fixed Cartesian-to-irrep map from the installed library's spherical harmonics. An orthonormal STF basis is projected through a least-squares calibration; inverse transforms reconstruct the Cartesian vector/tensor. No axis-order assumption or incorrect l=2 reshape is used. The calibration is coordinate independent and occurs on CPU at construction. Round-trip and rotated TP tests exercise it.

The tensor product uses internal shared weights across edges. Different layers and encoder stages own separate weights. SH degrees are 0,1,2, normalized directions. Other learned channel maps preserve the Fiber representation rather than using arbitrary flattened MLPs on magnetic components.

## Installation and verified environment

Actual remote CI ran Python 3.11.16, PyTorch 2.14.1+cpu, CuEq core/Torch 0.9.0. The optional package range is broader for installation flexibility but only this explicitly reported CuEq version was exercised. See [V020_VALIDATION.md](V020_VALIDATION.md) for exact evidence.

```bash
python -m pip install 'cuequivariance==0.9.0' 'cuequivariance-torch==0.9.0'
python -m pip install -e '.[dev,cueq]'
python -c 'import cuequivariance, cuequivariance_torch'
pytest tests/test_cueq.py -m 'not cuda' -q
protein-jepa demo --config configs/effdock_cueq_naive.yaml --output runs/cueq-naive
```

For CUDA, use a compatible CUDA-enabled PyTorch build and the NVIDIA ops wheel matching the CUDA major and CuEq package versions. Start FP32 and run explicit GPU preflights before full presets. A skipped `pytest -m cuda` is not a pass. Detailed commands are in [UPGRADE_RUNBOOK.md](UPGRADE_RUNBOOK.md).

## Checkpoint and symmetry limits

Old baseline configurations retain their default operator. Architecture, backend, shape or ablation changes are not supported for exact resume. Naive-to-fused checkpoint migration is not implemented as a portable resume path, even if learned shapes happen to agree. Do not discard learned-state mismatches. The model's contract is proper-rotation SO(3), not reflection O(3); source EFF-Dock parity copies are not silently identified.

Primary API references: [FullyConnectedTensorProduct](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.FullyConnectedTensorProduct.html), [SphericalHarmonics](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SphericalHarmonics.html), and [NVIDIA installation](https://docs.nvidia.com/cuda/cuequivariance/#installation).
