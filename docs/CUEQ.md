# CuEquivariance backend contract

## Three explicit modes

| backend | Implementation | Build-environment verification |
|---|---|---|
| `reference` | Native PyTorch Cartesian SO(3) contractions | CPU executed |
| `cueq-naive` | CuEq FullyConnectedTensorProduct + SphericalHarmonics, `method="naive"` | Not executed; dependencies unavailable |
| `cueq-cuda` | CuEq `fused_tp` and `uniform_1d` kernels | Not executed; no CUDA GPU |

Unavailable backends raise an error. A passing reference test is **not** a CuEq kernel test. Optional CuEq tests skip explicitly when dependencies are absent.

## Internal representation and basis bridge

The portable encoder stores scalar S, Cartesian vector V, and symmetric-traceless tensor T. T uses a 3×3 matrix but has five independent components. CuEq's `mul_ir` flattened representation has width:

```text
C0 + 3*C1 + 5*C2
```

`CuEqMessage` computes a fixed change of basis using spherical harmonics from the **installed CuEq version**. An orthonormal Cartesian STF basis B_k maps a tensor to five components. Spherical harmonics evaluated on deterministic directions determine invertible matrices M1 and M2. Packing uses V·M1 and (T:B)·M2; unpacking uses the inverses. The calibration is a numerical basis conversion, not a trainable geometry encoder or learned target.

This avoids assuming a particular ordering/sign convention for l=2 harmonics. A calibration residual check fails fast on unsupported conventions. `test_cueq_bridge_and_cpu_equivariance` verifies roundtrip, equivariance and backward when packages are present.

## Actual CuEq operation

```python
irreps = cue.Irreps(cue.SO3, "C0x0 + C1x1 + C2x2")
harmonics = cue.Irreps(cue.SO3, "1x0 + 1x1 + 1x2")
# Actual code substitutes numeric multiplicities.
tp = cuet.FullyConnectedTensorProduct(
    irreps, harmonics, irreps,
    layout=cue.mul_ir,
    shared_weights=True,
    internal_weights=True,
    method="naive",  # or fused_tp
)
```

Invariant radial/relation gates modulate output channels. Message aggregation and scalar gates remain in PyTorch. Calling this backend does not imply all graph construction or attention has been fused into a CUDA kernel.

## Installation and validation order

```bash
python -m pip install -e '.[dev,cueq]'
pytest -q -m cueq
protein-jepa demo --config configs/cueq_naive.yaml --output runs/cueq_naive
```

For a GPU environment, install the NVIDIA CuEq operations distribution appropriate to the installed PyTorch/CUDA combination following the official documentation. Exact CUDA wheel compatibility was not established in this environment; do not treat a guessed package combination as tested.

Then run:

```bash
pytest -q -m cuda
protein-jepa train --config configs/cueq_gpu.yaml \
  --manifest data/manifest.jsonl --output runs/cueq_gpu
```

Before large-scale training, verify memory use, throughput, gradient finite checks, rotation tests, and checkpoint resume on the actual GPU. A100/4090/H100 throughput is not reported because none was measured.

## Backend checkpoint compatibility

Reference and CuEq implement different parameterizations. They have the same Fiber input/output types, but are not weight-identical implementations. Checkpoints encode the backend and exact model configuration; resume rejects a changed backend.

The optional API was implemented against NVIDIA's documented `method`, `layout`, `FullyConnectedTensorProduct`, and `SphericalHarmonics` interfaces. `cueq>=0.8` is an installation range, not a claim that every version in that range was tested. Pin the exact successfully validated versions in the deployment environment.

Sources: [CuEq FCTP](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.FullyConnectedTensorProduct.html), [Spherical harmonics](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SphericalHarmonics.html).
