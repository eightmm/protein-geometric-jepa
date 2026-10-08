# v0.2.0 validation — EFF-Dock-inspired JEPA

Validation date: 2026-10-07. Code commit: `4de2b0e916fc254be9856229909d0478425bf318`.
Subsequent publication commits add documentation, diagnostic records and the ablation-config writer; they do not change this model operator. This report separates analytic CPU, actual CuEq CPU, and unexecuted CUDA validation.

## 1. Analytic CPU reference

Local environment: Python 3.13, PyTorch 2.10.0+cpu, one Torch thread, no CuEq installation and no CUDA device.

| Check | Result | Evidence |
|---|---|---|
| Complete suite | **123 passed, 9 skipped** | [pytest.txt](../reports/interaction/pytest.txt) |
| Both interaction variants, 128/256 residues, nine tasks each | **36/36 finite forward/backward** | [crop_diagnostics.json](../reports/interaction/crop_diagnostics.json) |
| New variant, 18 optimizer/EMA steps, all nine tasks twice | Passed | [smoke summary](../reports/interaction/smoke_summary.json), [log](../reports/interaction/smoke.log) |
| New variant, 2-process CPU/Gloo, 18 steps | Passed | [DDP summary](../reports/interaction/ddp_summary.json), [log](../reports/interaction/ddp.log) |
| Ten generated ablation configurations, one seed, tiny AA-infill backward | **10/10 passed** | [ablation_smoke.json](../reports/interaction/ablation_smoke.json) |

The nine local skips are optional CuEq/CUDA cases, NOT passes. Existing model-fixture tests now execute for both `baseline` and `effdock`: encoder/predictor/global equivariance, clean-BB isolation, hidden-target mutation and task gradients. New tests cover eight-block stacks, zero norm derivatives, directed edge roles, cutoff-edge equivalence, dropout with shared random masks, conditional normalization, configuration errors, exact resume with dropout, and old-v0.1 config defaulting.

The exact-resume test compares every online/teacher/predictor state tensor bitwise on the same CPU configuration. It does not establish cross-device or changed-world-size reproducibility. The two-process smoke establishes distributed execution, not global-batch covariance regularization.

## 2. Independent GitHub reference CI

[CPU correctness run 37573920416](https://github.com/eightmm/protein-geometric-jepa/actions/runs/37573920416) ran the same code commit on **Python 3.11, 3.12 and 3.13**. All three jobs succeeded: dependency installation, full pytest suite, original baseline CLI demo and new EFF-Dock-style CLI demo.

## 3. Actual CuEquivariance CPU — executed, not skipped

[CuEq CPU run 37573920388](https://github.com/eightmm/protein-geometric-jepa/actions/runs/37573920388), [job 112638454109](https://github.com/eightmm/protein-geometric-jepa/actions/runs/37573920388/job/112638454109), finished successfully. Observed environment: Ubuntu 24.04, Python **3.11.16**, PyTorch **2.14.1+cpu**, `cuequivariance==0.9.0`, `cuequivariance-torch==0.9.0`, NumPy **2.4.6**. These are the versions recorded in this run, not a claim about every supported version combination.

Executed commands:

```bash
python -c 'import cuequivariance, cuequivariance_torch'
python -m pytest tests/test_cueq.py -m 'not cuda' -q --junitxml=cueq.xml
python -m protein_jepa.cli demo --config configs/effdock_cueq_naive.yaml \
  --output /tmp/jepa-effdock-cueq --length 24 --count 4
python scripts/validate_effdock.py --config configs/effdock_cueq_naive.yaml \
  --variants effdock --lengths 32 --output cueq-smoke.json
```

The test command reported **7 passed, 2 deselected in 9.05s**. The two deselected tests require CUDA; they were intentionally excluded, not executed. The seven tests include the Cartesian↔CuEq basis round-trip, TP equivariance, baseline/new full-model backward, new BB/SC/AA node/atom/global equivariance, all-nine-task gradients, and masked-target mutation isolation. The actual `FullyConnectedTensorProduct` and `SphericalHarmonics` naive implementations were invoked.

The CLI completed **18 optimizer steps** across all nine tasks twice. The final AA-infill loss was `0.14437100291252136`, with gradient norm `0.3391108810901642`. The separate 32-residue diagnostic completed all nine task backward passes with finite gradients. These values document successful execution; they are not accuracy or convergence claims.

The workflow artifact `cueq-naive-verification` contains `cueq.xml`, the installed-package list, the task diagnostic JSON and training summary. The run warned that optional `opt_einsum_fx` was not installed; it did not prevent naive execution. A compact, explicitly transcribed record is retained in [cueq_ci_summary.json](../reports/interaction/cueq_ci_summary.json).

## 4. Measured cost, with narrow interpretation

The tiny analytic reference model used **75,255** trainable parameters for baseline and **117,890** for the new variant, excluding the EMA teacher. This is not the full training preset and not the CuEq model parameter count. In a one-shot 256-residue AA-infill measurement, baseline took approximately **0.359 s** and the new variant **0.511 s**. Timing includes a complete JEPA forward/backward, with no repeated warmup benchmark and no optimizer step.

Different tasks and teacher representations make raw loss comparisons unsuitable for ranking representation quality. No GPU speedup, docking improvement or downstream accuracy improvement has been established.

## 5. Not validated / not claimed

- CuEq **CUDA fused kernels**, CUDA/NCCL multi-GPU, GPU memory and throughput.
- Actual CuEq 128/256-residue runs in this release's CI; those lengths were tested with the analytic reference, while actual CuEq tested 32 residues plus 16/24-crop training.
- Full preset pretraining on a real protein corpus or any downstream task benchmark.
- EFF-Dock checkpoint compatibility, reflection/O(3) parity preservation or docking-force predictions.
- Cross-backend checkpoint portability; baseline/effdock and reference/CuEq remain distinct architectures/operators.

Run the GPU gates in [UPGRADE_RUNBOOK.md](UPGRADE_RUNBOOK.md) before expensive GPU training. Old [VALIDATION.md](VALIDATION.md) is the historical v0.1 report, not evidence for the new block.
