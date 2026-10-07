# Implementation status — v0.3.0

This is an executable research implementation, not a pretrained protein model or a docking model.

## v0.3.0 (JEPA objective and predictor)

v0.3 fixes defects found by executing v0.2: JEPA targets were a fixed random projection for l>0 (untrained projectors), the AA atom target path contained an untrained activation, the predictor was one cross-attention layer whose l>0 output could not leave the span of context vectors, and atom queries were built from observed teacher atoms. Targets are now normalized teacher-encoder states, the predictor is a two-stage equivariant transformer with topology-derived atom queries, masks are exact-size multi-block spans, tasks mix per sample, and EMA follows a schedule. The reference message now has the full CG path set. Encoder extensions (`sc_context: spatial`, `effdock_directional`, `effdock_ffn: bilinear`, `effdock_adaptive_cutoff`) are implemented as opt-in experiments. **Checkpoints from v0.1/v0.2 are rejected (format 2).** Details, council record and evidence: [JEPA_V030_KO.md](JEPA_V030_KO.md), [reports/v030](../reports/v030/).

Verified for v0.3 on CPU: 171 tests pass and 2 CUDA tests skip; actual CuEq 0.9.0 CPU tests (8) execute; four 18-step smoke demos, 2-process Gloo DDP, and 128/256-residue forward/backward for baseline/effdock/effdock-full pass. Not verified: CUDA/NCCL, real-corpus training, and whether representation quality improved (requires the ablation grid and frozen probes).

The sections below describe the v0.2 interaction release and remain accurate unless superseded above.

## Implemented and integrated

Sequence Transformer, clean backbone encoder, sidechain atom stem, BB-to-AA activation reuse, AA atom/residue fusion, internal-coordinate and chi encoders, atom/residue/global latent heads, EMA target model, nine JEPA tasks, dependency-aware masking and visible-only graphs remain available. The new interaction can be selected in all five structural stages with `model.interaction: effdock`; old configurations remain `baseline`.

New operators include per-degree RMSNorm, shared TP with bounded dual radial scales, directed role/bond embeddings, learned distance decay, invariant content attention, three aggregation choices, stable norm rescaling, direction-preserving dropout, invariant conditional normalization and residual equivariant FFN. The actual CuEq backend is separate from the analytic reference. See [upgrade analysis](EFFDOCK_UPGRADE_KO.md).

The package includes runnable reference/CuEq CPU/CUDA presets, checkpoint/resume, torchrun DDP, feature export, an architecture validation script and a ten-variant ablation configuration writer. No jobs are submitted automatically by the ablation writer.

## Verified

**Analytic CPU:** 123 tests pass, nine optional backend/GPU tests skip. Both architectures pass 128/256-residue forward/backward across all nine tasks. The new architecture passes 18-step training, CPU/Gloo two-process training, exact same-configuration CPU resume, and isolation/equivariance regression tests.

**Actual CuEq 0.9.0 CPU:** GitHub CI successfully executes seven CuEq tests, the new architecture's 18-step training and nine 32-residue task backward passes. CUDA tests are excluded there. Independent reference CI passes on Python 3.11/3.12/3.13.

Exact environments, commit/run IDs, raw local reports and interpretation limits are in [V020_VALIDATION.md](V020_VALIDATION.md). Historical [VALIDATION.md](VALIDATION.md) and top-level old reports describe v0.1, not the new operator.

## Not verified or not implemented

CuEq CUDA fused kernels, CUDA/NCCL multi-GPU training, GPU throughput/memory benchmark, real-corpus pretraining and downstream performance remain unverified. Pretrained weights, EFF-Dock checkpoint import, ligand/NA encoders and force/refinement heads are not supplied. Existing model inputs are still canonical single-chain records; sequence alignment and homology clustering are external dataset responsibilities.

The hierarchy remains BB/SC/AA, not a single atom+residue heterogeneous graph as in EFF-Dock. SO(3) is supported; O(3) reflection parity equivalence is not claimed. Sparse edge output does not make chunked pairwise neighbor discovery subquadratic. The trainer processes records in Python and regularizer statistics are rank-local.

## Publication and compatibility

Code and specifications are published to `eightmm/protein-geometric-jepa`. The owner's public repository visibility is unchanged; neither EFF-Dock nor PLMol is modified. Baseline checkpoints remain baseline. Changing architecture/backend/ablation flags intentionally rejects resume rather than silently reusing incompatible state.

[UPGRADE_RUNBOOK.md](UPGRADE_RUNBOOK.md) gives the execution gates. Quality improvements are hypotheses for the documented ablations, not conclusions from smoke tests.
