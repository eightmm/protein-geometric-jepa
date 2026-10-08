# Implementation status

## Typed latents (as originally agreed)

Per-view learned typed latent heads sit after the encoders and are EMA-tracked with them: semantic scalars, l=1/l=2 irreps, learned circles (S^1), and opt-in S^2 directions / SO(3) frames. The predictor reads only the typed context latents, so every head weight that shapes a teacher target is trained by the prediction loss (verified with regularizers off). Per-kind losses; a sem variance+covariance and a per-channel circle floor; heat-kernel MMD (torus/sphere) and an all-Euclidean baseline as ablations. Rigid augmentation is off by default (measured no-op for this exactly equivariant model). The CuEq backend uses the ir_mul layout, so CPU paths work when the CUDA ops wheel is installed; actual CuEq CUDA tests pass on a Blackwell GPU. A 22-protein stochastic overfit (random 128/256 crops, fresh masks, task mixing, EMA) reaches 6.8x-chance centred retrieval with effective rank 44; without the covariance term rank collapses to 3. Checkpoint format 7 (samples are a pure function of the step; optimizer states are lists for AdamW + optional Muon); formats 3-5 load for inference but do not resume. Details: [TYPED_LATENT_KO.md](TYPED_LATENT_KO.md), [reports/typed_latent](../reports/typed_latent/).


Every section of the owner's design spec is mapped to code and tests in [SPEC_COMPLIANCE_KO.md](SPEC_COMPLIANCE_KO.md): residue pair frame geometry (6.3), SC local-frame coordinates (7.1), a rename-invariant atom loss (7.3), selectable teacher-free/cosine/mean-readout/raw-reconstruction baselines (13.3, 18.3, 19.2, 19.6), l>0 collapse diagnostics (20.3), downstream entry points with overlapping-window inference (25) and ablation configs for every comparison (27). Deferred by the spec itself: crop-to-larger-parent and full-protein global objectives (14.3), which need a teacher record different from the student's and so lie outside the packed-batch contract.

This is an executable research implementation, not a pretrained protein model or a docking model.

## JEPA objective and predictor fixes

Earlier fixes, found by executing the first implementation: JEPA targets were a fixed random projection for l>0 (untrained projectors), the AA atom target path contained an untrained activation, the predictor was one cross-attention layer whose l>0 output could not leave the span of context vectors, and atom queries were built from observed teacher atoms. Targets are now normalized teacher-encoder states, the predictor is a two-stage equivariant transformer with topology-derived atom queries, masks are exact-size multi-block spans, tasks mix per sample, and EMA follows a schedule. The reference message now has the full CG path set. Encoder extensions (`sc_context: spatial`, `effdock_directional`, `effdock_ffn: bilinear`, `effdock_adaptive_cutoff`) are implemented as opt-in experiments. **Checkpoints from an earlier revision/an earlier revision are rejected (format 2).** Details, council record and evidence: [JEPA_TARGETS_KO.md](JEPA_TARGETS_KO.md), [reports/targets](../reports/targets/).

Verified earlier on CPU (historical): 171 tests pass and 2 CUDA tests skip; actual CuEq 0.9.0 CPU tests (8) execute; four 18-step smoke demos, 2-process Gloo DDP, and 128/256-residue forward/backward for baseline/effdock/effdock-full pass. Not verified: CUDA/NCCL, real-corpus training, and whether representation quality improved (requires the ablation grid and frozen probes).

The sections below describe the interaction architecture and remain accurate unless superseded above.

## Implemented and integrated

Sequence Transformer, clean backbone encoder, sidechain atom stem, BB-to-AA activation reuse, AA atom/residue fusion, internal-coordinate and chi encoders, atom/residue/global latent heads, EMA target model, nine JEPA tasks, dependency-aware masking and visible-only graphs remain available. The new interaction can be selected in all five structural stages with `model.interaction: effdock`; old configurations remain `baseline`.

New operators include per-degree RMSNorm, shared TP with bounded dual radial scales, directed role/bond embeddings, learned distance decay, invariant content attention, three aggregation choices, stable norm rescaling, direction-preserving dropout, invariant conditional normalization and residual equivariant FFN. The actual CuEq backend is separate from the analytic reference. See [upgrade analysis](EFFDOCK_UPGRADE_KO.md).

The package includes runnable reference/CuEq CPU/CUDA presets, checkpoint/resume, torchrun DDP, feature export, an architecture validation script and a ten-variant ablation configuration writer. No jobs are submitted automatically by the ablation writer.

## Verified

**Analytic CPU:** 123 tests pass, nine optional backend/GPU tests skip. Both architectures pass 128/256-residue forward/backward across all nine tasks. The new architecture passes 18-step training, CPU/Gloo two-process training, exact same-configuration CPU resume, and isolation/equivariance regression tests.

**Actual CuEq 0.9.0 CPU:** GitHub CI successfully executes seven CuEq tests, the new architecture's 18-step training and nine 32-residue task backward passes. CUDA tests are excluded there. Independent reference CI passes on Python 3.11/3.12/3.13.

Exact environments, commit/run IDs, raw local reports and interpretation limits are in [INTERACTION_VALIDATION.md](INTERACTION_VALIDATION.md). Historical [VALIDATION.md](VALIDATION.md) and top-level old reports describe an earlier revision, not the new operator.

## Not verified or not implemented

CuEq CUDA fused kernels, CUDA/NCCL multi-GPU training, GPU throughput/memory benchmark, real-corpus pretraining and downstream performance remain unverified. Pretrained weights, EFF-Dock checkpoint import, ligand/NA encoders and force/refinement heads are not supplied. Existing model inputs are still canonical single-chain records; sequence alignment and homology clustering are external dataset responsibilities.

The hierarchy remains BB/SC/AA, not a single atom+residue heterogeneous graph as in EFF-Dock. SO(3) is supported; O(3) reflection parity equivalence is not claimed. Sparse edge output does not make chunked pairwise neighbor discovery subquadratic. The trainer processes records in Python and regularizer statistics are rank-local.

## Publication and compatibility

Code and specifications are published to `eightmm/protein-geometric-jepa`. The owner's public repository visibility is unchanged; neither EFF-Dock nor PLMol is modified. Baseline checkpoints remain baseline. Changing architecture/backend/ablation flags intentionally rejects resume rather than silently reusing incompatible state.

[UPGRADE_RUNBOOK.md](UPGRADE_RUNBOOK.md) gives the execution gates. Quality improvements are hypotheses for the documented ablations, not conclusions from smoke tests.
