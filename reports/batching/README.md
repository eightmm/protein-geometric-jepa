# Packed batching throughput

Train steps (forward, backward, AdamW, EMA) of the default model with
`interaction: effdock`, `backend: cueq-cuda`, crops 128/256, mixed 9 tasks per
microbatch, on 22 single-chain PDB records (`reports/typed_latent/overfit22_records.txt`).
GPU: one RTX PRO 6000 Blackwell, torch 2.11+cu128, cuequivariance 0.9.0.

Both codes ran back to back while another training job shared the GPU, so
absolute numbers are pessimistic; the ratio is the comparable quantity.

- Before: one sample at a time, so crops/s does not grow with the batch size.
- Now: samples of the same task are one disjoint-union batch (graphs keep
  edges inside each record, transformers and predictor attention are padded
  per record, losses and diagnostics are per-record segment reductions).
  Throughput grows with the batch: 5.6 -> 35.9 crops/s at batch 72 (6.4x).
- Packed and separate computation agree to 2.2e-7 relative loss on 81 real
  samples with the CuEq CUDA kernels; `test_packed_group_equals_separate_samples`
  checks every task on CPU.

Profile notes that drove the changes: per-residue Python loops in chi
torsions and bond tables (now lookup tables), and the relative-position bias
lookup whose advanced-index backward serialized millions of pairs into 66
buckets (now an embedding lookup).
