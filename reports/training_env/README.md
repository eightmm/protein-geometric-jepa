# Training environments: verification record

Default model (`interaction: effdock`, `backend: cueq-cuda`), crops 128/256,
22 single-chain PDB records as an npz manifest. One RTX PRO 6000 Blackwell
(another job shared the GPU during the measurements), torch 2.11+cu128,
cuequivariance 0.9.0.

## Multi-process DDP

| setup | result |
|---|---|
| CPU, gloo, 2 ranks, `dropout: 0.1`, `accumulation_steps: 2`, `loader_workers: 1` (forkserver), 4 steps; full run vs stop at 2 + resume | identical weights (bitwise) |
| one GPU shared by 2 ranks, gloo, CuEq CUDA, batch 18, `accumulation_steps: 2`, `loader_workers: 2`, `pack_size: 9`; full vs stop/resume | runs and resumes; max weight difference 1.4e-5 (CUDA atomics are nondeterministic) |
| NCCL across GPUs, multi-node Slurm | **not run** (single-GPU machine) |

## Precision

Batch 72 train step and the relative change of per-sample losses under a
rigid rotation of the inputs (eval mode):

| mode | s/step | crops/s | rotation error |
|---|---|---|---|
| float32 (`highest`) | 1.981 | 36.3 | 3.2e-7 |
| TF32 matmul | 1.949 | 36.9 | 5.6e-4 |
| bf16 autocast | does not run (index_add dtype mismatch) | | |

TF32 buys 1.6% and costs 1700x in equivariance error, so `train()` pins
float32 matmul precision.

## Loader workers

Batch 72, real `train()` loop: 0 workers 29.3 crops/s, 4 workers 29.4 crops/s.
Building crops and masks takes about 3% of a step for these small records;
workers matter for large corpora or slow storage, where file decompression
grows. Host-to-device copies of the records take about 0.1 s per step.
