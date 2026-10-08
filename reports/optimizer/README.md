# AdamW vs Muon + AdamW (22 proteins)

Same recipe as `reports/spec_completion/`: default-size model,
`interaction: effdock`, `backend: cueq-cuda`, current defaults
(`sc_local_frame: false`, `decay_exclusions: true`), stochastic 22-protein
overfit, batch 4, lr 1e-3 constant, 3000 steps. The overfit uses no weight
decay. Only `optimizer` differs. Single runs on a shared GPU.

| step 3000 | AdamW | Muon + AdamW |
|---|---|---|
| retrieval (node_top1, chance 0.016) | 0.148 | **0.380** |
| loss | 1.050 | **0.880** |
| pooled effective rank | 48.5 | 48.2 |
| per-view rank seq / bb_internal / chi / bb / aa / sc | 25.6 / 20.8 / 31.6 / 15.5 / 24.6 / 8.0 | 41.9 / 35.3 / 37.6 / 21.6 / 28.2 / 8.2 |

Muon reaches 2.6x the retrieval of AdamW at the same step with more spread
latents in every view. Muon runs on hidden `Linear` and QKV matrices
(`match_rms_adamw` scaling, one learning rate); everything else stays on
AdamW. One seed and a memorization setting: the gap needs confirming on a
held-out set before relying on it, but it is large.

The default stays `optimizer: adamw` because Muon needs PyTorch >= 2.9 while
the package allows torch >= 2.2.
