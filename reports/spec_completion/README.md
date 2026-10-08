# Spec-completion regression check (22 proteins)

Same recipe as `reports/typed_latent/overfit22_chain_crop.*`: default-size
model, `interaction: effdock`, `backend: cueq-cuda`, 22 single-chain PDB
records (`reports/typed_latent/overfit22_records.txt`), stochastic mode
(random 128/256 crops, fresh masks, mixed tasks, EMA 0.99 -> 0.999), batch 4,
lr 1e-3, 3000 steps, evaluation of 198 fixed (crop, task) pairs every 300
steps. Single runs; the GPU was shared with other jobs (times are not comparable).

| run | code | retrieval (node_top1) | chance | pooled effective rank |
|---|---|---|---|---|
| before | prior main (`overfit22_chain_crop`) | 0.073 | 0.016 | 13.7 |
| `tier1` | pair frames + SC local frame + rename-invariant atom loss | 0.124 | 0.016 | 2.7 |
| `tier1_nopair` | tier1 without pair frame features | 0.155 | 0.016 | 2.0 |
| `tier1_nosc` | tier1 without SC local frame | 0.121 | 0.016 | 49.2 |
| `full_views` | tier1 + baselines/API (all defaults then on) | 0.178 | 0.016 | 2.5 |

Per-view effective rank of `full_views` at step 3000 (the pooled rank mixes
views of different scale): seq 30.7, chi 33.2, bb_internal 27.2, aa 28.0,
bb 17.5, sc 12.8. No view collapsed.

Reading:

- Every new configuration retrieves 1.7-2.4x better than before at 3000 steps.
- The pooled-rank drop comes from the SC local-frame input: without it the
  pooled rank is 49 with unchanged retrieval. It inflates the SC latent scale,
  so it is now opt-in (`sc_local_frame: false`); the spec lists it as optional.
- Pair frame features stay on (spec 6.3 requires them); the single-run
  difference against `tier1_nopair` is within what one seed can show.
- Pooled rank alone is a misleading collapse signal once views differ in
  scale; the overfit diagnostic now also reports `view_effective_rank`.
