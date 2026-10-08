# SC shape and vectorized geometry

Local implementation and CPU evidence, 2026-10-08. No new GPU training,
real-data throughput or downstream quality result is claimed.

## Representation

Fresh configurations enable `sc_shape_features`. Visible SC heavy atoms,
excluding OXT, provide unweighted centroid offset from an observed CA and
population covariance about their centroid. The scalar inputs are offset
norm/4, covariance trace/16 and separate shape/anchor validity flags; the
vector is offset/4 and the l=2 tensor is STF covariance/16. A bias-free
projection adds a 0.1 residual to existing SC node states. AA consumes it
through the existing SC fusion; pure BB and query generation are unchanged.
Latent output widths and objectives are unchanged. SC absence yields zero
features and false validity; CA absence removes only the anchor-dependent
offset. A single observed SC atom has valid zero covariance.

`sc_shape_features: false` is the `no_sc_shape` ablation. Older stored configs
missing the option restore false and their parameter layout. Format 6 is
retained; resume rejects a different architecture configuration. Feature
refactoring can change arithmetic at float32 rounding precision, so exact
cross-source-version bitwise training replay is not claimed.

## Vectorization and reuse

Four named BB torsion windows share one batched dihedral calculation, and
three BB bond directions share one normalization. SC moments use masked
reductions and a batched 3x3 covariance product, without a residue/atom Python
loop. A full six-view encoder call computes BB features once instead of five
times. SC-only calls calculate the N-CA-C frame without the unused BB angles.
Geometry reuse stays within one forward and one observation mask.

The implementation adapts named slots, explicit observation masks and point
geometry from the local `plmol` library; it adds no dependency on `plmol` and
does not copy its SASA, residue properties or CA-as-SC fallback.

## Verification

- Full CPU suite: **319 passed, 2 CUDA skips**, including installed CuEq CPU
  paths. After unused-import cleanup and the missing-coordinate NaN assertion,
  38 data/geometry/parser regressions passed. Syntax, Ruff F checks and all
  39 local documentation links passed.
- Analytic centroid/covariance, empty/single-atom/missing-anchor cases,
  permutation and rigid-motion invariance, hidden-coordinate isolation,
  record packing, BB isolation, and online training of EMA target-path weights.
- Zero SC shape projection restores the encoder without shape exactly.
  Shared geometry matches standalone paths exactly for full and masked inputs.
- The immutable pre-change feature implementation was compared on 28 cases:
  lengths 1/2/3/4/14/128/256, full/masked/broken/packed records. All validity
  masks match; largest numeric difference is `1.1920929e-7`.
- With shared common weights and shape disabled, before/after encoder states
  differ by at most `5.9604645e-7` on 128/256-residue synthetic fixtures.

## CPU timing

Raw measurements, configurations and source SHA256s are in
[benchmark.json](benchmark.json). The baseline is an immutable snapshot of
the local working tree before this SC-shape/vectorization change, including
the preceding global-contract changes; it is not a published commit.

Same-process, fixed allowed CPU, one Torch thread, shared common weights,
randomized interleaving, three rounds:

| BB feature kernel | Before | Vectorized | Reduction |
|---|---:|---:|---:|
| 128 residues | 0.995 ms | 0.637 ms | 35.9% |
| 256 residues | 1.067 ms | 0.762 ms | 28.6% |

End-to-end encoder speedup is **not established**. Separate-process runs
varied substantially even for the unchanged chi kernel. Paired all-view
timings also varied; the 128-residue shape model ranged from 84.6 to 126.3 ms.
A further eight-forward rotated-order probe measured the following CPU-time
medians (wall and CPU medians were nearly equal in that probe):

| All six views | Before | Vectorized, shape off | Vectorized, shape on |
|---|---:|---:|---:|
| 128 residues | 90.55 ms | 87.88 ms | 88.10 ms |
| 256 residues | 225.75 ms | 227.50 ms | 230.42 ms |

The controlled kernel improvement and reduced call count are supported.
Shape inputs add work, and graph/encoder time dominates these fixtures. The
cause of earlier timing variation was not isolated; it cannot be attributed
to a particular host scheduling effect from these data. These are synthetic
CPU microbenchmarks, not GPU throughput or scientific-quality evidence.

Re-run current measurements:

```bash
PYTHONPATH=src python scripts/benchmark_geometry.py --output /tmp/shape-on.json
PYTHONPATH=src python scripts/benchmark_geometry.py --no-sc-shape --output /tmp/shape-off.json
```

Compare quality with the existing held-out retrieval/global diagnostics and
independently audited frozen probes before concluding that shape inputs help.
