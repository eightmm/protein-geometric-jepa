# Protein JEPA objective discussion

Read-only objective audit and public-literature council, 2026-10-08. Runtime
source, training defaults and existing working-tree changes were preserved.
No pretraining, GPU work, held-out experiment or publication was performed.
The proposed weights and improvements below are hypotheses, not validated
recipes.

## Discussion scope and evidence

Codex `gpt-6-astra` and Claude's served `claude-opus-5-5` completed independent
openings and one rebuttal round, all four calls exiting successfully. The
public council did not inspect the repository; rebuttals used bounded peer
excerpts and disclosed that limit. The parent read the complete answers,
verified primary references, inspected local source and ran the local probes.
Agreement is not implementation or performance verification.

Automatic approval review rejected the initial attempt to send private
repository architecture to these providers: the discussion request did not
explicitly authorize that sensitive egress. The completed council used only
public theory and papers, with local/task/history inspection forbidden. Local
code findings stayed local and were not sent back to the council.

Public council thread: `jepa-public-objective-discussion-20261008`; operation:
`ask-20261008T083707Z-c54231e2`. Source-bound CPU probes are in
[local_probes.json](local_probes.json). They describe mathematical or synthetic
behavior, not a learned shortcut or scientific-quality result.

## Current objective

The existing architecture is a valid geometric JEPA baseline: online encoders
and typed heads, an EMA teacher evaluated without gradients, and a non-EMA
predictor receiving allowed context plus mask/position tokens. Atom loss
compares normalized hidden latents, not reconstructed coordinates. Teacher
access to complete target geometry is intentional. Only student preprocessing,
context and queries must exclude forbidden hidden geometry.

For microbatch b with represented tasks T_b and context views V_b:

```text
L_b = mean over present tasks t [mean over samples of task t:
        D_node + 0.1 D_global + 0.2 D_atom]
    + mean over context views v [0.05 R_variance
                                 + 0.04 R_covariance + 0.05 R_circle]
```

Each node type is normalized over valid entries/channels, then its term is
summed: semantic MSE, vector Frobenius/3, STF tensor Frobenius/5, circular
1-cos, and optional direction/frame distances. Geometry terms are enabled
only where the context supplies an orientation. Global uses semantic and
irreps; regularization uses raw online context **node** semantic/circle states
with rank-local statistics.

Source: [pair loss](../../src/protein_jepa/models/jepa.py),
[task/view reduction](../../src/protein_jepa/models/jepa.py),
[typed distances](../../src/protein_jepa/models/latents.py),
[task registry](../../src/protein_jepa/objectives/tasks.py),
[sample schedule](../../src/protein_jepa/data/sampling.py),
[training accumulation](../../src/protein_jepa/train.py).
Relevant one-based lines at the audited source: jepa 210–241 and 277–313;
latents 257–277; tasks 16–25; sampling 22–38; train 143.

The nine tasks form three useful families:

| Family | Tasks | Uniform-schedule exposure |
|---|---|---:|
| Masked geometry | bb_infill, sc_infill, aa_infill | 3/9 |
| Sequence/structure cross-view | seq_to_bb, bb_to_seq | 2/9 |
| Geometry representation conversion | cart_to_internal, internal_to_bb, sc_to_chi, chi_to_sc | 4/9 |

These fractions describe schedule exposure, not measured gradient influence.
Conversion is allowed to contain deterministically shared geometry. Inverse
conversion is not always complete: torsions alone omit lengths, bond angles,
anchors and potentially closure constraints. SC may use observed BB as a
condition; statistical information about sequence from BB is not leakage.

## Confirmed limits and unverified risks

1. **Task equality does not mean equal type budget.** In the synthetic probe,
   bb_to_seq has semantic loss only; seq_to_bb and several conversions have
   semantic/circle losses; geometry infilling has semantic/vector/tensor/circle
   losses. Adding direction/frame terms changes the summed budget. Existing
   per-type dimension normalization is appropriate, but does not determine
   task priorities or equalize gradient influence.
2. **Accumulation partitions change the reduction.** For the same four pairs
   A,A,B,C, one task-balanced batch gives family coefficients 1/3,1/3,1/3;
   averaging microbatches (A,A) and (B,C) gives 1/2,1/4,1/4. Actual no-update
   synthetic forwards produced 2.372670 versus 2.800428. This is documented
   current behavior, not a newly discovered implementation violation. It
   matters when comparing objectives across accumulation/batch settings.
3. **SC specificity is a hypothesis needing a control.** Current sc_infill
   targets AA, while BB at the target residue is observed. AA's target mixes
   BB and SC, so an aggregate score may reward known BB strongly. Source proves
   the mixture, not that the trained model ignores SC. Added SC shape inputs
   and existing sidechain atom latent losses do not automatically quantify the
   SC-specific contribution.
4. **Anti-collapse statistics are insufficient.** A protein-independent
   position-only Fourier code has variance penalty 0, covariance penalty about
   1e-13 and effective rank 16. The circle floor is 0.1 for one constant point
   but 0 for a balanced pair of antipodes. These are counterexamples to
   sufficient semantic/diversity claims, not proof of an actual trained-model
   shortcut. Globals have diagnostics but no separate diversity regularizer;
   whether an added regularizer helps remains unknown.
5. **Robust regression must respect SO(3).** For the same vector rotated 45
   degrees, componentwise L1 changes from 1 to sqrt(2), while its L2 norm stays
   1 within float32 rounding. A scalar SmoothL1 ablation or robust penalty on
   an invariant irrep residual norm is defensible; blindly replacing vector/
   tensor Frobenius losses with componentwise L1 is not.

## Ranked next design

| Priority | Proposal | Strongest objection | Discriminating comparison and rejection rule |
|---|---|---|---|
| 1 | Define masked geometry as the primary family; explicitly budget cross-view and conversion auxiliaries. Make per-type weights explicit and reduce over the intended optimizer-step task population. | Easy conversions can teach useful structure and train context heads; suppressing them may damage transfer. | Compare current mixing with fixed family budgets at equal data/compute. Reject a new budget if held-out context-dependent/frozen-probe performance worsens or required teacher-path weights stop training. Verify partition invariance for the prediction reduction; keep regularizer sampling scope explicit. |
| 2 | Add an SC-specific EMA node target for masked sidechains, conditioned on allowed BB/sequence/environment. Keep AA infilling for joint packing. | Local SC teacher states may themselves be easy or dominated by known BB/identity. SC-absent residues yield invalid targets. | Compare AA-target and SC-target versions with identical masks/inputs. Require additional sensitivity to actual hidden SC and improvement over matched BB+sequence priors. Query construction must not use teacher SC presence. The SC typed head must receive student prediction gradients through allowed visible-SC context or another active task. Reject if only local conversion improves or BB-only context performs equally. |
| 3 | Preserve current normalization and regularizers as the reference; diagnose between-protein/global diversity and position shortcuts before replacing them. | Strong whitening/uniform priors can erase useful correlations; small batches cannot supply reliable global covariance. | Use matched length/position/visibility controls, per-protein strata, raw versus normalized statistics, and global invariant diagnostics. Add a weak global scalar regularizer only if a registered collapse criterion triggers; reject if rank rises without context dependence/transfer. |
| 4 | Give sequence-to-structure an explicitly invariant prediction contract; optionally test a small latent-relation head. | Pair targets are redundant with node targets, may be expensive, and still predict conditional summaries. | First retain the existing invariant loss. If needed, predict parameter-free teacher-latent contractions with an invariant head, using sequence/mask-selected pairs. Reject if it helps only the relation decoder or requires hidden-geometry-based query selection. |
| 5 | Test robustness or probabilistic heads only for a demonstrated failure: scalar outliers or multi-conformer uncertainty. | Distributional heads introduce mode collapse/calibration problems and complicate a representation-learning baseline. | Compare scalar MSE/SmoothL1 or invariant radial robust loss with fixed targets. Mixtures require relevant conformer data and calibration metrics. Reject if benefit is limited to training loss or physical ambiguity is not actually evaluated. |

A coherent prospective hierarchy is:

```text
L = alpha_I * masked typed-latent prediction
  + alpha_X * sequence/structure latent prediction
  + alpha_C * geometry-conversion latent prediction
  + existing eligible global/atom latent terms
  + type-appropriate anti-collapse regularization
```

Each family mean and its eligible type/component weights must be defined
explicitly. An illustrative pilot ratio alpha_I:alpha_X:alpha_C = 1:0.3:0.1
is a starting hypothesis only. Keep the current global/atom/regularizer values
as the reference while isolating one change at a time; changing every weight
and target together would obscure the cause of any result. SC-specific targets
and new relation heads are contract/architecture changes requiring dedicated
config, checkpoint and masking tests if later implemented.

## Parent decisions after rebuttals

- **Adopt:** masked contextual latent prediction as the main scientific
  question; explicit type/task budgets; homolog-aware evaluation; symmetry
  handling; context controls alongside collapse statistics.
- **Correct:** teacher seeing a complete target is allowed. BB conditioning,
  residue-identity correlation and visible geometric neighbours are legitimate
  information under the specified task, not automatically leakage.
- **Reject blanket removal of conversion auxiliaries:** deterministic input
  transforms add no observations but may provide inductive bias. Their
  contribution requires transfer comparisons, not a declaration of uselessness.
- **Do not remove current var/cov just because EMA exists:** historical fitting
  evidence supports keeping it as the reference; new-revision held-out benefit
  remains unverified. A detached auxiliary head is a probe, not a loss that
  updates the encoder.
- **Distinguish ambiguity from reconstruction:** squared latent prediction has
  a conditional-mean optimum for a fixed teacher. It is not itself averaging
  atom coordinates or promising conformer generation. Learned circles are not
  named physical torsions, and l=2 tensors are not generic pairwise distance maps.
- **Reject conclusive interpretation of random-teacher controls:** freezing a
  teacher at initialization changes target quality and dynamics together. It
  is a random-feature distillation baseline, not a clean diagnosis of why a
  learned teacher's loss decreased. Fixed-checkpoint context contrasts and
  mid-training fixed-teacher continuations are additional controls, not proofs.
- **Preserve disagreement:** auxiliary necessity, optimal fixed/adaptive
  weights, early global regularization and uncertainty heads are empirical
  choices. The council did not resolve them by experiment.

## Smallest useful experiment sequence

1. Before any training change, verify explicit type budgets and step-level
   prediction reduction on identical pairs; report valid node/global/atom
   denominators and branch-specific gradient probes. Existing eval-mode gradient
   probes are not actual accumulated/DDP step contributions.
2. Compare latent-infilling-focused, conversion-focused and combined recipes
   with the same encoder, data split, masks and compute. An inactive target
   head or teacher path must be removed/bypassed rather than left as an
   untrained EMA target when a task is disabled.
3. Compare SC-specific and AA targets only after that baseline is stable.
   Sequence-only, BB-conditioned and all-atom deployment modes get separate
   evaluations. A joint ranking must not hide a regression in one mode.
4. Use fixed-checkpoint true-context/content-removal/matched-context-swap
   comparisons. Content removal is an intervention, not a trained position-only
   baseline. A trained null predictor should use the same frozen teacher and
   fitting split as its comparator; letting its teacher change introduces
   another confound.
5. Pair independently audited homology splits with equal-capacity frozen probes,
   random-encoder and explicit-geometry baselines, and an independent protein
   task. Record seed variance before calling an auxiliary useful. The chosen
   split defines the generalization question; a single identity threshold is
   not universal proof of homology separation.

Reject a claimed improvement if it appears only in deterministic conversions,
vanishes under context controls, raises rank without independent transfer,
or depends on comparison of losses from different moving teachers. No pilot
was run here; local mathematical/synthetic probes are not such a pilot.

## Primary research checked

- [I-JEPA, method and masking](https://arxiv.org/html/2301.08243v3): masked
  contextual latent targets, an EMA teacher and target-position queries;
  masking choices were central in its image experiments. This motivates a
  protein design hypothesis, not an established protein recipe.
- [V-JEPA, objective](https://arxiv.org/html/2404.08471v1): uses L1 latent
  regression with EMA/stop-gradient in its visual setting. Its componentwise
  loss cannot be transferred unmodified to rotating Cartesian irreps.
- [data2vec](https://arxiv.org/abs/2202.03555): contextual target prediction
  across modalities; target averaging/normalization is a candidate, not a
  reason to discard the existing protein normalization without a comparison.
- [VICReg](https://arxiv.org/abs/2105.04906): variance/covariance regularization
  supplies an anti-collapse mechanism, not a semantic or shortcut guarantee.
- [LeJEPA](https://arxiv.org/abs/2511.08544): presents isotropic Gaussian
  regularization and a distinct teacher-free recipe. Adding SIGReg to an EMA
  model is a hybrid requiring its own evaluation, not automatically LeJEPA.
- [Beyond Gaussian Worlds](https://arxiv.org/abs/2609.21656): recovery claims
  depend on geometry, positive-pair dynamics and distribution matching.
  Its spherical/toroidal results do not establish that physical protein
  torsions should be uniformly distributed or that this mixed-type model
  satisfies the theorem's assumptions.

New direct coordinate/FAPE supervision, full-chain/cross-crop targets,
contrastive negatives, larger manifold bundles and simultaneous SIGReg/MMD
replacement are deferred. They may become separate justified experiments,
but do not make a model more JEPA merely by adding terms.
