# GFT rebuild — architecture & traceability (master doc)

Status: **Complete — all six modules written, self-tested, documented.** The
pipeline runs end to end on synthetic data (features → data → gft/ga →
ablation → viz); on the real fleet it needs only `data.build_cache()` first.

This document is the entry point for the rebuilt project. Every module has a
companion `NN_<module>.md` in this folder recording the decisions behind it, so
each artifact is traceable to a justification. Nothing here touches the old files
— the rebuild lives entirely under `gftree/`.

---

## 1. Why a clean rebuild rather than an edit

The old pipeline earned its results but accreted three kinds of weight that a
full-tree, subtree-ablation project should not carry forward:

1. **Dead capability.** `learn_mf` (learnable membership functions, the whole of
   old `gft.py` §1b: `_decode`/`_encode`, gap-encoding, `MF_PAD`, `plot_memberships`)
   was measured and *dropped* — fixed-MF won on RMSE with ~50× lower run-to-run
   variance. Roughly a third of the old core exists only to support a feature we
   no longer use. The rebuilt core is fixed-MF only.
2. **Scattered ablation.** Model selection and evidence are spread across
   `run.py` (single split), `confirm.py` (repeated CV), `leaf_search.py` (leaf
   selection), `add_age_NASA.py` (a one-off patch to backfill a metric), and
   `nn_baselines.py`. Reproducing "the ablation" means running four scripts in the
   right order. The rebuild collapses this into one `ablation.py` that runs every
   arm in a single pass.
3. **Four-modifier ceiling.** The old preprocessing hard-coded the HPT/LPT
   modifiers. The full tree diagnoses all five components, so the data layer had
   to be widened to ten modifiers regardless of anything else — which is why it
   was the natural first thing to rebuild.

## 2. Module layout (new)

| module | replaces | status |
|---|---|---|
| `features.py` | `ncmapss_features.py` | **done, tested** |
| `data.py` | `datasets.py` | **done, tested** |
| `gft.py` (new) | `gft.py` (tree+FIS+fit/predict, simplified) | **done, tested** |
| `ga.py` | the GA buried in old `gft.py` §4 | **done, tested** |
| `ablation.py` | `run.py` + `confirm.py` + `leaf_search.py` + `add_age_NASA.py` | **done, tested** |
| `viz.py` | `gft_analysis.py` + `visualize_final.py` | **done, tested** |

The subtree decomposition is resolved (see §5): the core fits **single-branch
trees** (one fault mode's leaves → its spool(s) → RUL) via `fit_branch`, and the
**assembled** tree via `fit_full`, warm-started from the branches.

The core is split into `gft.py` (fuzzy inference + tree + subtree specs +
fit/predict, kept deliberately small) and `ga.py` (the generic real-coded
optimizer). The optimizer is model-agnostic, so isolating it keeps the "train
subtrees and the full tree" requirement to a single fit harness that hands the GA
a loss — the GA never needs to know which tree it is optimizing.

## 3. Cleanup — disposition of every old file

Deleted outright (no successor):
- `test.py` — two-line scratch, absorbed into module self-tests.
- `add_age_NASA.py` — existed only to backfill the age-NASA column into a stale
  CSV. The new `ablation.py` computes age-NASA natively, so the patch is moot.

Merged / superseded (successor in the table above):
- `run.py`, `confirm.py`, `leaf_search.py` → `ablation.py`.
- `gft_analysis.py`, `visualize_final.py` → `viz.py`.
- `ncmapss_features.py` → `features.py`; `datasets.py` → `data.py`;
  old `gft.py` → new `gft.py` + `ga.py`.

Dropped from the pipeline (kept only as optional external context):
- `nn_baselines.py` — PyTorch dependency, and the new ablation is GFT-internal
  (full vs subtrees vs sensor-free baselines). The deep-net comparison is a
  separate, same-domain context experiment; it is not re-derived here and is not
  needed to run the ablation. Left out of the rebuild; can be re-attached later.

Old-core functions with **no successor** (dead in a fixed-MF world):
`_decode`, `_encode`, `_mono_encode`, `seed_genome`'s MF branch, `GAP_MIN`,
`MF_PAD`, `n_mf_params`, `plot_memberships`, `loo_cv` (leave-one-*unit*-out,
never used — LOSO is the protocol), `official_split` (unused).

Outputs (`*.png`, `*.csv`) are regenerated, not carried over.

## 4. What the rebuilt core keeps (the parts that are load-bearing)

These survived scrutiny in the old project and are ported verbatim in spirit:
- **Fixed Ruspini partition + zero-order Sugeno** on a full rule grid.
- **Structural monotonicity** (`hp`, `lp`, `RUL` non-increasing by construction),
  because damage cannot fall with more damage — physics, not a preference.
- **`[0,1]` anchoring of latent spool nodes** (old fix #1), removing the gauge
  freedom that let a healthy engine read ~0.48 "damage".
- **Degrading-row conditioning of spool-input MF centres** (old fix #2), so the
  anchored ramp grades instead of flooring.
- **Per-unit condition residualization + causal EWMA**, the leak-free damage
  signal.
- **Capped RUL target, NASA + RMSE scoring, by-unit / by-mode evaluation.**

## 5. Subtree decomposition — PER-COMPONENT SINGLE BRANCHES

A "subtree" is a **single-component branch**: one component's leaf(s) → its damage
node → `RUL`, with every other component absent ("dead"). Trained on that
component's single-fault dataset. Five branches:

```
HPT branch :  hpt leaves -> hpt damage -> RUL(hpt)     trained on the HPT-only file
LPT branch :  lpt leaves -> lpt damage -> RUL(lpt)     trained on the LPT-only file
fan branch :  fan leaves -> fan damage -> RUL(fan)     trained on the fan-only file
LPC branch :  lpc leaves -> lpc damage -> RUL(lpc)     trained on the LPC-only file
HPC branch :  hpc leaves -> hpc damage -> RUL(hpc)     trained on the HPC-only file
```

Each branch answers two questions that a joint fit confounds:
1. **Leaf observability** — which sensors give a good RMSE/ρ against this
   component's θ (the `leaf_search` question, now per branch). The HPT_flow verdict
   (ρ≈0.27, unreadable → dropped) is the template.
2. **Single-mode RUL** — does this branch *alone*, in its own failure scenario,
   predict RUL? A branch whose θ is readable but whose damage does not track RUL is
   a different, weaker result than one that does both.

Then the surviving branches are assembled into the **full tree** and fit jointly,
where the third question lives:
3. **Leak / redundancy** — which branch goes inert once the others are present
   (its signal leaked into another node), and can it be trimmed without hurting
   RUL. This is the fan-branch lesson made systematic.

**Core requirements this imposes** (the two functions called out in the request):
- `fit_branch(component, single_fault_frame, sensors, ...)` — build and fit a
  single-component branch; report leaf θ fit AND branch RUL.
- `fit_full(leaf_config, pooled_frame, ...)` — build and fit the aggregated tree;
  expose every node's activation so the leak/trim analysis can see which branch
  died.
Both are thin wrappers over ONE general `fit(tree_spec, frame, ...)`; a branch, a
bare leaf, and the full tree are all just tree specs, and the composite loss
(a θ term per supervised leaf + a RUL term when a root is present) covers all
three without special cases. The full-tree aggregation topology
(component damage → spool → RUL, vs. flat) is a builder option, itself an ablation
knob, defaulting to the shaft-grouped physical form.

## 6. Build order once §5 is confirmed

1. `gft.py` — FIS, tree spec, `SUBTREES` dict, `build_tree`, `predict_tree`,
   `fit`/`predict`/`evaluate`. Self-test: monotonicity + anchoring invariants on
   random genomes, per (sub)tree.
2. `ga.py` — extract and self-test the optimizer standalone.
3. `ablation.py` — one entry point: leaf selection → per-arm repeated CV
   (in-distribution) → LOSO (leave-one-mode-out) → age baselines, NASA-first,
   writing one tidy results table per arm.
4. `viz.py` — ablation-focused figures: per-arm RMSE/NASA distributions,
   spool-damage curves, RUL parity, and the leaf-diagnostic panels — driven off
   the ablation's saved models, not a separate fit.

## 7. Test evidence so far

- `features.py`: synthetic 3-component file → 6 modifier columns present, age ≥ 1,
  onset fires, physics metadata (`components_on`, `theta_of`) correct.
- `data.py`: synthetic 3-file fleet (HPT / fan / HPC+HPT+LPT) → pooling fills all
  ten modifiers, unit ids remapped unique, coverage table correct, by-unit split
  disjoint, LOSO yields one fold per file.
