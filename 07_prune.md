# 07 — prune.py (greedy backward leaf elimination)

Purpose: find the parsimonious HONEST tree, and the redundancy ordering, from the
finding that extra leaves are redundant lifetime signals.

## Why this design (and not a 2^n sweep)

Ten leaves = 1024 subsets × a full-tree GA fit each — days of compute for a ranking
swamped by the seed noise seen at 1089 params. Greedy backward is O(n²) ≈ 55 fits:
from the full set, drop the leaf whose removal HELPS or LEAST HURTS RUL, repeat.
Redundant leaves are exactly the ones whose removal costs nothing, so the drop
order walks down the redundancy gradient — the trajectory is the result, not just
the endpoint.

## Two disciplines the finding forces

- **Multi-seed.** Verdicts flip run-to-run at this size; every candidate is fit at
  `seeds` and compared on mean ± spread.
- **Score both axes.** RUL (RMSE/NASA) AND `min_leaf_R2`. The soft-floor run showed
  leaves hiding at R²≈0 with no diagnostic value, so a subset only counts as good
  if its surviving leaves are genuinely readable. The θ floor (`gft.W_THETA_FLOOR`)
  stays ACTIVE during every fit, so a leaf that cannot stay honest is penalized and
  surfaces as a low `min_leaf_R2`.

## Interface

- `greedy_prune(pooled, leaves, train_ds, eval_ds=None, seeds=(0,1,2), gens, pop,
  min_leaves=1)` → `{trajectory, all_points, pareto, recommended}`. Scoring is on
  `eval_ds` (defaults to `train_ds`, in-sample like the rest of the pipeline; pass
  a held-out list for an out-of-sample prune).
- `_recommend`: among greedy-kept sets whose leaves are all honest
  (`min_leaf_R2 ≥ r2_floor`, default 0.3), the lowest-RMSE one; if none clear the
  bar, the most-honest achievable, flagged (the conflict is unresolved).
- `save_prune(out, outdir)` → `trajectory.csv`, `pareto.csv`, `all_points.csv`, and
  `recommended_leaves.json` (best_leaves.json-compatible → feeds `gft` directly).
- `viz.plot_pruning(out, save)` → trajectory (RMSE + min-leaf-R² vs leaf count) and
  the (RMSE, min-leaf-R²) Pareto scatter.

## Test evidence

`python prune.py` on the synthetic all-modes fleet reproduces the phenomenon: the
6-leaf set has a collapsed leaf (min_leaf_R2 −0.25) and mediocre RUL; dropping
fan → LPC → HPC (the redundancy order) improves BOTH RUL and fidelity to a knee at
3 leaves (RMSE 1.36, min_leaf_R2 0.80), after which dropping further hurts RUL. The
recommendation lands on that knee; the Pareto front holds the 3- and 2-leaf sets.

## Caveat carried from the experiments

Score pruning candidates under the HARD floor (the shipped `gft` weights), never a
soft one — the soft-floor run proved a gentle penalty lets leaves hide at R²≈0, so
a soft-floor prune would credit diagnostically-empty leaves as fine.
