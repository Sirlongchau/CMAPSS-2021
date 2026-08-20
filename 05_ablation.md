# 05 — ablation.py (the staged run)

Purpose: run the whole curriculum in one command and emit the tables that decide
which sensors, which leaves, and which branches survive. Replaces `run.py`,
`confirm.py`, `leaf_search.py`, and `add_age_NASA.py`.

## Stage 1 — identify (per-leaf sensor selection on single-fault files)

For each single-fault mode, `leaf_search` fits each leaf **in isolation**
(sensors → its θ, one free FIS — cheap, so the sensor search is fast) over the
candidate sets in `LEAVES`, and reports `theta_R2`, `theta_rho`, and a
cross-mode **leak** score: the identified leaf is applied to a foil mode (a
different-shaft single-fault file) and its spurious output magnitude is measured —
low leak = specific, high leak = reads global damage. `_best_leaves` then picks
the winning sensor set per leaf **globally**, so a component seen in two modes
(HPC in DS05 and DS06) yields one leaf, and records a cross-mode **agreement**
row — the DS05-vs-DS06 HPC check (do the two modes agree on sensors and rho?).
Finally one `fit_branch` per mode gives the branch-alone RUL: can this fault
mode's branch predict its own RUL (`branch_RMSE/NASA/rho_RUL`)?

Curriculum (derived from the caller's `mode_components`, matching the plan):
HPT/DS01, LPT/DS07, HPC/DS05, fan/DS04, and **HPC+LPC/DS06 jointly** (1a) — the
DS06 fit activates HPC→hp and LPC→lp under one root.

## Stage 2 — assemble (warm-started full fit)

`assemble` fits the full hp/lp/RUL tree on the all-modes files (DS08a/c), passing
the Stage-1 branch models to `gft.fit_full` so the GA population is **seeded** from
the identified branches and then runs **free** — the compute-saver, not a
constraint.

## Stage 3 — autopsy (leak table + trim test)

`leak_table` reports, per single-fault mode in the ASSEMBLED tree, the peak damage
each spool reaches: the own spool should climb; another spool climbing (or the own
spool staying flat) is flagged. `trim_test` then refits the full tree WITH and
WITHOUT a flagged branch and compares RUL on that branch's own mode:
`RMSE_without ≤ RMSE_with` ⇒ **redundant (trim)**; a real cost ⇒ **starved/leaked
(keep, and report as a finding)** — the discipline that a branch is only trimmed
on evidence it adds nothing, never on "looks dead".

## Entry point

`run_ablation(pooled, mode_components, train_ds, ...)` runs all three stages,
prints each table, and returns `{best_leaves, search, branch_report, agreement,
assembled, leaks, trims, branch_models}` for `viz.py`.

## Test evidence

`python ablation.py` on a synthetic six-mode fleet:
- Stage 1 searches every leaf, dedups HPC across DS05/DS06 (one leaf,
  `sensor_agreement=True`, `rho_spread≈0.002`), and every single-fault branch
  predicts its own RUL (rho_RUL ≥ 0.9);
- Stage 2 assembles and fits the warm-started full tree;
- Stage 3 flags the **fan** branch as inert (`lp_peak≈0.03`) — reproducing the
  real fan finding — and runs the with/without trim test on it.

Note: on the tiny synthetic fleet at low generations the trim *verdict* flips
run-to-run (4 units/mode is underpowered); the detection and machinery are stable,
and a real run at full budget gives a stable verdict.
