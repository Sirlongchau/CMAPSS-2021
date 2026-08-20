# 06 — viz.py (ablation figures)

Purpose: turn the ablation result dict into figures. Every plot is driven off
`ablation.run_ablation`'s output plus the pooled frame, so a figure always matches
the run that produced it — there is no separate fit, and nothing here can silently
disagree with the tables.

## Panels

- **leaf_search.png** — one panel per leaf, a bar per candidate sensor set
  (Stage-1 `theta_rho`), coloured by cross-mode leak (green = specific, red =
  leaky), winner dashed. This is the "which sensors read this component" evidence
  at a glance.
- **branch_rul.png** — per single-fault mode, branch-alone RUL RMSE and NASA:
  does one fault mode's branch predict its own life.
- **leak.png** — per mode in the assembled tree, own-spool vs other-spool peak
  damage, with a `LEAK` marker where the wrong spool fired. This is the picture
  that motivates each trim decision.
- **assembled.png** — per single-fault mode, spool-damage trajectories (own spool
  solid, others dashed) and RUL parity. Correct attribution shows the own spool
  climbing while the other stays flat; a flat own spool (as the fan shows) is the
  inert-branch signature, visible next to the modes that work.

## Interface

`visualize(out, pooled, mode_components, outdir="figures") -> [paths]` renders all
four and returns the file list. Individual `plot_*` functions are available if a
single figure is wanted.

## Design notes

Matplotlib only (Agg backend, no interactive dependency), so it runs headless in a
pipeline. Deliberately scoped to the ablation — the old general-purpose surface /
membership / per-unit galleries are gone; if a specific diagnostic is needed later
it is a small addition here, not a second module.

## Test evidence

`python viz.py` runs a short ablation on the synthetic fleet and renders all four
PNGs without error. Visual check: `leaf_search` shows per-set θ-rho with the leak
colouring; `assembled` shows own-spool attribution for HPT/HPC/LPT and the flat
(inert) fan spool beside them.
