# 04 — gft.py (fuzzy tree core)

Purpose: the tree engine, rebuilt small, with the two fit entry points the
curriculum needs.

## Architecture (fixed)

```
HPT, HPC       -> hp        (HP shaft)
fan, LPC, LPT  -> lp        (LP shaft)
hp, lp         -> RUL
```

Spool grouping is read from `features.SHAFT`, so it is the physics, declared once.

## Node kinds and their constraints

- **leaf** (sensors → one θ modifier): FREE grid — θ need not be monotone in a raw
  sensor. Antecedent centres are quantiles of the residualized, EWMA-smoothed
  sensor.
- **spool** (leaf θ → damage): MONOTONE decreasing (more negative modifier ⇒ more
  damage) and ANCHORED to span exactly `[0,1]`, removing the gauge freedom in a
  latent node's scale. Its antecedent centres sit on the θ's degrading range
  (`since_onset > 0`) so the ramp grades instead of flooring.
- **root** (spool damage → RUL): MONOTONE decreasing and ANCHORED to `[0, rul_cap]`,
  so `RUL(no damage)=cap` and `RUL(full damage)=0` hold BY CONSTRUCTION (the
  0-corner result carried over from the previous project, now native).

## The two fit entry points (the contract)

- **`fit_branch(active, frame)`** builds a tree with ONLY the active components'
  leaves → the spool(s) they sit on → RUL; dead branches are never created.
  `active` may be one component (single-fault identification: HPT/DS01, LPT/DS07,
  HPC/DS05, fan/DS04) or several. **DS06 = `["HPC","LPC"]`** builds HPC→hp AND
  LPC→lp — two active spools under one root, i.e. decision **1(a)**: both leaves
  trained jointly, and DS05's HPC-only fit is the reference the ablation compares
  DS06's HPC leaf against. Returns the model plus, in `report`, the two answers
  each single-fault fit owes: per-leaf θ quality (`<mod>:R2`, `<mod>:rho`) and the
  branch's standalone RUL (`RMSE`, `NASA`, `rho_RUL`).
- **`fit_full(frame, branch_models=...)`** builds the assembled tree and, if given
  the identified branch models, splices their genes into a full-tree seed and
  warm-starts the GA from it. Per the agreed contract the seed is **free** — the
  GA may move any gene; the splice only saves search, standing in for the heavy
  cold fit the old `confirm.py` paid for. `splice_seed` copies each branch node's
  genes into the full genome's matching node slice (by name, shape-checked);
  unmatched nodes keep their random init.

## Simplifications vs the old core

Fixed MFs only — the genome is consequents, nothing else. Deleted with no
successor: learnable-MF decode/encode, gap genes, `MF_PAD`, membership plotting,
leave-one-unit-out, and the unused official-split helper. Result: one `_fis`, one
`_mono_decode`, one `build_tree`.

## Loss

Per-leaf θ RMSE normalized by that θ's spread (keeps every leaf honest to its own
component) plus the RUL RMSE normalized by RUL spread. Weights `W_THETA`, `W_RUL`.

## Test evidence

`python gft.py`:
- **invariants** on 200 random genomes: partition-of-unity exact, every spool/root
  grid monotone, and the root grid's corners land exactly on `0` and `cap`;
- **single-branch HPT fit**: `HPT_eff` tracks its θ at ρ = 0.97 (R² 0.94) and the
  branch alone predicts RUL at RMSE 2.13 / NASA 0.12 — both curriculum questions
  answered by one call;
- **warm-started full fit**: HPT + LPT branch genomes splice into the full tree,
  the GA runs free, RUL stays within `[0, cap]`;
- **DS06 joint branch** (separately checked): `["HPC","LPC"]` yields hp and lp
  spools under one root, both leaves tracking (ρ 0.99 / 0.98).
