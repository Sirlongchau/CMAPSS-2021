"""
run_full.py -- compare aggregation strategies for the full tree in one run.

Four arms, all on the SAME identified leaves and the SAME DS08 train set, so the
only thing that varies is how leaf theta is aggregated into RUL:

  shaft-coupled     baseline: grouping=shaft, joint theta+RUL fit (the old behaviour)
  station-coupled   Route 3 : grouping=station (cold/hot) -- the leak experiment
  shaft-decoupled   Route 2 : frozen honest leaves + separately-trained aggregator
  station-decoupled Route 2 + Route 3

Every arm is scored on BOTH axes -- RUL (RMSE/NASA) and min leaf theta-R2 -- at
several seeds (verdicts flip run-to-run at this size), and the shaft-vs-station
leak tables are printed to test whether cold/hot grouping reduces the cross-mode
leak. Per-arm assembled figures and the comparison table are written to `outdir`.

The θ floor stays active in the COUPLED arms (so a leaf that can't stay honest is
penalised); the DECOUPLED arms don't need it -- their leaves are frozen honest by
construction, which is the whole point.

RUNTIME: this is (#arms x #seeds) full-tree fits. At gens=300, pop=120, 4 arms,
3 seeds that is 12 heavy fits -- start with seeds=(0,) or a subset of `arms` for a
quick look, then scale up. The cost levers are `seeds` and `pop`, not `gens`.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

import data
import ablation
import gft
import viz


# ==========================================================================
# arm definitions -- name -> a function seed -> fitted model
# ==========================================================================

def _arms(tr, leaves, gens, pop):
    return {
        "shaft-coupled":
            lambda s: gft.fit_full(tr, leaves=leaves, grouping="shaft",
                                   gens=gens, pop=pop, seed=s),
        "station-coupled":
            lambda s: gft.fit_full(tr, leaves=leaves, grouping="station",
                                   gens=gens, pop=pop, seed=s),
        "shaft-decoupled":
            lambda s: gft.fit_decoupled(tr, leaves=leaves, grouping="shaft",
                                        gens=gens, pop=pop, seed=s),
        "station-decoupled":
            lambda s: gft.fit_decoupled(tr, leaves=leaves, grouping="station",
                                        gens=gens, pop=pop, seed=s),
    }


def _score(model, frame):
    r = gft.evaluate(frame, model)
    lr2 = [v for k, v in r.items() if k.endswith(":R2")]
    return {"RMSE": r["RMSE"], "NASA": r["NASA"], "R2": r["R2"],
            "rho_RUL": r["rho_RUL"],
            "min_leaf_R2": float(np.min(lr2)) if lr2 else np.nan,
            "mean_leaf_R2": float(np.mean(lr2)) if lr2 else np.nan}


# ==========================================================================
# the comparison (testable core)
# ==========================================================================

def compare(pooled, leaves, train_ds, arms=None, seeds=(0, 1, 2),
            gens=300, pop=120, outdir="figures", verbose=True):
    """Run the arms, score both axes at each seed, print/return the comparison
    table, the shaft-vs-station leak tables, and the last model per arm."""
    os.makedirs(outdir, exist_ok=True)
    tr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    cov = data.coverage(pooled)
    modes_all = {r.ds: r.components.split("+") for r in cov.itertuples()}
    modes_single = {ds: c for ds, c in modes_all.items() if len(c) == 1}

    all_arms = _arms(tr, leaves, gens, pop)
    if arms:
        all_arms = {k: v for k, v in all_arms.items() if k in arms}

    rows, models = [], {}
    for name, fit in all_arms.items():
        if verbose:
            print(f"\n=== {name}  ({len(seeds)} seed(s), gens={gens}, pop={pop}) ===")
        per_seed, last = [], None
        for s in seeds:
            last = fit(s)
            sc = _score(last, tr)
            per_seed.append(sc)
            if verbose:
                print(f"  seed {s}: RMSE={sc['RMSE']:.2f}  NASA={sc['NASA']:.2f}  "
                      f"min_leaf_R2={sc['min_leaf_R2']:.2f}")
        d = pd.DataFrame(per_seed)
        rows.append({"arm": name,
                     "RMSE": d.RMSE.mean(), "RMSE_sd": d.RMSE.std(ddof=0),
                     "NASA": d.NASA.mean(), "R2": d.R2.mean(),
                     "rho_RUL": d.rho_RUL.mean(),
                     "min_leaf_R2": d.min_leaf_R2.mean(),
                     "min_leaf_R2_sd": d.min_leaf_R2.std(ddof=0),
                     "mean_leaf_R2": d.mean_leaf_R2.mean()})
        models[name] = last

    table = pd.DataFrame(rows)
    print("\n=========== ARM COMPARISON  (RUL vs leaf fidelity) ===========")
    print(table.round(3).to_string(index=False))
    table.round(4).to_csv(os.path.join(outdir, "arm_comparison.csv"), index=False)

    # leak hypothesis: shaft vs station on the coupled arms
    leaks = {}
    for g in ("shaft-coupled", "station-coupled"):
        if g in models:
            lk = ablation.leak_table(models[g], pooled, modes_single)
            leaks[g] = lk
            print(f"\n== leak table: {g} ==")
            print(lk.round(3).to_string(index=False))
            viz.plot_leak(lk, os.path.join(outdir, f"leak_{g}.png"))

    for name, m in models.items():
        viz.plot_assembled(m, pooled, modes_all,
                           os.path.join(outdir, f"assembled_{name}.png"))
    print(f"\nwrote {outdir}/arm_comparison.csv and per-arm figures")
    return table, models, leaks


# ==========================================================================
# real-data entry point
# ==========================================================================

if __name__ == "__main__":
    pooled = data.pooled()
    leaves = ablation.load_leaves("ablation_out/best_leaves.json")
    train = [d for d in pooled["ds"].unique() if d.startswith("DS08")]

    # start with seeds=(0,) for a quick look; use (0,1,2) for the real comparison.
    compare(pooled, leaves, train, seeds=(0, 1, 2), gens=300, pop=120,
            outdir="figures")