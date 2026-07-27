"""
confirm.py -- confirmatory evaluation of the DEPLOYABLE interpretable model.

Replaces run.py's single 80/10/10 split (one seed, one 11-unit test set, one
pathological unit owning ~27% of the error) with a repeated by-unit CV that
yields a DISTRIBUTION of held-out performance for a PRE-REGISTERED model:

    age-free (TREE_LP1)  |  learn_mf=False  |  monotone=True

Nothing here is selected on val. The architecture was fixed upstream by the leaf
ablation (leaf_search.py) and the decision to drop MF learning. This script
ESTIMATES the chosen model's performance; it does not choose it. `full` (TREE) is
reported alongside only to show the choice was not cherry-picked.

    python datasets.py     # once: cache
    python confirm.py
"""
import numpy as np
import pandas as pd

import gft
import datasets as ds

K_SPLITS = 12                 # random by-unit splits -- now the dominant variance source
GA_SEEDS = (0,)               # fixed-MF has ~0 GA-seed variance (sd ~0.02); 1 seed suffices
GENS, POP = 400, 120

# fixed-MF: learn_mf dropped per the decision; monotone kept for the damage-signal claim.
FIT_BASE = dict(gens=GENS, pop=POP, theta_weight=1.0, trend_weight=1.0,
                smooth_span=7, learn_mf=False, monotone=True)

DEPLOY  = ("LP-specific", gft.TREE_LP1)   # every leaf diagnoses its own spool -> FADEC-ready
COMPARE = ("full",        gft.TREE)       # reported, not selected

feats = ds.pooled()


def per_unit_rmse(frame, model, cap):
    """Per-unit held-out RUL RMSE. Aggregated across splits later, this finds
    pathological units ROBUSTLY -- over every split a unit lands in, not one."""
    pr = gft.predict_gft(frame, model)
    f = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    d = pr["RUL_hat"].to_numpy(float) - gft.cap(f["RUL"], cap).to_numpy(float)
    return (pd.DataFrame({"unit": f["unit"].to_numpy(), "sq": d ** 2})
            .groupby("unit")["sq"].mean().pow(0.5))


split_rows, unit_rows = [], []
for k in range(K_SPLITS):
    train, val, test = ds.split(feats, fracs=(0.8, 0.1, 0.1), seed=k)
    fitset = pd.concat([train, val], ignore_index=True)     # no val selection -> fold it in
    cap = gft.suggest_rul_cap(fitset)                       # fit-set only: no test leakage
    fit = dict(FIT_BASE, rul_cap=cap)

    base = gft.baselines(fitset, test, cap)                 # age-only reference for THIS test set
    age_rmse = float(base.loc[base["model"].str.startswith("age only"), "RMSE"].iloc[0])

    for label, tree in (DEPLOY, COMPARE):
        for s in GA_SEEDS:
            m = gft.fit_gft(fitset, tree, seed=s, verbose=False, **fit)
            e = gft.evaluate(test, m, label)
            e.update(split=k, seed=s, age_RMSE=age_rmse, beats_age=e["RMSE"] < age_rmse)
            split_rows.append(e)
            if label == DEPLOY[0] and s == GA_SEEDS[0]:
                for u, r in per_unit_rmse(test, m, cap).items():
                    unit_rows.append(dict(split=k, unit=int(u), rmse=float(r)))
    tail = split_rows[-len(GA_SEEDS) * 2:]
    print(f"  split {k:2d}: age={age_rmse:5.2f}  "
          + "  ".join(f"{r['model']}={r['RMSE']:.2f}" for r in tail))

S = pd.DataFrame(split_rows)
U = pd.DataFrame(unit_rows)

print("\n== held-out RUL over splits (mean +/- sd) ==")
print(S.groupby("model")[["RMSE", "NASA", "R2"]].agg(["mean", "std"]).round(3))

print("\nfraction of splits that beat the age baseline:")
print(S.groupby("model")["beats_age"].mean().round(2).to_string())

print("\n== worst units, aggregated over the splits they appear in ==")
print(U.groupby("unit")["rmse"].agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False).head(10).round(2).to_string())

print("\n== leaf theta quality for the deployable model (the FADEC payload) ==")
tcols = [c for c in S.columns if c.endswith(":R2") or c.endswith(":rho")]
print(S[S["model"] == DEPLOY[0]][tcols].agg(["mean", "std"]).round(3).to_string())

S.to_csv("confirm_splits.csv", index=False)
U.to_csv("confirm_units.csv", index=False)
print("\nwrote confirm_splits.csv, confirm_units.csv")