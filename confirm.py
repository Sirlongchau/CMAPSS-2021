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
    d = pr["RUL_hat"].to_numpy(float) - np.asarray(gft.cap(f["RUL"], cap), float)
    return (pd.DataFrame({"unit": f["unit"].to_numpy(), "sq": d ** 2})
            .groupby("unit")["sq"].mean().pow(0.5))


def per_unit_age_rmse(fitset, frame, cap):
    """Per-unit RMSE of the one-parameter age baseline (RUL = c - age), with c fit
    on the SAME fitset baselines() uses, so a per-mode age column matches the
    aggregate age_RMSE. This is what makes the fan branch's payoff legible: DS04
    (fan) lost to age before it had a leaf; per mode we can see it flip to a win."""
    y = np.asarray(gft.cap(fitset["RUL"], cap), float)
    a = fitset["age"].to_numpy(float)
    ok = (y < float(cap) - 1e-9) if cap else np.ones(len(y), bool)
    c = float(np.mean(y[ok] + a[ok])) if ok.sum() > 1 else float(np.mean(y + a))
    f = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    age_hat = np.clip(c - f["age"].to_numpy(float), 0.0, cap)
    d = age_hat - np.asarray(gft.cap(f["RUL"], cap), float)
    return (pd.DataFrame({"unit": f["unit"].to_numpy(), "sq": d ** 2})
            .groupby("unit")["sq"].mean().pow(0.5))


# The tree has leaves for HPT/LPT only, so fan-/LPC-/HPC-only units (DS04/05/06)
# carry no damage it can read: it must NOT be fit on them (they teach the root that
# the zero-damage corner can mean end-of-life) and the headline must NOT be scored
# on them (that measures a blind guess). The design here:
#   - split ONLY the diagnosable units into train/val/test (the normal CV);
#   - hold EVERY blind unit (DS04 fan, DS05 HPC, DS06 LPC+HPC) out of training and
#     evaluate ALL of them on every split -- the LP-LEAK experiment. With no leaf for
#     those components, any RUL skill on them comes purely from their degradation
#     LEAKING into the HPT/LPT sensors. If that still tracks RUL, it is a real, if
#     bounded, generalisation result; if not, it is the honest limit of the model.
# Using every blind unit each split (not the ~10% that would land in a pooled test
# fold) gives full, low-variance statistics on that leak.
feats_diag, feats_blind = ds.split_diagnosable(feats)
print(f"diagnosable units: {feats_diag['unit'].nunique()}  |  "
      f"blind (held out of training, kept in test): {feats_blind['unit'].nunique()}  "
      f"[{', '.join(sorted(feats_blind['ds'].unique()))}]")

split_rows, unit_rows = [], []
for k in range(K_SPLITS):
    train, val, test_d = ds.split(feats_diag, fracs=(0.8, 0.1, 0.1), seed=k)
    fitset = pd.concat([train, val], ignore_index=True)     # diagnosable only, by construction
    cap = gft.suggest_rul_cap(fitset)                       # fit-set only: no test leakage
    fit = dict(FIT_BASE, rul_cap=cap)

    scopes = [("diagnosable", test_d), ("blind", feats_blind)]  # ALL blind units, every split

    for label, tree in (DEPLOY, COMPARE):
        for s in GA_SEEDS:
            m = gft.fit_gft(fitset, tree, seed=s, verbose=False, **fit)
            for scope, tset in scopes:
                if tset.empty:
                    continue
                base = gft.baselines(fitset, tset, cap)     # age reference for THIS subset
                age_rmse = float(base.loc[base["model"].str.startswith("age only"),
                                          "RMSE"].iloc[0])
                e = gft.evaluate(tset, m, label)
                e.update(split=k, seed=s, scope=scope, age_RMSE=age_rmse,
                         beats_age=e["RMSE"] < age_rmse)
                split_rows.append(e)
            if label == DEPLOY[0] and s == GA_SEEDS[0]:
                evalset = pd.concat([test_d, feats_blind], ignore_index=True)
                pr = per_unit_rmse(evalset, m, cap)
                ar = per_unit_age_rmse(fitset, evalset, cap)
                leak = gft.flag_undiagnosable(gft.predict_gft(evalset, m))  # peak hp/lp per unit
                leak = leak.set_index("unit")["damage_peak"] if not leak.empty else pd.Series(dtype=float)
                info = evalset.groupby("unit")[["diagnosable", "ds", "failure_mode"]].first()
                for u, r in pr.items():
                    unit_rows.append(dict(
                        split=k, unit=int(u), rmse=float(r),
                        age_rmse=float(ar.get(u, np.nan)),
                        leak_peak=float(leak.get(u, np.nan)),   # how hard hp/lp lit up
                        diagnosable=bool(info.loc[u, "diagnosable"]),
                        ds=str(info.loc[u, "ds"]),
                        failure_mode=str(info.loc[u, "failure_mode"])))
    dep = [r for r in split_rows[-len(GA_SEEDS) * 2 * 2:]
           if r["model"] == DEPLOY[0]]
    print(f"  split {k:2d}: "
          + "  ".join(f"{r['scope'][:4]} {r['model']}={r['RMSE']:.2f}"
                      f"(age {r['age_RMSE']:.2f})" for r in dep))

S = pd.DataFrame(split_rows)
U = pd.DataFrame(unit_rows)

print("\n== held-out RUL over splits, by scope (mean +/- sd) ==")
print(S.groupby(["scope", "model"])[["RMSE", "NASA", "R2"]]
        .agg(["mean", "std"]).round(3))

print("\nfraction of splits that beat the age baseline, by scope:")
print(S.groupby(["scope", "model"])["beats_age"].mean().round(2).to_string())
print("\n(BLIND = fan/HPC/LPC units with no leaf, held OUT of training. Any skill "
      "here is the fault LEAKING into the HPT/LPT sensors -- a generalisation probe, "
      "not the headline. beats_age there = the leak carries real RUL information.)")

print("\n== worst DIAGNOSABLE units, aggregated over the splits they appear in ==")
Ud = U[U["diagnosable"]]
print(Ud.groupby("unit")["rmse"].agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False).head(10).round(2).to_string())

print("\n== per failure mode: deployable vs age, with the LP-leak measured ==")
M = (U.groupby(["ds", "failure_mode", "diagnosable"])
       .agg(model_RMSE=("rmse", "mean"), age_RMSE=("age_rmse", "mean"),
            leak_peak=("leak_peak", "mean"), units=("unit", "nunique"))
       .reset_index()
       .sort_values(["diagnosable", "ds"], ascending=[False, True]))
M["beats_age"] = M["model_RMSE"] < M["age_RMSE"]
print(M.round(2).to_string(index=False))
print("  diagnosable=False rows are the BLIND modes, held out of training. leak_peak "
      "= mean peak hp/lp damage the model read on them: >0 means the un-modelled "
      "fault LEAKED into the HPT/LPT sensors. Where such a row still has "
      "model_RMSE < age_RMSE, the leak alone carried usable RUL -- the generalisation "
      "result. DS04 is the fan mode: compare its model_RMSE to the ~15 it scored when "
      "blind under LOSO, and to age here.")

print("\n== leaf theta quality for the deployable model (the FADEC payload) ==")
tcols = [c for c in S.columns if c.endswith(":R2") or c.endswith(":rho")]
Sd = S[(S["model"] == DEPLOY[0]) & (S["scope"] == "diagnosable")]
print(Sd[tcols].agg(["mean", "std"]).round(3).to_string())

S.to_csv("confirm_splits.csv", index=False)
U.to_csv("confirm_units.csv", index=False)
print("\nwrote confirm_splits.csv, confirm_units.csv")