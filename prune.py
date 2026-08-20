"""
prune.py -- greedy backward LEAF elimination for the full tree.

Three runs established the finding this module operationalizes: adding leaves past
a small set trades one half of the contract for the other -- free leaves give good
RUL but fake diagnostics (negative theta R2); floored leaves give honest
diagnostics but broken RUL (R2 < 0, and a softer floor made it worse). The extra
components are REDUNDANT lifetime signals, not independent ones. This module finds
the parsimonious honest tree and, as a by-product, the redundancy ORDERING.

Why greedy backward and not a 2^n sweep: 10 leaves is 1024 subsets x a full-tree
GA fit each -- days of compute for a ranking swamped by seed noise. Greedy backward
is O(n^2) ~ 55 fits: from the full set, drop the single leaf whose removal HELPS or
LEAST HURTS RUL, repeat. Redundant leaves are exactly the ones whose removal costs
nothing, so the drop order walks down the redundancy gradient -- the trajectory is
the result, not just the endpoint.

Two disciplines the finding forces:
  * MULTI-SEED. Trim verdicts flip run-to-run at 1089 params, so every candidate is
    fit at several seeds and compared on mean +/- spread.
  * SCORE BOTH AXES. RUL (RMSE/NASA) AND min leaf theta-R2. A subset only counts as
    good if its surviving leaves are actually readable, not merely non-negative --
    the soft-floor run showed leaves hiding at R2 ~ 0 with no diagnostic value. The
    theta FLOOR (gft.W_THETA_FLOOR) stays ACTIVE during every fit so a leaf that
    cannot stay honest shows up as costly.

Output: elimination trajectory, the (RUL, leaf-R2) Pareto front, and the
recommended leaf set as JSON (best_leaves.json-compatible, feeds gft directly).
"""

from __future__ import annotations
import copy
import json
import os
import numpy as np
import pandas as pd

import gft


# ==========================================================================
# leaf-set bookkeeping (a leaf = (component, leaf_name); config = gft leaves dict)
# ==========================================================================

def _leaf_list(leaves):
    return [(comp, name, target, sensors)
            for comp, entries in leaves.items()
            for (name, target, sensors) in entries]


def _n_leaves(leaves):
    return sum(len(v) for v in leaves.values())


def _remove(leaves, comp, name):
    """Return a copy of `leaves` with leaf (comp, name) removed; drop the component
    key if it has no leaves left (its spool then loses that input)."""
    out = copy.deepcopy(leaves)
    out[comp] = [e for e in out[comp] if e[0] != name]
    if not out[comp]:
        del out[comp]
    return out


# ==========================================================================
# fit + score a leaf set across seeds
# ==========================================================================

def _fit_eval(pooled, leaves, train_ds, eval_ds, seeds, gens, pop):
    """Fit the full tree from `leaves` at each seed, score on eval_ds. Returns mean
    (+ spread) of RUL metrics and the MIN surviving-leaf theta-R2 -- the fidelity
    axis. The floor is active inside gft.fit_full, so a leaf that can't stay honest
    is penalized during the fit and shows up here as a low min_leaf_R2."""
    fr_tr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    fr_ev = pooled[pooled["ds"].isin(eval_ds)].reset_index(drop=True)
    rows = []
    for s in seeds:
        m = gft.fit_full(fr_tr, leaves=leaves, gens=gens, pop=pop, seed=s)
        rep = gft.evaluate(fr_ev, m)
        lr2 = [v for k, v in rep.items() if k.endswith(":R2")]
        rows.append({"RMSE": rep["RMSE"], "NASA": rep["NASA"], "R2": rep["R2"],
                     "rho_RUL": rep["rho_RUL"],
                     "min_leaf_R2": float(np.min(lr2)) if lr2 else np.nan,
                     "mean_leaf_R2": float(np.mean(lr2)) if lr2 else np.nan})
    d = pd.DataFrame(rows)
    return {"RMSE": d.RMSE.mean(), "RMSE_sd": d.RMSE.std(ddof=0),
            "NASA": d.NASA.mean(), "R2": d.R2.mean(), "rho_RUL": d.rho_RUL.mean(),
            "min_leaf_R2": d.min_leaf_R2.mean(),
            "min_leaf_R2_sd": d.min_leaf_R2.std(ddof=0),
            "mean_leaf_R2": d.mean_leaf_R2.mean()}


# ==========================================================================
# greedy backward elimination
# ==========================================================================

def greedy_prune(pooled, leaves, train_ds, eval_ds=None, seeds=(0, 1, 2),
                 gens=60, pop=100, min_leaves=1, verbose=True):
    """From the full leaf set, repeatedly drop the leaf whose removal gives the
    LOWEST resulting RUL RMSE (helps or least hurts), down to `min_leaves`.

    Returns dict with:
      trajectory : the kept set at each size (size, dropped, RUL + fidelity metrics)
      all_points : every subset evaluated (for the Pareto front)
      pareto     : non-dominated subsets on (RMSE down, min_leaf_R2 up)
      recommended: the chosen leaf set (dict) + why it was chosen
    Scoring is on eval_ds (defaults to train_ds -- IN-SAMPLE, matching the rest of
    the pipeline; pass a held-out ds list for an out-of-sample prune)."""
    eval_ds = eval_ds or train_ds
    current = copy.deepcopy(leaves)
    traj, points = [], []

    base = _fit_eval(pooled, current, train_ds, eval_ds, seeds, gens, pop)
    traj.append({"size": _n_leaves(current), "dropped": "(full set)",
                 "_leaves": copy.deepcopy(current), **base})
    if verbose:
        print(f"[{_n_leaves(current):2d} leaves] full set: "
              f"RMSE={base['RMSE']:.2f}+/-{base['RMSE_sd']:.2f}  "
              f"min_leaf_R2={base['min_leaf_R2']:.2f}")

    while _n_leaves(current) > min_leaves:
        trials = []
        for comp, name, _, _ in _leaf_list(current):
            trial = _remove(current, comp, name)
            if _n_leaves(trial) == 0:
                continue
            met = _fit_eval(pooled, trial, train_ds, eval_ds, seeds, gens, pop)
            trials.append((comp, name, trial, met))
            points.append({"size": _n_leaves(trial), "dropped": f"{comp}.{name}",
                           "_leaves": trial, **met})
        if not trials:
            break
        comp, name, current, met = min(trials, key=lambda t: t[3]["RMSE"])  # least hurt
        traj.append({"size": _n_leaves(current), "dropped": f"{comp}.{name}",
                     "_leaves": copy.deepcopy(current), **met})
        if verbose:
            print(f"[{_n_leaves(current):2d} leaves] drop {comp}.{name:9s}: "
                  f"RMSE={met['RMSE']:.2f}+/-{met['RMSE_sd']:.2f}  "
                  f"min_leaf_R2={met['min_leaf_R2']:.2f}")

    all_points = traj + points
    pareto = _pareto_front(all_points)
    rec = _recommend(traj)
    return {"trajectory": _tidy(traj), "all_points": _tidy(all_points),
            "pareto": _tidy(pareto), "recommended": rec}


def _pareto_front(points):
    """Non-dominated on (RMSE lower better, min_leaf_R2 higher better)."""
    out = []
    for p in points:
        if any((q["RMSE"] <= p["RMSE"] and q["min_leaf_R2"] >= p["min_leaf_R2"]
                and (q["RMSE"] < p["RMSE"] or q["min_leaf_R2"] > p["min_leaf_R2"]))
               for q in points if q is not p):
            continue
        out.append(p)
    # dedup by (size, dropped)
    seen, uniq = set(), []
    for p in sorted(out, key=lambda x: x["RMSE"]):
        key = (p["size"], p["dropped"])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _recommend(traj, r2_floor=0.3):
    """The parsimonious HONEST tree: among the greedy-kept sets whose surviving
    leaves are genuinely readable (min_leaf_R2 >= r2_floor), take the one with the
    lowest RUL RMSE. If none clear the fidelity bar, fall back to the set with the
    highest min_leaf_R2 (best achievable honesty) and flag it."""
    honest = [t for t in traj if t["min_leaf_R2"] >= r2_floor]
    if honest:
        pick = min(honest, key=lambda t: t["RMSE"])
        why = (f"lowest RUL RMSE ({pick['RMSE']:.2f}) among sets with all leaves "
               f"honest (min_leaf_R2 >= {r2_floor})")
    else:
        pick = max(traj, key=lambda t: t["min_leaf_R2"])
        why = (f"NO set kept all leaves honest at min_leaf_R2 >= {r2_floor}; "
               f"reporting the most-honest achievable (min_leaf_R2 "
               f"{pick['min_leaf_R2']:.2f}) -- the honest/RUL conflict is unresolved")
    return {"leaves": pick["_leaves"], "size": pick["size"],
            "RMSE": pick["RMSE"], "min_leaf_R2": pick["min_leaf_R2"], "why": why}


def _tidy(points):
    cols = ["size", "dropped", "RMSE", "RMSE_sd", "NASA", "R2", "rho_RUL",
            "min_leaf_R2", "min_leaf_R2_sd", "mean_leaf_R2"]
    return pd.DataFrame([{k: p.get(k) for k in cols} for p in points])


# ==========================================================================
# persistence
# ==========================================================================

def save_prune(out, outdir="prune_out"):
    os.makedirs(outdir, exist_ok=True)
    out["trajectory"].to_csv(os.path.join(outdir, "trajectory.csv"), index=False)
    out["pareto"].to_csv(os.path.join(outdir, "pareto.csv"), index=False)
    out["all_points"].to_csv(os.path.join(outdir, "all_points.csv"), index=False)
    rec = out["recommended"]
    cfg = {c: [[n, t, list(s)] for (n, t, s) in e] for c, e in rec["leaves"].items()}
    with open(os.path.join(outdir, "recommended_leaves.json"), "w") as f:
        json.dump({"leaves": cfg, "size": rec["size"], "RMSE": rec["RMSE"],
                   "min_leaf_R2": rec["min_leaf_R2"], "why": rec["why"]}, f, indent=2)
    print(f"saved trajectory/pareto/all_points CSV + recommended_leaves.json -> {outdir}/")
    print("recommended:", rec["why"])
    return outdir


# ==========================================================================
# self-test on the synthetic fleet
# ==========================================================================

def _self_test():
    import ablation
    pooled, modes = ablation._synth_fleet()
    leaves = copy.deepcopy(gft.DEFAULT_LEAVES)      # 6 leaves across 5 components
    print(f"pruning from {_n_leaves(leaves)} leaves on synthetic DS08 ...")
    out = greedy_prune(pooled, leaves, train_ds=["DS08"], seeds=(0,),
                       gens=12, pop=16, min_leaves=2, verbose=True)
    print("\ntrajectory:")
    print(out["trajectory"].round(3).to_string(index=False))
    print("\npareto front:")
    print(out["pareto"].round(3).to_string(index=False))
    assert not out["trajectory"].empty
    assert out["trajectory"]["size"].iloc[0] == _n_leaves(leaves)
    assert out["trajectory"]["size"].iloc[-1] == 2
    assert isinstance(out["recommended"]["leaves"], dict)
    save_prune(out, outdir="/tmp/prune_out")
    print("\nprune self-test OK")


if __name__ == "__main__":
    _self_test()
