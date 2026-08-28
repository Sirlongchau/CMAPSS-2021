"""
run_full.py -- ONE reproducibility driver for every number in the paper (benchmark §4).

`run_all(pooled, leaves, train_ds, ...)` regenerates, in order, writing a CSV per block
and a manifest.json to `outdir`, and printing §4-mapped tables:

  §4a-P1 + §4d  p1_and_age()      -- by-unit CV on DS08, age OFF vs ON (selection)   [freeze_fit]
  §4c           leaf_block()      -- per-leaf held-out theta R2/rho + specificity     [leaf_check]
  §4b           compare()         -- 4 architecture arms, RUL vs min-leaf R2, leak    [this file]
  §4a-P2        loso()            -- leave-one-dataset-out, per unseen mode + mean    [data]
  §4a-P3        native_test()     -- native dev->test, spent ONCE, + per-mode         [freeze_fit]
  §1            manifest          -- rul_cap, n_params, seeds, date, leaf JSON

Every block reuses the already-tested modules (freeze_fit, leaf_check) so this file is
orchestration, not new algorithms. P1/P2 are selection/characterisation (mean +/- sd over
seeds); P3 is the comparable row and is evaluated once. Frozen candidate = decoupled ·
shaft · age=True unless overridden.

RUNTIME on the real fleet is large (P1 = 2 x seeds full fits; §4b = arms x seeds; LOSO =
one fit per DS file). Start with `quick=True` (few seeds, small budget) for a dry run,
then scale. Synthetic self-test: `python run_full.py --selftest`.
"""

from __future__ import annotations
import os, json, datetime
import numpy as np
import pandas as pd

import data, ablation, gft, viz
import freeze_fit, leaf_check


# ==========================================================================
# §4b  architecture arms  (unchanged core)
# ==========================================================================

def _arms(tr, leaves, gens, pop):
    return {
        "shaft-coupled":     lambda s: gft.fit_full(tr, leaves=leaves, grouping="shaft", gens=gens, pop=pop, seed=s),
        "station-coupled":   lambda s: gft.fit_full(tr, leaves=leaves, grouping="station", gens=gens, pop=pop, seed=s),
        "shaft-decoupled":   lambda s: gft.fit_decoupled(tr, leaves=leaves, grouping="shaft", gens=gens, pop=pop, seed=s),
        "station-decoupled": lambda s: gft.fit_decoupled(tr, leaves=leaves, grouping="station", gens=gens, pop=pop, seed=s),
    }


def _score(model, frame):
    r = gft.evaluate(frame, model)
    lr2 = [v for k, v in r.items() if k.endswith(":R2")]
    return {"RMSE": r["RMSE"], "NASA": r["NASA"], "R2": r["R2"], "rho_RUL": r["rho_RUL"],
            "min_leaf_R2": float(np.min(lr2)) if lr2 else np.nan,
            "mean_leaf_R2": float(np.mean(lr2)) if lr2 else np.nan}


def compare(pooled, leaves, train_ds, arms=None, seeds=(0, 1, 2), gens=300, pop=120,
            outdir="figures", verbose=True):
    """4 arms x seeds; RUL and min-leaf R2 both axes; shaft-vs-station leak tables."""
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
        per_seed, last = [], None
        for s in seeds:
            last = fit(s); per_seed.append(_score(last, tr))
        d = pd.DataFrame(per_seed)
        rows.append({"arm": name, "RMSE": d.RMSE.mean(), "RMSE_sd": d.RMSE.std(ddof=0),
                     "NASA": d.NASA.mean(), "R2": d.R2.mean(), "rho_RUL": d.rho_RUL.mean(),
                     "min_leaf_R2": d.min_leaf_R2.mean(),
                     "min_leaf_R2_sd": d.min_leaf_R2.std(ddof=0),
                     "mean_leaf_R2": d.mean_leaf_R2.mean()})
        models[name] = last
    table = pd.DataFrame(rows)
    if verbose:
        print(table.round(3).to_string(index=False))
    leaks = {}
    for g in ("shaft-coupled", "station-coupled"):
        if g in models:
            leaks[g] = ablation.leak_table(models[g], pooled, modes_single)
    return table, models, leaks


# ==========================================================================
# helpers
# ==========================================================================

def _n_params(model):
    return int(len(model["genome"]))


def _rul_cap(pooled, train_ds):
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    return float(gft.suggest_rul_cap(fr))


# ==========================================================================
# §4a-P1 + §4d   CV and the age switch
# ==========================================================================

def p1_and_age(pooled, leaves, train_ds, seeds=(0, 1, 2, 3, 4),
               leaf_gens=300, leaf_pop=80, gens=120, pop=120, outdir="figures",
               also_eval_ds=None):
    res = {}
    for age in (False, True):
        res[age] = freeze_fit.finalize_multi(
            pooled, leaves, train_ds, seeds=seeds, age=age,
            leaf_gens=leaf_gens, leaf_pop=leaf_pop, gens=gens, pop=pop,
            outdir=outdir, also_eval_ds=also_eval_ds, debug_worst=True)
    # age switch table (dev fold)
    off, on = res[False]["summary"], res[True]["summary"]
    def dev(s): return s[s["split"] == "dev"].iloc[0]
    switch = pd.DataFrame([{
        "switch": "age-at-root",
        "off_dev_RMSE": dev(off)["RMSE_mean"], "on_dev_RMSE": dev(on)["RMSE_mean"],
        "dRMSE": dev(on)["RMSE_mean"] - dev(off)["RMSE_mean"],
        "off_dev_NASA": dev(off)["NASA_mean"], "on_dev_NASA": dev(on)["NASA_mean"],
        "d_min_leaf_R2": 0.0,   # age is root-only: leaves identical, by construction
    }])
    return {"cv_age_off": res[False], "cv_age_on": res[True], "age_switch": switch}


# ==========================================================================
# §4c   per-leaf diagnostic fidelity (held-out theta + specificity)
# ==========================================================================

def leaf_block(pooled, leaves, train_ds, age=True, seed=0, leaf_gens=300, leaf_pop=80):
    """Per-leaf held-out theta fidelity (leaf_check.check) merged with specificity
    (default = flat-when-healthy) AND cross-shaft specificity (opposite-shaft foil, which
    separates true global leak from same-shaft sibling coupling for LP leaves)."""
    fid = leaf_check.check(pooled, leaves, train_ds, seed=seed)   # held-out R2/rho + tail slope
    tr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    m, _ = freeze_fit.freeze(tr, leaves, age=age, leaf_gens=leaf_gens,
                             leaf_pop=leaf_pop, seed=seed)
    spec = leaf_check.specificity(m, pooled)[["target", "specificity"]]
    spec_x = (leaf_check.specificity(m, pooled, cross_shaft=True)[["target", "specificity"]]
              .rename(columns={"specificity": "spec_cross_shaft"}))
    return fid.merge(spec, on="target", how="left").merge(spec_x, on="target", how="left")


# ==========================================================================
# §4a-P2   leave-one-dataset-out
# ==========================================================================

def loso(pooled, leaves, ages=(False, True), grouping="shaft", gens=120, pop=120, seed=0):
    """Leave-one-dataset-out, per held-out mode + mean, for EACH age setting -- the age
    trade-off on the generalization axis, not just in-distribution."""
    out = []
    for age in ages:
        rows = []
        for tr, te, ds in data.leave_one_dataset_out(pooled):
            m = gft.fit_decoupled(tr, leaves=leaves, grouping=grouping, age=age,
                                  gens=gens, pop=pop, seed=seed)
            r = freeze_fit.evaluate_on(m, te, f"LOSO:{ds}"); r["age"] = age
            rows.append(r)
        t = pd.DataFrame(rows)
        mean_row = {"split": "LOSO:mean", "age": age, "n_units": int(t["n_units"].sum()),
                    "RMSE": t["RMSE"].mean(), "NASA": t["NASA"].mean(),
                    "R2": t["R2"].mean(), "rho_RUL": t["rho_RUL"].mean()}
        out.append(pd.concat([t, pd.DataFrame([mean_row])], ignore_index=True))
    return pd.concat(out, ignore_index=True)


# ==========================================================================
# §4a-P3   native dev -> test, spent once (per age)
# ==========================================================================

def native_test(pooled, leaves, train_ds, ages=(False, True), grouping="shaft",
                leaf_gens=300, leaf_pop=80, gens=120, pop=120, seed=0):
    if "split" not in pooled.columns:
        print("  [P3] no native 'split' column in pooled -- skipping (real fleet only).")
        return None
    tr = pooled[(pooled["ds"].isin(train_ds)) & (pooled["split"] == "dev")].reset_index(drop=True)
    te = pooled[(pooled["ds"].isin(train_ds)) & (pooled["split"] == "test")].reset_index(drop=True)
    if not len(te):
        print("  [P3] no native test rows for train_ds -- skipping.")
        return None
    out = []
    for age in ages:
        model, frozen = freeze_fit.freeze(tr, leaves, grouping=grouping, age=age,
                                          leaf_gens=leaf_gens, leaf_pop=leaf_pop, seed=seed)
        model = freeze_fit.fit_rul(tr, model, frozen, gens=gens, pop=pop, seed=seed)
        r = freeze_fit.evaluate_on(model, te, "P3:native_pooled"); r["age"] = age
        out.append(r)
        for ds in sorted(pooled["ds"].unique()):
            fd = pooled[(pooled["ds"] == ds) & (pooled["split"] == "test")].reset_index(drop=True)
            if len(fd):
                rr = freeze_fit.evaluate_on(model, fd, f"P3:{ds}"); rr["age"] = age
                out.append(rr)
    return pd.DataFrame(out)


# ==========================================================================
# spool-degeneration A/B: does age slacken the damage representation?
# ==========================================================================

def spool_ab(per_unit_off, per_unit_on):
    """Mean peak spool damage over ALL held-out units, age OFF vs ON, per spool. A drop
    under age = the intermediate damage nodes carry less signal (age explains RUL at the
    root, relieving pressure on the spools). Reads the peak_* columns finalize_multi
    already records; no refit."""
    peak_cols = [c for c in per_unit_on.columns if c.startswith("peak_")]
    rows = []
    for c in peak_cols:
        off = float(per_unit_off[c].mean()); on = float(per_unit_on[c].mean())
        rows.append({"spool_metric": c, "age_off": off, "age_on": on, "delta": on - off})
    return pd.DataFrame(rows)


# ==========================================================================
# orchestrator
# ==========================================================================

def _w(df, outdir, name):
    if df is not None:
        os.makedirs(outdir, exist_ok=True)
        df.round(4).to_csv(os.path.join(outdir, name), index=False)


def run_all(pooled, leaves, train_ds, seeds=(0, 1, 2, 3, 4), age=True,
            arms=None, leaf_gens=300, leaf_pop=80, gens=120, pop=120, arm_gens=300,
            outdir="results", also_eval_ds=None):
    os.makedirs(outdir, exist_ok=True)
    print("=" * 70, "\n§1 setting + frozen rule base\n", "=" * 70)
    rc = _rul_cap(pooled, train_ds)
    # fit the single canonical frozen candidate once: authoritative model for n_params and
    # the rule dump (the reproducible artifact the paper's explainability claim points to).
    tr0 = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    m_frozen, frz = freeze_fit.freeze(tr0, leaves, age=age, leaf_gens=leaf_gens,
                                      leaf_pop=leaf_pop, seed=seeds[0])
    m_frozen = freeze_fit.fit_rul(tr0, m_frozen, frz, gens=gens, pop=pop, seed=seeds[0])
    freeze_fit.save_leaves(m_frozen, frz, os.path.join(outdir, "frozen_leaves.json"))
    rules = freeze_fit.dump_rules(m_frozen, os.path.join(outdir, "rules.json"))
    cdiag = freeze_fit.component_diag(m_frozen, pooled)
    freeze_fit.plot_component_diag(m_frozen, pooled, os.path.join(outdir, "component_diag.png"))
    print("component-damage diagnosis (misdiag_ratio high = false positive):")
    print(cdiag.round(3).to_string(index=False)); _w(cdiag, outdir, "component_diag.csv")
    manifest = {"date": datetime.date.today().isoformat(), "seeds": list(seeds),
                "rul_cap": rc, "n_params": _n_params(m_frozen),
                "n_rules_total": rules["n_rules_total"], "age_frozen": age,
                "grouping": "shaft", "n_leaves": sum(len(v) for v in leaves.values()),
                "train_ds": list(train_ds)}
    print(manifest)

    print("\n" + "=" * 70, "\n§4a-P1 (CV) + §4d (age switch)\n", "=" * 70)
    pa = p1_and_age(pooled, leaves, train_ds, seeds=seeds, leaf_gens=leaf_gens,
                    leaf_pop=leaf_pop, gens=gens, pop=pop, outdir=outdir,
                    also_eval_ds=also_eval_ds)
    _w(pa["cv_age_on"]["summary"], outdir, "p1_cv_age_on.csv")
    _w(pa["cv_age_off"]["summary"], outdir, "p1_cv_age_off.csv")
    _w(pa["cv_age_on"]["diag"], outdir, "p1_diag_age_on.csv")
    _w(pa["cv_age_on"]["blind_by_ds"], outdir, "p1_blind_by_ds_age_on.csv")
    _w(pa["age_switch"], outdir, "age_switch.csv")

    print("\n" + "=" * 70, "\nspool degeneration A/B (age off vs on, all held-out units)\n", "=" * 70)
    sp = spool_ab(pa["cv_age_off"]["per_unit"], pa["cv_age_on"]["per_unit"])
    print(sp.round(3).to_string(index=False)); _w(sp, outdir, "spool_ab.csv")

    print("\n" + "=" * 70, "\n§4c leaf fidelity (held-out theta + specificity + cross-shaft)\n", "=" * 70)
    lf = leaf_block(pooled, leaves, train_ds, age=age, leaf_gens=leaf_gens, leaf_pop=leaf_pop)
    print(lf.round(3).to_string(index=False)); _w(lf, outdir, "leaf_fidelity.csv")

    print("\n" + "=" * 70, "\n§4b architecture arms\n", "=" * 70)
    arm_tab, _, _ = compare(pooled, leaves, train_ds, arms=arms, seeds=seeds,
                            gens=arm_gens, pop=pop, outdir=outdir, verbose=True)
    _w(arm_tab, outdir, "arm_comparison.csv")

    print("\n" + "=" * 70, "\n§4a-P2 LOSO (age off vs on)\n", "=" * 70)
    lo = loso(pooled, leaves, ages=(False, True), gens=gens, pop=pop)
    print(lo.round(3).to_string(index=False)); _w(lo, outdir, "loso.csv")

    print("\n" + "=" * 70, "\n§4a-P3 native test, spent once (age off vs on)\n", "=" * 70)
    p3 = native_test(pooled, leaves, train_ds, ages=(False, True), leaf_gens=leaf_gens,
                     leaf_pop=leaf_pop, gens=gens, pop=pop)
    if p3 is not None:
        print(p3.round(3).to_string(index=False)); _w(p3, outdir, "p3_native.csv")

    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nall blocks written to {outdir}/  (manifest.json + one CSV per block)")
    return {"manifest": manifest, "p1_age": pa, "spool_ab": sp, "leaf": lf,
            "arms": arm_tab, "loso": lo, "p3": p3}


# ==========================================================================
# entry points
# ==========================================================================

def _self_test():
    import copy
    pooled, _ = ablation._synth_fleet()
    # synthesize a native split so the P3 path is exercised (real fleet has one)
    rng = np.random.default_rng(0)
    u2s = {u: ("test" if rng.random() < 0.3 else "dev") for u in pooled["unit"].unique()}
    pooled = pooled.assign(split=pooled["unit"].map(u2s))
    leaves = copy.deepcopy(gft.DEFAULT_LEAVES)
    train = [d for d in pooled["ds"].unique() if d.startswith("DS08")]
    out = run_all(pooled, leaves, train, seeds=(0, 1), age=True,
                  arms=("shaft-coupled", "shaft-decoupled"),
                  leaf_gens=8, leaf_pop=8, gens=8, pop=8, arm_gens=8,
                  outdir="/tmp/results", also_eval_ds=["DS01"])
    for f in ("p1_cv_age_on.csv", "age_switch.csv", "spool_ab.csv", "leaf_fidelity.csv",
              "arm_comparison.csv", "loso.csv", "p3_native.csv", "manifest.json",
              "rules.json", "frozen_leaves.json", "component_diag.csv", "component_diag.png"):
        assert os.path.exists(os.path.join("/tmp/results", f)), f
    assert out["manifest"]["n_params"] > 0
    assert out["manifest"]["n_rules_total"] == out["manifest"]["n_params"]   # rules == params
    assert "spec_cross_shaft" in out["leaf"].columns
    assert set(out["loso"]["age"].unique()) == {False, True}
    assert set(out["p3"]["age"].unique()) == {False, True}
    print("\nrun_full self-test OK (all blocks + rule base; age off/on for LOSO+P3)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _self_test()
    else:
        pooled = data.pooled()
        # FULL 10-leaf set (all 5 components x eff+flow) to test whether the component
        # tier helps generalization with every branch present. For the pruned 7-leaf
        # candidate instead, wrap this in freeze_fit.select_leaves({...}).
        leaves = ablation.load_leaves("ablation_out/best_leaves.json")
        train = [d for d in pooled["ds"].unique() if d.startswith("DS08")]
        run_all(pooled, leaves, train, seeds=(0, 1, 2, 3, 4), age=False,
                also_eval_ds=["DS01", "DS04", "DS05", "DS06", "DS07"], outdir="results_full10")