"""
ablation.py -- the whole curriculum in ONE run. Replaces run.py + confirm.py +
leaf_search.py + add_age_NASA.py.

Three stages, exactly the plan we fixed:

  STAGE 1  IDENTIFY. For each single-fault file, search candidate sensor sets for
           each leaf and answer the two questions a single-fault fit owes:
             (a) which sensors give the best theta RMSE/rho for this leaf, and
             (b) does this branch ALONE predict RUL on its own fault mode.
           Also measures CROSS-MODE LEAK: apply the identified leaf to a foil
           mode and see if it spuriously reports damage (low specificity).
           Curriculum: HPT/DS01, LPT/DS07, HPC/DS05, fan/DS04, and HPC+LPC/DS06
           jointly (1a) -- DS06's HPC leaf is checked against DS05's.

  STAGE 2  ASSEMBLE. Build the full tree from the identified leaves and fit it on
           the all-modes files (DS08a/c), warm-started (free) from the Stage-1
           branch models.

  STAGE 3  AUTOPSY. On the assembled tree, per single-fault mode, report which
           spool/leaf actually lit up (leak table). For any branch that looks
           inert or leaked-around, run the with/without RUL ablation on its own
           mode to separate REDUNDANT (removing it costs nothing -> trim) from
           STARVED/LEAKED (removing it hurts -> keep, and report as a finding).

Runs on a real pooled frame (data.pooled()) or on the synthetic fleet in the
self-test, so it is exercised end to end without the multi-GB files.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

import ga
import gft
from features import COMPONENTS, theta_of, SHAFT


# ==========================================================================
# Persistence -- the identified config must survive the process
# ==========================================================================
# best_leaves is otherwise only an in-memory dict; save it (and the evidence
# tables) so a full-tree run can consume the config WITHOUT re-running Stage 1.

def save_config(out, outdir="ablation_out"):
    """Write the ablation's chosen config + evidence to disk. Produces:
      best_leaves.json  -- the winning leaf/sensor config (feeds gft directly)
      search.csv, branch_report.csv, agreement.csv, leaks.csv, trims.csv
    Returns the config path."""
    os.makedirs(outdir, exist_ok=True)
    cfg = {c: [[name, target, list(sensors)] for name, target, sensors in entries]
           for c, entries in out["best_leaves"].items()}
    path = os.path.join(outdir, "best_leaves.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    for key in ("search", "branch_report", "agreement", "leaks", "trims"):
        df = out.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
    print(f"saved config -> {path}  (+ evidence CSVs in {outdir}/)")
    return path


def load_leaves(path="ablation_out/best_leaves.json"):
    """Load a saved leaves config into the (name, target, sensors) tuple form
    gft.fit_branch / gft.fit_full accept via `leaves=`."""
    with open(path) as f:
        cfg = json.load(f)
    return {c: [(name, target, list(sensors)) for name, target, sensors in entries]
            for c, entries in cfg.items()}

# --- candidate sensor sets per leaf (Stage-1 search space) ------------------
# leaf -> (target modifier, [candidate sensor sets]). Small on purpose; widen for
# a real run. The winners replace gft.DEFAULT_LEAVES for the assembled tree.
LEAVES = {
    "HPT": [("hpt_eff", "HPT_eff_mod",
             [["T48", "P40", "Nc"], ["T48", "P40", "Ps30"], ["T48", "T30", "Nc"]])],
    "LPT": [("lpt_flow", "LPT_flow_mod",
             [["T50", "P50", "P24"], ["T50", "P24", "Nf"], ["T50", "P50", "Nf"]]),
            ("lpt_eff", "LPT_eff_mod",
             [["T50", "P50", "Nf"], ["T50", "Nf", "Nc"]])],
    "HPC": [("hpc_eff", "HPC_eff_mod",
             [["Ps30", "T30", "Nc"], ["P24", "T30", "Nc"], ["Ps30", "P40", "Nc"]])],
    "fan": [("fan_flow", "fan_flow_mod",
             [["P21", "P15", "P24"], ["P21", "P24", "Nf"], ["P15", "P24", "Wf"]])],
    "LPC": [("lpc_eff", "LPC_eff_mod",
             [["P24", "T24", "Nf"], ["P24", "T24", "Nc"], ["T24", "Nf", "Wf"]])],
}
# foil = a component on the OTHER shaft; a specific leaf should NOT fire on it.
FOIL = {"HPT": "LPT", "HPC": "LPT", "fan": "HPT", "LPC": "HPT", "LPT": "HPT"}


# ==========================================================================
# Stage 1 -- leaf sensor identification
# ==========================================================================

def _leaf_theta_fit(frame, sensors, target, gens=25, pop=30, seed=0):
    """Fit a lone leaf (sensors -> target theta) and score theta tracking. Cheap:
    one free FIS, no spool/root, so the sensor search is fast."""
    P = gft.prep(frame, sensors)
    centres = [gft._quantile_centres(P[s]) for s in sensors]
    cols = [P[s].to_numpy() for s in sensors]
    y = frame[target].to_numpy(float)
    shape = tuple(len(c) for c in centres)
    n = int(np.prod(shape))
    span = float(y.max() - y.min()) or 1.0
    lo = np.full(n, y.min() - 0.25 * span)
    hi = np.full(n, y.max() + 0.25 * span)

    def loss(g):
        return gft.rmse(y, gft._fis(cols, centres, g))
    g, _, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed)
    yh = gft._fis(cols, centres, g)
    return {"R2": gft.r2(y, yh), "rho": gft.unit_corr(frame["unit"].to_numpy(), y, yh),
            "singletons": g, "centres": centres}


def leaf_search(single_fault_frame, component, foil_frame=None, **kw):
    """Search sensor sets for each of `component`'s leaves on its single-fault
    frame. Returns rows (one per leaf x sensor set) with theta quality and, if a
    foil frame is given, a leak score = how much the fitted leaf spuriously fires
    on a foil mode (lower = more specific)."""
    rows = []
    for name, target, cand in LEAVES[component]:
        for sensors in cand:
            r = _leaf_theta_fit(single_fault_frame, sensors, target, **kw)
            leak = np.nan
            if foil_frame is not None:
                Pf = gft.prep(foil_frame, sensors)
                yhf = gft._fis([Pf[s].to_numpy() for s in sensors],
                               r["centres"], r["singletons"])
                # foil theta is ~0; spurious magnitude relative to own signal
                own = abs(single_fault_frame[target]).mean() or 1.0
                leak = float(np.abs(yhf).mean() / own)
            rows.append({"component": component, "leaf": name, "target": target,
                         "sensors": "+".join(sensors), "theta_R2": r["R2"],
                         "theta_rho": r["rho"], "leak": leak})
    return pd.DataFrame(rows)


def _best_leaves(search_tables):
    """Pick the best sensor set per leaf (max theta_rho) across ALL tables, so a
    component identified in more than one mode (HPC in DS05 and DS06) yields ONE
    leaf, not a duplicate node. Returns (leaves_dict, agreement_rows) where
    agreement records, per leaf found in >1 mode, whether the modes agree on the
    winning sensor set -- the DS06-vs-DS05 HPC check."""
    allrows = pd.concat(search_tables, ignore_index=True)
    best, agree = {}, []
    for (comp, leaf, target), sub in allrows.groupby(["component", "leaf", "target"]):
        win = sub.loc[sub["theta_rho"].idxmax()]
        best.setdefault(comp, []).append((leaf, target, win["sensors"].split("+")))
        modes = sub["ds"].nunique()
        if modes > 1:
            # best sensor set per mode; do the modes agree?
            per_mode = sub.loc[sub.groupby("ds")["theta_rho"].idxmax()]
            agree.append({"leaf": leaf, "modes": modes,
                          "winner": win["sensors"], "won_in": win["ds"],
                          "sensor_agreement": per_mode["sensors"].nunique() == 1,
                          "rho_spread": round(float(per_mode["theta_rho"].max()
                                                    - per_mode["theta_rho"].min()), 3)})
    return best, pd.DataFrame(agree)


# ==========================================================================
# Stage 1 driver: identify every branch
# ==========================================================================

def identify(pooled, mode_components, gens=30, pop=30, seed=0):
    """Run Stage 1 across the single-fault modes. `mode_components` maps a ds tag
    to its active components. Returns (best_leaves, search_table, branch_models,
    branch_report)."""
    tables, models, brep = [], [], []
    # a foil frame per shaft: any single-HP and single-LP mode
    single = {ds: comps[0] for ds, comps in mode_components.items() if len(comps) == 1}
    hp_foil = next((ds for ds, c in single.items() if SHAFT[c] == "HP"), None)
    lp_foil = next((ds for ds, c in single.items() if SHAFT[c] == "LP"), None)

    for ds, comps in mode_components.items():
        fr = pooled[pooled["ds"] == ds]
        for comp in comps:
            foil_ds = lp_foil if SHAFT[comp] == "HP" else hp_foil
            foil_fr = pooled[pooled["ds"] == foil_ds] if foil_ds else None
            tbl = leaf_search(fr, comp, foil_fr, gens=gens, pop=pop, seed=seed)
            tbl["ds"] = ds
            tables.append(tbl)
    search_table = pd.concat(tables, ignore_index=True)
    best, agreement = _best_leaves(tables)

    # one branch fit per mode with the identified sensors (for branch RUL + warm start)
    for ds, comps in mode_components.items():
        fr = pooled[pooled["ds"] == ds]
        m = gft.fit_branch(comps, fr, leaves=best, gens=gens, pop=pop, seed=seed)
        models.append(m)
        rep = m["report"]
        brep.append({"ds": ds, "components": "+".join(comps),
                     "branch_RMSE": rep["RMSE"], "branch_NASA": rep["NASA"],
                     "branch_rho_RUL": rep["rho_RUL"],
                     **{f"{c}:rho": rep.get(f"{theta_of(c)[0]}:rho", np.nan)
                        for c in comps}})
    return best, search_table, models, pd.DataFrame(brep), agreement


# ==========================================================================
# Stage 2 -- assemble and fit the full tree, warm-started (free)
# ==========================================================================

def assemble(pooled, best_leaves, branch_models, train_ds, gens=50, pop=50, seed=0):
    """Fit the full hp/lp/RUL tree on the all-modes files, seeded from the branches."""
    fit_fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    return gft.fit_full(fit_fr, branch_models=branch_models, leaves=best_leaves,
                        gens=gens, pop=pop, seed=seed)


# ==========================================================================
# Stage 3 -- leak table + trim test on the assembled tree
# ==========================================================================

def leak_table(full_model, pooled, mode_components):
    """Per single-fault mode: peak damage each SPOOL reaches and each leaf's rho on
    that mode's units. The correct branch should light up; another spool lighting
    up is a LEAK (the mode is being read by the wrong component's sensors)."""
    spools = [n["name"] for n in full_model["meta"] if n["kind"] == "spool"]
    rows = []
    for ds, comps in mode_components.items():
        if len(comps) != 1:
            continue
        fr = pooled[pooled["ds"] == ds]
        pred = gft.predict_tree(fr, full_model, full_model["genome"])
        own_spool = gft.owner_spool(full_model, comps[0])
        row = {"ds": ds, "mode": comps[0], "own_spool": own_spool}
        for sp in spools:
            row[f"{sp}_peak"] = float(pred.groupby(fr["unit"].values)[sp + "_h"].max().mean())
        rows.append(row)
    tbl = pd.DataFrame(rows)
    if len(tbl):
        # leak flag: a non-own spool reaches a meaningful fraction of the own spool
        def flag(r):
            own = r[f"{r['own_spool']}_peak"]
            other = max((r[f"{s}_peak"] for s in spools if s != r["own_spool"]),
                        default=0.0)
            return other > 0.5 * max(own, 1e-6)
        tbl["leak"] = tbl.apply(flag, axis=1)
    return tbl


def trim_test(pooled, best_leaves, branch_models, train_ds, component,
              test_ds, gens=40, pop=40, seed=0):
    """Refit the full tree WITH and WITHOUT `component`'s branch and compare RUL on
    `test_ds`. Removing a branch that costs nothing => REDUNDANT (trim). Removing
    one that hurts => STARVED/LEAKED-around (keep; report as a finding)."""
    full = assemble(pooled, best_leaves, branch_models, train_ds,
                    gens=gens, pop=pop, seed=seed)
    reduced_leaves = {c: v for c, v in best_leaves.items() if c != component}
    reduced_models = [m for m in branch_models
                      if component not in _model_components(m)]
    red = assemble(pooled, reduced_leaves, reduced_models, train_ds,
                   gens=gens, pop=pop, seed=seed)
    te = pooled[pooled["ds"].isin(test_ds)].reset_index(drop=True)
    rf, rr = gft.evaluate(te, full), gft.evaluate(te, red)
    verdict = "redundant (trim)" if rr["RMSE"] <= rf["RMSE"] + 1e-6 else "starved/leaked (keep)"
    return {"component": component, "test_ds": "+".join(test_ds),
            "RMSE_with": rf["RMSE"], "RMSE_without": rr["RMSE"],
            "delta": rr["RMSE"] - rf["RMSE"], "verdict": verdict}


def dead_branches(full_model, pooled, mode_components, thresh=0.15):
    """Per single-fault mode, is the OWN spool essentially inert (peak damage below
    `thresh`)? A dead own-spool means that fault is being carried by a leak, not by
    its own leaf -- the branches to look at. Returns a tidy table for viz."""
    rows = []
    for ds, comps in mode_components.items():
        if len(comps) != 1:
            continue
        fr = pooled[pooled["ds"] == ds]
        pred = gft.predict_tree(fr, full_model, full_model["genome"])
        own = gft.owner_spool(full_model, comps[0])
        peak_own = float(pred.groupby(fr["unit"].values)[own + "_h"].max().mean())
        other = [n["name"] for n in full_model["meta"]
                 if n["kind"] == "spool" and n["name"] != own]
        peak_other = max((float(pred.groupby(fr["unit"].values)[s + "_h"].max().mean())
                          for s in other), default=0.0)
        rows.append({"ds": ds, "mode": comps[0], "own_spool": own,
                     "own_peak": round(peak_own, 3), "other_peak": round(peak_other, 3),
                     "dead": peak_own < thresh})
    return pd.DataFrame(rows)


def train_full(pooled, leaves, train_ds, gens=80, pop=60, seed=0, branch_models=None):
    """Train the full tree from a FIXED leaf config (e.g. loaded from
    best_leaves.json) on `train_ds`. Optional branch_models warm-start it. Returns
    the fitted model -- ready for viz.plot_assembled to show the dead branches."""
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    return gft.fit_full(fr, branch_models=branch_models, leaves=leaves,
                        gens=gens, pop=pop, seed=seed)


def _model_components(model):
    comps = set()
    for n in model["meta"]:
        if n["kind"] == "leaf":
            for c, (e, fl) in COMPONENTS.items():
                if n["target"] in (e, fl):
                    comps.add(c)
    return comps


# ==========================================================================
# Single entry point
# ==========================================================================

def run_ablation(pooled, mode_components, train_ds, gens=30, pop=30, seed=0,
                 do_trim=True):
    """The whole staged run. Returns a dict of tables + the assembled model."""
    print("=" * 70, "\nSTAGE 1 -- identify leaves on single-fault files\n", "=" * 70)
    best, search, models, brep, agreement = identify(pooled, mode_components,
                                                     gens=gens, pop=pop, seed=seed)
    print("\nleaf sensor search (best per leaf marked *):")
    show = search.sort_values(["component", "leaf", "theta_rho"],
                              ascending=[True, True, False])
    print(show.round(3).to_string(index=False))
    print("\nidentified leaves:")
    for c, entries in best.items():
        for name, target, sens in entries:
            print(f"  {c:4s} {name:9s} <- {'+'.join(sens)}")
    if len(agreement):
        print("\ncross-mode agreement (a leaf found in >1 mode -- e.g. HPC in DS05 & DS06):")
        print(agreement.to_string(index=False))
    print("\nbranch-alone RUL (does one fault mode's branch predict its own RUL?):")
    print(brep.round(3).to_string(index=False))

    print("\n" + "=" * 70, "\nSTAGE 2 -- assemble full tree on", train_ds,
          "(warm-started)\n", "=" * 70)
    full = assemble(pooled, best, models, train_ds, gens=gens * 2, pop=pop, seed=seed)
    rep = gft.evaluate(pooled[pooled["ds"].isin(train_ds)], full)
    print(f"assembled tree: n_params={rep['n_params']}  RMSE={rep['RMSE']:.2f}  "
          f"NASA={rep['NASA']:.2f}  R2={rep['R2']:.2f}")

    print("\n" + "=" * 70, "\nSTAGE 3 -- leak table + trim autopsy\n", "=" * 70)
    leaks = leak_table(full, pooled, mode_components)
    print("per single-fault mode: which spool lit up (leak = wrong spool fired):")
    print(leaks.round(3).to_string(index=False))

    trims = []
    if do_trim:
        flagged = list(leaks.loc[leaks.get("leak", False), "mode"]) if len(leaks) else []
        for comp in flagged:
            ds_for = [ds for ds, cs in mode_components.items() if cs == [comp]]
            if ds_for:
                trims.append(trim_test(pooled, best, models, train_ds, comp,
                                       ds_for, gens=gens, pop=pop, seed=seed))
        if trims:
            print("\ntrim test on leaked/inert branches:")
            print(pd.DataFrame(trims).round(3).to_string(index=False))
        else:
            print("\nno branch flagged as leaked/inert -- nothing to trim.")

    return {"best_leaves": best, "search": search, "branch_report": brep,
            "agreement": agreement, "assembled": full, "leaks": leaks,
            "trims": pd.DataFrame(trims), "branch_models": models}


# ==========================================================================
# Self-test on a synthetic fleet (no real .h5 needed)
# ==========================================================================

def _synth_fleet():
    modes = {"DS01": ["HPT"], "DS04": ["fan"], "DS05": ["HPC"],
             "DS07": ["LPT"], "DS06": ["HPC", "LPC"],
             "DS08": ["fan", "LPC", "HPC", "HPT", "LPT"]}
    frames = []
    for i, (ds, comps) in enumerate(modes.items()):
        f = gft._synth(comps, units=4, cycles=40, seed=i + 1)
        f["ds"] = ds
        f["unit"] = 1000 * (i + 1) + f["unit"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True), modes


def _self_test():
    pooled, modes = _synth_fleet()
    curriculum = {ds: c for ds, c in modes.items() if ds != "DS08"}
    out = run_ablation(pooled, curriculum, train_ds=["DS08"],
                       gens=18, pop=22, seed=0, do_trim=True)
    assert len(out["best_leaves"]) == 5, out["best_leaves"].keys()
    assert not out["branch_report"].empty
    assert "RUL_hat" in gft.predict_tree(pooled[pooled.ds == "DS08"],
                                         out["assembled"], out["assembled"]["genome"]).columns
    print("\nablation self-test OK")


if __name__ == "__main__":
    _self_test()