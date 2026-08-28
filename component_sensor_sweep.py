"""
component_sensor_sweep.py -- EXHAUSTIVE sensor-set ablation for component honesty.

Question (answered exhaustively, not indicatively): is there ANY 3-sensor set that makes
each component independently diagnosable from cruise means? For every 3-subset of the 13
sensors and every theta target, refit the leaf (theta-frozen, standalone -- reuses
leaf_train._fit_leaf) and measure, on HELD-OUT units, cycle-level:
    rho          observability (per-unit corr of theta_hat with own theta)
    spec         1 - mean|theta_hat|_own-healthy / mean|theta_hat|_own-degraded
    spec_xshaft  same but healthy rows restricted to OPPOSITE-shaft-degrading -> isolates
                 true cross-shaft global-damage leak from same-shaft sibling coupling
A component is honestly diagnosable iff some set clears an observability floor AND a
cross-shaft specificity floor. If none does for fan/LPC/LPT, that is the definitive
observability-ceiling result.

Design for a ONE-TIME run:
  * RESUMABLE -- every (target, sensors) row is appended to the CSV immediately; rerun the
    same command to continue after any interruption (done pairs are skipped).
  * HELD-OUT + cycle-level metrics -- not in-sample, so a winner is not gamed by a sensor
    that merely correlates with fault-mode identity in training.
  * TWO STAGES in run_ablation(): exhaustive single-seed rank -> multi-seed CONFIRM of the
    top-k per target -> best_component.json. You never re-run the 2860 fits to stabilise the
    handful that matter.

Snippet (real fleet):
    import data, component_sensor_sweep as css
    pooled = data.pooled()
    css.run_ablation(pooled, outdir="ablation_component")   # writes CSVs + best_component.json
"""

from __future__ import annotations
import itertools, os, json
import numpy as np
import pandas as pd

import gft, data, leaf_train
from features import SENSORS, COMPONENTS, SHAFT


def _targets():
    ts = []
    for _, (e, fl) in COMPONENTS.items():
        ts += [e, fl]
    return ts


def _comp_of(target):
    for comp, (e, fl) in COMPONENTS.items():
        if target in (e, fl):
            return comp
    return None


def _row_shaft_deg(frame, eps=1e-6):
    """Per-row booleans: is any HP / any LP component actively degrading at that cycle?"""
    hp = np.zeros(len(frame), bool)
    lp = np.zeros(len(frame), bool)
    for comp, (e, fl) in COMPONENTS.items():
        cols = [c for c in (e, fl) if c in frame.columns]
        if not cols:
            continue
        thmin = np.minimum.reduce([frame[c].to_numpy(float) for c in cols])
        d = thmin < -eps
        (hp if SHAFT[comp] == "HP" else lp)[d] = True
    return hp, lp


def _metrics(theta_hat, own_theta, units, hp_deg, lp_deg, shaft, eps=1e-6):
    ah = np.abs(np.asarray(theta_hat, float))
    deg = own_theta < -eps
    heal = np.abs(own_theta) <= eps
    opp = lp_deg if shaft == "HP" else hp_deg
    xheal = heal & opp
    base = ah[deg].mean() if deg.any() else np.nan
    ok = (base is not np.nan) and np.isfinite(base) and base > eps
    spec = (1.0 - ah[heal].mean() / base) if (ok and heal.any()) else np.nan
    specx = (1.0 - ah[xheal].mean() / base) if (ok and xheal.any()) else np.nan
    rho = gft.unit_corr(units, own_theta, theta_hat)
    return rho, spec, specx, int(deg.sum()), int(heal.sum()), int(xheal.sum())


def sweep(pooled, out_csv, targets=None, sensor_pool=None, k=3,
          leaf_gens=30, leaf_pop=16, split_seed=0, fit_seed=0, resume=True, log_every=50):
    """Exhaustive single-seed sweep. Appends each (target, sensors) result to `out_csv`
    immediately (resumable). Returns the full table."""
    targets = targets or _targets()
    pool = list(sensor_pool) if sensor_pool else list(SENSORS)
    tr, ho, _ = data.split(pooled, seed=split_seed)            # hold-out spans fault modes
    hp_ho, lp_ho = _row_shaft_deg(ho)
    units_ho = ho["unit"].to_numpy()

    done = set()
    header = True
    if os.path.exists(out_csv):
        header = False
        if resume:
            prev = pd.read_csv(out_csv)
            done = set(zip(prev["target"].astype(str), prev["sensors"].astype(str)))

    combos = ["+".join(c) for c in itertools.combinations(pool, k)]
    total = len(targets) * len(combos)
    i = 0
    for target in targets:
        comp = _comp_of(target); shaft = SHAFT[comp]
        own_ho = ho[target].to_numpy(float)
        for sensors_str in combos:
            i += 1
            if (target, sensors_str) in done:
                continue
            sensors = sensors_str.split("+")
            m = leaf_train._fit_leaf(tr, sensors, target, k, leaf_gens, leaf_pop, fit_seed)
            th = leaf_train._predict(m, ho)
            rho, spec, specx, nd, nh, nx = _metrics(th, own_ho, units_ho, hp_ho, lp_ho, shaft)
            pd.DataFrame([{"target": target, "component": comp, "shaft": shaft,
                           "sensors": sensors_str, "rho": rho, "spec": spec,
                           "spec_xshaft": specx, "n_deg": nd, "n_heal": nh, "n_xheal": nx}]
                         ).to_csv(out_csv, mode="a", header=header, index=False)
            header = False
            if i % log_every == 0:
                print(f"  {i}/{total}  {target:14s} {sensors_str:14s} "
                      f"rho={rho:.2f} spec={spec:.2f} spec_x={specx:.2f}")
    return pd.read_csv(out_csv)


def confirm_and_pick(pooled, ablation_df, outdir, topk=6, rho_floor=0.5,
                     seeds=(0, 1, 2), k=3, leaf_gens=45, leaf_pop=24, split_seed=0):
    """Multi-seed CONFIRM of the top-k (by spec_xshaft, subject to rho>=rho_floor) per
    target; pick the best per target -> best_component.json (+ confirm CSV)."""
    os.makedirs(outdir, exist_ok=True)
    tr, ho, _ = data.split(pooled, seed=split_seed)
    hp_ho, lp_ho = _row_shaft_deg(ho); units_ho = ho["unit"].to_numpy()
    rows, best = [], {}
    for target, g in ablation_df.groupby("target"):
        comp = _comp_of(target); shaft = SHAFT[comp]
        own_ho = ho[target].to_numpy(float)
        cand = (g[g["rho"] >= rho_floor].sort_values("spec_xshaft", ascending=False).head(topk)
                if (g["rho"] >= rho_floor).any()
                else g.sort_values("spec_xshaft", ascending=False).head(topk))
        best_row = None
        for _, r in cand.iterrows():
            sensors = str(r["sensors"]).split("+")
            R, S, X = [], [], []
            for s in seeds:
                m = leaf_train._fit_leaf(tr, sensors, target, k, leaf_gens, leaf_pop, s)
                th = leaf_train._predict(m, ho)
                rho, spec, specx, *_ = _metrics(th, own_ho, units_ho, hp_ho, lp_ho, shaft)
                R.append(rho); S.append(spec); X.append(specx)
            row = {"target": target, "component": comp, "shaft": shaft,
                   "sensors": r["sensors"],
                   "rho": float(np.nanmean(R)), "rho_sd": float(np.nanstd(R)),
                   "spec": float(np.nanmean(S)),
                   "spec_xshaft": float(np.nanmean(X)), "spec_xshaft_sd": float(np.nanstd(X))}
            rows.append(row)
            if best_row is None or row["spec_xshaft"] > best_row["spec_xshaft"]:
                best_row = row
        if best_row is not None:
            best.setdefault(comp, {})[target] = best_row
    confirm = pd.DataFrame(rows)
    confirm.round(4).to_csv(os.path.join(outdir, "component_ablation_confirm.csv"), index=False)
    with open(os.path.join(outdir, "best_component.json"), "w") as f:
        json.dump({"rho_floor": rho_floor, "seeds": list(seeds), "best": best}, f, indent=2)
    print(f"wrote {outdir}/component_ablation_confirm.csv and best_component.json")
    return confirm, best


def run_ablation(pooled, outdir="ablation_component", sensor_pool=None, k=3,
                 leaf_gens=30, leaf_pop=16, confirm_seeds=(0, 1, 2), rho_floor=0.5, topk=6):
    """One command: exhaustive resumable sweep -> multi-seed confirm top-k -> best_component.json."""
    os.makedirs(outdir, exist_ok=True)
    csv = os.path.join(outdir, "component_ablation.csv")
    print("stage 1: exhaustive sweep (resumable) ...")
    abl = sweep(pooled, csv, sensor_pool=sensor_pool, k=k, leaf_gens=leaf_gens, leaf_pop=leaf_pop)
    print("\nstage 2: multi-seed confirm of top-k per target ...")
    confirm, best = confirm_and_pick(pooled, abl, outdir, topk=topk, rho_floor=rho_floor,
                                     seeds=confirm_seeds, k=k)
    # compact verdict per component/modifier
    print("\nbest per target (held-out, multi-seed):")
    print(confirm.sort_values(["component", "spec_xshaft"], ascending=[True, False])
          .groupby("component").head(1)
          [["component", "target", "sensors", "rho", "spec", "spec_xshaft"]]
          .round(3).to_string(index=False))
    return {"ablation": abl, "confirm": confirm, "best": best}


def _self_test():
    import ablation
    pooled, _ = ablation._synth_fleet()
    # restrict to a 5-sensor pool -> C(5,3)=10 combos, quick; and 2 targets
    out = run_ablation(pooled, outdir="/tmp/abl",
                       sensor_pool=["T48", "T30", "Nc", "P24", "Nf"],
                       leaf_gens=8, leaf_pop=8, confirm_seeds=(0,), topk=3)
    assert os.path.exists("/tmp/abl/component_ablation.csv")
    assert os.path.exists("/tmp/abl/best_component.json")
    assert {"rho", "spec", "spec_xshaft"} <= set(out["ablation"].columns)
    # resume: second call should add nothing (all pairs done)
    n1 = len(pd.read_csv("/tmp/abl/component_ablation.csv"))
    sweep(pooled, "/tmp/abl/component_ablation.csv",
          sensor_pool=["T48", "T30", "Nc", "P24", "Nf"], leaf_gens=8, leaf_pop=8)
    n2 = len(pd.read_csv("/tmp/abl/component_ablation.csv"))
    assert n1 == n2, f"resume failed: {n1} -> {n2}"
    print("\ncomponent_sensor_sweep self-test OK (resumable; CSV + best_component.json)")


if __name__ == "__main__":
    _self_test()
