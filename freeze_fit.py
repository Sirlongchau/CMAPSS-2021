"""
freeze_fit.py -- staged decoupled pipeline: large leaf training -> freeze -> observe ->
save JSON -> train the aggregator on RUL over the frozen leaves.

Same two phases as gft.fit_decoupled, but SPLIT so the frozen leaves can be inspected and
PERSISTED before (and independently of) aggregator training -- freeze once, reuse for
several RUL fits (e.g. with/without age). Reuses gft's real primitives throughout:
build_tree, _freeze_leaves (phase 1), _rul_loss_fn + ga.optimize (phase 2). Leaves are
theta-only and pinned during phase 2, so RUL can never distort them.

    select_leaves(all, keep)                 -> filter a loaded config to chosen modifiers
    m, frozen = freeze(frame, leaves, ...)   -> LARGE leaf training + freeze
    save_leaves(m, frozen, path)             -> per-leaf params (centres + consequents) to JSON
    frozen = load_frozen(m, path)            -> restore frozen leaves into a model
    m = fit_rul(frame, m, frozen, ...)       -> train spool+root on RUL, leaves pinned
    run(frame, leaves, ...)                  -> all of the above + observe (metrics + graph)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

import gft, ga, data


def _grid(k):
    cols = int(np.ceil(np.sqrt(k))); rows = int(np.ceil(k / cols)); return rows, cols


# --------------------------------------------------------------------------
# choose the leaf set
# --------------------------------------------------------------------------

def select_leaves(all_leaves, keep):
    """keep = {component: [modifiers]}, e.g. {"HPT":["eff"], "HPC":["eff","flow"],
    "fan":["eff"], "LPC":["eff"], "LPT":["eff","flow"]}. Filters a loaded leaves config
    down to the chosen modifiers (targets like 'HPT_eff_mod')."""
    out = {}
    for comp, entries in all_leaves.items():
        mods = keep.get(comp)
        if not mods:
            continue
        sel = [(nm, tg, list(se)) for (nm, tg, se) in entries
               if any(tg.endswith(f"{m}_mod") for m in mods)]
        if sel:
            out[comp] = sel
    return out


# --------------------------------------------------------------------------
# phase 1: large leaf training + freeze
# --------------------------------------------------------------------------

def freeze(frame, leaves, grouping="shaft", age=False,
           leaf_gens=300, leaf_pop=80, seed=0):
    """Build the tree skeleton and fit each leaf to its OWN theta with a large GA budget,
    then freeze. Returns (model, frozen_genome). model['genome'] is set to the frozen
    genome so leaves can be observed immediately (aggregator genes are placeholder)."""
    spec = gft.full_spec(leaves, grouping, age)
    rul_cap = gft.suggest_rul_cap(frame)
    model = gft.build_tree(spec, frame, rul_cap)
    frozen = gft._freeze_leaves(frame, model, gens=leaf_gens, pop=leaf_pop, seed=seed)
    model["genome"] = frozen.copy()
    model["spec"] = spec; model["grouping"] = grouping
    model["age"] = age; model["decoupled"] = True
    return model, frozen


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_leaves(model, frozen, path="frozen_leaves.json"):
    """Per-leaf params to JSON: inputs, MF centres, and fitted rule consequents. Enough to
    reconstruct the leaf's output and to reload into a model built on the same frame."""
    leaves = {}
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        leaves[n["target"]] = {
            "name": n["name"],
            "inputs": list(n["inputs"]),
            "centres": [np.asarray(c).tolist() for c in n["centres"]],
            "consequents": np.asarray(frozen[n["rules"]]).tolist(),
        }
    blob = {"grouping": model.get("grouping", "shaft"),
            "age": bool(model.get("age", False)),
            "rul_cap": float(model["rul_cap"]),
            "leaves": leaves}
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"saved {len(leaves)} frozen leaves -> {path}")
    return path


def load_frozen(model, path="frozen_leaves.json"):
    """Restore frozen leaf consequents from JSON into a genome for `model`. Validates
    length (MF count / sensors) and warns if saved centres differ from the model's frame."""
    with open(path) as f:
        blob = json.load(f)
    frozen = model["genome"].copy() if "genome" in model else 0.5 * (model["lo"] + model["hi"])
    saved = blob["leaves"]
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        s = saved.get(n["target"])
        if s is None:
            raise KeyError(f"{n['target']} missing in {path}")
        cons = np.asarray(s["consequents"], float)
        want = n["rules"].stop - n["rules"].start
        if cons.shape[0] != want:
            raise ValueError(f"{n['target']}: {cons.shape[0]} consequents, model wants {want} "
                             "(MF count or sensor set changed)")
        for cs, cn in zip(s["centres"], n["centres"]):
            if not np.allclose(np.asarray(cs), np.asarray(cn), atol=1e-4):
                print(f"  WARN {n['target']}: saved centres != current frame's "
                      "(different data?) -- consequents may not align")
                break
        frozen[n["rules"]] = cons
    return frozen


# --------------------------------------------------------------------------
# phase 2: train aggregator on RUL, leaves pinned
# --------------------------------------------------------------------------

def fit_rul(frame, model, frozen, gens=120, pop=120, seed=0, lambda_spec=0.0):
    """Pin leaf genes to `frozen`, train spool+root on RUL (gft._rul_loss_fn). Mirrors
    fit_decoupled's phase 2 exactly. lambda_spec>0 adds the component-specificity penalty.
    Sets model['genome']."""
    lo, hi = model["lo"].copy(), model["hi"].copy()
    for n in model["meta"]:
        if n["kind"] == "leaf":
            lo[n["rules"]] = frozen[n["rules"]]
            hi[n["rules"]] = frozen[n["rules"]]
    loss = gft._rul_loss_fn(frame, model, lambda_spec=lambda_spec)
    g, f, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed, seeds=[frozen])
    model["genome"] = g
    model["loss"] = f
    return model


# --------------------------------------------------------------------------
# full staged run + observe
# --------------------------------------------------------------------------

def run(frame, leaves, grouping="shaft", age=False,
        leaf_gens=300, leaf_pop=80, gens=120, pop=120, seed=0,
        json_path="frozen_leaves.json", outdir="figures"):
    import os
    os.makedirs(outdir, exist_ok=True)

    print("phase 1: large leaf training + freeze ...")
    model, frozen = freeze(frame, leaves, grouping, age, leaf_gens, leaf_pop, seed)

    rep0 = gft.evaluate(frame, model)
    leaftab = pd.DataFrame([{"target": n["target"],
                             "R2": rep0.get(n["target"] + ":R2"),
                             "rho": rep0.get(n["target"] + ":rho")}
                            for n in model["meta"] if n["kind"] == "leaf"])
    print("frozen leaf fidelity (in-sample):")
    print(leaftab.round(3).to_string(index=False))

    import leaf_check
    fig = leaf_check.plot_trajectories(model, frame, os.path.join(outdir, "frozen_leaves.png"))
    print("observed ->", fig)

    save_leaves(model, frozen, json_path)

    print("\nphase 2: train aggregator on RUL (leaves pinned) ...")
    model = fit_rul(frame, model, frozen, gens=gens, pop=pop, seed=seed)
    rep = gft.evaluate(frame, model)
    print(f"RUL (decoupled): RMSE={rep['RMSE']:.3f}  NASA={rep['NASA']:.3f}  "
          f"R2={rep['R2']:.3f}  rho_RUL={rep['rho_RUL']:.3f}")
    return model, frozen, {"leaf_fidelity": leaftab, "rul": rep}


def _term_labels(k):
    if k == 3:
        return ["low", "mid", "high"]
    if k == 2:
        return ["low", "high"]
    return [f"L{j}" for j in range(k)]


def dump_rules(model, path=None):
    """Export the tree's fuzzy rule base as JSON, one block per NODE (leaf / spool / root),
    to authenticate the inspectable-rule-base claim. Per node: its inputs, the membership
    functions (term label + centre in the model's processed input space), and every rule as
    IF <antecedent terms> THEN <consequent> -- the consequent is the REAL post-anchor output
    (gft._consequents), so spool rules read in [0,1] damage and root rules in [0, rul_cap].
    Rules are enumerated in the model's own C-order (matches _fis), and the total rule count
    equals n_params. Also emits a flat `readable` list of plain-language rules per node."""
    import itertools
    g = model["genome"]
    nodes = []
    for n in model["meta"]:
        cons = gft._consequents(n, g)                  # actual per-rule output values
        shape, inputs, centres = n["shape"], list(n["inputs"]), n["centres"]
        out_label = (n["target"] if n["kind"] == "leaf"
                     else "RUL" if n["kind"] == "root"
                     else f"{n['name']}_damage")
        mfs = {name: [{"term": _term_labels(len(c))[j], "centre": round(float(c[j]), 5)}
                      for j in range(len(c))]
               for name, c in zip(inputs, centres)}
        rules, readable = [], []
        for flat, terms in enumerate(itertools.product(*[range(s) for s in shape])):
            ant = [{"input": name, "term": _term_labels(len(c))[j]}
                   for name, c, j in zip(inputs, centres, terms)]
            val = round(float(cons[flat]), 5)
            rules.append({"if": ant, "then": val})
            cond = " AND ".join(f"{a['input']} is {a['term']}" for a in ant)
            readable.append(f"IF {cond} THEN {out_label} = {val}")
        nodes.append({"name": n["name"], "kind": n["kind"], "output": out_label,
                      "inputs": inputs, "n_rules": len(rules),
                      "membership_functions": mfs, "rules": rules, "readable": readable})
    blob = {"n_nodes": len(nodes),
            "n_rules_total": sum(x["n_rules"] for x in nodes),
            "n_params": int(len(g)), "rul_cap": float(model["rul_cap"]),
            "grouping": model.get("grouping", "shaft"), "age": bool(model.get("age", False)),
            "nodes": nodes}
    if path:
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"dumped {blob['n_rules_total']} rules across {blob['n_nodes']} nodes -> {path}")
    return blob


def component_diag(model, pooled, eps=1e-6):
    """Diagnosis check for the per-component damage nodes (<c>_dmg). Components are RUL-fit
    (phase 2), not theta-frozen, so verify each reads ITS OWN degradation, not global damage:
      rho_dmg_theta  per-unit corr(dmg, that component's true theta) over degrading units,
                     sign-flipped so a good node -> +1 (dmg rises as theta falls)
      peak_deg       mean peak dmg on units where the component degrades   (want -> ~1)
      peak_healthy   mean peak dmg on units where it is HEALTHY             (want -> ~0)
      misdiag_ratio  peak_healthy / peak_deg -- HIGH = false-positive misdiagnosis (the
                     node lights up when its own component is fine = the specificity issue
                     at the damage level, e.g. LPT reading LP-global damage)."""
    from features import COMPONENTS
    pred = gft.predict_tree(pooled, model, model["genome"])
    units = pooled["unit"].to_numpy()
    rows = []
    for n in model["meta"]:
        if n["kind"] != "component":
            continue
        c = n["name"]
        dmg = pred[c + "_dmg"].to_numpy()
        mods = [t for t in COMPONENTS.get(c, ()) if t in pooled.columns]
        th = (np.minimum.reduce([pooled[t].to_numpy(float) for t in mods])
              if mods else np.zeros(len(pooled)))
        peak_deg, peak_heal, rhos = [], [], []
        for u in np.unique(units):
            mu = units == u
            pk = float(np.max(dmg[mu]))
            if (th[mu].max() - th[mu].min()) > eps:
                peak_deg.append(pk)
                if np.std(dmg[mu]) > eps and np.std(th[mu]) > eps:
                    rhos.append(np.corrcoef(th[mu], dmg[mu])[0, 1])
            else:
                peak_heal.append(pk)
        pd_ = float(np.mean(peak_deg)) if peak_deg else np.nan
        ph_ = float(np.mean(peak_heal)) if peak_heal else np.nan
        rows.append({"component": c,
                     "rho_dmg_theta": float(-np.mean(rhos)) if rhos else np.nan,
                     "peak_deg": pd_, "peak_healthy": ph_,
                     "misdiag_ratio": (ph_ / pd_) if (peak_heal and peak_deg and pd_ > eps) else np.nan})
    return pd.DataFrame(rows)


def plot_component_diag(model, pooled, save, max_units=8, seed=0):
    """Per component: predicted damage over cycle, DEGRADING units (blue, should rise to ~1)
    vs HEALTHY units (red, should stay ~0). Red curves that climb = misdiagnosis."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from features import COMPONENTS
    pred = gft.predict_tree(pooled, model, model["genome"])
    m = pooled[["unit", "cycle"]].copy()
    comps = [n["name"] for n in model["meta"] if n["kind"] == "component"]
    units = pooled["unit"].to_numpy()
    rng = np.random.default_rng(seed)
    rows_g, cols_g = _grid(len(comps))
    fig, ax = plt.subplots(rows_g, cols_g, figsize=(3.8 * cols_g, 3.0 * rows_g), squeeze=False)
    for i, c in enumerate(comps):
        a = ax[i // cols_g][i % cols_g]
        dmg = pred[c + "_dmg"].to_numpy()
        mods = [t for t in COMPONENTS.get(c, ()) if t in pooled.columns]
        th = (np.minimum.reduce([pooled[t].to_numpy(float) for t in mods])
              if mods else np.zeros(len(pooled)))
        deg_u, heal_u = [], []
        for u in np.unique(units):
            mu = units == u
            (deg_u if (th[mu].max() - th[mu].min()) > 1e-6 else heal_u).append(u)
        for grp, col in ((rng.permutation(deg_u)[:max_units], "steelblue"),
                         (rng.permutation(heal_u)[:max_units], "crimson")):
            for u in grp:
                mu = units == u; o = np.argsort(pooled["cycle"].to_numpy()[mu])
                a.plot(pooled["cycle"].to_numpy()[mu][o], dmg[mu][o], color=col, lw=1.0, alpha=0.6)
        a.set_title(f"{c}_dmg  (blue=degrading, red=healthy)", fontsize=9)
        a.set_xlabel("cycle"); a.set_ylabel("damage"); a.set_ylim(-0.02, 1.02)
    for j in range(len(comps), rows_g * cols_g):
        ax[j // cols_g][j % cols_g].axis("off")
    fig.suptitle("Per-component damage -- red (healthy) rising = misdiagnosis")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(save, dpi=110); plt.close(fig)
    return save


# --------------------------------------------------------------------------
# evaluate + plot RUL on any frame; finalize on held-out dev/test
# --------------------------------------------------------------------------

def evaluate_on(model, frame, label=""):
    """RUL metrics of a fitted model on `frame` (gft.evaluate). rul_cap stays the one
    from training -- dev/test are not peeked at for capping."""
    rep = gft.evaluate(frame, model)
    return {"split": label, "n_units": int(frame["unit"].nunique()),
            "RMSE": rep["RMSE"], "NASA": rep["NASA"], "R2": rep["R2"],
            "rho_RUL": rep["rho_RUL"]}


def plot_rul(model, frame, save, title="", max_units=6, seed=0):
    """RUL of the tree on `frame`: left = true-vs-pred parity (capped, RMSE in title),
    right = per-unit RUL over cycle (black true, blue predicted). Works on any frame
    (train/dev/test/other datasets)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pred = gft.predict_tree(frame, model, model["genome"])
    m = frame[["unit", "cycle", "RUL"]].merge(pred[["unit", "cycle", "RUL_hat"]],
                                              on=["unit", "cycle"])
    cp = gft.cap(m["RUL"].to_numpy(float), model["rul_cap"])
    yh = m["RUL_hat"].to_numpy(float)
    rmse = gft.rmse(cp, yh)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    hi = float(np.max(cp)) * 1.05 + 1e-6
    ax[0].scatter(cp, yh, s=8, alpha=0.4, c="steelblue", edgecolors="none")
    ax[0].plot([0, hi], [0, hi], "k--", lw=1)
    ax[0].set_xlim(0, hi); ax[0].set_ylim(0, hi)
    ax[0].set_xlabel("true RUL (capped)"); ax[0].set_ylabel("pred RUL")
    ax[0].set_title(f"RUL parity -- RMSE {rmse:.2f}")

    rng = np.random.default_rng(seed)
    units = list(m["unit"].unique())
    sel = list(rng.choice(units, size=min(max_units, len(units)), replace=False))
    for u in sel:
        d = m[m["unit"] == u].sort_values("cycle")
        ax[1].plot(d["cycle"], gft.cap(d["RUL"].to_numpy(float), model["rul_cap"]),
                   color="k", lw=1.0, alpha=0.6)
        ax[1].plot(d["cycle"], d["RUL_hat"], color="steelblue", lw=1.2, alpha=0.85)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("RUL")
    ax[1].set_title(f"RUL over life -- {len(sel)} units (black true, blue pred)")
    fig.suptitle(("RUL -- " + title).strip(" -"))
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(save, dpi=110); plt.close(fig)
    return save


def finalize(pooled, leaves, train_ds, grouping="shaft", age=False,
             leaf_gens=300, leaf_pop=80, gens=120, pop=120, seed=0,
             json_path="frozen_leaves.json", outdir="figures", also_eval_ds=None):
    """Honest dev/test finalization. Splits train_ds into train/dev/test BY UNIT
    (data.split), fits leaves + aggregator on TRAIN ONLY, reports RUL on all splits, and
    plots dev/test RUL. `also_eval_ds` = extra whole datasets to score as cross-mode
    generalization (e.g. single-fault DS0x files). Returns (model, frozen, metrics_table)."""
    import os; os.makedirs(outdir, exist_ok=True)
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    tr, dev, test = data.split(fr, seed=seed)

    print("phase 1: large leaf training + freeze (TRAIN units only) ...")
    model, frozen = freeze(tr, leaves, grouping, age, leaf_gens, leaf_pop, seed)
    save_leaves(model, frozen, json_path)
    print("phase 2: aggregator on RUL (TRAIN units) ...")
    model = fit_rul(tr, model, frozen, gens=gens, pop=pop, seed=seed)

    rows = [evaluate_on(model, tr, "train"),
            evaluate_on(model, dev, "dev"),
            evaluate_on(model, test, "test")]
    if also_eval_ds:
        for d in also_eval_ds:
            fd = pooled[pooled["ds"] == d].reset_index(drop=True)
            if len(fd):
                rows.append(evaluate_on(model, fd, f"ds:{d}"))
    tab = pd.DataFrame(rows)
    print("\nRUL by split:")
    print(tab.round(3).to_string(index=False))

    plot_rul(model, dev, os.path.join(outdir, "rul_dev.png"), "dev")
    plot_rul(model, test, os.path.join(outdir, "rul_test.png"), "test")
    print(f"wrote {outdir}/rul_dev.png and {outdir}/rul_test.png")
    return model, frozen, tab


# --------------------------------------------------------------------------
# per-unit diagnostics + multi-seed finalization
# --------------------------------------------------------------------------

def _unit_metrics(model, frame, last_k=10, smooth=5):
    """Per-unit held-out diagnostics that RMSE alone hides:
      rmse_u    -- per-unit RUL RMSE (capped)
      rebound   -- (max predicted RUL AFTER its minimum - that minimum)/rul_cap, on a
                   median-smoothed curve. Predicted RUL should only fall over life; a
                   large rebound = the non-monotone 'comes back to life' failure.
      late_bias -- mean(pred - true) over the last `last_k` cycles. POSITIVE = optimistic
                   (says more life remains than there is) -- the unsafe direction.
      peak_<spool> -- max damage each spool reaches over life. A blind unit's spools stay
                   flat (peak << 1) because its leaves under-read degradation.
      ds        -- the unit's fault mode, for grouping the blind population.
    """
    pred = gft.predict_tree(frame, model, model["genome"])
    spools = [n["name"] for n in model["meta"] if n["kind"] == "spool"]
    keep = ["unit", "cycle", "RUL_hat"] + [sp + "_h" for sp in spools]
    m = frame[["unit", "cycle", "RUL", "ds"]].merge(pred[keep], on=["unit", "cycle"])
    cap = model["rul_cap"]
    rows = []
    for u in m["unit"].unique():
        d = m[m["unit"] == u].sort_values("cycle")
        true = gft.cap(d["RUL"].to_numpy(float), cap)
        ph = d["RUL_hat"].to_numpy(float)
        if len(ph) < 3:
            continue
        phs = pd.Series(ph).rolling(smooth, min_periods=1, center=True).median().to_numpy()
        imin = int(np.argmin(phs))
        rebound = float((phs[imin:].max() - phs[imin]) / cap) if cap else np.nan
        k = min(last_k, len(ph))
        late_bias = float(np.mean(ph[-k:] - true[-k:]))
        peaks = {f"peak_{sp}": float(d[sp + "_h"].max()) for sp in spools}
        rows.append({"unit": u, "ds": d["ds"].iloc[0], "n_cycles": len(ph),
                     "rmse_u": gft.rmse(true, ph), "rebound": rebound,
                     "late_bias": late_bias,
                     "peak_damage": max(peaks.values()) if peaks else np.nan, **peaks})
    return pd.DataFrame(rows)


def _blind_report(ho, q=0.9):
    """The leaf-blind population: held-out unit-evals with late_bias in the top decile
    (most optimistic). Grouped by fault mode, with mean peak spool damage so you can see
    which spool stays flat. Returns (threshold, blind_rows, by_mode_table)."""
    empty = ho.iloc[0:0]
    if not len(ho):
        return np.nan, empty, empty
    thr = float(ho["late_bias"].quantile(q))
    blind = ho[ho["late_bias"] >= thr].copy()
    peak_cols = [c for c in ho.columns if c.startswith("peak_")]
    agg = {"n": ("unit", "size"), "late_bias": ("late_bias", "mean"),
           "rmse_u": ("rmse_u", "mean")}
    for pc in peak_cols:
        agg[pc] = (pc, "mean")
    by_ds = (blind.groupby("ds").agg(**agg).reset_index()
             .sort_values("n", ascending=False))
    return thr, blind, by_ds


def plot_unit_debug(model, frame_unit, unit, save):
    """One unit, three panels: RUL true vs pred; spool damages; leaf theta_hat -- all over
    cycle. Locates where a RUL failure (e.g. a late rebound) originates in the tree."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pred = gft.predict_tree(frame_unit, model, model["genome"])
    d = frame_unit[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"]).sort_values("cycle")
    c = d["cycle"].to_numpy()
    spools = [n["name"] for n in model["meta"] if n["kind"] == "spool"]
    leaves = [n["target"] for n in model["meta"] if n["kind"] == "leaf"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(c, gft.cap(d["RUL"].to_numpy(float), model["rul_cap"]), "k", lw=1.2, label="true")
    ax[0].plot(c, d["RUL_hat"], "steelblue", lw=1.4, label="pred")
    ax[0].set_title(f"unit {unit} -- RUL"); ax[0].set_xlabel("cycle"); ax[0].legend(fontsize=8)
    for sp in spools:
        ax[1].plot(c, d[sp + "_h"], lw=1.3, label=sp)
    ax[1].set_title("spool damage"); ax[1].set_xlabel("cycle"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend(fontsize=7)
    for tg in leaves:
        ax[2].plot(c, d[tg + "_hat"], lw=1.0, label=tg)
    ax[2].set_title("leaf theta_hat"); ax[2].set_xlabel("cycle"); ax[2].legend(fontsize=6)
    fig.suptitle(f"Unit {unit} debug -- trace a RUL failure to spool / leaf")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(save, dpi=110); plt.close(fig)
    return save


def finalize_multi(pooled, leaves, train_ds, seeds=(0, 1, 2, 3, 4), age=False,
                   grouping="shaft", leaf_gens=300, leaf_pop=80, gens=120, pop=120,
                   outdir="figures", also_eval_ds=None, debug_worst=True):
    """Multi-seed dev/test finalization. Per seed: split by unit, freeze on train, fit RUL
    on train, score all splits. Reports per-split RMSE/NASA/R2/rho as mean +/- sd across
    seeds, the held-out PER-UNIT RMSE distribution, the non-monotone fraction, and the
    late-life bias (optimism). Auto-plots the worst held-out unit. Run once with age=False
    and once with age=True for a clean A/B. Returns dict(summary, per_unit, worst)."""
    import os; os.makedirs(outdir, exist_ok=True)
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    split_rows, unit_rows, models = [], [], {}
    for s in seeds:
        tr, dev, test = data.split(fr, seed=s)
        model, frozen = freeze(tr, leaves, grouping, age, leaf_gens, leaf_pop, s)
        model = fit_rul(tr, model, frozen, gens=gens, pop=pop, seed=s)
        models[s] = model
        for name, frm in [("train", tr), ("dev", dev), ("test", test)]:
            r = evaluate_on(model, frm, name); r["seed"] = s; split_rows.append(r)
        for name, frm in [("dev", dev), ("test", test)]:
            um = _unit_metrics(model, frm); um["seed"] = s; um["split"] = name
            unit_rows.append(um)
        for d in (also_eval_ds or []):
            fd = pooled[pooled["ds"] == d].reset_index(drop=True)
            if len(fd):
                r = evaluate_on(model, fd, f"ds:{d}"); r["seed"] = s; split_rows.append(r)

    sp = pd.DataFrame(split_rows)
    order = ["train", "dev", "test"] + sorted(x for x in sp["split"].unique() if x.startswith("ds:"))
    summary = (sp.groupby("split")
                 .agg(RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                      NASA_mean=("NASA", "mean"), NASA_sd=("NASA", "std"),
                      R2_mean=("R2", "mean"), rho_mean=("rho_RUL", "mean"))
                 .reindex(order).dropna(how="all").reset_index())
    per_unit = pd.concat(unit_rows, ignore_index=True)
    ho = per_unit[per_unit["split"].isin(["dev", "test"])]
    diag = pd.DataFrame([{
        "n_unit_evals": len(ho),
        "perunit_RMSE_median": ho["rmse_u"].median(),
        "perunit_RMSE_p90": ho["rmse_u"].quantile(0.9),
        "perunit_RMSE_max": ho["rmse_u"].max(),
        "nonmono_frac(reb>0.15)": float((ho["rebound"] > 0.15).mean()),
        "late_bias_mean(+=optimistic)": ho["late_bias"].mean(),
    }])

    print(f"\n=== finalize_multi  age={age}  seeds={tuple(seeds)} ===")
    print(summary.round(3).to_string(index=False))
    print("\nheld-out per-unit diagnostics:")
    print(diag.round(3).to_string(index=False))

    thr, blind, blind_by_ds = _blind_report(ho, q=0.9)
    if len(blind_by_ds):
        rest = ho[ho["late_bias"] < thr]
        print(f"\nblind units (late_bias >= p90 = {thr:.2f}) by fault mode "
              f"-- leaves under-read degradation (peak_* = mean peak spool damage):")
        print(blind_by_ds.round(3).to_string(index=False))
        print(f"  mean peak_damage: blind={blind['peak_damage'].mean():.3f}  "
              f"rest={rest['peak_damage'].mean():.3f}  "
              f"(flat spools => the leaves never drove damage up)")

    worst = None
    if debug_worst and len(ho):
        w = ho.sort_values("rmse_u").iloc[-1]
        worst = {"unit": w["unit"], "seed": int(w["seed"]), "split": w["split"],
                 "rmse_u": float(w["rmse_u"]), "rebound": float(w["rebound"]),
                 "late_bias": float(w["late_bias"])}
        fu = fr[fr["unit"] == w["unit"]].reset_index(drop=True)
        plot_unit_debug(models[int(w["seed"])], fu, w["unit"],
                        os.path.join(outdir, f"worst_unit_age{int(age)}.png"))
        print(f"\nworst held-out unit: {worst}  -> {outdir}/worst_unit_age{int(age)}.png")
    return {"summary": summary, "per_unit": per_unit, "diag": diag,
            "blind": blind, "blind_by_ds": blind_by_ds, "worst": worst}


# --------------------------------------------------------------------------
# self-test (synthetic; not run by the assistant unless invoked)
# --------------------------------------------------------------------------

CHOSEN = {
    "HPT": [("hpt_eff",  "HPT_eff_mod",  ["T48", "T30", "Nc"])],
    "HPC": [("hpc_eff",  "HPC_eff_mod",  ["Ps30", "T30", "Nc"]),
            ("hpc_flow", "HPC_flow_mod", ["P24", "T30", "Nc"])],
    "fan": [("fan_eff",  "fan_eff_mod",  ["T24", "P24", "Nf"])],
    "LPC": [("lpc_eff",  "LPC_eff_mod",  ["P24", "T24", "Nf"])],
    "LPT": [("lpt_eff",  "LPT_eff_mod",  ["T50", "Nf", "Nc"]),
            ("lpt_flow", "LPT_flow_mod", ["T50", "P50", "P24"])],
}


def _self_test():
    import ablation, os
    pooled, _ = ablation._synth_fleet()
    fr = pooled[pooled["ds"] == "DS08"].reset_index(drop=True)

    # small budgets for the smoke test
    model, frozen, res = run(fr, CHOSEN, leaf_gens=30, leaf_pop=16, gens=20, pop=16,
                             json_path="/tmp/frozen_leaves.json", outdir="/tmp/ff")
    assert len(res["leaf_fidelity"]) == 7
    assert os.path.exists("/tmp/frozen_leaves.json")
    assert os.path.exists("/tmp/ff/frozen_leaves.png")
    assert {"RMSE", "NASA", "R2", "rho_RUL"} <= set(res["rul"].keys())

    # JSON round-trip: reload frozen leaves into a fresh model and check they match
    m2, _ = freeze(fr, CHOSEN, leaf_gens=1, leaf_pop=4, seed=0)   # skeleton (cheap)
    frozen2 = load_frozen(m2, "/tmp/frozen_leaves.json")
    for n in m2["meta"]:
        if n["kind"] == "leaf":
            assert np.allclose(frozen2[n["rules"]], frozen[n["rules"]]), n["target"]
    # dev/test finalization + RUL plots
    print("\n-- finalize (held-out dev/test) --")
    _, _, tab = finalize(pooled, CHOSEN, ["DS08"], leaf_gens=20, leaf_pop=12,
                         gens=15, pop=12, json_path="/tmp/frozen_leaves.json",
                         outdir="/tmp/ff", also_eval_ds=["DS01"])
    assert set(tab["split"]) >= {"train", "dev", "test"}
    assert os.path.exists("/tmp/ff/rul_dev.png") and os.path.exists("/tmp/ff/rul_test.png")

    # multi-seed finalization + per-unit diagnostics + worst-unit debug plot
    print("\n-- finalize_multi (2 seeds) --")
    res = finalize_multi(pooled, CHOSEN, ["DS08"], seeds=(0, 1), age=False,
                         leaf_gens=15, leaf_pop=10, gens=12, pop=10,
                         outdir="/tmp/ff", also_eval_ds=["DS01"])
    assert {"summary", "per_unit", "diag", "worst"} <= set(res)
    assert {"rmse_u", "rebound", "late_bias", "ds", "peak_damage"} <= set(res["per_unit"].columns)
    assert "blind_by_ds" in res
    assert set(res["summary"]["split"]) >= {"train", "dev", "test"}
    if res["worst"]:
        assert os.path.exists("/tmp/ff/worst_unit_age0.png")

    print("\nfreeze_fit self-test OK (round-trip + dev/test + multi-seed verified)")


if __name__ == "__main__":
    _self_test()