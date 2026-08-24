"""
leaf_train.py -- standalone (leaf-only) training sweep to separate UNDER-TRAINING from
an OBSERVABILITY CEILING, with predicted-vs-real graphs.

Leaf-only: fits each leaf's sensors -> its OWN theta, NOT in the tree and NOT on RUL.
Mirrors gft._freeze_leaves exactly (quantile MF centres + GA on the rule consequents,
same loss, same bounds, same ga.optimize) but exposes a CAPACITY knob k = number of MFs
per input (rule grid = k**n_sensors). Sweeping k on HELD-OUT units answers the question:

    rho_out rises with k   -> under-trained / capacity-limited (fixable)
    rho_out flat with k     -> observability ceiling (more training won't help)

Graphs: (1) capacity curve rho_out vs k per leaf; (2) predicted-vs-real theta over cycle
per leaf at its best k, on held-out units (the trend view, curve complexity visible).

Only gft + ga + data public primitives. Leaf-only by construction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import gft, ga, data


def _grid(k):
    cols = int(np.ceil(np.sqrt(k))); rows = int(np.ceil(k / cols)); return rows, cols


def _rho_active(units, y, yh, eps=1e-6):
    """Per-unit rho over ONLY units whose target varies -- the rare-mode-fair view
    (flat units score 0 and dilute the plain mean; matters for the fan)."""
    u = np.asarray(units); y = np.asarray(y, float)
    span = pd.Series(y).groupby(u).transform(lambda v: v.max() - v.min()).to_numpy()
    m = np.abs(span) > eps
    return gft.unit_corr(u[m], y[m], np.asarray(yh, float)[m]) if m.any() else np.nan


def _fit_leaf(train, sensors, target, k, gens, pop, seed):
    """One leaf, standalone: sensors -> target. Fixed quantile MF centres (n=k), GA on
    the k**d rule consequents. Identical recipe to gft._freeze_leaves' per-leaf step."""
    P = gft.prep(train, sensors)
    cols = [P[s].to_numpy() for s in sensors]
    centres = [gft._quantile_centres(P[s].to_numpy(), n=k) for s in sensors]
    y = train[target].to_numpy(float)
    n_rules = int(np.prod([len(c) for c in centres]))
    span = float(y.max() - y.min()) or 1.0
    lo = np.full(n_rules, y.min() - 0.25 * span)
    hi = np.full(n_rules, y.max() + 0.25 * span)
    loss = lambda g: gft.rmse(y, gft._fis(cols, centres, g))
    g, _, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed)
    return {"sensors": list(sensors), "target": target, "k": k,
            "centres": centres, "genome": g}


def _predict(model, frame):
    P = gft.prep(frame, model["sensors"])
    cols = [P[s].to_numpy() for s in model["sensors"]]
    return gft._fis(cols, model["centres"], model["genome"])


def _leaf_iter(leaves):
    for comp, entries in leaves.items():
        for (name, target, sensors) in entries:
            yield target, list(sensors)


def sweep(pooled, leaves, train_ds, ks=(2, 3, 4, 5), seeds=(0, 1, 2, 3, 4),
          gens=150, pop=60, verbose=True):
    """Per (leaf, k): mean held-out rho / rho_active / R2 over unit-disjoint seed splits."""
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    rows = []
    for target, sensors in _leaf_iter(leaves):
        for k in ks:
            ro, ra, r2 = [], [], []
            for s in seeds:
                tr, va, _ = data.split(fr, seed=s)
                m = _fit_leaf(tr, sensors, target, k, gens, pop, s)
                yh = _predict(m, va)
                y = va[target].to_numpy(float); u = va["unit"].to_numpy()
                ro.append(gft.unit_corr(u, y, yh)); ra.append(_rho_active(u, y, yh))
                r2.append(gft.r2(y, yh))
            rows.append({"target": target, "sensors": "+".join(sensors), "k": k,
                         "n_rules": k ** len(sensors),
                         "rho_out": np.nanmean(ro), "rho_out_sd": np.nanstd(ro),
                         "rho_active_out": np.nanmean(ra), "R2_out": np.nanmean(r2)})
            if verbose:
                print(f"  {target:14s} k={k} rules={k**len(sensors):3d}  "
                      f"rho_out={np.nanmean(ro):.3f}  rho_active={np.nanmean(ra):.3f}")
    return pd.DataFrame(rows)


def best_k(df):
    """{target: k with highest rho_out}."""
    idx = df.groupby("target")["rho_out"].idxmax()
    return {df.loc[i, "target"]: int(df.loc[i, "k"]) for i in idx}


def plot_capacity(df, save):
    """rho_out vs k, one line per leaf. Rising = under-trained; flat = ceiling."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for tgt, g in df.groupby("target"):
        g = g.sort_values("k")
        ax.errorbar(g["k"], g["rho_out"], yerr=g["rho_out_sd"], marker="o",
                    capsize=3, label=tgt)
    ax.set_xlabel("MF count per input (k) -- capacity"); ax.set_ylabel("held-out rho")
    ax.set_title("Leaf capacity sweep -- rising = under-trained, flat = ceiling")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=110); plt.close(fig)
    return save


def plot_leaf_fits(pooled, leaves, train_ds, k_map, save, seed=0, gens=150, pop=60):
    """Predicted (blue) vs real (black) theta over cycle, per leaf at its best k, on the
    held-out units of one split. The trend view for analysing curve shape / fit quality."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    tr, va, _ = data.split(fr, seed=seed)
    pairs = list(_leaf_iter(leaves))
    rows_g, cols_g = _grid(len(pairs))
    fig, ax = plt.subplots(rows_g, cols_g, figsize=(3.8 * cols_g, 3.0 * rows_g),
                           squeeze=False)
    uarr = va["unit"].to_numpy(); cyc = va["cycle"].to_numpy()
    uniq = list(dict.fromkeys(uarr))
    for i, (target, sensors) in enumerate(pairs):
        a = ax[i // cols_g][i % cols_g]
        k = int(k_map.get(target, 3)) if isinstance(k_map, dict) else int(k_map)
        m = _fit_leaf(tr, sensors, target, k, gens, pop, seed)
        yh = _predict(m, va); y = va[target].to_numpy(float)
        for u in uniq:
            mu = uarr == u; o = np.argsort(cyc[mu])
            a.plot(cyc[mu][o], y[mu][o], color="k", lw=1.0, alpha=0.55)
            a.plot(cyc[mu][o], yh[mu][o], color="steelblue", lw=1.2, alpha=0.85)
        a.set_title(f"{target}  (k={k})", fontsize=9)
        a.set_xlabel("cycle"); a.set_ylabel("theta")
    for j in range(len(pairs), rows_g * cols_g):
        ax[j // cols_g][j % cols_g].axis("off")
    fig.suptitle("Standalone leaf fits -- black = real theta, blue = predicted (held-out units)")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(save, dpi=110); plt.close(fig)
    return save


def run(pooled, leaves, train_ds, ks=(2, 3, 4, 5), seeds=(0, 1, 2, 3, 4),
        gens=150, pop=60, outdir="figures"):
    """Full sweep + both graphs. Returns the capacity table."""
    import os; os.makedirs(outdir, exist_ok=True)
    print("capacity sweep (held-out over seeds):")
    df = sweep(pooled, leaves, train_ds, ks=ks, seeds=seeds, gens=gens, pop=pop)
    print("\n" + df.round(3).to_string(index=False))
    bk = best_k(df)
    plot_capacity(df, os.path.join(outdir, "leaf_capacity.png"))
    plot_leaf_fits(pooled, leaves, train_ds, bk, os.path.join(outdir, "leaf_fits.png"),
                   seed=seeds[0], gens=gens, pop=pop)
    print("\nbest k per leaf:", bk)
    print(f"wrote {outdir}/leaf_capacity.png and {outdir}/leaf_fits.png")
    return df


def _self_test():
    import copy, ablation, os
    pooled, _ = ablation._synth_fleet()
    leaves = copy.deepcopy(gft.DEFAULT_LEAVES)
    df = run(pooled, leaves, ["DS08"], ks=(2, 3), seeds=(0, 1),
             gens=15, pop=12, outdir="/tmp/lt")
    assert {"target", "k", "rho_out", "rho_active_out", "R2_out"} <= set(df.columns)
    assert len(df) == sum(len(v) for v in leaves.values()) * 2   # leaves x ks
    assert os.path.exists("/tmp/lt/leaf_capacity.png")
    assert os.path.exists("/tmp/lt/leaf_fits.png")
    print("\nleaf_train self-test OK")


if __name__ == "__main__":
    _self_test()