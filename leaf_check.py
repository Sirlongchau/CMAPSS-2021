"""
leaf_check.py -- post-hoc honesty checks on a chosen leaf set, in ONE place.

Does NOT touch the prune. Four functions, all on a fitted decoupled model + a frame:

  fit(pooled, leaves, train_ds)      -> the model you'd freeze (fit on full train)
  check(pooled, leaves, train_ds)    -> per-leaf table: in-sample vs HELD-OUT R2/rho,
                                        and all-points vs deep-TAIL slope (fits its own
                                        unit-disjoint split internally)
  specificity(model, pooled)         -> per-leaf COMPONENT-specificity, confound-free,
                                        from the fleet's fault-mode structure
  plot_trajectories(model, frame,..) -> per-leaf theta(t) vs theta_hat(t) over cycles
                                        (the TREND view; complements the parity plot)

Why two views of "fidelity":
  * parity (pred vs real, diagonal): a CALIBRATION view -- shows scale/offset error, and
    hides trend-following. A leaf with rho 0.9 but R2 0.4 tracks the trend yet compresses
    magnitude, and looks "off the diagonal".
  * trajectories (theta vs cycle): a TREND view -- shows whether theta_hat follows the
    degradation path and how complex the curve is; magnitude error is read as a vertical
    gap, not as "bad correlation".

Why specificity here is NOT leaf_fidelity's retracted metric:
  the old one was rho(theta_hat, theta_own) - rho(theta_hat, theta_foil), confounded
  because within a unit everything is monotone in time so both rhos ~ equal -> ~0 for
  every leaf. This one uses the CROSS-UNIT fault-mode structure instead: a component-
  specific leaf is ACTIVE on units where its component degrades and FLAT on units where
  it is healthy. No within-unit correlation, so no monotone confound.

Only gft + data public API.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import gft
import data


def _grid(k):
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    return rows, cols


def _slope(x, y, min_n=5):
    """OLS slope of y on x; NaN when too few points or x barely varies (the flat-clamp
    case -- left un-scored rather than faked)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < min_n or np.var(x) < 1e-12:
        return np.nan
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def fit(pooled, leaves, train_ds, grouping="shaft", seed=0, age=False, **fit_kw):
    """The decoupled model you'd freeze: fit on the FULL train_ds pool (not a split).
    Use its output for specificity / plot_trajectories. (check() fits its own held-out
    split separately, on purpose.)"""
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    return gft.fit_decoupled(fr, leaves=leaves, grouping=grouping, age=age,
                             seed=seed, **fit_kw)


def check(pooled, leaves, train_ds, grouping="shaft", seed=0, frac=0.5, age=False):
    """Fit ONCE on a unit-disjoint split of train_ds, then per leaf: in-sample vs
    held-out R2/rho, and all-points vs deep-tail slope. Returns one DataFrame."""
    fr = pooled[pooled["ds"].isin(train_ds)].reset_index(drop=True)
    tr, va, _ = data.split(fr, seed=seed)                       # split BY UNIT
    m = gft.fit_decoupled(tr, leaves=leaves, grouping=grouping, age=age, seed=seed)

    rep_in  = gft.evaluate(tr, m)
    rep_out = gft.evaluate(va, m)
    pred = gft.predict_tree(va, m, m["genome"])

    rows = []
    for n in m["meta"]:
        if n["kind"] != "leaf":
            continue
        tgt = n["target"]
        t = va[tgt].to_numpy(float)
        th = pred[tgt + "_hat"].to_numpy()
        tail = t <= frac * float(t.min())
        rows.append({
            "target": tgt,
            "R2_in":  rep_in.get(f"{tgt}:R2"),   "R2_out":  rep_out.get(f"{tgt}:R2"),
            "rho_in": rep_in.get(f"{tgt}:rho"),  "rho_out": rep_out.get(f"{tgt}:rho"),
            "slope_all":  _slope(t, th),
            "slope_tail": _slope(t[tail], th[tail]),
            "R2_tail":    gft.r2(t[tail], th[tail]) if tail.sum() >= 5 else np.nan,
            "n_tail":     int(tail.sum()),
        })
    return pd.DataFrame(rows)


def specificity(model, pooled, eps=1e-6):
    """Confound-free component-specificity of the FROZEN leaves, from fault-mode
    structure. For each leaf, compare theta_hat's within-unit amplitude on units where
    its OWN component degrades vs units where it is HEALTHY:

        specificity = 1 - amp_healthy / amp_degrading      (1 = specific, 0 = global damage)

    A specific leaf is flat when its component is healthy. Pass the FULL pooled fleet
    (needs fault-mode diversity: units where the component does NOT degrade). Leaves with
    no healthy units get specificity NaN (n_heal reported)."""
    pred = gft.predict_tree(pooled, model, model["genome"])
    units = pooled["unit"].to_numpy()
    uniq = np.unique(units)
    rows = []
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        tgt = n["target"]
        th = pred[tgt + "_hat"].to_numpy()
        y = pooled[tgt].to_numpy(float)
        amp_deg, amp_heal = [], []
        for u in uniq:
            mu = units == u
            own_varies = (y[mu].max() - y[mu].min()) > eps
            amp = float(np.std(th[mu]))
            (amp_deg if own_varies else amp_heal).append(amp)
        ad = float(np.mean(amp_deg)) if amp_deg else np.nan
        ah = float(np.mean(amp_heal)) if amp_heal else np.nan
        spec = (1.0 - ah / ad) if (amp_deg and amp_heal and ad > eps) else np.nan
        rows.append({"target": tgt, "amp_degrading": ad, "amp_healthy": ah,
                     "specificity": spec, "n_deg": len(amp_deg), "n_heal": len(amp_heal)})
    return pd.DataFrame(rows)


def plot_trajectories(model, frame, save, n_units=6, seed=0):
    """Per leaf, overlay real theta(t) (black) and predicted theta_hat(t) (blue) against
    cycle, for a spread of units -- the TREND view. Reveals trend-following and curve
    shape that the pred-vs-real parity plot conflates with scale error. Writes a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = gft.predict_tree(frame, model, model["genome"])
    leaves = [n for n in model["meta"] if n["kind"] == "leaf"]
    uniq = list(dict.fromkeys(frame["unit"].to_numpy()))
    rng = np.random.default_rng(seed)
    sel = list(rng.choice(uniq, size=min(n_units, len(uniq)), replace=False))

    unit_arr = frame["unit"].to_numpy()
    cyc = frame["cycle"].to_numpy()
    rows_g, cols_g = _grid(len(leaves))
    fig, ax = plt.subplots(rows_g, cols_g, figsize=(3.8 * cols_g, 3.0 * rows_g),
                           squeeze=False)
    for i, n in enumerate(leaves):
        a = ax[i // cols_g][i % cols_g]
        tgt = n["target"]
        th = pred[tgt + "_hat"].to_numpy()
        y = frame[tgt].to_numpy(float)
        for u in sel:
            mu = unit_arr == u
            o = np.argsort(cyc[mu])
            a.plot(cyc[mu][o], y[mu][o], color="k", lw=1.0, alpha=0.55)
            a.plot(cyc[mu][o], th[mu][o], color="steelblue", lw=1.2, alpha=0.85)
        a.set_title(tgt, fontsize=9)
        a.set_xlabel("cycle"); a.set_ylabel("theta")
    for j in range(len(leaves), rows_g * cols_g):
        ax[j // cols_g][j % cols_g].axis("off")
    fig.suptitle("Leaf trajectories -- black = real theta, blue = predicted "
                 f"({len(sel)} units vs cycle)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save


def _self_test():
    import copy, ablation
    pooled, _ = ablation._synth_fleet()
    leaves = copy.deepcopy(gft.DEFAULT_LEAVES)

    print("== check (held-out + tail) ==")
    df = check(pooled, leaves, ["DS08"], seed=0)
    print(df.round(3).to_string(index=False))
    assert {"R2_in", "R2_out", "slope_all", "slope_tail", "n_tail"} <= set(df.columns)
    assert len(df) == sum(len(v) for v in leaves.values())

    print("\n== fit on full DS08, then specificity on FULL pooled ==")
    m = fit(pooled, leaves, ["DS08"], seed=0, gens=12, pop=16, leaf_gens=8, leaf_pop=8)
    sp = specificity(m, pooled)
    print(sp.round(3).to_string(index=False))
    assert {"specificity", "amp_degrading", "amp_healthy"} <= set(sp.columns)
    # synthetic leaves ARE component-specific -> specificity should be clearly positive,
    # unlike the retracted output-metric which returned ~0 for every leaf.
    assert (sp["specificity"].dropna() > 0.2).all(), sp[["target", "specificity"]]

    print("\n== plot_trajectories ==")
    p = plot_trajectories(m, pooled[pooled["ds"] == "DS08"], "/tmp/leaf_traj.png")
    import os
    assert os.path.exists(p)
    print("wrote", p)
    print("\nleaf_check self-test OK")


if __name__ == "__main__":
    _self_test()