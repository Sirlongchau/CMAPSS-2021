"""
viz.py -- figures for the ablation, and only the ablation. Everything here is
driven off the dict returned by ablation.run_ablation (plus the pooled frame), so
the plots always match the run that produced them -- no separate fit.

Panels:
  leaf_search   : per leaf, theta-rho across candidate sensor sets (the Stage-1
                  sensor selection), coloured by cross-mode leak.
  branch_rul    : per single-fault mode, does the branch ALONE predict RUL.
  leak          : per mode in the ASSEMBLED tree, own spool vs other spool peak
                  damage -- the leak picture that drives trimming.
  assembled     : per single-fault mode, spool-damage trajectories + RUL parity.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gft
from features import SHAFT


def _grid(n, cols=3):
    return (int(np.ceil(n / cols)), min(n, cols))


def plot_leaf_search(search, save):
    """One panel per leaf: theta_rho for each candidate sensor set (Stage 1)."""
    leaves = list(search.groupby(["component", "leaf"]))
    rows, cols = _grid(len(leaves))
    fig, ax = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for i, ((comp, leaf), sub) in enumerate(leaves):
        a = ax[i // cols][i % cols]
        sub = sub.sort_values("theta_rho")
        leak = sub["leak"].fillna(0).to_numpy()
        colors = plt.cm.RdYlGn_r(np.clip(leak, 0, 1)) if np.isfinite(leak).any() \
            else "steelblue"
        a.barh(range(len(sub)), sub["theta_rho"], color=colors)
        a.set_yticks(range(len(sub)))
        a.set_yticklabels(sub["sensors"], fontsize=7)
        a.set_xlim(0, 1)
        a.set_title(f"{comp}.{leaf}  (theta rho)", fontsize=9)
        a.axvline(sub["theta_rho"].max(), ls="--", c="k", lw=0.8)
    for j in range(len(leaves), rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.suptitle("Stage 1 -- leaf sensor search (bar = theta rho, colour = leak; "
                 "dashed = winner)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save


def plot_branch_rul(brep, save):
    """Per mode: branch-alone RUL RMSE and NASA (does one branch predict RUL?)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(brep))
    ax[0].bar(x, brep["branch_RMSE"], color="steelblue")
    ax[0].set_xticks(x); ax[0].set_xticklabels(brep["ds"], rotation=45, ha="right")
    ax[0].set_ylabel("branch RUL RMSE"); ax[0].set_title("branch-alone RMSE")
    ax[1].bar(x, brep["branch_NASA"], color="indianred")
    ax[1].set_xticks(x); ax[1].set_xticklabels(brep["ds"], rotation=45, ha="right")
    ax[1].set_ylabel("branch RUL NASA"); ax[1].set_title("branch-alone NASA")
    fig.suptitle("Stage 1 -- does a single fault mode's branch predict its own RUL?")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save


def plot_leak(leaks, save):
    """Per single-fault mode: own spool peak vs the other spool's peak in the
    ASSEMBLED tree. A tall 'other' bar (or a flat 'own' bar) is the leak."""
    if leaks.empty:
        return None
    spools = [c[:-5] for c in leaks.columns if c.endswith("_peak")]
    x = np.arange(len(leaks))
    w = 0.8 / max(len(spools), 1)
    fig, a = plt.subplots(figsize=(1.6 * len(leaks) + 3, 3.6))
    for j, sp in enumerate(spools):
        a.bar(x + j * w, leaks[f"{sp}_peak"], w, label=f"{sp} peak")
    for i, r in leaks.reset_index(drop=True).iterrows():
        a.annotate("own", (i + spools.index(r["own_spool"]) * w, 0.02),
                   ha="center", fontsize=7, color="k")
        if r.get("leak", False):
            a.text(i + 0.3, max(r[f"{s}_peak"] for s in spools) + 0.03, "LEAK",
                   color="firebrick", fontweight="bold", fontsize=8)
    a.set_xticks(x + w * (len(spools) - 1) / 2)
    a.set_xticklabels([f"{r['ds']}\n{r['mode']}" for _, r in leaks.iterrows()])
    a.set_ylabel("peak latent damage"); a.legend(fontsize=8)
    a.set_title("Stage 3 -- which spool lights up per single-fault mode (leak = wrong spool)")
    fig.tight_layout()
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save


def plot_assembled(full_model, pooled, mode_components, save, max_units=2):
    """Per single-fault mode: spool-damage trajectories (a few units) and RUL
    parity, from the assembled tree. Shows attribution: does the OWN spool climb?"""
    modes = [(ds, c[0]) for ds, c in mode_components.items() if len(c) == 1]
    rows = len(modes)
    fig, ax = plt.subplots(rows, 2, figsize=(9, 2.6 * rows), squeeze=False)
    spools = [n["name"] for n in full_model["meta"] if n["kind"] == "spool"]
    for i, (ds, mode) in enumerate(modes):
        fr = pooled[pooled["ds"] == ds]
        pred = gft.predict_tree(fr, full_model, full_model["genome"])
        m = fr[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
        own = "hp" if SHAFT[mode] == "HP" else "lp"
        a0 = ax[i][0]
        for u in list(np.unique(m["unit"]))[:max_units]:
            d = m[m["unit"] == u].sort_values("cycle")
            for sp in spools:
                a0.plot(d["cycle"], d[sp + "_h"],
                        lw=2 if sp == own else 1.2,
                        ls="-" if sp == own else "--",
                        label=f"{sp}{' (own)' if sp == own else ''}" if u == m['unit'].iloc[0] else None)
        a0.set_title(f"{ds} ({mode}) -- spool damage", fontsize=9)
        a0.set_ylim(-0.02, 1.02); a0.set_xlabel("cycle"); a0.legend(fontsize=7)
        a1 = ax[i][1]
        cp = gft.cap(m["RUL"], full_model["rul_cap"])
        a1.scatter(cp, m["RUL_hat"], s=8, alpha=0.5)
        hi = float(np.max(cp)) * 1.05 + 1e-6
        a1.plot([0, hi], [0, hi], "k--", lw=1)
        a1.set_xlim(0, hi); a1.set_ylim(0, hi)
        a1.set_title(f"{ds} -- RUL parity (RMSE {gft.rmse(cp, m['RUL_hat']):.1f})",
                     fontsize=9)
        a1.set_xlabel("true RUL"); a1.set_ylabel("pred RUL")
    fig.suptitle("Assembled tree -- attribution (own spool solid) and RUL fit")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save


def visualize(out, pooled, mode_components, outdir="figures"):
    """Render every ablation figure into `outdir`. Returns the list of files."""
    os.makedirs(outdir, exist_ok=True)
    made = []
    made.append(plot_leaf_search(out["search"], os.path.join(outdir, "leaf_search.png")))
    made.append(plot_branch_rul(out["branch_report"], os.path.join(outdir, "branch_rul.png")))
    lk = plot_leak(out["leaks"], os.path.join(outdir, "leak.png"))
    if lk:
        made.append(lk)
    made.append(plot_assembled(out["assembled"], pooled, mode_components,
                               os.path.join(outdir, "assembled.png")))
    print("wrote:", *[os.path.basename(m) for m in made])
    return made


def _self_test():
    import ablation
    pooled, modes = ablation._synth_fleet()
    curriculum = {ds: c for ds, c in modes.items() if ds != "DS08"}
    out = ablation.run_ablation(pooled, curriculum, train_ds=["DS08"],
                                gens=15, pop=20, seed=0, do_trim=False)
    made = visualize(out, pooled, curriculum, outdir="/tmp/gftree_figs")
    assert len(made) >= 3, made
    print("viz self-test OK")


if __name__ == "__main__":
    _self_test()


def plot_pruning(prune_out, save):
    """The pruning trajectory (RUL RMSE and min-leaf-R2 vs leaf count) plus the
    (RMSE, min-leaf-R2) Pareto scatter -- the figure that shows redundancy: leaves
    whose removal improves BOTH axes are redundant passengers."""
    traj = prune_out["trajectory"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    x = traj["size"].to_numpy()
    a.errorbar(x, traj["RMSE"], yerr=traj["RMSE_sd"].fillna(0), marker="o",
               color="steelblue", label="RUL RMSE")
    a.set_xlabel("leaves kept"); a.set_ylabel("RUL RMSE", color="steelblue")
    a.invert_xaxis()
    a2 = a.twinx()
    a2.errorbar(x, traj["min_leaf_R2"], yerr=traj["min_leaf_R2_sd"].fillna(0),
                marker="s", color="firebrick", label="min leaf R2")
    a2.axhline(0, color="firebrick", ls=":", lw=0.8)
    a2.set_ylabel("min leaf theta-R2", color="firebrick")
    rec = prune_out["recommended"]
    a.axvline(rec["size"], color="green", ls="--", lw=1)
    a.set_title(f"elimination trajectory (recommend {rec['size']} leaves)")
    for _, r in traj.iloc[1:].iterrows():
        a.annotate(r["dropped"].split(".")[-1], (r["size"], r["RMSE"]),
                   fontsize=6, rotation=30, ha="right")
    ap = prune_out["all_points"]
    b.scatter(ap["RMSE"], ap["min_leaf_R2"], s=14, c="0.7", label="evaluated")
    pf = prune_out["pareto"].sort_values("RMSE")
    b.plot(pf["RMSE"], pf["min_leaf_R2"], "-o", color="darkgreen", label="Pareto")
    b.axhline(0, color="k", ls=":", lw=0.8)
    b.set_xlabel("RUL RMSE (lower better)")
    b.set_ylabel("min leaf theta-R2 (higher better)")
    b.set_title("(RUL, leaf-fidelity) Pareto front"); b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save