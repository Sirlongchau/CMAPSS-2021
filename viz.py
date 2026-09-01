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
        own = gft.owner_spool(full_model, mode)
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


def plot_theta_fit(model, frame, save, max_units=25):
    """Per leaf: predicted theta vs real theta (parity), the DIAGNOSTIC-fidelity
    figure. A tight diagonal = the leaf reads its component; scatter/negative slope
    = it does not. R2 and per-unit rho annotated. Complements plot_assembled, which
    shows the aggregation (spool damage + RUL); this shows the leaves themselves."""
    pred = gft.predict_tree(frame, model, model["genome"])
    leaves = [n for n in model["meta"] if n["kind"] == "leaf"]
    rows, cols = _grid(len(leaves))
    fig, ax = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    units = frame["unit"].to_numpy()
    keep = set(list(dict.fromkeys(units))[:max_units])   # thin for legibility
    m = np.array([u in keep for u in units])
    for i, n in enumerate(leaves):
        a = ax[i // cols][i % cols]
        t = frame[n["target"]].to_numpy(float)
        th = pred[n["target"] + "_hat"].to_numpy()
        a.scatter(t[m], th[m], s=7, alpha=0.35, c="steelblue", edgecolors="none")
        lo = float(min(t.min(), th.min())); hi = float(max(t.max(), th.max()))
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        r2 = gft.r2(t, th); rho = gft.unit_corr(units, t, th)
        a.set_title(f"{n['target']}\nR2={r2:.2f}  rho={rho:.2f}", fontsize=9)
        a.set_xlabel("real theta"); a.set_ylabel("pred theta")
    for j in range(len(leaves), rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.suptitle("Leaf diagnostic fidelity -- predicted vs real theta (dashed = y=x)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, dpi=110)
    plt.close(fig)
    return save

# ==========================================================================
# control surfaces of every FIS + per-unit test diagnostics (reused by export_all)
# ==========================================================================

def _out_label(n):
    return ({"leaf": n["target"], "component": n["name"] + "_dmg",
             "spool": n["name"] + "_h", "root": "RUL"})[n["kind"]]


def _input_domains(n, P, frame):
    """(lo, hi) per input, in the space that FIS node consumes: prepped sensors for leaves,
    theta-hat range for components, [0,1] damage for spools/root, age range for age."""
    doms = []
    for j, s in enumerate(n["inputs"]):
        if n["kind"] == "leaf":
            v = P[s].to_numpy()
            doms.append((float(np.percentile(v, 1)), float(np.percentile(v, 99))))
        elif s == "age":
            v = frame["age"].to_numpy(float); doms.append((float(v.min()), float(v.max())))
        elif n["kind"] == "component":
            c = np.asarray(n["centres"][j]); doms.append((float(c.min()) * 1.15, 0.0))
        else:
            doms.append((0.0, 1.0))
    return doms


def plot_control_surfaces(model, frame, outdir, res=45):
    """Control surface of every FIS, grouped by node kind (one figure per kind). 1-input ->
    1D curve; >=2 inputs -> 2D surface over the first two inputs, any others at their median."""
    P = gft.prep(frame, model["sensors"])
    by_kind = {}
    for n in model["meta"]:
        by_kind.setdefault(n["kind"], []).append(n)
    saved = []
    for kind, nodes in by_kind.items():
        rows_g, cols_g = _grid(len(nodes))
        fig, ax = plt.subplots(rows_g, cols_g, figsize=(4.2 * cols_g, 3.4 * rows_g), squeeze=False)
        for i, n in enumerate(nodes):
            a = ax[i // cols_g][i % cols_g]
            cons = gft._consequents(n, model["genome"])
            doms = _input_domains(n, P, frame); d = len(n["inputs"])
            if d == 1:
                x = np.linspace(*doms[0], 200)
                a.plot(x, gft._fis([x], n["centres"], cons), color="steelblue")
                a.set_xlabel(n["inputs"][0]); a.set_ylabel(_out_label(n))
                a.set_title(f"{n['name']} -> {_out_label(n)}", fontsize=9)
            else:
                x0 = np.linspace(*doms[0], res); x1 = np.linspace(*doms[1], res)
                X0, X1 = np.meshgrid(x0, x1)
                cols = []
                for j in range(d):
                    cols.append(X0.ravel() if j == 0 else X1.ravel() if j == 1
                                else np.full(X0.size, float(np.mean(doms[j]))))
                Z = gft._fis(cols, n["centres"], cons).reshape(X0.shape)
                cf = a.contourf(X0, X1, Z, levels=20, cmap="viridis")
                fig.colorbar(cf, ax=a, shrink=0.85)
                a.set_xlabel(n["inputs"][0]); a.set_ylabel(n["inputs"][1])
                a.set_title(f"{n['name']} -> {_out_label(n)}" + (" (slice)" if d > 2 else ""),
                            fontsize=8)
        for j in range(len(nodes), rows_g * cols_g):
            ax[j // cols_g][j % cols_g].axis("off")
        fig.suptitle(f"Control surfaces -- {kind} FISes")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(outdir, f"control_surfaces_{kind}.png")
        fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    return saved


def plot_test_units(model, frame, outdir, max_units=6, seed=0):
    """One figure per unit: RUL true vs pred, spool damage, per-component damage D_c, and a
    small-multiple grid of each leaf's theta true vs predicted -- all over cycle."""
    from matplotlib.gridspec import GridSpec
    pred = gft.predict_tree(frame, model, model["genome"])
    leaves = [n["target"] for n in model["meta"] if n["kind"] == "leaf"]
    comps = [n["name"] for n in model["meta"] if n["kind"] == "component"]
    spools = [n["name"] for n in model["meta"] if n["kind"] == "spool"]
    keep = ["unit", "cycle", "RUL_hat"] + [t + "_hat" for t in leaves] \
        + [c + "_dmg" for c in comps] + [s + "_h" for s in spools]
    m = frame[["unit", "cycle", "RUL"] + leaves].merge(pred[keep], on=["unit", "cycle"])
    uniq = list(dict.fromkeys(m["unit"].to_numpy()))
    rng = np.random.default_rng(seed)
    sel = list(rng.choice(uniq, size=min(max_units, len(uniq)), replace=False))
    nL, tcols = len(leaves), 4
    trows = int(np.ceil(nL / tcols))
    saved = []
    for u in sel:
        d = m[m["unit"] == u].sort_values("cycle"); c = d["cycle"].to_numpy()
        fig = plt.figure(figsize=(4.6 * 3, 3.0 * (1 + trows)))
        gs = GridSpec(1 + trows, tcols, figure=fig)
        aR, aS, aC = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
        aR.plot(c, gft.cap(d["RUL"].to_numpy(float), model["rul_cap"]), "k", lw=1.4, label="true")
        aR.plot(c, d["RUL_hat"], "steelblue", lw=1.6, label="pred")
        aR.set_title("RUL"); aR.set_xlabel("cycle"); aR.legend(fontsize=8)
        for s in spools:
            aS.plot(c, d[s + "_h"], lw=1.4, label=s)
        aS.set_title("spool damage"); aS.set_xlabel("cycle"); aS.set_ylim(-0.02, 1.02); aS.legend(fontsize=8)
        for cc in comps:
            aC.plot(c, d[cc + "_dmg"], lw=1.2, label=cc)
        aC.set_title("component damage $D_c$"); aC.set_xlabel("cycle"); aC.set_ylim(-0.02, 1.02)
        aC.legend(fontsize=7)
        for k, tg in enumerate(leaves):
            a = fig.add_subplot(gs[1 + k // tcols, k % tcols])
            a.plot(c, d[tg].to_numpy(float), "k", lw=1.0)
            a.plot(c, d[tg + "_hat"], "steelblue", lw=1.0)
            a.set_title(tg, fontsize=8); a.set_xlabel("cycle")
        fig.suptitle(f"Unit {u} -- test (black=true, blue=pred)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(outdir, f"unit_{u}.png"); fig.savefig(p, dpi=105); plt.close(fig); saved.append(p)
    return saved