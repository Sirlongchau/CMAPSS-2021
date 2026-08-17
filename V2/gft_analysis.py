r"""
gft_analysis.py -- read predictions from predict_gft() and draw diagnostics.

    print_metrics       : per-unit RUL RMSE (+ NASA score).
    plot_theta          : predicted vs actual theta -- ONE FIGURE PER modifier.
    plot_nodes          : latent spool health (*_h) vs cycle.
    plot_rul            : RUL vs cycle, per unit, real vs predicted.
    plot_scatter        : predicted vs true RUL with the y=x line.
    plot_fitness        : GA best-loss per generation.
    plot_surfaces       : learned control surface -- ONE FIGURE PER sub-FIS.
    analyze             : do all of the above in one call.
    save_figures        : write every figure to a folder (created if needed,
                          existing .png replaced).

Each node is {name, inputs, n_rules, target, root}; a genome slice per node
holds its singletons. Surfaces are evaluated with the same maths as gft._fis,
with sensor axes shown in raw units and theta/[0,1] node axes in their own.
"""

from __future__ import annotations
import os
import glob
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import gft

_PAL = [f"C{i}" for i in range(20)]


def _color(u):
    return _PAL[(int(u) - 1) % len(_PAL)]


def _grid(n):
    cols = min(int(np.floor(n ** 0.5)), 3) or 1
    return int(np.ceil(n / cols)), cols


def _slice(meta, genome, node):
    """This node's rule consequents, decoded. For a monotone node the genome
    stores [base | steps]; gft.node_singletons rebuilds the actual grid so the
    control surface reflects what the model computes."""
    return gft.node_singletons(node, genome)


def _finish(fig, save, name, show):
    if save:
        p = f"{save}{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("saved:", p)
    if show:
        plt.show()
    plt.close(fig)


def _prepare_dir(folder, clear=True):
    """Create `folder` if missing; wipe existing .png so a rerun replaces them.
    Returns a filename prefix ('folder/') for _finish."""
    os.makedirs(folder, exist_ok=True)
    if clear:
        for p in glob.glob(os.path.join(folder, "*.png")):
            os.remove(p)
    return os.path.join(folder, "")


# ==========================================================================
# metrics
# ==========================================================================

def print_metrics(feats, pred):
    m = feats[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
    rows = []
    for u in np.unique(m["unit"]):
        d = m[m["unit"] == u]
        rows.append({"unit": int(u), "n": len(d),
                     "RUL_RMSE": round(gft.rmse(d["RUL"], d["RUL_hat"]), 2),
                     "NASA": round(gft.nasa_score(d["RUL"], d["RUL_hat"]), 3)})
    df = pd.DataFrame(rows)
    df.loc["ALL"] = {"unit": "ALL", "n": len(m),
                     "RUL_RMSE": round(gft.rmse(m["RUL"], m["RUL_hat"]), 2),
                     "NASA": round(gft.nasa_score(m["RUL"], m["RUL_hat"]), 3)}
    print(df.to_string(index=False))
    return df


# ==========================================================================
# theta: predicted vs actual, ONE FIGURE PER modifier
# ==========================================================================

def plot_theta(feats, pred, size=3.4, save=None, show=True):
    """For every predicted theta (a '*_hat' column that matches a truth column
    in `feats`) draw its own figure, faceted BY UNIT: one panel per unit with
    real (solid) vs predicted (dashed). Previously every unit was overlaid on a
    single axis, which hid per-unit behaviour on the pooled fleet.

    A unit whose TRUE modifier is identically zero never degraded that component
    in its failure mode (e.g. a fan-only unit -- the tree has no leaf for the
    fan, so those units' theta targets are all 0). Such panels are shaded and
    flagged, so a flat-zero 'fit' is not mistaken for a good one."""
    mods = [c[:-4] for c in pred.columns
            if c.endswith("_hat") and c != "RUL_hat" and c[:-4] in feats.columns]
    for mod in mods:
        m = feats[["unit", "cycle", mod]].merge(
            pred[["unit", "cycle", mod + "_hat"]], on=["unit", "cycle"])
        units = np.unique(m["unit"])
        rows, cols = _grid(len(units))
        fig = plt.figure(figsize=(size * cols, max(size, rows * size * 0.85)))
        gs = gridspec.GridSpec(rows, cols)
        for n, u in enumerate(units):
            ax = fig.add_subplot(gs[n])
            d = m[m["unit"] == u].sort_values("cycle")
            ax.plot(d["cycle"], d[mod], "-", color=_color(u), lw=1.8, label="real")
            ax.plot(d["cycle"], d[mod + "_hat"], "--", color=_color(u), lw=1.4,
                    label="pred")
            if float(np.abs(d[mod]).max()) < 1e-6:      # component not degraded here
                ax.set_facecolor("0.94")
                ax.text(0.5, 0.5, "θ ≡ 0\n(not degraded)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="0.45")
            ax.set_title(f"Unit {int(u)}", fontsize=9)
            ax.set_xlabel("cycle", fontsize=8); ax.set_ylabel(mod, fontsize=8)
            ax.tick_params(labelsize=7)
            if n == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{mod}: real (—) vs predicted (- -)   "
                     f"overall RMSE={gft.rmse(m[mod], m[mod + '_hat']):.4f}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _finish(fig, save, f"theta_{mod}", show)


def plot_nodes(pred, size=3.4, save=None, show=True):
    """Latent spool health (*_h) vs cycle, faceted BY UNIT: one panel per unit,
    every latent node drawn as its own line, so a single engine's hp/lp damage
    curves are read together. Previously all units were overlaid on one axis per
    node, which made per-unit trajectories unreadable on the pooled fleet."""
    nodes = [c for c in pred.columns if c.endswith("_h")]
    if not nodes:
        return
    units = np.unique(pred["unit"])
    blind = _blind_units(pred)
    rows, cols = _grid(len(units))
    node_colors = {c: f"C{j}" for j, c in enumerate(nodes)}
    fig = plt.figure(figsize=(size * cols, max(size, rows * size * 0.85)))
    gs = gridspec.GridSpec(rows, cols)
    for n, u in enumerate(units):
        ax = fig.add_subplot(gs[n])
        d = pred[pred["unit"] == u].sort_values("cycle")
        for col in nodes:
            ax.plot(d["cycle"], d[col], "-", color=node_colors[col], lw=1.6,
                    alpha=0.9, label=col)
        ax.set_ylim(-0.02, 1.02)
        is_blind = int(u) in blind
        if is_blind:
            ax.set_facecolor("0.96")
            ax.text(0.5, 0.5, "no damage read\n(blind mode)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="firebrick", alpha=0.7)
        ax.set_xlabel("cycle", fontsize=8); ax.set_ylabel("latent health", fontsize=8)
        ax.set_title(f"Unit {int(u)}" + ("  ⚠" if is_blind else ""), fontsize=9,
                     color=("firebrick" if is_blind else "black"))
        ax.tick_params(labelsize=7)
        if n == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Latent spool health vs cycle (per unit)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _finish(fig, save, "nodes", show)


# ==========================================================================
# RUL plots
# ==========================================================================

def _blind_units(pred, thresh=0.2):
    """Units the tree read no damage on (peak latent health < thresh): its RUL is
    a no-damage prior, not a reading. Uses gft.flag_undiagnosable when *_h exist."""
    if not any(c.endswith("_h") for c in pred.columns):
        return set()
    fu = gft.flag_undiagnosable(pred, damage_thresh=thresh)
    return set(int(u) for u in fu.loc[fu["undiagnosable"], "unit"])


def plot_rul(feats, pred, size=12, save=None, show=True):
    m = feats[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
    units = np.unique(m["unit"])
    blind = _blind_units(pred)
    rows, cols = _grid(len(units))
    fig = plt.figure(figsize=(size, max(size, rows * 2.6)))
    gs = gridspec.GridSpec(rows, cols)
    for n, u in enumerate(units):
        ax = fig.add_subplot(gs[n])
        d = m[m["unit"] == u].sort_values("cycle")
        ax.plot(d["cycle"], d["RUL"], "-", color=_color(u), lw=2, label="real")
        ax.plot(d["cycle"], d["RUL_hat"], "o", color=_color(u), mfc="none",
                ms=4, alpha=0.8, label="pred")
        is_blind = int(u) in blind
        ax.set_title(f"Unit {int(u)}" + ("  ⚠ blind" if is_blind else ""),
                     color=("firebrick" if is_blind else "black"))
        if is_blind:
            ax.set_facecolor("0.96")
        ax.set_xlabel("cycle"); ax.set_ylabel("RUL")
        if n == 0:
            ax.legend()
    fig.suptitle("RUL: real vs predicted   (⚠ blind = fault in a component the tree "
                 "has no leaf for)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _finish(fig, save, "rul", show)


def plot_scatter(feats, pred, size=3.4, save=None, show=True):
    """RUL parity, faceted BY UNIT: one y=x panel per unit with that unit's own
    held-out RMSE in the title. Previously every unit shared a single parity
    axis, so a few tail units dominated the picture."""
    m = feats[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
    units = np.unique(m["unit"])
    rows, cols = _grid(len(units))
    hi = float(m["RUL"].max()) * 1.05
    fig = plt.figure(figsize=(size * cols, max(size, rows * size * 0.9)))
    gs = gridspec.GridSpec(rows, cols)
    for n, u in enumerate(units):
        ax = fig.add_subplot(gs[n])
        d = m[m["unit"] == u]
        ax.plot(d["RUL"], d["RUL_hat"], "o", color=_color(u), mfc="none",
                ms=4, alpha=0.6)
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("True RUL", fontsize=8); ax.set_ylabel("Pred RUL", fontsize=8)
        ax.set_title(f"Unit {int(u)}  (RMSE={gft.rmse(d['RUL'], d['RUL_hat']):.1f})",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"RUL parity per unit   overall RMSE={gft.rmse(m['RUL'], m['RUL_hat']):.2f}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _finish(fig, save, "scatter", show)


def plot_fitness(model, size=7, save=None, show=True):
    h = model["history"]
    fig, ax = plt.subplots(figsize=(size, size * 0.6))
    ax.plot(np.arange(1, len(h) + 1), h, "-o", ms=3)
    ax.set_xlabel("generation"); ax.set_ylabel("best loss (NASA + λ·θ)")
    ax.set_title("GA fitness"); ax.grid(alpha=0.3)
    ax.annotate(f"{h[-1]:.4g}", xy=(len(h), h[-1]),
                xytext=(-40, 12), textcoords="offset points")
    plt.tight_layout()
    _finish(fig, save, "fitness", show)


# ==========================================================================
# control surfaces: ONE FIGURE PER sub-FIS
# ==========================================================================

def _disp_range(inp):
    """Axis [lo, hi] in display units (raw for sensors, native for nodes)."""
    c = gft.disp(inp, inp["centers"])
    return float(c.min()), float(c.max())


def _eval_disp(node, Xdisp, singletons):
    """Evaluate a node's FIS at display-unit points Xdisp (n, d)."""
    cols, centers = [], []
    for k, inp in enumerate(node["inputs"]):
        x = Xdisp[:, k]
        if inp["kind"] == "col":
            x = (x - inp["mu"]) / inp["sd"]
        cols.append(x); centers.append(inp["centers"])
    return gft._fis(cols, centers, singletons)


def _surface(node, singletons, ix, iy, res=45):
    d = len(node["inputs"])
    xlo, xhi = _disp_range(node["inputs"][ix])
    ylo, yhi = _disp_range(node["inputs"][iy])
    XX, YY = np.meshgrid(np.linspace(xlo, xhi, res), np.linspace(ylo, yhi, res))
    P = np.empty((XX.size, d))
    for k in range(d):
        if k == ix:
            P[:, k] = XX.ravel()
        elif k == iy:
            P[:, k] = YY.ravel()
        else:
            lo, hi = _disp_range(node["inputs"][k])
            P[:, k] = 0.5 * (lo + hi)
    return XX, YY, _eval_disp(node, P, singletons).reshape(XX.shape)


def _out_label(node):
    return node["target"] or ("RUL" if node["root"] else node["name"])


def plot_surface_node(node, singletons, kind="surface", res=45, size=5,
                      save=None, show=True):
    """One figure for a single sub-FIS: a curve (1 input), or one panel per
    input-pair (2-3 inputs), the remaining input frozen at its centre."""
    d = len(node["inputs"])
    pairs = list(itertools.combinations(range(d), 2))
    olabel = _out_label(node)

    if d == 1:                                            # 1-input curve
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        lo, hi = _disp_range(node["inputs"][0])
        xs = np.linspace(lo, hi, res)
        ys = _eval_disp(node, xs.reshape(-1, 1), singletons)
        ax.plot(xs, ys, lw=2)
        ax.set_xlabel(node["inputs"][0]["src"]); ax.set_ylabel(olabel)
        ax.set_title(node["name"])
        plt.tight_layout()
        _finish(fig, save, f"surface_{node['name']}", show)
        return

    ncols = len(pairs)
    fig = plt.figure(figsize=(size * ncols, size))
    for j, (ix, iy) in enumerate(pairs):
        xlab = node["inputs"][ix]["src"]
        ylab = node["inputs"][iy]["src"]
        XX, YY, ZZ = _surface(node, singletons, ix, iy, res)
        if kind == "contour":
            ax = fig.add_subplot(1, ncols, j + 1)
            cf = ax.contourf(XX, YY, ZZ, levels=18, cmap="viridis")
            for cx in _disp_axis_centers(node["inputs"][ix]):
                ax.axvline(cx, color="w", lw=0.6, alpha=0.5)
            for cy in _disp_axis_centers(node["inputs"][iy]):
                ax.axhline(cy, color="w", lw=0.6, alpha=0.5)
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax = fig.add_subplot(1, ncols, j + 1, projection="3d")
            ax.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, alpha=0.95)
            ax.set_zlabel(olabel); ax.view_init(elev=28, azim=-125)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{xlab} × {ylab}")
    fig.suptitle(f"{node['name']}  →  {olabel}   [{kind}]")
    if kind == "contour":
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.subplots_adjust(top=0.88, bottom=0.08, wspace=0.15)
    _finish(fig, save, f"surface_{node['name']}", show)


def _disp_axis_centers(inp):
    return gft.disp(inp, inp["centers"])


# ==========================================================================
# membership functions: what the GA did to the antecedents
# ==========================================================================

def plot_memberships(model, size=3.2, save=None, show=True):
    """One panel per FIS input: the LEARNED Ruspini partition (solid, coloured)
    over the quantile partition it was seeded from (dashed grey).

    THIS is the figure that shows whether tuning the antecedents earned its keep.
    Watch the theta inputs of hp/lp: with quantile centres nearly all the
    resolution sits in the healthy plateau (theta ~ 0), which is why hp_h used to
    flatten out. If the GA has dragged centres towards the degraded tail, the
    partition is now spending its resolution where the signal is."""
    meta, g = model["meta"], model["genome"]
    ins = [(m, i, c) for m in meta
           for i, c in zip(m["inputs"], gft.centers(m, g))]
    ncol = min(4, len(ins))
    nrow = int(np.ceil(len(ins) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow),
                             squeeze=False)
    for ax, (m, inp, c) in zip(axes.ravel(), ins):
        cL, c0 = gft.disp(inp, c), gft.disp(inp, inp["c0"])
        lo, hi = min(cL.min(), c0.min()), max(cL.max(), c0.max())
        pad = 0.05 * ((hi - lo) or 1.0)
        xs = np.linspace(lo - pad, hi + pad, 500)
        M0, ML = gft._memberships(xs, c0), gft._memberships(xs, cL)
        for j in range(M0.shape[1]):
            ax.plot(xs, M0[:, j], "--", lw=1.4, color="0.65")
            ax.plot(xs, ML[:, j], "-", lw=1.8, color=f"C{j}")
        ax.set_title(f"{m['name']} <- {inp['src']}", fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(ins):]:
        ax.axis("off")
    fig.suptitle("Membership functions:  learned (solid)  vs  quantile seed (dashed)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finish(fig, save, "memberships", show)


def plot_surfaces(model, kind="surface", res=45, size=5, save=None, show=True):
    """Draw every sub-FIS as its OWN figure (surface_<node>.png)."""
    for node in model["meta"]:
        sing = _slice(model["meta"], model["genome"], node)
        plot_surface_node(node, sing, kind=kind, res=res, size=size,
                          save=save, show=show)


# ==========================================================================
# entry points
# ==========================================================================

def analyze(feats, pred, model=None, surfaces=False, surface_kind="surface",
            savedir=None, show=True):
    """Print metrics and draw every figure. If `savedir` is given, all figures
    are written there (folder created if needed, existing .png replaced)."""
    save = _prepare_dir(savedir) if savedir else None
    print_metrics(feats, pred)
    plot_theta(feats, pred, save=save, show=show)     # one fig per theta
    plot_nodes(pred, save=save, show=show)
    plot_rul(feats, pred, save=save, show=show)
    plot_scatter(feats, pred, save=save, show=show)
    if model is not None:
        plot_fitness(model, save=save, show=show)
        if model.get("learn_mf"):
            plot_memberships(model, save=save, show=show)
        if surfaces:
            plot_surfaces(model, kind=surface_kind, save=save, show=show)


def save_figures(feats, pred, model=None, folder="figures",
                 surfaces=True, surface_kind="contour"):
    """Save every figure into `folder` (no on-screen display)."""
    analyze(feats, pred, model=model, surfaces=surfaces,
            surface_kind=surface_kind, savedir=folder, show=False)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    df = gft._synthetic_frame(units=6, cycles=60, seed=1)
    model = gft.fit_gft(df, gens=60, pop=60, verbose=False)
    pred = gft.predict_gft(df, model)
    save_figures(df, pred, model=model, folder="/home/claude/figures",
                 surface_kind="contour")
    print("analysis self-test OK")