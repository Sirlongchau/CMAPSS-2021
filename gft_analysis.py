"""
gft_analysis.py -- read RUL predictions from predict_gft() and draw:

    print_metrics       : per-unit RUL RMSE (+ NASA score).
    plot_theta          : ground-truth θ modifiers vs cycle (reference shape).
    plot_nodes          : learned mid-node health ∈[0,1] vs cycle (sanity check).
    plot_rul            : RUL vs cycle, per unit, real vs predicted.
    plot_scatter        : predicted vs true RUL with the y=x line.
    plot_fitness        : GA best-loss per generation.
    plot_surfaces       : the learned control surface of every sub-FIS.
    analyze             : do all of the above in one call.
    save_figures        : write every figure to a folder (created if needed,
                          existing .png replaced).

The tree stores each node as {name, inputs, n_rules, root}; a genome slice per
node holds its singletons. plot_surfaces evaluates a node's FIS on a dense grid
over two of its inputs (others frozen at their centre) -- the same maths as
gft._fis, so it needs nothing from the model but its topology + genome.
"""

from __future__ import annotations
import os
import glob
import itertools
from typing import Optional

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
    k = 0
    for m in meta:
        if m is node:
            return genome[k:k + m["n_rules"]]
        k += m["n_rules"]
    raise KeyError(node["name"])


def _range(inp):
    """Axis [lo, hi] in the input's own space (z for cols, [0,1] for nodes)."""
    c = inp["centers"]
    return float(c.min()), float(c.max())


# ==========================================================================
# metrics + RUL plots
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


def plot_rul(feats, pred, size=12, save=None, show=True):
    m = feats[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
    units = np.unique(m["unit"])
    rows, cols = _grid(len(units))
    fig = plt.figure(figsize=(size, max(size, rows * 2.6)))
    gs = gridspec.GridSpec(rows, cols)
    for n, u in enumerate(units):
        ax = fig.add_subplot(gs[n])
        d = m[m["unit"] == u].sort_values("cycle")
        ax.plot(d["cycle"], d["RUL"], "-", color=_color(u), lw=2, label="real")
        ax.plot(d["cycle"], d["RUL_hat"], "o", color=_color(u), mfc="none",
                ms=4, alpha=0.8, label="pred")
        ax.set_title(f"Unit {int(u)}"); ax.set_xlabel("cycle"); ax.set_ylabel("RUL")
        if n == 0:
            ax.legend()
    fig.suptitle("RUL: real vs predicted")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _finish(fig, save, "rul", show)


def plot_scatter(feats, pred, size=7, save=None, show=True):
    m = feats[["unit", "cycle", "RUL"]].merge(pred, on=["unit", "cycle"])
    fig, ax = plt.subplots(figsize=(size, size))
    for u in np.unique(m["unit"]):
        d = m[m["unit"] == u]
        ax.plot(d["RUL"], d["RUL_hat"], "o", color=_color(u), mfc="none",
                ms=4, alpha=0.6, label=f"Unit {int(u)}")
    lim = [0, float(m["RUL"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL")
    ax.set_title(f"RUL parity (RMSE={gft.rmse(m['RUL'], m['RUL_hat']):.2f})")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    _finish(fig, save, "scatter", show)


def _series_grid(df, series, ylabel_each, suptitle, ylim=None,
                 size=12, save=None, name="", show=True):
    """One subplot per series, one coloured line per unit, x = cycle."""
    if not series:
        return
    units = np.unique(df["unit"])
    rows, cols = _grid(len(series))
    fig = plt.figure(figsize=(size, max(size * 0.5, rows * 3)))
    gs = gridspec.GridSpec(rows, cols)
    for n, col in enumerate(series):
        ax = fig.add_subplot(gs[n])
        for u in units:
            d = df[df["unit"] == u].sort_values("cycle")
            ax.plot(d["cycle"], d[col], "-", color=_color(u), lw=1.6, alpha=0.9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("cycle"); ax.set_ylabel(ylabel_each(col)); ax.set_title(col)
    fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _finish(fig, save, name, show)


def plot_theta(feats, size=12, save=None, show=True):
    """Ground-truth health modifiers (θ) vs cycle -- the reference degradation
    shape the mid nodes should track. Uses any '*_mod' columns in `feats`."""
    theta = [c for c in feats.columns if c.endswith("_mod")]
    _series_grid(feats, theta, lambda c: "θ", "Ground-truth θ modifiers vs cycle",
                 size=size, save=save, name="theta", show=show)


def plot_nodes(pred, size=12, save=None, show=True):
    """Learned intermediate-node health (∈[0,1]) vs cycle -- sanity-check the
    tree's mid layers. Uses the '*_h' columns returned by predict_gft."""
    nodes = [c for c in pred.columns if c.endswith("_h")]
    _series_grid(pred, nodes, lambda c: "health", "Learned node health vs cycle",
                 ylim=(-0.02, 1.02), size=size, save=save, name="nodes", show=show)


def plot_fitness(model, size=7, save=None, show=True):
    h = model["history"]
    fig, ax = plt.subplots(figsize=(size, size * 0.6))
    ax.plot(np.arange(1, len(h) + 1), h, "-o", ms=3)
    ax.set_xlabel("generation"); ax.set_ylabel("best NASA score")
    ax.set_title("GA fitness"); ax.grid(alpha=0.3)
    ax.annotate(f"{h[-1]:.4g}", xy=(len(h), h[-1]),
                xytext=(-40, 12), textcoords="offset points")
    plt.tight_layout()
    _finish(fig, save, "fitness", show)


# ==========================================================================
# control surfaces
# ==========================================================================

def _surface(meta, singletons, node, ix, iy, res=45):
    """Dense (XX, YY, ZZ) for `node` over inputs (ix, iy); others at centre."""
    d = len(node["inputs"])
    xlo, xhi = _range(node["inputs"][ix])
    ylo, yhi = _range(node["inputs"][iy])
    XX, YY = np.meshgrid(np.linspace(xlo, xhi, res), np.linspace(ylo, yhi, res))
    cols, centers = [], []
    for k, inp in enumerate(node["inputs"]):
        if k == ix:
            cols.append(XX.ravel())
        elif k == iy:
            cols.append(YY.ravel())
        else:
            lo, hi = _range(inp)
            cols.append(np.full(XX.size, 0.5 * (lo + hi)))
        centers.append(inp["centers"])
    ZZ = gft._fis(cols, centers, singletons).reshape(XX.shape)
    return XX, YY, ZZ


def _panels(meta, genome):
    out = []
    for node in meta:
        d = len(node["inputs"])
        pairs = list(itertools.combinations(range(d), 2)) or [(0, 0)]
        for ix, iy in pairs:
            out.append((node, _slice(meta, genome, node), ix, iy))
    return out


def plot_surfaces(model, kind="surface", res=45, size=13, save=None, show=True):
    """Learned surface of every sub-FIS. kind: 'surface' (3-D) or 'contour'."""
    meta, genome = model["meta"], model["genome"]
    panels = _panels(meta, genome)
    rows, cols = _grid(len(panels))
    fig = plt.figure(figsize=(size, max(size * 0.5, rows * 3.4)))
    gs = gridspec.GridSpec(rows, cols, hspace=0.45, wspace=0.3)
    for n, (node, sing, ix, iy) in enumerate(panels):
        d = len(node["inputs"])
        xlab = node["inputs"][ix]["src"]
        ylab = node["inputs"][iy]["src"]
        if d == 1:                                     # single-input curve
            ax = fig.add_subplot(gs[n])
            lo, hi = _range(node["inputs"][0])
            xs = np.linspace(lo, hi, res)
            ys = gft._fis([xs], [node["inputs"][0]["centers"]], sing)
            ax.plot(xs, ys, lw=2)
            ax.set_xlabel(xlab); ax.set_title(node["name"])
            continue
        XX, YY, ZZ = _surface(meta, sing, node, ix, iy, res)
        if kind == "contour":
            ax = fig.add_subplot(gs[n])
            cf = ax.contourf(XX, YY, ZZ, levels=18, cmap="viridis")
            for cx in node["inputs"][ix]["centers"]:
                ax.axvline(cx, color="w", lw=0.6, alpha=0.5)
            for cy in node["inputs"][iy]["centers"]:
                ax.axhline(cy, color="w", lw=0.6, alpha=0.5)
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax = fig.add_subplot(gs[n], projection="3d")
            ax.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, alpha=0.95)
            ax.set_zlabel(node["name"]); ax.view_init(elev=28, azim=-125)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{node['name']}  ({xlab}×{ylab})")
    fig.suptitle(f"Control surfaces [{kind}]")
    if kind == "contour":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.04, right=0.97)
    _finish(fig, save, f"surfaces_{kind}", show)


# ==========================================================================
# convenience
# ==========================================================================

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


def analyze(feats, pred, model=None, surfaces=False, surface_kind="surface",
            savedir=None, show=True):
    """Print metrics and draw every figure in one call.

    savedir : if given, all figures are written to that folder (created if it
              doesn't exist, existing .png replaced) instead of / as well as
              being shown. Set show=False to only save.
    """
    save = _prepare_dir(savedir) if savedir else None
    print_metrics(feats, pred)
    plot_theta(feats, save=save, show=show)          # ground-truth θ / cycle
    plot_nodes(pred, save=save, show=show)           # learned mid-node health
    plot_rul(feats, pred, save=save, show=show)
    plot_scatter(feats, pred, save=save, show=show)
    if model is not None:
        plot_fitness(model, save=save, show=show)
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
    # save everything into a folder (created if missing, old figures replaced)
    save_figures(df, pred, model=model, folder="/home/claude/figures",
                 surfaces=True, surface_kind="contour")
    print("analysis self-test OK")