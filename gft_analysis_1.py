"""
gft_analysis.py
===============

Turn the raw (unit, cycle) prediction table from predict_gft() into readable
REAL-vs-PREDICTED plots, in the visual style of the NASA/Kaggle N-CMAPSS
notebook (one colour per engine unit, traces vs flight cycle) -- AND now the
CONTROL SURFACES of every sub-FIS in both GFT stages.

It joins:
    feats  -> ground truth   (theta__<comp>__<mod>, target__-__RUL)
    pred   -> predictions     (theta__<comp>__<mod>__hat, RUL_hat)
on (id__-__unit, id__-__cycle), then draws:

    1. plot_theta            : theta modifiers vs cycle, per unit, real vs pred.
    2. plot_rul              : RUL vs cycle, per unit, real vs predicted.
    3. plot_scatter          : predicted vs true RUL, y=x diagonal.
    4. plot_fitness          : GA loss per epoch, one panel per stage.
    5. plot_control_surfaces : the learned FIS mapping surfaces (NEW).
    6. per_unit_metrics / print_metrics : numbers.

CONTROL SURFACES (new)
----------------------
A FIS is a function  inputs -> output. `plot_control_surfaces` evaluates each
learned sub-FIS on a dense grid over its own input box and renders the result
as a 3-D surface (or a top-down filled contour with the fuzzy-rule grid drawn
in). Works for both TSK order 0 and order 1, and for the standardised
(z-scored) nodes of gft_ncmapss_1 as well as the raw-input nodes of
gft_ncmapss -- the module and per-node standardisation are auto-detected, so
you never have to tell it which pipeline built the model.

    Stage 1 : 4 two-input FIS -> one surface each (sensor x sensor -> theta).
    Stage 2 : one three-input FIS -> the three input PAIRS as surfaces, the
              remaining input held at the centre of its range.

Usage
-----
    from ncmapss_gft_features import extract_features, Config
    from gft_ncmapss_1 import fit_gft, predict_gft, GFTConfig
    from gft_analysis import analyze

    feats, _ = extract_features("N-CMAPSS_DS02.h5", Config(split="dev"))
    model = fit_gft(feats, GFTConfig())
    pred  = predict_gft(feats, model)

    analyze(feats, pred, model=model, surfaces=True, show=True)   # everything
    # or just the surfaces:
    from gft_analysis import plot_control_surfaces
    plot_control_surfaces(model, kind="surface")     # 3-D
    plot_control_surfaces(model, kind="contour")     # top-down + rule grid
"""

from __future__ import annotations

import sys
import importlib
import itertools
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

KEYS = ["id__-__unit", "id__-__cycle"]

# default degrading modifiers (match the notebook's parallel-coordinates pick)
DEFAULT_MODS = ["theta__HPT__eff", "theta__LPT__eff", "theta__LPT__flow"]

# 20-colour per-unit palette, same idea as the notebook's color_dic_unit
_PALETTE = [f"C{i}" for i in range(20)]


def _color(unit) -> str:
    return _PALETTE[(int(unit) - 1) % len(_PALETTE)]


def _grid(n: int):
    """Rows/cols for a compact subplot grid (<=4 columns), notebook-style."""
    cols = min(int(np.floor(n ** 0.5)), 4) or 1
    rows = int(np.ceil(n / cols))
    return rows, cols


def _nice(col: str) -> str:
    """Human-readable axis label from a <role>__<node>__<var> column name."""
    s = (col.replace("theta__", "θ ").replace("in__", "").replace("s2__", "")
            .replace("target__-__", "").replace("op__-__", "")
            .replace("__-__", " ").replace("__", " "))
    return s.strip()


def merge_truth_pred(feats: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Inner-join truth and predictions on (unit, cycle)."""
    keep = KEYS + [c for c in feats.columns
                   if c.startswith("theta__") or c == "target__-__RUL"]
    m = feats[keep].merge(pred, on=KEYS, how="inner")
    return m.sort_values(KEYS).reset_index(drop=True)


# ==========================================================================
# Time-series / parity plots  (unchanged behaviour)
# ==========================================================================

def plot_theta(feats: pd.DataFrame, pred: pd.DataFrame,
               mods: Optional[List[str]] = None, labelsize: int = 15,
               size: int = 12, save: Optional[str] = None, show: bool = True):
    """
    theta vs cycle, one subplot per modifier, one colour per unit.
    Solid line = real theta, dashed line = predicted theta_hat.
    """
    m = merge_truth_pred(feats, pred)
    mods = mods or [c for c in (mods or DEFAULT_MODS) if c in feats.columns] \
        or [c for c in feats.columns if c.startswith("theta__")]

    rows, cols = _grid(len(mods))
    fig = plt.figure(figsize=(size, max(size, rows * 3)))
    gs = gridspec.GridSpec(rows, cols)

    for n, mod in enumerate(mods):
        ax = fig.add_subplot(gs[n])
        for u in np.unique(m["id__-__unit"]):
            d = m[m["id__-__unit"] == u].sort_values("id__-__cycle")
            x = d["id__-__cycle"].to_numpy()
            ax.plot(x, d[mod], "-", color=_color(u), alpha=0.9, lw=1.8)
            if mod + "__hat" in d.columns:
                ax.plot(x, d[mod + "__hat"], "--", color=_color(u),
                        alpha=0.9, lw=1.4)
        ax.set_xlabel("Time [cycle]", fontsize=labelsize)
        ax.set_ylabel(mod.replace("theta__", "θ ").replace("__", " ") + " [-]",
                      fontsize=labelsize)
        ax.tick_params(labelsize=labelsize - 3)
    # one shared legend explaining the linestyle code
    fig.legend([plt.Line2D([0], [0], color="k", ls="-"),
                plt.Line2D([0], [0], color="k", ls="--")],
               ["real θ", "predicted θ̂"], loc="upper right", fontsize=labelsize)
    fig.suptitle("Health parameters: real vs predicted", fontsize=labelsize + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _finish(fig, save, "theta", show)


def plot_rul(feats: pd.DataFrame, pred: pd.DataFrame, labelsize: int = 15,
             size: int = 12, save: Optional[str] = None, show: bool = True):
    """RUL vs cycle, one subplot per unit: real (solid) vs predicted (markers)."""
    m = merge_truth_pred(feats, pred)
    units = np.unique(m["id__-__unit"])
    rows, cols = _grid(len(units))
    fig = plt.figure(figsize=(size, max(size, rows * 2.6)))
    gs = gridspec.GridSpec(rows, cols)

    for n, u in enumerate(units):
        ax = fig.add_subplot(gs[n])
        d = m[m["id__-__unit"] == u].sort_values("id__-__cycle")
        x = d["id__-__cycle"].to_numpy()
        ax.plot(x, d["target__-__RUL"], "-", color=_color(u), lw=2, label="real")
        ax.plot(x, d["RUL_hat"], "o", color=_color(u), mfc="none",
                ms=4, alpha=0.8, label="pred")
        ax.set_title(f"Unit {int(u)}", fontsize=labelsize - 2)
        ax.set_xlabel("Time [cycle]", fontsize=labelsize - 3)
        ax.set_ylabel("RUL [cycle]", fontsize=labelsize - 3)
        ax.tick_params(labelsize=labelsize - 4)
        if n == 0:
            ax.legend(fontsize=labelsize - 4)
    fig.suptitle("RUL: real vs predicted", fontsize=labelsize + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _finish(fig, save, "rul", show)


def plot_scatter(feats: pd.DataFrame, pred: pd.DataFrame, labelsize: int = 15,
                 size: int = 7, save: Optional[str] = None, show: bool = True):
    """Predicted vs true RUL, coloured per unit, with the y=x reference line."""
    m = merge_truth_pred(feats, pred)
    fig, ax = plt.subplots(figsize=(size, size))
    for u in np.unique(m["id__-__unit"]):
        d = m[m["id__-__unit"] == u]
        ax.plot(d["target__-__RUL"], d["RUL_hat"], "o", color=_color(u),
                mfc="none", ms=4, alpha=0.6, label=f"Unit {int(u)}")
    lim = [0, float(m["target__-__RUL"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True RUL [cycle]", fontsize=labelsize)
    ax.set_ylabel("Predicted RUL [cycle]", fontsize=labelsize)
    ax.set_title(f"RUL parity  (RMSE = {_rmse(m['target__-__RUL'], m['RUL_hat']):.2f})",
                 fontsize=labelsize)
    ax.tick_params(labelsize=labelsize - 3)
    ax.legend(fontsize=labelsize - 5, ncol=2)
    plt.tight_layout()
    _finish(fig, save, "scatter", show)


# ==========================================================================
# CONTROL SURFACES  (new)  --  the learned FIS input->output mappings
# ==========================================================================

def _module_of(node):
    """
    Return the gft_ncmapss[_1] module that defines this node's class, so the
    exact matching decode / fis_output / _slices are used (order-0/1 and the
    z-score layout differ between the two files). Falls back to importing the
    known module names if the class' own module was stripped (e.g. __main__).
    """
    mod = sys.modules.get(type(node).__module__)
    if mod is not None and hasattr(mod, "decode") and hasattr(mod, "fis_output"):
        return mod
    for name in ("gft_ncmapss_1", "gft_ncmapss"):
        m = sys.modules.get(name)
        if m is None:
            try:
                m = importlib.import_module(name)
            except ImportError:
                m = None
        if m is not None and hasattr(m, "decode"):
            return m
    raise ImportError("Cannot find the GFT module (need decode/fis_output/_slices).")


def _has_std(node) -> bool:
    """True if the node standardises its inputs (gft_ncmapss_1 nodes)."""
    return hasattr(node, "standardize") and hasattr(node, "mu")


def _axis_range(node, k: int, pad: float = 0.0):
    """Raw-input [lo, hi] display range for input k (maps z-box back if needed)."""
    if _has_std(node):
        lo = node.mu[k] + node.c_lo[k] * node.sd[k]
        hi = node.mu[k] + node.c_hi[k] * node.sd[k]
    else:
        lo, hi = node.c_lo[k], node.c_hi[k]
    if pad:
        s = (hi - lo) * pad
        lo, hi = lo - s, hi + s
    return float(lo), float(hi)


def _centers_raw(gm, node, gslice):
    """Decoded antecedent centres, mapped back to RAW input units per input."""
    centers, _ = gm.decode(node, gslice)
    out = []
    for k, c in enumerate(centers):
        c = np.asarray(c, float)
        out.append(node.mu[k] + c * node.sd[k] if _has_std(node) else c)
    return out


def _eval_node(gm, node, gslice, X_raw) -> np.ndarray:
    """Evaluate one sub-FIS at RAW input points X_raw (n, d) -> output (n,)."""
    centers, cons = gm.decode(node, gslice)
    X = node.standardize(X_raw) if _has_std(node) else np.asarray(X_raw, float)
    return gm.fis_output(X, centers, cons)


def fis_surface(gm, node, gslice, ix: int, iy: int, res: int = 45,
                fixed: "dict | None" = None):
    """
    Dense evaluation grid for a sub-FIS over inputs (ix, iy). Any other input
    is held fixed (at `fixed[k]` if given, else the centre of its range).
    Returns (XX, YY, ZZ) meshes in RAW input units.
    """
    d = len(node.inputs)
    xlo, xhi = _axis_range(node, ix)
    ylo, yhi = _axis_range(node, iy)
    XX, YY = np.meshgrid(np.linspace(xlo, xhi, res), np.linspace(ylo, yhi, res))
    P = np.empty((XX.size, d))
    for k in range(d):
        if k == ix:
            P[:, k] = XX.ravel()
        elif k == iy:
            P[:, k] = YY.ravel()
        else:
            if fixed and k in fixed:
                P[:, k] = fixed[k]
            else:
                lo, hi = _axis_range(node, k)
                P[:, k] = 0.5 * (lo + hi)
    ZZ = _eval_node(gm, node, gslice, P).reshape(XX.shape)
    return XX, YY, ZZ


def _panels_for(gm, nodes, genome):
    """(node, gslice, ix, iy) for every input PAIR of every node in a stage."""
    panels = []
    for node, (a, b) in zip(nodes, gm._slices(nodes)):
        gslice = genome[a:b]
        d = len(node.inputs)
        pairs = list(itertools.combinations(range(d), 2))
        if not pairs:                      # 1-input FIS -> a single "curve" panel
            pairs = [(0, 0)]
        for ix, iy in pairs:
            panels.append((node, gslice, ix, iy))
    return panels


def _draw_panel(fig, cell, gm, node, gslice, ix, iy, kind, res, labelsize):
    d = len(node.inputs)

    # --- 1-input FIS: draw the y = FIS(x) curve --------------------------
    if d == 1:
        ax = fig.add_subplot(cell)
        lo, hi = _axis_range(node, 0)
        xs = np.linspace(lo, hi, res)
        ys = _eval_node(gm, node, gslice, xs.reshape(-1, 1))
        ax.plot(xs, ys, color="C0", lw=2)
        for cx in _centers_raw(gm, node, gslice)[0]:
            ax.axvline(cx, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel(_nice(node.inputs[0]), fontsize=labelsize - 3)
        ax.set_ylabel(_nice(node.output), fontsize=labelsize - 3)
        ax.set_title(node.name, fontsize=labelsize - 2)
        return

    XX, YY, ZZ = fis_surface(gm, node, gslice, ix, iy, res=res)
    xlab, ylab = _nice(node.inputs[ix]), _nice(node.inputs[iy])

    # note which inputs were frozen and at what value
    others = [k for k in range(d) if k not in (ix, iy)]
    frozen = ""
    if others:
        parts = []
        for k in others:
            lo, hi = _axis_range(node, k)
            parts.append(f"{_nice(node.inputs[k])}={0.5 * (lo + hi):.3g}")
        frozen = "  |  " + ", ".join(parts)

    if kind == "contour":
        ax = fig.add_subplot(cell)
        cf = ax.contourf(XX, YY, ZZ, levels=18, cmap="viridis")
        cr = _centers_raw(gm, node, gslice)               # draw the rule grid
        for cx in cr[ix]:
            ax.axvline(cx, color="w", lw=0.6, alpha=0.55)
        for cy in cr[iy]:
            ax.axhline(cy, color="w", lw=0.6, alpha=0.55)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(xlab, fontsize=labelsize - 3)
        ax.set_ylabel(ylab, fontsize=labelsize - 3)
    else:                                                 # 3-D surface
        ax = fig.add_subplot(cell, projection="3d")
        ax.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0,
                        antialiased=True, alpha=0.95)
        ax.set_xlabel(xlab, fontsize=labelsize - 4)
        ax.set_ylabel(ylab, fontsize=labelsize - 4)
        ax.set_zlabel(_nice(node.output), fontsize=labelsize - 4)
        ax.view_init(elev=28, azim=-125)
        ax.tick_params(labelsize=labelsize - 6)

    ax.set_title(node.name + frozen, fontsize=labelsize - 3)


def plot_control_surfaces(model: dict, stage="both", kind: str = "surface",
                          res: int = 45, labelsize: int = 15, size: int = 13,
                          save: Optional[str] = None, show: bool = True):
    """
    Draw the learned control surface of every sub-FIS.

    stage : "both" (default), 1, or 2.
    kind  : "surface" (3-D) or "contour" (top-down, with the fuzzy-rule grid).
    res   : grid resolution per axis.

    One figure per stage. Stage 1's four 2-input FIS give four surfaces;
    Stage 2's single 3-input FIS gives the three input-pair surfaces with the
    third input frozen at the centre of its range.
    """
    if kind not in ("surface", "contour"):
        raise ValueError("kind must be 'surface' or 'contour'")
    gm = _module_of(model["nodes1"][0])

    todo = []
    if stage in ("both", 1, "1"):
        todo.append((1, "Stage 1  —  sensors → θ̂",
                     model["nodes1"], model["genome1"]))
    if stage in ("both", 2, "2"):
        todo.append((2, "Stage 2  —  [damage, slope, load] → RUL",
                     model["nodes2"], model["genome2"]))

    for idx, title, nodes, genome in todo:
        panels = _panels_for(gm, nodes, genome)
        rows, cols = _grid(len(panels))
        fig = plt.figure(figsize=(size, max(size * 0.5, rows * 3.6)))
        hspace = 0.45 if kind == "surface" else 0.30
        gs = gridspec.GridSpec(rows, cols, hspace=hspace, wspace=0.30)
        for n, (node, gslice, ix, iy) in enumerate(panels):
            _draw_panel(fig, gs[n], gm, node, gslice, ix, iy, kind, res, labelsize)
        order = getattr(nodes[0], "order", 0)
        fig.suptitle(f"{title}    [TSK-{order} · {kind}]", fontsize=labelsize + 2)
        if kind == "contour":
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.subplots_adjust(top=0.92, bottom=0.06, left=0.04, right=0.97)
        _finish(fig, save, f"surface_stage{idx}", show)


# ==========================================================================
# GA fitness curves
# ==========================================================================

def plot_fitness(model: dict, labelsize: int = 15, size: int = 12,
                 save: Optional[str] = None, show: bool = True):
    """
    GA training curves: best fitness (loss) per epoch, one panel per stage.
    The two stages use different loss units (Stage 1 = theta RMSE, Stage 2 =
    RUL NASA score), so they get separate axes.
    """
    panels = [("Stage 1  (θ RMSE)", model.get("history1")),
              ("Stage 2  (RUL NASA score)", model.get("history2"))]
    panels = [(t, h) for t, h in panels if h]
    fig = plt.figure(figsize=(size, max(4, size * 0.35)))
    gs = gridspec.GridSpec(1, len(panels))
    for n, (title, hist) in enumerate(panels):
        ax = fig.add_subplot(gs[n])
        ep = np.arange(1, len(hist) + 1)
        ax.plot(ep, hist, "-o", color=_PALETTE[n], ms=3, alpha=0.9)
        ax.set_title(title, fontsize=labelsize)
        ax.set_xlabel("Epoch (generation)", fontsize=labelsize - 2)
        ax.set_ylabel("Best fitness (loss)", fontsize=labelsize - 2)
        ax.tick_params(labelsize=labelsize - 4)
        ax.grid(alpha=0.3)
        # annotate the final value so the plateau is readable
        ax.annotate(f"{hist[-1]:.4g}", xy=(ep[-1], hist[-1]),
                    xytext=(-40, 12), textcoords="offset points",
                    fontsize=labelsize - 4)
    fig.suptitle("GA fitness evolution", fontsize=labelsize + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finish(fig, save, "fitness", show)


# ==========================================================================
# Metrics  (so print() is informative, not a wall of numbers)
# ==========================================================================

def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def per_unit_metrics(feats: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-unit RUL RMSE and mean theta RMSE across modifiers."""
    m = merge_truth_pred(feats, pred)
    theta_mods = [c for c in feats.columns if c.startswith("theta__")
                  and c + "__hat" in pred.columns]
    rows = []
    for u in np.unique(m["id__-__unit"]):
        d = m[m["id__-__unit"] == u]
        th = np.mean([_rmse(d[c], d[c + "__hat"]) for c in theta_mods]) \
            if theta_mods else np.nan
        rows.append({"unit": int(u), "n_cycles": len(d),
                     "theta_RMSE": round(th, 5),
                     "RUL_RMSE": round(_rmse(d["target__-__RUL"], d["RUL_hat"]), 3)})
    df = pd.DataFrame(rows)
    df.loc["ALL"] = {"unit": "ALL", "n_cycles": len(m),
                     "theta_RMSE": round(np.mean(df["theta_RMSE"]), 5),
                     "RUL_RMSE": round(_rmse(m["target__-__RUL"], m["RUL_hat"]), 3)}
    return df


def print_metrics(feats: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    df = per_unit_metrics(feats, pred)
    print(df.to_string(index=False))
    return df


# ==========================================================================
# Convenience + IO
# ==========================================================================

def _finish(fig, save: Optional[str], name: str, show: bool):
    if save:
        path = f"{save}{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("saved:", path)
    if show:
        plt.show()
    plt.close(fig)


def analyze(feats: pd.DataFrame, pred: pd.DataFrame,
            model: Optional[dict] = None, mods: Optional[List[str]] = None,
            surfaces: bool = False, surface_kind: str = "surface",
            save_prefix: Optional[str] = None, show: bool = True):
    """
    Print metrics and draw the figures in one call.

    Pass `model` to also plot the GA fitness curves and (if surfaces=True) the
    learned control surfaces. `surface_kind` is "surface" (3-D) or "contour".
    """
    print_metrics(feats, pred)
    if model is not None:
        plot_fitness(model, save=save_prefix, show=show)
        if surfaces:
            plot_control_surfaces(model, kind=surface_kind,
                                  save=save_prefix, show=show)
    plot_theta(feats, pred, mods=mods, save=save_prefix, show=show)
    plot_rul(feats, pred, save=save_prefix, show=show)
    plot_scatter(feats, pred, save=save_prefix, show=show)


# ==========================================================================
# Self-test: reuse the GFT synthetic pipeline, save example figures
# ==========================================================================

if __name__ == "__main__":
    matplotlib.use("Agg")  # headless for the self-test
    try:                                   # prefer the TSK order-1 capable module
        from gft_ncmapss_1 import _synthetic_frame, fit_gft, predict_gft, GFTConfig
        cfg = GFTConfig(n_terms=3, pop=40, gens=40, order1=1, order2=0)
    except ImportError:
        from gft_ncmapss import _synthetic_frame, fit_gft, predict_gft, GFTConfig
        cfg = GFTConfig(n_terms=3, pop=40, gens=40)

    df = _synthetic_frame(units=6, cycles=60, seed=1)
    model = fit_gft(df, cfg)
    pred = predict_gft(df, model)

    analyze(df, pred, model=model, surfaces=True, surface_kind="surface",
            save_prefix="/home/claude/_demo_", show=False)
    # also produce the top-down contour view of the surfaces
    plot_control_surfaces(model, kind="contour",
                          save="/home/claude/_demo_contour_", show=False)