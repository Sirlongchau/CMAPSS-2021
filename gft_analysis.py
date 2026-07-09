"""
gft_analysis.py
===============

Turn the raw (unit, cycle) prediction table from predict_gft() into readable
REAL-vs-PREDICTED plots, in the visual style of the NASA/Kaggle N-CMAPSS
notebook (one colour per engine unit, traces vs flight cycle).

It joins:
    feats  -> ground truth   (theta__<comp>__<mod>, target__-__RUL)
    pred   -> predictions     (theta__<comp>__<mod>__hat, RUL_hat)
on (id__-__unit, id__-__cycle), then draws:

    1. plot_theta   : theta modifiers vs cycle, per unit, real (solid line)
                      vs predicted (dashed) -- like the notebook's theta plots.
    2. plot_rul     : RUL vs cycle, per unit, real vs predicted.
    3. plot_scatter : predicted vs true RUL, y=x diagonal (prognostics view).
    4. per_unit_metrics / print_metrics : numbers, so print() is useful too.

Usage
-----
    from ncmapss_gft_features import extract_features, Config
    from gft_ncmapss import fit_gft, predict_gft, GFTConfig
    from gft_analysis import analyze

    feats, _ = extract_features("N-CMAPSS_DS02.h5", Config(split="dev"))
    model = fit_gft(feats, GFTConfig())
    pred  = predict_gft(feats, model)

    analyze(feats, pred, show=True)          # or save_prefix="ds02_"
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

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


def merge_truth_pred(feats: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Inner-join truth and predictions on (unit, cycle)."""
    keep = KEYS + [c for c in feats.columns
                   if c.startswith("theta__") or c == "target__-__RUL"]
    m = feats[keep].merge(pred, on=KEYS, how="inner")
    return m.sort_values(KEYS).reset_index(drop=True)


# ==========================================================================
# Plots
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
# Metrics  (so print() is informative, not a wall of numbers)
# ==========================================================================

def plot_fitness(model: dict, labelsize: int = 15, size: int = 12,
                 save: Optional[str] = None, show: bool = True):
    """
    GA training curves: best fitness (loss) per epoch, one panel per stage.
    The two stages use different loss units (Stage 1 = theta RMSE, Stage 2 =
    RUL RMSE in cycles), so they get separate axes.
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
            save_prefix: Optional[str] = None, show: bool = True):
    """Print metrics and draw the figures in one call. Pass `model` to also
    plot the GA fitness curves."""
    print_metrics(feats, pred)
    if model is not None:
        plot_fitness(model, save=save_prefix, show=show)
    plot_theta(feats, pred, mods=mods, save=save_prefix, show=show)
    plot_rul(feats, pred, save=save_prefix, show=show)
    plot_scatter(feats, pred, save=save_prefix, show=show)


# ==========================================================================
# Self-test: reuse the GFT synthetic pipeline, save example figures
# ==========================================================================

if __name__ == "__main__":
    matplotlib.use("Agg")  # headless for the self-test
    from gft_ncmapss import _synthetic_frame, fit_gft, predict_gft, GFTConfig

    df = _synthetic_frame(units=6, cycles=60, seed=1)
    model = fit_gft(df, GFTConfig(n_terms=3, pop=40, gens=40))
    pred = predict_gft(df, model)

    analyze(df, pred, model=model, save_prefix="/home/claude/_demo_", show=False)