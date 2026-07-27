"""
leaf_search.py -- which sensors actually diagnose each health parameter?

For every theta target we currently pick 3 inputs by engineering judgement
(HPT_eff <- T48, P40, Nc, ...). This searches input combinations empirically and
keeps the best on HELD-OUT theta.

TWO TRAPS THIS AVOIDS, deliberately:

  1. It scores each leaf on ITS OWN theta (per-unit rho and R2), NOT on downstream
     RUL. Selecting on RUL would reward inputs that help through the age-correlated
     path -- the exact confound the whole project fights.
  2. It fits each leaf STANDALONE (one FIS, sensors -> theta_hat), not inside the
     full tree, so "which sensors carry this theta" is isolated from every other
     moving part (spools, root, the RUL loss).

Same preprocessing as gft (per-unit residual + EWMA) and the same pooled, cached
fleet (datasets.pooled), so a winning combination here transfers to gft.TREE.

    python leaf_search.py                 # all targets, physical prefilter, size 3
    python leaf_search.py --target HPT_flow_mod --k 2 --all-sensors

Wire a result back by editing gft._LEAVES with the chosen inputs.
"""

from __future__ import annotations
import argparse
import itertools

import numpy as np
import pandas as pd

import gft
import datasets as ds
from ncmapss_features import SENSORS, THETA


# ==========================================================================
# 1. Physical candidate prefilter
# ==========================================================================
# Restrict each target to sensors that could plausibly read that component, so the
# search is C(8,3)=56 rather than C(13,3)=286. This is a PRIOR, not a hard rule:
# --all-sensors removes it entirely. Kept deliberately generous -- the point of the
# search is to test the engineering choice, not to smuggle it back in as a filter.
#
#   HPT (high-pressure turbine)  : hot-section temps/pressures + core speed
#   LPT (low-pressure turbine)   : LPT exit temp/pressure + fan speed + bypass
CANDIDATES = {
    "HPT_eff_mod":  ["T48", "T30", "P40", "Ps30", "Nc", "Nf", "Wf", "T50"],
    "HPT_flow_mod": ["T48", "T30", "P40", "Ps30", "Nc", "Nf", "Wf", "P24"],
    "LPT_eff_mod":  ["T50", "P50", "Nf", "Nc", "P24", "T48", "Wf", "P21"],
    "LPT_flow_mod": ["T50", "P50", "P24", "Nf", "P21", "P15", "Nc", "Wf"],
}

# The current engineering choice, for reference in the output table.
BASELINE = {
    "HPT_eff_mod":  ["T48", "P40", "Nc"],
    "HPT_flow_mod": ["T48", "P40", "Ps30"],
    "LPT_eff_mod":  ["T50", "P50", "Nf"],
    "LPT_flow_mod": ["T50", "P50", "P24"],
}

# Which spool each target belongs to, and the OPPOSITE-spool target used as the
# specificity foil. A truly component-specific input set predicts its own spool's
# theta well and the other spool's theta poorly. An input set that is really just
# reading global damage (e.g. T48 as a "something is degrading" proxy) predicts
# BOTH about equally -- that is the pattern the raw rho leaderboard cannot see.
FOIL = {
    "HPT_eff_mod":  "LPT_eff_mod",
    "HPT_flow_mod": "LPT_flow_mod",
    "LPT_eff_mod":  "HPT_eff_mod",
    "LPT_flow_mod": "HPT_flow_mod",
}


# ==========================================================================
# 2. Standalone single-leaf FIS  (one FIS, sensors -> theta)
# ==========================================================================

def _fit_leaf(train, inputs, target, k=3, learn_mf=True, gens=120, pop=50,
              smooth_span=7, seed=0):
    """Fit ONE Sugeno FIS mapping `inputs` -> `target`, by GA, on `train`.

    Reuses gft's Ruspini FIS, gap-encoded learnable centres and GA -- but a tiny
    self-contained genome ([gap genes | singletons]) so it is fast and independent
    of the tree machinery. Returns a picklable dict usable by _predict_leaf."""
    x = {c: train[c].to_numpy(float) for c in inputs}
    mu = {c: float(x[c].mean()) for c in inputs}
    sd = {c: (float(x[c].std()) or 1.0) for c in inputs}
    z = {c: (x[c] - mu[c]) / sd[c] for c in inputs}
    c0 = {c: gft._centers(z[c], k) for c in inputs}
    dom = {c: gft._domain(z[c].min(), z[c].max()) for c in inputs}
    y = train[target].to_numpy(float)

    d = len(inputs)
    n_mf = d * (k + 1) if learn_mf else 0
    n_rule = k ** d
    lo = np.concatenate([np.full(n_mf, gft.GAP_MIN),
                         np.full(n_rule, y.min() - 0.25 * (np.ptp(y) or 1))])
    hi = np.concatenate([np.full(n_mf, 1.0),
                         np.full(n_rule, y.max() + 0.25 * (np.ptp(y) or 1))])

    def centres(g):
        out = []
        for j, c in enumerate(inputs):
            if learn_mf:
                genes = g[j * (k + 1):(j + 1) * (k + 1)]
                w = np.maximum(genes, 1e-9)
                out.append(dom[c][0] + (dom[c][1] - dom[c][0]) * np.cumsum(w / w.sum())[:k])
            else:
                out.append(c0[c])
        return out

    def predict(g, zz):
        return gft._fis([zz[c] for c in inputs], centres(g), g[n_mf:])

    theta0 = float(np.std(y)) or 1.0

    def loss(g):
        return gft.rmse(y, predict(g, z)) / theta0

    # seed the GA at the quantile placement, like gft
    x0 = 0.5 * (lo + hi)
    if learn_mf:
        for j, c in enumerate(inputs):
            zc = np.clip((c0[c] - dom[c][0]) / (dom[c][1] - dom[c][0]), 1e-3, 1 - 1e-3)
            x0[j * (k + 1):(j + 1) * (k + 1)] = np.clip(
                np.diff(np.concatenate(([0.0], zc, [1.0]))), gft.GAP_MIN, 1.0)

    g, _ = gft.genetic_optimize(loss, lo, hi, pop=pop, gens=gens, seed=seed,
                                x0=x0, verbose=False, label="leaf")
    return dict(inputs=inputs, target=target, k=k, learn_mf=learn_mf,
                mu=mu, sd=sd, c0=c0, dom=dom, n_mf=n_mf, genome=g)


def _predict_leaf(model, frame):
    inp = model["inputs"]
    z = {c: (frame[c].to_numpy(float) - model["mu"][c]) / model["sd"][c] for c in inp}
    k, g = model["k"], model["genome"]
    cent = []
    for j, c in enumerate(inp):
        if model["learn_mf"]:
            genes = g[j * (k + 1):(j + 1) * (k + 1)]
            w = np.maximum(genes, 1e-9)
            dlo, dhi = model["dom"][c]
            cent.append(dlo + (dhi - dlo) * np.cumsum(w / w.sum())[:k])
        else:
            cent.append(model["c0"][c])
    return gft._fis([z[c] for c in inp], cent, g[model["n_mf"]:])


# ==========================================================================
# 3. Scoring a candidate on held-out theta
# ==========================================================================

def _score(train, val, inputs, target, **fit_kw):
    """Fit standalone on train, score the LEAF's theta on val. Ranking metric is
    per-unit rho (does theta_hat move with theta), with R2 alongside."""
    m = _fit_leaf(train, list(inputs), target, **fit_kw)
    yh = _predict_leaf(m, val)
    y = val[target].to_numpy(float)
    return {"inputs": "+".join(inputs),
            "rho": gft._unit_corr(val["unit"].to_numpy(), y, yh),
            "R2": gft.r2(y, yh),
            "NRMSE": gft.rmse(y, yh) / (float(np.std(y)) or 1.0)}


def _prep(frame, sensors, residual, ref_cycles, smooth_span):
    """Same per-unit residual + EWMA gft applies, so results transfer."""
    if residual:
        frame = gft.residualize(frame, sensors, gft.CONDITIONS, ref_cycles)
    return gft.smooth_inputs(frame, sensors, smooth_span)


# ==========================================================================
# 4. The search
# ==========================================================================

def search_target(feats, target, k=3, all_sensors=False, residual=True,
                  ref_cycles=20, smooth_span=7, gens=120, pop=50, seed=0,
                  top=8):
    """Rank every k-subset of the candidate sensors for one theta target."""
    pool = SENSORS if all_sensors else CANDIDATES.get(target, SENSORS)
    pool = [s for s in pool if s in feats.columns]
    combos = list(itertools.combinations(pool, k))

    # split ONCE by unit; residualize/smooth each side once for all combos
    train, val, _ = ds.split(feats, seed=seed)
    tr = _prep(train, pool, residual, ref_cycles, smooth_span)
    va = _prep(val, pool, residual, ref_cycles, smooth_span)

    print(f"\n{target}: {len(combos)} combinations of {len(pool)} sensors "
          f"{'(all sensors)' if all_sensors else '(physical prefilter)'}")
    rows = []
    for i, combo in enumerate(combos):
        rows.append(_score(tr, va, combo, target, k=k, gens=gens, pop=pop, seed=seed))
        if (i + 1) % 10 == 0 or i + 1 == len(combos):
            print(f"  {i + 1}/{len(combos)} tested", end="\r")
    print()
    df = pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)

    base = BASELINE.get(target, [])
    baseset = frozenset(base)
    # NB: compare as SETS. itertools.combinations emits sensors in pool order, so
    # the engineering choice ["T48","P40","Nc"] comes out as "Nc+T48+P40" and a
    # plain string comparison silently never matches (it made is_baseline all-False
    # in --all-sensors runs, so the "engineering choice ranks #N" line never fired).
    df["is_baseline"] = df["inputs"].map(lambda s: frozenset(s.split("+")) == baseset)
    rank = df.index[df["is_baseline"]].tolist()
    print(f"top {top} by per-unit rho:")
    print(df.head(top).to_string(index=False))
    if rank:
        r = rank[0]
        print(f"  engineering choice ({'+'.join(base)}) ranks "
              f"#{r + 1}/{len(df)}: rho={df.loc[r,'rho']:.3f}  R2={df.loc[r,'R2']:.3f}")
    else:
        print(f"  engineering choice ({'+'.join(base)}) not in candidate pool "
              f"(try --all-sensors)")
    return df


# ==========================================================================
# 5. Confirm: seed-robustness + cross-spool specificity
# ==========================================================================
# The raw search is ONE split, ONE seed, and its leaderboard top is bunched inside
# the seed-to-seed noise. Two questions the leaderboard cannot answer:
#   (a) is a combo's rho REAL or a lucky seed?          -> refit over N seeds
#   (b) does it diagnose ITS component, or just read     -> specificity =
#       global damage (the T48-everywhere problem)?         rho_own - rho_foil
# A combo with high rho_own AND high specificity is a genuine, component-specific
# diagnostic. A combo with high rho_own but ~0 specificity is riding a global
# damage signal and would quietly destroy the per-component interpretability.

def _score_multi(feats, inputs, target, foil, seeds, residual=True,
                 ref_cycles=20, smooth_span=7, gens=120, pop=50):
    """Fit `inputs`->target across seeds; also fit the SAME inputs to the opposite
    spool's `foil` target. Returns per-seed own/foil rho and R2."""
    pool = list(inputs)
    own_rho, own_r2, foil_rho = [], [], []
    for s in seeds:
        tr, va, _ = ds.split(feats, seed=s)
        trp = _prep(tr, pool, residual, ref_cycles, smooth_span)
        vap = _prep(va, pool, residual, ref_cycles, smooth_span)
        u = vap["unit"].to_numpy()

        m = _fit_leaf(trp, pool, target, gens=gens, pop=pop, seed=s)
        yh = _predict_leaf(m, vap)
        y = vap[target].to_numpy(float)
        own_rho.append(gft._unit_corr(u, y, yh))
        own_r2.append(gft.r2(y, yh))

        if foil and foil in vap.columns:                 # SAME inputs -> foil theta
            mf = _fit_leaf(trp, pool, foil, gens=gens, pop=pop, seed=s)
            yhf = _predict_leaf(mf, vap)
            yf = vap[foil].to_numpy(float)
            foil_rho.append(gft._unit_corr(u, yf, yhf))
    own_rho, own_r2 = np.array(own_rho), np.array(own_r2)
    fr = np.array(foil_rho) if foil_rho else np.array([np.nan])
    return {"inputs": "+".join(inputs),
            "rho": own_rho.mean(), "rho_sd": own_rho.std(),
            "R2": own_r2.mean(), "R2_sd": own_r2.std(),
            "foil_rho": fr.mean(),
            "specificity": own_rho.mean() - fr.mean()}


def confirm_target(feats, target, shortlist, seeds=(0, 1, 2, 3, 4), **fit_kw):
    """Re-evaluate a shortlist of combos for one target across `seeds`, and add the
    cross-spool specificity. The engineering BASELINE is always included."""
    foil = FOIL.get(target)
    combos = [tuple(c.split("+")) for c in shortlist]
    base = tuple(BASELINE.get(target, []))
    # set-compare, so a shortlist entry like "Nc+T48+P40" is recognised as the
    # baseline ["T48","P40","Nc"] rather than being appended a second time
    if base and not any(frozenset(c) == frozenset(base) for c in combos):
        combos.append(base)

    print(f"\n{target}: confirming {len(combos)} combos over {len(seeds)} seeds "
          f"(foil = {foil})")
    rows = []
    for i, combo in enumerate(combos):
        r = _score_multi(feats, combo, target, foil, seeds, **fit_kw)
        r["is_baseline"] = (frozenset(combo) == frozenset(base))
        rows.append(r)
        print(f"  {i + 1}/{len(combos)} done", end="\r")
    print()
    df = (pd.DataFrame(rows)
          .sort_values("specificity", ascending=False)   # specificity-first sort
          .reset_index(drop=True))
    cols = ["inputs", "rho", "rho_sd", "R2", "foil_rho", "specificity", "is_baseline"]
    print(df[cols].round(3).to_string(index=False))
    print("  rho         = mean per-unit corr on OWN target (higher better)")
    print("  foil_rho    = SAME inputs predicting the OTHER spool (lower = more specific)")
    print("  specificity = rho - foil_rho. Near 0 => the combo reads GLOBAL damage,")
    print("                not this component. Prefer high rho AND high specificity.")
    return df


def _shortlist_from_csv(path, target, n):
    """Top-n combos for a target from an earlier search's CSV (by rho)."""
    df = pd.read_csv(path)
    df = df[df["target"] == target].sort_values("rho", ascending=False)
    return df["inputs"].head(n).tolist()


def _prescreen(feats, target, all_combos, n, seed=0, min_rho=0.0, **fit_kw):
    """Cheap ONE-seed specificity screen over the WHOLE candidate list, to pick the
    shortlist for the expensive multi-seed confirm.

    Why not just confirm the top-n by rho? Because rho and specificity disagree:
    the most component-SPECIFIC combo can sit well down the rho ranking (and a
    not-yet-converged search only fixes the ordering loosely anyway). This scans
    every combo once, ranks by (single-seed) specificity, and returns the top n --
    so a genuinely specific combo that the rho leaderboard buried still gets its
    full multi-seed hearing. min_rho drops combos too weak to be worth confirming
    regardless of specificity."""
    foil = FOIL.get(target)
    rows = []
    for i, combo in enumerate(all_combos):
        inp = tuple(combo.split("+"))
        r = _score_multi(feats, inp, target, foil, seeds=(seed,), **fit_kw)
        rows.append(r)
        if (i + 1) % 20 == 0 or i + 1 == len(all_combos):
            print(f"  prescreen {i + 1}/{len(all_combos)}", end="\r")
    print()
    df = pd.DataFrame(rows)
    df = df[df["rho"] >= min_rho]
    df = df.sort_values("specificity", ascending=False)
    return df["inputs"].head(n).tolist()


def _all_combos_from_csv(path, target):
    df = pd.read_csv(path)
    return df[df["target"] == target]["inputs"].tolist()


def main():
    p = argparse.ArgumentParser(description="Per-leaf sensor-input ablation.")
    p.add_argument("--target", choices=THETA, help="one target (default: all four)")
    p.add_argument("--k", type=int, default=3, help="inputs per FIS (default 3)")
    p.add_argument("--all-sensors", action="store_true",
                   help="ignore the physical prefilter; search all 13 sensors")
    p.add_argument("--gens", type=int, default=120)
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--smooth", type=int, default=7)
    p.add_argument("--out", default="leaf_search.csv")
    # --- confirm mode -----------------------------------------------------
    p.add_argument("--confirm", metavar="CSV", nargs="?", const="leaf_search.csv",
                   help="seed-robustness + cross-spool specificity on the top-N "
                        "combos from an earlier search CSV (default leaf_search.csv)")
    p.add_argument("--topn", type=int, default=5,
                   help="how many combos per target to confirm (default 5)")
    p.add_argument("--seeds", type=int, default=5,
                   help="number of seeds for confirm mode (default 5)")
    p.add_argument("--by-specificity", action="store_true",
                   help="pick the confirm shortlist by a cheap 1-seed specificity "
                        "PRESCREEN over the WHOLE csv, not by top-rho. Surfaces "
                        "component-specific combos the rho leaderboard buried.")
    p.add_argument("--min-rho", type=float, default=0.35,
                   help="in --by-specificity, drop combos below this rho before "
                        "ranking on specificity (default 0.35)")
    a = p.parse_args()

    feats = ds.pooled()
    targets = [a.target] if a.target else THETA
    fit_kw = dict(residual=not a.no_residual, smooth_span=a.smooth,
                  gens=a.gens, pop=a.pop)

    # ---- CONFIRM: validate a shortlist instead of searching --------------
    if a.confirm:
        seeds = tuple(range(a.seeds))
        allres = []
        for t in targets:
            if a.by_specificity:
                all_combos = _all_combos_from_csv(a.confirm, t)
                print(f"\n{t}: specificity prescreen over {len(all_combos)} combos "
                      f"(1 seed) -> top {a.topn} by specificity, min_rho={a.min_rho}")
                shortlist = _prescreen(feats, t, all_combos, a.topn,
                                       seed=0, min_rho=a.min_rho, **fit_kw)
            else:
                shortlist = _shortlist_from_csv(a.confirm, t, a.topn)
            df = confirm_target(feats, t, shortlist, seeds=seeds, **fit_kw)
            df.insert(0, "target", t)
            allres.append(df)
        out = pd.concat(allres, ignore_index=True)
        out.to_csv("leaf_confirm.csv", index=False)
        print("\n" + "=" * 70)
        print("ADOPT (best by specificity, with rho stable across seeds)")
        print("=" * 70)
        best = out.sort_values("specificity", ascending=False).groupby("target").first()
        print(best[["inputs", "rho", "rho_sd", "R2", "specificity"]].round(3).to_string())
        print("\nfull table -> leaf_confirm.csv")
        print("A combo only earns adoption if rho is high, rho_sd is small (stable "
              "across seeds), AND specificity is clearly positive.")
        return

    # ---- SEARCH ----------------------------------------------------------
    allres = []
    for t in targets:
        df = search_target(feats, t, k=a.k, all_sensors=a.all_sensors,
                           seed=a.seed, **fit_kw)
        df.insert(0, "target", t)
        allres.append(df)
    out = pd.concat(allres, ignore_index=True)
    out.to_csv(a.out, index=False)

    print("\n" + "=" * 70)
    print("BEST INPUT SET PER TARGET (by held-out per-unit rho)")
    print("=" * 70)
    best = out.sort_values("rho", ascending=False).groupby("target").first()
    print(best[["inputs", "rho", "R2", "NRMSE"]].round(3).to_string())
    print(f"\nfull table -> {a.out}")
    print("Next: python leaf_search.py --confirm  (seed-robustness + specificity)")
    print("Then adopt by editing gft._LEAVES and re-running run.py.")


if __name__ == "__main__":
    main()
