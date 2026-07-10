r"""
gft.py -- a small GENETIC FUZZY TREE for N-CMAPSS RUL, with per-component
supervision.

A REAL tree of small zero-order Sugeno sub-FIS (each 1-3 inputs -> 1 output),
trained by a genetic algorithm on a HYBRID loss:

  * LEAF FIS predict the physical health parameters theta directly
    ([sensors] -> theta_hat), supervised by a theta RMSE term -- so every leaf
    output is comparable to a real modifier and can be plotted vs the truth.
  * The upper (spool + root) FIS have NO target of their own; they are pulled
    only by the RUL term (NASA score). The GA optimises the whole tree at once.

Topology follows the engine (C-MAPSS, Fig. 1): HP shaft = HPC+HPT, LP shaft =
Fan+LPC+LPT. In DS02 only HPT and LPT degrade, so the default tree supervises
the HPT/LPT modifiers and aggregates them per shaft:

    sensors --> [hpt_eff][hpt_flow]      [lpt_eff][lpt_flow]     (leaf -> theta)
                     \       /                \       /
                     [ hp ] (HP damage)       [ lp ] (LP damage)  (spool, in[0,1])
                          \                      /
                          [ RUL(hp, lp, age) ]                    (root -> RUL)

Each FIS: Ruspini partition (triangles summing to 1) + product t-norm on a full
rule grid -> firing weights sum to 1, so the Sugeno output is a weighted average
of per-rule singletons. Antecedent centres are FIXED from data (sensors: data
quantiles in z-space; theta-node inputs: quantiles of that theta; [0,1] nodes:
0..1). The GA only tunes the singletons. Edit TREE to grow/prune.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ==========================================================================
# 1. Fuzzy inference (zero-order Sugeno on a Ruspini partition)
# ==========================================================================

def _centers(x, n):
    """n ordered set-centres at data quantiles (spread out if near-constant)."""
    c = np.unique(np.quantile(np.asarray(x, float), np.linspace(0, 1, n)))
    if c.size < n:
        lo, hi = float(np.min(x)), float(np.max(x))
        s = (hi - lo) or 1.0
        c = np.linspace(lo - 0.01 * s, hi + 0.01 * s, n)
    return c


def _memberships(x, c):
    """(n_samples, n_terms) triangular memberships; each row sums to 1."""
    x = np.asarray(x, float).ravel()
    m = c.size
    M = np.empty((x.size, m))
    for i in range(m):
        if i == 0:
            xp, fp = [c[0], c[1]], [1.0, 0.0]
        elif i == m - 1:
            xp, fp = [c[m - 2], c[m - 1]], [0.0, 1.0]
        else:
            xp, fp = [c[i - 1], c[i], c[i + 1]], [0.0, 1.0, 0.0]
        M[:, i] = np.interp(x, xp, fp)
    return M


def _fis(cols, centers, singletons):
    """One FIS: product firing over the full grid, then weighted-average."""
    F = _memberships(cols[0], centers[0])
    for d in range(1, len(centers)):
        Md = _memberships(cols[d], centers[d])
        F = (F[:, :, None] * Md[:, None, :]).reshape(F.shape[0], -1)
    denom = F.sum(1)
    denom[denom == 0] = 1.0
    return (F @ singletons) / denom


# ==========================================================================
# 2. Tree architecture  (name, [inputs], theta_target_or_None)
# ==========================================================================
# An input is a data column (cruise sensor / "age") OR an earlier node's name.
# A node with a theta target is a supervised leaf (output = theta_hat); a node
# with target None is latent (spool damage in [0,1], or the RUL root).
# Children must precede parents. Max 3 inputs per node. Root must be named "RUL".

TREE = [
    ("hpt_eff",  ["T48", "P40", "P45"], "HPT_eff_mod"),
    ("hpt_flow", ["T48", "P40", "P45"], "HPT_flow_mod"),
    ("lpt_eff",  ["T50", "P50", "epr"], "LPT_eff_mod"),
    ("lpt_flow", ["T50", "P50", "epr"], "LPT_flow_mod"),
    ("hp",  ["hpt_eff", "hpt_flow"], None),      # HP-shaft damage  (in [0,1])
    ("lp",  ["lpt_eff", "lpt_flow"], None),      # LP-shaft damage  (in [0,1])
    ("RUL", ["hp", "lp", "age"], None),          # prognosis        (root)
]

CONDITIONS = ["alt", "Mach", "TRA", "T2"]


def _node_names(tree):
    return {n for n, _, _ in tree}


def _leaf_sensors(tree):
    names = _node_names(tree)
    return sorted({c for _, ins, _ in tree for c in ins
                   if c not in names and c != "age"})


def build_tree(frame, tree=TREE, n_terms=5):
    """Resolve inputs against the frame and fix the fuzzy centres. Also record
    each node's OUTPUT domain (theta quantiles / [0,1]) so parents can place
    their own input centres consistently."""
    out_dom, meta = {}, []
    for name, inputs, target in tree:
        tgt = target if (target in frame.columns) else None
        ins = []
        for inp in inputs:
            if inp in out_dom:                                 # child node output
                ins.append({"kind": "node", "src": inp, "centers": out_dom[inp]})
            elif inp in frame.columns:                         # sensor / age
                x = frame[inp].to_numpy(float)
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                ins.append({"kind": "col", "src": inp, "mu": mu, "sd": sd,
                            "centers": _centers((x - mu) / sd, n_terms)})
        if not ins:
            continue
        n_rules = int(np.prod([len(i["centers"]) for i in ins]))
        # output domain: theta quantiles for supervised leaves, else [0,1]
        out_dom[name] = (_centers(frame[tgt].to_numpy(float), n_terms)
                         if tgt is not None else np.linspace(0, 1, n_terms))
        meta.append({"name": name, "inputs": ins, "n_rules": n_rules,
                     "target": tgt, "root": name == "RUL"})
    return meta


def n_params(meta):
    return int(sum(m["n_rules"] for m in meta))


def genome_bounds(meta, frame, rul_hi, cons_pad=0.25):
    """Singleton box per node: theta range for supervised leaves, [0,1] for
    latent spool nodes, [0, rul_hi] for the root."""
    lo, hi = [], []
    for m in meta:
        if m["root"]:
            a, b = 0.0, rul_hi
        elif m["target"] is not None:
            y = frame[m["target"]].to_numpy(float)
            span = (y.max() - y.min()) or 1.0
            a, b = y.min() - cons_pad * span, y.max() + cons_pad * span
        else:
            a, b = 0.0, 1.0
        lo += [a] * m["n_rules"]
        hi += [b] * m["n_rules"]
    return np.array(lo), np.array(hi)


def predict_tree(frame, meta, genome):
    """Evaluate the whole tree; returns {node_name: output_array}."""
    vals, k = {}, 0
    for m in meta:
        s = genome[k:k + m["n_rules"]]
        k += m["n_rules"]
        cols, centers = [], []
        for i in m["inputs"]:
            if i["kind"] == "node":
                cols.append(vals[i["src"]])            # already in its domain
            else:
                cols.append((frame[i["src"]].to_numpy(float) - i["mu"]) / i["sd"])
            centers.append(i["centers"])
        y = _fis(cols, centers, s)
        # latent spool nodes are clamped to [0,1]; theta leaves / root unclamped
        vals[m["name"]] = y if (m["root"] or m["target"] is not None) else np.clip(y, 0, 1)
    return vals


# ==========================================================================
# 3. Losses
# ==========================================================================

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def nasa_score(y_true, y_pred):
    """Asymmetric PHM score: late (dangerous) predictions cost more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))))


# ==========================================================================
# 4. Genetic algorithm  (real-coded, minimises a scalar loss)
# ==========================================================================

def _tournament(fit, rng):
    idx = rng.choice(len(fit), 3, replace=False)
    return idx[np.argmin(fit[idx])]


def genetic_optimize(loss, lo, hi, pop=200, gens=120, seed=1, x0=None,
                     verbose=True, label=""):
    """Tournament selection + BLX-alpha crossover + Gaussian mutation, with
    elitism and a slowly cooling mutation step. Returns (best, history)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n, span = len(lo), hi - lo
    P = rng.uniform(lo, hi, (pop, n))
    if x0 is not None:
        P[0] = np.clip(x0, lo, hi)
    fit = np.array([loss(g) for g in P])
    best, best_fit, sigma, history = P[fit.argmin()].copy(), fit.min(), 0.3, []
    for t in range(gens):
        newP = [P[i].copy() for i in fit.argsort()[:2]]
        while len(newP) < pop:
            pa, pb = P[_tournament(fit, rng)], P[_tournament(fit, rng)]
            lo_c = np.minimum(pa, pb) - 0.3 * np.abs(pa - pb)
            hi_c = np.maximum(pa, pb) + 0.3 * np.abs(pa - pb)
            child = rng.uniform(lo_c, hi_c)
            mask = rng.random(n) < 0.35
            child[mask] += rng.normal(0, 1, mask.sum()) * sigma * span[mask]
            newP.append(np.clip(child, lo, hi))
        P = np.array(newP)
        fit = np.array([loss(g) for g in P])
        if fit.min() < best_fit:
            best_fit, best = fit.min(), P[fit.argmin()].copy()
        history.append(best_fit)
        sigma = max(0.1, sigma * 0.97)
        if verbose and (t % 10 == 0 or t == gens - 1):
            print(f"  [{label}] gen {t + 1:3d}/{gens}  best={best_fit:.4f}  sigma={sigma:.3f}")
    return best, history


# ==========================================================================
# 5. Condition residuals  (strip the dominant flight-condition effect)
# ==========================================================================

def _design(frame, conds):
    W = frame[conds].to_numpy(float)
    return np.column_stack([np.ones(len(W)), W, W ** 2])


def _healthy_mask(frame):
    if "since_onset" in frame.columns:
        m = frame["since_onset"].to_numpy() <= 0
        if m.sum() >= 20:
            return m
    return frame["age"].to_numpy() <= np.median(frame["age"].to_numpy())


def fit_baselines(frame, sensors, conds):
    X, mask = _design(frame, conds), _healthy_mask(frame)
    return {s: np.linalg.lstsq(X[mask], frame[s].to_numpy(float)[mask], rcond=None)[0]
            for s in sensors if s in frame.columns}


def apply_baselines(frame, coefs, conds):
    X, f = _design(frame, conds), frame.copy()
    for s, b in coefs.items():
        f[s] = f[s].to_numpy(float) - X @ b
    return f


# ==========================================================================
# 6. Fit / predict
# ==========================================================================

def fit_gft(frame, tree=TREE, n_terms=3, pop=80, gens=120, residualize=True,
            rul_cap=None, theta_weight=1.0, seed=0, verbose=True):
    """Train the tree end-to-end on a hybrid loss:
        loss = NASA_score(RUL)  +  theta_weight * mean_leaf( RMSE(theta)/range ).
    theta_weight=0 recovers pure RUL training. Returns a plain-dict model."""
    frame = frame.reset_index(drop=True)
    conds = [c for c in CONDITIONS if c in frame.columns]
    baselines = None
    if residualize and len(conds) >= 2:
        baselines = fit_baselines(frame, _leaf_sensors(tree), conds)
        frame = apply_baselines(frame, baselines, conds)

    meta = build_tree(frame, tree, n_terms)
    y = frame["RUL"].to_numpy(float)
    if rul_cap:
        y = np.minimum(y, float(rul_cap))
    lo, hi = genome_bounds(meta, frame, float(y.max()))

    theta_nodes = [m for m in meta if m["target"] is not None]
    Yt = {m["name"]: frame[m["target"]].to_numpy(float) for m in theta_nodes}
    span = {m["name"]: (Yt[m["name"]].max() - Yt[m["name"]].min()) or 1.0
            for m in theta_nodes}

    def loss(g):
        vals = predict_tree(frame, meta, g)
        l = nasa_score(y, vals["RUL"])
        if theta_nodes:
            l += theta_weight * np.mean(
                [rmse(Yt[m["name"]], vals[m["name"]]) / span[m["name"]]
                 for m in theta_nodes])
        return l

    if verbose:
        print("tree:", " ".join(m["name"] for m in meta),
              "| params:", n_params(meta),
              "| supervised leaves:", [m["target"] for m in theta_nodes])
    genome, history = genetic_optimize(loss, lo, hi, pop=pop, gens=gens,
                                       seed=seed, x0=0.5 * (lo + hi),
                                       verbose=verbose, label="GFT")
    return {"tree": tree, "meta": meta, "genome": genome, "history": history,
            "baselines": baselines, "conds": conds, "rul_cap": rul_cap,
            "theta_weight": theta_weight}


def predict_gft(frame, model):
    """Full inference. Returns unit, cycle, each theta_hat, each latent node
    health (*_h), and RUL_hat."""
    frame = frame.reset_index(drop=True)
    if model["baselines"] is not None:
        frame = apply_baselines(frame, model["baselines"], model["conds"])
    vals = predict_tree(frame, model["meta"], model["genome"])
    out = frame[["unit", "cycle"]].copy()
    for m in model["meta"]:
        if m["root"]:
            continue
        if m["target"] is not None:
            out[m["target"] + "_hat"] = vals[m["name"]]      # comparable to theta
        else:
            out[m["name"] + "_h"] = vals[m["name"]]          # latent [0,1]
    out["RUL_hat"] = np.clip(vals["RUL"], 0.0, model["rul_cap"])
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def inspect(model):
    """Print each FIS's inputs, target, grid shape and learned singleton range."""
    g, k = model["genome"], 0
    for m in model["meta"]:
        s = g[k:k + m["n_rules"]]
        k += m["n_rules"]
        shape = tuple(len(i["centers"]) for i in m["inputs"])
        srcs = [i["src"] for i in m["inputs"]]
        tgt = m["target"] or ("RUL" if m["root"] else "-")
        print(f"{m['name']:9s} <- {srcs}  ->{tgt:14s} grid{shape}  "
              f"out=[{s.min():.3g}, {s.max():.3g}]")


# ==========================================================================
# 7. Self-test on a signal-bearing synthetic frame
# ==========================================================================

def _synthetic_frame(units=6, cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, units + 1):
        eol = cycles + int(rng.integers(-8, 8))
        onset = int(0.5 * eol)
        for k in range(1, eol + 1):
            prog = max(0, k - onset) / max(1, eol - onset)
            d = 0.03 * (0.3 * k / eol + 0.7 * prog ** 1.6)     # damage >= 0
            alt = 30000 + rng.normal(0, 6000)
            mach = 0.75 + rng.normal(0, 0.05)
            tra = 65 + 10 * np.sin(k / 7.0) + rng.normal(0, 4)
            t2 = 520 + rng.normal(0, 8)
            fc = 0.02 * (alt - 30000) / 1000 + 3 * (mach - 0.75) + 0.05 * (tra - 65)
            def sens(gain):
                return 1.0 + fc + gain * d + rng.normal(0, 0.01)
            rows.append({
                "unit": u, "cycle": k, "age": float(k),
                "alt": alt, "Mach": mach, "TRA": tra, "T2": t2,
                "T48": sens(9.0), "P40": sens(2.0), "P45": sens(1.5),   # HPT
                "T50": sens(7.0), "P50": sens(1.5), "epr": sens(1.0),   # LPT
                "HPT_eff_mod": -d,        "HPT_flow_mod": -0.6 * d,
                "LPT_eff_mod": -0.8 * d,  "LPT_flow_mod": -0.5 * d,
                "since_onset": float(max(0, k - onset)),
                "RUL": float(eol - k)})
    return pd.DataFrame(rows)


def _self_test():
    df = _synthetic_frame(units=8, cycles=60, seed=1)
    train = df[df.unit <= 6].reset_index(drop=True)
    test = df[df.unit > 6].reset_index(drop=True)

    c = _centers(train["T48"].to_numpy(), 3)
    M = _memberships(np.linspace(-3, 3, 2000), c)
    print(f"Ruspini partition-of-unity max|sum-1| = {np.abs(M.sum(1) - 1).max():.1e}\n")

    model = fit_gft(train, gens=120, pop=80, theta_weight=1.0, verbose=True)
    pr_tr, pr_te = predict_gft(train, model), predict_gft(test, model)

    theta_cols = [m["target"] for m in model["meta"] if m["target"]]
    th_tr = np.mean([rmse(train[c], pr_tr[c + "_hat"]) for c in theta_cols])
    th_te = np.mean([rmse(test[c], pr_te[c + "_hat"]) for c in theta_cols])
    base = rmse(test["RUL"], np.full(len(test), train["RUL"].mean()))
    print("\n--- results ---")
    print(f"theta RMSE  train={th_tr:.4f}   test={th_te:.4f}")
    print(f"RUL RMSE    train={rmse(pr_tr['RUL_hat'], train['RUL']):.2f}"
          f"   test={rmse(pr_te['RUL_hat'], test['RUL']):.2f}"
          f"   (baseline mean={base:.2f})")
    print(f"RUL NASA    test={nasa_score(test['RUL'], pr_te['RUL_hat']):.3f}\n")
    inspect(model)


if __name__ == "__main__":
    _self_test()