r"""
gft.py -- a small GENETIC FUZZY TREE for N-CMAPSS RUL.

This is a REAL tree, not a bag of independent FIS. Small zero-order Sugeno
sub-FIS (each 1-3 inputs -> 1 output) are NESTED: leaf FIS read cruise sensors
and output a latent "module health" in [0, 1]; those feed spool-level FIS; the
spool healths feed the root FIS that outputs RUL. The whole tree is trained
END-TO-END by a genetic algorithm on the RUL score -- only the root has a
target, every intermediate node is discovered by the GA.

Topology follows the engine (C-MAPSS, Fig. 1): the LP shaft carries Fan+LPC+LPT
and the HP shaft carries HPC+HPT, so:

    sensors --> [fan] [lpc] [hpc] [hpt] [lpt]        (leaf health FIS)
                   \    |            |   /
        LP shaft ---[ lp ]      [ hp ]--- HP shaft   (spool health FIS)
                        \        /
                        [ RUL(lp, hp, age) ]         (root prognosis FIS)

Every FIS: inputs fuzzified with a Ruspini partition (triangles summing to 1),
product t-norm on a full rule grid -> firing weights sum to 1, so the Sugeno
output is a plain weighted average of per-rule singletons. Antecedent centres
are FIXED from the data (or evenly in [0,1] for node inputs); the GA only tunes
the singletons. Edit TREE to grow or prune the tree.
"""

from __future__ import annotations  # noqa: E402
import numpy as np
import pandas as pd


# ==========================================================================
# 1. Fuzzy inference (zero-order Sugeno on a Ruspini partition)
# ==========================================================================

def _centers(x: np.ndarray, n: int) -> np.ndarray:
    """n ordered set-centres at data quantiles (spread out if near-constant)."""
    c = np.unique(np.quantile(np.asarray(x, float), np.linspace(0, 1, n)))
    if c.size < n:
        lo, hi = float(np.min(x)), float(np.max(x))
        s = (hi - lo) or 1.0
        c = np.linspace(lo - 0.01 * s, hi + 0.01 * s, n)
    return c


def _memberships(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(n_samples, n_terms) triangular memberships; each row sums to 1."""
    x = np.asarray(x, float).ravel()
    m = c.size
    M = np.empty((x.size, m))
    for i in range(m):
        if i == 0:                       # left shoulder
            xp, fp = [c[0], c[1]], [1.0, 0.0]
        elif i == m - 1:                 # right shoulder
            xp, fp = [c[m - 2], c[m - 1]], [0.0, 1.0]
        else:                            # interior triangle
            xp, fp = [c[i - 1], c[i], c[i + 1]], [0.0, 1.0, 0.0]
        M[:, i] = np.interp(x, xp, fp)
    return M


def _fis(cols, centers, singletons: np.ndarray) -> np.ndarray:
    """One FIS: product firing over the full grid, then weighted-average."""
    F = _memberships(cols[0], centers[0])
    for d in range(1, len(centers)):
        Md = _memberships(cols[d], centers[d])
        F = (F[:, :, None] * Md[:, None, :]).reshape(F.shape[0], -1)
    denom = F.sum(1)
    denom[denom == 0] = 1.0
    return (F @ singletons) / denom


# ==========================================================================
# 2. Tree architecture  (edit this list to add/remove nodes or inputs)
# ==========================================================================
# (node_name, [inputs]).  An input is a data column (a cruise sensor / "age")
# OR the name of an earlier node (its health output, in [0, 1]).
# Children must be listed before their parents.  Max 3 inputs per node.

TREE = [
    ("fan", ["Nf",  "P21",  "P15"]),      # LP: fan
    ("lpc", ["T24", "P24",  "W22"]),      # LP: low-pressure compressor
    ("hpc", ["T30", "Ps30", "Nc"]),       # HP: high-pressure compressor
    ("hpt", ["T48", "P40",  "P45"]),      # HP: high-pressure turbine (degrades)
    ("lpt", ["T50", "P50",  "epr"]),      # LP: low-pressure turbine (degrades)
    ("hp",  ["hpc", "hpt"]),              # HP-shaft health
    ("lp",  ["fan", "lpc", "lpt"]),       # LP-shaft health
    ("RUL", ["hp",  "lp",  "age"]),       # prognosis
]

CONDITIONS = ["alt", "Mach", "TRA", "T2"]


def _node_names(tree):
    return {n for n, _ in tree}


def _leaf_sensors(tree):
    names = _node_names(tree)
    return sorted({c for _, ins in tree for c in ins
                   if c not in names and c != "age"})


def build_tree(frame: pd.DataFrame, tree=TREE, n_terms: int = 3):
    """Resolve each node's inputs against the frame, fixing the fuzzy centres.
    Data inputs are z-scored (centres in z-space); node inputs use [0,1]."""
    built, meta = set(), []
    for name, inputs in tree:
        ins = []
        for inp in inputs:
            if inp in built:                                   # child FIS output
                ins.append({"kind": "node", "src": inp,
                            "centers": np.linspace(0, 1, n_terms)})
            elif inp in frame.columns:                         # sensor / age
                x = frame[inp].to_numpy(float)
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                ins.append({"kind": "col", "src": inp, "mu": mu, "sd": sd,
                            "centers": _centers((x - mu) / sd, n_terms)})
            # else: input not available -> silently dropped
        if not ins:
            continue
        n_rules = int(np.prod([len(i["centers"]) for i in ins]))
        meta.append({"name": name, "inputs": ins,
                     "n_rules": n_rules, "root": name == "RUL"})
        built.add(name)
    return meta


def n_params(meta) -> int:
    return int(sum(m["n_rules"] for m in meta))


def genome_bounds(meta, rul_hi: float):
    """Singleton box: [0,1] for health nodes, [0, rul_hi] for the root."""
    lo, hi = [], []
    for m in meta:
        top = rul_hi if m["root"] else 1.0
        lo += [0.0] * m["n_rules"]
        hi += [top] * m["n_rules"]
    return np.array(lo), np.array(hi)


def predict_tree(frame: pd.DataFrame, meta, genome: np.ndarray) -> dict:
    """Evaluate the whole tree; returns {node_name: output_array}."""
    vals, k = {}, 0
    for m in meta:
        s = genome[k:k + m["n_rules"]]
        k += m["n_rules"]
        cols, centers = [], []
        for i in m["inputs"]:
            if i["kind"] == "node":
                cols.append(np.clip(vals[i["src"]], 0, 1))
            else:
                cols.append((frame[i["src"]].to_numpy(float) - i["mu"]) / i["sd"])
            centers.append(i["centers"])
        y = _fis(cols, centers, s)
        vals[m["name"]] = y if m["root"] else np.clip(y, 0.0, 1.0)
    return vals


# ==========================================================================
# 3. Losses
# ==========================================================================

def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM score: late (dangerous) predictions cost more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))))


# ==========================================================================
# 4. Genetic algorithm  (real-coded, minimises a scalar loss)
# ==========================================================================

def _tournament(fit, rng):
    idx = rng.choice(len(fit), 3, replace=False)
    return idx[np.argmin(fit[idx])]


def genetic_optimize(loss, lo, hi, pop=80, gens=100, seed=0, x0=None,
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
        newP = [P[i].copy() for i in fit.argsort()[:2]]          # elitism
        while len(newP) < pop:
            pa, pb = P[_tournament(fit, rng)], P[_tournament(fit, rng)]
            lo_c = np.minimum(pa, pb) - 0.3 * np.abs(pa - pb)
            hi_c = np.maximum(pa, pb) + 0.3 * np.abs(pa - pb)
            child = rng.uniform(lo_c, hi_c)
            mask = rng.random(n) < 0.2
            child[mask] += rng.normal(0, 1, mask.sum()) * sigma * span[mask]
            newP.append(np.clip(child, lo, hi))
        P = np.array(newP)
        fit = np.array([loss(g) for g in P])
        if fit.min() < best_fit:
            best_fit, best = fit.min(), P[fit.argmin()].copy()
        history.append(best_fit)
        sigma = max(0.05, sigma * 0.97)
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

def fit_gft(frame, tree=TREE, n_terms=3, pop=80, gens=120,
            residualize=True, rul_cap=None, seed=0, verbose=True):
    """Train the whole tree end-to-end on the NASA RUL score.
    Returns a plain dict model (topology + genome + history + baselines)."""
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
    lo, hi = genome_bounds(meta, float(y.max()))

    def loss(g):
        return nasa_score(y, predict_tree(frame, meta, g)["RUL"])

    if verbose:
        print("tree:", " ".join(m["name"] for m in meta),
              "| params:", n_params(meta))
    genome, history = genetic_optimize(loss, lo, hi, pop=pop, gens=gens,
                                       seed=seed, x0=0.5 * (lo + hi),
                                       verbose=verbose, label="GFT")
    return {"tree": tree, "meta": meta, "genome": genome, "history": history,
            "baselines": baselines, "conds": conds, "rul_cap": rul_cap}


def predict_gft(frame, model) -> pd.DataFrame:
    """Full inference. Returns unit, cycle, each node health, and RUL_hat."""
    frame = frame.reset_index(drop=True)
    if model["baselines"] is not None:
        frame = apply_baselines(frame, model["baselines"], model["conds"])
    vals = predict_tree(frame, model["meta"], model["genome"])
    out = frame[["unit", "cycle"]].copy()
    for m in model["meta"]:
        if not m["root"]:
            out[m["name"] + "_h"] = vals[m["name"]]
    rul = np.clip(vals["RUL"], 0.0, model["rul_cap"])
    out["RUL_hat"] = rul
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def inspect(model):
    """Print each FIS's grid shape and its learned singletons."""
    g, k = model["genome"], 0
    for m in model["meta"]:
        s = g[k:k + m["n_rules"]]
        k += m["n_rules"]
        shape = tuple(len(i["centers"]) for i in m["inputs"])
        srcs = [i["src"] for i in m["inputs"]]
        print(f"{m['name']:4s} <- {srcs}  grid{shape}  "
              f"singletons=[{s.min():.3g}, {s.max():.3g}]")


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
            def sens(dmg_gain, cond_gain=1.0):     # big condition + small damage
                return 1.0 + cond_gain * fc + dmg_gain * d + rng.normal(0, 0.01)
            rows.append({
                "unit": u, "cycle": k, "age": float(k),
                "alt": alt, "Mach": mach, "TRA": tra, "T2": t2,
                "Nf": sens(0.2), "P21": sens(0.1), "P15": sens(0.1),
                "T24": sens(0.3), "P24": sens(0.2), "W22": sens(0.2),
                "T30": sens(0.5), "Ps30": sens(0.3), "Nc": sens(0.2),
                "T48": sens(9.0), "P40": sens(2.0), "P45": sens(1.5),   # HPT
                "T50": sens(7.0), "P50": sens(1.5), "epr": sens(1.0),   # LPT
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

    model = fit_gft(train, gens=120, pop=80, verbose=True)
    pr_tr, pr_te = predict_gft(train, model), predict_gft(test, model)
    base = rmse(test["RUL"], np.full(len(test), train["RUL"].mean()))
    print("\n--- results ---")
    print(f"RUL RMSE  train={rmse(pr_tr['RUL_hat'], train['RUL']):.2f}"
          f"   test={rmse(pr_te['RUL_hat'], test['RUL']):.2f}"
          f"   (baseline mean={base:.2f})")
    print(f"RUL NASA  test={nasa_score(test['RUL'], pr_te['RUL_hat']):.3f}\n")
    inspect(model)
    show = pr_te.copy(); show["RUL_true"] = test["RUL"].values
    print("\nsample (test unit, late life):")
    print(show[["unit", "cycle", "hp_h", "lp_h", "RUL_hat", "RUL_true"]]
          .tail(6).to_string(index=False))


if __name__ == "__main__":
    _self_test()