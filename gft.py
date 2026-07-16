r"""
gft.py -- a small GENETIC FUZZY TREE for N-CMAPSS RUL, with per-component
supervision.

A REAL tree of small zero-order Sugeno sub-FIS (each 1-3 inputs -> 1 output),
trained by a genetic algorithm on a HYBRID loss:

  * LEAF FIS predict the physical health parameters theta directly
    ([sensors] -> theta_hat), supervised by a theta RMSE term.
  * The upper (spool + root) FIS have NO target of their own; they are pulled
    only by the RUL term (NASA score). The GA optimises the whole tree at once.

    sensors --> [hpt_eff][hpt_flow]      [lpt_eff][lpt_flow]     (leaf -> theta)
                     \       /                \       /
                     [ hp ] (HP damage)       [ lp ] (LP damage)  (spool, in[0,1])
                          \                      /
                            [ RUL(hp, lp) ]                       (root -> RUL)

Each FIS: Ruspini partition (triangles summing to 1) + product t-norm on a full
rule grid -> firing weights sum to 1, so the Sugeno output is a weighted average
of per-rule singletons.

THE GENOME HOLDS BOTH HALVES OF THE FUZZY SYSTEM:
  * the ANTECEDENTS -- where the membership functions sit (section 1b), and
  * the CONSEQUENTS -- one singleton per rule.

Score models with evaluate() / loo_cv() on HELD-OUT units, against the baselines
in baselines() (section 7). Edit TREE to grow/prune.
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


def _fis(cols, centers_, singletons):
    """One FIS: product firing over the full grid, then weighted-average."""
    F = _memberships(cols[0], centers_[0])
    for d in range(1, len(centers_)):
        Md = _memberships(cols[d], centers_[d])
        F = (F[:, :, None] * Md[:, None, :]).reshape(F.shape[0], -1)
    denom = F.sum(1)
    denom[denom == 0] = 1.0
    return (F @ singletons) / denom


# ==========================================================================
# 1b. Learnable membership functions  (the antecedent half of the GFS)
# ==========================================================================
# Under a Ruspini partition the fuzzy sets of a variable ARE its ordered centres
# c_1 < ... < c_k -- there is nothing else to tune. So we evolve them. But a GA
# mutating positions directly keeps producing out-of-order (invalid) partitions.
#
# Fix: evolve the GAPS, not the positions. k+1 positive genes, normalised to sum
# to 1, give k breakpoints inside the padded domain [dlo, dhi]:
#
#       c_i = dlo + (dhi - dlo) * (w_1 + ... + w_i),      w = g / sum(g)
#
# Gaps are positive, so the centres come out ORDERED, always. No repair operator,
# no penalty term, no invalid genome possible. The partition stays Ruspini, so
# "firing strengths sum to 1 -> Sugeno output is a dot product" still holds.
#
# WHY IT MATTERS: theta is ~0 for most of a unit's life, so its data QUANTILES
# drop two of three centres on top of each other at zero and squeeze the whole
# degraded regime onto one triangle's ramp. That is what flattens hp_h after
# cycle ~32. A learnable centre can just move to where the signal actually is.

GAP_MIN, MF_PAD = 0.02, 0.05     # min gap gene; domain padding (frac of span)


def _domain(lo, hi):
    s = (float(hi) - float(lo)) or 1.0
    return float(lo) - MF_PAD * s, float(hi) + MF_PAD * s


def _decode(inp, genes):
    """k ordered centres from k+1 positive gap genes."""
    w = np.maximum(genes, 1e-9)
    return inp["dlo"] + (inp["dhi"] - inp["dlo"]) * np.cumsum(w / w.sum())[:inp["k"]]


def _encode(inp, c):
    """Inverse of _decode. Used to SEED the GA at the old quantile placement, so
    learning the MFs can never start off worse than not learning them."""
    z = np.clip((np.asarray(c, float) - inp["dlo"]) / (inp["dhi"] - inp["dlo"]),
                1e-3, 1 - 1e-3)
    return np.clip(np.diff(np.concatenate(([0.0], z, [1.0]))), GAP_MIN, 1.0)


# ==========================================================================
# 2. Tree architecture  (name, [inputs], theta_target_or_None)
# ==========================================================================
# An input is a data column (cruise sensor / "age") OR an earlier node's name.
# A node with a theta target is a supervised leaf (output = theta_hat); a node
# with target None is latent (spool damage in [0,1], or the RUL root).
# Children must precede parents. Max 3 inputs per node. Root must be named "RUL".
#
# Per-input membership counts: a bare name uses the default n_terms; ("name", k)
# gives that input k fuzzy sets. A node's rule count is the PRODUCT of its
# inputs' set counts, so (5, 3, 3) -> 45 rules.

_LEAVES = [
    ("hpt_eff",  ["T48", "P40", "Nc"],   "HPT_eff_mod"),   # HP: eff  -> speed
    ("hpt_flow", ["T48", "P40", "Ps30"], "HPT_flow_mod"),  # HP: flow -> pressure
    ("lpt_eff",  ["T50", "P50", "Nf"],   "LPT_eff_mod"),   # LP: eff  -> speed
    ("lpt_flow", ["T50", "P50", "P24"],  "LPT_flow_mod"),  # LP: flow -> pressure
    ("hp",  ["hpt_eff", "hpt_flow"], None),      # global HP-shaft damage
    ("lp",  ["lpt_eff", "lpt_flow"], None),      # global LP-shaft damage
]

# DEFAULT: the root sees ONLY the two shaft-damage nodes. `age` is ABLATED -- with
# a few units of similar lifetime it is a shortcut, RUL ~= mean_EOL - age is
# learnable without touching a single sensor, and the GA will take that deal.
TREE     = _LEAVES + [("RUL", ["hp", "lp"], None)]
TREE_AGE = _LEAVES + [("RUL", ["hp", "lp", "age"], None)]   # ablation control

CONDITIONS = ["alt", "Mach", "TRA", "T2"]


def _parse_input(inp, default):
    """('name', k) -> (name, k);  'name' -> (name, default)."""
    if isinstance(inp, (tuple, list)):
        return inp[0], int(inp[1])
    return inp, default


def _leaf_sensors(tree):
    names = {n for n, _, _ in tree}
    return sorted({_parse_input(r, 0)[0] for _, ins, _ in tree for r in ins}
                  - names - {"age"})


def build_tree(frame, tree=TREE, n_terms=3, learn_mf=True):
    """Resolve inputs against the frame, seed each input's centres at the data
    quantiles, and hand every node its slice of the genome.

    GENOME LAYOUT, per node in order:  [ gap genes of its inputs | its singletons ]
    An input's slice is inp["mf"]; a node's singletons are node["rules"].
    learn_mf=False gives every input zero gap genes -> the old fixed-MF model."""
    out_dom, meta, k = {}, [], 0
    for name, inputs, target in tree:
        tgt = target if (target in frame.columns) else None
        ins = []
        for raw in inputs:
            src, n = _parse_input(raw, n_terms)
            if src in out_dom:                                # a child node's output
                th = out_dom[src]                             # theta data, or None
                c0 = _centers(th, n) if th is not None else np.linspace(0, 1, n)
                dlo, dhi = _domain(*((th.min(), th.max()) if th is not None
                                     else (0.0, 1.0)))
                i = {"kind": "node", "src": src, "mu": 0.0, "sd": 1.0}
            elif src in frame.columns:                        # a sensor / age
                x = frame[src].to_numpy(float)
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                z = (x - mu) / sd
                c0, (dlo, dhi) = _centers(z, n), _domain(z.min(), z.max())
                i = {"kind": "col", "src": src, "mu": mu, "sd": sd}
            else:
                continue
            i.update(k=n, c0=c0, centers=c0, dlo=dlo, dhi=dhi,
                     n_mf=(n + 1) if (learn_mf and n >= 2) else 0)
            i["mf"] = slice(k, k + i["n_mf"])                  # gap genes ...
            k += i["n_mf"]
            ins.append(i)
        if not ins:
            continue
        n_rules = int(np.prod([i["k"] for i in ins]))
        meta.append({"name": name, "inputs": ins, "n_rules": n_rules, "target": tgt,
                     "root": name == "RUL", "rules": slice(k, k + n_rules)})
        k += n_rules                                          # ... then singletons
        out_dom[name] = frame[tgt].to_numpy(float) if tgt is not None else None
    return meta


def n_params(meta):
    return int(max(m["rules"].stop for m in meta)) if meta else 0


def n_mf_params(meta):
    return int(sum(i["n_mf"] for m in meta for i in m["inputs"]))


def centers(node, genome):
    """This node's Ruspini centres, one array per input: decoded from the genome
    if the antecedents are learned, else the fixed quantile ones."""
    return [_decode(i, genome[i["mf"]]) if i["n_mf"] else i["c0"]
            for i in node["inputs"]]


def seed_genome(meta, lo, hi):
    """GA seed = the previous model: quantile centres, mid-box singletons."""
    x0 = 0.5 * (lo + hi)
    for m in meta:
        for i in m["inputs"]:
            if i["n_mf"]:
                x0[i["mf"]] = _encode(i, i["c0"])
    return x0


def bake_centers(meta, genome):
    """Store the learned centres in meta so inspect() and the plots see them."""
    for m in meta:
        for i, c in zip(m["inputs"], centers(m, genome)):
            i["centers"] = c
    return meta


def genome_bounds(meta, frame, rul_hi, cons_pad=0.25):
    """Box for the whole genome. Gap genes: [GAP_MIN, 1]. Singletons: the theta
    range (supervised leaf) / [0,1] (spool) / [0, rul_hi] (root)."""
    lo, hi = np.zeros(n_params(meta)), np.zeros(n_params(meta))
    for m in meta:
        for i in m["inputs"]:
            lo[i["mf"]], hi[i["mf"]] = GAP_MIN, 1.0
        if m["root"]:
            a, b = 0.0, rul_hi
        elif m["target"] is not None:
            y = frame[m["target"]].to_numpy(float)
            span = float(y.max() - y.min())
            if span <= 1e-9:                 # (near-)constant target: pin tightly
                v = float(y.mean())          # -> the leaf predicts ~this constant
                eps = max(1e-6, abs(v) * 1e-3)
                a, b = v - eps, v + eps
            else:
                a, b = y.min() - cons_pad * span, y.max() + cons_pad * span
        else:
            a, b = 0.0, 1.0
        lo[m["rules"]], hi[m["rules"]] = a, b
    return lo, hi


def predict_tree(frame, meta, genome):
    """Evaluate the whole tree; returns {node_name: output_array}. BOTH the
    centres and the singletons are read out of the genome."""
    vals = {}
    for m in meta:
        cols = [vals[i["src"]] if i["kind"] == "node"
                else (frame[i["src"]].to_numpy(float) - i["mu"]) / i["sd"]
                for i in m["inputs"]]
        y = _fis(cols, centers(m, genome), genome[m["rules"]])
        # latent spool nodes are clamped to [0,1]; theta leaves / root unclamped
        vals[m["name"]] = y if (m["root"] or m["target"]) else np.clip(y, 0, 1)
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


def genetic_optimize(loss, lo, hi, pop=80, gens=120, seed=0, x0=None,
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
            mask = rng.random(n) < 0.2
            child[mask] += rng.normal(0, 1, mask.sum()) * sigma * span[mask]
            newP.append(np.clip(child, lo, hi))
        P = np.array(newP)
        fit = np.array([loss(g) for g in P])
        if fit.min() < best_fit:
            best_fit, best = fit.min(), P[fit.argmin()].copy()
        history.append(best_fit)
        sigma = max(0.12, sigma * 0.97)
        if verbose and (t % 10 == 0 or t == gens - 1):
            print(f"  [{label}] gen {t + 1:3d}/{gens}  best={best_fit:.4f}  sigma={sigma:.3f}")
    return best, history


# ==========================================================================
# 5. Condition residuals  (strip the dominant flight-condition effect)
# ==========================================================================

def _design(frame, conds):
    W = frame[conds].to_numpy(float)
    return np.column_stack([np.ones(len(W)), W, W ** 2])


def residualize(frame, sensors, conds, ref_cycles=20):
    """Subtract each engine's OWN new-condition signature.

    For every unit, fit  x_s ~ [1, w, w^2]  on its FIRST `ref_cycles` cycles, then
    subtract that model from all of its cycles. So "T48 residual" means "hotter
    than THIS engine ran when it was new" -- a damage signal.

    A single fleet-wide baseline (the previous version) instead means "hotter than
    the AVERAGE engine ran when new", which is damage + manufacturing scatter +
    flight-class bias. That extra noise lands straight in the leaves, and pooling
    ten datasets multiplies it. Hence per-unit.

    No leakage: it uses only early-life SENSOR data from the engine itself, which
    you have at deployment. It touches neither RUL nor the hs flag, so it works
    identically on a held-out unit -- there is nothing to store and re-apply."""
    conds = [c for c in conds if c in frame.columns]
    cols = [s for s in sensors if s in frame.columns]
    if len(conds) < 2 or not cols:
        return frame
    f = frame.reset_index(drop=True)
    X, Y = _design(f, conds), f[cols].to_numpy(float)
    u, cyc, R = f["unit"].to_numpy(), f["cycle"].to_numpy(), Y.copy()
    for k in np.unique(u):
        m = np.flatnonzero(u == k)
        m = m[np.argsort(cyc[m])]
        ref = m[:max(ref_cycles, X.shape[1] + 2)]        # too short -> use all
        B = np.linalg.lstsq(X[ref], Y[ref], rcond=None)[0]
        R[m] = Y[m] - X[m] @ B
    f[cols] = R
    return f


def smooth_inputs(frame, cols, span):
    """EWMA-smooth `cols` along cycles WITHIN each unit (span<=1 = no-op).

    The FIS is memoryless: it maps one cycle's cruise-mean sensors to theta_hat
    with no temporal context, so per-cycle sensor noise passes straight into the
    prediction. Smoothing the inputs here (identically at fit and predict time)
    is the direct cure for the cycle-to-cycle jitter in theta_hat / RUL_hat."""
    if not span or span <= 1:
        return frame
    order = frame.index
    f = frame.sort_values(["unit", "cycle"]).copy()
    for c in cols:
        if c in f.columns:
            f[c] = f.groupby("unit")[c].transform(
                lambda s: s.ewm(span=span, min_periods=1).mean())
    return f.loc[order]


def _prep(frame, tree, residual, ref_cycles, smooth_span):
    """The preprocessing both fit and predict must apply, identically."""
    sensors = _leaf_sensors(tree)
    if residual:
        frame = residualize(frame, sensors, CONDITIONS, ref_cycles)
    return smooth_inputs(frame, sensors, smooth_span)


# ==========================================================================
# 6. Fit / predict
# ==========================================================================

def fit_gft(frame, tree=TREE, n_terms=3, pop=80, gens=120, residual=True,
            ref_cycles=20, smooth_span=0, rul_cap=None, theta_weight=1.0,
            learn_mf=True, seed=0, verbose=True):
    """Train the tree end-to-end on a hybrid loss:
        loss = NASA_score(RUL)  +  theta_weight * mean_leaf( RMSE(theta)/range ).

    learn_mf   : also evolve the membership functions (section 1b). The GA is
                 seeded at the quantile placement, so this contains the fixed-MF
                 model -- it cannot start worse. False = antecedent ablation.
    residual   : subtract each engine's OWN new-condition signature, fitted on its
                 first `ref_cycles` cycles (section 5). Nothing is stored: the same
                 transform is recomputed at predict time on the held-out unit.
    rul_cap    : y = min(RUL, cap). RECOMMENDED. Before onset the engine is
                 healthy, theta is flat and RUL is set by the unit's LIFETIME,
                 not by any sensor; an uncapped target makes the loss reward an
                 age-regressor. See suggest_rul_cap().
    smooth_span: EWMA span on the sensors (~5-10 tames the theta jitter).

    Fits on `frame`. Score with evaluate() on a HELD-OUT frame -- in-sample
    numbers here are not evidence. Returns a plain-dict model."""
    frame = _prep(frame.reset_index(drop=True), tree, residual, ref_cycles,
                  smooth_span)
    meta = build_tree(frame, tree, n_terms, learn_mf)
    y = cap(frame["RUL"], rul_cap)
    lo, hi = genome_bounds(meta, frame, float(y.max()))

    leaves = [m for m in meta if m["target"]]
    Yt = {m["name"]: frame[m["target"]].to_numpy(float) for m in leaves}
    span = {n: (v.max() - v.min()) or 1.0 for n, v in Yt.items()}

    def loss(g):
        vals = predict_tree(frame, meta, g)
        l = nasa_score(y, vals["RUL"])
        if leaves:
            l += theta_weight * np.mean([rmse(Yt[m["name"]], vals[m["name"]])
                                         / span[m["name"]] for m in leaves])
        return l

    if verbose:
        print(f"tree: {' '.join(m['name'] for m in meta)} | params: {n_params(meta)}"
              f" ({n_mf_params(meta)} antecedent + "
              f"{n_params(meta) - n_mf_params(meta)} rule)"
              f" | leaves: {[m['target'] for m in leaves]}")
    genome, history = genetic_optimize(loss, lo, hi, pop=pop, gens=gens, seed=seed,
                                       x0=seed_genome(meta, lo, hi),
                                       verbose=verbose, label="GFT")
    bake_centers(meta, genome)
    return {"tree": tree, "meta": meta, "genome": genome, "history": history,
            "rul_cap": rul_cap, "theta_weight": theta_weight, "learn_mf": learn_mf,
            "residual": residual, "ref_cycles": ref_cycles,
            "smooth_span": smooth_span}


def predict_gft(frame, model):
    """Full inference. Returns unit, cycle, each theta_hat, each latent node
    health (*_h), and RUL_hat."""
    frame = _prep(frame.reset_index(drop=True), model["tree"], model["residual"],
                  model["ref_cycles"], model["smooth_span"])
    vals = predict_tree(frame, model["meta"], model["genome"])
    out = frame[["unit", "cycle"]].copy()
    for m in model["meta"]:
        if m["root"]:
            continue
        if m["target"]:
            out[m["target"] + "_hat"] = vals[m["name"]]      # comparable to theta
        else:
            out[m["name"] + "_h"] = vals[m["name"]]          # latent [0,1]
    out["RUL_hat"] = np.clip(vals["RUL"], 0.0, model["rul_cap"])
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def disp(inp, c):
    """Centres back in display units (raw sensor units for `col` inputs)."""
    return inp["mu"] + np.asarray(c, float) * inp["sd"]


def inspect(model):
    """Per FIS: inputs, target, grid, singleton range, and -- when the
    antecedents were learned -- the tuned centres next to the quantile seed."""
    meta, g = model["meta"], model["genome"]
    print(f"genome: {n_mf_params(meta)} antecedent + "
          f"{n_params(meta) - n_mf_params(meta)} rule = {n_params(meta)} params")
    for m in meta:
        s = g[m["rules"]]
        tgt = m["target"] or ("RUL" if m["root"] else "-")
        print(f"{m['name']:9s} <- {[i['src'] for i in m['inputs']]}  ->{tgt:14s} "
              f"grid{tuple(i['k'] for i in m['inputs'])}  "
              f"out=[{s.min():.3g}, {s.max():.3g}]")
        for i, c in zip(m["inputs"], centers(m, g)):
            if i["n_mf"]:
                print(f"    {i['src']:>10s}  centres {np.round(disp(i, c), 4)}"
                      f"   (was {np.round(disp(i, i['c0']), 4)})")


# ==========================================================================
# 7. Held-out evaluation + the baselines the tree has to beat
# ==========================================================================
# The tree has ~200 free genes against ~300 cycle rows. In-sample numbers mean
# nothing: split by UNIT (loo_cv), and always print `RUL = c - age` next to it.

def cap(y, rul_cap):
    """Piecewise-linear (capped) RUL convention: min(RUL, cap)."""
    y = np.asarray(y, float)
    return np.minimum(y, float(rul_cap)) if rul_cap else y


def suggest_rul_cap(frame, q=0.5):
    """RUL at degradation onset (q-quantile over units) -- the natural place to
    flatten the target, since nothing before it is visible in the sensors."""
    post = frame[frame.get("since_onset", pd.Series(0, index=frame.index)) > 0]
    if post.empty:
        return None
    return float(np.quantile(post.groupby("unit")["RUL"].max().to_numpy(float), q))


def split_units(frame, holdout):
    """Split by UNIT -- never by row; rows within a unit are not independent."""
    hold = set(np.atleast_1d(holdout).ravel().tolist())
    return (frame[~frame["unit"].isin(hold)].reset_index(drop=True),
            frame[frame["unit"].isin(hold)].reset_index(drop=True))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss if ss > 1e-12 else 0.0


def unit_rho(frame, y, yhat):
    """Mean per-unit Pearson correlation. For a memoryless FIS the plain RMSE is
    dominated by the per-cycle noise floor; rho asks the question that matters --
    does the estimate MOVE WITH the truth."""
    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = float(np.sqrt((a @ a) * (b @ b)))
        return float(a @ b) / d if d > 1e-12 else 0.0
    u, y, yhat = frame["unit"].to_numpy(), np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean([rho(y[u == k], yhat[u == k]) for k in np.unique(u)]))


def _score(frame, yhat, rul_cap):
    y = cap(frame["RUL"], rul_cap)
    return {"RMSE": rmse(y, yhat), "MAE": float(np.mean(np.abs(y - yhat))),
            "NASA": nasa_score(y, yhat), "R2": r2(y, yhat)}


def baselines(train, test, rul_cap=None):
    """The sensor-free references. `RUL = c - age` uses ZERO sensors and ONE
    parameter; if the fuzzy tree cannot beat it, the tree is decoration."""
    y, a = cap(train["RUL"], rul_cap), train["age"].to_numpy(float)
    ok = (y < float(rul_cap) - 1e-9) if rul_cap else np.ones(len(y), bool)
    c = float(np.mean(y[ok] + a[ok])) if ok.sum() > 1 else float(np.mean(y + a))
    mu = float(np.mean(y))
    age_hat = np.clip(c - test["age"].to_numpy(float), 0.0, rul_cap or None)
    return pd.DataFrame([
        dict(model="mean RUL", n_params=1, **_score(test, np.full(len(test), mu), rul_cap)),
        dict(model="age only (RUL = c - age)", n_params=1, **_score(test, age_hat, rul_cap)),
    ])


def evaluate(frame, model, name="GFT"):
    """Score a fitted model on a HELD-OUT frame. Reports theta as R2 and per-unit
    correlation -- an RMSE of 8e-4 on a variable whose range is 1.3e-2 sounds tiny
    and means almost nothing."""
    pr = predict_gft(frame, model)
    f = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    yh = pr["RUL_hat"].to_numpy(float)
    out = dict(model=name, n_params=n_params(model["meta"]),
               **_score(f, yh, model["rul_cap"]))
    out["rho_RUL"] = unit_rho(f, cap(f["RUL"], model["rul_cap"]), yh)
    for m in model["meta"]:
        if m["target"]:
            y, t = f[m["target"]].to_numpy(float), pr[m["target"] + "_hat"].to_numpy(float)
            out[f"{m['target']}:R2"] = r2(y, t)
            out[f"{m['target']}:rho"] = unit_rho(f, y, t)
    return out


def loo_cv(frame, tree=TREE, name="GFT", seeds=(0,), **fit_kw):
    """Leave-ONE-UNIT-out CV. With 6 units this is the only honest protocol."""
    rows = []
    for u in np.sort(frame["unit"].unique()):
        tr, te = split_units(frame, u)
        for s in seeds:
            model = fit_gft(tr, tree=tree, seed=s, verbose=False, **fit_kw)
            r = evaluate(te, model, name)
            r.update(fold=int(u), seed=int(s))
            rows.append(r)
            print(f"  [{name}] hold-out unit {int(u)} seed {s}: "
                  f"RMSE={r['RMSE']:.2f}  NASA={r['NASA']:.3f}  R2={r['R2']:.3f}")
    return pd.DataFrame(rows)


def summarize(df, keys=("RMSE", "MAE", "NASA", "R2")):
    """mean +/- sd across folds/seeds -- one GA run on a 200-D non-convex
    landscape is an anecdote, not a result."""
    g = df.groupby("model")[list(keys)]
    return g.mean().round(3).join(g.std().round(3), rsuffix="_sd")


# ==========================================================================
# 8. Self-test on a signal-bearing synthetic frame
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
                "T48": sens(9.0), "P40": sens(2.0), "Nc": sens(1.5),      # HPT
                "Ps30": sens(1.3),
                "T50": sens(7.0), "P50": sens(1.5), "Nf": sens(1.0),      # LPT
                "P24": sens(1.2),
                "HPT_eff_mod": -d,        "HPT_flow_mod": -0.6 * d,
                "LPT_eff_mod": -0.8 * d,  "LPT_flow_mod": -0.5 * d,
                "since_onset": float(max(0, k - onset)),
                "RUL": float(eol - k)})
    return pd.DataFrame(rows)


def _self_test():
    df = _synthetic_frame(units=6, cycles=60, seed=1)
    rc = suggest_rul_cap(df)

    # the learned partition must stay a valid Ruspini partition for ANY genome
    meta = build_tree(df, TREE, 3, learn_mf=True)
    lo, hi = genome_bounds(meta, df, rc)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        g = rng.uniform(lo, hi)
        for node in meta:
            for c in centers(node, g):
                assert np.all(np.diff(c) > 0), "centres out of order!"
                M = _memberships(np.linspace(c[0] - 5, c[-1] + 5, 400), c)
                worst = max(worst, float(np.abs(M.sum(1) - 1).max()))
    print(f"100 random genomes: centres always ordered, max|sum(mu)-1| = {worst:.1e}")

    # the GA seed must decode back to the OLD quantile centres, exactly
    x0, fixed = seed_genome(meta, lo, hi), build_tree(df, TREE, 3, learn_mf=False)
    err = max(float(np.abs(c - i["c0"]).max())
              for a, b in zip(meta, fixed)
              for c, i in zip(centers(a, x0), b["inputs"]))
    print(f"GA seed reproduces the quantile centres: max err = {err:.1e}")

    print(f"\nrul_cap = {rc:.0f}   (RUL at degradation onset)\n")
    tr, te = split_units(df, holdout=6)
    print(baselines(tr, te, rc).round(3).to_string(index=False), "\n")

    rows = []
    for lm in (False, True):
        m = fit_gft(tr, gens=80, pop=60, rul_cap=rc, smooth_span=7,
                    theta_weight=5.0, learn_mf=lm, verbose=False)
        rows.append(evaluate(te, m, f"GFT learn_mf={lm}"))
    print(pd.DataFrame(rows)[["model", "n_params", "RMSE", "NASA", "R2",
                              "HPT_eff_mod:R2"]].round(3).to_string(index=False))
    print()
    inspect(m)


if __name__ == "__main__":
    _self_test()