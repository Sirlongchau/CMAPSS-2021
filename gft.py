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

# MONOTONE NODES (see section 2b). A node listed here has its output constrained
# to be monotone in each input BY CONSTRUCTION -- its singleton grid is built from
# non-negative steps, so no genome the GA can produce will violate it.
#
# The value is either a single sign broadcast over ALL of the node's inputs, or a
# per-input tuple. -1 = non-increasing, +1 = non-decreasing. A scalar is preferred
# when every input shares a sign, because it stays correct if the node's arity
# changes (e.g. TREE vs TREE_AGE give the root 2 vs 3 inputs).
#
#   hp / lp : inputs are damage MODIFIERS (more negative = more degraded), output
#             is shaft DAMAGE, which can only RISE as a modifier falls
#             -> non-increasing in each input.
#   RUL     : inputs are shaft DAMAGE in [0,1] (and, in TREE_AGE, `age`). More
#             damage means less remaining life; more accumulated cycles likewise
#             -> non-increasing in every input.
#
# The signs are physics, not preference: damage cannot fall with more damage. The
# LEAVES are deliberately absent -- their monotonicity depends on how cleanly the
# per-unit residual isolated the health effect, which is not yet guaranteed.
MONOTONE = {
    "hp":  -1,
    "lp":  -1,
    "RUL": -1,
}


def _mono_signs(name, n_inputs, monotone=True):
    """Resolve MONOTONE[name] to a per-input sign tuple, or None if unconstrained.
    A scalar broadcasts over every input; a tuple must match the node's arity."""
    if not monotone or name not in MONOTONE:
        return None
    s = MONOTONE[name]
    if np.isscalar(s):
        return tuple(int(np.sign(s)) for _ in range(n_inputs))
    if len(s) != n_inputs:
        raise ValueError(
            f"MONOTONE[{name!r}] gives {len(s)} signs but the node has {n_inputs} "
            f"inputs. Use a scalar (e.g. -1) to broadcast over all inputs, which "
            f"stays correct when the node's arity changes (TREE vs TREE_AGE).")
    return tuple(int(np.sign(v)) for v in s)

_LEAVES = [
    # Leaf inputs and membership are the OUTPUT of leaf_search.py, not a guess.
    # Each was fitted standalone and scored on HELD-OUT units, over 5 seeds, with a
    # cross-spool specificity check (does it read ITS component, or global damage?).
    #
    #   leaf       rho    R2    specificity   verdict
    #   hpt_eff   0.61   0.65     +0.07       strongest & most stable leaf -> keep
    #   lpt_flow  0.59   0.44     +0.38*      best rho AND R2 of any candidate
    #   lpt_eff   0.55   0.45     -0.07       predictive but NOT spool-specific
    #   hpt_flow  0.25   0.13     -0.34       unreadable: DROPPED (see below)
    #   (*) the flow-pair specificity is inflated -- its foil, HPT_flow, is itself
    #       unpredictable, so foil_rho is low for any LPT_flow combo. Judge lpt_flow
    #       on rho/R2, not on that number.
    ("hpt_eff",  ["T48", "P40", "Nc"],   "HPT_eff_mod"),   # HP: eff  -> speed
    ("lpt_flow", ["T50", "P50", "P24"],  "LPT_flow_mod"),  # LP: flow -> pressure
    ("lpt_eff",  ["T50", "P50", "Nf"],   "LPT_eff_mod"),   # LP: eff  -> speed

    # HPT_flow_mod is deliberately ABSENT. An exhaustive search over all 286
    # 3-sensor combinations of the 13 measured sensors put its ceiling at rho=0.27
    # (every other modifier reaches ~0.63), with negative held-out R2 throughout and
    # a specificity of -0.34: candidate inputs predicted the OTHER spool better than
    # HPT flow itself. It is not observable from cruise-mean sensors by a memoryless
    # FIS. Keeping it would have fed the tree a leaf that diagnoses nothing.

    ("hp", ["hpt_eff"],             None),   # HP-shaft damage (single input)
    ("lp", ["lpt_flow", "lpt_eff"], None),   # LP-shaft damage
]

# DEFAULT: the root sees ONLY the two shaft-damage nodes. `age` is ABLATED -- with
# a few units of similar lifetime it is a shortcut, RUL ~= mean_EOL - age is
# learnable without touching a single sensor, and the GA will take that deal.
TREE     = _LEAVES + [("RUL", ["hp", "lp"], None)]
TREE_AGE = _LEAVES + [("RUL", ["hp", "lp", "age"], None)]   # ablation control

# LPT_eff is predictive (R2 0.45) but reads global damage as much as the LP spool
# (specificity -0.07). TREE_LP1 drops it, leaving a fully spool-SPECIFIC tree of two
# leaves. Run both on validation: if dropping it barely costs RUL accuracy, take
# TREE_LP1 -- it is the cleaner claim.
TREE_LP1 = [l for l in _LEAVES if l[0] != "lpt_eff"]
TREE_LP1 = [(n, ["lpt_flow"], t) if n == "lp" else (n, i, t)
            for n, i, t in TREE_LP1] + [("RUL", ["hp", "lp"], None)]

CONDITIONS = ["alt", "Mach", "TRA", "T2"]


# ==========================================================================
# 2b. Structural monotone consequents
# ==========================================================================
# A Sugeno rule grid is a generic approximator: nothing stops the singletons from
# folding into a non-monotone surface, and on noisy data the GA WILL buy a fold if
# it shaves training loss. For the damage/RUL nodes that is unphysical -- the true
# surface is monotone. So we bake monotonicity into the parametrization.
#
# A grid is stored as [ base | non-negative steps ] (prod(shape) genes, same count
# as free singletons). The steps are cumulatively summed along every axis, so the
# result is monotone from a corner; per-axis signs choose the direction. This is
# the same idea as the MF gap-encoding: the non-monotone region is removed from the
# search space entirely -- no penalty, no weight to tune, no possible violation,
# and the GA wastes no budget rediscovering that folds are bad.

def _mono_decode(base, steps, shape, signs, out=None):
    """Monotone grid (flattened) from a base value + non-negative steps.
    Multidim cumsum of non-negative increments is non-decreasing from a corner;
    decreasing axes are flipped so their running total grows the other way.

    `base` is the grid MINIMUM (the corner where the cumsum is 0), so the grid
    spans [base, base + total_steps] -- which a box constraint alone cannot hold
    inside the node's output range. `out=(lo, hi)` clips it there. Clipping is a
    monotone map, so the surface stays monotone; without it a spool grid can run
    past 1 and saturate against the [0,1] clamp in predict_tree, wasting the
    node's whole dynamic range."""
    s = np.maximum(np.asarray(steps, float), 0.0).copy()
    s[0] = 0.0                                      # corner carries the base only
    G = s.reshape(shape)
    rev = tuple(slice(None, None, -1) if sg < 0 else slice(None) for sg in signs)
    G = G[rev]
    for ax in range(G.ndim):
        G = np.cumsum(G, axis=ax)
    G = G[rev]
    G = float(base) + G
    if out is not None:
        G = np.clip(G, out[0], out[1])
    return G.ravel()


def _mono_encode(grid, shape, signs):
    """Inverse: recover (base, steps) from a monotone grid, to seed the GA at a
    valid monotone start. `steps` come out non-negative iff `grid` is monotone."""
    G = np.asarray(grid, float).reshape(shape)
    corner = tuple(-1 if sg < 0 else 0 for sg in signs)
    base = float(G[corner])
    rev = tuple(slice(None, None, -1) if sg < 0 else slice(None) for sg in signs)
    A = (G - base)[rev]
    for ax in range(A.ndim):
        A = np.diff(A, axis=ax, prepend=0)
    out = A[rev].ravel().copy()
    out[0] = base
    return out


def _parse_input(inp, default):
    """('name', k) -> (name, k);  'name' -> (name, default)."""
    if isinstance(inp, (tuple, list)):
        return inp[0], int(inp[1])
    return inp, default


def _leaf_sensors(tree):
    names = {n for n, _, _ in tree}
    return sorted({_parse_input(r, 0)[0] for _, ins, _ in tree for r in ins}
                  - names - {"age"})


def build_tree(frame, tree=TREE, n_terms=3, learn_mf=True, monotone=True):
    """Resolve inputs against the frame, seed each input's centres at the data
    quantiles, and hand every node its slice of the genome.

    GENOME LAYOUT, per node in order:  [ gap genes of its inputs | its singletons ]
    An input's slice is inp["mf"]; a node's singletons are node["rules"].
    learn_mf=False gives every input zero gap genes -> the old fixed-MF model.

    monotone=True encodes the nodes named in MONOTONE as structural monotone grids
    (section 2b): their rule slice holds [base | non-negative steps] rather than raw
    singletons, so the surface cannot fold. monotone=False = free grids (ablation)."""
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
        shape = tuple(i["k"] for i in ins)
        n_rules = int(np.prod(shape))
        signs = _mono_signs(name, len(ins), monotone)
        meta.append({"name": name, "inputs": ins, "n_rules": n_rules, "target": tgt,
                     "root": name == "RUL", "rules": slice(k, k + n_rules),
                     "shape": shape, "mono": signs})
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


def node_singletons(node, genome):
    """This node's rule consequents as a flat singleton vector. For a monotone
    node the rule slice stores [base | steps]; decode it (and clip to the node's
    output range, which genome_bounds recorded as out_lo/out_hi)."""
    g = genome[node["rules"]]
    if node.get("mono") is not None:
        out = ((node["out_lo"], node["out_hi"])
               if "out_lo" in node else None)
        return _mono_decode(g[0], g, node["shape"], node["mono"], out)
    return g


def seed_genome(meta, lo, hi):
    """GA seed = the previous model: quantile centres, mid-box singletons. For a
    monotone node, genome index rules.start doubles as the grid `base` (its bounds
    were set to the output range, not [0, span]), and _mono_decode ignores it as a
    step -- so the plain mid-box is already a valid, if steep, monotone seed."""
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
    """Box for the whole genome. Gap genes: [GAP_MIN, 1]. Free-node singletons:
    the theta range (supervised leaf) / [0,1] (spool) / [0, rul_hi] (root).

    A MONOTONE node's slice is [base | steps]: `base` gets the node's output range,
    and each `step` is bounded [0, span] -- non-negative (enforcing monotonicity)
    and no larger than the whole range (a single step cannot exceed the output
    span). The corner step (index 0) is unused by _mono_decode; it is pinned to 0."""
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
        m["out_lo"], m["out_hi"] = a, b           # used to clip monotone grids
        if m.get("mono") is not None:             # [base | non-negative steps]
            # base is the grid MINIMUM and the grid spans [base, base+total], so a
            # box alone cannot keep it inside [a, b] -- node_singletons clips. The
            # per-step cap still matters: it stops one gene swallowing the whole
            # range and flattening the surface into a single step.
            r = m["rules"]
            path = sum(k - 1 for k in m["shape"]) or 1
            lo[r], hi[r] = 0.0, (b - a) / path
            lo[r.start], hi[r.start] = a, b           # base spans the output range
        else:
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
        y = _fis(cols, centers(m, genome), node_singletons(m, genome))
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

def _unit_corr(units, y, yhat):
    """Mean per-unit Pearson correlation (nan-safe). A unit whose theta or
    prediction is flat contributes 0 (no correlation to speak of), not nan."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    rs = []
    for k in np.unique(units):
        m = units == k
        a, b = y[m] - y[m].mean(), yhat[m] - yhat[m].mean()
        d = float(np.sqrt((a @ a) * (b @ b)))
        rs.append(float(a @ b) / d if d > 1e-12 else 0.0)
    return float(np.mean(rs)) if rs else 0.0


def fit_gft(frame, tree=TREE, n_terms=3, pop=80, gens=120, residual=True,
            ref_cycles=20, smooth_span=0, rul_cap=None, theta_weight=1.0,
            trend_weight=0.0, learn_mf=True, monotone=True, seed=0, verbose=True):
    """Train the tree end-to-end on a hybrid loss.

    LOSS (every term is in 'fraction of a trivial baseline's error', so the weights
    are interpretable and a leaf's variance no longer sets its influence):

        loss =        NASA(RUL)      / NASA(age-only baseline)
             + w_th * mean_leaf[ RMSE(theta)     / std(theta_leaf) ]
             + w_tr * mean_leaf[ 1 - mean_unit_corr(theta_hat, theta) ]

    * The RUL term is divided by the NASA score of `RUL = c - age`, so a value of 1
      means 'as good as the zero-sensor baseline'; the tree must drive it below 1.
    * Each leaf's RMSE is divided by std(theta_leaf) = RMSE(predict-the-mean), its
      OWN trivial error. A near-constant leaf (small std -- the HPT_flow problem in
      the pool) no longer explodes and drowns the informative leaves.
    * trend_weight adds a per-unit CORRELATION term. Runs show the leaves capture
      the SHAPE of degradation (per-unit rho ~ 0.5-0.6) while missing the SCALE
      (negative R2). RMSE punishes scale error and gives shape no credit; this term
      rewards shape directly -- what a memoryless FIS on noisy sensors can actually
      deliver. Try 0.5-1.0. 0 = off.

    theta_weight : weight on the (baseline-normalized) leaf RMSE term.
    trend_weight : weight on the per-unit correlation term (0 = off).
    learn_mf     : also evolve the membership functions (section 1b). Seeded at the
                   quantile placement, so it cannot start worse. False = ablation.
    monotone     : constrain MONOTONE nodes (hp, lp, RUL) to monotone surfaces by
                   construction (section 2b). False = free grids, for the ablation.
    residual     : subtract each engine's OWN new-condition signature (section 5).
    rul_cap      : y = min(RUL, cap). RECOMMENDED -- see suggest_rul_cap().
    smooth_span  : EWMA span on the sensors (~5-10 tames the theta jitter).

    Fits on `frame`. Score with evaluate() on a HELD-OUT frame -- in-sample
    numbers here are not evidence. Returns a plain-dict model."""
    frame = _prep(frame.reset_index(drop=True), tree, residual, ref_cycles,
                  smooth_span)
    meta = build_tree(frame, tree, n_terms, learn_mf, monotone)
    y = cap(frame["RUL"], rul_cap)
    yv = np.asarray(y, float)
    lo, hi = genome_bounds(meta, frame, float(yv.max()))
    units = frame["unit"].to_numpy()

    # --- per-term normalizers, computed ONCE from trivial baselines ----------
    a = frame["age"].to_numpy(float)                     # RUL: age-only NASA score
    ok = (yv < float(rul_cap) - 1e-9) if rul_cap else np.ones(len(yv), bool)
    c_age = float(np.mean(yv[ok] + a[ok])) if ok.sum() > 1 else float(np.mean(yv + a))
    nasa0 = nasa_score(y, np.clip(c_age - a, 0.0, rul_cap or None)) or 1.0

    leaves = [m for m in meta if m["target"]]
    Yt = {m["name"]: frame[m["target"]].to_numpy(float) for m in leaves}
    theta0 = {n: (float(np.std(v)) or 1.0) for n, v in Yt.items()}   # std = trivial RMSE

    def loss(g):
        vals = predict_tree(frame, meta, g)
        l = nasa_score(y, vals["RUL"]) / nasa0
        if leaves and theta_weight:
            l += theta_weight * np.mean(
                [rmse(Yt[m["name"]], vals[m["name"]]) / theta0[m["name"]]
                 for m in leaves])
        if leaves and trend_weight:
            l += trend_weight * np.mean(
                [1.0 - _unit_corr(units, Yt[m["name"]], vals[m["name"]])
                 for m in leaves])
        return l

    if verbose:
        mono = [m["name"] for m in meta if m.get("mono") is not None]
        print(f"tree: {' '.join(m['name'] for m in meta)} | params: {n_params(meta)}"
              f" ({n_mf_params(meta)} antecedent + "
              f"{n_params(meta) - n_mf_params(meta)} rule)"
              f" | leaves: {[m['target'] for m in leaves]}"
              f" | monotone: {mono} | w_th={theta_weight} w_tr={trend_weight}")
    genome, history = genetic_optimize(loss, lo, hi, pop=pop, gens=gens, seed=seed,
                                       x0=seed_genome(meta, lo, hi),
                                       verbose=verbose, label="GFT")
    bake_centers(meta, genome)
    return {"tree": tree, "meta": meta, "genome": genome, "history": history,
            "rul_cap": rul_cap, "theta_weight": theta_weight,
            "trend_weight": trend_weight, "learn_mf": learn_mf,
            "monotone": monotone, "residual": residual, "ref_cycles": ref_cycles,
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
    antecedents were learned -- the tuned centres next to the quantile seed.
    Monotone nodes are flagged and their singletons are the DECODED grid."""
    meta, g = model["meta"], model["genome"]
    print(f"genome: {n_mf_params(meta)} antecedent + "
          f"{n_params(meta) - n_mf_params(meta)} rule = {n_params(meta)} params")
    for m in meta:
        s = node_singletons(m, g)
        tgt = m["target"] or ("RUL" if m["root"] else "-")
        tag = f"  MONO{m['mono']}" if m.get("mono") is not None else ""
        print(f"{m['name']:9s} <- {[i['src'] for i in m['inputs']]}  ->{tgt:14s} "
              f"grid{tuple(i['k'] for i in m['inputs'])}  "
              f"out=[{s.min():.3g}, {s.max():.3g}]{tag}")
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

    # every monotone node's surface must be monotone for ANY genome
    mm = build_tree(df, TREE, 3, learn_mf=True, monotone=True)
    lo2, hi2 = genome_bounds(mm, df, rc)
    bad = 0
    for _ in range(200):
        g = rng.uniform(lo2, hi2)
        for node in mm:
            if node.get("mono") is None:
                continue
            G = node_singletons(node, g).reshape(node["shape"])
            for ax, sg in enumerate(node["mono"]):
                d = np.diff(G, axis=ax)
                if sg < 0 and np.any(d > 1e-9):
                    bad += 1
                if sg > 0 and np.any(d < -1e-9):
                    bad += 1
    print(f"200 random genomes: monotone nodes never violate their sign "
          f"({'OK' if bad == 0 else str(bad) + ' VIOLATIONS'})")

    print(f"\nrul_cap = {rc:.0f}   (RUL at degradation onset)\n")
    tr, te = split_units(df, holdout=6)
    print(baselines(tr, te, rc).round(3).to_string(index=False), "\n")

    rows = []
    for mono in (False, True):
        m = fit_gft(tr, gens=80, pop=60, rul_cap=rc, smooth_span=7,
                    theta_weight=5.0, learn_mf=True, monotone=mono, verbose=False)
        rows.append(evaluate(te, m, f"GFT monotone={mono}"))
    print(pd.DataFrame(rows)[["model", "n_params", "RMSE", "NASA", "R2",
                              "HPT_eff_mod:R2"]].round(3).to_string(index=False))
    print()
    inspect(m)


if __name__ == "__main__":
    _self_test()
