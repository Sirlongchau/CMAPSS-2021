"""
gft_ncmapss.py
==============

Two-stage GENETIC FUZZY TREE (GFT) for N-CMAPSS, from scratch.
No fuzzy toolbox (no skfuzzy): pure inference + a parametric rule base a GA
can optimise.

    Stage 1  -- one small FIS per (component, modifier)
               [cruise sensors]  ->  theta_hat        (diagnosis)
    Stage 2  -- one FIS
               [damage, slope, load]  ->  RUL          (prognosis)

The two trees are chained: Stage 1's theta_hat builds Stage 2's inputs.

--------------------------------------------------------------------------
FUZZY MODEL (one line to remember)

Each FIS is a ZERO-ORDER TAKAGI-SUGENO system on a FULL GRID of rules with
SINGLETON consequents. Inputs are fuzzified with a RUSPINI PARTITION
(triangular sets that sum to 1 everywhere -- a "partition of unity").

With a Ruspini partition + product t-norm + full grid, the rule firing
strengths of one FIS sum to exactly 1 at every point:

    sum_rules  prod_d mu_{d,i_d}(x_d)  =  prod_d ( sum_i mu_{d,i}(x_d) )  =  1

so the Sugeno weighted average is just a dot product

    y(x) = sum_rules  firing_rule(x) * c_rule            (c = consequents)

The only free parameters are the consequents c (one scalar per grid cell).
They enter LINEARLY, which is exactly what makes them easy for a GA (or,
if you preferred, ordinary least squares). Antecedent centres are fixed
from data quantiles, so the whole model is small and interpretable.

--------------------------------------------------------------------------
WHY THIS SHAPE (ties back to the maths we derived)

    Stage 1  = fuzzy gain-scheduled inverse of the sensor->theta map:
               the grid tiles the operating space, each cell holds a local
               linear-ish map r -> theta. Keeping w-driven sensors as
               antecedents is how the model tracks the w-dependent gain.
    Stage 2  = first-passage estimator RUL = H(state, tangent, load):
               damage (state), its cycle slope (tangent) and a cumulative
               mission-severity proxy (load, from W history) -> RUL.

--------------------------------------------------------------------------
INPUT: any per-cycle DataFrame using the naming convention of
ncmapss_gft_features.py (columns  <role>__<node>__<var>). A self-test at the
bottom builds a signal-bearing synthetic frame so you can watch it learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# ==========================================================================
# 1. RUSPINI FUZZIFICATION  (triangular sets forming a partition of unity)
# ==========================================================================

def ruspini_centers(x: np.ndarray, n_terms: int) -> np.ndarray:
    """
    Ordered term centres for one variable, placed at data quantiles so the
    fuzzy sets are populated. Guards against ties / constant inputs.
    """
    x = np.asarray(x, float)
    c = np.quantile(x, np.linspace(0.0, 1.0, n_terms))
    c = np.unique(c)
    if c.size < n_terms:                       # near-constant input -> spread out
        lo, hi = float(x.min()), float(x.max())
        span = (hi - lo) or 1.0
        c = np.linspace(lo - 0.01 * span, hi + 0.01 * span, n_terms)
    return c


def memberships(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Ruspini memberships of samples `x` for the given ordered `centers`.

    Returns M of shape (n_samples, n_terms) with each ROW summing to 1.
    Implemented with np.interp: each triangular set is the piecewise-linear
    hat 0->1->0 through (c_{i-1}, c_i, c_{i+1}); the two end sets are
    shouldered (flat 1 beyond the extreme centres), which np.interp gives
    for free by clamping outside its knot range.
    """
    x = np.asarray(x, float).ravel()
    c = np.asarray(centers, float)
    m = c.size
    M = np.empty((x.size, m))
    for i in range(m):
        if i == 0:                      # left shoulder: 1 at c0, down to 0 at c1
            xp, fp = [c[0], c[1]], [1.0, 0.0]
        elif i == m - 1:                # right shoulder: 0 at c_{m-2}, 1 at c_{m-1}
            xp, fp = [c[m - 2], c[m - 1]], [0.0, 1.0]
        else:                           # interior triangle peaked at c_i
            xp, fp = [c[i - 1], c[i], c[i + 1]], [0.0, 1.0, 0.0]
        M[:, i] = np.interp(x, xp, fp)
    return M


# ==========================================================================
# 2. FIS  (zero-order Sugeno, full grid, singleton consequents)
# ==========================================================================

@dataclass
class FISNode:
    """
    One FIS. Both the Ruspini centres AND the singleton consequents are now
    GA parameters, packed per node as

        genome_slice = [ centres_input0 , centres_input1 , ... , consequents ]

    The node only stores the fixed metadata: how many terms per input, the
    box the centres may live in (data range), and quantile centres used to
    seed the GA. Decoding sorts the centres so the partition stays valid.
    """
    name: str
    inputs: List[str]              # feature column names (antecedents)
    output: str                    # target column name (for training)
    n_terms: List[int]             # fuzzy sets per input
    c_lo: List[float]              # per-input centre lower bound (data min)
    c_hi: List[float]              # per-input centre upper bound (data max)
    init_centers: List[np.ndarray] # per-input quantile centres (GA seed)

    @property
    def n_rules(self) -> int:
        r = 1
        for m in self.n_terms:
            r *= m
        return r

    @property
    def n_centers(self) -> int:
        return int(sum(self.n_terms))

    @property
    def n_params(self) -> int:
        return self.n_centers + self.n_rules


def _sanitize_centers(c: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sort, clip to the data box, and force a strictly increasing sequence
    so the Ruspini triangles stay well defined even after mutation."""
    c = np.sort(np.clip(c, lo, hi))
    eps = (hi - lo) * 1e-3 or 1e-6
    for i in range(1, len(c)):
        if c[i] <= c[i - 1]:
            c[i] = c[i - 1] + eps
    return c


def decode(node: FISNode, g: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Split a node's genome slice into (centres_per_input, consequents)."""
    centers, k = [], 0
    for d, m in enumerate(node.n_terms):
        centers.append(_sanitize_centers(g[k:k + m], node.c_lo[d], node.c_hi[d]))
        k += m
    return centers, g[k:]


def _firing(X: np.ndarray, centers: List[np.ndarray]) -> np.ndarray:
    """
    Rule firing strengths, shape (n_samples, n_rules).

    Product t-norm over inputs, expanded over the full grid in C-order
    (input 0 most significant). Rows sum to 1 by the Ruspini property.
    """
    F = memberships(X[:, 0], centers[0])                    # (n, m0)
    for d in range(1, len(centers)):
        Md = memberships(X[:, d], centers[d])               # (n, md)
        F = (F[:, :, None] * Md[:, None, :]).reshape(F.shape[0], -1)
    return F


def fis_output(X: np.ndarray, centers: List[np.ndarray],
               consequents: np.ndarray) -> np.ndarray:
    """Sugeno inference: y = (firing . consequents) / sum(firing).
    The division is a no-op under a Ruspini partition but keeps things safe
    if a centre set ever degenerates."""
    F = _firing(X, centers)
    denom = F.sum(axis=1)
    denom[denom == 0.0] = 1.0
    return (F @ consequents) / denom


# ==========================================================================
# 3. GENOME  (a model is a list of nodes + one flat vector of centres+conseq.)
# ==========================================================================

def n_params(nodes: List[FISNode]) -> int:
    return int(sum(n.n_params for n in nodes))


def _slices(nodes: List[FISNode]) -> List[Tuple[int, int]]:
    """Contiguous [start, stop) of each node's parameters inside the genome."""
    out, k = [], 0
    for n in nodes:
        out.append((k, k + n.n_params))
        k += n.n_params
    return out


def genome_box(nodes: List[FISNode], cons_lo: float, cons_hi: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (lo, hi, seed) genome vectors: centre bounds come from each
    input's data range, consequent bounds from the target range. The seed
    puts centres at the quantiles and consequents at the target midpoint, so
    the GA starts from a sensible (non-random) antecedent layout."""
    lo, hi, seed = [], [], []
    mid = 0.5 * (cons_lo + cons_hi)
    for node in nodes:
        for d, m in enumerate(node.n_terms):
            lo += [node.c_lo[d]] * m
            hi += [node.c_hi[d]] * m
            seed += list(node.init_centers[d])
        lo += [cons_lo] * node.n_rules
        hi += [cons_hi] * node.n_rules
        seed += [mid] * node.n_rules
    return np.array(lo), np.array(hi), np.array(seed)


def predict_nodes(frame: pd.DataFrame, nodes: List[FISNode],
                  genome: np.ndarray) -> Dict[str, np.ndarray]:
    """Run every FIS in `nodes`, return {output_name: prediction}."""
    preds = {}
    for node, (a, b) in zip(nodes, _slices(nodes)):
        centers, cons = decode(node, genome[a:b])
        X = frame[node.inputs].to_numpy(float)
        preds[node.output] = fis_output(X, centers, cons)
    return preds


def build_nodes(frame: pd.DataFrame,
                specs: List[Tuple[str, List[str], str]],
                n_terms: int) -> List[FISNode]:
    """
    Turn (name, input_cols, output_col) specs into FISNodes, dropping inputs
    absent from `frame`. Centre bounds = data [min, max] per input; seed
    centres = quantiles (the GA then tunes them).
    """
    nodes = []
    for name, inputs, output in specs:
        cols = [c for c in inputs if c in frame.columns]
        if not cols or output not in frame.columns:
            continue
        c_lo, c_hi, init = [], [], []
        for c in cols:
            x = frame[c].to_numpy(float)
            c_lo.append(float(np.min(x)))
            c_hi.append(float(np.max(x)))
            init.append(ruspini_centers(x, n_terms))
        nodes.append(FISNode(name=name, inputs=cols, output=output,
                             n_terms=[n_terms] * len(cols),
                             c_lo=c_lo, c_hi=c_hi, init_centers=init))
    return nodes


# ==========================================================================
# 4. GENETIC ALGORITHM  (generic, real-coded, minimises a scalar loss)
# ==========================================================================

def genetic_optimize(loss: Callable[[np.ndarray], float],
                     n: int, lo: np.ndarray, hi: np.ndarray,
                     pop: int = 40, gens: int = 60, seed: int = 0,
                     elite: int = 2, p_mut: float = 0.4,
                     sigma0: float = 0.3, sigma_min: float = 0.02,
                     cool: float = 0.93, reheat: float = 4.0,
                     patience: int = 8, tol: float = 1e-4,
                     x0: "np.ndarray | None" = None,
                     verbose: bool = False, log_every: int = 1,
                     label: str = "") -> Tuple[np.ndarray, List[float]]:
    """
    Minimise `loss(genome)` over a box [lo, hi]^n.

    Tournament selection + BLX-alpha crossover + Gaussian mutation, elitism,
    and an ADAPTIVE mutation step `sigma` (in fractions of each gene's span):

        - starts at sigma0 (broad exploration),
        - geometric cooldown by `cool` each generation while the best keeps
          improving (settle into exploitation),  floored at sigma_min,
        - if the best loss is FLAT for `patience` generations (relative
          improvement < tol), RE-HEAT: sigma <- min(sigma0, sigma*reheat),
          to kick the search out of a plateau.

    "improvement" is measured on the best-so-far, so the schedule reacts to
    real progress rather than to the generation counter. Returns
    (best_genome, history); history[t] = best loss after generation t.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = hi - lo

    P = rng.uniform(lo, hi, size=(pop, n))
    if x0 is not None:                       # seed one individual at a good guess
        P[0] = np.clip(np.asarray(x0, float), lo, hi)
    fit = np.array([loss(g) for g in P])
    history = []

    sigma = sigma0                      # current mutation strength
    best_so_far = float(fit.min())      # for stagnation detection
    since_improve = 0                   # generations since a real improvement
    best_genome = P[np.argmin(fit)].copy()

    for t in range(gens):
        # --- elitism: carry the best few unchanged --------------------------
        order = np.argsort(fit)
        newP = [P[order[i]].copy() for i in range(elite)]

        # --- fill the rest with children ------------------------------------
        while len(newP) < pop:
            # tournament selection (k=3) for two parents
            a = order[np.argmin(fit[rng.choice(pop, 3, replace=False)])]
            b = order[np.argmin(fit[rng.choice(pop, 3, replace=False)])]
            pa, pb = P[a], P[b]
            # BLX-alpha crossover
            alpha = 0.3
            lo_c = np.minimum(pa, pb) - alpha * np.abs(pa - pb)
            hi_c = np.maximum(pa, pb) + alpha * np.abs(pa - pb)
            child = rng.uniform(lo_c, hi_c)
            # Gaussian mutation: the mask is the vectorised "for each gene:
            # if rand < p_mut: perturb". Strength = current adaptive sigma.
            mask = rng.random(n) < p_mut
            child[mask] += rng.normal(0.0, 1.0, mask.sum()) * sigma * span[mask]
            np.clip(child, lo, hi, out=child)
            newP.append(child)

        P = np.array(newP)
        fit = np.array([loss(g) for g in P])
        cur = float(fit.min())
        history.append(cur)
        if cur < best_so_far - tol * max(abs(best_so_far), 1e-9):
            best_so_far, since_improve = cur, 0
            best_genome = P[np.argmin(fit)].copy()
        else:
            since_improve += 1

        # --- adaptive mutation controller -----------------------------------
        reheated = since_improve >= patience
        if reheated:
            sigma = min(sigma0, sigma * reheat)     # kick out of the plateau
            since_improve = 0
        else:
            sigma = max(sigma_min, sigma * cool)    # otherwise cool down

        if verbose and (t % log_every == 0 or t == gens - 1):
            flag = "  <-- reheat" if reheated else ""
            print(f"  [{label}] epoch {t + 1:3d}/{gens}   best={cur:.5f}   "
                  f"mean={fit.mean():.5f}   sigma={sigma:.3f}   "
                  f"stall={since_improve}{flag}")

    return best_genome, history


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray,
               a_under: float = 1.0 / 13.0, a_over: float = 1.0 / 10.0) -> float:
    """
    NASA / PHM asymmetric prognostics score (mean per sample).

    d = y_pred - y_true.  Late predictions (d>0: you think more life remains
    than there is) are penalised harder (/10) than early ones (d<0, /13):

        score = mean( exp(|d|/13) - 1 )   where d < 0   (early, safe)
                mean( exp( d /10) - 1 )    where d >= 0  (late, dangerous)

    Uses expm1 for accuracy; 0 at a perfect prediction. Minimising this
    instead of RMSE makes the model prefer to under-predict RUL.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d < 0.0, np.expm1(-d * a_under), np.expm1(d * a_over))
    return float(np.mean(s))


def _roughness_rms(nodes: List[FISNode], genome: np.ndarray) -> float:
    """
    RMS of neighbouring-consequent differences over every FIS grid.

    Adjacent grid cells share an antecedent term, so penalising their
    difference makes the fitted TS surface vary gently instead of fitting
    per-cell noise. Consequents are decoded from the genome (which now also
    carries the centres).
    """
    sq, cnt = 0.0, 0
    for node, (a, b) in zip(nodes, _slices(nodes)):
        _, cons = decode(node, genome[a:b])
        C = cons.reshape(tuple(node.n_terms))
        for ax in range(C.ndim):
            d = np.diff(C, axis=ax)
            sq += float(np.sum(d * d))
            cnt += d.size
    return np.sqrt(sq / cnt) if cnt else 0.0


def smooth_per_unit(frame: pd.DataFrame, cols: List[str], span: int) -> pd.DataFrame:
    """
    EWMA-smooth the given columns along cycles WITHIN each unit.

    The FIS is memoryless, so temporal denoising must happen outside it.
    Applied identically at fit and predict time. span<=1 is a no-op.
    """
    if not span or span <= 1:
        return frame
    order = frame.index
    f = frame.sort_values(["id__-__unit", "id__-__cycle"]).copy()
    for c in cols:
        if c in f.columns:
            f[c] = (f.groupby("id__-__unit")[c]
                    .transform(lambda s: s.ewm(span=span, min_periods=1).mean()))
    return f.loc[order]


def _stage1_input_cols() -> List[str]:
    cols: List[str] = []
    for _, ins, _ in STAGE1_SPECS:
        cols += ins
    return sorted(set(cols))


# ==========================================================================
# 5. TREE ARCHITECTURE  (Stage 1 diagnosis, Stage 2 prognosis)
# ==========================================================================

# Stage 1: (name, input feature columns, theta target). Two measurable
# sensors per node -> 2-input FIS -> few rules. Prunable non-degrading
# components are simply not listed.
STAGE1_SPECS: List[Tuple[str, List[str], str]] = [
    ("HPT_eff",  ["in__HPT__T48","in__HPT__Ps30", "in__HPT__P40"], "theta__HPT__eff"),
    ("HPT_flow", ["in__HPT__T48", "in__HPT__P40"], "theta__HPT__flow"),
    ("LPT_eff",  ["in__LPT__T50", "in__LPT__P50"], "theta__LPT__eff"),
    ("LPT_flow", ["in__LPT__T50", "in__LPT__P50"], "theta__LPT__flow"),
]

# Stage 2 reads three derived channels (built from Stage 1 output + W history)
STAGE2_SPECS: List[Tuple[str, List[str], str]] = [
    ("RUL", ["s2__damage", "s2__slope", "s2__load"], "target__-__RUL"),
]

# Operating-condition context used to build the healthy baseline f(w).
COND_COLS = ["op__-__alt", "op__-__Mach", "op__-__TRA", "op__-__T2"]


# ==========================================================================
# 4b. CONDITION RESIDUALS  (remove the dominant w-effect before the FIS)
#
# Raw cruise-mean sensors are dominated by S_s(w, theta0): flight-to-flight
# condition variation buries the small degradation term J_theta(w)*dtheta.
# We fit a healthy baseline f(w) ~= S_s(w, theta0) on the HEALTHY cycles and
# replace each sensor by its residual  r = x_s - f(w), IN PLACE (same column
# name) so the rest of the tree is untouched. f is a degree-2 polynomial in
# the 4 operating conditions -- crude but it strips the bulk of the w-effect.
# ==========================================================================

def _healthy_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows to fit the baseline on: pre-onset if we know it, else early life."""
    if "time__-__since_onset" in frame.columns:
        m = (frame["time__-__since_onset"].to_numpy() <= 0)
        if m.sum() >= 20:
            return m
    # fallback: first half of each unit's cycles
    age = frame.groupby("id__-__unit")["id__-__cycle"].rank(pct=True).to_numpy()
    return age <= 0.5


def _poly_design(frame: pd.DataFrame, cond_cols: List[str]) -> np.ndarray:
    """Design matrix [1, w, w^2] for the available operating conditions."""
    W = frame[cond_cols].to_numpy(float)
    return np.column_stack([np.ones(len(W)), W, W ** 2])


def fit_condition_baselines(frame: pd.DataFrame, sensor_cols: List[str],
                            cond_cols: List[str]) -> Dict[str, np.ndarray]:
    """Least-squares baseline coefficients per sensor, fit on healthy rows."""
    X = _poly_design(frame, cond_cols)
    mask = _healthy_mask(frame)
    coefs = {}
    for s in sensor_cols:
        y = frame[s].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        coefs[s] = beta
    return coefs


def apply_condition_baselines(frame: pd.DataFrame, coefs: Dict[str, np.ndarray],
                              cond_cols: List[str]) -> pd.DataFrame:
    """Replace each sensor by its residual  x_s - f(w)  (in place, same name)."""
    X = _poly_design(frame, cond_cols)
    f = frame.copy()
    for s, beta in coefs.items():
        f[s] = f[s].to_numpy(float) - X @ beta
    return f


def make_stage2_inputs(frame: pd.DataFrame,
                       theta_pred: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build Stage 2's three inputs from Stage 1 predictions and the W history.

        s2__damage : mean of (-theta_hat) over the predicted modifiers.
                     0 healthy, grows with degradation  -> STATE.
        s2__slope  : per-unit cycle slope of s2__damage  -> TANGENT
                     (rolling mean of the cycle-to-cycle difference).
        s2__load   : per-unit cumulative mission-severity proxy from cruise
                     throttle (op__-__TRA), normalised  -> LOAD / exposure.
    """
    out = frame[["id__-__unit", "id__-__cycle"]].copy()
    dmg = np.mean([-np.asarray(v) for v in theta_pred.values()], axis=0)
    out["s2__damage"] = dmg

    out = out.sort_values(["id__-__unit", "id__-__cycle"])
    g = out.groupby("id__-__unit")["s2__damage"]
    out["s2__slope"] = (
        g.diff().groupby(out["id__-__unit"])
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )

    if "op__-__TRA" in frame.columns:
        tra = frame.loc[out.index, "op__-__TRA"].to_numpy(float)
        sev = (tra - np.nanmean(tra)) / (np.nanstd(tra) or 1.0)   # z-scored thrust
    else:
        sev = np.ones(len(out))
    out["_sev"] = sev
    out["s2__load"] = (
        out.groupby("id__-__unit")["_sev"].cumsum().to_numpy() / 100.0
    )
    return out.drop(columns="_sev").sort_index()


# ==========================================================================
# 6. FIT / PREDICT  (clean end-to-end pipeline)
# ==========================================================================

@dataclass
class GFTConfig:
    n_terms: int = 5          # fuzzy sets per input  (low / med / high)
    pop: int = 100
    gens: int = 200
    seed: int = 0
    verbose: bool = True      # print fitness once per epoch during training
    log_every: int = 1        # 1 = every epoch; raise to thin the log
    cons_pad: float = 0.25    # consequent search box = target range +/- pad*span
    smooth_lambda: float = 0.01   # surface smoothness: neighbour-consequent penalty
    smooth_span: int = 0         # temporal smoothness: EWMA span on inputs (0=off)
    residualize: bool = True     # feed FIS the condition-matched residual x_s - f(w)
    # --- adaptive mutation controller (see genetic_optimize) --------------
    p_mut: float = 0.2           # per-gene mutation probability
    sigma0: float = 0.3          # initial mutation strength (fraction of span)
    sigma_min: float = 0.02      # cooldown floor
    cool: float = 0.93           # geometric cooldown per improving generation
    reheat: float = 4.0          # sigma multiplier applied on stagnation
    patience: int = 4            # flat generations before a reheat


def _fit_stage(frame: pd.DataFrame, nodes: List[FISNode],
               target_cols: List[str], cfg: GFTConfig, seed: int,
               metric: Callable[[np.ndarray, np.ndarray], float] = _rmse,
               label: str = ""
               ) -> Tuple[np.ndarray, List[float], Tuple[float, float]]:
    """
    GA-fit each FIS (centres + consequents) against its target.

    Loss = mean over nodes of `metric(y_true, y_pred)`  (+ optional roughness).
    `metric` is RMSE for Stage 1 (theta) and the NASA score for Stage 2 (RUL).
    The consequent box comes from the target range; centre boxes from the data
    (built by genome_box). The GA is seeded at the quantile-centre layout.
    """
    N = n_params(nodes)
    Y = {n.output: frame[n.output].to_numpy(float) for n in nodes}
    allY = np.concatenate([Y[c] for c in target_cols])
    span = (float(allY.max()) - float(allY.min())) or 1.0
    cons_lo = float(allY.min()) - cfg.cons_pad * span
    cons_hi = float(allY.max()) + cfg.cons_pad * span
    lo, hi, seed_genome = genome_box(nodes, cons_lo, cons_hi)

    def loss(genome: np.ndarray) -> float:
        preds = predict_nodes(frame, nodes, genome)
        data = float(np.mean([metric(Y[c], preds[c]) for c in target_cols]))
        if cfg.smooth_lambda > 0.0:
            return data + cfg.smooth_lambda * _roughness_rms(nodes, genome)
        return data

    genome, history = genetic_optimize(
        loss, N, lo, hi, pop=cfg.pop, gens=cfg.gens, seed=seed,
        p_mut=cfg.p_mut, sigma0=cfg.sigma0, sigma_min=cfg.sigma_min,
        cool=cfg.cool, reheat=cfg.reheat, patience=cfg.patience,
        x0=seed_genome, verbose=cfg.verbose, log_every=cfg.log_every, label=label)
    return genome, history, (cons_lo, cons_hi)


def fit_gft(frame: pd.DataFrame, cfg: GFTConfig = GFTConfig()) -> dict:
    """Train both trees. Returns a plain-dict model (no heavy objects).
    The model carries per-stage GA fitness histories ('history1'/'history2')
    and consequent search boxes ('bounds1'/'bounds2')."""
    # (a) condition residuals: strip the dominant w-effect from the sensors
    s1_cols = _stage1_input_cols()
    conds = [c for c in COND_COLS if c in frame.columns]
    baselines = None
    if cfg.residualize and len(conds) >= 2:
        baselines = fit_condition_baselines(frame, s1_cols, conds)
        frame = apply_condition_baselines(frame, baselines, conds)
    elif cfg.residualize and cfg.verbose:
        print("residualize=True but operating-condition columns missing -> skipped")

    # (b) optional temporal denoising of Stage-1 inputs (FIS is memoryless)
    frame = smooth_per_unit(frame, s1_cols, cfg.smooth_span)

    # ---- Stage 1: sensors -> theta_hat -----------------------------------
    nodes1 = build_nodes(frame, STAGE1_SPECS, cfg.n_terms)
    if cfg.verbose:
        print("Stage 1 FIS:", [n.name for n in nodes1])
    genome1, hist1, bounds1 = _fit_stage(frame, nodes1, [n.output for n in nodes1],
                                         cfg, seed=cfg.seed, metric=_rmse,
                                         label="Stage 1")

    # ---- chain: build Stage 2 inputs from Stage 1 predictions ------------
    theta_pred = predict_nodes(frame, nodes1, genome1)
    s2 = make_stage2_inputs(frame, theta_pred)
    s2["target__-__RUL"] = frame["target__-__RUL"].to_numpy(float)

    # ---- Stage 2: [damage, slope, load] -> RUL  (NASA asymmetric score) --
    nodes2 = build_nodes(s2, STAGE2_SPECS, cfg.n_terms)
    if cfg.verbose:
        print("Stage 2 FIS:", [n.name for n in nodes2])
    genome2, hist2, bounds2 = _fit_stage(s2, nodes2, [n.output for n in nodes2],
                                         cfg, seed=cfg.seed + 1, metric=nasa_score,
                                         label="Stage 2")

    return {"nodes1": nodes1, "genome1": genome1, "history1": hist1, "bounds1": bounds1,
            "nodes2": nodes2, "genome2": genome2, "history2": hist2, "bounds2": bounds2,
            "baselines": baselines, "cond_cols": conds, "cfg": cfg}


def predict_gft(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Full inference: sensors -> theta_hat -> [damage, slope, load] -> RUL.
    Returns a frame with the predicted thetas and RUL alongside the ids."""
    cfg = model.get("cfg", GFTConfig())
    s1_cols = _stage1_input_cols()
    if model.get("baselines") is not None:                 # same residualisation as fit
        frame = apply_condition_baselines(frame, model["baselines"], model["cond_cols"])
    frame = smooth_per_unit(frame, s1_cols, cfg.smooth_span)

    theta_pred = predict_nodes(frame, model["nodes1"], model["genome1"])
    s2 = make_stage2_inputs(frame, theta_pred)
    rul = predict_nodes(s2, model["nodes2"], model["genome2"])["target__-__RUL"]

    out = frame[["id__-__unit", "id__-__cycle"]].copy()
    for k, v in theta_pred.items():
        out[k + "__hat"] = v
    out["RUL_hat"] = np.clip(rul, 0.0, None)
    return out.sort_values(["id__-__unit", "id__-__cycle"]).reset_index(drop=True)


def inspect_consequents(model: dict) -> None:
    """
    Print each FIS's GA-learned centres and consequent grid, plus the
    consequent box and the fraction of consequents pinned at a bound (high =
    box too tight, widen via GFTConfig.cons_pad).
    """
    for tag in ("1", "2"):
        nodes, genome, (lo, hi) = (model["nodes" + tag], model["genome" + tag],
                                   model["bounds" + tag])
        print(f"[stage {tag}]   consequent box = [{lo:.4g}, {hi:.4g}]")
        for node, (a, b) in zip(nodes, _slices(nodes)):
            centers, C = decode(node, genome[a:b])
            grid = tuple(node.n_terms)
            pinned = np.mean(np.isclose(C, lo, atol=1e-3) | np.isclose(C, hi, atol=1e-3))
            print(f"  {node.name:9s} grid{grid}  conseq.range=[{C.min():.4g}, "
                  f"{C.max():.4g}]  pinned={pinned:.0%}")
            print("     learned centers :",
                  [np.round(c, 4).tolist() for c in centers])
            print("     consequents     :", np.round(C.reshape(grid), 4).tolist())


# ==========================================================================
# 7. SELF-TEST  (signal-bearing synthetic frame so learning is visible)
# ==========================================================================

def _synthetic_frame(units=6, cycles=60, seed=0, condition_dominated=True) -> pd.DataFrame:
    """
    Per-cycle feature frame in the extractor's schema, with realistic
    structure so the trees have something to learn:
      - theta drifts negative after an onset (steeper afterwards),
      - each flight has its own operating point (alt/Mach/TRA/T2),
      - sensors carry a LARGE condition effect + a small damage effect
        (condition_dominated=True mimics real DS02, where raw sensors are
        buried under flight-to-flight variation -> residualisation matters),
      - full W context columns + time__-__since_onset are provided,
      - RUL = EOL - cycle.
    """
    rng = np.random.default_rng(seed)
    cw = 1.0 if condition_dominated else 0.1     # weight of the condition effect
    rows = []
    for u in range(1, units + 1):
        eol = cycles + int(rng.integers(-8, 8))
        onset = int(0.5 * eol)
        for k in range(1, eol + 1):
            prog = max(0, k - onset) / max(1, eol - onset)
            d = 0.03 * (0.3 * (k / eol) + 0.7 * prog ** 1.6)      # damage >= 0
            # --- this flight's operating point (varies flight to flight) ----
            alt = 30000 + rng.normal(0, 6000)
            mach = 0.75 + rng.normal(0, 0.05)
            tra = 65 + 10 * np.sin(k / 7.0) + rng.normal(0, 4)
            t2 = 520 + rng.normal(0, 8)
            # condition effect on each sensor (nonlinear in w) -- the confounder
            fc = (0.02 * (alt - 30000) / 1000 + 3.0 * (mach - 0.75)
                  + 0.05 * (tra - 65) + 0.03 * (t2 - 520))
            # sensors = baseline + big condition term + small damage term + noise
            T48 = 1.0 + cw * fc + 9.0 * d + rng.normal(0, 0.01)
            P40 = 1.0 + 0.4 * cw * fc + 2.0 * d + rng.normal(0, 0.01)
            T50 = 1.0 + 0.8 * cw * fc + 7.0 * d + rng.normal(0, 0.01)
            P50 = 1.0 + 0.3 * cw * fc + 1.5 * d + rng.normal(0, 0.01)
            rows.append({
                "id__-__unit": u, "id__-__cycle": k,
                "op__-__alt": alt, "op__-__Mach": mach,
                "op__-__TRA": tra, "op__-__T2": t2,
                "in__HPT__T48": T48, "in__HPT__P40": P40,
                "in__LPT__T50": T50, "in__LPT__P50": P50,
                "theta__HPT__eff": -d, "theta__HPT__flow": -0.6 * d,
                "theta__LPT__eff": -0.8 * d, "theta__LPT__flow": -0.5 * d,
                "time__-__since_onset": float(max(0, k - onset)),
                "target__-__RUL": float(eol - k)})
    return pd.DataFrame(rows)


def _self_test():
    df = _synthetic_frame(units=6, cycles=60, seed=1)
    train = df[df["id__-__unit"] <= 4].reset_index(drop=True)   # units 1-4
    test = df[df["id__-__unit"] > 4].reset_index(drop=True)     # units 5-6

    # --- sanity: Ruspini partition of unity ------------------------------
    c = ruspini_centers(train["in__HPT__T48"].to_numpy(), 3)
    M = memberships(np.linspace(0.9, 1.5, 2000), c)
    print(f"Ruspini partition-of-unity max|sum-1| = {np.abs(M.sum(1) - 1).max():.2e}")

    # --- train ------------------------------------------------------------
    cfg = GFTConfig(n_terms=3, pop=100, gens=200, verbose=True)
    print("\nTraining GFT ...")
    model = fit_gft(train, cfg)

    # --- evaluate ---------------------------------------------------------
    pr_tr = predict_gft(train, model)
    pr_te = predict_gft(test, model)

    def theta_rmse(frame, pred):
        cols = [n.output for n in model["nodes1"]]
        return np.mean([_rmse(pred[c + "__hat"], frame[c]) for c in cols])

    base_te = _rmse(test["target__-__RUL"], np.full(len(test),
                                                    train["target__-__RUL"].mean()))
    print("\n--- results ---")
    print(f"Stage1 theta RMSE   train={theta_rmse(train, pr_tr):.4f}"
          f"   test={theta_rmse(test, pr_te):.4f}")
    print(f"Stage2 RUL   RMSE   train={_rmse(pr_tr['RUL_hat'], train['target__-__RUL']):.2f}"
          f"   test={_rmse(pr_te['RUL_hat'], test['target__-__RUL']):.2f}"
          f"   (baseline mean={base_te:.2f})")
    print("\nsample (test unit, late life):")
    show = pr_te.copy()
    show["RUL_true"] = test["target__-__RUL"].values
    print(show[["id__-__unit", "id__-__cycle", "theta__HPT__eff__hat",
                "RUL_hat", "RUL_true"]].tail(6).to_string(index=False))


if __name__ == "__main__":
    _self_test()