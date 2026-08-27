"""
gft.py -- the fuzzy tree core, rebuilt small. Fixed membership functions only
(learnable MFs were measured and dropped), one FIS, one decode, two fit entry
points that serve the whole curriculum:

    fit_branch(active_components, frame)   -- one fault mode's leaves -> spool(s)
                                              -> RUL, every other branch pruned.
                                              The per-leaf sensor ablation.
    fit_full(frame, seeds=branch_models)   -- the assembled tree
                                              HPT,HPC->hp ; fan,LPC,LPT->lp ;
                                              hp,lp->RUL, warm-started (free) from
                                              the identified branches.

Target architecture (physics, from features.SHAFT):
    HPT, HPC          -> hp        (HP shaft)
    fan, LPC, LPT     -> lp        (LP shaft)
    hp, lp            -> RUL

Node kinds and their constraints:
    leaf  : sensors -> one theta modifier. FREE grid (theta need not be monotone
            in a raw sensor). Antecedent centres = quantiles of the residualized,
            EWMA-smoothed sensor.
    spool : leaf theta -> [0,1] damage. MONOTONE decreasing (more negative
            modifier => more damage) and ANCHORED to span exactly [0,1] (kills the
            gauge freedom in a latent node's scale).
    root  : spool damage -> RUL. MONOTONE decreasing and ANCHORED to [0, rul_cap],
            so RUL(no damage)=cap and RUL(full damage)=0 hold BY CONSTRUCTION.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import ga
from features import (SENSORS, CONDITIONS, COMPONENTS, SHAFT, THETA, GROUPINGS,
                      components_on, theta_of)

# ---- config --------------------------------------------------------------
N_TERMS = 3            # fuzzy sets per input (low/mid/high)
EWMA_SPAN = 7          # causal smoothing window on residualized sensors
HEALTHY_FRAC = 0.2     # first fraction of a unit's life used to fit its baseline
W_THETA = 3.0          # leaf supervision weight -- raised from 1.0: with 10 leaves
                       # the averaged theta term was out-competed by the single RUL
                       # term, so the GA sacrificed leaf fidelity (negative theta R2)
                       # to fit RUL. 3x restores leaf leverage without ignoring RUL.
W_RUL = 1.0            # RUL weight in the loss
W_THETA_FLOOR = 4.0    # HARD penalty per unit of NEGATIVE leaf theta R2. R2<0 means
                       # a leaf predicts its modifier worse than a constant -- it has
                       # stopped reading its component and become free RUL parameters.
                       # This makes that regime expensive by construction, so a leaf
                       # can only be kept if it stays an honest diagnostic.

# spool grouping is the shaft physics; the single source is features.SHAFT
SPOOLS = {"hp": components_on("HP"), "lp": components_on("LP")}   # hp:[HPC,HPT] lp:[fan,LPC,LPT]

# Default observable leaves per component: (leaf_name, target_modifier, sensors).
# These are starting points from the project's leaf_search history + physics; the
# ablation overrides `sensors` with the searched winners and drops unobservable
# leaves (e.g. HPT_flow). The core only needs SOME default to be runnable.
DEFAULT_LEAVES = {
    "HPT": [("hpt_eff",  "HPT_eff_mod",  ["T48", "P40", "Nc"])],
    "HPC": [("hpc_eff",  "HPC_eff_mod",  ["Ps30", "T30", "Nc"])],
    "fan": [("fan_flow", "fan_flow_mod", ["P21", "P15", "P24"])],
    "LPC": [("lpc_eff",  "LPC_eff_mod",  ["P24", "T24", "Nf"])],
    "LPT": [("lpt_flow", "LPT_flow_mod", ["T50", "P50", "P24"]),
            ("lpt_eff",  "LPT_eff_mod",  ["T50", "P50", "Nf"])],
}


# ==========================================================================
# 1. Preprocessing -- per-unit condition residual + causal EWMA (leak-free)
# ==========================================================================

def residualize(frame, sensors, conditions=CONDITIONS, healthy_frac=HEALTHY_FRAC):
    """Remove operating-condition variation per unit: regress each sensor on the
    conditions using only the unit's HEALTHY head, then subtract that fit across
    the whole life. Per-unit => no cross-unit leak; healthy-fit => the residual is
    the health signal, not the flight."""
    out = frame.copy()
    cond = [c for c in conditions if c in frame.columns]
    for _, idx in frame.groupby("unit").groups.items():
        g = frame.loc[idx]
        n_heal = max(N_TERMS + 1, int(healthy_frac * len(g)))
        H = np.column_stack([np.ones(n_heal), g[cond].to_numpy()[:n_heal]])
        A = np.column_stack([np.ones(len(g)), g[cond].to_numpy()])
        for s in sensors:
            beta, *_ = np.linalg.lstsq(H, g[s].to_numpy()[:n_heal], rcond=None)
            out.loc[idx, s] = g[s].to_numpy() - A @ beta
    return out


def smooth(frame, sensors, span=EWMA_SPAN):
    """Causal EWMA per unit -- uses only the past, so no future leak."""
    out = frame.copy()
    for s in sensors:
        out[s] = frame.groupby("unit")[s].transform(lambda x: x.ewm(span=span).mean())
    return out


def prep(frame, sensors):
    return smooth(residualize(frame, sensors), sensors)


# ==========================================================================
# 2. Fuzzy inference -- Ruspini triangular partition + zero-order Sugeno
# ==========================================================================

def _memberships(x, c):
    """Triangular partition of unity over sorted centres c (shoulders clamp)."""
    x = np.asarray(x, float)
    m = len(c)
    M = np.zeros((len(x), m))
    for i in range(m):
        if i == 0:
            M[:, i] = np.interp(x, [c[0], c[1]], [1.0, 0.0])
        elif i == m - 1:
            M[:, i] = np.interp(x, [c[m - 2], c[m - 1]], [0.0, 1.0])
        else:
            M[:, i] = np.interp(x, [c[i - 1], c[i], c[i + 1]], [0.0, 1.0, 0.0])
    return M


def _fis(cols, centres, singl):
    """Zero-order Sugeno on the full rule grid. `cols` are input arrays, `centres`
    the per-input centre vectors, `singl` the flat consequents (len = prod shape)."""
    w = _memberships(cols[0], centres[0])
    for i in range(1, len(cols)):
        w = (w[:, :, None] * _memberships(cols[i], centres[i])[:, None, :])
        w = w.reshape(w.shape[0], -1)
    den = w.sum(1)
    return (w @ singl) / np.where(den > 1e-12, den, 1.0)


def _mono_decode(base, steps, shape, signs, out=None, anchor=False):
    """Consequent grid from non-negative steps => monotone per axis (sign). If
    anchor: min-max normalise then scale into `out`, pinning both corners by
    construction (spool -> [0,1]; root -> [0, rul_cap] so RUL=0 at full damage)."""
    s = np.maximum(np.asarray(steps, float), 0.0).copy()
    s[0] = 0.0
    G = s.reshape(shape)
    rev = tuple(slice(None, None, -1) if g < 0 else slice(None) for g in signs)
    G = G[rev]
    for ax in range(G.ndim):
        G = np.cumsum(G, axis=ax)
    G = G[rev]
    G = float(base) + G
    if anchor:
        lo_, hi_ = float(G.min()), float(G.max())
        G = (G - lo_) / (hi_ - lo_) if (hi_ - lo_) > 1e-9 else np.zeros_like(G)
        if out is not None:
            G = float(out[0]) + G * (float(out[1]) - float(out[0]))
    elif out is not None:
        G = np.clip(G, out[0], out[1])
    return G.ravel()


# ==========================================================================
# 3. Tree specification -- branch and full, from a leaves dict
# ==========================================================================

def _leaves_for(components, leaves=None):
    leaves = leaves or DEFAULT_LEAVES
    return {c: leaves[c] for c in components if c in leaves}


def branch_spec(active, leaves=None, grouping="shaft", age=False):
    """Spec for a tree with ONLY the active components. FOUR levels:
    leaves (sensors->theta) -> COMPONENT nodes (a component's theta modifiers -> a single
    component damage in [0,1]) -> spool nodes (component damages grouped -> spool damage
    [0,1]) -> RUL. The component tier makes each component a [0,1] scalar, so a change of
    `grouping` (shaft hp/lp vs station cold/hot) is a pure re-wiring of component->spool
    edges, not a re-partition of raw leaves. age=True appends `age` at the ROOT only."""
    lv = _leaves_for(active, leaves)
    spec = []
    comp_leaves = {}                                   # component -> [leaf names]
    for comp, entries in lv.items():
        for name, target, sensors in entries:
            spec.append({"name": name, "kind": "leaf", "inputs": list(sensors),
                         "target": target})
            comp_leaves.setdefault(comp, []).append(name)
    for comp, leaf_names in comp_leaves.items():        # component tier
        spec.append({"name": comp, "kind": "component", "inputs": leaf_names,
                     "target": None})
    used = {}                                          # spool tier: components grouped
    for comp in comp_leaves:
        used.setdefault(group_of(comp, grouping), []).append(comp)
    for node, comp_names in used.items():
        spec.append({"name": node, "kind": "spool", "inputs": comp_names, "target": None})
    root_inputs = list(used.keys()) + (["age"] if age else [])
    spec.append({"name": "RUL", "kind": "root", "inputs": root_inputs, "target": None})
    return spec


def full_spec(leaves=None, grouping="shaft", age=False):
    """The assembled target tree under a grouping: every component's leaves ->
    intermediate nodes -> RUL (optionally with age at the root)."""
    return branch_spec(list(COMPONENTS), leaves, grouping, age)


def group_of(component, grouping="shaft"):
    """Intermediate node a component's leaf feeds, under the named grouping."""
    return GROUPINGS[grouping][component]


def owner_spool(model, component):
    """Which spool node does `component` feed, in this BUILT model? Traverses
    leaf -> component -> spool (grouping-agnostic; leak/dead-branch/viz code uses this)."""
    e, fl = theta_of(component)
    leaf_names = [n["name"] for n in model["meta"]
                  if n["kind"] == "leaf" and n["target"] in (e, fl)]
    comp_nodes = [n["name"] for n in model["meta"]
                  if n["kind"] == "component" and any(l in n["inputs"] for l in leaf_names)]
    for n in model["meta"]:
        if n["kind"] == "spool" and any(c in n["inputs"] for c in comp_nodes):
            return n["name"]
    for n in model["meta"]:                              # fallback: legacy leaf->spool
        if n["kind"] == "spool" and any(l in n["inputs"] for l in leaf_names):
            return n["name"]
    return None


# ==========================================================================
# 4. Build -- fix the MFs from data, lay out the genome
# ==========================================================================

def _quantile_centres(x, n=N_TERMS):
    q = np.quantile(np.asarray(x, float), np.linspace(0.05, 0.95, n))
    # guarantee strictly increasing (degenerate if a channel is constant)
    for i in range(1, n):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + 1e-6
    return q


def build_tree(spec, frame, rul_cap):
    """Resolve a spec against data: fixed centres per input, rule slices, signs,
    anchoring, output ranges, and per-node genome bounds. Fixed-MF, so this is the
    whole 'compile' step -- the genome is consequents only."""
    sensors = sorted({s for n in spec if n["kind"] == "leaf" for s in n["inputs"]})
    P = prep(frame, sensors)
    deg = frame.get("since_onset", pd.Series(1.0, index=frame.index)) > 0
    tcols = {n["name"]: n["target"] for n in spec if n["kind"] == "leaf"}

    meta, lo, hi, k = [], [], [], 0
    for n in spec:
        name, kind, inputs = n["name"], n["kind"], n["inputs"]
        centres = []
        for src in inputs:
            if kind == "leaf":                          # antecedent = a sensor
                centres.append(_quantile_centres(P[src]))
            elif kind == "component":                   # antecedent = a leaf theta
                t = frame[tcols[src]].to_numpy(float)
                sub = t[deg.to_numpy()] if deg.any() else t
                centres.append(_quantile_centres(sub if len(sub) else t))
            elif kind == "spool":                       # antecedent = a component damage [0,1]
                centres.append(np.linspace(0.0, 1.0, N_TERMS))
            else:                                       # root inputs
                if src == "age":                          # lifetime prior: age quantiles
                    centres.append(_quantile_centres(frame["age"].to_numpy(float)))
                else:                                     # spool damage in [0,1]
                    centres.append(np.linspace(0.0, 1.0, N_TERMS))
        shape = tuple(len(c) for c in centres)
        n_rules = int(np.prod(shape))
        mono = kind in ("component", "spool", "root")
        # sign = direction of the node output in each input.
        #   component: inputs are theta (negative = worse) -> damage DECREASES in theta -> -1
        #   spool:     inputs are component DAMAGE (high = worse) -> spool damage INCREASES -> +1
        #   root:      inputs are spool damage / age -> RUL DECREASES -> -1  (RUL=0 at full damage)
        if kind == "spool":
            signs = tuple(+1 for _ in inputs)
        elif mono:
            signs = tuple(-1 for _ in inputs)
        else:
            signs = None
        anchor = mono
        out = ((0.0, 1.0) if kind in ("component", "spool") else
               (0.0, float(rul_cap)) if kind == "root" else None)

        # genome bounds for this node's consequents
        if mono:                                        # [base | steps], base dead
            b_lo = [0.0] + [0.0] * (n_rules - 1)
            b_hi = [0.0] + [1.0] * (n_rules - 1)        # ratios only; anchor rescales
        else:                                           # free leaf singletons
            t = frame[n["target"]].to_numpy(float)
            span = float(t.max() - t.min()) or 1.0
            b_lo = [t.min() - 0.25 * span] * n_rules
            b_hi = [t.max() + 0.25 * span] * n_rules
        lo += b_lo
        hi += b_hi

        meta.append({"name": name, "kind": kind, "inputs": inputs,
                     "target": n["target"], "centres": centres, "shape": shape,
                     "signs": signs, "anchor": anchor, "out": out,
                     "rules": slice(k, k + n_rules)})
        k += n_rules

    return {"meta": meta, "lo": np.array(lo), "hi": np.array(hi),
            "sensors": sensors, "rul_cap": float(rul_cap)}


def _consequents(node, genome):
    g = genome[node["rules"]]
    if node["kind"] == "leaf":
        return g
    return _mono_decode(g[0], g, node["shape"], node["signs"],
                        out=node["out"], anchor=node["anchor"])


# ==========================================================================
# 5. Predict
# ==========================================================================

def predict_tree(frame, model, genome):
    meta = model["meta"]
    P = prep(frame, model["sensors"])
    vals = {}
    out = pd.DataFrame({"unit": frame["unit"].to_numpy(),
                        "cycle": frame["cycle"].to_numpy()})
    age_col = frame["age"].to_numpy(float)
    for n in meta:
        cols = [age_col if s == "age" else
                (P[s].to_numpy() if n["kind"] == "leaf" else vals[s])
                for s in n["inputs"]]
        y = _fis(cols, n["centres"], _consequents(n, genome))
        vals[n["name"]] = y
        if n["kind"] == "leaf":
            out[n["target"] + "_hat"] = y
        elif n["kind"] == "component":
            out[n["name"] + "_dmg"] = y
        elif n["kind"] == "spool":
            out[n["name"] + "_h"] = y
        else:
            out["RUL_hat"] = y
    return out


# ==========================================================================
# 6. Metrics
# ==========================================================================

def cap(y, rul_cap):
    y = np.asarray(y, float)
    return np.minimum(y, float(rul_cap)) if rul_cap else y


def rmse(y, yh):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yh, float)) ** 2)))


def r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 1e-12 else 0.0


def nasa(y, yh):
    """PHM08 asymmetric score (mean): late estimates (predicting more life than
    remains) are penalised harder (a=10) than early ones (a=13)."""
    d = np.asarray(yh, float) - np.asarray(y, float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.mean(s))


def unit_corr(units, y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    units = np.asarray(units)
    rs = []
    for u in np.unique(units):
        m = units == u
        if m.sum() < 3 or np.std(y[m]) < 1e-9 or np.std(yh[m]) < 1e-9:
            continue
        rs.append(np.corrcoef(y[m], yh[m])[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def suggest_rul_cap(frame):
    if "since_onset" in frame.columns:
        onset = frame[frame["since_onset"] <= 0].groupby("unit")["RUL"].min()
        if len(onset):
            return float(np.median(onset))
    return float(np.quantile(frame["RUL"], 0.9))


# ==========================================================================
# 7. Loss + fit
# ==========================================================================

def _loss_fn(frame, model):
    meta = model["meta"]
    P = prep(frame, model["sensors"])
    units = frame["unit"].to_numpy()
    y_rul = cap(frame["RUL"].to_numpy(), model["rul_cap"])
    leaves = [n for n in meta if n["kind"] == "leaf"]
    theta = {n["name"]: frame[n["target"]].to_numpy(float) for n in leaves}
    tstd = {k: (np.std(v) or 1.0) for k, v in theta.items()}
    tss = {k: (float(np.sum((v - v.mean()) ** 2)) or 1.0)   # total SS, for R2
           for k, v in theta.items()}
    cols_cache = {s: P[s].to_numpy() for s in model["sensors"]}
    cols_cache["age"] = frame["age"].to_numpy(float)

    def loss(genome):
        vals, tterm, floor = {}, [], 0.0
        for n in meta:
            cols = [cols_cache[s] if (n["kind"] == "leaf" or s == "age") else vals[s]
                    for s in n["inputs"]]
            yv = _fis(cols, n["centres"], _consequents(n, genome))
            vals[n["name"]] = yv
            if n["kind"] == "leaf":
                t = theta[n["name"]]
                tterm.append(rmse(t, yv) / tstd[n["name"]])
                r2 = 1.0 - float(np.sum((t - yv) ** 2)) / tss[n["name"]]
                if r2 < 0.0:                          # leaf worse than a constant
                    floor += -r2                      # how far below zero
        rul_term = rmse(y_rul, vals["RUL"]) / (np.std(y_rul) or 1.0)
        return (W_THETA * (np.mean(tterm) if tterm else 0.0)
                + W_RUL * rul_term
                + W_THETA_FLOOR * floor)

    return loss


def _fit(spec, frame, rul_cap=None, seeds=None, gens=60, pop=60, seed=0,
         grouping="shaft", verbose=False):
    rul_cap = rul_cap if rul_cap is not None else suggest_rul_cap(frame)
    model = build_tree(spec, frame, rul_cap)
    loss = _loss_fn(frame, model)
    g, f, hist = ga.optimize(loss, model["lo"], model["hi"], gens=gens, pop=pop,
                             seed=seed, seeds=seeds, verbose=verbose)
    model["genome"] = g
    model["loss"] = f
    model["spec"] = spec
    model["grouping"] = grouping
    return model


def fit_branch(active, frame, leaves=None, grouping="shaft", **kw):
    """Fit a single-branch (or multi-active) tree on the rows where that branch is
    relevant. Returns the model plus per-leaf theta quality and the branch's
    standalone RUL score -- the two questions each single-fault fit must answer."""
    active = [active] if isinstance(active, str) else list(active)
    spec = branch_spec(active, leaves, grouping)
    model = _fit(spec, frame, grouping=grouping, **kw)
    model["report"] = evaluate(frame, model)
    return model


def fit_full(frame, branch_models=None, leaves=None, grouping="shaft", age=False, **kw):
    """Fit the assembled tree (grouping-dependent intermediate nodes -> RUL). If
    branch_models are given, splice their identified genes into a full-tree seed and
    warm-start the GA from it (free). age=True adds age at the root."""
    spec = full_spec(leaves, grouping, age)
    rul_cap = kw.pop("rul_cap", None)
    rul_cap = rul_cap if rul_cap is not None else suggest_rul_cap(frame)
    model = build_tree(spec, frame, rul_cap)                 # to know the layout
    seeds = None
    if branch_models:
        seeds = [splice_seed(model, branch_models)]
    return _fit(spec, frame, rul_cap=rul_cap, seeds=seeds, grouping=grouping, **kw)


# ==========================================================================
# 7b. DECOUPLED fit (Route 2) -- freeze honest leaves, train only aggregators
# ==========================================================================
# The crosspoint (honest leaves vs good RUL) lives in the aggregation: leaves lose
# fidelity only when a single RUL loss is allowed to distort them. fit_decoupled
# removes that coupling. Phase 1: fit each leaf to its OWN theta on the assembly
# frame (theta-only) and FREEZE it -- honest by construction, RUL can never touch
# it. Phase 2: with leaf genes pinned, the GA trains only the spool + root
# consequents on RUL. The leaves stay transparent; the aggregators absorb the RUL
# burden. RUL is then capped by how much life signal survives in the honest leaves.

def _freeze_leaves(frame, model, gens=30, pop=30, seed=0):
    """Fit each leaf's consequents to its theta on `frame` (theta-only, per leaf --
    independent, so cheap). Returns a full-length genome with leaf slices filled."""
    P = prep(frame, model["sensors"])
    genome = 0.5 * (model["lo"] + model["hi"])
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        cols = [P[s].to_numpy() for s in n["inputs"]]
        y = frame[n["target"]].to_numpy(float)
        lo, hi = model["lo"][n["rules"]], model["hi"][n["rules"]]

        def loss(g, cols=cols, y=y, c=n["centres"]):
            return rmse(y, _fis(cols, c, g))
        g, _, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed)
        genome[n["rules"]] = g
    return genome


def _rul_loss_fn(frame, model):
    """RUL-only loss (used once leaves are frozen -- no theta term, no floor)."""
    P = prep(frame, model["sensors"])
    y_rul = cap(frame["RUL"].to_numpy(), model["rul_cap"])
    ystd = np.std(y_rul) or 1.0
    cols_cache = {s: P[s].to_numpy() for s in model["sensors"]}
    cols_cache["age"] = frame["age"].to_numpy(float)

    def loss(genome):
        vals = {}
        for n in model["meta"]:
            cols = [cols_cache[s] if (n["kind"] == "leaf" or s == "age") else vals[s]
                    for s in n["inputs"]]
            vals[n["name"]] = _fis(cols, n["centres"], _consequents(n, genome))
        return rmse(y_rul, vals["RUL"]) / ystd
    return loss


def fit_decoupled(frame, leaves=None, grouping="shaft", age=False, gens=60, pop=60,
                  seed=0, leaf_gens=30, leaf_pop=30):
    """Route 2: honest frozen leaves + a separately-trained aggregator. Breaks the
    diagnosis/prognosis crosspoint by construction -- the leaves are theta-fit and
    never moved by RUL, so they stay transparent; the spool+root are RUL-fit over
    those frozen signals. age=True adds age at the ROOT only: a lifetime prior for
    the aggregator that CANNOT degenerate the frozen leaves (why age is safe here)."""
    spec = full_spec(leaves, grouping, age)
    rul_cap = suggest_rul_cap(frame)
    model = build_tree(spec, frame, rul_cap)
    frozen = _freeze_leaves(frame, model, gens=leaf_gens, pop=leaf_pop, seed=seed)

    lo, hi = model["lo"].copy(), model["hi"].copy()
    for n in model["meta"]:                                  # pin leaf genes
        if n["kind"] == "leaf":
            lo[n["rules"]] = frozen[n["rules"]]
            hi[n["rules"]] = frozen[n["rules"]]

    loss = _rul_loss_fn(frame, model)
    g, f, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed, seeds=[frozen])
    model["genome"] = g
    model["loss"] = f
    model["spec"] = spec
    model["grouping"] = grouping
    model["age"] = age
    model["decoupled"] = True
    return model


def splice_seed(full_model, branch_models):
    """Build a full-tree seed genome: start random-in-bounds, then copy each
    branch model's node genes into the full genome's matching node slice (matched
    by node name). Nodes absent from a branch keep their random init."""
    rng = np.random.default_rng(0)
    seed = full_model["lo"] + rng.random(len(full_model["lo"])) * (
        full_model["hi"] - full_model["lo"])
    full_by_name = {n["name"]: n for n in full_model["meta"]}
    for bm in branch_models:
        for n in bm["meta"]:
            dst = full_by_name.get(n["name"])
            if dst is not None and dst["shape"] == n["shape"]:
                seed[dst["rules"]] = bm["genome"][n["rules"]]
    return seed


# ==========================================================================
# 8. Evaluate
# ==========================================================================

def evaluate(frame, model):
    pred = predict_tree(frame, model, model["genome"])
    y = cap(frame["RUL"].to_numpy(), model["rul_cap"])
    yh = pred["RUL_hat"].to_numpy()
    rep = {"n_params": len(model["genome"]),
           "RMSE": rmse(y, yh), "NASA": nasa(y, yh), "R2": r2(y, yh),
           "rho_RUL": unit_corr(frame["unit"].to_numpy(), y, yh)}
    for n in model["meta"]:
        if n["kind"] == "leaf":
            t = frame[n["target"]].to_numpy(float)
            th = pred[n["target"] + "_hat"].to_numpy()
            rep[f"{n['target']}:R2"] = r2(t, th)
            rep[f"{n['target']}:rho"] = unit_corr(frame["unit"].to_numpy(), t, th)
    return rep


# ==========================================================================
# 9. Self-test -- invariants + a single-branch fit + warm-started full fit
# ==========================================================================

def _synth(components, units=5, cycles=45, seed=0):
    """Build a pooled-style frame directly (skip the .h5 round-trip) so the core
    is testable in isolation. Sensors respond to the active components' damage."""
    rng = np.random.default_rng(seed)
    gain = {"fan": {"P21": 1.4, "P15": 1.2, "P24": 0.6},
            "LPC": {"P24": 1.4, "T24": 1.2, "Nf": 0.6},
            "HPC": {"Ps30": 1.4, "T30": 1.3, "Nc": 0.9},
            "HPT": {"T48": 9.0, "P40": 2.0, "Nc": 1.5},
            "LPT": {"T50": 7.0, "P50": 1.5, "P24": 0.6}}
    rows = []
    for u in range(units):
        eol = cycles + int(rng.integers(-5, 6)); onset = int(0.5 * eol)
        for c in range(1, eol + 1):
            d = 0.03 * max(0, c - onset) / max(1, eol - onset)
            r = {"unit": u, "cycle": c, "age": float(c),
                 "since_onset": float(max(0, c - onset)),
                 "RUL": float(eol - c),
                 "alt": 30000 + rng.normal(0, 3000), "Mach": 0.75,
                 "TRA": 65.0, "T2": 520.0}
            for s in SENSORS:
                r[s] = 1.0 + rng.normal(0, 0.004)
            for comp in components:
                for s, gm in gain[comp].items():
                    r[s] += gm * d
            for comp in COMPONENTS:
                e, fl = theta_of(comp)
                on = comp in components
                r[e] = -d if on else 0.0
                r[fl] = -0.6 * d if on else 0.0
            rows.append(r)
    return pd.DataFrame(rows)


def _self_test():
    print("== invariants ==")
    df = _synth(["HPT"], seed=1)
    cap_ = suggest_rul_cap(df)
    model = build_tree(branch_spec(["HPT"]), df, cap_)
    rng = np.random.default_rng(0)
    for _ in range(200):                       # monotonicity + PoU on random genomes
        g = model["lo"] + rng.random(len(model["lo"])) * (model["hi"] - model["lo"])
        for n in model["meta"]:
            M = _memberships(np.linspace(-0.05, 0.05, 50), n["centres"][0])
            assert abs(M.sum(1) - 1).max() < 1e-9, "partition of unity broken"
            if n["kind"] in ("spool", "root"):
                G = _consequents(n, g).reshape(n["shape"])
                assert np.all(np.diff(G, axis=0) <= 1e-9), "spool/root not monotone"
    # anchored corners: root spans exactly [0, cap]; spool spans [0,1]
    root = [n for n in model["meta"] if n["kind"] == "root"][0]
    Groot = _consequents(root, g)
    assert abs(Groot.min()) < 1e-6 and abs(Groot.max() - cap_) < 1e-6, (Groot.min(), Groot.max(), cap_)
    print("  partition-of-unity, monotonicity, and 0/cap corners all hold")

    print("\n== single-branch fit (HPT only, DS01-style) ==")
    m_hpt = fit_branch("HPT", df, gens=40, pop=40, seed=0)
    rep = m_hpt["report"]
    print(f"  HPT_eff:  R2={rep['HPT_eff_mod:R2']:.2f}  rho={rep['HPT_eff_mod:rho']:.2f}"
          f"   branch RUL: RMSE={rep['RMSE']:.2f}  NASA={rep['NASA']:.2f}")
    assert rep["HPT_eff_mod:rho"] > 0.5, "leaf should track its own theta"

    print("\n== warm-started full fit (splice HPT + LPT branches) ==")
    df_full = _synth(["HPT", "LPT"], seed=2)
    m_lpt = fit_branch("LPT", _synth(["LPT"], seed=3), gens=30, pop=40, seed=0)
    m_full = fit_full(df_full, branch_models=[m_hpt, m_lpt], gens=40, pop=40, seed=0)
    r = evaluate(df_full, m_full)
    print(f"  full tree: n_params={r['n_params']}  RUL RMSE={r['RMSE']:.2f}  "
          f"NASA={r['NASA']:.2f}  R2={r['R2']:.2f}")
    pred = predict_tree(df_full, m_full, m_full["genome"])
    assert pred["RUL_hat"].min() >= -1e-6, "RUL must stay >= 0"
    assert pred["RUL_hat"].max() <= m_full["rul_cap"] + 1e-6, "RUL must stay <= cap"
    print("  RUL_hat within [0, cap], warm-start spliced without error")
    print("\ngft self-test OK")


if __name__ == "__main__":
    _self_test()