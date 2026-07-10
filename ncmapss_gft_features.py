"""
ncmapss_gft_features.py
=======================

Feature extraction for a TWO-STAGE Genetic Fuzzy Tree (GFT) on the N-CMAPSS
run-to-failure dataset (Arias Chao et al., 2021).

Group names, var-name decoding and the dev/test layout match the official
NASA/Kaggle example notebook. That notebook also collapses theta with
df_T.drop_duplicates() -> confirming theta is constant within a flight cycle,
which is exactly why every feature here is aggregated to one row per
(unit, cycle). notebook_frames() bridges back to its plotting helpers.

    Stage 1  (one sub-FIS per engine component)
        cruise sensor features  ->  theta_<component>_<mod>      [diagnosis]

    Stage 2  (aggregating GFT)
        theta, d(theta)/dcycle, time-features  ->  RUL           [prognosis]

--------------------------------------------------------------------------
WHY CYCLE-LEVEL?
    theta is imposed ONCE PER FLIGHT (constant within a cycle at 1 Hz and
    stepping between cycles). Deriving theta on the raw 1 Hz stream gives ~0
    in flight and spikes at cycle borders. So EVERYTHING here is aggregated
    to one row per (unit, cycle); derivatives are taken cycle-to-cycle.

--------------------------------------------------------------------------
EXPLICIT NAMING CONVENTION  (this is what makes GFT nesting mechanical)

    Every feature column is  <role>__<node>__<var>   (separator = "__").

        role   in {id, op, in, theta, dtheta, slope, time, hi, target}
        node   the FIS this variable belongs to (e.g. HPT, LPT, GFT_RUL, -)
        var    the physical / derived quantity

    Split a column with  col.split("__")  ->  [role, node, var].
    A FeatureSchema object (see build_schema) lists, per FIS node, the exact
    input column names and the output column name, so wiring the tree is:

        for node in schema.stage1:
            fis = make_fis(inputs=node.inputs, output=node.outputs)   # your code

--------------------------------------------------------------------------
Roles
    id__     : identifiers (unit, cycle)                         -- keys, not FIS inputs
    op__     : cruise operating-condition context (alt/Mach/TRA/T2)
    in__     : STAGE-1 sub-FIS INPUTS (cruise sensor means, per component)
    theta__  : STAGE-1 TARGETS / STAGE-2 INPUTS (health parameters)
    dtheta__ : STAGE-2 INPUTS  (first difference of theta per cycle)
    slope__  : STAGE-2 INPUTS  (smoothed local slope of theta)
    time__   : STAGE-2 INPUTS  (age, time since onset, time since maintenance)
    hi__     : STAGE-2 INPUT   (health-index proxy from theta)
    target__ : STAGE-2 TARGET  (RUL, in cycles)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ==========================================================================
# 1. DOMAIN MAPS  --  which sensors feed which component FIS, and which
#    theta each component owns. Edit these to reshape the tree.
# ==========================================================================

# component -> sensor variable names (looked up across X_s AND X_v).
# Only sensors physically informative for that component are routed to it;
# this is exactly the per-node input reduction that GFTs exploit.
COMPONENT_SENSORS: Dict[str, List[str]] = {
    "Fan": ["Nf", "NRf", "P15", "P21", "W21", "SmFan"],
    "LPC": ["T24", "P24", "W22", "W25", "SmLPC"],
    "HPC": ["T30", "Ps30", "P30", "NRc", "SmHPC"],
    "HPT": ["T48", "P40", "P45", "W48"],
    "LPT": ["T50", "P50", "W50", "epr"],
}

# component -> its two health parameters, mapped to short target names.
# Source theta names in the .h5 T-group:  <comp>_eff_mod / <comp>_flow_mod
COMPONENT_THETA: Dict[str, Dict[str, str]] = {
    "Fan": {"eff": "fan_eff_mod", "flow": "fan_flow_mod"},
    "LPC": {"eff": "LPC_eff_mod", "flow": "LPC_flow_mod"},
    "HPC": {"eff": "HPC_eff_mod", "flow": "HPC_flow_mod"},
    "HPT": {"eff": "HPT_eff_mod", "flow": "HPT_flow_mod"},
    "LPT": {"eff": "LPT_eff_mod", "flow": "LPT_flow_mod"},
}

# In DS01-style data only HPT (and HPT+LPT) actually degrade; the other
# component modifiers are near-constant initial offsets. Non-active nodes
# are still emitted but flagged, so you can prune Stage 1 to the movers.
DEFAULT_ACTIVE_COMPONENTS = ("HPT", "LPT")


# ==========================================================================
# 2. CONFIG
# ==========================================================================

@dataclass
class Config:
    split: str = "dev"                 # "dev" or "test"
    # Cruise selection: keep samples whose altitude is within `cruise_alt_frac`
    # of the per-cycle max altitude (stable, condition-consistent segment).
    cruise_alt_frac: float = 0.95
    min_cruise_samples: int = 30       # cycles with fewer cruise pts are dropped
    slope_window: int = 5              # cycles used for the smoothed theta slope
    # Maintenance-reset detection: a positive jump of theta (health recovery)
    # larger than this (per cycle) starts a new "time since maintenance" count.
    # In monotone datasets none are found and time__since_maint == age_cycles.
    maint_recovery_threshold: float = 5e-3
    active_components: tuple = DEFAULT_ACTIVE_COMPONENTS


# ==========================================================================
# 3. LOADING  --  read the .h5 groups into a single tidy 1 Hz frame
# ==========================================================================

def _decode_var_names(arr) -> List[str]:
    """
    The *_var groups store variable names as byte-strings (and sometimes as a
    2-D (n,1) array). Return a flat list of clean python strings.

    Equivalent to the Kaggle notebook's  list(np.array(var, dtype='U20'))
    but robust to bytes vs. numpy.str_ and to 2-D shapes.
    """
    flat = np.array(arr).ravel()
    names = []
    for n in flat:
        if isinstance(n, (bytes, bytearray)):
            n = n.decode()
        names.append(str(n).strip())
    return names


def load_h5_raw(path: str, split: str = "dev") -> pd.DataFrame:
    """
    Load one N-CMAPSS .h5 file into a per-timestamp (1 Hz) DataFrame.

    Columns produced:
        unit, cycle, Fc, hs,
        alt, Mach, TRA, T2,                     (operating conditions W)
        <all X_s sensors>, <all X_v sensors>,   (measured + virtual)
        <all theta>,                            (health parameters T)
        RUL                                     (target Y)
    """
    if h5py is None:
        raise ImportError("h5py is required to read .h5 files (pip install h5py).")

    def g(h, key):
        node = h.get(key)
        return None if node is None else np.array(node)

    with h5py.File(path, "r") as h:
        s = split
        W   = g(h, f"W_{s}")
        Xs  = g(h, f"X_s_{s}")
        Xv  = g(h, f"X_v_{s}")
        T   = g(h, f"T_{s}")
        Y   = g(h, f"Y_{s}")
        A   = g(h, f"A_{s}")

        W_var  = _decode_var_names(g(h, "W_var"))
        Xs_var = _decode_var_names(g(h, "X_s_var"))
        Xv_var = _decode_var_names(g(h, "X_v_var"))
        T_var  = _decode_var_names(g(h, "T_var"))
        A_var  = _decode_var_names(g(h, "A_var"))

    frames = []
    frames.append(pd.DataFrame(A,  columns=A_var))      # unit, cycle, Fc, hs
    frames.append(pd.DataFrame(W,  columns=W_var))      # alt, Mach, TRA, T2
    frames.append(pd.DataFrame(Xs, columns=Xs_var))
    frames.append(pd.DataFrame(Xv, columns=Xv_var))
    frames.append(pd.DataFrame(T,  columns=T_var))
    df = pd.concat(frames, axis=1)
    df["RUL"] = np.asarray(Y).ravel()

    # normalise identifier dtypes
    for c in ("unit", "cycle"):
        if c in df.columns:
            df[c] = df[c].round().astype(int)
    return df


# ==========================================================================
# 4. CRUISE MASK  --  reduce flight-condition variability before aggregating
# ==========================================================================

def cruise_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Boolean mask keeping the high-altitude (cruise) portion of each cycle."""
    if "alt" not in df.columns:
        return pd.Series(True, index=df.index)
    cyc_max = df.groupby(["unit", "cycle"])["alt"].transform("max")
    return df["alt"] >= (cfg.cruise_alt_frac * cyc_max)


# ==========================================================================
# 5. STAGE-1 FEATURES  --  cruise sensor means per component  ->  in__<comp>__<sensor>
#    and the theta targets                                    ->  theta__<comp>__<eff|flow>
# ==========================================================================

def _present(cols: List[str], available) -> List[str]:
    return [c for c in cols if c in available]


def build_cycle_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (unit, cycle). Emits id__, op__, in__ and theta__ columns."""
    avail = set(df.columns)
    cruise = df[cruise_mask(df, cfg)].copy()

    grouped = cruise.groupby(["unit", "cycle"], sort=True)
    n_cruise = grouped.size().rename("n_cruise")

    out = pd.DataFrame(index=n_cruise.index)

    # ---- operating-condition context (cruise means) -> op__ ---------------
    for w in ("alt", "Mach", "TRA", "T2"):
        if w in avail:
            out[f"op__-__{w}"] = grouped[w].mean()

    # ---- STAGE-1 INPUTS: per-component cruise sensor means -> in__ --------
    for comp, sensors in COMPONENT_SENSORS.items():
        for s in _present(sensors, avail):
            out[f"in__{comp}__{s}"] = grouped[s].mean()

    # ---- STAGE-1 TARGETS: theta (constant in a cycle -> take the mean) ----
    for comp, mods in COMPONENT_THETA.items():
        for kind, src in mods.items():
            if src in avail:
                out[f"theta__{comp}__{kind}"] = grouped[src].mean()

    # ---- carry RUL target and cruise count -------------------------------
    if "RUL" in avail:
        out["target__-__RUL"] = grouped["RUL"].median()
    out["_n_cruise"] = n_cruise

    out = out.reset_index()  # -> columns unit, cycle
    out = out.rename(columns={"unit": "id__-__unit", "cycle": "id__-__cycle"})

    # drop cycles with too little cruise data (noisy aggregates)
    out = out[out["_n_cruise"] >= cfg.min_cruise_samples].drop(columns="_n_cruise")
    return out.sort_values(["id__-__unit", "id__-__cycle"]).reset_index(drop=True)


# ==========================================================================
# 6. STAGE-2 FEATURES  --  derivatives, slopes, time, HI proxy
# ==========================================================================

def _theta_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("theta__")]


def add_theta_dynamics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-unit cycle-to-cycle derivative (dtheta__) and smoothed slope (slope__)."""
    df = df.sort_values(["id__-__unit", "id__-__cycle"]).copy()
    g = df.groupby("id__-__unit", sort=False)

    for tcol in _theta_cols(df):
        _, comp, kind = tcol.split("__")
        # first difference per cycle
        df[f"dtheta__{comp}__{kind}"] = g[tcol].diff()
        # robust local slope: rolling mean of the first difference
        df[f"slope__{comp}__{kind}"] = (
            g[tcol].diff()
            .groupby(df["id__-__unit"])
            .transform(lambda s: s.rolling(cfg.slope_window, min_periods=1).mean())
        )
    # first cycle of each unit has no previous value -> fill 0 change
    dyn_cols = [c for c in df.columns if c.startswith(("dtheta__", "slope__"))]
    df[dyn_cols] = df[dyn_cols].fillna(0.0)
    return df


def add_time_features(df: pd.DataFrame, raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    time__age_cycles     : cycles elapsed since start of life (1-indexed)
    time__since_onset    : cycles since hs flipped 0->1 (0 while healthy)
    time__since_maint    : cycles since last theta-recovery reset,
                           == age_cycles when no maintenance events exist.
    """
    df = df.sort_values(["id__-__unit", "id__-__cycle"]).copy()
    unit = df["id__-__unit"]

    # age = running cycle count per unit (1-indexed)
    df["time__-__age_cycles"] = (df.groupby(unit).cumcount() + 1).astype(float)

    # onset: age at the first cycle whose health-state flag hs == 1.
    # Fully vectorized (no groupby.apply -> robust to single-unit splits).
    if "hs" in raw.columns:
        hs_cycle = (
            raw.groupby(["unit", "cycle"])["hs"].max()
            .rename("hs").reset_index()
            .rename(columns={"unit": "id__-__unit", "cycle": "id__-__cycle"})
        )
        df = df.merge(hs_cycle, on=["id__-__unit", "id__-__cycle"], how="left")
        df["hs"] = df["hs"].fillna(0)
        onset_age = (
            df.loc[df["hs"] >= 1]
            .groupby("id__-__unit")["time__-__age_cycles"].min()
        )
        oa = df["id__-__unit"].map(onset_age)                     # NaN if never degrades
        df["time__-__since_onset"] = (
            (df["time__-__age_cycles"] - oa).clip(lower=0).fillna(0.0)
        )
        df = df.drop(columns="hs")
    else:
        df["time__-__since_onset"] = 0.0

    # maintenance resets: a positive jump of the HI proxy = health recovery.
    # since_maint = position within the block that starts at each reset;
    # with no resets (monotone DS) it equals age_cycles.
    df = add_hi_proxy(df)
    d_hi = df.groupby(df["id__-__unit"])["hi__-__proxy"].diff()
    is_reset = (d_hi > cfg.maint_recovery_threshold).fillna(False)
    block = is_reset.groupby(df["id__-__unit"]).cumsum()
    reset_grp = (
        df["id__-__unit"].astype(int).astype(str) + "_" + block.astype(int).astype(str)
    )
    df["time__-__since_maint"] = (df.groupby(reset_grp).cumcount() + 1).astype(float)
    return df


def add_hi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    hi__proxy : simple health-index proxy from theta (NOT the ground-truth HI,
    which ships only with DS03). All modifiers start ~0 and drift negative, so
    proxy = 1 - ||theta|| / ref  in [~1 healthy .. lower when degraded].
    """
    if "hi__-__proxy" in df.columns:
        return df
    tcols = _theta_cols(df)
    if not tcols:
        df["hi__-__proxy"] = np.nan
        return df
    norm = np.sqrt((df[tcols] ** 2).sum(axis=1))
    ref = norm.replace(0, np.nan).quantile(0.99)
    ref = ref if (ref and not np.isnan(ref)) else 1.0
    df["hi__-__proxy"] = 1.0 - (norm / ref)
    return df


# ==========================================================================
# 7. SCHEMA  --  declares the tree topology for GFT wiring
# ==========================================================================

@dataclass
class FISNode:
    name: str                       # e.g. "HPT" or "GFT_RUL"
    inputs: List[str]               # exact feature-column names
    outputs: List[str]              # exact target/output column names
    active: bool = True
    role: str = "stage1"            # "stage1" | "stage2"


@dataclass
class FeatureSchema:
    stage1: List[FISNode] = field(default_factory=list)
    stage2: Optional[FISNode] = None

    def describe(self) -> str:
        lines = ["STAGE 1 (per-component diagnosis sub-FIS):"]
        for n in self.stage1:
            flag = "" if n.active else "  [inactive/near-constant -> prunable]"
            lines.append(f"  [{n.name}]{flag}")
            lines.append(f"      inputs : {n.inputs}")
            lines.append(f"      output : {n.outputs}")
        if self.stage2:
            n = self.stage2
            lines.append("STAGE 2 (aggregating GFT -> RUL):")
            lines.append(f"  [{n.name}]")
            lines.append(f"      inputs : {n.inputs}")
            lines.append(f"      output : {n.outputs}")
        return "\n".join(lines)


def build_schema(features: pd.DataFrame, cfg: Config) -> FeatureSchema:
    """Derive the node->columns wiring from the columns actually present."""
    cols = list(features.columns)

    def by(role: str, node: Optional[str] = None) -> List[str]:
        out = []
        for c in cols:
            parts = c.split("__")
            if parts[0] != role:
                continue
            if node is not None and parts[1] != node:
                continue
            out.append(c)
        return out

    stage1: List[FISNode] = []
    for comp in COMPONENT_SENSORS:
        ins = by("in", comp)
        outs = by("theta", comp)
        if not outs:                       # no theta for this comp in file
            continue
        stage1.append(
            FISNode(
                name=comp,
                inputs=ins,
                outputs=outs,
                active=comp in cfg.active_components,
                role="stage1",
            )
        )

    # Stage 2 consumes theta + dynamics + time + HI proxy of the ACTIVE comps.
    active = set(cfg.active_components)
    def active_only(prefix: str) -> List[str]:
        return [c for c in cols
                if c.startswith(prefix) and c.split("__")[1] in active]

    stage2_inputs = (
        active_only("theta__")
        + active_only("dtheta__")
        + active_only("slope__")
        + by("hi")
        + by("time")
    )
    stage2 = FISNode(
        name="GFT_RUL",
        inputs=stage2_inputs,
        outputs=by("target"),
        active=True,
        role="stage2",
    )
    return FeatureSchema(stage1=stage1, stage2=stage2)


# ==========================================================================
# 8. TOP-LEVEL PIPELINE
# ==========================================================================

_COL_ORDER = {"id": 0, "op": 1, "in": 2, "theta": 3, "dtheta": 4,
              "slope": 5, "hi": 6, "time": 7, "target": 8}


def _order_columns(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[sorted(feats.columns,
                        key=lambda c: (_COL_ORDER.get(c.split("__")[0], 9), c))]


def _extract_single(path: str, split: str, cfg: Config) -> pd.DataFrame:
    """Run the whole pipeline for ONE split ('dev' or 'test')."""
    raw = load_h5_raw(path, split)
    feats = build_cycle_features(raw, cfg)
    feats = add_theta_dynamics(feats, cfg)   # groups by unit -> no cross-split leak
    feats = add_time_features(feats, raw, cfg)
    feats = add_hi_proxy(feats)
    return feats


def extract_features(path: str, cfg: Optional[Config] = None):
    """
    Full pipeline: raw .h5  ->  (features_df, schema).

    features_df : one row per (unit, cycle), every column named
                  <role>__<node>__<var>.
    schema      : FeatureSchema describing how to wire the two-stage GFT.

    cfg.split :
        "dev"  -> development units only  (training)
        "test" -> test units only         (final evaluation)
        "both" -> both, with an extra  id__-__split  column ('dev'/'test').
                  Dev and test units never share numbers, and every derived
                  quantity is computed within a split, so training on dev and
                  validating on test is just a filter on id__-__split.
                  (The Kaggle notebook concatenates the two with no flag; we
                  add the flag so the boundary stays explicit.)
    """
    cfg = cfg or Config()

    if cfg.split == "both":
        parts = []
        for sp in ("dev", "test"):
            f = _extract_single(path, sp, cfg)
            f.insert(0, "id__-__split", sp)
            parts.append(f)
        feats = pd.concat(parts, axis=0, ignore_index=True)
    else:
        feats = _extract_single(path, cfg.split, cfg)

    feats = _order_columns(feats)
    schema = build_schema(feats, cfg)
    return feats, schema


# ==========================================================================
# 8b. NOTEBOOK BRIDGE  --  reuse the NASA/Kaggle example's plotting helpers
# ==========================================================================

def notebook_frames(path: str, split: str = "dev"):
    """
    Return the raw per-timestamp frames the way the NASA/Kaggle example
    notebook names them: (df_W, df_X_s, df_X_v, df_T, df_A, Y).

    df_T here is the FULL 1 Hz frame; to reproduce the notebook's per-cycle
    theta table do:  df_T.drop_duplicates(subset=['unit', 'cycle'])
    (theta is constant within a flight -- this is why we aggregate by cycle).
    """
    raw = load_h5_raw(path, split)
    W_cols  = [c for c in ("alt", "Mach", "TRA", "T2") if c in raw.columns]
    Xs_cols = [c for grp in ("Fan", "LPC", "HPC", "HPT", "LPT")
               for c in COMPONENT_SENSORS[grp] if c in raw.columns]
    theta_cols = [v for m in COMPONENT_THETA.values()
                  for v in m.values() if v in raw.columns]
    df_A = raw[[c for c in ("unit", "cycle", "Fc", "hs") if c in raw.columns]].copy()
    df_W = raw[W_cols].copy();          df_W[["unit", "cycle"]] = df_A[["unit", "cycle"]]
    df_T = raw[theta_cols].copy();      df_T[["unit", "cycle"]] = df_A[["unit", "cycle"]]
    df_X_s = raw[[c for c in Xs_cols]].copy()
    df_X_v = raw[[c for c in raw.columns
                  if c not in set(W_cols) | set(theta_cols)
                  | {"unit", "cycle", "Fc", "hs", "RUL"} and c not in Xs_cols]].copy()
    return df_W, df_X_s, df_X_v, df_T, df_A, raw["RUL"].to_numpy()


# ==========================================================================
# 9. SELF-TEST  --  build a tiny synthetic file and run the pipeline
# ==========================================================================

_SYNTH_XS_VAR = ["Wf", "Nf", "Nc", "T24", "T30", "T48", "T50",
                 "P15", "P21", "P24", "Ps30", "P40", "P50"]
_SYNTH_XV_VAR = ["T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32", "W48",
                 "W50", "epr", "SmFan", "SmLPC", "SmHPC", "NRf", "NRc", "PCNfR", "phi"]
_SYNTH_T_VAR = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod",
                "HPC_eff_mod", "HPC_flow_mod", "HPT_eff_mod", "HPT_flow_mod",
                "LPT_eff_mod", "LPT_flow_mod"]


def _synth_split(units, cycles, spc, seed):
    """Generate arrays (W, Xs, Xv, T, Y, A) for a set of unit ids."""
    rng = np.random.default_rng(seed)
    W, Xs, Xv, T, Y, A = [], [], [], [], [], []
    for u in units:
        eol, onset = cycles, cycles // 2
        for c in range(1, cycles + 1):
            alt = np.concatenate([np.linspace(0, 35000, spc // 2),
                                  np.full(spc - spc // 2, 35000.0)])
            W.append(np.stack([alt, rng.normal(0.7, 0.02, spc),
                               rng.normal(65, 3, spc), rng.normal(520, 2, spc)], axis=1))
            Xs.append(rng.normal(1.0, 0.01, (spc, len(_SYNTH_XS_VAR))))
            Xv.append(rng.normal(1.0, 0.01, (spc, len(_SYNTH_XV_VAR))))
            deg = -0.02 * max(0, c - onset) / max(1, eol - onset)
            th = rng.normal(0, 1e-4, (spc, len(_SYNTH_T_VAR)))
            th[:, _SYNTH_T_VAR.index("HPT_eff_mod")] += deg
            th[:, _SYNTH_T_VAR.index("LPT_eff_mod")] += 0.8 * deg
            th[:, _SYNTH_T_VAR.index("LPT_flow_mod")] += 0.6 * deg
            T.append(th)
            A.append(np.stack([np.full(spc, u), np.full(spc, c),
                               np.full(spc, 3), np.full(spc, 1 if c >= onset else 0)], axis=1))
            Y.append(np.full((spc, 1), float(eol - c)))
    return (np.vstack(W), np.vstack(Xs), np.vstack(Xv),
            np.vstack(T), np.vstack(Y), np.vstack(A))


def _make_synthetic_h5(path: str, dev_units=(1, 2), test_units=(3,),
                       cycles: int = 8, spc: int = 120):
    """Create an N-CMAPSS-like file with BOTH dev and test groups + var names."""
    W_var, A_var = ["alt", "Mach", "TRA", "T2"], ["unit", "cycle", "Fc", "hs"]
    dev = _synth_split(dev_units, cycles, spc, seed=0)
    test = _synth_split(test_units, cycles, spc, seed=1)
    with h5py.File(path, "w") as h:
        for suffix, data in [("dev", dev), ("test", test)]:
            for name, arr in zip(["W", "X_s", "X_v", "T", "Y", "A"], data):
                h.create_dataset(f"{name}_{suffix}", data=arr)
        for name, var in [("W_var", W_var), ("X_s_var", _SYNTH_XS_VAR),
                          ("X_v_var", _SYNTH_XV_VAR), ("T_var", _SYNTH_T_VAR),
                          ("A_var", A_var)]:
            h.create_dataset(name, data=np.array(var, dtype="S"))


def _self_test():
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "_ncmapss_synth.h5")
    _make_synthetic_h5(tmp, dev_units=(1, 2), test_units=(3,), cycles=8)

    feats, schema = extract_features(tmp, Config(split="dev", min_cruise_samples=10))
    print("DEV features shape:", feats.shape)
    print("\ncolumns:")
    for c in feats.columns:
        print("   ", c)
    print("\n" + schema.describe())
    print("\nhead:")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(feats.head(6).to_string(index=False))

    # split="both" path + notebook bridge
    feats_both, _ = extract_features(tmp, Config(split="both", min_cruise_samples=10))
    print("\nBOTH features shape:", feats_both.shape,
          "| splits:", feats_both["id__-__split"].unique().tolist())
    df_W, df_X_s, df_X_v, df_T, df_A, Y = notebook_frames(tmp, "dev")
    per_cycle_theta = df_T.drop_duplicates(subset=["unit", "cycle"])
    print("notebook bridge -> df_A cols:", list(df_A.columns),
          "| per-cycle theta rows:", len(per_cycle_theta))
    os.remove(tmp)


# ==========================================================================
# 10. CLI
# ==========================================================================

def main():
    p = argparse.ArgumentParser(description="N-CMAPSS -> GFT feature extraction")
    p.add_argument("h5", nargs="?", help="path to N-CMAPSS_DSxx.h5 (omit to self-test)")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--out", default=None, help="write features to .parquet/.csv")
    args = p.parse_args()

    if args.h5 is None:
        _self_test()
        return

    feats, schema = extract_features(args.h5, Config(split=args.split))
    print(schema.describe())
    print("\nfeatures shape:", feats.shape)
    if args.out:
        if args.out.endswith(".csv"):
            feats.to_csv(args.out, index=False)
        else:
            feats.to_parquet(args.out, index=False)
        print("written:", args.out)


if __name__ == "__main__":
    main()
