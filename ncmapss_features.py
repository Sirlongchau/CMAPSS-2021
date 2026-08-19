"""
ncmapss_features.py -- N-CMAPSS .h5  ->  one row per (unit, cycle).

theta is imposed once per flight (constant within a cycle at 1 Hz), so every
feature is a cruise-mean aggregated to the cycle level. Output columns use
plain physical names so the fuzzy tree can reference sensors directly:

    unit, cycle, age, RUL, since_onset,
    alt, Mach, TRA, T2,                 (operating conditions, cruise means)
    <cruise-mean sensors: Nf, T48, P40, ... from X_s and X_v>

`age` = 1-indexed cycle count per unit; `since_onset` = cycles since the
health-state flag hs flipped 0->1 (0 while healthy). No smoothing, no schema
objects, no config class -- just a function that returns a DataFrame.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

CONDITIONS = ["alt", "Mach", "TRA", "T2"]

# REAL measured sensors only (C-MAPSS Table 2, X_s). Virtual sensors (Table 3,
# X_v: T40 P30 P45 W21 W22 W25 W31 W32 W48 W50 epr Sm* NR* PCNfR phi) are
# deliberately NOT loaded, so nothing downstream can use a virtual sensor.
SENSORS = ["Wf", "Nf", "Nc", "T24", "T30", "T48", "T50",
           "P15", "P21", "P24", "Ps30", "P40", "P50"]

# Health-parameter modifiers (theta). There are two per rotating sub-component:
#   TREE_THETA  -- the modifiers the tree's leaves cover (HPT/LPT). Used for the
#                  leaf_search target list and for DIAGNOSABILITY: a unit the tree
#                  can read must degrade HPT or LPT. HPT_flow is kept in the search
#                  list (that is how it was shown unobservable) though it has no
#                  leaf; a unit is still diagnosable on HPT via HPT_eff.
#   ALL_THETA   -- every component's eff/flow pair, all EXTRACTED. This matters even
#                  though the tree reads only HPT/LPT: the fan/LPC/HPC channels are
#                  what LABEL a unit's failure mode, so the blind fan/HPC/LPC units
#                  (DS04/05/06) can be identified and held out of training yet still
#                  put in the test set -- the LP-leak generalisation experiment. A
#                  dedicated fan branch was tried and rolled back (fan readable alone
#                  but not separable from the LP spool in-tree; see gft._LEAVES), so
#                  fan/LPC/HPC are NOT tree components here.
TREE_THETA = ["HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"]
ALL_THETA = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod",
             "HPC_eff_mod", "HPC_flow_mod", "HPT_eff_mod", "HPT_flow_mod",
             "LPT_eff_mod", "LPT_flow_mod"]
THETA = TREE_THETA          # backwards-compatible alias (datasets, leaf_search)

# component -> its two modifiers, and the components the tree can actually read.
COMPONENT_THETA = {
    "fan": ["fan_eff_mod", "fan_flow_mod"],
    "LPC": ["LPC_eff_mod", "LPC_flow_mod"],
    "HPC": ["HPC_eff_mod", "HPC_flow_mod"],
    "HPT": ["HPT_eff_mod", "HPT_flow_mod"],
    "LPT": ["LPT_eff_mod", "LPT_flow_mod"],
}
TREE_COMPONENTS = ["HPT", "LPT"]      # the only spools with leaves in gft.TREE


def _names(arr):
    """Decode a *_var group (byte-strings, possibly 2-D) to clean strings."""
    return [(n.decode() if isinstance(n, (bytes, bytearray)) else str(n)).strip()
            for n in np.array(arr).ravel()]


def load_raw(path, split="dev") -> pd.DataFrame:
    """Load one .h5 split into a per-timestamp (1 Hz) DataFrame."""
    if h5py is None:
        raise ImportError("h5py is required (pip install h5py).")

    def get(h, key):
        node = h.get(key)
        return None if node is None else np.array(node)

    with h5py.File(path, "r") as h:
        parts, cols = [], []
        # X_v (virtual sensors) is intentionally NOT loaded -- real sensors only.
        for grp, var in [("A", "A_var"), ("W", "W_var"),
                         ("X_s", "X_s_var"), ("T", "T_var")]:
            data = get(h, f"{grp}_{split}")
            if data is not None:
                parts.append(data)
                cols += _names(get(h, var))
        Y = get(h, f"Y_{split}")
    df = pd.DataFrame(np.concatenate(parts, axis=1), columns=cols)
    df["RUL"] = np.asarray(Y).ravel()
    for c in ("unit", "cycle"):
        df[c] = df[c].round().astype(int)
    return df


def cycle_features(path, split="dev", cruise_frac=0.95, min_cruise=30):
    """One row per (unit, cycle): cruise means of sensors + conditions,
    plus age, since_onset and RUL."""
    df = load_raw(path, split)

    # keep the high-altitude (cruise) part of each cycle before averaging
    if "alt" in df.columns:
        cyc_max = df.groupby(["unit", "cycle"])["alt"].transform("max")
        df = df[df["alt"] >= cruise_frac * cyc_max]

    keep = ([c for c in CONDITIONS if c in df.columns]
            + [c for c in SENSORS + ALL_THETA if c in df.columns])
    g = df.groupby(["unit", "cycle"], sort=True)
    out = g[keep].mean()
    out["RUL"] = g["RUL"].median()
    out["_n"] = g.size()

    if "hs" in df.columns:                        # onset = first cycle with hs==1
        out["_hs"] = g["hs"].max()

    out = out[out["_n"] >= min_cruise].drop(columns="_n").reset_index()
    out = out.sort_values(["unit", "cycle"]).reset_index(drop=True)

    out["age"] = out.groupby("unit").cumcount() + 1.0
    if "_hs" in out.columns:
        onset = out.loc[out["_hs"] >= 1].groupby("unit")["age"].min()
        out["since_onset"] = (out["age"] - out["unit"].map(onset)).clip(lower=0).fillna(0.0)
        out = out.drop(columns="_hs")
    else:
        out["since_onset"] = 0.0
    return out


# ---- tiny synthetic .h5 so the pipeline is testable without the real file --

def _make_synthetic_h5(path, dev_units=(1, 2), test_units=(3,), cycles=8, spc=120):
    xs = ["Wf", "Nf", "Nc", "T24", "T30", "T48", "T50",
          "P15", "P21", "P24", "Ps30", "P40", "P50"]
    xv = ["P30", "P45", "W22", "W25", "W48", "W50", "epr",
          "NRf", "NRc", "SmFan", "SmLPC", "SmHPC"]
    tv = ["HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"]

    def split(units, seed):
        rng = np.random.default_rng(seed)
        W, Xs, Xv, T, Y, A = [], [], [], [], [], []
        for u in units:
            onset = cycles // 2
            for c in range(1, cycles + 1):
                alt = np.concatenate([np.linspace(0, 35000, spc // 2),
                                      np.full(spc - spc // 2, 35000.0)])
                W.append(np.stack([alt, rng.normal(0.7, 0.02, spc),
                                   rng.normal(65, 3, spc), rng.normal(520, 2, spc)], 1))
                Xs.append(rng.normal(1, 0.01, (spc, len(xs))))
                Xv.append(rng.normal(1, 0.01, (spc, len(xv))))
                deg = -0.02 * max(0, c - onset) / max(1, cycles - onset)
                th = rng.normal(0, 1e-4, (spc, len(tv)))
                th[:, tv.index("HPT_eff_mod")] += deg
                th[:, tv.index("LPT_eff_mod")] += 0.8 * deg
                T.append(th)
                A.append(np.stack([np.full(spc, u), np.full(spc, c),
                                   np.full(spc, 3), np.full(spc, 1 if c >= onset else 0)], 1))
                Y.append(np.full((spc, 1), float(cycles - c)))
        return [np.vstack(z) for z in (W, Xs, Xv, T, Y, A)]

    with h5py.File(path, "w") as h:
        for suffix, seed in [("dev", 0), ("test", 1)]:
            data = split(dev_units if suffix == "dev" else test_units, seed)
            for name, arr in zip(["W", "X_s", "X_v", "T", "Y", "A"], data):
                h.create_dataset(f"{name}_{suffix}", data=arr)
        for name, var in [("W_var", CONDITIONS), ("X_s_var", xs), ("X_v_var", xv),
                          ("T_var", tv), ("A_var", ["unit", "cycle", "Fc", "hs"])]:
            h.create_dataset(name, data=np.array(var, dtype="S"))


def _self_test():
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "_ncmapss_synth.h5")
    _make_synthetic_h5(tmp, cycles=8)
    feats = cycle_features(tmp, split="dev", min_cruise=10)
    print("dev features shape:", feats.shape)
    print("columns:", list(feats.columns))
    print(feats.head().to_string(index=False))
    os.remove(tmp)


if __name__ == "__main__":
    _self_test()