"""
features.py -- N-CMAPSS .h5  ->  one row per (unit, cycle), full-engine edition.

This is the rebuilt preprocessing for the FULL fuzzy tree. The old pipeline loaded
only the HPT/LPT health parameters (four modifiers); the full tree diagnoses all
five rotating components, so we now load all TEN modifiers:

    fan   : fan_eff_mod   fan_flow_mod          (LP shaft)
    LPC   : LPC_eff_mod   LPC_flow_mod          (LP shaft)
    HPC   : HPC_eff_mod   HPC_flow_mod          (HP shaft)
    HPT   : HPT_eff_mod   HPT_flow_mod          (HP shaft)
    LPT   : LPT_eff_mod   LPT_flow_mod          (LP shaft)

theta is imposed once per flight (constant within a cycle at 1 Hz), so every
feature is a cruise-mean aggregated to the cycle level. Output columns use plain
physical names so the fuzzy tree can reference sensors and modifiers directly:

    unit, cycle, age, RUL, since_onset,
    alt, Mach, TRA, T2,                 (operating conditions, cruise means)
    <cruise-mean real sensors: Nf, T48, P40, ... from X_s>,
    <whichever of the 10 theta modifiers the file actually carries>

Design decisions kept from the old pipeline (still correct, see docs/01_features.md):
  * REAL measured sensors only (X_s). Virtual sensors (X_v) are never loaded, so
    nothing downstream can cheat with a virtual channel.
  * cruise-mean aggregation on the high-altitude part of each cycle.
  * `age` = 1-indexed cycle count per unit; `since_onset` = cycles since the
    health-state flag hs flipped 0->1 (0 while healthy).
  * a file that never degraded a component simply lacks that modifier column; we
    load what is present and let the pooling layer (data.py) fill the rest with 0.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ==========================================================================
# Canonical physical metadata -- the single source of truth for the whole
# package. tree.py builds its leaves/spools from COMPONENTS + SHAFT; data.py
# fills missing theta from THETA. Nothing else defines these names.
# ==========================================================================

CONDITIONS = ["alt", "Mach", "TRA", "T2"]

# REAL measured sensors only (C-MAPSS Table 2, X_s). Virtual sensors (Table 3,
# X_v) are deliberately NOT loaded.
SENSORS = ["Wf", "Nf", "Nc", "T24", "T30", "T48", "T50",
           "P15", "P21", "P24", "Ps30", "P40", "P50"]

# The five rotating components, each with an efficiency and a flow modifier, and
# the shaft each sits on. This mapping IS the physics the tree groups by:
#   LP shaft (N1): fan, LPC (booster), LPT
#   HP shaft (N2): HPC, HPT
COMPONENTS = {
    "fan": ("fan_eff_mod", "fan_flow_mod"),
    "LPC": ("LPC_eff_mod", "LPC_flow_mod"),
    "HPC": ("HPC_eff_mod", "HPC_flow_mod"),
    "HPT": ("HPT_eff_mod", "HPT_flow_mod"),
    "LPT": ("LPT_eff_mod", "LPT_flow_mod"),
}
SHAFT = {"fan": "LP", "LPC": "LP", "LPT": "LP", "HPC": "HP", "HPT": "HP"}

# Flat list of all ten modifiers, in component order.
THETA = [m for pair in COMPONENTS.values() for m in pair]


def components_on(shaft):
    """Component names sitting on a given shaft ('HP' or 'LP')."""
    return [c for c, s in SHAFT.items() if s == shaft]


def theta_of(component):
    """The (eff, flow) modifier names of a component."""
    return COMPONENTS[component]


# ==========================================================================
# Loading
# ==========================================================================

def _names(arr):
    """Decode a *_var group (byte-strings, possibly 2-D) to clean strings."""
    return [(n.decode() if isinstance(n, (bytes, bytearray)) else str(n)).strip()
            for n in np.array(arr).ravel()]


def load_raw(path, split="dev") -> pd.DataFrame:
    """Load one .h5 split into a per-timestamp (1 Hz) DataFrame.

    X_v (virtual sensors) is intentionally NOT loaded -- real sensors only. Only
    the theta modifiers physically present in this file's T_var are read; a file
    that never degraded a component has no column for it here."""
    if h5py is None:
        raise ImportError("h5py is required (pip install h5py).")

    def get(h, key):
        node = h.get(key)
        return None if node is None else np.array(node)

    with h5py.File(path, "r") as h:
        parts, cols = [], []
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
    """One row per (unit, cycle): cruise means of sensors + conditions + whichever
    theta modifiers are present, plus age, since_onset and RUL."""
    df = load_raw(path, split)

    # keep the high-altitude (cruise) part of each cycle before averaging
    if "alt" in df.columns:
        cyc_max = df.groupby(["unit", "cycle"])["alt"].transform("max")
        df = df[df["alt"] >= cruise_frac * cyc_max]

    keep = ([c for c in CONDITIONS if c in df.columns]
            + [c for c in SENSORS + THETA if c in df.columns])
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


# ==========================================================================
# Tiny synthetic .h5 so the pipeline is testable without the multi-GB files.
# `components` picks which components degrade; only their modifiers move and only
# their modifier names go into T_var -- mimicking real files that carry a subset.
# data.py reuses this to build a per-component multi-file fleet for its tests.
# ==========================================================================

def _make_synthetic_h5(path, units=(1, 2, 3), cycles=40, spc=120,
                       components=("HPT", "LPT"), seed=0, split_name="dev"):
    xv = ["P30", "P45", "W22", "W25", "W48", "W50", "epr",
          "NRf", "NRc", "SmFan", "SmLPC", "SmHPC"]
    # T_var carries only the degrading components' modifiers (like the real files)
    tv = [m for c in components for m in COMPONENTS[c]]
    rng = np.random.default_rng(seed)

    # per-component sensor sensitivity: which sensors a component's damage moves.
    gain = {
        "fan": {"Nf": 1.0, "P21": 1.4, "P15": 1.2},
        "LPC": {"P24": 1.4, "T24": 1.2, "Nf": 0.6},
        "HPC": {"Ps30": 1.4, "T30": 1.3, "Nc": 0.9},
        "HPT": {"T48": 9.0, "P40": 2.0, "Nc": 1.5},
        "LPT": {"T50": 7.0, "P50": 1.5, "P24": 0.6},
    }

    W, Xs, Xv, T, Y, A = [], [], [], [], [], []
    for u in units:
        onset = cycles // 2
        for c in range(1, cycles + 1):
            prog = max(0, c - onset) / max(1, cycles - onset)
            d = 0.03 * (0.3 * c / cycles + 0.7 * prog ** 1.6)   # damage >= 0
            alt = np.concatenate([np.linspace(0, 35000, spc // 2),
                                  np.full(spc - spc // 2, 35000.0)])
            mach = rng.normal(0.7, 0.02, spc)
            tra = rng.normal(65, 3, spc)
            t2 = rng.normal(520, 2, spc)
            W.append(np.stack([alt, mach, tra, t2], 1))

            xs = rng.normal(1.0, 0.01, (spc, len(SENSORS)))
            for comp in components:
                for s, gm in gain[comp].items():
                    xs[:, SENSORS.index(s)] += gm * d
            Xs.append(xs)
            Xv.append(rng.normal(1.0, 0.01, (spc, len(xv))))

            th = rng.normal(0, 1e-4, (spc, len(tv)))
            for comp in components:
                th[:, tv.index(COMPONENTS[comp][0])] += -d          # eff  falls
                th[:, tv.index(COMPONENTS[comp][1])] += -0.6 * d    # flow falls
            T.append(th)

            A.append(np.stack([np.full(spc, u), np.full(spc, c),
                               np.full(spc, 3),
                               np.full(spc, 1 if c >= onset else 0)], 1))
            Y.append(np.full((spc, 1), float(cycles - c)))

    data = {"W": np.vstack(W), "X_s": np.vstack(Xs), "X_v": np.vstack(Xv),
            "T": np.vstack(T), "Y": np.vstack(Y), "A": np.vstack(A)}
    varnames = {"W_var": CONDITIONS, "X_s_var": SENSORS, "X_v_var": xv,
                "T_var": tv, "A_var": ["unit", "cycle", "Fc", "hs"]}
    with h5py.File(path, "w") as h:
        for name, arr in data.items():
            h.create_dataset(f"{name}_{split_name}", data=arr)
        for name, var in varnames.items():
            h.create_dataset(name, data=np.array(var, dtype="S"))


def _self_test():
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "_ncmapss_full.h5")
    _make_synthetic_h5(tmp, cycles=20, components=("HPT", "LPT", "fan"))
    feats = cycle_features(tmp, split="dev", min_cruise=10)
    print("dev features shape:", feats.shape)
    print("columns:", list(feats.columns))
    theta_present = [c for c in THETA if c in feats.columns]
    print("theta present:", theta_present)
    assert set(theta_present) == {"HPT_eff_mod", "HPT_flow_mod",
                                  "LPT_eff_mod", "LPT_flow_mod",
                                  "fan_eff_mod", "fan_flow_mod"}, theta_present
    assert (feats["age"] >= 1).all()
    assert (feats["since_onset"] >= 0).all()
    # onset must actually fire (hs flips mid-life)
    assert feats["since_onset"].max() > 0
    print("physics metadata:  COMPONENTS =", list(COMPONENTS))
    print("                   LP components =", components_on("LP"))
    print("                   HP components =", components_on("HP"))
    print(feats.head().to_string(index=False))
    os.remove(tmp)
    print("\nfeatures self-test OK")


if __name__ == "__main__":
    _self_test()