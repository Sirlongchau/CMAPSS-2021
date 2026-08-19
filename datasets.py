"""
datasets.py -- pool the ten N-CMAPSS files into ONE cycle-level frame, and split
it in a way that cannot leak.

WHY POOL. With six units of near-identical lifetime, `RUL = c - age` is a strong
baseline and the GA will happily learn it without touching a sensor. Across all
ten files the lifetimes and flight classes vary, that shortcut collapses, and the
tree is forced onto the sensors. It also takes the parameter ratio from ~1.5 rows
per gene to ~25.

WHY NOT AN 80/10/10 ROW SPLIT. Rows are (unit, cycle) cruise means. Unit 4 cycle
31 and unit 4 cycle 32 are the same engine at the same damage state, differing by
noise. Put one in train and one in test and you have measured interpolation on a
curve you already fitted, not prediction. THE UNIT IS THE INDEPENDENT SAMPLE.
Everything here splits on it.

Typical use:

    import datasets as ds
    print(ds.probe())                 # what theta / units does each file have?
    ds.build_cache()                  # one .h5 at a time -> cache/*.parquet
    feats = ds.pooled()               # all compatible files, globally unique units
    tr, va, te = ds.split(feats)      # 80/10/10 BY UNIT, stratified by dataset
"""

from __future__ import annotations
import re
import glob
import os

import numpy as np
import pandas as pd

from ncmapss_features import (cycle_features, THETA, TREE_THETA,
                              COMPONENT_THETA, TREE_COMPONENTS)

CACHE = "cache"
PATTERN = "N-CMAPSS_DS*.h5"


# ==========================================================================
# Failure modes -- which components each subset actually degrades
# ==========================================================================
# From the N-CMAPSS design (Arias Chao et al. 2021). The tree has leaves for HPT and
# LPT only (TREE_COMPONENTS), so a unit is DIAGNOSABLE iff its mode degrades one of
# those. DS04 (fan), DS05 (HPC), DS06 (LPC+HPC) are BLIND: the tree has no leaf for
# those components. We hold blind units OUT of training but keep them in the TEST
# set (labeled), to measure how far the fan/HPC/LPC signal LEAKING into the HPT/LPT
# sensors carries RUL with no dedicated leaf -- a bounded generalisation result.
# (A fan branch was tried and rolled back: readable alone, not separable in-tree.)
#
# This table is DOCUMENTATION and a cross-check. The diagnosable flag itself is
# derived from the data (tag_diagnosable), so files not listed here (or future
# ones) are still handled correctly -- the two must agree, and _reconcile_diagnosable
# checks that they do.
FAILURE_MODES = {                          # subset tag -> components degraded
    "DS01":  {"HPT"},
    "DS02":  {"HPT", "LPT"},               # not in the design table given; from data
    "DS03":  {"HPT", "LPT"},
    "DS04":  {"fan"},                      # fan-only -> BLIND (no fan leaf)
    "DS05":  {"HPC"},                      # HPC-only -> BLIND
    "DS06":  {"LPC", "HPC"},               # LPC+HPC  -> BLIND
    "DS07":  {"LPT"},
    "DS08a": {"fan", "LPC", "HPC", "HPT", "LPT"},   # all modes -> readable via HPT/LPT
    "DS08c": {"fan", "LPC", "HPC", "HPT", "LPT"},
}


def mode_key(tag):
    """Map a pooled ds tag to its FAILURE_MODES key by stripping the file suffix:
    'DS01-005' -> 'DS01', 'DS08a-009' -> 'DS08a', 'DS04' -> 'DS04'. Without this the
    suffixed files never match FAILURE_MODES, so their failure_mode reads '?' and
    _reconcile_diagnosable silently skips them."""
    m = re.match(r"(DS\d+[a-z]?)", str(tag))
    return m.group(1) if m else str(tag)


def mode_is_diagnosable(components, tree_components=TREE_COMPONENTS):
    """A failure mode is diagnosable iff it degrades a component the tree reads."""
    return bool(set(components) & set(tree_components))


def tag_diagnosable(frame, tree_theta=TREE_THETA, eps=1e-6):
    """Add a per-unit boolean `diagnosable`: can THIS tree read this unit's
    degradation at all? True iff at least one HPT/LPT modifier is non-trivially
    non-zero somewhere in the unit's life.

    DATA-DRIVEN on purpose: the actual theta is the single source of truth, so
    this is correct for DS02/DS09 or any file absent from FAILURE_MODES. A
    fan-/LPC-/HPC-only unit has all TREE_THETA identically 0 and comes out False --
    the tree has no leaf for its degrading component, so hp=lp=0 for its whole life
    and the root cannot explain its RUL. Also carries `damage_theta_peak` (the
    largest |modifier| the tree can see on the unit) for inspection."""
    f = frame.copy()
    cols = [c for c in tree_theta if c in f.columns]
    if not cols:
        f["damage_theta_peak"] = 0.0
        f["diagnosable"] = False
        return f
    peak = f.groupby("unit")[cols].apply(lambda d: float(np.abs(d.to_numpy()).max()))
    f["damage_theta_peak"] = f["unit"].map(peak).astype(float)
    f["diagnosable"] = f["damage_theta_peak"] > eps
    return f


def split_diagnosable(frame):
    """(diagnosable, undiagnosable) sub-frames. Fit and score the HPT/LPT damage
    model on the first; report the second separately as a known blind spot."""
    if "diagnosable" not in frame.columns:
        frame = tag_diagnosable(frame)
    return (frame[frame["diagnosable"]].reset_index(drop=True),
            frame[~frame["diagnosable"]].reset_index(drop=True))


def files(pattern=PATTERN):
    return sorted(glob.glob(pattern))


# ==========================================================================
# 0. Verify -- an .h5 that half-downloaded will take the whole run down with it
# ==========================================================================

def verify(pattern=PATTERN):
    """Try to open every file and report what happened. Run this before anything
    else, especially after a fresh download.

    The classic failure is a TRUNCATED file: HDF5 stores the byte offset where
    the file should end, and h5py refuses to open it if the file on disk is
    shorter ("truncated file: eof = N, stored_eof = M"). A shortfall of a few
    bytes means the download was cut off, or the file was never fully hydrated.

    KEEP THESE FILES OUT OF ONEDRIVE / DROPBOX / GOOGLE DRIVE. With sync-on-demand
    a file can sit on disk as a partially materialised placeholder, and h5py seeks
    all over it -- the worst possible access pattern for a synced folder. Put them
    in a plain local directory."""
    import h5py
    rows = []
    for p in files(pattern):
        size = os.path.getsize(p)
        try:
            with h5py.File(p, "r") as h:
                keys = len(list(h.keys()))
            rows.append(dict(file=os.path.basename(p), GB=round(size / 2 ** 30, 2),
                             status="ok", detail=f"{keys} groups"))
        except Exception as e:
            msg = str(e).split("(")[-1].rstrip(")")
            rows.append(dict(file=os.path.basename(p), GB=round(size / 2 ** 30, 2),
                             status="BROKEN", detail=msg))
    df = pd.DataFrame(rows)
    bad = df[df["status"] == "BROKEN"]
    if len(bad):
        print(f"\n{len(bad)} unreadable file(s) -- re-download them, and make sure "
              f"they are NOT in a cloud-synced folder:")
        print(bad.to_string(index=False))
    return df


def _open(path):
    """h5py.File, or None with a message. Never lets one bad file stop the run."""
    import h5py
    try:
        return h5py.File(path, "r")
    except Exception as e:
        print(f"  SKIP {os.path.basename(path)}: {e}")
        return None


# ==========================================================================
# 1. Probe -- read the fault modes off the files instead of trusting a table
# ==========================================================================

def probe(pattern=PATTERN):
    """One row per .h5: which theta modifiers it actually contains, how many
    units, and the lifetime spread. Run this FIRST -- the tree's four leaves
    (HPT/LPT eff & flow) only exist in some of the files; others degrade the fan,
    the LPC or the HPC instead. Unreadable files are reported and skipped; see
    verify()."""
    rows = []
    for p in files(pattern):
        h = _open(p)
        if h is None:
            continue
        with h:
            th = [n.decode().strip() if isinstance(n, bytes) else str(n).strip()
                  for n in np.array(h["T_var"]).ravel()]
            out = {"file": os.path.basename(p), "theta": th,
                   "has_tree_theta": set(THETA) <= set(th)}
            for sp in ("dev", "test"):
                A = h.get(f"A_{sp}")
                Y = h.get(f"Y_{sp}")
                if A is None:
                    continue
                u = np.array(A)[:, 0].round().astype(int)
                life = pd.Series(np.array(Y).ravel()).groupby(u).max() + 1
                out[f"{sp}_units"] = int(len(np.unique(u)))
                out[f"{sp}_life"] = f"{life.min():.0f}-{life.max():.0f}"
        rows.append(out)
    if not rows:
        raise RuntimeError(f"no readable .h5 matched {pattern!r} -- run verify()")
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Cache -- one .h5 at a time; never hold ten of them in RAM
# ==========================================================================


def _file_theta(path):
    """Cheap read of a file's theta modifier names (the T_var name list only, not
    the multi-GB arrays), intersected with ALL_THETA -- i.e. exactly the theta
    columns cycle_features WOULD write for this file. Used to detect a cache that
    predates a schema change (e.g. fan theta added to extraction). None if the
    file can't be opened."""
    h = _open(path)
    if h is None:
        return None
    with h:
        th = [n.decode().strip() if isinstance(n, bytes) else str(n).strip()
              for n in np.array(h["T_var"]).ravel()]
    return [c for c in ALL_THETA if c in th]

def build_cache(pattern=PATTERN, cache=CACHE, force=False, **kw):
    """Reduce each .h5 to cycle-level cruise means and park it as parquet.

    load_raw() pulls a whole split into memory as one array, so a 3.7 GB file is
    several GB while it is being reduced. Do them ONE AT A TIME (this loop), and
    afterwards the whole fleet is a few thousand rows and a few MB.

    A file that fails to open (truncated download, half-hydrated cloud file) is
    reported and skipped -- the rest of the fleet still caches. Returns the list
    of files that failed."""
    os.makedirs(cache, exist_ok=True)
    broken = []
    for p in files(pattern):
        tag = os.path.basename(p).replace("N-CMAPSS_", "").replace(".h5", "")
        out = os.path.join(cache, f"{tag}.parquet")
        if os.path.exists(out) and not force:
            # Reuse the cache ONLY if it already holds every theta column this file
            # would now yield. A schema change (e.g. fan theta added to ALL_THETA)
            # leaves old parquets missing columns; pooled() would then silently
            # fill them with 0 and mislabel the file. Detect that and rebuild it.
            want = _file_theta(p)
            if want is not None:
                import pyarrow.parquet as pq
                have = set(pq.read_schema(out).names)         # footer only, cheap
                missing = [c for c in want if c not in have]
                if missing:
                    print(f"  REBUILD {tag}: cache predates schema, missing "
                          f"{missing}")
                else:
                    print(f"  skip {tag} (cached)")
                    continue
            else:
                print(f"  skip {tag} (cached, unreadable source -- cannot verify)")
                continue
        parts, failed = [], False
        for sp in ("dev", "test"):
            try:
                parts.append(cycle_features(p, split=sp, **kw).assign(split=sp))
            except OSError as e:                       # unreadable / truncated file
                print(f"  BROKEN {tag}: {e}")
                failed = True
                break
            except Exception as e:                     # split absent -- fine
                print(f"  {tag}/{sp}: {e}")
        if failed:
            broken.append(p)
            continue
        if parts:
            pd.concat(parts, ignore_index=True).assign(ds=tag).to_parquet(out)
            print(f"  {tag}: {sum(len(x) for x in parts)} cycle-rows -> {out}")
    if broken:
        print(f"\n{len(broken)} file(s) could not be read and were LEFT OUT of the "
              f"cache:\n  " + "\n  ".join(os.path.basename(b) for b in broken) +
              "\nRe-download them, and keep them out of any cloud-synced folder.")
    return broken


# ==========================================================================
# 3. Pool -- globally unique unit ids, missing theta filled with 0
# ==========================================================================

def pooled(cache=CACHE, include=None, theta=THETA, fill_missing_theta=True):
    """Concatenate the cached files into one frame.

    unit ids COLLIDE across files (every .h5 numbers its units from 1), so the
    pooled `unit` is remapped to ds_index*1000 + local_unit; the originals stay in
    `unit_local` and `ds`. `split` keeps each file's own dev/test tag.

    A file whose T_var lacks e.g. HPT_flow_mod never degraded that component, so
    the modifier is physically 0 for all its units. fill_missing_theta writes that
    0 in. For the LEAF this is valid supervision -- it teaches the leaf to output
    zero when its component reads nominal. But it does NOT make the unit
    diagnosable: a unit that degrades ONLY the fan/LPC/HPC has every tree modifier
    at 0 for its whole life, so the RUL root sees no damage and can only guess the
    no-damage prior. `pooled` therefore tags each unit `diagnosable`
    (tag_diagnosable) and attaches its `failure_mode`, so callers can fit and score
    the HPT/LPT damage model on the units it can actually read and report the rest
    separately (see split_diagnosable). Set fill_missing_theta False to keep only
    files that carry all four tree modifiers."""
    got = sorted(glob.glob(os.path.join(cache, "*.parquet")))
    if not got:
        raise FileNotFoundError(f"no parquet in {cache}/ -- run build_cache() first")
    frames = []
    for i, p in enumerate(got):
        tag = os.path.basename(p).replace(".parquet", "")
        if include and tag not in include:
            continue
        f = pd.read_parquet(p)
        missing = [t for t in theta if t not in f.columns]
        if missing and not fill_missing_theta:
            print(f"  drop {tag}: missing {missing}")
            continue
        for t in missing:
            f[t] = 0.0
        if missing:
            print(f"  {tag}: {missing} not degraded in this file -> theta = 0")
        f["unit_local"] = f["unit"]
        f["unit"] = 1000 * (i + 1) + f["unit"]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    # label each unit's failure mode (documented) and whether the tree can read it
    # (data-driven), then reconcile the two so a drift is caught early.
    out["failure_mode"] = out["ds"].map(
        lambda t: "+".join(sorted(FAILURE_MODES[mode_key(t)]))
        if mode_key(t) in FAILURE_MODES else "?")
    out = tag_diagnosable(out)
    _reconcile_diagnosable(out)

    n_units = out["unit"].nunique()
    n_diag = out.loc[out["diagnosable"], "unit"].nunique()
    print(f"pooled: {len(out)} cycle-rows, {n_units} units, "
          f"{out['ds'].nunique()} datasets  "
          f"({n_diag} diagnosable, {n_units - n_diag} blind to HPT/LPT leaves)")
    blind = sorted(out.loc[~out["diagnosable"], "ds"].unique())
    if blind:
        print(f"  UNDIAGNOSABLE modes (no HPT/LPT degradation, tree has no leaf): "
              f"{', '.join(blind)}")
    return out


def _reconcile_diagnosable(frame):
    """Warn if the data-driven `diagnosable` flag disagrees with FAILURE_MODES for
    any known subset -- e.g. a mislabelled file or an unexpected zero theta."""
    key = frame["ds"].map(mode_key)
    for tag, comps in FAILURE_MODES.items():
        m = key == tag
        if not m.any():
            continue
        expect = mode_is_diagnosable(comps)
        got = bool(frame.loc[m, "diagnosable"].any())
        if expect != got:
            print(f"  WARNING: {tag} FAILURE_MODES says diagnosable={expect} but the "
                  f"data says {got} -- check T_var / fill_missing_theta.")


# ==========================================================================
# 4. Splits -- always by unit
# ==========================================================================

def split(frame, fracs=(0.8, 0.1, 0.1), seed=0, stratify="ds"):
    """80/10/10 BY UNIT, stratified within each dataset so every fault mode is
    represented in all three sets. A unit is never split across sets.

    Small datasets are rounded UP to at least one val and one test unit each --
    with ~10 units per file, exact 80/10/10 within a file is not achievable and a
    silently empty test set is far worse than a slightly off ratio.

    Tune hyper-parameters (theta_weight, n_terms, gens) on VAL. Touch TEST once."""
    rng = np.random.default_rng(seed)
    keys = {"train": [], "val": [], "test": []}
    groups = frame.groupby(stratify) if stratify else [(None, frame)]
    for _, g in groups:
        u = rng.permutation(g["unit"].unique())
        n = len(u)
        if n < 3:                                   # too few to carve up
            keys["train"] += list(u)
            continue
        n_val = max(1, int(round(fracs[1] * n)))
        n_test = max(1, int(round(fracs[2] * n)))
        keys["test"] += list(u[:n_test])
        keys["val"] += list(u[n_test:n_test + n_val])
        keys["train"] += list(u[n_test + n_val:])
    out = tuple(frame[frame["unit"].isin(v)].reset_index(drop=True)
                for v in keys.values())
    assert not (set(keys["train"]) & set(keys["val"])), "unit in train AND val"
    assert not (set(keys["train"]) & set(keys["test"])), "unit in train AND test"
    assert not (set(keys["val"]) & set(keys["test"])), "unit in val AND test"
    print("split by unit -- " + "   ".join(
        f"{k}: {len(v)} units / {len(f)} rows"
        for (k, v), f in zip(keys.items(), out)))
    return out


def official_split(frame):
    """The .h5 files' own dev/test groups. Use this for the headline number --
    it is the only one comparable with published N-CMAPSS results."""
    return (frame[frame["split"] == "dev"].reset_index(drop=True),
            frame[frame["split"] == "test"].reset_index(drop=True))


def leave_one_dataset_out(frame):
    """Yield (train, test, ds) with one whole FILE held out. This is the hard
    question -- can the tree read a fault mode it has never seen? A fuzzy tree
    with physically named leaves ought to degrade more gracefully here than a
    black box, and if it does, that is the contribution."""
    for d in sorted(frame["ds"].unique()):
        m = frame["ds"] == d
        yield (frame[~m].reset_index(drop=True),
               frame[m].reset_index(drop=True), d)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print("verifying files ...")
    print(verify().to_string(index=False))
    print("\nprobing fault modes ...")
    print(probe().to_string(index=False))
    print("\nbuilding cache (one file at a time) ...")
    build_cache()
    feats = pooled()
    split(feats)