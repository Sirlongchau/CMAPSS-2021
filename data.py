"""
data.py -- pool the N-CMAPSS files into ONE cycle-level frame and split it so it
cannot leak. Rebuilt to carry all ten theta modifiers (the full tree needs every
component supervised where it is present).

WHY POOL. A single fault-mode file has near-uniform lifetimes, so `RUL = c - age`
is a strong shortcut the GA will learn without touching a sensor. Across the ten
files lifetimes and flight classes vary, the shortcut collapses, and the tree is
forced onto the sensors. Pooling also lifts the row-per-parameter ratio.

WHY SPLIT BY UNIT, NEVER BY ROW. Rows are (unit, cycle) cruise means; two cycles
of one engine are the same machine at neighbouring damage states. Put one in
train and one in test and you have measured interpolation, not prediction. THE
UNIT IS THE INDEPENDENT SAMPLE.

WHY LEAVE-ONE-DATASET-OUT IS THE HEADLINE GENERALIZATION TEST. Each N-CMAPSS file
is (broadly) one failure mode. Holding a whole file out asks: can the tree read a
fault mode it has never trained on? For a component-named fuzzy tree that is the
contribution, so leave_one_dataset_out is first-class here, not an afterthought.

Typical use:
    import data
    print(data.probe())           # what theta / units does each file have?
    data.build_cache()            # one .h5 at a time -> cache/*.parquet
    feats = data.pooled()         # all files, globally unique units, theta filled
    print(data.coverage(feats))   # which datasets degrade which components
    tr, va, te = data.split(feats)
"""

from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd

from features import cycle_features, THETA, COMPONENTS

CACHE = "cache"
PATTERN = "N-CMAPSS_DS*.h5"


def files(pattern=PATTERN):
    return sorted(glob.glob(pattern))


# ==========================================================================
# 0. Verify -- a half-downloaded .h5 takes the whole run down with it
# ==========================================================================

def verify(pattern=PATTERN):
    """Try to open every file and report what happened. The classic failure is a
    TRUNCATED file (HDF5 stores the expected end offset and h5py refuses to open a
    file shorter than that). Keep these files OUT of cloud-synced folders: with
    sync-on-demand a file can sit as a partial placeholder and h5py seeks all over
    it -- the worst access pattern for a synced folder."""
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
        print(f"\n{len(bad)} unreadable file(s) -- re-download, keep out of any "
              f"cloud-synced folder:")
        print(bad.to_string(index=False))
    return df


def _open(path):
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
    """One row per .h5: which theta modifiers it carries, unit count, lifetime
    spread. The full tree's ten leaves only exist where the file degraded that
    component; others are physically 0. Unreadable files are reported and skipped."""
    rows = []
    for p in files(pattern):
        h = _open(p)
        if h is None:
            continue
        with h:
            th = [n.decode().strip() if isinstance(n, bytes) else str(n).strip()
                  for n in np.array(h["T_var"]).ravel()]
            comps = [c for c, pair in COMPONENTS.items() if set(pair) <= set(th)]
            out = {"file": os.path.basename(p), "theta": th,
                   "components": comps, "n_theta": len(th)}
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

def build_cache(pattern=PATTERN, cache=CACHE, force=False, **kw):
    """Reduce each .h5 to cycle-level cruise means and park it as parquet. Do them
    ONE AT A TIME -- a 3.7 GB file is several GB while being reduced; afterwards
    the whole fleet is a few thousand rows. A file that fails to open is reported
    and skipped; returns the list that failed."""
    os.makedirs(cache, exist_ok=True)
    broken = []
    for p in files(pattern):
        tag = os.path.basename(p).replace("N-CMAPSS_", "").replace(".h5", "")
        out = os.path.join(cache, f"{tag}.parquet")
        if os.path.exists(out) and not force:
            print(f"  skip {tag} (cached)")
            continue
        parts, failed = [], False
        for sp in ("dev", "test"):
            try:
                parts.append(cycle_features(p, split=sp, **kw).assign(split=sp))
            except OSError as e:
                print(f"  BROKEN {tag}: {e}")
                failed = True
                break
            except Exception as e:
                print(f"  {tag}/{sp}: {e}")
        if failed:
            broken.append(p)
            continue
        if parts:
            pd.concat(parts, ignore_index=True).assign(ds=tag).to_parquet(out)
            print(f"  {tag}: {sum(len(x) for x in parts)} cycle-rows -> {out}")
    if broken:
        print(f"\n{len(broken)} file(s) left out of the cache:\n  "
              + "\n  ".join(os.path.basename(b) for b in broken))
    return broken


# ==========================================================================
# 3. Pool -- globally unique unit ids, missing theta filled with 0
# ==========================================================================

def pooled(cache=CACHE, include=None, theta=THETA, fill_missing_theta=True):
    """Concatenate the cached files into one frame.

    Unit ids COLLIDE across files (each numbers from 1), so pooled `unit` becomes
    ds_index*1000 + local_unit; the originals stay in `unit_local` and `ds`.

    A file whose T_var lacks a modifier never degraded that component, so the
    modifier is physically 0 for all its units. fill_missing_theta writes that 0
    in -- VALID supervision: it teaches the leaf to output zero when the sensors
    look nominal, which is exactly what a healthy component should read."""
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
            print(f"  {tag}: {len(missing)} modifier(s) not degraded here -> theta = 0")
        f["unit_local"] = f["unit"]
        f["unit"] = 1000 * (i + 1) + f["unit"]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    print(f"pooled: {len(out)} cycle-rows, {out['unit'].nunique()} units, "
          f"{out['ds'].nunique()} datasets")
    return out


def coverage(frame):
    """Per dataset: which components actually degrade (theta not ~0 anywhere).
    The ablation uses this to know where each leaf is supervised vs. pinned to 0,
    and to read leave-one-dataset-out as leave-one-FAILURE-MODE-out."""
    rows = []
    for d, g in frame.groupby("ds"):
        active = [c for c, (e, fl) in COMPONENTS.items()
                  if (g[e].abs().max() > 1e-6 or g[fl].abs().max() > 1e-6)]
        rows.append({"ds": d, "units": g["unit"].nunique(),
                     "components": "+".join(active) or "(none)"})
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


# ==========================================================================
# 4. Splits -- always by unit
# ==========================================================================

def split(frame, fracs=(0.8, 0.1, 0.1), seed=0, stratify="ds"):
    """80/10/10 BY UNIT, stratified within each dataset so every fault mode is
    represented in all three sets. A unit is never split across sets. Small files
    are rounded UP to at least one val and one test unit each."""
    rng = np.random.default_rng(seed)
    keys = {"train": [], "val": [], "test": []}
    groups = frame.groupby(stratify) if stratify else [(None, frame)]
    for _, g in groups:
        u = rng.permutation(g["unit"].unique())
        n = len(u)
        if n < 3:
            keys["train"] += list(u)
            continue
        n_val = max(1, int(round(fracs[1] * n)))
        n_test = max(1, int(round(fracs[2] * n)))
        keys["test"] += list(u[:n_test])
        keys["val"] += list(u[n_test:n_test + n_val])
        keys["train"] += list(u[n_test + n_val:])
    out = tuple(frame[frame["unit"].isin(v)].reset_index(drop=True)
                for v in keys.values())
    assert not (set(keys["train"]) & set(keys["val"]))
    assert not (set(keys["train"]) & set(keys["test"]))
    assert not (set(keys["val"]) & set(keys["test"]))
    print("split by unit -- " + "   ".join(
        f"{k}: {len(v)} units / {len(f)} rows"
        for (k, v), f in zip(keys.items(), out)))
    return out


def leave_one_dataset_out(frame):
    """Yield (train, test, ds) with one whole FILE held out -- the leave-one-
    failure-mode-out generalization test."""
    for d in sorted(frame["ds"].unique()):
        m = frame["ds"] == d
        yield (frame[~m].reset_index(drop=True),
               frame[m].reset_index(drop=True), d)


# ==========================================================================
# 5. Self-test on a synthetic multi-file fleet (no real .h5 needed)
# ==========================================================================

def _build_synthetic_fleet(tmpdir):
    """Three synthetic files, each degrading a different component set, so pooling
    / theta-fill / coverage / splits can be exercised end to end."""
    from features import _make_synthetic_h5
    specs = [("DS01", ("HPT",)), ("DS04", ("fan",)), ("DS08", ("HPT", "LPT", "HPC"))]
    for tag, comps in specs:
        _make_synthetic_h5(os.path.join(tmpdir, f"N-CMAPSS_{tag}.h5"),
                           units=(1, 2, 3, 4), cycles=30, components=comps,
                           seed=hash(tag) % 1000)
    return specs


def _self_test():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gftree_data_")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        specs = _build_synthetic_fleet(tmp)
        print("probe:")
        print(probe().to_string(index=False))
        build_cache(min_cruise=10)
        feats = pooled()
        # every one of the ten modifiers must exist after pooling (filled with 0)
        assert all(t in feats.columns for t in THETA), "missing theta after pool"
        print("\ncoverage:")
        print(coverage(feats).to_string(index=False))
        tr, va, te = split(feats, seed=0)
        # by-unit disjointness already asserted inside split(); check LOSO count
        modes = [d for _, _, d in leave_one_dataset_out(feats)]
        assert len(modes) == len(specs), modes
        print("\nLOSO folds:", modes)
        print("\ndata self-test OK")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    _self_test()