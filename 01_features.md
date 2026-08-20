# 01 — features.py (preprocessing)

Purpose: turn one N-CMAPSS `.h5` split into one row per `(unit, cycle)`, carrying
every channel the full tree can reference.

## Decisions

**All ten modifiers, load-what's-present.** The old code fixed four modifiers
(HPT/LPT). The full tree needs all five components, so `THETA` now lists all ten
(`fan/LPC/HPC/HPT/LPT × eff/flow`). `cycle_features` still only keeps the
modifiers a given file actually contains — real N-CMAPSS files carry only the
modifiers they degraded — and the pooling layer fills the rest with zero. This
keeps `features.py` ignorant of the fleet: it reports one file faithfully.

**Physics metadata lives here, once.** `COMPONENTS` (component → its eff/flow
modifier names) and `SHAFT` (component → LP/HP shaft) are declared in this module
and imported everywhere else. The tree groups leaves into spools by `SHAFT`; the
data layer fills missing theta from `THETA`. Putting these in one place means a
naming change (or a sixth component) is a one-line edit, not a hunt.

**Real sensors only.** `X_v` (virtual sensors) is still never loaded — a hard
guarantee that nothing downstream can read a virtual channel.

**Cruise-mean aggregation, unchanged.** theta is constant within a cycle at 1 Hz,
so a cycle-level cruise mean loses no health information while collapsing 120
timestamps to one row. The high-altitude filter (`alt ≥ cruise_frac·max`) removes
climb/descent transients before averaging.

**age / since_onset, unchanged.** `age` = 1-indexed per-unit cycle count.
`since_onset` = cycles since `hs` flipped 0→1, clipped at 0. These are the two
temporal quantities the tree is *allowed* to see only as the age baseline / cap
anchor — never as a tree input (the age-free requirement).

## Interface

- `cycle_features(path, split, cruise_frac=0.95, min_cruise=30) -> DataFrame`
- `load_raw(path, split) -> DataFrame` (per-timestamp, real sensors + present theta)
- constants: `SENSORS`, `CONDITIONS`, `COMPONENTS`, `SHAFT`, `THETA`
- helpers: `components_on(shaft)`, `theta_of(component)`
- `_make_synthetic_h5(..., components=(...))` — flexible synthetic generator; each
  synthetic "file" degrades only the named components and lists only their
  modifiers in `T_var`, mirroring the real files. Reused by `data.py`'s tests.

## Test evidence

`python features.py` on a synthetic file degrading `HPT+LPT+fan`:
- exactly the six expected modifier columns present, none of the other four;
- `age ≥ 1` everywhere, `since_onset ≥ 0`, and onset fires (max > 0);
- `components_on("LP") = [fan, LPC, LPT]`, `components_on("HP") = [HPC, HPT]`.
