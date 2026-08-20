# 02 — data.py (pooling, splits, LOSO)

Purpose: assemble the ten cached files into one leak-proof frame and provide the
two evaluation protocols the ablation reports on.

## Decisions

**Pool, and remap units.** Unit ids collide across files (each numbers from 1), so
pooled `unit = 1000·file_index + local_unit`, with `unit_local`/`ds` retained.
Pooling is what kills the `RUL = c − age` shortcut: one file's near-uniform
lifetimes make it learnable without sensors; ten files' varied lifetimes and
flight classes collapse it.

**Fill missing theta with zero — this is supervision, not padding.** A file that
never degraded a component genuinely had that modifier at ~0 for all its units.
Writing 0 in teaches the corresponding leaf to output zero when its sensors look
nominal, which is the correct healthy reading. `fill_missing_theta=False` keeps
only all-ten files if a stricter experiment ever wants that.

**Split by unit, never by row.** Two cycles of one engine are the same machine at
neighbouring damage states; splitting them across train/test measures
interpolation, not prediction. The unit is the independent sample. `split`
stratifies by `ds` so every fault mode appears in train/val/test, and rounds small
files up to at least one val and one test unit.

**Leave-one-dataset-out is first-class.** Each file is (broadly) one failure mode,
so `leave_one_dataset_out` is the leave-one-failure-mode-out generalization test —
the headline question for a component-named tree. It sits here as a primary API,
not a diagnostic afterthought.

**New: `coverage(frame)`.** Reports, per dataset, which components actually
degrade (theta not ~0 anywhere). The ablation needs this to know where each leaf
is supervised versus pinned to zero, and to label each LOSO fold with the mode it
holds out.

## Interface

- `verify()`, `probe()` — integrity + fault-mode inventory before any run.
- `build_cache()` — one `.h5` at a time → `cache/*.parquet`.
- `pooled(...) -> DataFrame` — the fleet, unique units, theta filled.
- `coverage(frame) -> DataFrame` — component activity per dataset.
- `split(frame, fracs, seed, stratify)` — 80/10/10 by unit, stratified.
- `leave_one_dataset_out(frame)` — yields `(train, test, ds)` per held-out mode.

## Test evidence

`python data.py` builds a synthetic 3-file fleet (`DS01=HPT`, `DS04=fan`,
`DS08=HPC+HPT+LPT`) and checks end to end:
- `probe` reads each file's modifier set and maps it to components;
- after `pooled()` all ten modifiers exist (absent ones filled with 0);
- `coverage` reports `DS01→HPT`, `DS04→fan`, `DS08→HPC+HPT+LPT`;
- by-unit split is disjoint across train/val/test (asserted);
- LOSO yields exactly one fold per file.

The benign `DS0x/test: need at least one array to concatenate` line is expected:
the synthetic generator writes only a `dev` split, and `build_cache` skips the
absent `test` split gracefully. Real files carry both.
