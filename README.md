# Genetic Fuzzy Tree for N-CMAPSS RUL

A nested genetic fuzzy tree that predicts remaining useful life from N-CMAPSS
turbofan data, with physically named intermediate nodes: each leaf diagnoses one
engine health parameter, each spool node reports shaft damage.

## Start here

| file | read it for |
|---|---|
| **`PROJECT_STATE.md`** | **Read this first.** Full project context: every design decision and the evidence that forced it, current results, open problems, dead ends already ruled out, and known gotchas. Written so a new session can pick the project up with no other context. |
| `gft_documentation.pdf` | Formal code documentation: the maths, the encodings, and a per-function API reference. Source in `gft_documentation.tex`. |

## Code

| file | role |
|---|---|
| `gft.py` | The model. FIS, learnable Ruspini antecedents, structurally monotone consequents, tree definitions, GA, loss, fit/predict, held-out evaluation and baselines. Self-test: `python gft.py` |
| `datasets.py` | Verify / probe / cache / pool the `.h5` fleet; unit-wise splits; leave-one-dataset-out. `python datasets.py` |
| `ncmapss_features.py` | Raw `.h5` → cycle-level cruise means. Unmodified from the original project. |
| `leaf_search.py` | Per-leaf sensor-input ablation, with seed robustness and cross-spool specificity confirmation. |
| `gft_analysis.py` | All figures. Self-test: `python gft_analysis.py` |
| `run.py` | The protocol: pool → split by unit → validate arms → test the winning arm once → leave-one-dataset-out → figures. |

## Running

```bash
python datasets.py     # once: verify files, probe fault modes, build the parquet cache
python run.py          # the full protocol
```

Requires `numpy`, `pandas`, `matplotlib`, `h5py`, `pyarrow`.

Put the `N-CMAPSS_DS*.h5` files in the working directory — **not** in a
cloud-synced folder (see `PROJECT_STATE.md` §8).

## Invariants

`python gft.py` must print all three:

```
100 random genomes: centres always ordered, max|sum(mu)-1| = 0.0e+00
GA seed reproduces the quantile centres: max err = 4.6e-16
200 random genomes: monotone nodes never violate their sign (OK)
```

If any fails, the fuzzy partition, the GA seeding, or the monotonicity encoding is
broken — fix that before trusting any result.
