# Genetic Fuzzy Tree for N-CMAPSS RUL — Project State

**Purpose of this document:** hand the whole project to a new session with no loss of
context. It records what the model is, every design decision *and the evidence that
forced it*, what is already ruled out, what is still open, and how to run everything.

---

## 1. What this is

A **nested Genetic Fuzzy Tree (GFT)** that predicts remaining useful life (RUL) for
N-CMAPSS turbofan data, trained end-to-end by a real-coded genetic algorithm.

The scientific pitch is **not** "beat an LSTM on RMSE". It is:

> A prognostic model whose intermediate nodes are *physically named and readable*
> — each leaf diagnoses one engine health parameter, each spool node reports shaft
> damage — that generalises to **unseen failure modes** better than a lifetime prior.

Everything in the codebase is subordinate to that claim. Accuracy that costs
interpretability has repeatedly been rejected on purpose.

### Model form

Each node is a **zero-order Takagi–Sugeno FIS** on a full rule grid, with a
**Ruspini partition** (triangles summing to 1). With the product t-norm this makes
firing strengths sum to exactly 1, so the output is a plain dot product with the
rule singletons:

```
y(x) = Σ_r f_r(x) · c_r ,   Σ_r f_r(x) = 1
```

The genome carries **both halves of the fuzzy system**: the antecedents (where the
membership functions sit) and the consequents (one singleton per rule).

---

## 2. Current architecture

```
sensors ─→ residualize (per unit) ─→ EWMA smooth
              │
              ├─ hpt_eff  (T48, P40, Nc)  ─→ HPT_eff_mod   ─→ [ hp ]  ─┐
              ├─ lpt_flow (T50, P50, P24) ─→ LPT_flow_mod ─┐           │
              └─ lpt_eff  (T50, P50, Nf)  ─→ LPT_eff_mod  ─┴→ [ lp ] ─┴→ RUL(hp, lp)
```

* **Leaves** are supervised on the θ health modifiers.
* **Spool nodes** `hp`, `lp` are latent damage in [0,1], no target of their own.
  `hp` is a **single-input node** (a monotone 1-D calibration) — legitimate and
  intentional, see §4.4.
* **Root** `RUL(hp, lp)` — `age` is deliberately **absent** (§4.2).

Three tree variants live in `gft.py`:

| constant | params | role |
|---|---|---|
| `TREE` | 158 | default |
| `TREE_LP1` | 109 | drops the coupled `lpt_eff`; fully spool-specific |
| `TREE_AGE` | 180 | ablation control — restores `age` at the root |

`HPT_flow_mod` **does not appear**. It was removed on evidence (§4.5).

---

## 3. Latest results (pooled fleet, 9 datasets, 99 units, 7473 cycle-rows)

Split by **unit**, 80/10/10 stratified by dataset: train 77 / val 11 / test 11 units.
`rul_cap = 74` cycles.

### Validation (mean ± sd over 3 seeds)

| model | RMSE | sd | R² | note |
|---|---|---|---|---|
| GFT +age | 8.135 | 0.158 | 0.858 | *ablation control, not reportable* |
| GFT free-grid | 9.082 | 0.602 | 0.823 | *monotone ablation* |
| **GFT fixed-MF** | **9.340** | **0.024** | 0.813 | most stable deployable arm |
| GFT LP-specific | 9.495 | 0.193 | 0.807 | 109 params |
| GFT full | 9.622 | 1.263 | 0.799 | high variance |
| age only (`RUL = c − age`) | 10.649 | — | 0.757 | zero sensors, 1 param |
| mean RUL | 21.684 | — | −0.008 | |

### Held-out θ quality — **the big win of this run**

| leaf | R² | ρ (per-unit) |
|---|---|---|
| HPT_eff_mod | **0.417** | 0.623 |
| LPT_flow_mod | **0.400** | 0.618 |
| LPT_eff_mod | **0.583** | 0.600 |

**All three leaves are now positive R².** In the previous run three of four were
*negative* (HPT_eff −0.39, HPT_flow −1.46, LPT_flow −0.86). The combination of loss
normalisation + correlation term + dropping HPT_flow fixed the diagnostic layer.

### Test (touched once)

| model | RMSE | R² |
|---|---|---|
| age only | 10.854 | 0.760 |
| GFT full | 11.355 | 0.737 |

**The tree lost on test.** Two caveats, both material:

1. **Arm-selection bug (now fixed).** `run.py` hardcoded the `full` arm for test —
   the *worst and least stable* val arm (sd 1.263). `fixed-MF` (9.340 ± 0.024) should
   have been selected. Fixed: the test model is now the winning *deployable* val arm.
2. **Unit 8012 dominates.** RMSE 22.38, NASA **15.16** (next worst 2.84). It is
   **7.1 % of test rows but 27.4 % of squared error**. Excluding it, test RMSE is
   **10.03** vs the age baseline's 10.85 — i.e. the tree wins. This single unit
   decides the headline number and has not been investigated.

### Leave-one-dataset-out (unseen failure mode)

Beats `age only` on **6 / 9**:

| held out | GFT | age only | |
|---|---|---|---|
| DS08c-008 | 13.96 | 23.19 | **−9.2** |
| DS08a-009 | 10.60 | 15.13 | −4.5 |
| DS01-005 | 11.89 | 14.14 | −2.3 |
| DS02-006 | 8.21 | 9.72 | −1.5 |
| DS05 | 10.35 | 11.89 | −1.5 |
| DS03-012 | 9.58 | 10.25 | −0.7 |
| DS04 | 14.43 | 13.10 | +1.3 |
| DS06 | 10.29 | 9.87 | +0.4 |
| DS07 | 9.77 | 8.34 | +1.4 |

Stable pattern across every run: **the tree wins hardest exactly where `age` fails**
(DS08c/DS08a — units with atypical lifetimes) and loses mildly where degradation is
near-linear in cycle (DS04, DS07). That asymmetry *is* the result.

---

## 4. Design decisions and the evidence that forced them

Do not re-open these without new evidence. Each cost a full experimental cycle.

### 4.1 Everything is scored on held-out **units**

Rows are `(unit, cycle)` cruise means. Unit 4 cycle 31 and cycle 32 are the same
engine at the same damage state. A random **row** split measures interpolation, not
prediction. Every split is by unit (`split_units`, `datasets.split`, `loo_cv`), with
asserts that no unit appears in two sets.

*Origin:* the original code fit and predicted on the same frame — every number in the
first version was training error.

### 4.2 `age` is removed from the root; the RUL target is capped

With few units of similar lifetime, `RUL = c − age` is learnable without touching a
sensor, and the GA takes that deal. Two counters:

* **`age` ablated** from the root (`TREE_AGE` restores it, to *measure* the shortcut).
* **`rul_cap`** = median RUL at degradation onset (from the `hs` flag via
  `since_onset`). Before onset, RUL is set by the unit's total lifetime and is not
  knowable from sensors; an uncapped target forces the loss to fit an unlearnable
  region and `age` is the only feature that can.

*Current gap:* `+age` still beats the age-free model by ~1.5 cycles RMSE. Report both;
the gap quantifies how much of prognosis here is condition-monitoring vs a clock.

### 4.3 The loss is normalised by trivial baselines, plus a correlation term

```
loss = NASA(RUL)/NASA(age-only)
     + w_th · mean_leaf[ RMSE(θ)/std(θ_leaf) ]
     + w_tr · mean_leaf[ 1 − mean_unit_corr(θ̂, θ) ]
```

* NASA normalised by the age baseline ⇒ a value < 1 literally means "beats the
  zero-sensor baseline".
* Each leaf divided by **its own** trivial error (`std(θ)`), not by range. The old
  `RMSE/range` made low-variance leaves explode and drown the informative ones.
* The **correlation term** rewards *shape*. Runs consistently showed leaves with
  ρ ≈ 0.5–0.6 but negative R² — they tracked degradation but got the scale wrong, and
  RMSE punished that while giving shape no credit.

Defaults `w_th = 1.0`, `w_tr = 1.0`. Because every term is now "fraction of a trivial
baseline", the weights are interpretable (the old magic `theta_weight=5` was just
where two unnormalised numbers happened to coincide).

### 4.4 Membership functions are learned, via a gap encoding

Under a Ruspini partition the fuzzy sets of a variable **are** its ordered centres.
Fixing them at data quantiles was pathological: θ ≈ 0 for most of a unit's life, so
the median sits at ≈ 0, two of three centres collapse into the healthy plateau, and
the entire degraded regime gets one triangle's ramp.

Encoding: each input contributes **k+1 non-negative gap genes**, normalised and
accumulated inside a padded domain:

```
c_i = d_lo + (d_hi − d_lo) · Σ_{j≤i} w_j ,    w = g / Σg
```

* Ordering `c_1 < … < c_k` holds for **any** gene values ⇒ no repair operator, no
  penalty, no invalid genome possible.
* Partition stays Ruspini exactly (self-test: `max|Σμ − 1| = 0`).
* The GA is **seeded at the quantile placement** (self-test: reproduces it to 1e-16),
  so learning MFs cannot *start* worse.

*Status:* MF learning has **never shown a reliable held-out gain**. In the latest run
`fixed-MF` (9.340 ± 0.024) beat `full` (9.622 ± 1.263) and was ~50× more stable.
Treat the 56 antecedent genes as, at best, neutral. See §6.

### 4.5 Structural monotonicity on `hp`, `lp`, `RUL`

A Sugeno grid is a generic approximator, so nothing stops a folded, non-monotone
surface — and on noisy data the GA buys folds. For the damage/RUL nodes that is
unphysical: damage cannot fall with more damage.

Encoding: the rule slice stores `[base | non-negative steps]`; the steps are
cumulatively summed along each axis with per-axis signs. `MONOTONE = {"hp": -1,
"lp": -1, "RUL": -1}` — a **scalar broadcasts over any arity**, so the same spec works
for the 2-input `TREE` root and the 3-input `TREE_AGE` root.

**Leaves are deliberately NOT constrained** — their sign depends on the residual
cleanly isolating the health effect, which is not guaranteed.

*Status:* an earlier run showed a clean win (8.87 vs 9.20, all 3 seeds, plus much
lower variance). The latest run shows `free-grid` (9.082 ± 0.602) ≈ `full`
(9.622 ± 1.263) — within noise, no longer a demonstrated gain. The loss and the tree
both changed in between, so the earlier result does not transfer. **Keep monotonicity
for the interpretability claim; do not currently claim it improves accuracy.**

### 4.6 Leaf inputs are chosen by exhaustive search, not judgement

`leaf_search.py` fits each leaf standalone (`sensors → θ`), scores it on held-out
units, and ranks. Two traps it avoids by construction: it scores on the leaf's **own
θ** (never downstream RUL, which would reward the age-correlated path), and fits
**standalone** (not inside the tree), isolating observability from everything else.

Then `--confirm` adds **seed robustness** (mean ± sd over N seeds) and **cross-spool
specificity**:

```
specificity = ρ(own target) − ρ(same inputs → opposite spool's target)
```

Near zero ⇒ the combo reads *global damage*, not that component.

Results over all 286 three-sensor combinations of the 13 measured sensors:

| leaf | ρ | R² | specificity | verdict |
|---|---|---|---|---|
| HPT_eff (T48,P40,Nc) | 0.606 | 0.646 | +0.071 | **keep** — strongest, most stable (sd 0.035) |
| LPT_flow (T50,P50,P24) | 0.588 | 0.442 | +0.376* | **keep** — best ρ and R² of any candidate |
| LPT_eff (T50,P50,Nf) | 0.549 | 0.451 | −0.071 | predictive but **not spool-specific** |
| HPT_flow | 0.250 | 0.126 | −0.340 | **dropped** |

\* **Caveat:** the flow-pair specificity is inflated. Its foil (HPT_flow) is itself
unpredictable, so `foil_rho` is automatically low for any LPT_flow combo. Judge
LPT_flow on ρ/R², not on that number. The **eff**-pair specificity is valid (both
targets reach ρ ≈ 0.6).

Also found: **T48 appears in ~80–100 % of every target's top-10** — it is a global
damage proxy, which is why ρ-ranking alone is misleading and the specificity check
was necessary.

### 4.7 Per-unit condition residual

For each unit, a quadratic condition model is fitted on that engine's **own** first
`ref_cycles` cycles and subtracted from all its cycles:

```
r_u(t) = x_s(t) − [1, w, w²] β_u
```

So "T48 residual" means *hotter than **this** engine ran when new* — a damage signal.
The previous fleet-wide baseline meant *hotter than the average engine ran when new*
= damage + manufacturing scatter + flight-class bias, and pooling 10 datasets
multiplies that error.

No leakage: uses only early-life **sensor** data from the engine itself, touches
neither RUL nor `hs`, and is stateless — recomputed identically on held-out units, so
fit and predict cannot drift apart.

### 4.8 Pooling the fleet

`datasets.py` pools all files into one frame. This is not just "more data":

* **It destroys the age shortcut.** Lifetimes vary across datasets, so `age only`
  collapses and the tree is forced onto the sensors.
* **Parameter ratio** goes from ~1.5 rows/gene to ~47.
* **It supplies a validation split**, which is what makes tuning `w_th`/`w_tr`
  legitimate at all.

Unit IDs collide across files (each `.h5` numbers from 1) and are remapped to
`1000·(file_index+1) + local_unit`; originals kept in `unit_local`/`ds`.

---

## 5. Files

| file | role |
|---|---|
| `gft.py` | model: FIS, MF gap-encoding, monotone consequents, tree, GA, loss, fit/predict, held-out evaluation + baselines. Self-test: `python gft.py` |
| `datasets.py` | verify / probe / cache / pool the `.h5` fleet; unit-wise splits; LOSO iterator. `python datasets.py` |
| `ncmapss_features.py` | raw `.h5` → cycle-level cruise means (**unchanged from original**) |
| `gft_analysis.py` | all figures: RUL, θ, latent nodes, parity, fitness, **memberships**, control surfaces. Self-test: `python gft_analysis.py` |
| `leaf_search.py` | per-leaf input ablation + `--confirm` (seed robustness + cross-spool specificity) |
| `run.py` | the protocol: pool → split → validate arms → test winning arm once → LOSO → figures |
| `simplified_tree.tex` | LaTeX code documentation (**stale** — see §6) |

### Running

```bash
python datasets.py                      # once: verify, probe fault modes, build cache
python run.py                           # the full protocol
python leaf_search.py --all-sensors     # re-derive leaf inputs (slow)
python leaf_search.py --confirm --by-specificity --topn 5
```

---

## 6. Open problems, ranked

1. **Unit 8012.** NASA 15.16, 27 % of test squared error from 7 % of rows. Plot its θ
   trajectories and RUL. Either it fails in a mode nothing else in the fleet shares,
   or its labels are wrong. Until known, report pooled metrics with *and* without it.
2. **Re-run with the arm-selection fix.** The reported test loss (11.36 vs 10.85) used
   the worst val arm. `fixed-MF` should have gone to test. This may flip the headline.
3. **MF learning earns nothing.** `fixed-MF` beat `full` on mean *and* had 1/50th the
   variance. Either drop `learn_mf` (−56 genes, simpler model, better story) or find
   why it destabilises. Do not keep it on the assumption it helps.
4. **`age` still worth ~1.5 cycles.** The sensor path has not closed the gap.
5. **GA plateaus and is under-searched at the end** (still descending at gen 400,
   σ floored at 0.05). *Already tested:* 1000 gens + σ_min 0.12 made things **worse**
   — see §7. The optimiser is not the bottleneck; do not spend more budget here.
6. **`simplified_tree.tex` is stale.** It documents the old 4-leaf tree, the old
   loss, and does not mention monotonicity, the leaf search, or the current
   architecture. Regenerate before submission.
7. **Memory-NN for HPT_flow.** The only remaining route to that leaf (§7). The
   agreed design is **Option B**: the network is an adaptive *front-end* producing a
   cleaner sensor state for an unchanged fuzzy tree — *not* a competing θ predictor.
   Admission criterion agreed: the NN's correction `δ_t` must be **bounded**
   (`|δ| ≤ kσ` on top of a transparent EWMA baseline) and demonstrably **noise-like**
   (white, zero-mean innovation sequence). If `δ_t` develops structure, the NN is
   doing state inference, not denoising, and the interpretability claim fails.

---

## 7. Already ruled out — do not redo

* **More GA budget.** 1000 generations at σ_min 0.12 gave *worse* held-out results
  than 400 at σ_min 0.05 (test RMSE 10.81 vs 9.88; every leaf's R² degraded). A more
  aggressive optimiser gets better at exploiting the easy RUL term, not at the hard θ
  term. **The optimiser was never the bottleneck.**
* **Better inputs for HPT_flow.** Exhaustive: all 286 combinations, ceiling ρ = 0.27
  (every other modifier reaches ~0.63), negative R² throughout, specificity −0.34
  (candidates predicted the *other* spool better). Not observable from cruise-mean
  sensors by a memoryless FIS.
* **Rescuing LPT_eff's specificity.** A specificity-first prescreen over the *whole*
  search space found a best of **+0.018** — noise, with R² 0.067. LPT efficiency is
  not separately observable in the pooled data.
* **Ranking leaf inputs by ρ alone.** T48 dominates every leaderboard as a global
  damage proxy; ρ-ranking picks leaves that read the wrong spool.
* **Virtual sensors (`X_v`) as leaf inputs.** They are outputs of the same simulator
  that generated the θ targets (leakage risk) and do not exist on a real engine
  (breaks the deployability claim). Legitimate alternative: **Type-2 reconstructions**
  built from measured sensors only (ratios, corrected parameters, temperature
  spreads) — not yet tried, and the right next move if more leaf signal is needed.

---

## 8. Gotchas and fixed bugs

* **`is_baseline` compared joined strings.** `itertools.combinations` emits sensors in
  *pool* order, so `["T48","P40","Nc"]` came out `"Nc+T48+P40"` and never matched —
  every `--all-sensors` run had `is_baseline` silently all-False. Now compares sets.
* **Monotone grids escaped their output range.** `base` is the grid **minimum** (the
  corner where the cumsum is zero), so the grid spans `[base, base+total]` — a box
  constraint cannot bound `base + total`. Spool grids ran past 1 and saturated against
  the `[0,1]` clamp, wasting the node's whole dynamic range. Fixed by clipping in
  `_mono_decode` (clipping is a monotone map, so monotonicity is preserved). Verified:
  500 random genomes, zero overshoot, zero violations.
* **Test-arm hardcoding.** `run.py` sent a fixed config to test regardless of which arm
  won validation. Fixed; arms are now tagged `selectable` so the ablation controls
  (`free-grid`, `+age`) can win on RMSE without being reported.
* **`MONOTONE` arity.** Use a **scalar** (`-1`) so it broadcasts; a tuple must match
  the node's input count and breaks when swapping `TREE` ↔ `TREE_AGE`.
* **`analyze()` needs the capped frame.** It compares `frame["RUL"]` to `RUL_hat`; pass
  `frame.assign(RUL=gft.cap(frame["RUL"], rul_cap))` or you plot a capped prediction
  against an uncapped truth.
* **Keep the `.h5` files out of OneDrive.** One file arrived truncated (32 bytes short)
  because of sync-on-demand. `datasets.verify()` now detects and reports this;
  `probe`/`build_cache` skip broken files instead of crashing.
* **Self-test invariants** (run `python gft.py`) — all three must pass:
  `centres always ordered, max|Σμ−1| = 0` · `GA seed reproduces quantile centres to
  ~1e-16` · `monotone nodes never violate their sign`.

---

## 9. The honest one-paragraph summary

The tree is a competent RUL regressor (val R² ≈ 0.81) whose **diagnostic layer now
works** — all three leaves have positive held-out R² and per-unit ρ ≈ 0.6, after four
rounds of fixing the loss, the residual, and the leaf inputs. It generalises to unseen
fault modes better than a lifetime prior on 6 of 9 datasets, winning hardest exactly
where that prior fails. It does **not** yet cleanly beat `RUL = c − age` on the pooled
test split, and the two reasons are known and addressable: a fixed arm-selection bug,
and one pathological unit carrying a quarter of the error. The interpretability
machinery — learnable Ruspini antecedents, structurally monotone damage nodes,
evidence-selected leaf inputs — is complete, verified by invariant tests, and is the
contribution.
