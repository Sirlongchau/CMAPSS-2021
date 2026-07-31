# GFT / N-CMAPSS — State update

Covers the work since the last `PROJECT_STATE.md`: the confirmatory evaluation,
the neural baselines, the figure-driven diagnosis of a representational
degeneracy, the two model fixes that followed, and the decisions taken. Written
to be merged into `PROJECT_STATE.md`.

---

## 1. Where the project was (entry state)

- **Deployable model:** age-free, fixed-MF, monotone, **LP-specific** (`TREE_LP1`,
  69 params). Age-free is a hard requirement, not a preference — the intended use
  is an interpretable *damage* predictor (future FADEC coupling), and a clock
  tells a controller nothing actionable.
- **Reported result rested on one split.** The "tree lost on test" headline came
  from a single 80/10/10 by-unit split (one seed) in which one unit (8012) owned
  ~27% of the error, plus an arm-selection bug that shipped the wrong arm.
- **Diagnostic layer worked but modestly:** θ recovered at ρ ≈ 0.6, R² 0.3–0.5.
- **`learn_mf` had been dropped** on RMSE + variance grounds (fixed-MF beat full
  and had ~50× lower run-to-run variance).

## 2. Advancements this session

### 2.1 Statistical footing (`confirm.py`)
- Replaced the single split with a **repeated 12-split by-unit CV** of the
  pre-registered deployable model. GA-seed variance is negligible once MF-learning
  is off, so the compute went to split variance (the real source).
- **Result (pre-fix, anchored numbers in §2.4):** LP-specific RMSE
  **10.85 ± 0.99** vs age **12.65 ± 1.30**, beating age on **11/12** splits by RMSE.
  `full` 10.42 ± 1.12.
- **Wired the NASA-vs-age comparison** (the age baseline's NASA was already
  computed, just discarded). On the risk-aware metric the tree beat age on
  **12/12** splits (~30% NASA reduction) — a stronger and more decision-relevant
  result than the RMSE margin, because `c − age` over-predicts life on short-lived
  units (the *late*, dangerous direction NASA penalizes).
- **Unit 8012 closed.** Aggregated over the splits each unit appears in, 8012
  (~14.6) is not even the worst (8009 ~17.8, 7009 ~17.2). It looked pathological
  only because it was the lone tail unit in the old single test split. There is no
  dominating unit — a fat error tail clustered in the atypical-lifetime datasets
  (DS07/08/09), consistent with the thesis.

### 2.2 Neural baselines (`nn_baselines.py`)
Same-input protocol (identical residualization, by-unit splits, LOSO, cap, target;
MLP is memoryless, CNN/LSTM see a within-unit window, split by unit).
- **In-distribution CV:** MLP **9.37**, LSTM 9.95, 1D-CNN 10.11 (mean RMSE). The
  **memoryless MLP beats both temporal nets** — cruise-mean features already
  summarize condition, so temporal capacity buys nothing, and the memoryless FIS
  is not handicapped. The MLP also beats the GFT on RMSE; on NASA only the LSTM
  and GFT beat age on all 12 splits (MLP 11/12, CNN 9/12 — erratic).
- **Unseen failure mode (LOSO):** neural nets win the *easy* modes (DS01–03,05,06,
  MLP best), but the **GFT generalizes best on the hardest atypical-lifetime modes**
  (DS08c 13.96 vs LSTM 14.44 vs age 23.41; DS08a 10.60), and — unlike the MLP —
  **never fails catastrophically**. The in-distribution-champion MLP blows up on
  DS04 (RMSE 24.99, R² −0.13, NASA 44.9); the GFT degrades gracefully (14.43).

### 2.3 Figure-driven diagnosis
Visual inspection of the deployable model exposed a **representational degeneracy
invisible to RMSE/NASA**: the latent spool signals lived in **[0.48, 0.96]**, not
[0,1] — a healthy engine read ~0.48 "damage," the hp/lp maps were near-binary
switches, and RUL predictions were compressed into a **[~10, 57]** band (true range
[0,74]), under-predicting the healthy plateau and flooring at ~10 at end of life
(a systematic *late* bias). Root cause: the latent scale is a **gauge freedom** —
the RUL node can rescale hp/lp, so the GA never used [0,1], and fixed quantile MFs
on the spool inputs collapsed their resolution near θ ≈ 0.

### 2.4 Two model fixes (in `gft.py`)
- **Fix #1 — anchor spool nodes to [0,1]** (`_mono_decode`, `genome_bounds`,
  `build_tree`, `node_singletons`). Min-max normalize each latent spool grid so the
  healthy corner → 0 and max-damage → 1, removing the gauge freedom. Affine and
  increasing, so monotonicity/signs are preserved; the base gene becomes dead and
  is pinned to 0. New self-test invariant: anchored nodes span [0,1] exactly.
  - **Effect:** `hp` now spans [0,1] (baseline → 0). θ quality improved
    (HPT_eff R² 0.43 → 0.48, LPT_flow 0.30 → 0.33; `full`'s LPT_flow 0.07 → **0.18**
    — the latents now *mean* something). RUL metrics dipped slightly
    (LP-specific RMSE 10.85 → 11.56, NASA 1.51 → 1.70): expected, because anchoring
    removes a degree of freedom the GA used to compress predictions and hedge error,
    and because it *exposed* the resolution problem instead of hiding it.
- **Fix #2 — place spool-input centres on the degrading rows** (`build_tree`).
  Global quantiles collapse near 0 (θ ≈ 0 for most of life), so the anchored ramp
  crammed into a sliver and even healthy units read high (the observed `lp` floor
  at ~0.6). Conditioning on `since_onset > 0` spreads the centres across the actual
  damage range. Self-test confirms centres now span ~85% of the θ range.
  - **Effect:** `lp` now spans the full [0,1] and grades rather than floors.
- **MF resolution helper — `refine(tree, k_spool, k_root)`.** Raises term counts
  only on the cheap 1–2 input spool/root nodes (leaves stay at 3³). `TREE_LP1`
  69 → **89 params**. Grades the ramp and steepens the RUL surface near (1,1);
  needs held-out validation before adoption.

### 2.5 The RUL floor (RUL(1,1) ≈ 9)
Diagnosed as **E[RUL | both spools saturated]**, not an artifact to enforce away.
The spools saturate *early* (15–40 cycles before EOL), so the (1,1) corner spans a
long RUL window and the FIS outputs the loss-minimizing mean (~9). Hard-pinning
RUL(1,1)=0 would make the model *early* (over-conservative) on most of that window
to be right on the few true-EOL rows — a real RMSE cost for a memoryless model. The
honest cure is a variable that keeps moving after damage saturates (a temporal
input), which is a thesis-level decision (it reintroduces a clock). **Left as-is.**

## 3. Decisions taken (and why)

1. **Deploy LP-specific, report `full` alongside.** `full` edges RUL RMSE
   (post-anchor gap ~0.9 cyc) but LP-specific carries the cleaner LP diagnostic
   (LPT_flow R² 0.33 vs 0.18). For a component-attributable damage predictor,
   spool-specificity wins the tie.
2. **Keep the model age-free.** Confirmed by the deployment target (interpretable
   damage for control). Age remains a baseline and an ablation probe (`TREE_AGE`),
   never the shipped model.
3. **Adopt fixes #1 and #2 despite the small RMSE dip.** Reverting to recover the
   metric would re-optimize the error into a physically meaningless representation —
   the exact trap the figures caught. A meaningful [0,1] damage signal is worth
   ~0.7 cycles for an interpretability paper.
4. **Do not hard-enforce the RUL floor** (see §2.5).
5. **Reframe the paper.** NASA-first (12/12, the risk metric), RMSE second; LOSO =
   *leave-one-failure-mode-out* as the headline (the datasets are the fault modes,
   stratified CV keeps every mode in training so CV is in-distribution only); neural
   baselines as same-domain context, framed as "matches deep nets to ~1 cycle at
   1–2% of the params, generalizes better to the hardest unseen modes, never fails
   catastrophically, and is interpretable" — not "beats deep nets."

## 4. Next steps

1. **Re-fit with fixes #1 + #2 and re-run `confirm.py`.** Expectation: the NASA/RMSE
   lost to anchoring recovers (the graded ramp removes the abrupt-saturation late
   errors) — target RMSE ≤ 10.85, NASA ≤ 1.51 for LP-specific — **with** clean
   figures. Confirm `lp_h` → 0 at healthy and the RUL parity band opening up.
2. **A/B `refine(TREE_LP1, 5, 5)`** on held-out units. Adopt only if the +20 params
   buy held-out accuracy; report as an ablation either way.
3. **Deployable-arm LOSO** (`loso_deployable.py`, to be written): LP-specific /
   fixed-MF / anchored across the nine modes, replacing the stale full-tree LOSO
   column in the article (Tables 2 and 4).
4. **NN baselines over 3 seeds** for LOSO so the DS04 MLP blow-up is defensible as a
   robust effect, not a single-seed fluke.
5. **Regenerate all figures** from the fixed model; swap the article's figure
   placeholders for real `figures/final/*`.
6. **Finish the article:** fill the NN table, refresh the LOSO table, insert real
   citations (still `\todo`), authors/date.
7. **Update `PROJECT_STATE.md` §8** with the two resolved gotchas: latent-scale
   gauge freedom (anchoring) and spool-input MF collapse (degrading-conditioned
   centres).

### Deferred / thesis-level
- **Temporal input for the RUL floor.** Adding `since_onset` (or a rate feature) to
  the RUL node would let it resolve the saturated window and drop the floor toward
  0 — but it reintroduces a clock-like signal and must be weighed against the
  age-free contribution. Decide deliberately; do not slip it in for a metric gain.
- **Leaf-side scale compression.** If `lp_h` still floors above ~0.2 after re-fit,
  the residual cause is the `lpt_flow` leaf under-reading toward 0 for healthy
  units — a separate, smaller leaf-calibration issue.