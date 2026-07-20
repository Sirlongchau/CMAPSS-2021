"""
run.py -- the honest protocol, on the pooled fleet.

  1. POOL.       All ten .h5 files -> one cycle-level frame (datasets.py).
                 Ten datasets of varying lifetime kill the `RUL = c - age`
                 shortcut that six near-identical units made irresistible.
  2. SPLIT BY UNIT.  80/10/10, stratified by dataset. A random ROW split would
                 put unit 4 cycle 31 in train and cycle 32 in test -- the same
                 engine at the same damage state. That is leakage, not a score.
  3. TUNE ON VAL, TOUCH TEST ONCE.  theta_weight / n_terms / gens are chosen on
                 val. `age only` is printed next to everything: if the tree
                 cannot beat one parameter and zero sensors, it is decoration.
  4. ABLATE.     age in the root (TREE_AGE), and learned vs fixed membership
                 functions (learn_mf), so every claim has its control.

    python datasets.py     # probe + build the cache (once, slow)
    python run.py
"""

import numpy as np
import pandas as pd

import gft
import datasets as ds
from gft_analysis import analyze

GENS, POP, SEEDS = 400, 120, (0, 1, 2)
THETA_W, SMOOTH = 5.0, 7

pd.set_option("display.width", 200, "display.max_columns", 40)


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


feats = ds.pooled()
train, val, test = ds.split(feats, fracs=(0.8, 0.1, 0.1), seed=0)

RUL_CAP = gft.suggest_rul_cap(train)          # from TRAIN only -- never from test
FIT = dict(gens=GENS, pop=POP, theta_weight=THETA_W, smooth_span=SMOOTH,
           rul_cap=RUL_CAP)
banner(f"rul_cap = {RUL_CAP:.0f} cycles (RUL at degradation onset, train units)")

# ---- model selection: everything below is scored on VAL -------------------
banner("VALIDATION  (mean +/- sd over seeds)")
rows = [gft.baselines(train, val, RUL_CAP)]
#  name                        tree          learn_mf  monotone
arms = [("GFT full",           gft.TREE,     True,     True),
        ("GFT free-grid",      gft.TREE,     True,     False),   # monotone ablation
        ("GFT fixed-MF",       gft.TREE,     False,    True),    # antecedent ablation
        ("GFT +age",           gft.TREE_AGE, True,     True)]    # age ablation
for name, tree, mf, mono in arms:
    for s in SEEDS:
        m = gft.fit_gft(train, tree, learn_mf=mf, monotone=mono, seed=s,
                        verbose=False, **FIT)
        rows.append(pd.DataFrame([gft.evaluate(val, m, name)]))
        print(f"  {name:16s} seed {s}: RMSE={rows[-1]['RMSE'][0]:.2f}")
res = pd.concat(rows, ignore_index=True)
print(gft.summarize(res).to_string())
print("""
  full vs 'age only'   -- do the sensors and the fuzzy tree contribute anything?
  full vs free-grid    -- what the MONOTONICITY constraint bought. free-grid lets
                          hp/lp/RUL fold; if full is >= free-grid on VAL, the
                          physical prior is free accuracy AND a readable surface.
  full vs fixed-MF     -- what tuning the ANTECEDENTS bought.
  +age vs full         -- how much of the score is still the age shortcut.""")

tcols = [c for c in res.columns if ":" in c]
banner("held-out theta quality (R2 and per-unit correlation)")
print(res[res["model"] == "GFT full"][tcols].mean().round(3).to_string())

# ---- final: refit on train+val, score ONCE on test ------------------------
banner("TEST  (fit on train+val, scored once)")
full = pd.concat([train, val], ignore_index=True)
model = gft.fit_gft(full, gft.TREE, learn_mf=True, monotone=True, seed=0, **FIT)
final = pd.concat([gft.baselines(full, test, RUL_CAP),
                   pd.DataFrame([gft.evaluate(test, model, "GFT full")])],
                  ignore_index=True)
print(final[["model", "n_params", "RMSE", "MAE", "NASA", "R2"]].round(3)
      .to_string(index=False))
gft.inspect(model)

# ---- can it read a fault mode it has never seen? -------------------------
banner("LEAVE-ONE-DATASET-OUT  (unseen failure mode)")
loso = []
for tr, te, name in ds.leave_one_dataset_out(feats):
    m = gft.fit_gft(tr, gft.TREE, learn_mf=True, monotone=True, seed=0, verbose=False, **FIT)
    r = gft.evaluate(te, m, name)
    r.update(gft.baselines(tr, te, RUL_CAP).iloc[1][["RMSE"]].add_prefix("age_"))
    loso.append(r)
    print(f"  held out {name:12s} RMSE={r['RMSE']:.2f}  (age only: {r['age_RMSE']:.2f})")
pd.DataFrame(loso).to_csv("loso_results.csv", index=False)

# ---- figures -------------------------------------------------------------
# analyze() compares frame["RUL"] to RUL_hat -- hand it the CAPPED frame, or you
# are plotting a capped prediction against an uncapped truth.
analyze(test.assign(RUL=gft.cap(test["RUL"], RUL_CAP)), gft.predict_gft(test, model),
        model=model, surfaces=True, surface_kind="contour", savedir="figures/test")
res.to_csv("val_results.csv", index=False)
print("\nwrote val_results.csv, loso_results.csv, figures/test/")