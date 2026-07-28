"""
visualize_final.py -- fit the confirmed deployable model ONCE and render every
figure for graphical analysis.

confirm.py fits 24 throwaway models across splits and keeps only metrics; it
produces no model to look at. This fits a single clean instance of the deployable
arm and hands it to gft_analysis.analyze(), which renders:

    * memberships     -- the (fixed) Ruspini partition per FIS input
    * control surfaces -- each sub-FIS response, monotone where constrained
    * latent nodes    -- the hp_h / lp_h damage curves over each unit's life
    * theta parity    -- theta_hat vs theta for each leaf
    * RUL trajectories + parity, and GA convergence

    python visualize_final.py

Note: with learn_mf=False the memberships plot shows the learned partition sitting
exactly on the quantile seed (solid == dashed) -- that is correct now, not a bug.
"""
import pandas as pd

import gft
import datasets as ds
from gft_analysis import analyze

SEED = 0
SAVEDIR = "figures/final"

feats = ds.pooled()
train, val, test = ds.split(feats, fracs=(0.8, 0.1, 0.1), seed=SEED)
fitset = pd.concat([train, val], ignore_index=True)
cap = gft.suggest_rul_cap(fitset)

# the confirmed deployable arm: LP-specific, fixed-MF, monotone
model = gft.fit_gft(fitset, gft.TREE_LP1, learn_mf=False, monotone=True,
                    theta_weight=1.0, trend_weight=1.0, smooth_span=7,
                    rul_cap=cap, gens=400, pop=120, seed=SEED)

# per-FIS inputs, grid, singleton ranges, monotone tags, tuned-vs-seed centres
gft.inspect(model)

# analyze() compares frame["RUL"] to RUL_hat -- hand it the CAPPED frame, or you
# plot a capped prediction against an uncapped truth (PROJECT_STATE.md sec 8).
analyze(test.assign(RUL=gft.cap(test["RUL"], cap)),
        gft.predict_gft(test, model),
        model=model, surfaces=True, surface_kind="contour", savedir=SAVEDIR)

print(f"\nwrote figures to {SAVEDIR}/")