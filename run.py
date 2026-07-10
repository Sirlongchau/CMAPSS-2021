from ncmapss_features import cycle_features
from gft import fit_gft, predict_gft, inspect
from gft_analysis import analyze

feats = cycle_features("N-CMAPSS_DS01-005.h5", split="dev")

# Hybrid loss: leaves supervised on theta (theta_weight), whole tree on RUL.
# theta_weight=0 -> pure RUL training; raise it to trust the theta labels more.
model = fit_gft(feats, gens=1000, residualize=True, theta_weight=10.0)
pred = predict_gft(feats, model)

# one figure per theta + one figure per sub-FIS, all written to figures/
analyze(feats, pred, model=model, surfaces=True, surface_kind="contour",
        savedir="figures")
inspect(model)