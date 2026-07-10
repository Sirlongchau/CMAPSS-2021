from ncmapss_features import cycle_features
from gft import fit_gft, predict_gft, inspect
from gft_analysis import analyze

feats = cycle_features("N-CMAPSS_DS02-006.h5", split="dev")
model = fit_gft(feats, gens=300, residualize=True)   # rul_cap=75 also works
pred = predict_gft(feats, model)

analyze(feats, pred, model=model, surfaces=True, surface_kind="contour",
        savedir="figures")   # figures/ created if missing, old .png replaced
inspect(model)