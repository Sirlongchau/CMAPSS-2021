from ncmapss_gft_features import extract_features, Config
from gft_ncmapss import fit_gft, predict_gft, GFTConfig, inspect_consequents
from gft_analysis_1 import analyze

feats, _ = extract_features("N-CMAPSS_DS02-006.h5", Config(split="dev"))
model = fit_gft(feats, GFTConfig(gens=200, residualize=True, smooth_span=5))
pred  = predict_gft(feats, model)
analyze(feats, pred, model=model, surfaces=True, surface_kind="contour")
inspect_consequents(model)