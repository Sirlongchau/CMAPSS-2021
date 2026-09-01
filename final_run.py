import data, ablation, freeze_fit
pooled = data.pooled()
leaves = ablation.load_leaves("best_leaves_updated.json")
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]
tr = pooled[pooled.ds.isin(train)].reset_index(drop=True)

# fit the final trapezoid model (frozen leaves + frozen component damage + RUL-fit aggregator)
m, knees = freeze_fit.fit_trapezoid(tr, leaves, grouping="shaft", age=False, ref_frame=pooled)

# choose a test set (native test, or a held-out split), then export everything
test = pooled[(pooled.ds.isin(train)) & (pooled["split"] == "test")].reset_index(drop=True)
freeze_fit.export_all(m, test, outdir="export", ref_frame=pooled, max_units=8)