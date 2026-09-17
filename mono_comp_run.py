import data, ablation, freeze_fit
pooled = data.pooled()
leaves = ablation.load_leaves("best_leaves_updated.json")     # or best_leaves_updated.json from the ablation

# 1. Validate the frozen construction: knees from TRAIN, honesty measured on HELD-OUT, 5 splits.
val = freeze_fit.validate_trapezoid(pooled, leaves, split_seeds=(0,1,2,3,4))
print(val.round(3).to_string(index=False))   # per component: spec, spec_xshaft, ±sd across splits

# 2. Build the model with frozen trapezoid components (knees fleet-wide, RUL on top).
train = pooled[pooled.ds.astype(str).str.startswith("DS08")].reset_index(drop=True)
m, knees = freeze_fit.fit_trapezoid(train, leaves, age=False, ref_frame=pooled)
print(knees.round(4).to_string(index=False))  # the frozen p10/p50/p90 per modifier
