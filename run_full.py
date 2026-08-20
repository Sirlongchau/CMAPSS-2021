import data, ablation, gft, viz

pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]

full = ablation.train_full(
    pooled, leaves, train_ds=train,
    gens=400, pop=120, seed=0,          # see notes below
    branch_models=None,                  # warm-start optional; see note 3
)

print(gft.evaluate(pooled[pooled.ds.isin(train)], full))
print(ablation.dead_branches(full, pooled,
      {r.ds: r.components.split("+") for r in data.coverage(pooled).itertuples()
       if len(r.components.split("+")) == 1}))
viz.plot_assembled(full, pooled,
                   {r.ds: r.components.split("+") for r in data.coverage(pooled).itertuples()},
                   "figures/full_400gen.png")