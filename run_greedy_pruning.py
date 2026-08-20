import data, ablation, prune, viz
pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")   # your 10-leaf config
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]

out = prune.greedy_prune(pooled, leaves, train_ds=train,
                         seeds=(0, 1, 2), gens=60, pop=120, min_leaves=1)
prune.save_prune(out, "prune_out")
viz.plot_pruning(out, "figures/pruning.png")
print(out["trajectory"].round(3).to_string(index=False))