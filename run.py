import data, ablation, prune, viz
pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")   # the 10-leaf config
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]

out = prune.greedy_prune(
    pooled, leaves, train_ds=train, seeds=(0, 1, 2),
    gens=80, pop=120,
    keep_all_components=True,          # never drop a component's last leaf
    decoupled=True, grouping="shaft",  # the chosen architecture
)
prune.save_prune(out, "prune_decoupled_out")
viz.plot_pruning(out, "figures/prune_decoupled.png")
print(out["trajectory"][["size","dropped","RMSE","RMSE_sd","min_leaf_R2"]].round(3).to_string(index=False))