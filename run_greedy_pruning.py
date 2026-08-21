import data, ablation, prune, viz, gft
pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")   # your 10-leaf config
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]

out = prune.greedy_prune(pooled, leaves, train_ds=train, seeds=(0,1,2),
                         gens=80, pop=120, keep_all_components=True,
                         decoupled=True, grouping="shaft")   # now NASA-keyed
print(out["trajectory"].round(3).to_string(index=False))
rec_leaves = out["recommended"]["leaves"]    
print(out["recommended"]["why"]) 
prune.save_prune(out, "prune_out")
# then, on the recommended leaf set, test age:
m_age = gft.fit_decoupled(pooled[pooled.ds.isin(train)], leaves=rec_leaves,
                          grouping="shaft", age=True, gens=80, pop=120)


viz.plot_theta_fit(m_age, pooled[pooled.ds.isin(train)], "figures/theta_fit.png")
viz.plot_pruning(out, "figures/pruning.png")
