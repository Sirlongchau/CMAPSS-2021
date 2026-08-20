import data, ablation, viz
#data.verify()
#data.build_cache()
pooled = data.pooled()
cov = data.coverage(pooled)
modes = {r.ds: r.components.split("+") for r in cov.itertuples()
         if len(r.components.split("+")) <= 2}          # single/joint-fault files
train = [d for d in pooled.ds.unique() if d.startswith("DS08")]

out = ablation.run_ablation(pooled, modes, train_ds=train, gens=60, pop=60)
ablation.save_config(out, "ablation_out")               # <-- persist the winners

# later, or in a fresh process — train the full tree straight from the saved config:
leaves = ablation.load_leaves("ablation_out/best_leaves.json")
full   = ablation.train_full(pooled, leaves, train_ds=train,
                             gens=80, pop=60, branch_models=out["branch_models"])
print(ablation.dead_branches(full, pooled, modes))       # which own-spools stay inert
viz.plot_assembled(full, pooled, modes, "figures/full_tree.png")