import os
import data, ablation, prune, leaf_check
pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]

# 1. Prune (plain module — specificity is handled downstream, not in the prune).
out = prune.greedy_prune(
    pooled, leaves, train_ds=train, seeds=(0, 1, 2),
    gens=80, pop=120, keep_all_components=True,
    decoupled=True, grouping="shaft")
prune.save_prune(out, "prune_out")
print(out["trajectory"].round(3).to_string(index=False))

rec = out["recommended"]["leaves"]

# 2. Held-out fidelity + deep-tail slope (check() fits its own unit-disjoint split).
print("\n== held-out R2/rho + tail slope ==")
print(leaf_check.check(pooled, rec, train).round(3).to_string(index=False))

# 3. Fit the recommended set once (the model you'd freeze) for the model-based views.
m = leaf_check.fit(pooled, rec, train, seed=0)

# 4. Component-specificity — pass the FULL pooled fleet (needs fault-mode diversity,
#    i.e. units where each component is healthy). Read the ORDERING: HPT_flow low
#    (global damage), HPT_eff high (component-specific).
print("\n== specificity ==")
print(leaf_check.specificity(m, pooled).round(3).to_string(index=False))

# 5. Trajectory view — real theta vs predicted theta over cycles, per unit.
os.makedirs("figures", exist_ok=True)
leaf_check.plot_trajectories(m, pooled[pooled.ds.isin(train)], "figures/leaf_traj.png")
print("\nwrote figures/leaf_traj.png")