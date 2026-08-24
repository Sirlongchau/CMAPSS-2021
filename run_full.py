import data, ablation, freeze_fit
pooled = data.pooled()
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]
leaves = freeze_fit.select_leaves(
    ablation.load_leaves("ablation_out/best_leaves.json"),
    {"HPT":["eff"], "HPC":["eff","flow"], "fan":["eff"], "LPC":["eff"], "LPT":["eff","flow"]})

noage = freeze_fit.finalize_multi(pooled, leaves, train, seeds=(0,1,2,3,4), age=False,
                                  also_eval_ds=["DS01","DS04"], outdir="figures")
age   = freeze_fit.finalize_multi(pooled, leaves, train, seeds=(0,1,2,3,4), age=True,
                                  also_eval_ds=["DS01","DS04"], outdir="figures")


# for multiple runs, you can do something like this:
# ### freeze ONCE
# #m0, frozen = freeze_fit.freeze(train, leaves, leaf_gens=300, leaf_pop=80)
# freeze_fit.save_leaves(m0, frozen, "frozen_leaves.json")

# # fit the tree twice on the SAME frozen leaves
# m_noage = freeze_fit.fit_rul(train, m0, frozen)                      # age=False

# m1, _   = freeze_fit.freeze(train, leaves, age=True, leaf_gens=1, leaf_pop=1)  # skeleton only
# frozen1 = freeze_fit.load_frozen(m1, "frozen_leaves.json")           # same leaves
# m_age   = freeze_fit.fit_rul(train, m1, frozen1)                     # age=True