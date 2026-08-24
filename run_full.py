import data, ablation, freeze_fit
pooled = data.pooled()
train  = pooled[pooled.ds.astype(str).str.startswith("DS08")].reset_index(drop=True)

leaves = freeze_fit.select_leaves(
    ablation.load_leaves("ablation_out/best_leaves.json"),
    {"HPT": ["eff"], "HPC": ["eff", "flow"], "fan": ["eff"],
     "LPC": ["eff"], "LPT": ["eff", "flow"]})          # the 7-leaf set you specified

model, frozen, res = freeze_fit.run(
    train, leaves, grouping="shaft", age=False,
    leaf_gens=300, leaf_pop=80,     # LARGE leaf training (phase 1)
    gens=120, pop=120,              # aggregator on RUL (phase 2)
    json_path="frozen_leaves.json", outdir="figures")


# for multiple runs, you can do something like this:
# ### freeze ONCE
# #m0, frozen = freeze_fit.freeze(train, leaves, leaf_gens=300, leaf_pop=80)
# freeze_fit.save_leaves(m0, frozen, "frozen_leaves.json")

# # fit the tree twice on the SAME frozen leaves
# m_noage = freeze_fit.fit_rul(train, m0, frozen)                      # age=False

# m1, _   = freeze_fit.freeze(train, leaves, age=True, leaf_gens=1, leaf_pop=1)  # skeleton only
# frozen1 = freeze_fit.load_frozen(m1, "frozen_leaves.json")           # same leaves
# m_age   = freeze_fit.fit_rul(train, m1, frozen1)                     # age=True