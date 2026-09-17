import data, ablation, leaf_train
pooled = data.pooled()
leaves = ablation.load_leaves("ablation_out/best_leaves.json")   # or your 7-leaf recommended set
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]
df = leaf_train.run(pooled, leaves, train, ks=(2,3,4,5), seeds=(0,1,2,3,4), gens=150, pop=60)