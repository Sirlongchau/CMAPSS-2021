import gft, datasets as ds
f = ds.pooled(); cap = gft.suggest_rul_cap(f)
print((f["RUL"] >= cap).mean())