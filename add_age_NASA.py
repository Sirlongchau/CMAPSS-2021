"""
add_age_nasa.py -- add the NASA-vs-age comparison to an EXISTING confirm_splits.csv
WITHOUT retraining any tree.

Why no retrain is needed: the model's own NASA is already in confirm_splits.csv
(the `NASA` column, written by evaluate()). Only the AGE baseline's NASA is
missing, and computing it involves no fitting -- gft.baselines() is pure
arithmetic on the age line. The splits are deterministic in the seed, so we
reproduce each split's fit-set, recompute the age NASA, and merge on `split`.

Use this ONLY if you don't want to re-run confirm.py. If you re-run the updated
confirm.py, it already writes age_NASA / beats_age_nasa and you don't need this.

    python add_age_nasa.py
"""
import pandas as pd

import gft
import datasets as ds

S = pd.read_csv("confirm_splits.csv")
feats = ds.pooled()

age = []
for k in sorted(S["split"].unique()):
    train, val, test = ds.split(feats, fracs=(0.8, 0.1, 0.1), seed=int(k))
    fitset = pd.concat([train, val], ignore_index=True)
    cap = gft.suggest_rul_cap(fitset)                       # identical to the run
    row = gft.baselines(fitset, test, cap)                  # no fitting
    row = row[row["model"].str.startswith("age only")].iloc[0]
    age.append(dict(split=int(k), age_NASA=float(row["NASA"])))

S = S.merge(pd.DataFrame(age), on="split", how="left", suffixes=("", "_new"))
if "age_NASA_new" in S.columns:                             # column already existed
    S["age_NASA"] = S["age_NASA_new"]; S = S.drop(columns="age_NASA_new")
S["beats_age_nasa"] = S["NASA"] < S["age_NASA"]

print("fraction of splits that beat the age baseline:")
print(S.groupby("model")[["beats_age", "beats_age_nasa"]].mean().round(2).to_string())

S.to_csv("confirm_splits.csv", index=False)
print("\nupdated confirm_splits.csv (added age_NASA, beats_age_nasa)")