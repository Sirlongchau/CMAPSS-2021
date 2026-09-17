import traceback, numpy as np, data, gft, ablation, freeze_fit
pooled = data.pooled()
leaves = freeze_fit.select_leaves(ablation.load_leaves("ablation_out/best_leaves.json"),
    {"HPT":["eff"],"HPC":["eff","flow"],"fan":["eff"],"LPC":["eff"],"LPT":["eff","flow"]})

# monkeypatch _fis to report the first bad call instead of dying cryptically
_orig = gft._fis
def _fis_dbg(cols, centres, singl):
    lens = [len(c) for c in cols]
    if any(l == 0 for l in lens) or len(set(lens)) != 1:
        print("BAD _fis: col lens", lens, "centre shapes", [len(c) for c in centres],
              "n_singl", len(singl))
        raise RuntimeError("bad _fis inputs")
    return _orig(cols, centres, singl)
gft._fis = _fis_dbg

tr, te, ds = next(iter(data.leave_one_dataset_out(pooled)))
try:
    gft.fit_decoupled(tr, leaves=leaves, grouping="shaft", age=True,
                      gens=4, pop=6, leaf_gens=4, leaf_pop=6, seed=0)
    print("no error on fold", ds)
except Exception:
    traceback.print_exc()