"""
freeze_fit.py -- staged decoupled pipeline: large leaf training -> freeze -> observe ->
save JSON -> train the aggregator on RUL over the frozen leaves.

Same two phases as gft.fit_decoupled, but SPLIT so the frozen leaves can be inspected and
PERSISTED before (and independently of) aggregator training -- freeze once, reuse for
several RUL fits (e.g. with/without age). Reuses gft's real primitives throughout:
build_tree, _freeze_leaves (phase 1), _rul_loss_fn + ga.optimize (phase 2). Leaves are
theta-only and pinned during phase 2, so RUL can never distort them.

    select_leaves(all, keep)                 -> filter a loaded config to chosen modifiers
    m, frozen = freeze(frame, leaves, ...)   -> LARGE leaf training + freeze
    save_leaves(m, frozen, path)             -> per-leaf params (centres + consequents) to JSON
    frozen = load_frozen(m, path)            -> restore frozen leaves into a model
    m = fit_rul(frame, m, frozen, ...)       -> train spool+root on RUL, leaves pinned
    run(frame, leaves, ...)                  -> all of the above + observe (metrics + graph)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

import gft, ga


# --------------------------------------------------------------------------
# choose the leaf set
# --------------------------------------------------------------------------

def select_leaves(all_leaves, keep):
    """keep = {component: [modifiers]}, e.g. {"HPT":["eff"], "HPC":["eff","flow"],
    "fan":["eff"], "LPC":["eff"], "LPT":["eff","flow"]}. Filters a loaded leaves config
    down to the chosen modifiers (targets like 'HPT_eff_mod')."""
    out = {}
    for comp, entries in all_leaves.items():
        mods = keep.get(comp)
        if not mods:
            continue
        sel = [(nm, tg, list(se)) for (nm, tg, se) in entries
               if any(tg.endswith(f"{m}_mod") for m in mods)]
        if sel:
            out[comp] = sel
    return out


# --------------------------------------------------------------------------
# phase 1: large leaf training + freeze
# --------------------------------------------------------------------------

def freeze(frame, leaves, grouping="shaft", age=False,
           leaf_gens=300, leaf_pop=80, seed=0):
    """Build the tree skeleton and fit each leaf to its OWN theta with a large GA budget,
    then freeze. Returns (model, frozen_genome). model['genome'] is set to the frozen
    genome so leaves can be observed immediately (aggregator genes are placeholder)."""
    spec = gft.full_spec(leaves, grouping, age)
    rul_cap = gft.suggest_rul_cap(frame)
    model = gft.build_tree(spec, frame, rul_cap)
    frozen = gft._freeze_leaves(frame, model, gens=leaf_gens, pop=leaf_pop, seed=seed)
    model["genome"] = frozen.copy()
    model["spec"] = spec; model["grouping"] = grouping
    model["age"] = age; model["decoupled"] = True
    return model, frozen


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_leaves(model, frozen, path="frozen_leaves.json"):
    """Per-leaf params to JSON: inputs, MF centres, and fitted rule consequents. Enough to
    reconstruct the leaf's output and to reload into a model built on the same frame."""
    leaves = {}
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        leaves[n["target"]] = {
            "name": n["name"],
            "inputs": list(n["inputs"]),
            "centres": [np.asarray(c).tolist() for c in n["centres"]],
            "consequents": np.asarray(frozen[n["rules"]]).tolist(),
        }
    blob = {"grouping": model.get("grouping", "shaft"),
            "age": bool(model.get("age", False)),
            "rul_cap": float(model["rul_cap"]),
            "leaves": leaves}
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"saved {len(leaves)} frozen leaves -> {path}")
    return path


def load_frozen(model, path="frozen_leaves.json"):
    """Restore frozen leaf consequents from JSON into a genome for `model`. Validates
    length (MF count / sensors) and warns if saved centres differ from the model's frame."""
    with open(path) as f:
        blob = json.load(f)
    frozen = model["genome"].copy() if "genome" in model else 0.5 * (model["lo"] + model["hi"])
    saved = blob["leaves"]
    for n in model["meta"]:
        if n["kind"] != "leaf":
            continue
        s = saved.get(n["target"])
        if s is None:
            raise KeyError(f"{n['target']} missing in {path}")
        cons = np.asarray(s["consequents"], float)
        want = n["rules"].stop - n["rules"].start
        if cons.shape[0] != want:
            raise ValueError(f"{n['target']}: {cons.shape[0]} consequents, model wants {want} "
                             "(MF count or sensor set changed)")
        for cs, cn in zip(s["centres"], n["centres"]):
            if not np.allclose(np.asarray(cs), np.asarray(cn), atol=1e-4):
                print(f"  WARN {n['target']}: saved centres != current frame's "
                      "(different data?) -- consequents may not align")
                break
        frozen[n["rules"]] = cons
    return frozen


# --------------------------------------------------------------------------
# phase 2: train aggregator on RUL, leaves pinned
# --------------------------------------------------------------------------

def fit_rul(frame, model, frozen, gens=120, pop=120, seed=0):
    """Pin leaf genes to `frozen`, train spool+root on RUL (gft._rul_loss_fn). Mirrors
    fit_decoupled's phase 2 exactly. Sets model['genome']."""
    lo, hi = model["lo"].copy(), model["hi"].copy()
    for n in model["meta"]:
        if n["kind"] == "leaf":
            lo[n["rules"]] = frozen[n["rules"]]
            hi[n["rules"]] = frozen[n["rules"]]
    loss = gft._rul_loss_fn(frame, model)
    g, f, _ = ga.optimize(loss, lo, hi, gens=gens, pop=pop, seed=seed, seeds=[frozen])
    model["genome"] = g
    model["loss"] = f
    return model


# --------------------------------------------------------------------------
# full staged run + observe
# --------------------------------------------------------------------------

def run(frame, leaves, grouping="shaft", age=False,
        leaf_gens=300, leaf_pop=80, gens=120, pop=120, seed=0,
        json_path="frozen_leaves.json", outdir="figures"):
    import os
    os.makedirs(outdir, exist_ok=True)

    print("phase 1: large leaf training + freeze ...")
    model, frozen = freeze(frame, leaves, grouping, age, leaf_gens, leaf_pop, seed)

    rep0 = gft.evaluate(frame, model)
    leaftab = pd.DataFrame([{"target": n["target"],
                             "R2": rep0.get(n["target"] + ":R2"),
                             "rho": rep0.get(n["target"] + ":rho")}
                            for n in model["meta"] if n["kind"] == "leaf"])
    print("frozen leaf fidelity (in-sample):")
    print(leaftab.round(3).to_string(index=False))

    import leaf_check
    fig = leaf_check.plot_trajectories(model, frame, os.path.join(outdir, "frozen_leaves.png"))
    print("observed ->", fig)

    save_leaves(model, frozen, json_path)

    print("\nphase 2: train aggregator on RUL (leaves pinned) ...")
    model = fit_rul(frame, model, frozen, gens=gens, pop=pop, seed=seed)
    rep = gft.evaluate(frame, model)
    print(f"RUL (decoupled): RMSE={rep['RMSE']:.3f}  NASA={rep['NASA']:.3f}  "
          f"R2={rep['R2']:.3f}  rho_RUL={rep['rho_RUL']:.3f}")
    return model, frozen, {"leaf_fidelity": leaftab, "rul": rep}


# --------------------------------------------------------------------------
# self-test (synthetic; not run by the assistant unless invoked)
# --------------------------------------------------------------------------

CHOSEN = {
    "HPT": [("hpt_eff",  "HPT_eff_mod",  ["T48", "T30", "Nc"])],
    "HPC": [("hpc_eff",  "HPC_eff_mod",  ["Ps30", "T30", "Nc"]),
            ("hpc_flow", "HPC_flow_mod", ["P24", "T30", "Nc"])],
    "fan": [("fan_eff",  "fan_eff_mod",  ["T24", "P24", "Nf"])],
    "LPC": [("lpc_eff",  "LPC_eff_mod",  ["P24", "T24", "Nf"])],
    "LPT": [("lpt_eff",  "LPT_eff_mod",  ["T50", "Nf", "Nc"]),
            ("lpt_flow", "LPT_flow_mod", ["T50", "P50", "P24"])],
}


def _self_test():
    import ablation, os
    pooled, _ = ablation._synth_fleet()
    fr = pooled[pooled["ds"] == "DS08"].reset_index(drop=True)

    # small budgets for the smoke test
    model, frozen, res = run(fr, CHOSEN, leaf_gens=30, leaf_pop=16, gens=20, pop=16,
                             json_path="/tmp/frozen_leaves.json", outdir="/tmp/ff")
    assert len(res["leaf_fidelity"]) == 7
    assert os.path.exists("/tmp/frozen_leaves.json")
    assert os.path.exists("/tmp/ff/frozen_leaves.png")
    assert {"RMSE", "NASA", "R2", "rho_RUL"} <= set(res["rul"].keys())

    # JSON round-trip: reload frozen leaves into a fresh model and check they match
    m2, _ = freeze(fr, CHOSEN, leaf_gens=1, leaf_pop=4, seed=0)   # skeleton (cheap)
    frozen2 = load_frozen(m2, "/tmp/frozen_leaves.json")
    for n in m2["meta"]:
        if n["kind"] == "leaf":
            assert np.allclose(frozen2[n["rules"]], frozen[n["rules"]]), n["target"]
    print("\nfreeze_fit self-test OK (round-trip verified)")


if __name__ == "__main__":
    _self_test()