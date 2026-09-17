import data, ablation, freeze_fit, gft, run_full, viz, os
pooled = data.pooled()
leaves = ablation.load_leaves("best_leaves_updated.json")
train  = [d for d in pooled.ds.unique() if d.startswith("DS08")]           # all subsets (full-native protocol)

# # CV + LOSO (these split internally, no leakage)
rulfit = freeze_fit.finalize_multi(pooled, leaves, train, seeds=(0,1,2,3,4), age=False,
                                    components="rulfit",    outdir="ab_rulfit")
trapz  = freeze_fit.finalize_multi(pooled, leaves, train, seeds=(0,1,2,3,4), age=False,
                                   components="trapezoid", outdir="ab_trapezoid")
# print(rulfit["summary"].round(3).to_string(index=False))
print(trapz["summary"].round(3).to_string(index=False))
print(run_full.loso(pooled, leaves, ages=(False,), components="trapezoid").round(3).to_string(index=False))
# print(run_full.loso(pooled, leaves, ages=(False,), components="rulfit").round(3).to_string(index=False))

# FINAL model: train on native DEV only (no test leakage), test on native TEST
tr_dev = pooled[(pooled["ds"].isin(train)) & (pooled["split"] == "dev")].reset_index(drop=True)
test   = pooled[(pooled["ds"].isin(train)) & (pooled["split"] == "test")].reset_index(drop=True)
m, knees = freeze_fit.fit_trapezoid(tr_dev, leaves, grouping="shaft", age=False, ref_frame=tr_dev)

# # export everything (writes the LP cube among the control surfaces)
freeze_fit.export_all(m, test, outdir="export_mono", ref_frame=tr_dev, max_units=8)

# # make the cube explicit + list what was written
figs = viz.plot_control_surfaces(m, tr_dev, "export_mono")
print("\ncontrol-surface figures written to export/:")
for f in figs:
    print("   ", os.path.basename(f))
print("LP cube ->", os.path.abspath("export_mono/control_surfaces_lp_cube.png"))

rep = gft.evaluate(test, m)
print(f"\nNATIVE dev->test: n_units={test['unit'].nunique()} n_rows={len(test)}  "
      f"RMSE={rep['RMSE']:.3f} R2={rep['R2']:.3f} rho={rep['rho_RUL']:.3f} "
      f"NASA_cohen={0.5*rep['RMSE']+0.5*rep['NASA']:.3f}")