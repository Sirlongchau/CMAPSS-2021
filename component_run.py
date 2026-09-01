import data, component_sensor_sweep as css
pooled = data.pooled()

# 1. adopt: build the honest config from the exhaustive CSV you already have
leaves, tags = css.build_best_leaves("ablation_component/component_ablation.csv",
                                     out_json="best_leaves_updated.json",
                                     rho_floor=0.5, spec_floor=0.4)

# 2. confirm the diagnostic winners survive across splits (not LOSO — see note)
diag = [(t["target"], t["sensors"]) for t in tags.to_dict("records") if t["tag"]=="diagnostic"]
print(css.robustness_check(pooled, diag).round(3).to_string(index=False))