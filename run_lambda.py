import data, component_sensor_sweep as css
pooled = data.pooled()
out = css.run_ablation(pooled, outdir="ablation_component")   # ~2860 fits + confirm; one command