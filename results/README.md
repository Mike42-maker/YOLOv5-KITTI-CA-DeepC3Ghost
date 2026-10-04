# Result provenance

validation_summary.csv is the public main-results table. Its values are rounded to exactly match the ablation table in the submitted final paper icftic2026_submission_final/ICFTIC2026_Xiao.tex (seed 42, 640x640, batch size 8).

benchmark_summary.csv contains the detailed forward benchmark values from the submitted paper comparison protocol: RTX 4060 Laptop GPU, FP32, batch size 1, 50 warmup forwards, and 500 timed forwards. The extra decimal places are measurement detail and are not mixed into the rounded main-results table.

ca_model_smoke.json is an architecture construction smoke test, not a trained accuracy result. experiment_protocol.json records the common evaluation protocol. KITTI data, checkpoints, and training logs are intentionally excluded.
