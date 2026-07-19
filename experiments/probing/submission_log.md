# Submission Log

## Baseline: `submission_physical_prior_v1.csv`
- **Method**: Route C (Linear Head) with L2 Physical Priors.
- **Priors Injected**: Rice ↔ flood_frac_avg (+1.0); Cotton ↔ mean_oct13 (+1.0), delta_oct13_aug14 (+1.0); Maize/Bajra ↔ mean_aug14 (+1.0), delta_oct13_aug14 (-1.0).
- **Validation**: Replaced flat regional allocation with highly variant per-village predictions. No cap violations.

---

## Probe Submissions

| File | Village ID | Crop Name | Delta Added (ha) | Notes |
|---|---|---|---|---|
| `submission_probe_01_25_Groundnut.csv` | 25 | Groundnut | +60.0 | |
| `submission_probe_02_25_Cotton.csv` | 25 | Cotton | +60.0 | |
| `submission_probe_03_25_Rice.csv` | 25 | Rice | +60.0 | |
| `submission_probe_04_25_Bajra.csv` | 25 | Bajra | +60.0 | |
| `submission_probe_05_25_Maize.csv` | 25 | Maize | +60.0 | |
| `submission_probe_06_27_Groundnut.csv` | 27 | Groundnut | +60.0 | |
| `submission_probe_07_27_Cotton.csv` | 27 | Cotton | +60.0 | |
| `submission_probe_08_27_Rice.csv` | 27 | Rice | +60.0 | |
| `submission_probe_09_27_Bajra.csv` | 27 | Bajra | +60.0 | |
| `submission_probe_10_27_Maize.csv` | 27 | Maize | +60.0 | |
| `submission_probe_11_12_Groundnut.csv` | 12 | Groundnut | +60.0 | |
| `submission_probe_12_12_Cotton.csv` | 12 | Cotton | +60.0 | |
