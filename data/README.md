# Data Directory — Required Real Files (NOT in git)

**Current state: NO real competition data is present on this machine.** Everything in
`tests/` runs against a synthetic fixture. The pipeline will refuse to write a real
submission from synthetic data (see `--allow-synthetic` guard).

## What you must supply

Place the following before running the real pipeline:

| Path | Source | Notes |
|---|---|---|
| `data/raw/Sample_submission_file.csv` | Kaggle competition data page | Defines exact ID values and row order |
| `data/raw/villages_clean.shp` (+ `.shx/.dbf/.prj`) | Kaggle competition data page | 29 villages, EPSG:4326; project to EPSG:32643 for areas |
| `data/raw/capella/<date>/…` | Kaggle competition data page | 4 dates: 2025-06-06, 2025-06-19, 2025-08-14, 2025-10-13 (SLC + GEO `.tif` + `.json`) |
| `data/processed/village_features.csv` | Legacy repo (port) or regenerate from rasters | Must match the schema in `src/s4r/config.py` (`REQUIRED_COLUMNS`) |
| `data/weak_labels/annotations.csv` | Manual visual inspection (Section 4.3 of spec) | Copy `TEMPLATE.csv`, fill rows |

## Porting the legacy `village_features.csv`

The legacy feature table may use different column names. Rename columns to match
`s4r.config.FEATURE_COLUMNS`:
`mean_<date>`, `std_<date>`, `coverage_<date>` for date in `jun06, jun19, aug14, oct13`,
plus `flood_frac_avg, traj_slope, traj_range, traj_curvature, delta_aug14_jun19, delta_oct13_aug14`,
and identity columns `village_id, village_name, area_ha`.
Extra columns are allowed and ignored. `s4r.data.ingest.load_features` will fail loudly
listing anything missing.

## Kaggle download (once credentials exist)

```bash
pip install kaggle  # or: uv tool install kaggle
# put API token in ~/.kaggle/kaggle.json (Kaggle → Account → Create New Token)
kaggle competitions download -c <competition-slug> -p data/raw/
```

## Compliance reminder

Never upload/transmit Capella rasters to third-party services. All processing is local.
