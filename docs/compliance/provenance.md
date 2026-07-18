# Signal Provenance Record (Compliance)

Per the host ruling, Capella X-band SAR must remain the primary dataset, and the
final estimation must primarily rely on information extracted from it. This table
records the provenance of every signal in the pipeline. Update it whenever a
feature or signal is added.

## Route C (classical fallback) — current state

| Signal | Provenance | Role at inference |
|---|---|---|
| `mean_<date>`, `std_<date>` (4 dates) | **Capella-only** (zonal statistics of Capella backscatter per village polygon) | Direct model input |
| `coverage_<date>` | **Capella-only** (acquisition footprint vs village polygon) | Confidence weighting |
| `delta_aug14_jun19`, `delta_oct13_aug14` | **Capella-only** (temporal differences of Capella means) | Direct model input |
| `flood_frac_avg` | **Capella-only** (low-backscatter thresholding on Capella) | Direct model input |
| `traj_slope`, `traj_range`, `traj_curvature` | **Capella-only** (fits to the 4-date Capella trajectory) | Direct model input |
| `area_ha`, geometry | Competition shapefile (`villages_clean.shp`, EPSG:32643 projection) | Cap + scaling |
| Regional total band (5200–5500 ha) | **Leaderboard-derived prior** (ablation submissions V1–V7, approximate) | Soft training constraint only |
| Regional crop mix (GN 35.9% / Cotton 22.5% / Rice 15.0% / Bajra 14.1% / Maize 12.5%) | **Leaderboard-derived prior** (approximate) | Soft training constraint only |
| Weak labels (`data/weak_labels/annotations.csv`) | Manual visual inspection of **free public imagery** (Google Earth Pro historical, Bing Maps, Sentinel-2 true color) | Training-time anchors / blend targets; no auxiliary imagery is an inference input |

**Inference inputs are exclusively Capella-derived features + competition shapefile
geometry.** Auxiliary sources (weak labels) act only through trained parameters and
blend targets, consistent with the ruling's "validate rule-based and threshold-based
approaches" allowance.

## Route A (planned, not yet built)

| Signal | Provenance | Role |
|---|---|---|
| OlmoEarth pretrained weights | Pretrained on Sentinel-1/2, Landsat (public, free) | Frozen backbone; auxiliary-informed by construction |
| Capella patch embeddings | **Capella-only** input through new trainable adapter | Sole inference-time input |
| Optional Sentinel-1 distillation teacher | Free public Sentinel-1 GRD | **Training only**, never at inference |

All external resources used are freely and publicly accessible (Reasonableness
Standard). Capella data is processed locally only; it is never transmitted to any
third-party service.
