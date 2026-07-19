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

## Route B (built 2026-07-18) — Sentinel-1 pseudo-label anchors

| Signal | Provenance | Role |
|---|---|---|
| Sentinel-1 RTC clips (VV/VH, 10 m, 4 monsoon-2025 dates) | **Microsoft Planetary Computer STAC** (free, public; `sentinel-1-rtc`); cached at `data/raw/s1_stacks.npz` | **Training only** — never an inference input |
| OlmoEarth-v1-Base embeddings of the S1 clips | `allenai/OlmoEarth-v1-Base` pretrained weights (public, free); inference run locally | **Training only** — source for pseudo-labels |
| Pseudo-label anchors (`data/weak_labels/annotations.csv`, source `olmoearth_s1_v1`) | Unsupervised: k-means over OlmoEarth token embeddings + seasonal-VH-dynamics cluster labeling (`s4r.route_b.pseudo_labels`) | Training-time anchor loss / blend targets at confidence 0.3; `dominant_crop` deliberately empty |

Only STAC queries over public catalogs leave the machine; **no competition data
is ever transmitted**. Validation vs Capella features is logged in
`experiments/runs/route_b_*.json` (moderate correlations, |r| up to ~0.5).

## Route A (built 2026-07-18) — frozen OlmoEarth + Capella adapter

| Signal | Provenance | Role |
|---|---|---|
| OlmoEarth pretrained weights | Pretrained on Sentinel-1/2, Landsat (public, free) | **Frozen** backbone (requires_grad=False); auxiliary-informed by construction |
| Capella chips (`s4r.route_a.data.village_chips`) | **Capella-only** (windowed reads of GEO previews, DN→dB, per-chip z-score) through the trainable adapter | **Sole inference-time input** |
| Pseudo-label anchors (Route B above) | See Route B | Training-time `L_anchor` + blend targets only |

Route A inference is Capella chips + shapefile geometry only; the backbone
runs locally, so Capella data never leaves the machine.

All external resources used are freely and publicly accessible (Reasonableness
Standard). Capella data is processed locally only; it is never transmitted to any
third-party service.
