# Methodology — Phase 2: LLP-Constrained Crop Area Estimation

## Status and honesty note

This pipeline synthesizes Learning-from-Label-Proportions (LLP) ideas with
leaderboard-derived aggregate priors and (in the planned Route A) a foundation-model
backbone. **This exact combination has no direct precedent in published literature**
— published LLP crop-mapping work (e.g., La Rosa et al.) uses government crop
statistics as the proportion source and targets classification, not direct hectare
regression. We treat this as R&D under active validation, not a proven method. A
classical fallback (Route C) is maintained and submittable at all times.

## Problem framing

29 villages × 5 crops, no per-village labels. Known (approximately, from leaderboard
reverse engineering): the regional cultivated total (5200–5500 ha band, point 5269)
and the regional crop mix. The MSE metric squares per-village errors, so single
large errors dominate (the "V7 lesson": one ~600 ha over-allocation cost >1000 MSE).
The correct framework for "reliable aggregate + no local labels" is LLP: individual
predictions are constrained only through their aggregate.

## Route C architecture (implemented)

- **Inputs per village:** 10 standardized Capella-derived features
  (`s4r.config.MODEL_FEATURES`), village area, per-date coverage → confidence
  `c_i ∈ [0,1]` (`s4r.features.coverage`).
- **Head (66 parameters):** `frac_model = alpha·sigmoid(w_t·x + b_t)`;
  `shares_model = softmax(W_s·x + b_s)`. Small by design: N=29, no labels — larger
  heads memorize noise (cf. Ramos-Pollán et al., on-orbit LLP training, as a sizing
  reference, not a hard rule).
- **Structural constraints (never loss-only):** non-negativity and the crop simplex
  via activations; per-village cap `total_i ≤ alpha·area_i` (alpha = 0.38 default,
  sweep 0.35–0.40) because both blend endpoints are ≤ alpha; shrinkage via a
  confidence-weighted convex blend toward the regional mean allocation
  (`c_i = 0` ⇒ exactly regional mean — encodes the V4 lesson: never zero out,
  never extrapolate low-coverage villages).
- **Weak-label anchors (spec §4.3):** manual estimates from free public imagery
  become (a) an anchor loss term and (b) replacement blend targets, so
  zero-coverage villages can still be moved by human evidence.
- **Losses (aggregate-only, soft bands not point targets):**
  `L_total` (band 5200–5500 ha), `L_mix` (per-crop share bands, ±2 pp),
  `L_shrink`, `L_anchor`, plus L2. Weighted sum minimized by multi-restart
  L-BFGS-B (`s4r.fallback.train`). Every run logs its full configuration and
  loss breakdown to `experiments/runs/*.json`.
- **Submission gate:** `s4r.submission.writer` refuses any output violating
  non-negativity, the cap, the total band, or sample-file ID order — and refuses
  synthetic-sourced predictions outright.

## Route B (implemented) — Sentinel-1 pseudo-label anchors

Sentinel-1 RTC clips (Planetary Computer, monsoon-2025 dates mirroring the
Capella acquisitions) are embedded by a frozen OlmoEarth-v1-Base encoder;
k-means over token embeddings plus a seasonal-VH-dynamics rule flags
agricultural clusters; the per-village fraction of agricultural tokens becomes
a `cultivated_fraction_est` anchor at confidence 0.3 (`s4r.route_b`).
Validation against Capella features showed **moderate** correlations
(|r| up to ~0.5, e.g. Spearman −0.60 vs `delta_aug14_jun19`) — treated as
weak-but-real signal, hence the low anchor confidence. `dominant_crop` is left
empty: an unsupervised pipeline cannot credibly name crops. Training-time
signal only; never an inference input.

## Route A (implemented, under validation)

Frozen OlmoEarth-v1-Base trunk (89M params, requires_grad=False) + trainable
1-channel Capella patch-embedding adapter and 768→10 projection; the pooled
features feed a torch mirror of the SAME 66-parameter LLP head (numerical
equivalence with the numpy head is test-enforced) and the same aggregate
losses, with `L_anchor` consuming the Route B pseudo-labels. Capella is the
only inference-time input. Known genuine domain gap: OlmoEarth never saw
X-band SAR (wavelength/resolution/speckle differ from Sentinel-1 C-band);
budget for it to underperform Route C.

## Hyperparameters

See `s4r.fallback.train.TrainConfig` (all serialized per run): alpha, loss weights
(w_total, w_mix, w_shrink, w_anchor), L2 lambda, restarts, seed, maxiter.

## Data provenance

See `docs/compliance/provenance.md`. Inference inputs are exclusively
Capella-derived; priors and weak labels act only as training-time constraints.
