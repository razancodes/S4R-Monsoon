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

## Route A (planned)

Frozen OlmoEarth backbone + new trainable Capella patch-embedding layer; pooled
per-village embeddings replace the engineered features; identical LLP head and
losses. Capella is the only inference-time input. Known genuine domain gap:
OlmoEarth never saw X-band SAR (wavelength/resolution/speckle differ from
Sentinel-1 C-band); budget for it to underperform Route C.

## Hyperparameters

See `s4r.fallback.train.TrainConfig` (all serialized per run): alpha, loss weights
(w_total, w_mix, w_shrink, w_anchor), L2 lambda, restarts, seed, maxiter.

## Data provenance

See `docs/compliance/provenance.md`. Inference inputs are exclusively
Capella-derived; priors and weak labels act only as training-time constraints.
