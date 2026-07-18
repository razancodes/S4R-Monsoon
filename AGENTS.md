# AGENTS.md — Rules for AI Coding Agents Working on This Repo

This file is the contract for ANY coding agent (Claude, Gemini, Copilot, Antigravity, …)
editing this repository. Read it fully before changing anything. If an instruction here
conflicts with what you were asked to do, STOP and surface the conflict to the human
instead of proceeding.

## 1. What this project is (60-second context)

Live Kaggle competition: predict cultivated hectares for **5 crops × 29 villages** from
Capella X-band SAR imagery. Metric is MSE over all 145 values — **squared** errors mean
one badly over-allocated village can cost more than everything else combined (this
happened: one ~600 ha error added >1000 MSE, see "V7 lesson" below). There are **no
training labels** — only leaderboard-derived aggregate priors. The architecture is
Learning-from-Label-Proportions (LLP): a tiny constrained model whose *aggregate*
predictions must match regional priors while per-village outputs stay differentiated
and heavily safeguarded.

Stack: Python 3.12+, `uv`, numpy/scipy/pandas/rasterio/geopandas, pytest. No deep
learning frameworks in Route C. Run everything through `uv run …`.

## 2. HARD INVARIANTS — never weaken these, in code or in tests

These encode expensive real-world failures. Changing any of them is a
product decision for the human, not a refactor:

1. **Non-negativity is structural.** Predictions are non-negative because of
   sigmoid/softmax activations in `src/s4r/fallback/head.py` — never replace with a
   loss penalty or a post-hoc clip.
2. **Per-village cap:** total predicted ha for a village must never exceed
   `alpha * area_ha` (`ALPHA_CAP = 0.38`, sweepable 0.35–0.40, never removable).
   The cap is structural in the head AND audited in `losses.cap_violations` AND
   gated in the submission writer. All three layers stay.
3. **Low-coverage shrinkage:** villages with poor SAR coverage are convex-blended
   toward the regional-mean baseline (`features/coverage.py` confidence × head blend).
   Never zero out a village, never let a low-coverage village deviate freely.
4. **Total band:** the sum of all predictions must land in **5200–5500 ha**.
5. **Crop-mix bands:** aggregate shares must stay within ±2 pp of the priors in
   `config.REGIONAL_MIX`.
6. **Synthetic-data guard:** `submission.writer.write_submission` refuses predictions
   derived from synthetic features unless `allow_synthetic=True`. Never default that
   flag to True, never remove the `is_synthetic` column plumbing.
7. **Compliance:** Capella-derived features are the only inference-time inputs.
   Auxiliary data (Sentinel, OlmoEarth, manual imagery inspection) may influence
   *training* only. Never upload/transmit competition rasters to any external service.
   Every new signal must be added to `docs/compliance/provenance.md`.
8. **Priors are approximate.** The numbers in `config.py` (5269 ha point, mix
   percentages) came from leaderboard reverse-engineering with uncertainty. Do not
   "tune" them to more decimals, do not treat them as ground truth, do not remove
   the tolerance bands around them.

## 3. Repository map

```
src/s4r/config.py             All constants/priors/schema. THE single source of truth.
src/s4r/data/synthetic.py     Synthetic 29-village fixture (tests only; never submit).
src/s4r/data/ingest.py        Strict loaders. Fail-loudly schema contract.
src/s4r/features/extract.py   Regenerates village_features.csv from raw Capella GEO
                              previews (histogram-exact zonal stats, windowed reads).
src/s4r/features/coverage.py  Coverage → confidence in [0,1].
src/s4r/fallback/baseline.py  Hedged regional-mean allocation (safety-net submission).
src/s4r/fallback/head.py      66-param head; ALL hard constraints structural here.
src/s4r/fallback/train.py     Multi-restart L-BFGS on aggregate losses; JSON run logs.
src/s4r/losses/aggregate.py   Band penalties, shrink/anchor losses, cap audit.
src/s4r/weak_labels/ingest.py Manual annotation CSV loader (validated).
src/s4r/submission/writer.py  Submission gate: refuses constraint violations.
src/s4r/cli.py                `extract` and `route-c` subcommands.
tests/                        Mirrors src. Unit tests use synthetic fixtures;
                              integration tests auto-skip without real data.
data/                         Gitignored payloads. See data/README.md.
anrf-aise-hack-*_copy/        RAW COMPETITION DATA (gitignored). READ-ONLY — never
                              modify, move, or commit anything inside it.
experiments/runs/             Auto-written JSON run logs (gitignored).
outputs/                      Generated submissions (gitignored).
docs/                         Methodology, compliance provenance, plans.
```

Data flow: raw rasters + shapefile → `extract` → `data/processed/village_features.csv`
→ `ingest` (validate) → `standardize_features` + `coverage_confidence` →
`train`/`baseline` → `write_submission` (gate) → `outputs/*.csv`.

## 4. How to make edits (required workflow)

1. **Test-first.** Every behavior change starts with a failing test. Every bug fix
   starts with a test that reproduces the bug. No exceptions — this repo's tests
   encode the anti-footgun rules, and "it looks right" is how the V7 disaster shipped.
2. **Run the suite before claiming done:**
   ```bash
   uv run pytest -q            # full suite; ~50s with real data present, all must pass
   ```
   Integration tests (`tests/test_extract_integration.py`) auto-skip if the raw data
   directory is absent — passing-by-skip on a machine WITH the data present means
   something is broken in test collection; investigate.
3. **Never edit generated artifacts** (`data/processed/*.csv`, `outputs/*.csv`,
   `experiments/runs/*.json`) by hand. Regenerate via the CLI.
4. **Schema changes** (new feature column): update `config.FEATURE_COLUMNS` (and
   `MODEL_FEATURES` only if the model should consume it), the extractor, the synthetic
   generator, and the provenance doc — in the same commit, with tests for each.
   `ingest.load_features` must keep failing loudly on missing columns.
5. **Keep the head small.** N=29 with zero labels. Do not add parameters/layers
   casually; the 66-param sizing is deliberate. If you think you need more capacity,
   that's a human decision.
6. **Commit style:** conventional prefixes (`feat:`, `fix:`, `chore:`, `test:`),
   one logical change per commit, imperative subject. Do not push or open PRs unless
   the human asked. Never commit anything gitignored (especially rasters/CSVs —
   redistributing competition data violates the rules).
7. **Determinism:** anything stochastic takes an explicit `seed`. Trainer runs must
   stay reproducible from their JSON run log.

## 5. Running the pipeline

```bash
# Regenerate features from raw data (only needed when extraction logic changes):
uv run python -m s4r.cli extract \
  --data-dir anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge_copy \
  --out data/processed/village_features.csv

# Safety-net baseline submission (hedged regional mean):
uv run python -m s4r.cli route-c --baseline-only \
  --features data/processed/village_features.csv \
  --sample anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge_copy/Sample_submission_file.csv \
  --out outputs/submission_baseline.csv

# Trained Route C (add --weak-labels data/weak_labels/annotations.csv when available):
uv run python -m s4r.cli route-c \
  --features data/processed/village_features.csv \
  --sample anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge_copy/Sample_submission_file.csv \
  --out outputs/submission.csv
```

Expected sanity marks after `extract`: 29 villages, area sum 21,006.71 ha, exactly 4
zero-coverage villages (IDs 1, 12, 25, 27). If any of these differ, extraction is
broken — do not proceed to training.

## 6. Things that look wrong but are deliberate (do NOT "fix")

- **Zero-coverage villages get exactly the regional-mean allocation.** Not a bug —
  it's the V4 lesson (zeroing them scored far worse). Only weak-label anchors may
  move them.
- **The trained model barely deviates from baseline without weak labels.** Expected:
  inside the tolerance bands the aggregate losses exert no differentiation pressure.
  Differentiation is supposed to come from anchors and (future) Route A embeddings.
- **`standardize_features` fills NaN with 0.** Post-z-score, 0 == column mean; the
  confidence blend already discounts those villages. Don't impute differently.
- **uint8 "dB" values are uncalibrated** (20·log10 of preview DN). Features are
  standardized before use, so only relative values matter. Don't add calibration.
- **`MIX_VECTOR` order ≠ submission column order.** Internal crop order is
  `config.CROPS`; the writer maps to Kaggle's column order. Always index through
  `config`, never by position assumptions.
- **Sum-check tolerances in tests** (e.g. area ±0.1 ha) reflect float/projection
  reality; don't tighten to exact equality.

## 7. Kaggle interaction rules

- Max 5 submissions/day, 2 final. Every submission is a scarce experiment — never
  burn one without a run-log entry recording exactly what config produced the file.
- Do not attempt to reverse-engineer per-village ground truth from leaderboard
  scores (underdetermined; overfits public LB). At most, ranges of aggregate priors
  may be re-confirmed — a human decision.
- Before handing any CSV to the human for submission: verify row count 29, IDs 1–29
  in sample order, exact column header
  `ID,Rice_ha,Cotton_ha,Maize_ha,Bajra_ha,Groundnut_ha`, all values ≥ 0, total in
  band, no cap violations. (`write_submission` enforces all of this — which is why
  bypassing it is forbidden.)

## 8. Roadmap context (so you don't build the wrong thing)

- **Route C (done):** classical features + LLP head. This file's rules all apply.
- **Route A (next, gated on Route C's leaderboard score):** frozen OlmoEarth backbone
  + trainable Capella patch-embedding adapter feeding the SAME LLP head and losses.
  When building it: backbone stays frozen, Capella remains the only inference input,
  Sentinel data may appear only as a training-time teacher.
- **Route B:** Sentinel-based sanity checks only. Never an inference input.
- Plans live in `docs/superpowers/plans/`. The methodology write-up
  (`docs/methodology.md`) must never claim the approach is "proven" — it is novel R&D.
