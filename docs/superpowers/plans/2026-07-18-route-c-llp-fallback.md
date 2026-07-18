# Route C — LLP-Constrained Classical Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Route C pipeline — an LLP-constrained, feature-driven allocation model (no foundation model) that produces a valid Kaggle submission CSV for the ANRF AISEHack 2026 SAR crop mapping competition, plus the shared infrastructure (config, losses, ingestion, submission validation, weak labels) that Route A will later reuse.

**Architecture:** A small linear head maps per-village standardized SAR features to (a) a cultivated-fraction via `alpha * sigmoid(...)` (structurally cap-safe) and (b) a 5-crop share vector via softmax (structurally simplex/non-negative). Predictions are convex-blended toward the regional-mean baseline weighted by per-village coverage confidence (structural V4-lesson shrinkage). The head's ~66 parameters are trained by multi-restart L-BFGS on aggregate-only LLP losses: total-area band, crop-mix band, shrinkage, optional weak-label anchors, and L2 regularization.

**Tech Stack:** Python 3.14, `uv` project management, numpy, scipy, pandas, pytest. No deep-learning framework needed for Route C.

## Global Constraints (verbatim from spec — every task implicitly includes these)

- Non-negativity of all predictions (structural, via activation functions).
- Per-village total agricultural hectares must never exceed `alpha * area_ha`, alpha ≈ 0.35–0.40 (default 0.38, sweepable, never disabled).
- Low-coverage villages must shrink toward the regional mean, never be zeroed or aggressively extrapolated.
- Capella must be the sole input at inference time in the final submitted model.
- Sum of all predictions must land within the total-area tolerance band **5200–5500 ha** (point estimate 5269 ha).
- Regional crop mix priors: Groundnut 35.9%, Cotton 22.5%, Rice 15.0%, Bajra 14.1%, Maize 12.5% — soft bands (±2 pp), not exact equality.
- Submission format: `ID,Rice_ha,Cotton_ha,Maize_ha,Bajra_ha,Groundnut_ha` — exact column order, one row per village, matching `Sample_submission_file.csv`.
- Always maintain a working, submittable fallback throughout development.
- Never claim the LLP + foundation-model synthesis is a "proven" method in documentation.
- Total village landmass: 21,006.71 ha across 29 villages. Zero-coverage village IDs: 1 (Manpura), 12 (Kotna), 25 (Pilol), 27 (Alindra); partial-low: 3 (Sankhyad ~0.4%), 5 (Khanpur ~1.2%), 11 (Chhani ~0.6%).
- **Data reality:** no real competition data exists on this machine yet. All tests run on a synthetic fixture. The submission writer must refuse synthetic-sourced data unless `--allow-synthetic` is passed. `data/README.md` documents exactly what the user must supply.

## File Structure

```
pyproject.toml                     — uv project, deps, pytest config
data/README.md                     — required real-data files + acquisition instructions
data/raw/ data/processed/ data/weak_labels/   (gitignored contents, .gitkeep)
src/s4r/__init__.py
src/s4r/config.py                  — crops, priors, bands, alpha, schema, feature lists
src/s4r/data/__init__.py
src/s4r/data/synthetic.py          — synthetic 29-village fixture generator
src/s4r/data/ingest.py             — feature-table + sample-submission loaders/validators
src/s4r/features/__init__.py
src/s4r/features/coverage.py       — coverage-confidence computation
src/s4r/losses/__init__.py
src/s4r/losses/aggregate.py        — band penalty, L_total, L_mix, L_shrink, L_anchor, L2, cap audit
src/s4r/fallback/__init__.py
src/s4r/fallback/baseline.py       — hedged regional-mean allocation (V5-style anchor)
src/s4r/fallback/head.py           — linear head forward pass + param (un)flattening
src/s4r/fallback/train.py          — objective, multi-restart L-BFGS, run logging
src/s4r/weak_labels/__init__.py
src/s4r/weak_labels/ingest.py      — manual annotation CSV loader/validator
src/s4r/submission/__init__.py
src/s4r/submission/writer.py       — CSV writer + hard-constraint validator + comparison report
src/s4r/cli.py                     — `python -m s4r.cli route-c ...` end-to-end entry
experiments/runs/                  — per-run JSON logs (gitignored contents)
docs/compliance/provenance.md      — per-feature Capella-vs-auxiliary provenance table
docs/methodology.md                — write-up stub (honesty note included)
tests/…                            — mirrors src layout
```

---

### Task 1: Repo scaffold + uv project

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/s4r/__init__.py`, package `__init__.py` files, `data/README.md`, `.gitkeep` files, `tests/test_import.py`

**Interfaces:**
- Produces: importable package `s4r`; `uv run pytest` works.

- [x] **Step 1: Scaffold**

`pyproject.toml`:
```toml
[project]
name = "s4r"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "scipy>=1.12", "pandas>=2.2"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/s4r"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/raw/*
data/processed/*
data/weak_labels/*
!data/**/.gitkeep
experiments/runs/*
!experiments/runs/.gitkeep
outputs/*
!outputs/.gitkeep
```

`data/README.md` documents required files (from user / Kaggle):
- `data/raw/Sample_submission_file.csv`, `data/raw/villages_clean.shp` (+ sidecars), `data/raw/capella/<date>/…` (4 dates: 2025-06-06, 2025-06-19, 2025-08-14, 2025-10-13), legacy `village_features.csv` → `data/processed/village_features.csv` (renamed to the schema in `src/s4r/config.py` if needed).

`tests/test_import.py`:
```python
def test_import():
    import s4r  # noqa: F401
```

- [x] **Step 2: `uv sync && uv run pytest -q`** → 1 passed
- [x] **Step 3: Commit** `chore: scaffold s4r uv project`

### Task 2: Config module

**Files:** Create `src/s4r/config.py`, `tests/test_config.py`

**Interfaces (produced, used by everything):**
```python
CROPS: list[str]                      # ["Rice","Cotton","Maize","Bajra","Groundnut"]
SUBMISSION_COLUMNS: list[str]         # ["ID","Rice_ha","Cotton_ha","Maize_ha","Bajra_ha","Groundnut_ha"]
REGIONAL_MIX: dict[str, float]        # sums to 1.0
MIX_VECTOR: np.ndarray                # (5,) in CROPS order
TOTAL_AREA_BAND = (5200.0, 5500.0); TOTAL_AREA_POINT = 5269.0
TOTAL_LANDMASS_HA = 21006.71
BASELINE_FRAC = TOTAL_AREA_POINT / TOTAL_LANDMASS_HA
ALPHA_CAP = 0.38; MIX_TOL = 0.02
DATES = ["jun06","jun19","aug14","oct13"]
ZERO_COVERAGE_IDS = [1,12,25,27]; LOW_COVERAGE_IDS = [3,5,11]
FEATURE_COLUMNS: list[str]            # full schema feature columns
MODEL_FEATURES: list[str]             # 10-feature subset fed to the head
REQUIRED_COLUMNS: list[str]           # ["village_id","village_name","area_ha"] + FEATURE_COLUMNS
```
`FEATURE_COLUMNS = [f"mean_{d}"], [f"std_{d}"], [f"coverage_{d}"] for d in DATES] + ["flood_frac_avg","traj_slope","traj_range","traj_curvature","delta_aug14_jun19","delta_oct13_aug14"]`
`MODEL_FEATURES = ["mean_jun06","mean_jun19","mean_aug14","mean_oct13","delta_aug14_jun19","delta_oct13_aug14","flood_frac_avg","traj_slope","traj_range","traj_curvature"]`

Note in a comment: Rice 0.150 + Cotton 0.225 + Maize 0.125 + Bajra 0.141 + Groundnut 0.359 = 1.000; mix values are Kaggle-column order != CROPS order pitfall — MIX_VECTOR must follow CROPS order.

- [x] **Step 1: Failing tests** (mix sums to 1; MIX_VECTOR ordering matches REGIONAL_MIX via CROPS; BASELINE_FRAC ≈ 0.2508 < ALPHA_CAP; SUBMISSION_COLUMNS exact; MODEL_FEATURES ⊂ FEATURE_COLUMNS)
- [x] **Step 2: Implement, pytest passes, commit** `feat: competition config and priors`

### Task 3: Synthetic fixture generator

**Files:** Create `src/s4r/data/synthetic.py`, `tests/test_synthetic.py`

**Interfaces:**
```python
def make_synthetic_features(seed: int = 0) -> pd.DataFrame
# 29 rows, REQUIRED_COLUMNS + "is_synthetic"=True column,
# area_ha scaled so sum == 21006.71,
# coverage_* == 0.0 for IDs 1,12,25,27; ~0.004/0.012/0.006 for IDs 3/5/11; feature values NaN where coverage is 0
```
Village names include known real ones (Manpura=1, Sankhyad=3, Khanpur=5, Chhani=11, Kotna=12, Pilol=25, Alindra=27, plus Koyali, Angadh, Asoj at other IDs; remaining synthetic names `Village_<id>`).

- [x] **Step 1: Failing tests** (29 rows; area sum ≈ 21006.71 within 0.01; zero-coverage IDs have coverage 0 and NaN means; `is_synthetic` column all True; deterministic for same seed)
- [x] **Step 2: Implement, pytest, commit** `feat: synthetic 29-village fixture generator`

### Task 4: Ingestion + validation

**Files:** Create `src/s4r/data/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
```python
class DataValidationError(ValueError): ...
def load_features(path: str | Path) -> pd.DataFrame
# raises DataValidationError on: missing REQUIRED_COLUMNS, row count != 29,
# NaN/nonpositive area_ha, duplicate village_id. Adds "is_synthetic" False if absent.
def load_sample_submission(path: str | Path) -> pd.DataFrame
# validates columns == SUBMISSION_COLUMNS, 29 rows; returns df (ID order preserved)
def standardize_features(df: pd.DataFrame) -> np.ndarray
# (29, len(MODEL_FEATURES)) z-scored per column (nan-aware), NaN→0.0 after scaling
```

- [x] **Step 1: Failing tests** (loads synthetic fixture written to tmp_path; drops a column → raises; 28 rows → raises; standardize output shape, ~0 mean, NaN→0 for zero-coverage rows)
- [x] **Step 2: Implement, pytest, commit** `feat: feature-table ingestion with strict validation`

### Task 5: Coverage confidence

**Files:** Create `src/s4r/features/coverage.py`, `tests/test_coverage.py`

**Interfaces:**
```python
def coverage_confidence(df: pd.DataFrame, saturation: float = 0.5) -> np.ndarray
# c_i = clip(mean(coverage_jun06..oct13) / saturation, 0, 1); (29,) float
```
Zero-coverage villages → 0.0 exactly (never zeroed *predictions* — they blend to baseline downstream, encoding the V4 lesson).

- [x] **Step 1: Failing tests** (IDs 1,12,25,27 → 0.0; full-coverage row → 1.0; monotone in coverage)
- [x] **Step 2: Implement, pytest, commit** `feat: coverage confidence`

### Task 6: Baseline hedged allocation (safety-net anchor)

**Files:** Create `src/s4r/fallback/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
```python
def baseline_allocation(area_ha: np.ndarray) -> np.ndarray
# (29,5): every village total = BASELINE_FRAC * area_ha_i, shares = MIX_VECTOR
```
This reproduces the V5-hedging idea (MSE 1662 anchor): spread proportionally to village area, regional mix everywhere.

- [x] **Step 1: Failing tests** (grand total == 5269 ± 0.5; per-crop aggregate proportions == REGIONAL_MIX ± 1e-9; no village exceeds ALPHA_CAP·area; non-negative)
- [x] **Step 2: Implement, pytest, commit** `feat: hedged regional-mean baseline allocation`

### Task 7: Aggregate LLP losses

**Files:** Create `src/s4r/losses/aggregate.py`, `tests/test_losses.py`

**Interfaces:**
```python
def band_penalty(x: float, lo: float, hi: float) -> float          # relu(lo-x)^2 + relu(x-hi)^2
def loss_total(totals: np.ndarray) -> float                        # band_penalty(sum, *TOTAL_AREA_BAND) / 100.0**2
def loss_mix(pred: np.ndarray) -> float                            # per-crop share vs MIX_VECTOR ± MIX_TOL, / 0.01**2, summed
def loss_shrink(frac_model, shares_model, conf) -> float
# mean_i (1-c_i)·[ ((frac_i-BASELINE_FRAC)/BASELINE_FRAC)^2 + ||shares_i-MIX_VECTOR||^2 ]
def loss_anchor(frac, anchors: pd.DataFrame | None) -> float
# anchors columns: village_index, cultivated_fraction_est, weight → mean w·(frac_i - est)^2 / BASELINE_FRAC^2; 0.0 if None/empty
def l2_penalty(theta: np.ndarray, lam: float) -> float
def cap_violations(pred: np.ndarray, area_ha: np.ndarray, alpha: float) -> np.ndarray  # bool mask, post-hoc audit
```

- [x] **Step 1: Failing tests with hand-computed values** (band inside=0, 100 over hi → known value; loss_mix zero at exact prior mix; loss_shrink zero when conf all 1; anchor arithmetic; cap mask on constructed violation)
- [x] **Step 2: Implement, pytest, commit** `feat: aggregate LLP loss components`

### Task 8: Route C head

**Files:** Create `src/s4r/fallback/head.py`, `tests/test_head.py`

**Interfaces:**
```python
N_FEATURES = len(MODEL_FEATURES)  # 10
def n_params() -> int             # (10+1) + 5*10 + 5 = 66
def unflatten(theta: np.ndarray) -> tuple[w_t, b_t, W_s, b_s]
def forward(theta, X, area_ha, conf, alpha: float = ALPHA_CAP) -> dict
# returns {"frac_model","shares_model","frac","shares","totals","pred"}
# frac_model = alpha*sigmoid(X@w_t+b_t); shares_model = softmax(X@W_s.T+b_s)
# frac = conf*frac_model + (1-conf)*BASELINE_FRAC       (structural shrink + cap: both terms ≤ alpha)
# shares = conf[:,None]*shares_model + (1-conf)[:,None]*MIX_VECTOR
# totals = frac*area_ha ; pred = totals[:,None]*shares  (29,5)
```

- [x] **Step 1: Failing tests** (shapes; for 200 random thetas: pred ≥ 0, totals ≤ alpha·area + 1e-9, shares rows sum to 1; conf=0 rows equal baseline_allocation rows exactly)
- [x] **Step 2: Implement, pytest, commit** `feat: cap-safe LLP head with structural shrinkage`

### Task 9: Trainer + run logging

**Files:** Create `src/s4r/fallback/train.py`, `tests/test_train.py`

**Interfaces:**
```python
@dataclass
class TrainConfig:  # all hyperparameters explicit, serialized into run log
    alpha: float = ALPHA_CAP; w_total: float = 1.0; w_mix: float = 1.0
    w_shrink: float = 0.1; w_anchor: float = 1.0; lam: float = 1e-3
    n_restarts: int = 8; seed: int = 0; maxiter: int = 300

def objective(theta, X, area_ha, conf, cfg, anchors=None) -> float   # weighted loss sum
def train(X, area_ha, conf, cfg, anchors=None, run_dir="experiments/runs") -> dict
# multi-restart scipy.optimize.minimize(method="L-BFGS-B"), keep best;
# returns {"theta","pred","loss_components": {...}, "run_log_path"}
# writes JSON run log: cfg fields, per-component losses, restart losses, aggregate total, per-crop mix, timestamp
```

- [x] **Step 1: Failing tests** (on synthetic fixture: trained aggregate total inside 5200–5500; per-crop mix within MIX_TOL+0.005 of priors; no cap violations; final loss ≤ init loss; run-log JSON exists and round-trips)
- [x] **Step 2: Implement, pytest (allow ~1 min runtime), commit** `feat: multi-restart L-BFGS LLP trainer with run logging`

### Task 10: Weak-label ingestion (+ template)

**Files:** Create `src/s4r/weak_labels/ingest.py`, `data/weak_labels/TEMPLATE.csv` (committed), `tests/test_weak_labels.py`

**Interfaces:**
```python
def load_weak_labels(path, features_df) -> pd.DataFrame
# CSV columns: village_id,cultivated_fraction_est,dominant_crop,confidence(0-1),source,notes
# validates: village_id ∈ features_df, 0 ≤ fraction ≤ 1, crop ∈ CROPS or "", conf ∈ (0,1]
# returns anchors df with village_index (row position in features_df) and weight=confidence
```
Template row example: `8,0.05,,0.9,"Google Earth Pro 2025-08 historical","Koyali refinery — near-zero cropland"`

- [x] **Step 1: Failing tests** (valid file loads and maps village_index; unknown id → raises; fraction 1.2 → raises)
- [x] **Step 2: Implement, pytest, commit** `feat: weak-label annotation ingestion`

### Task 11: Submission writer, validator, comparison report

**Files:** Create `src/s4r/submission/writer.py`, `tests/test_submission.py`

**Interfaces:**
```python
class SubmissionError(ValueError): ...
def validate_predictions(pred, area_ha, alpha) -> list[str]
# returns list of violation strings (empty = OK): negativity, cap, total outside band; caller decides severity
def write_submission(pred, features_df, sample_df, out_path, allow_synthetic=False) -> Path
# maps rows by village_id → sample "ID" order; raises SubmissionError if is_synthetic.any() and not allow_synthetic,
# or if validate_predictions non-empty, or IDs mismatch sample
def comparison_report(preds: dict[str, np.ndarray], features_df, alpha) -> pd.DataFrame
# one row per village: name, area_ha, per-route totals, cap headroom, flags; plus aggregate footer rows (total, per-crop mix)
```

- [x] **Step 1: Failing tests** (round-trip CSV has exact SUBMISSION_COLUMNS and 29 rows in sample ID order; synthetic without flag → raises; injected cap violation → raises; report contains both routes' totals)
- [x] **Step 2: Implement, pytest, commit** `feat: submission writer with hard-constraint validation`

### Task 12: CLI end-to-end + docs

**Files:** Create `src/s4r/cli.py`, `tests/test_cli.py`, `docs/compliance/provenance.md`, `docs/methodology.md`

**Interfaces:**
```bash
uv run python -m s4r.cli route-c --features F.csv --sample S.csv --out outputs/sub.csv \
    [--baseline-only] [--weak-labels W.csv] [--alpha 0.38] [--allow-synthetic] [--seed 0]
```
argparse; wires: load → standardize → confidence → (baseline | train) → report printed to stdout → write_submission. Exit code 1 with clear message on DataValidationError/SubmissionError.

`docs/compliance/provenance.md`: table of every MODEL_FEATURE → "Capella-only" (all Route C features are Capella-derived); priors → "leaderboard-derived, approximate"; weak labels → "manual visual inspection of free public imagery (validation role)".
`docs/methodology.md`: architecture, losses, hyperparameters; explicit honesty note: LLP+foundation-model synthesis is novel, not a proven published method; Route C is the maintained fallback.

- [x] **Step 1: Failing CLI test** (subprocess/monkeypatched run on synthetic fixture with `--allow-synthetic` writes valid CSV; without flag exits nonzero)
- [x] **Step 2: Implement, pytest full suite green, commit** `feat: route-c CLI and compliance docs`

---

## Out of scope (separate future plans)
- Route A (OlmoEarth Capella adapter) — gated on Route C leaderboard result per spec §9.6.
- Route B (Sentinel calibration), Capella patch extraction, real feature regeneration from rasters — blocked on real data.

## Self-Review Notes
- Spec coverage: §3.3 losses (1–6) → Tasks 7–9 (L_cap made structural in head + post-hoc audit; non-negativity structural); §4.3 weak labels → Task 10; §7 deliverables 1,2,4,5,6 → Tasks 1,12,11,12,12; §8 hard constraints → head design + validator.
- Type consistency: `forward` returns dict consumed by trainer/writer; `features_df` carries `is_synthetic`; anchors df produced by Task 10 matches `loss_anchor` contract in Task 7.
- Real-data gap is explicit: ingestion contract + data/README.md; nothing here submits synthetic output silently.
