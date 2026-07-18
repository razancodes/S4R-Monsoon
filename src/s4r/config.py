"""Competition constants, leaderboard-derived priors, and the feature-table schema.

Priors below were reverse-engineered from public-leaderboard MSE responses to
ablation submissions (V1-V7 in the legacy repo). They are soft, approximate
targets with uncertainty bands — NOT ground truth. Do not tighten them to more
decimal places without fresh leaderboard evidence.
"""

import numpy as np

# Internal crop order. NOTE: submission column order below is the Kaggle-defined
# order; MIX_VECTOR must always follow CROPS order.
CROPS = ["Rice", "Cotton", "Maize", "Bajra", "Groundnut"]
SUBMISSION_COLUMNS = ["ID", "Rice_ha", "Cotton_ha", "Maize_ha", "Bajra_ha", "Groundnut_ha"]

# Regional crop mix priors (leaderboard-derived): sums to exactly 1.000.
REGIONAL_MIX = {
    "Rice": 0.150,
    "Cotton": 0.225,
    "Maize": 0.125,
    "Bajra": 0.141,
    "Groundnut": 0.359,
}
MIX_VECTOR = np.array([REGIONAL_MIX[c] for c in CROPS])
MIX_TOL = 0.02  # ± tolerance band per crop share, in absolute proportion

# Regional total cultivated area band (ha) and best point estimate.
TOTAL_AREA_BAND = (5200.0, 5500.0)
TOTAL_AREA_POINT = 5269.0
TOTAL_LANDMASS_HA = 21006.71  # verified sum of all 29 village areas
BASELINE_FRAC = TOTAL_AREA_POINT / TOTAL_LANDMASS_HA  # ~0.2508

# Per-village cap: total cultivated ha must never exceed ALPHA_CAP * area_ha
# (V7 lesson). Sweep 0.35-0.40; never disable.
ALPHA_CAP = 0.38

N_VILLAGES = 29
DATES = ["jun06", "jun19", "aug14", "oct13"]

# SAR coverage facts from the legacy analysis.
ZERO_COVERAGE_IDS = [1, 12, 25, 27]        # Manpura, Kotna, Pilol, Alindra
LOW_COVERAGE_IDS = [3, 5, 11]              # Sankhyad ~0.4%, Khanpur ~1.2%, Chhani ~0.6%

# Feature-table schema (this repo owns the schema; legacy CSV columns must be
# renamed to match — see data/README.md).
FEATURE_COLUMNS = (
    [f"mean_{d}" for d in DATES]
    + [f"std_{d}" for d in DATES]
    + [f"coverage_{d}" for d in DATES]
    + [
        "flood_frac_avg",
        "traj_slope",
        "traj_range",
        "traj_curvature",
        "delta_aug14_jun19",
        "delta_oct13_aug14",
    ]
)
REQUIRED_COLUMNS = ["village_id", "village_name", "area_ha"] + FEATURE_COLUMNS

# Subset actually fed to the Route C head (keep small: N=29, no labels).
MODEL_FEATURES = [
    "mean_jun06",
    "mean_jun19",
    "mean_aug14",
    "mean_oct13",
    "delta_aug14_jun19",
    "delta_oct13_aug14",
    "flood_frac_avg",
    "traj_slope",
    "traj_range",
    "traj_curvature",
]
