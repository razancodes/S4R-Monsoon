"""Synthetic 29-village fixture for testing the pipeline without real data.

Values are random but structurally faithful: real village IDs/names where known,
the documented coverage gaps, and area_ha summing to the verified landmass total.
NEVER submit predictions derived from this fixture (the submission writer
enforces this via the is_synthetic flag).
"""

import numpy as np
import pandas as pd

from s4r import config

# Known real villages (ID -> name) from the legacy analysis.
KNOWN_NAMES = {
    1: "Manpura",
    3: "Sankhyad",
    5: "Khanpur",
    11: "Chhani",
    12: "Kotna",
    25: "Pilol",
    27: "Alindra",
    8: "Koyali",
    14: "Angadh",
    16: "Asoj",
}

# Mean coverage for partially covered villages.
PARTIAL_COVERAGE = {3: 0.004, 5: 0.012, 11: 0.006}


def make_synthetic_features(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.arange(1, config.N_VILLAGES + 1)

    raw_area = rng.uniform(200.0, 1500.0, size=config.N_VILLAGES)
    area_ha = raw_area * (config.TOTAL_LANDMASS_HA / raw_area.sum())

    rows = []
    for i, vid in enumerate(ids):
        name = KNOWN_NAMES.get(int(vid), f"Village_{vid}")
        row: dict = {
            "village_id": int(vid),
            "village_name": name,
            "area_ha": area_ha[i],
            "is_synthetic": True,
        }
        if vid in config.ZERO_COVERAGE_IDS:
            cov_per_date = np.zeros(4)
        elif int(vid) in PARTIAL_COVERAGE:
            cov_per_date = np.clip(
                rng.normal(PARTIAL_COVERAGE[int(vid)], 0.001, size=4), 0.0, 1.0
            )
        else:
            cov_per_date = rng.uniform(0.6, 1.0, size=4)

        # Plausible X-band backscatter (dB) trajectory with monsoon dip.
        base = rng.normal(-12.0, 2.0)
        means = base + np.array([0.0, rng.normal(-1, 0.5), rng.normal(-3, 1.0), rng.normal(1, 0.5)])
        for j, d in enumerate(config.DATES):
            covered = cov_per_date[j] > 0
            row[f"coverage_{d}"] = float(cov_per_date[j])
            row[f"mean_{d}"] = float(means[j]) if covered else np.nan
            row[f"std_{d}"] = float(abs(rng.normal(2.0, 0.5))) if covered else np.nan

        any_cov = cov_per_date.max() > 0
        row["flood_frac_avg"] = float(rng.uniform(0, 0.4)) if any_cov else np.nan
        row["traj_slope"] = float(rng.normal(0, 0.5)) if any_cov else np.nan
        row["traj_range"] = float(abs(rng.normal(4, 1))) if any_cov else np.nan
        row["traj_curvature"] = float(rng.normal(0, 1)) if any_cov else np.nan
        row["delta_aug14_jun19"] = float(means[2] - means[1]) if any_cov else np.nan
        row["delta_oct13_aug14"] = float(means[3] - means[2]) if any_cov else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[["village_id", "village_name", "area_ha"] + config.FEATURE_COLUMNS + ["is_synthetic"]]
