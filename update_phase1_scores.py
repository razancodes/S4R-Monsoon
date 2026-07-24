"""Update probe_results.csv with Phase 1 scores and compute true values."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from s4r.probing.delta_probe import compute_true_error

probe_csv = ROOT / "experiments" / "probing" / "probe_results.csv"
baseline_csv = ROOT / "experiments" / "runs" / "submission_physical_prior_v1.csv"

df = pd.read_csv(probe_csv)
baseline = pd.read_csv(baseline_csv)

# New scores from Phase 1
new_scores = {
    13: 1345.605,   # V1 Rice (delta=20)
    14: 1338.452,   # V1 Cotton (delta=20)
    15: 1345.645,   # V1 Maize (delta=20)
    16: 1345.863,   # V1 Bajra (delta=20)
    17: 1346.372,   # V1 Groundnut (delta=20)
    18: 1373.646,   # V12 Rice (delta=60)
    19: 1373.615,   # V12 Maize (delta=60)
    20: 1374.599,   # V12 Bajra (delta=60)
}

# Rows are 0-indexed, probes 13-20 are rows 12-19
for probe_num, mse_score in new_scores.items():
    row_idx = probe_num - 1  # 1-indexed probe -> 0-indexed row
    df.at[row_idx, 'perturbed_mse'] = mse_score

    vid = int(df.at[row_idx, 'village_id'])
    crop = df.at[row_idx, 'crop_name']
    delta = float(df.at[row_idx, 'delta_used'])
    baseline_mse = float(df.at[row_idx, 'baseline_mse'])

    true_error = compute_true_error(baseline_mse, mse_score, delta, n=145)
    pred_val = float(baseline[baseline['ID'] == vid][f'{crop}_ha'].iloc[0])
    true_val = max(0.0, pred_val + true_error)

    df.at[row_idx, 'computed_true_error'] = true_error
    df.at[row_idx, 'corrected_value'] = true_val

    print(f"Probe {probe_num}: V{vid} {crop:>10s} | delta={delta} | MSE={mse_score} | error={true_error:+.2f} | pred={pred_val:.2f} -> true={true_val:.2f}")

df.to_csv(probe_csv, index=False)
print(f"\nUpdated {probe_csv}")
print(f"Total scored probes: {df['perturbed_mse'].notna().sum()}")
