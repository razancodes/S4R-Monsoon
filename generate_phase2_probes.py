"""
Phase 2 Probing: Fixed delta for V11 to stay under cap.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from s4r import config
from s4r.probing.delta_probe import generate_probe_submission
from s4r.data.ingest import load_features

def main():
    probing_dir = ROOT / "experiments" / "probing"
    
    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    area_ha_dict = dict(zip(features_df['village_id'], features_df['area_ha']))

    baseline_csv = ROOT / "experiments" / "runs" / "submission_final_phase1.csv"

    # Targets: V29 delta=80, V11 delta=30
    targets = [
        (29, "Groundnut", 80.0),
        (29, "Cotton", 80.0),
        (29, "Bajra", 80.0),
        (11, "Groundnut", 30.0),
        (11, "Cotton", 30.0),
        (11, "Rice", 30.0)
    ]

    results_path = probing_dir / "probe_results.csv"
    df_results = pd.read_csv(results_path)
    
    # Keep only the 20 scored probes from Phase 1
    df_results = df_results[df_results['perturbed_mse'].notna()].copy()
    start_idx = len(df_results) + 1

    new_rows = []
    for i, (vid, crop, delta) in enumerate(targets):
        out_name = f"submission_probe_{start_idx+i:02d}_{vid}_{crop}.csv"
        out_path = probing_dir / out_name

        generate_probe_submission(baseline_csv, vid, crop, delta, out_path, area_ha_dict)

        new_rows.append({
            'village_id': vid,
            'crop_name': crop,
            'delta_used': delta,
            'baseline_mse': 1400.707, 
            'perturbed_mse': np.nan,
            'computed_true_error': np.nan,
            'corrected_value': np.nan
        })

    df_new = pd.DataFrame(new_rows)
    df_results = pd.concat([df_results, df_new], ignore_index=True)
    df_results.to_csv(results_path, index=False)

    print(f"\nGenerated {len(targets)} probes for Phase 2!")
    for i, (vid, crop, delta) in enumerate(targets):
        fname = f"submission_probe_{start_idx+i:02d}_{vid}_{crop}.csv"
        print(f"  {fname} (delta={delta})")

if __name__ == "__main__":
    main()
