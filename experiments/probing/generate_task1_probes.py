import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from s4r.probing.delta_probe import generate_probe_submission
from s4r.data.ingest import load_features

def main():
    probing_dir = ROOT / "experiments" / "probing"
    
    # Load features
    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    area_ha_dict = dict(zip(features_df['village_id'], features_df['area_ha']))
    
    route_c_csv = ROOT / "experiments" / "runs" / "submission_physical_prior_v1.csv"
    
    # Task 1 targets: Medium villages with coverage
    targets = [
        (15, "Groundnut", 40.0),
        (15, "Rice", 40.0),
        (18, "Groundnut", 40.0),
        (18, "Rice", 40.0)
    ]
    
    results_path = probing_dir / "probe_results.csv"
    df_results = pd.read_csv(results_path)
    
    new_rows = []
    
    start_idx = 13
    for i, (vid, crop, delta) in enumerate(targets):
        out_name = f"submission_probe_{start_idx+i:02d}_{vid}_{crop}.csv"
        out_path = probing_dir / out_name
        
        generate_probe_submission(route_c_csv, vid, crop, delta, out_path, area_ha_dict)
        
        new_rows.append({
            'village_id': vid,
            'crop_name': crop,
            'delta_used': delta,
            'baseline_mse': 1343.499,
            'perturbed_mse': np.nan,
            'computed_true_error': np.nan,
            'corrected_value': np.nan
        })
        
    df_new = pd.DataFrame(new_rows)
    df_results = pd.concat([df_results, df_new], ignore_index=True)
    df_results.to_csv(results_path, index=False)
    print("Added new targets to probe_results.csv")

if __name__ == "__main__":
    main()
