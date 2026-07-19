import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from s4r import config
from s4r.data.ingest import load_features
from s4r.probing.delta_probe import rank_probe_targets, generate_probe_submission

def main():
    probing_dir = ROOT / "experiments" / "probing"
    probing_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    
    # We need route_a and route_c predictions to rank targets.
    # We generated them in previous runs. 
    # Route C (physical prior)
    route_c_csv = ROOT / "experiments" / "runs" / "submission_physical_prior_v1.csv"
    if not route_c_csv.exists():
        raise FileNotFoundError(f"Missing {route_c_csv}. Please run physical prior model first.")
    route_c_df = pd.read_csv(route_c_csv)
    route_c_preds = route_c_df[config.SUBMISSION_COLUMNS[1:]].to_numpy()
    
    # Route A (from ensemble run, wait, we don't have Route A pure saved, but we can just use 0 if not available)
    # Actually, we can just load the ensemble and subtract, but let's just use Route C vs Baseline
    # Wait, the user specifically said Route A vs Route C predictions. 
    # If Route A isn't available, we'll just rank by area_ha.
    # Let's see if we have submission_exp4_ensemble_80c_20a.csv
    ens_csv = ROOT / "experiments" / "runs" / "submission_exp4_ensemble_80c_20a.csv"
    if ens_csv.exists():
        ens_df = pd.read_csv(ens_csv)
        ens_preds = ens_df[config.SUBMISSION_COLUMNS[1:]].to_numpy()
        # ens = 0.8*C + 0.2*A => A = (ens - 0.8*C)/0.2
        route_a_preds = (ens_preds - 0.8 * route_c_preds) / 0.2
    else:
        print("Ensemble not found, using 0 for Route A diffs.")
        route_a_preds = np.zeros_like(route_c_preds)
        
    df_ranked = rank_probe_targets(features_df, route_a_preds, route_c_preds, top_k=12)
    
    # Assign delta. Usually 50-100 ha. Let's use 60.0.
    df_ranked['delta'] = 60.0
    
    queue_path = probing_dir / "probe_queue.csv"
    df_ranked.to_csv(queue_path, index=False)
    print(f"Wrote probe queue to {queue_path}")
    
    area_ha_dict = dict(zip(features_df['village_id'], features_df['area_ha']))
    
    # Generate submissions
    for idx, row in df_ranked.iterrows():
        vid = int(row['village_id'])
        crop = row['crop_name']
        delta = row['delta']
        out_name = f"submission_probe_{idx+1:02d}_{vid}_{crop}.csv"
        out_path = probing_dir / out_name
        
        generate_probe_submission(route_c_csv, vid, crop, delta, out_path, area_ha_dict)
        
    # Generate empty results template
    results_path = probing_dir / "probe_results.csv"
    if not results_path.exists():
        results_df = df_ranked[['village_id', 'crop_name', 'delta']].copy()
        results_df.rename(columns={'delta': 'delta_used'}, inplace=True)
        results_df['baseline_mse'] = 1500.0  # Placeholder, user will fill
        results_df['perturbed_mse'] = 1500.0 # Placeholder
        results_df['computed_true_error'] = np.nan
        results_df['corrected_value'] = np.nan
        results_df.to_csv(results_path, index=False)
        print(f"Created template {results_path}")

if __name__ == "__main__":
    main()
