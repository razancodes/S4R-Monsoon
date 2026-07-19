import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from s4r import config
from s4r.submission.writer import validate_predictions
from s4r.probing.delta_probe import compute_true_error
from s4r.data.ingest import load_features

def main():
    probing_dir = ROOT / "experiments" / "probing"
    results_path = probing_dir / "probe_results.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}.")
        
    df_results = pd.read_csv(results_path)
    
    baseline_csv = ROOT / "experiments" / "runs" / "submission_physical_prior_v1.csv"
    df_sub = pd.read_csv(baseline_csv)
    
    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    area_ha_dict = dict(zip(features_df['village_id'], features_df['area_ha']))
    area = np.array([area_ha_dict[vid] for vid in df_sub['ID']])
    
    pred = df_sub[config.SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    
    diagnostics = []
    
    for idx, row in df_results.iterrows():
        vid = int(row['village_id'])
        crop = row['crop_name']
        
        # Compute true error if perturbed_mse is filled
        if pd.isna(row['perturbed_mse']):
            continue
            
        baseline_mse = row['baseline_mse']
        perturbed_mse = row['perturbed_mse']
        delta = row['delta_used']
        
        true_error = compute_true_error(baseline_mse, perturbed_mse, delta, n=145)
        df_results.at[idx, 'computed_true_error'] = true_error
        
        # Locate row in pred
        row_idx = df_sub.index[df_sub['ID'] == vid].tolist()[0]
        c_idx = config.CROPS.index(crop)
        
        orig_val = pred[row_idx, c_idx]
        corrected_val = orig_val + true_error
        
        # Non-negativity check
        corrected_val = max(0.0, corrected_val)
        df_results.at[idx, 'corrected_value'] = corrected_val
        
        diff = corrected_val - orig_val
        overestimate = diff < 0
        
        diagnostics.append(f"Village {vid} {crop}: Original {orig_val:.2f}, Corrected {corrected_val:.2f} (Error: {true_error:+.2f}). " 
                           f"Model was an {'OVERESTIMATE' if overestimate else 'UNDERESTIMATE'} by {abs(true_error):.2f} ha.")
        
        pred[row_idx, c_idx] = corrected_val
        
        # Re-enforce cap if necessary
        totals = pred.sum(axis=1)
        cap = config.ALPHA_CAP * area
        over = totals > cap
        if over[row_idx]:
            print(f"WARNING: Corrected value {corrected_val} for {crop} in village {vid} exceeded alpha cap. Re-normalizing.")
            scale = cap[row_idx] / totals[row_idx]
            pred[row_idx] *= scale
            
    # Save back computed errors
    df_results.to_csv(results_path, index=False)
    
    # Save corrected submission
    for c_idx, crop in enumerate(config.CROPS):
        df_sub[f"{crop}_ha"] = pred[:, c_idx]
        
    out_path = probing_dir / "submission_corrected_v1.csv"
    df_sub.to_csv(out_path, index=False)
    
    violations = validate_predictions(pred, area, config.ALPHA_CAP)
    if violations:
        print("CRITICAL WARNING: Corrected submission violates invariants:", violations)
    else:
        print(f"Corrected submission generated and passed all invariants: {out_path}")
        
    print("\n--- Diagnostic Report ---")
    for d in diagnostics:
        print(d)

if __name__ == "__main__":
    main()
