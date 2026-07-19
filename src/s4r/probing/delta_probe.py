import pandas as pd
import numpy as np
from pathlib import Path
from s4r import config
from s4r.submission.writer import validate_predictions

def compute_true_error(baseline_mse: float, perturbed_mse: float, delta: float, n: int = 145) -> float:
    """
    Computes the true error `e_i = true_i - pred_i` using the Kaggle LB delta method.
    n is the number of scored cells. For 29 villages x 5 crops, n = 145.
    """
    return (delta / 2.0) - (n * (perturbed_mse - baseline_mse)) / (2.0 * delta)

def generate_probe_submission(baseline_csv: str | Path, village_id: int, crop_name: str, delta: float, output_path: str | Path, area_ha_dict: dict) -> None:
    """
    Takes the current best submission, adds `delta` to exactly one (village, crop) cell.
    Re-normalizes only if required to maintain alpha cap, non-negativity.
    """
    df = pd.read_csv(baseline_csv)
    
    # Locate row
    idx = df.index[df['ID'] == village_id].tolist()
    if not idx:
        raise ValueError(f"Village ID {village_id} not found in submission.")
    row_idx = idx[0]
    
    col = f"{crop_name}_ha"
    if col not in df.columns:
        raise ValueError(f"Crop column {col} not found.")
        
    orig_val = df.at[row_idx, col]
    new_val = orig_val + delta
    
    # Non-negativity
    if new_val < 0:
        print(f"WARNING: Perturbation {delta} made {col} < 0. Clipping to 0. This contaminates the probe.")
        new_val = 0.0
        
    df.at[row_idx, col] = new_val
    
    # Extract prediction matrix to check invariants
    pred = df[config.SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    
    # Re-enforce cap if necessary
    area = np.array([area_ha_dict[vid] for vid in df['ID']])
    cap = config.ALPHA_CAP * area
    totals = pred.sum(axis=1)
    over = totals > cap
    if over[row_idx]:
        print(f"WARNING: Perturbation {delta} exceeded alpha cap for village {village_id}. Re-normalizing. This contaminates the probe.")
        scale = cap[row_idx] / totals[row_idx]
        pred[row_idx] *= scale
        
        # update dataframe
        for c_idx, crop in enumerate(config.CROPS):
            df.at[row_idx, f"{crop}_ha"] = pred[row_idx, c_idx]
            
    # Save
    df.to_csv(output_path, index=False)
    print(f"Probe generated: {output_path} (Village: {village_id}, Crop: {crop_name}, Delta: {delta})")


def rank_probe_targets(feature_table: pd.DataFrame, route_a_preds: np.ndarray, route_c_preds: np.ndarray, top_k: int = 12) -> pd.DataFrame:
    """
    Returns a ranked list of (village_id, crop_name) tuples prioritized by:
    1. area_ha (descending)
    2. |route_a_pred - route_c_pred| (descending)
    3. coverage_fraction == 0 (automatic top priority)
    """
    records = []
    
    # Identify zero coverage villages
    zero_cov_vids = set(config.ZERO_COVERAGE_IDS)
    
    for i, row in feature_table.iterrows():
        vid = int(row['village_id'])
        area = row['area_ha']
        is_zero_cov = vid in zero_cov_vids
        
        for c_idx, crop in enumerate(config.CROPS):
            diff = abs(route_a_preds[i, c_idx] - route_c_preds[i, c_idx])
            records.append({
                'village_id': vid,
                'crop_name': crop,
                'area_ha': area,
                'model_diff': diff,
                'is_zero_cov': is_zero_cov
            })
            
    df = pd.DataFrame(records)
    
    # Sort: is_zero_cov (True first), then area_ha (descending), then model_diff (descending)
    df.sort_values(by=['is_zero_cov', 'area_ha', 'model_diff'], ascending=[False, False, False], inplace=True)
    
    return df.head(top_k).reset_index(drop=True)
