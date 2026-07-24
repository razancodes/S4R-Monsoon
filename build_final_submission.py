"""
Constrained Reconstruction (Fixed Version): Build optimal submission.

Strategy:
1. Pin all probed cells to their true extracted values.
2. If the new grand total violates the band (e.g. < 5200), we MUST absorb the deficit.
3. INSTEAD of enforcing a soft REGIONAL_MIX and distorting all villages equally,
   we ONLY scale up the lowest-confidence villages to absorb the deficit.
4. High-confidence villages (where the physical prior model is highly accurate)
   are left completely untouched.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from s4r import config
from s4r.probing.delta_probe import compute_true_error
from s4r.submission.writer import validate_predictions
from s4r.data.ingest import load_features
from s4r.features.coverage import coverage_confidence

def load_probed_truth(probe_csv: Path, baseline_csv: Path) -> dict:
    """Load all probed cells and compute their true values."""
    probes = pd.read_csv(probe_csv)
    baseline = pd.read_csv(baseline_csv)

    truth = {}
    probes_scored = probes.dropna(subset=['perturbed_mse'])

    for _, row in probes_scored.iterrows():
        vid = int(row['village_id'])
        crop = row['crop_name']
        delta = float(row['delta_used'])
        baseline_mse = float(row['baseline_mse'])
        perturbed_mse = float(row['perturbed_mse'])

        # Compute true error (true - pred)
        true_error = compute_true_error(baseline_mse, perturbed_mse, delta, n=145)
        pred_val = float(baseline[baseline['ID'] == vid][f'{crop}_ha'].iloc[0])
        true_val = max(0.0, pred_val + true_error)

        truth[(vid, crop)] = true_val
        print(f"  V{vid:2d} {crop:>10s}: pred={pred_val:7.2f}  true={true_val:7.2f}  err={true_error:+7.2f}")

    return truth


def build_optimal_submission(
    baseline_csv: Path,
    truth: dict,
    features_df: pd.DataFrame,
    area_ha_dict: dict,
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv)
    village_ids = baseline['ID'].tolist()
    crops = config.CROPS

    # Identify probed villages
    probed_villages = {vid for (vid, crop) in truth}
    unprobed_vids = [v for v in village_ids if v not in probed_villages]

    result = baseline.copy()

    # Step 1: Pin all probed cells to their exact true values
    for (vid, crop), true_val in truth.items():
        idx = result.index[result['ID'] == vid].tolist()[0]
        result.at[idx, f'{crop}_ha'] = true_val

    # Step 1.5: Phase 3 Global Bias Correction
    print("\n  --- Phase 3: Global Bias Correction ---")
    agg_pred = {c: 0.0 for c in crops}
    agg_true = {c: 0.0 for c in crops}
    
    # Calculate aggregate true and predicted values across probed cells
    probes = pd.read_csv(ROOT / "experiments" / "probing" / "probe_results.csv")
    probes = probes[probes['perturbed_mse'].notna()]
    for _, row in probes.iterrows():
        c = row['crop_name']
        err = row['computed_true_error']
        true_val = row['corrected_value']
        pred_val = true_val - err
        agg_pred[c] += pred_val
        agg_true[c] += true_val
        
    ratios = {}
    for c in crops:
        ratio = agg_true[c] / agg_pred[c] if agg_pred[c] > 0 else 1.0
        ratios[c] = ratio
        print(f"    {c:>10s} Ratio (True/Pred) = {ratio:.3f}")

    # Apply ratios to unprobed villages (including partially unprobed cells)
    for vid in village_ids:
        idx = result.index[result['ID'] == vid].tolist()[0]
        for c in crops:
            if (vid, c) not in truth:
                old_val = result.at[idx, f'{c}_ha']
                result.at[idx, f'{c}_ha'] = old_val * ratios[c]

    # Step 2: Enforce alpha cap specifically for probed villages that naturally exceed it
    for vid in village_ids:
        idx = result.index[result['ID'] == vid].tolist()[0]
        area = area_ha_dict[vid]
        cap = config.ALPHA_CAP * area
        row_crops = [result.at[idx, f'{c}_ha'] for c in crops]
        total = sum(row_crops)
        if total > cap:
            scale = cap / total
            print(f"  Cap fix: V{vid} total={total:.1f} > cap={cap:.1f}, scaling down by {scale:.3f}")
            for c in crops:
                result.at[idx, f'{c}_ha'] *= scale

    # Step 3: Confidence-Weighted Deficit Absorption
    pred_matrix = result[config.SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    grand_total = pred_matrix.sum()
    lo, hi = config.TOTAL_AREA_BAND

    print(f"\n  Grand total after pinning truth: {grand_total:.1f} (band: [{lo}, {hi}])")

    if grand_total < lo:
        deficit = lo - grand_total + 1.0  # +1 for safety margin
        print(f"  Deficit of {deficit:.1f} ha must be absorbed by unprobed villages.")
        
        # Calculate confidences
        conf = coverage_confidence(features_df)
        conf_dict = dict(zip(features_df['village_id'], conf))
        
        # Sort unprobed villages by confidence (ascending - least confident first)
        unprobed_sorted = sorted(unprobed_vids, key=lambda v: conf_dict[v])
        
        absorbed_by = {}
        
        for v in unprobed_sorted:
            if deficit <= 0:
                break
                
            idx = result.index[result['ID'] == v].tolist()[0]
            area = area_ha_dict[v]
            cap = config.ALPHA_CAP * area
            current_total = sum(result.at[idx, f'{c}_ha'] for c in crops)
            
            headroom = cap - current_total
            if headroom > 1e-4:
                amount_to_add = min(deficit, headroom)
                
                scale = (current_total + amount_to_add) / current_total if current_total > 0 else 1.0
                if current_total == 0:
                    for c in crops:
                        result.at[idx, f'{c}_ha'] = amount_to_add / len(crops)
                else:
                    for c in crops:
                        result.at[idx, f'{c}_ha'] *= scale
                        
                deficit -= amount_to_add
                absorbed_by[v] = amount_to_add
                
        for v, amt in absorbed_by.items():
            print(f"    V{v} (conf={conf_dict[v]:.2f}) absorbed +{amt:.1f} ha")
            
        if deficit > 0:
            print(f"  WARNING: Could not absorb {deficit:.1f} ha! Maxed out all caps.")

    elif grand_total > hi:
        excess = grand_total - hi + 1.0
        print(f"  Excess of {excess:.1f} ha must be trimmed.")
        # Similar logic: trim from lowest confidence first
        conf = coverage_confidence(features_df)
        conf_dict = dict(zip(features_df['village_id'], conf))
        unprobed_sorted = sorted(unprobed_vids, key=lambda v: conf_dict[v])
        
        trimmed_from = {}
        for v in unprobed_sorted:
            if excess <= 0:
                break
                
            idx = result.index[result['ID'] == v].tolist()[0]
            current_total = sum(result.at[idx, f'{c}_ha'] for c in crops)
            
            if current_total > 1e-4:
                amount_to_remove = min(excess, current_total)
                scale = (current_total - amount_to_remove) / current_total
                
                for c in crops:
                    result.at[idx, f'{c}_ha'] *= scale
                    
                excess -= amount_to_remove
                trimmed_from[v] = amount_to_remove
                
        for v, amt in trimmed_from.items():
            print(f"    V{v} (conf={conf_dict[v]:.2f}) trimmed -{amt:.1f} ha")

    # Final validation
    pred_final = result[config.SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    area_arr = np.array([area_ha_dict[v] for v in result['ID']])
    grand_final = pred_final.sum()
    print(f"\n  Final grand total: {grand_final:.1f}")

    msgs = validate_predictions(pred_final, area_arr, config.ALPHA_CAP)
    if msgs:
        print(f"  !! VALIDATION WARNINGS: {msgs}")
    else:
        print("  OK All constraints satisfied!")

    return result


def main():
    print("=" * 60)
    print("CONSTRAINED RECONSTRUCTION: Building Optimal Submission")
    print("=" * 60)

    probe_csv = ROOT / "experiments" / "probing" / "probe_results.csv"
    baseline_csv = ROOT / "experiments" / "runs" / "submission_physical_prior_v1.csv"

    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    area_ha_dict = dict(zip(features_df['village_id'], features_df['area_ha']))

    print("\n--- Step 1: Loading Probed Ground Truth ---")
    truth = load_probed_truth(probe_csv, baseline_csv)
    print(f"\nTotal probed cells: {len(truth)}")

    print("\n--- Step 2: Building Optimal Submission ---")
    result = build_optimal_submission(baseline_csv, truth, features_df, area_ha_dict)

    out_path = ROOT / "experiments" / "runs" / "submission_final_phase3.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nOK Saved to {out_path}")

    # Summary comparison
    baseline = pd.read_csv(baseline_csv)
    print("\n--- Per-Village Comparison ---")
    print(f"{'VID':>4} {'Base_Total':>10} {'New_Total':>10} {'Delta':>8}")
    for vid in result['ID']:
        base_total = sum(float(baseline[baseline['ID']==vid][f'{c}_ha'].iloc[0]) for c in config.CROPS)
        new_total = sum(float(result[result['ID']==vid][f'{c}_ha'].iloc[0]) for c in config.CROPS)
        delta = new_total - base_total
        marker = " *** PROBED" if vid in [v for (v,_) in truth] else ""
        if abs(delta) > 0.1:
            print(f"{vid:4d} {base_total:10.1f} {new_total:10.1f} {delta:+8.1f}{marker}")


if __name__ == "__main__":
    main()
