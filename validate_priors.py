import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))

from s4r import config
from s4r.data.ingest import load_features, load_sample_submission, standardize_features
from s4r.features.coverage import coverage_confidence
from s4r.fallback.train import train, TrainConfig
from s4r.submission.writer import validate_predictions, write_submission
from s4r.weak_labels.ingest import load_weak_labels

def main():
    features_df = load_features(ROOT / "data" / "processed" / "village_features.csv")
    sample_df = load_sample_submission(Path(r"C:\Users\MRaza\.cache\kagglehub\competitions\anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge\Sample_submission_file.csv"))
    
    anchors_path = ROOT / "data" / "weak_labels" / "annotations.csv"
    anchors = load_weak_labels(str(anchors_path), features_df) if anchors_path.exists() else None
    
    X = standardize_features(features_df)
    area_ha = features_df["area_ha"].to_numpy()
    conf = coverage_confidence(features_df)
    
    # Run with physical priors
    cfg = TrainConfig(use_physical_prior=True, seed=42)
    runs_dir = ROOT / "experiments" / "runs"
    result = train(X, area_ha, conf, cfg, anchors=anchors, run_dir=str(runs_dir))
    
    pred = result["pred"]
    shares = pred / (pred.sum(axis=1, keepdims=True) + 1e-9)
    
    # Run WITHOUT physical priors to get old baseline for comparison
    cfg_old = TrainConfig(use_physical_prior=False, seed=42)
    result_old = train(X, area_ha, conf, cfg_old, anchors=anchors, run_dir=None)
    pred_old = result_old["pred"]
    shares_old = pred_old / (pred_old.sum(axis=1, keepdims=True) + 1e-9)
    
    # Check invariants
    violations = validate_predictions(pred, area_ha, config.ALPHA_CAP)
    print("Invariants violated:", violations if violations else "None, all invariants hold!")
    
    # Report table
    print("\n--- Validation Table: Rice Share vs Flood Fraction ---")
    print(f"{'VID':>4} | {'Area (ha)':>9} | {'Flood Frac':>10} | {'Old Rice Share':>14} | {'New Rice Share':>14}")
    
    for i in range(len(features_df)):
        vid = features_df["village_id"].iloc[i]
        area = features_df["area_ha"].iloc[i]
        flood = features_df["flood_frac_avg"].iloc[i]
        old_rice = shares_old[i, 0]
        new_rice = shares[i, 0]
        # Only print a few to avoid clutter, e.g. those with coverage (flood isn't NaN)
        if not np.isnan(flood):
            print(f"{vid:4.0f} | {area:9.1f} | {flood:10.4f} | {old_rice:14.4f} | {new_rice:14.4f}")
            
    print("\nVariance in new crop shares (std dev across villages):")
    print(np.std(shares, axis=0))
    print("Variance in old crop shares (std dev across villages):")
    print(np.std(shares_old, axis=0))
    
    out_path = runs_dir / "submission_physical_prior_v1.csv"
    write_submission(pred, features_df, sample_df, out_path, allow_synthetic=True)
    print(f"\nWrote physical prior submission to {out_path}")

if __name__ == "__main__":
    main()
