import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')
from s4r import config
from s4r.data.ingest import load_features

# Load current best submission
sub = pd.read_csv('experiments/runs/submission_physical_prior_v1.csv')
feat = load_features('data/processed/village_features.csv')

# Show predictions for ALL villages
print('=== Current Predictions (physical_prior_v1) ===')
print(f'{"VID":>4} {"Area":>8} {"Rice":>8} {"Cotton":>8} {"Maize":>8} {"Bajra":>8} {"GN":>8} {"Total":>8} {"ZeroCov":>7}')
for i, row in sub.iterrows():
    vid = int(row['ID'])
    is_zc = vid in config.ZERO_COVERAGE_IDS
    total = row['Rice_ha'] + row['Cotton_ha'] + row['Maize_ha'] + row['Bajra_ha'] + row['Groundnut_ha']
    area = float(feat[feat['village_id']==vid]['area_ha'].iloc[0])
    print(f'{vid:4d} {area:8.1f} {row["Rice_ha"]:8.1f} {row["Cotton_ha"]:8.1f} {row["Maize_ha"]:8.1f} {row["Bajra_ha"]:8.1f} {row["Groundnut_ha"]:8.1f} {total:8.1f} {str(is_zc):>7}')

# Show regional totals
print()
totals = {c: sub[f'{c}_ha'].sum() for c in config.CROPS}
grand = sum(totals.values())
print(f'Regional totals: {totals}')
print(f'Grand total: {grand:.1f} (band: {config.TOTAL_AREA_BAND})')
mix_str = {c: f"{v/grand*100:.1f}%" for c,v in totals.items()}
target_str = {c: f"{v*100:.1f}%" for c,v in config.REGIONAL_MIX.items()}
print(f'Regional mix: {mix_str}')
print(f'Target mix: {target_str}')

# Now show the probed ground truth vs predictions
print('\n=== PROBED GROUND TRUTH vs PREDICTIONS ===')
probes = pd.read_csv('experiments/probing/probe_results.csv')
probes_valid = probes.dropna(subset=['computed_true_error'])
print(f'{"VID":>4} {"Crop":>12} {"Predicted":>10} {"True":>10} {"Error":>10} {"SqErr":>10}')
total_sq = 0
for _, p in probes_valid.iterrows():
    vid = int(p['village_id'])
    crop = p['crop_name']
    true_val = p['corrected_value']
    pred_val = float(sub[sub['ID']==vid][f'{crop}_ha'].iloc[0])
    err = true_val - pred_val
    sq = err**2
    total_sq += sq
    print(f'{vid:4d} {crop:>12} {pred_val:10.1f} {true_val:10.1f} {err:+10.1f} {sq:10.1f}')

print(f'\nTotal squared error from {len(probes_valid)} probed cells: {total_sq:.0f}')
print(f'MSE contribution from probed cells: {total_sq/145:.1f}')
print(f'Remaining 133 cells MSE contribution: {1343.5 - total_sq/145:.1f}')
print(f'Average sq error per unprobed cell: {(145*1343.5 - total_sq)/133:.0f} -> avg |error| = {np.sqrt((145*1343.5 - total_sq)/133):.1f} ha')
