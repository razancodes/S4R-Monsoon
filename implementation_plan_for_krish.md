# Bias Correction Pipeline Analysis & Plan

The first-principles approach you've outlined safely navigates the huge risk of overfitting the public leaderboard when dealing with sparse test data (N=29). By using Empirical Bayes shrinkage and a post-hoc correction, we use the extracted bias dynamically without allowing L-BFGS to overfit a 66-parameter head to 12 exact points.

Here is the detailed analysis and step-by-step execution plan for this final sprint:

## Task 1: Expand the Probe Sample
**Analysis**: The first 12 probes were run on villages 25, 27, and 12. These are all zero-coverage villages (`is_zero_cov=True`), meaning the physical prior model relied heavily on regional constraints and `W_s` rather than village-specific SAR signals (which were missing/zeroed). We need to verify if the massive over/underestimation bias holds for villages with **actual SAR coverage**!
**Plan**: I will generate 4 new probes targeting "medium" villages that have strong SAR coverage (e.g., `coverage_fraction > 0`). Candidates: Village 15 (491 ha) or Village 18 (703 ha). I will queue 4 probes (e.g., Groundnut/Rice across 2 medium villages) and wait for you to score them.

## Task 2: Confidence-Shrunk Bias Estimates
**Analysis**: Empirical Bayes shrinkage `b_c_applied = b_c * (k / (k + k0))` prevents us from overreacting to noise. 
*Correction*: `k` is the number of **independent villages** probed, not individual crop-cells. With the original 3 villages and 2 new medium villages, `k = 5`. 
**Sensitivity Table for k=5**:
- `k0 = 4`: multiplier = 5 / 9 ≈ **0.55**
- `k0 = 6`: multiplier = 5 / 11 ≈ **0.45**
- `k0 = 8`: multiplier = 5 / 13 ≈ **0.38**
- `k0 = 10`: multiplier = 5 / 15 ≈ **0.33**

**Plan**: I will build the math into our script to calculate `b_c` from the combined pool of probe results. We will likely use `k0 = 6` (0.45x shrinkage).

## Task 3: Apply Post-Hoc Correction
**Analysis**: This is the crucial step. Instead of re-running the L-BFGS optimizer with a strong loss term, we directly shift the existing predictions:
`Pred_new(v, c) = Pred_prior(v, c) + b_c_applied`
**Constraint Handling**:
Shifting predictions up/down breaks the regional grand-total band (`[5200, 5500]`) and the simplex invariants. We will absorb this shock *only* in low-confidence villages, sorting villages by `coverage_confidence`. The highest confidence villages will retain the exact bias-corrected prediction. 
*Correction*: To prevent the mechanical satisfaction of constraints from pushing low-confidence villages into unrealistic extremes, we will apply a **bounded-absorption cap** (e.g., ±15-20% shift relative to their prior baseline). If a low-confidence village hits this bound, the remainder of the constraint shock rolls over to the next-lowest-confidence village until fully absorbed.
**Prior Nudge**: If the expanded probes show that, for example, Rice is consistently underestimated specifically in high-flood villages, we can optionally bump `W_s[Rice, flood]` manually. But the primary path is the post-hoc shrinkage.

## Task 4 & 5: Validation and Final Candidates
**Plan**: 
1. The correction script will run a strict pass through `s4r.submission.writer.validate_predictions()` to guarantee all constraints (alpha caps, non-negativity, total bands).
2. It will output a per-village delta report showing exactly how much each village shifted compared to the baseline.
3. It will generate `submission_bias_corrected_v1.csv` as our primary candidate, leaving `submission_physical_prior_v1.csv` completely untouched as the reserve candidate.
