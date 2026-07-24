# S4R Monsoon Crop Yield Estimation

## Overview

This repository contains the end-to-end pipeline for the S4R Monsoon Crop Yield challenge. The objective of this project is to accurately estimate crop yields across multiple regions utilizing Sentinel-1 Synthetic Aperture Radar (SAR) imagery. 

Our solution introduces a novel framework termed **Foundation-Model Anchored Constrained Optimization**. Instead of relying purely on unconstrained black-box machine learning models, our pipeline heavily utilizes domain-specific agronomic physical priors and the zero-shot capabilities of modern earth observation foundation models.

## Methodology

Our approach is built upon five core pillars, ensuring that our predictions are physically grounded, mathematically rigorous, and strictly adhere to the competition constraints.

### 1. Agronomic Physical Priors
At the core of our solution is a physical prior model that maps specific tabular SAR signatures directly to crop phenology. Synthetic Aperture Radar backscatter is highly sensitive to canopy structure, soil moisture, and standing water. By leveraging these physical interactions, we constructed specific features that directly map signals to crops. For example, the `flood_frac_avg` metric (an indicator of persistent standing water) is physically correlated with Rice cultivation, while specific temporal backscatter signatures (`mean_oct13`) map strongly to Cotton. This ensures our baseline predictions are backed by actual physical interactions on the ground rather than spurious statistical correlations.

### 2. Confidence-Weighted Fallback
Raw satellite imagery is prone to coverage gaps and missing passes. We implemented a `coverage_confidence` metric based directly on the completeness of the satellite pass coverage. Villages with dense, high-quality satellite passes receive a high confidence score, while villages with significant observational gaps are penalized. This metric allows us to gracefully manage predictions and identify regions where the physical prior model might be untrustworthy due to data sparsity.

### 3. Foundation Model Anchoring (OlmoEarth-1B)
For the most difficult villages in the dataset (e.g., those with near-zero coverage or highly anomalous physical priors), standard localized modeling fails. To handle these highest-uncertainty regions, we utilized raw Sentinel-1 stacks and ran zero-shot inference using the **OlmoEarth-1B foundation model**. This generated highly accurate pseudo-labels for a specific subset of the most challenging regions. These foundation model outputs act as immovable "anchors" in our optimization pipeline, stabilizing the global distribution of crop yields.

### 4. Global Bias Correction
By rigorously analyzing the discrepancy between our localized physical prior baseline and the highly accurate OlmoEarth-1B anchors, we identified systematic baseline biases. We apply a targeted global bias correction to all unanchored villages based on these findings, scaling specific crops to match the true physical distribution observed by the foundation model.

### 5. Strict Constrained Optimization
The competition enforces strict physical constraints: an alpha cap of 0.38 per village and a grand total area band between 5200 and 5500 hectares. Our custom rebalancing algorithm calculates the required area deficit or excess needed to satisfy these constraints globally. It then strictly sorts the unanchored villages by their coverage confidence. The optimizer iterates through the lowest-confidence villages, safely scaling their predicted areas up to the maximum allowable alpha cap to absorb any deficits. This mathematically guarantees that all constraints are met while perfectly preserving our high-confidence physical priors and our foundation model anchors.

## Repository Structure

* `data/`: Contains raw, processed, and weak label data. Pre-processed tabular features are cached here.
* `src/s4r/`: Core Python package containing the data ingestion logic, feature generation, coverage metrics, and optimization algorithms.
* `experiments/`: Contains experiment tracking, pipeline runners, and the generated submission files.
* `build_final_submission.py`: The primary execution script that runs the constrained optimization and generates the final CSV submission.

## Execution

To execute the pipeline and generate the final submission:

1. Clone the repository and navigate to the project root.
2. Install the package locally:
   ```bash
   pip install -e .
   ```
3. Run the final submission script:
   ```bash
   python build_final_submission.py
   ```

The final output will be successfully generated at `experiments/runs/submission_final_phase3.csv`.
