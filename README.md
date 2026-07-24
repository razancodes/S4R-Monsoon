# AISEHACK Monsoon Crop Yield Estimation

## Overview

This repository contains the end-to-end pipeline for the S4R Monsoon Crop Yield challenge. The objective of this project is to accurately estimate crop yields across multiple regions utilizing Synthetic Aperture Radar (SAR) imagery. 

Our solution introduces a novel framework termed **Foundation-Model Anchored Constrained Optimization**. Instead of relying purely on unconstrained black-box machine learning models, our pipeline heavily utilizes domain-specific agronomic physical priors and the zero-shot capabilities of modern earth observation foundation models, completely built across a multi-phase architecture.

## Pipeline Architecture & Methodology

Our pipeline is broken down into five deeply integrated phases, ensuring that predictions are physically grounded, robust to missing data, and mathematically bound by competition constraints.

### Phase 1: Agronomic Physical Priors & Confidence Scoring

At the core of our baseline is a physical prior model that maps specific tabular SAR signatures directly to crop phenology. Synthetic Aperture Radar backscatter is highly sensitive to canopy structure, soil moisture, and standing water.
* **Physical Mapping**: By leveraging these physical interactions, we constructed specific features that directly map signals to crops. For example, the `flood_frac_avg` metric (an indicator of persistent standing water) is physically correlated with Rice cultivation, while specific temporal backscatter signatures (`mean_oct13`) map strongly to Cotton.
* **Confidence-Weighted Fallback**: Raw satellite imagery is prone to coverage gaps. We implemented a `coverage_confidence` metric based directly on the completeness of the satellite pass coverage. Villages with dense passes receive a high confidence score, while villages with observational gaps are penalized. This metric elegantly identifies regions where the physical prior model is untrustworthy due to data sparsity.

### Phase 2: Auxiliary Sentinel-1 Data Fetch

Because the core dataset can have severe observational gaps or calibration anomalies, we rely on a completely separate, auxiliary dataset to establish ground truth for high-uncertainty regions.
* **Planetary Computer Integration**: The pipeline queries the Microsoft Planetary Computer STAC API to fetch raw Sentinel-1 Radiometric Terrain Corrected (RTC) imagery for the exact bounding boxes of the 29 competition villages.
* **Phenological Alignment**: It specifically targets four acquisition dates (June 06, June 19, August 14, October 13) that perfectly mirror the critical phenological milestones of the monsoon crop season.
* **Offline Caching**: The fetched `(H, W, T, 2)` SAR stacks (containing VV and VH polarizations) are heavily cropped using the village shapefiles and cached into an `.npz` file, strictly enforcing the rule that no external network calls are made during inference.

### Phase 3: Foundation Model Anchoring (OlmoEarth-V1 Base)

Rather than training a model from scratch on limited data, the pipeline uses the pre-trained weights of the **OlmoEarth-V1 Base Foundation Model** to extract robust physical representations of the land surface.
* **Zero-Shot Encoding**: The Sentinel-1 dB stacks are normalized and passed through the completely frozen OlmoEarth encoder. 
* **Patch Pooling**: The encoder generates high-dimensional embeddings for 80-meter spatial patches (tokens). These tokens are pooled across time and band-sets to create a highly dense representation of the physical surface for every valid pixel.

### Phase 4: Unsupervised Pseudo-Label Generation

To identify which tokens actually represent cropland without using any competition ground truth, the pipeline uses an elegant unsupervised clustering heuristic:
* **Global K-Means Clustering**: All tokens from all villages are standardized and clustered together using K-Means.
* **The Seasonal VH Heuristic**: The pipeline identifies "agricultural clusters" by analyzing the seasonal dynamics of the cross-polarization (VH) band. Monsoon croplands exhibit massive structural changes during the growing season (from bare soil to tall dense canopies), resulting in large VH variations. Water and urban areas remain relatively flat. Any cluster with a mean seasonal VH range exceeding **3.0 dB** is automatically flagged as cropland.
* **Cultivated Fraction Estimation**: For each village, the pipeline counts the fraction of its valid spatial tokens that belong to these agricultural clusters. This ratio becomes the cultivated fraction pseudo-label. These labels are assigned a modest confidence score (0.3) to act as stabilizing "anchors" rather than aggressive overwrites.

### Phase 5: Global Bias Correction & Constrained Optimization

The competition enforces strict physical constraints: an alpha cap of 0.38 per village and a grand total area band between 5200 and 5500 hectares. 
* **Global Bias Correction**: By analyzing the discrepancy between our localized physical prior baseline and the highly accurate OlmoEarth-V1 Base anchors, we identified systematic baseline biases. We apply a targeted global bias correction to all unanchored villages, scaling specific crops to match the true physical distribution.
* **Deficit Absorption**: Our custom rebalancing algorithm calculates the required area deficit or excess needed to satisfy the 5200-5500 ha band. It then sorts the unanchored villages by their `coverage_confidence`. The optimizer iterates through the lowest-confidence villages, safely scaling their predicted areas up to the maximum allowable alpha cap to absorb any deficits. This mathematically guarantees that all constraints are met while perfectly preserving our high-confidence physical priors and our foundation model anchors.

## Repository Structure

* `data/`: Contains raw, processed, and weak label data. Pre-processed tabular features and Sentinel-1 `.npz` caches are stored here.
* `src/s4r/`: Core Python package containing the data ingestion logic, feature generation, Sentinel-1 fetchers, OlmoEarth clustering algorithms, and optimization modules.
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
