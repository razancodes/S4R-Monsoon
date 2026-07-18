# OlmoEarth Use Guide: Strategy & Implementation

This guide documents the strategic role of the OlmoEarth foundation model in the **S4R-Monsoon** project, specifically detailing how it bridges the gap between Route B (sanity checks) and Route A (deep feature embeddings).

## 1. Strategic Overview

OlmoEarth is a multimodal, spatio-temporal foundation model (Vision Transformer) trained on Earth observation data (like Sentinel-1 and Sentinel-2). In this project, it serves two distinct purposes:
1. **Route B (Short-term):** As an automated pseudo-label generator using its native Sentinel-1 modality.
2. **Route A (Long-term):** As a frozen feature extractor for Capella X-band SAR imagery, paired with a trainable adapter.

---

## 2. Route B: Pseudo-Labeling & Domain Validation (Step 1)

Before attempting to build the complex Route A adapter, OlmoEarth must be deployed as part of Route B's Sentinel-based analysis. 

### Objective
Generate per-village "weak labels" (anchors) to break the Route C plateau, while simultaneously validating whether OlmoEarth's phenology patterns correlate with our Capella features.

### Procedure
1. **Fetch Data:** Download Sentinel-1 GRD imagery for the 29-village AOI (June–October) via Microsoft Planetary Computer or Copernicus Data Space.
2. **Native Inference:** Run OlmoEarth on this Sentinel-1 time series. Since OlmoEarth is already trained on C-band SAR (Sentinel-1), **no adapter is needed** and there is zero domain gap.
3. **Aggregation:** Pool the pixel-level outputs into village polygons to create per-village crop-type probabilities or agricultural fraction estimates.
4. **Integration (L_anchor):** Save these outputs as `data/weak_labels/annotations.csv`. Route C's `ingest.py` will read them, and they will populate the `L_anchor` term during Route C training. 
5. **Validation Cross-Check:** Compare these OlmoEarth pseudo-labels against the existing Capella features (e.g., `flood_frac`, `traj_range`).
    - *If they correlate:* Proceed to Route A.
    - *If they do not correlate:* Abort Route A (the X-band vs. C-band gap is too wide) and stick to Route C refinement.

> [!IMPORTANT]
> Route B data (Sentinel-1) is strictly for **training-time only**. It must never be used as an inference input for the final Kaggle submission.

---

## 3. Route A: Capella Adapter & Deep Embeddings (Step 2)

If the Route B cross-check proves successful, OlmoEarth is integrated directly into the inference pipeline via Route A.

### Objective
Replace the engineered Capella features (`extract.py`) with deep, pooled embeddings extracted by OlmoEarth from Capella patches.

### Procedure
1. **The Adapter:** Build a trainable CNN patch-embedding layer that translates 1-channel Capella X-band SAR into the token dimensions OlmoEarth expects.
2. **The Backbone:** The OlmoEarth ViT backbone is completely **frozen** (`requires_grad = False`).
3. **The Head:** The pooled sequence outputs from OlmoEarth are fed into the exact same 66-parameter Learning-from-Label-Proportions (LLP) head from Route C, maintaining all structural invariants (non-negativity, capacity caps).

---

## 4. Implementation: Loading OlmoEarth

### Primary Method (Recommended)
Use the Hugging Face `transformers` library to load the model programmatically. This automatically downloads and caches the necessary weights.

```python
import torch
from transformers import AutoModel

# Load the base OlmoEarth model
model_id = "allenai/OlmoEarth-v1-Base"
olmoearth_backbone = AutoModel.from_pretrained(model_id)

# Freeze backbone for Route A
for param in olmoearth_backbone.parameters():
    param.requires_grad = False
```

### Fallback Method (Manual Download)
> [!WARNING]
> Use this fallback **only** if the `transformers` API fails (e.g., network timeout, unexpected format errors, or environment constraints).

If programmatic loading fails, the agent must execute the following fallback:
1. Navigate to the Hugging Face Hub page for `allenai/OlmoEarth-v1-Base` (or the relevant fine-tuned variant).
2. Manually download the model weights (e.g., `model.safetensors` or `pytorch_model.bin`) and the configuration file (`config.json`).
3. Save these files locally in a directory, for example: `models/OlmoEarth-local/`.
4. Load the model pointing to the local directory:

```python
from transformers import AutoModel

# Load from the local directory instead of the Hub
local_model_path = "./models/OlmoEarth-local"
olmoearth_backbone = AutoModel.from_pretrained(local_model_path, local_files_only=True)
```
If using pure PyTorch without `transformers`, initialize the architecture manually and use `torch.load()` to ingest the downloaded state dictionary.
