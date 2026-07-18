# OlmoEarth Integration Guide (Route A)

> [!NOTE]
> Based on search results, OlmoEarth models are hosted by the Allen Institute for AI (`allenai`) on Hugging Face. They are Vision Transformer (ViT) based foundation models designed for Earth observation tasks.

You **do not need to manually download the weights** (like `.bin` or `.safetensors` files) from the Hugging Face website. The best practice is to use the `transformers` Python library, which programmatically downloads, caches, and loads the weights for you directly into PyTorch.

Here is the detailed, step-by-step procedure for integrating OlmoEarth into our project for **Route A**:

## Step 1: Install Required Libraries

Ensure your environment has PyTorch and the Hugging Face ecosystem libraries installed. Since this repository uses `uv`, you would add them to your `pyproject.toml` or run:

```bash
uv add torch torchvision transformers huggingface_hub
```

## Step 2: Load the Model Programmatically

We will use the `transformers` library to load the base model. This automatically pulls the weights from the Hugging Face Hub and stores them in your local cache (`~/.cache/huggingface/`).

```python
import torch
from transformers import AutoModel

# Load the base OlmoEarth model (replace with the exact model ID if different)
model_id = "allenai/OlmoEarth-v1-Base"

# Load the model without the pre-training head (just the backbone)
olmoearth_backbone = AutoModel.from_pretrained(model_id)
```

## Step 3: Freeze the OlmoEarth Backbone

According to our `AGENTS.md` rules for Route A, the OlmoEarth backbone must remain **frozen**. We only want to train our new Capella adapter and the existing LLP head. 

```python
# Freeze all parameters in the OlmoEarth backbone
for param in olmoearth_backbone.parameters():
    param.requires_grad = False
```

## Step 4: Build the Capella Patch-Embedding Adapter

OlmoEarth expects optical/multispectral inputs (like Sentinel-2), but we are feeding it Capella X-band SAR imagery. We need a trainable adapter (like a 2D convolution layer) to map our SAR patches into the embedding space that OlmoEarth expects.

```python
import torch.nn as nn

class CapellaAdapter(nn.Module):
    def __init__(self, sar_channels=1, embed_dim=768, patch_size=16):
        super().__init__()
        # Maps Capella SAR channels to the dimension OlmoEarth expects
        self.proj = nn.Conv2d(
            in_channels=sar_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, sar_channels, height, width)
        x = self.proj(x)
        # Flatten and transpose for transformer input
        x = x.flatten(2).transpose(1, 2)
        return x
```

## Step 5: Combine into the Final Route A Model

Now, we tie it all together: the adapter, the frozen backbone, the pooling mechanism to get a per-village embedding, and the existing 66-parameter LLP head.

```python
class RouteAModel(nn.Module):
    def __init__(self, olmoearth_backbone, llp_head):
        super().__init__()
        self.adapter = CapellaAdapter(sar_channels=1) # Adjust channels as needed
        self.backbone = olmoearth_backbone
        self.llp_head = llp_head # The existing 66-param head from Route C
        
    def forward(self, capella_patches, village_areas, coverage_confidences):
        # 1. Project Capella SAR to OlmoEarth embeddings (TRAINABLE)
        embeddings = self.adapter(capella_patches)
        
        # 2. Pass through frozen OlmoEarth backbone (FROZEN)
        # Depending on the specific OlmoEarth architecture, you might need to 
        # format the inputs as `inputs_embeds=embeddings`
        outputs = self.backbone(inputs_embeds=embeddings)
        
        # 3. Pool the sequence output into a single vector per village
        # (e.g., using the [CLS] token or mean pooling)
        pooled_features = outputs.last_hidden_state.mean(dim=1) 
        
        # 4. Pass to the existing LLP Head (TRAINABLE)
        frac_model, shares_model = self.llp_head(
            pooled_features, 
            village_areas, 
            coverage_confidences
        )
        
        return frac_model, shares_model
```

## Summary of Execution Flow
1. **Never manually download weights**; let `transformers.AutoModel.from_pretrained()` handle it.
2. Initialize the OlmoEarth model and set `requires_grad = False` on all its parameters.
3. Build a small, trainable projection layer (`CapellaAdapter`) to convert our X-band SAR imagery into the token format OlmoEarth expects.
4. Pool the frozen backbone's outputs to generate standard 1D features per village.
5. Feed those features directly into the existing `src/s4r/fallback/head.py` logic, maintaining all hard structural invariants (non-negativity, capacity caps, shrinkage) dictated by the project rules.
