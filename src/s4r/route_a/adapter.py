"""Capella Adapter: trainable 1-channel patch embedding over a frozen
OlmoEarth-v1-Base transformer trunk.

Data path per village:
    Capella chip (B, 1, H, W)
      -> CNN patch embed (trainable)          tokens (B, N, 768)
      -> + learned positional embedding       (trainable)
      -> frozen OlmoEarth blocks + norm       (requires_grad=False, eval)
      -> mean pool                            (B, 768)
      -> Linear projection (trainable)        (B, n_features=10)
      -> batch standardization (parameter-free, mirrors standardize_features)
      -> 66-parameter LLP head (torch mirror of fallback/head.py)

The head keeps the exact Route C sizing (66 params) and all structural
invariants (cap, non-negativity, confidence shrinkage). Capella is the only
inference-time input (AGENTS.md invariant 7).
"""

import torch
from torch import nn

from s4r import config
from s4r.fallback.head import N_FEATURES, n_params
from s4r.route_a.head_torch import head_forward

EMBED_DIM = 768
MAX_TOKENS = 1024


def load_frozen_backbone(load_weights: bool = True):
    """OlmoEarth-v1-Base encoder with every parameter frozen."""
    from olmoearth_pretrain_minimal import ModelID, load_model_from_id

    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=load_weights)
    encoder = model.encoder
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


class CapellaPatchEmbed(nn.Module):
    """CNN stem: 1-channel X-band chip -> OlmoEarth token dimension.

    Two stride-4 convs give an effective patch size of 16 px per token.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(96, embed_dim, kernel_size=4, stride=4),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_TOKENS, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)  # (B, D, h, w)
        b, d, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)  # (B, N, D)
        n = tokens.shape[1]
        if n > MAX_TOKENS:
            raise ValueError(f"{n} tokens exceeds positional table ({MAX_TOKENS})")
        return tokens + self.pos_embed[:, :n]


class RouteAModel(nn.Module):
    def __init__(self, backbone, alpha: float = config.ALPHA_CAP):
        super().__init__()
        self.alpha = alpha
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.patch_embed = CapellaPatchEmbed()
        self.proj = nn.Linear(EMBED_DIM, N_FEATURES)
        # zero theta would zero the head Jacobian w.r.t. features and starve
        # the adapter of gradient — start small but nonzero
        self.theta = nn.Parameter(torch.randn(n_params(), dtype=torch.float32) * 0.1)

    # --- parameter groups ----------------------------------------------------

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def adapter_parameters(self):
        return list(self.patch_embed.parameters()) + list(self.proj.parameters())

    def head_parameters(self):
        return [self.theta]

    def trainable_parameters(self):
        return self.adapter_parameters() + self.head_parameters()

    # --- forward -------------------------------------------------------------

    def features(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(patches)
        x = tokens
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        pooled = x.mean(dim=1)  # (B, 768)
        feats = self.proj(pooled)  # (B, n_features)
        # parameter-free batch standardization, mirroring standardize_features
        mu = feats.mean(dim=0, keepdim=True)
        sd = feats.std(dim=0, unbiased=False, keepdim=True)
        return (feats - mu) / (sd + 1e-8)

    def forward(
        self,
        patches: torch.Tensor,
        area_ha: torch.Tensor,
        conf: torch.Tensor,
        base_frac: torch.Tensor | None = None,
    ) -> dict:
        # head math in float64: float32 sigmoid saturation rounds frac to
        # exactly alpha and pushes totals past the writer's 1e-9 cap gate
        X = self.features(patches).double()
        return head_forward(
            self.theta.double(),
            X,
            area_ha.double(),
            conf.double(),
            alpha=self.alpha,
            base_frac=None if base_frac is None else base_frac.double(),
        )

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode even during training."""
        super().train(mode)
        self.backbone.eval()
        return self
