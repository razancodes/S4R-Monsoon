# OlmoEarth usage notes (replacement for missing use/integration guides)

The mission referenced `docs/olmoearth/olmoearth_use_guide.md` and
`olmoearth_integration_guide.md`, which never existed in this repo. Per human
decision (2026-07-18) this file records the verified facts from the public
sources instead:

- Model card: https://huggingface.co/allenai/OlmoEarth-v1-Base
- Loader/API: https://github.com/allenai/olmoearth_pretrain_minimal (PyPI:
  `olmoearth-pretrain-minimal`)
- Paper: https://allenai.org/papers/olmoearth

## Verified facts (probed locally, CPU, 2026-07-18)

- Load: `load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)`
  — NOT vanilla `transformers`; weights auto-download from HF into `~/.cache`.
- Architecture: FlexiViT encoder, **89.0M params, embedding dim 768**.
- Input container: `MaskedOlmoEarthSample` (NamedTuple). Relevant fields:
  `timestamps [B,T,3] = (day, month 0-indexed, year)`, `sentinel1
  [B,H,W,T,2]` with band order `['vv','vh']` (dB), `sentinel1_mask
  [B,H,W,T]` (0 = visible), optional `latlon [B,2]`.
- Normalization: `Normalizer(std_multiplier=2.0).normalize(Modality.SENTINEL1, x)`
  — returns float64; **cast to float32** before the encoder or conv bias
  dtype errors.
- Encoder call: `model.encoder(sample, patch_size=8, input_res=10,
  fast_pass=True)` → `{"tokens_and_masks": TokensAndMasks}` where
  `.sentinel1` has shape `(B, P_H, P_W, T, band_sets, 768)` and
  `.sentinel1_mask` marks valid tokens. With patch_size=8 at 10 m res one
  token covers 80 m × 80 m.
- Per-village embedding = masked mean over valid tokens.

## Project-specific constraints (from AGENTS.md)

- Sentinel-1 + OlmoEarth outputs are **training-time signal only** (Route B
  anchors, Route A teacher); Capella remains the sole inference input.
- Capella rasters must never be uploaded anywhere; OlmoEarth inference on
  Capella runs locally through the frozen backbone (Route A).
- Route A: backbone frozen (`requires_grad=False`), trainable 1-channel
  Capella patch-embed adapter, pooled features feed the SAME 66-param LLP
  head logic and aggregate losses as Route C.
- Known domain gap: OlmoEarth pretrained on C-band Sentinel-1 (5.4 GHz,
  10 m); Capella is X-band (9.4–9.9 GHz, GEO previews ~0.74 m, uncalibrated
  uint8 dB). Expect Route A to possibly underperform Route C — that is an
  acceptable, documented outcome.
