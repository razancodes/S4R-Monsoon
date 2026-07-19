# task.md — Route B (S1 Pseudo-Labels) + Route A (OlmoEarth Capella Adapter)

Mission tracking checklist. Decisions locked with the human (2026-07-18):
public OlmoEarth docs (no `docs/olmoearth/` files existed), Sentinel-1 from
Microsoft Planetary Computer (credential-free), CPU-only torch (31 GB disk,
4 GB VRAM), Route A gate LIFTED.

Correction vs. mission text: OlmoEarth loads via `olmoearth_pretrain_minimal`
(`load_model_from_id(ModelID.OLMOEARTH_V1_BASE)`), not vanilla `transformers`.
Same allenai/OlmoEarth-v1-Base weights (ViT-Base, 89M encoder, S1 bands
`['vv','vh']`, input `(B, H, W, T, C)`).

## Phase 0 — Environment & scaffolding
- [x] Add `route-a` dependency group: torch-cpu, olmoearth-pretrain-minimal,
      pystac-client, planetary-computer (keep Route C deps untouched)
- [x] Verify OlmoEarth-v1-Base loads on CPU; record encoder dim & API facts
      in `docs/olmoearth/notes.md` (replacement for the missing guides)
- [x] Write `docs/olmoearth/` notes: use-guide + integration facts from public docs

## Phase 1 — Route B: Sentinel-1 pseudo-labels
- [x] TDD: `tests/test_route_b.py` — S1 fetch utility contract (AOI from
      villages_clean shapefile, monsoon-2025 window, windowed/clipped reads only)
- [x] `src/s4r/route_b/s1_fetch.py` — MPC STAC query + per-village S1 VV/VH
      time-series clips (real data; mock only inside unit tests)
- [x] TDD: pseudo-label generation contract — output matches
      `weak_labels/ingest.py` schema exactly (village_id, cultivated_fraction_est
      ∈ [0,1], dominant_crop ∈ config.CROPS or empty, confidence ∈ (0,1],
      source, notes)
- [x] `src/s4r/route_b/pseudo_labels.py` — OlmoEarth encoder on S1 stacks →
      patch embeddings → per-village agricultural-fraction estimate
- [x] Run for all 29 villages → `data/weak_labels/annotations.csv`
- [x] Validation: correlation of pseudo-labels vs Capella features
      (flood_frac_avg, traj_*, mean_*) on the 25 covered villages; log to
      `experiments/runs/route_b_*.json` and report honestly (no assumed numbers)
- [x] Retrain Route C with `--weak-labels` anchors → check L_anchor active,
      submission gate passes
- [x] Compliance: add Route B rows to `docs/compliance/provenance.md`
      (S1+OlmoEarth = training-time only, never inference input)

## Phase 2 — Route A: Capella adapter + frozen backbone
- [x] TDD: `tests/test_route_a.py` written FIRST, must include:
      - [x] Gradient test: after `.backward()`, adapter grads nonzero,
            backbone grads exactly zero/None (requires_grad False)
      - [x] Invariant test: outputs obey `alpha * area_ha` cap and
            non-negativity structurally (same guarantees as Route C)
      - [x] Head-equivalence test: torch head == numpy `fallback/head.py`
            forward on identical inputs (tolerance ~1e-6)
- [x] `src/s4r/route_a/adapter.py` — trainable CNN patch-embed: 1-channel
      Capella X-band → OlmoEarth token dim; backbone loaded frozen
      (`requires_grad_(False)`, eval mode)
- [x] Token pooling + projection (in `adapter.py`, no separate pooling.py):
      mean pool of the token sequence → one 1D feature vector per village →
      Linear 768→`len(config.MODEL_FEATURES)` so the 66-param head sizing
      is preserved
- [x] `src/s4r/route_a/head_torch.py` — faithful torch mirror of
      `fallback/head.py` math (alpha·sigmoid, softmax, confidence blend,
      anchor base_frac); constraints structural, never loss-only
- [x] `src/s4r/route_a/train.py` — losses ported to torch (L_total, L_mix,
      L_shrink, L_anchor consuming Route B annotations, L2); seeded,
      JSON run log like Route C
- [x] Capella patch dataset: windowed reads from GEO previews per village
      polygon (read-only on `anrf-aise-hack-*_copy/`), downsample toward
      OlmoEarth-native resolution
- [x] CLI: `route-a` subcommand mirroring `route-c` flags

## Phase 3 — Execution & deliverables
- [x] Full suite green: `uv run pytest -q` (integration tests must RUN, not skip)
- [x] Train Route A with anchors; sanity: totals in 5200–5500 band, mix within
      ±2 pp, zero cap violations, zero-coverage villages at regional
      mean unless anchored
- [x] Generate `outputs/submission.csv` through `s4r.submission.writer` gate
      (real features ⇒ is_synthetic False; STOP and fix head math if gate rejects)
- [x] Update `docs/compliance/provenance.md` + `docs/methodology.md` Route A
      status (still "R&D", never "proven")
- [x] Final report: correlation results, loss breakdowns, run-log paths,
      submission comparison (baseline vs Route C+anchors vs Route A)
