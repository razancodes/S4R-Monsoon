"""Route A: frozen OlmoEarth backbone + trainable Capella adapter.

Mandated by the mission brief:
- Gradient test: adapter gradients flow, backbone gradients stay exactly zero.
- Invariant test: outputs structurally obey the alpha*area cap and
  non-negativity, exactly as Route C does.
Plus: the torch head must be numerically equivalent to the numpy Route C head
(same 66-parameter structure), and torch losses must match the numpy losses.

Tests auto-skip when the route-a dependency group (torch) is absent — mirrors
the integration-test convention in AGENTS.md.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s4r import config
from s4r.fallback import head as np_head
from s4r.losses import aggregate as np_losses


@pytest.fixture(scope="module")
def backbone():
    from s4r.route_a.adapter import load_frozen_backbone

    # load_weights=False: structure tests must not require the HF download
    return load_frozen_backbone(load_weights=False)


@pytest.fixture()
def model(backbone):
    from s4r.route_a.adapter import RouteAModel

    torch.manual_seed(0)
    return RouteAModel(backbone)


def _batch(n=4, hw=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    patches = torch.rand(n, 1, hw, hw, generator=g)
    area = torch.tensor([800.0, 400.0, 1200.0, 300.0][:n])
    conf = torch.tensor([1.0, 0.5, 0.0, 0.8][:n])
    return patches, area, conf


# --- backbone freezing -------------------------------------------------------


def test_backbone_parameters_are_frozen(model):
    from s4r.route_a.adapter import RouteAModel  # noqa: F401

    assert all(not p.requires_grad for p in model.backbone_parameters())
    assert any(p.requires_grad for p in model.adapter_parameters())


def test_gradients_flow_to_adapter_but_not_backbone(model):
    patches, area, conf = _batch()
    out = model(patches, area, conf)
    loss = out["pred"].sum()
    loss.backward()

    for p in model.backbone_parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0

    grads = [p.grad for p in model.adapter_parameters() if p.grad is not None]
    assert grads, "no adapter gradients at all"
    assert any(float(g.abs().max()) > 0 for g in grads)


# --- structural invariants ---------------------------------------------------


def test_outputs_obey_cap_and_nonnegativity_for_random_weights(backbone):
    from s4r.route_a.adapter import RouteAModel

    for seed in range(3):
        torch.manual_seed(seed)
        m = RouteAModel(backbone)
        # scale up head params to push sigmoid/softmax toward extremes
        with torch.no_grad():
            for p in m.head_parameters():
                p.mul_(50.0)
        patches, area, conf = _batch(seed=seed)
        out = m(patches, area, conf)
        pred = out["pred"].detach().numpy()
        assert (pred >= 0).all()
        totals = pred.sum(axis=1)
        assert (totals <= config.ALPHA_CAP * area.numpy() + 1e-6).all()


def test_zero_confidence_village_gets_exact_baseline(model):
    patches, area, conf = _batch()
    out = model(patches, area, conf)
    pred = out["pred"].detach().numpy()
    i = 2  # conf == 0.0
    expected = config.BASELINE_FRAC * float(area[i]) * config.MIX_VECTOR
    np.testing.assert_allclose(pred[i], expected, rtol=1e-5)


# --- numerical equivalence with Route C -------------------------------------


def test_torch_head_matches_numpy_head():
    from s4r.route_a.head_torch import head_forward

    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, size=np_head.n_params())
    X = rng.normal(0, 1, size=(29, np_head.N_FEATURES))
    area = rng.uniform(200, 1500, size=29)
    conf = rng.uniform(0, 1, size=29)
    base = rng.uniform(0, config.ALPHA_CAP, size=29)

    np_out = np_head.forward(theta, X, area, conf, base_frac=base)
    t_out = head_forward(
        torch.tensor(theta), torch.tensor(X), torch.tensor(area),
        torch.tensor(conf), base_frac=torch.tensor(base),
    )
    for key in ("frac", "shares", "totals", "pred"):
        np.testing.assert_allclose(t_out[key].numpy(), np_out[key], rtol=1e-6, atol=1e-9)


def test_torch_losses_match_numpy_losses():
    from s4r.route_a.losses_torch import loss_anchor, loss_mix, loss_shrink, loss_total

    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 60, size=(29, 5))
    totals = pred.sum(axis=1)
    frac_model = rng.uniform(0, 0.38, size=29)
    shares_model = rng.dirichlet(np.ones(5), size=29)
    conf = rng.uniform(0, 1, size=29)

    import pandas as pd

    anchors = pd.DataFrame(
        {
            "village_index": [0, 5, 10],
            "cultivated_fraction_est": [0.1, 0.3, 0.2],
            "weight": [0.3, 0.3, 0.5],
        }
    )
    frac = rng.uniform(0, 0.38, size=29)

    assert float(loss_total(torch.tensor(totals))) == pytest.approx(
        np_losses.loss_total(totals)
    )
    assert float(loss_mix(torch.tensor(pred))) == pytest.approx(np_losses.loss_mix(pred))
    assert float(
        loss_shrink(torch.tensor(frac_model), torch.tensor(shares_model), torch.tensor(conf))
    ) == pytest.approx(np_losses.loss_shrink(frac_model, shares_model, conf))
    assert float(loss_anchor(torch.tensor(frac), anchors)) == pytest.approx(
        np_losses.loss_anchor(frac, anchors)
    )


def test_trainer_reduces_loss_and_respects_constraints(backbone, tmp_path):
    from s4r.route_a.train import TrainAConfig, train_route_a

    from s4r.data.synthetic import make_synthetic_features

    features_df = make_synthetic_features()
    area = features_df["area_ha"].to_numpy()
    conf = np.clip(
        features_df[[f"coverage_{d}" for d in config.DATES]].to_numpy().mean(axis=1) / 0.5,
        0.0,
        1.0,
    )
    torch.manual_seed(0)
    chips = torch.rand(29, 1, 32, 32)

    cfg = TrainAConfig(epochs=40, lr=0.05, seed=0)
    result = train_route_a(
        chips, area, conf, cfg, anchors=None, run_dir=str(tmp_path), backbone=backbone
    )

    pred = result["pred"]
    assert pred.shape == (29, 5)
    assert (pred >= 0).all()
    assert (pred.sum(axis=1) <= config.ALPHA_CAP * area + 1e-6).all()
    assert result["loss_curve"][-1] < result["loss_curve"][0]
    assert result["run_log_path"] is not None
    import json

    log = json.loads(open(result["run_log_path"]).read())
    assert log["route"] == "A"
    assert log["config"]["seed"] == 0


def test_trainer_is_deterministic(backbone, tmp_path):
    from s4r.route_a.train import TrainAConfig, train_route_a

    area = np.full(29, 700.0)
    conf = np.full(29, 0.8)
    torch.manual_seed(0)
    chips = torch.rand(29, 1, 32, 32)
    cfg = TrainAConfig(epochs=5, seed=123)
    r1 = train_route_a(chips, area, conf, cfg, run_dir=None, backbone=backbone)
    r2 = train_route_a(chips, area, conf, cfg, run_dir=None, backbone=backbone)
    np.testing.assert_array_equal(r1["pred"], r2["pred"])


def test_head_param_count_is_66():
    from s4r.route_a.adapter import RouteAModel
    from s4r.route_a.adapter import load_frozen_backbone

    m = RouteAModel(load_frozen_backbone(load_weights=False))
    n = sum(p.numel() for p in m.head_parameters())
    assert n == np_head.n_params() == 66
