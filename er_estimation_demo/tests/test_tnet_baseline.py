import numpy as np
import torch
import pytest

from baselines.tnet import (
    TNetConfig,
    TNetModel,
    _build_tnet_optimizers,
    _truncated_power_basis,
    _interpolate_grid,
    fit_tnet_graph,
    predict_tnet_majority_effects,
)


def _toy_graph(n: int = 8):
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        adjacency[i, (i - 1) % n] = 1.0
        adjacency[i, (i + 1) % n] = 1.0
    x = np.linspace(-1.0, 1.0, n)
    treatment = np.asarray([i % 2 for i in range(n)], dtype=np.float64)
    degree = adjacency.sum(axis=1)
    exposure = adjacency @ treatment / degree
    y = 0.5 + 0.2 * x + 0.4 * treatment + 0.3 * exposure
    return adjacency, x, treatment, exposure, y


def test_truncated_power_basis_matches_degree_two_definition():
    z = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
    basis = _truncated_power_basis(z, n_basis=5, degree=2)
    knots = (1.0 / 3.0, 2.0 / 3.0)
    expected = torch.stack(
        [
            torch.ones_like(z),
            z,
            z.square(),
            torch.relu(z - knots[0]).square(),
            torch.relu(z - knots[1]).square(),
        ],
        dim=1,
    )
    assert basis.shape == (4, 5)
    torch.testing.assert_close(basis, expected, atol=1e-6, rtol=1e-6)


def test_tnet_optimizers_use_independent_official_style_learning_rates():
    config = TNetConfig(
        hidden_dim=12,
        grid_size=4,
        spline_basis=7,
        epochs=1,
        learning_rate_1step=1e-4,
        learning_rate_2step=1e-2,
    )
    model = TNetModel(input_dim=1, config=config)
    first_step, second_step = _build_tnet_optimizers(model, config)
    assert first_step.param_groups[0]["lr"] == 1e-4
    assert second_step.param_groups[0]["lr"] == 1e-2


def test_piecewise_grid_interpolation_matches_endpoints_and_midpoint():
    grid = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]], dtype=torch.float32)
    z = torch.tensor([0.0, 0.75], dtype=torch.float32)
    got = _interpolate_grid(grid, z)
    expected = torch.tensor([0.1, 0.3], dtype=torch.float32)
    torch.testing.assert_close(got, expected)


def test_tnet_model_nuisance_forward_has_expected_shapes():
    adjacency, x, treatment, exposure, _ = _toy_graph()
    config = TNetConfig(hidden_dim=12, grid_size=4, spline_basis=7, epochs=1)
    model = TNetModel(input_dim=1, config=config)
    a = torch.tensor(adjacency, dtype=torch.float32)
    xt = torch.tensor(x[:, None], dtype=torch.float32)
    tt = torch.tensor(treatment, dtype=torch.float32)
    zt = torch.tensor(exposure, dtype=torch.float32)
    h = model.encode(a, xt)
    g1 = model.g1_probability(h, tt)
    g2 = model.g2_probability(h, zt)
    mu = model.outcome_mean(h, tt, zt)
    eps = model.perturbation(tt, zt)
    assert h.shape == (8, 6)
    assert g1.shape == g2.shape == mu.shape == eps.shape == (8,)
    assert torch.all((g1 > 0) & (g1 < 1))
    assert torch.all(g2 > 0)


def test_fitted_tnet_returns_finite_majority_effects():
    adjacency, x, treatment, exposure, y = _toy_graph(n=10)
    config = TNetConfig(
        hidden_dim=10,
        grid_size=4,
        spline_basis=7,
        epochs=4,
        learning_rate_1step=5e-3,
        learning_rate_2step=2e-3,
        seed=13,
    )
    fitted = fit_tnet_graph(
        adjacency=adjacency,
        covariates=x[:, None],
        treatment=treatment,
        exposure=exposure,
        outcomes=y,
        config=config,
    )
    prediction = predict_tnet_majority_effects(
        fitted.model,
        adjacency=adjacency,
        covariates=x[:, None],
        treatment_prob=0.5,
    )
    assert prediction.node_effects.shape == (10, 3)
    assert prediction.arm_means.shape == (10, 4)
    assert np.isfinite(prediction.node_effects).all()
    assert np.isfinite(prediction.arm_means).all()
    assert np.isfinite(fitted.final_losses["nuisance"])
    assert np.isfinite(fitted.final_losses["targeted"])


def test_default_epoch_performs_fifty_real_targeted_optimizer_updates(monkeypatch):
    # Guard against collapsing the author's inner optimization loop to one step.
    import baselines.tnet as module
    optimizers = []
    build = module._build_tnet_optimizers
    def capture(model, config):
        pair = build(model, config)
        optimizers.extend(pair)
        return pair
    monkeypatch.setattr(module, "_build_tnet_optimizers", capture)
    a, x, t, z, y = _toy_graph()
    fit_tnet_graph(adjacency=a, covariates=x, treatment=t, exposure=z,
                   outcomes=y, config=TNetConfig(hidden_dim=4, epochs=1))
    base_steps = {int(s["step"]) for s in optimizers[0].state.values()}
    targeted_steps = {int(s["step"]) for s in optimizers[1].state.values()}
    assert base_steps == {1}
    assert targeted_steps == {50}


def test_outcome_normalization_restores_predictions_in_original_units():
    a, x, t, z, y = _toy_graph()
    config = TNetConfig(hidden_dim=4, epochs=2, targeted_steps=2, seed=123)
    predictions = []
    for outcomes in (y, 7.0 * y + 13.0):
        fitted = fit_tnet_graph(adjacency=a, covariates=x, treatment=t,
                               exposure=z, outcomes=outcomes, config=config)
        predictions.append(predict_tnet_majority_effects(
            fitted.model, adjacency=a, covariates=x, treatment_prob=0.5))
    np.testing.assert_allclose(predictions[1].arm_means,
                               7.0 * predictions[0].arm_means + 13.0, atol=2e-4)
    np.testing.assert_allclose(predictions[1].node_effects,
                               7.0 * predictions[0].node_effects, atol=2e-4)


def test_default_targeted_loss_weight_uses_fitted_sample_size():
    a, x, t, z, y = _toy_graph()
    fitted = fit_tnet_graph(adjacency=a, covariates=x, treatment=t, exposure=z,
                           outcomes=y, config=TNetConfig(hidden_dim=4, epochs=1))
    assert fitted.training_config["effective_beta"] == pytest.approx(20.0 / np.sqrt(8))


def test_constant_outcome_normalization_has_finite_predictions():
    a, x, t, z, _ = _toy_graph()
    fitted = fit_tnet_graph(adjacency=a, covariates=x, treatment=t, exposure=z,
        outcomes=np.full(8, 5.0), config=TNetConfig(hidden_dim=4, epochs=1, targeted_steps=1))
    result = predict_tnet_majority_effects(fitted.model, adjacency=a,
                                         covariates=x, treatment_prob=0.5)
    assert np.isfinite(result.arm_means).all()
