from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from pfn_interference.baselines.ns_pcr import (
    NSIConfig,
    NSI_METHOD_NAME,
    _psi_matrix,
    _psi_query,
    _unit_signatures,
    evaluate_nsi_ite_on_batch,
    fit_predict_nsi_ite,
)


def _factual_batch(n_units: int = 24) -> dict[str, torch.Tensor]:
    adjacency = torch.ones((1, n_units, n_units), dtype=torch.float32)
    adjacency[:, torch.arange(n_units), torch.arange(n_units)] = 0.0
    treatment = torch.tensor(
        [[float(index % 2) for index in range(n_units)]], dtype=torch.float32
    )
    degree = adjacency.sum(dim=-1)
    exposure = torch.bmm(adjacency, treatment.unsqueeze(-1)).squeeze(-1) / degree
    x = torch.linspace(-1.5, 1.5, n_units, dtype=torch.float32).unsqueeze(0)
    neighbor_mean_x = torch.bmm(adjacency, x.unsqueeze(-1)).squeeze(-1) / degree
    y = 0.4 + 0.7 * x + 0.5 * treatment + 0.25 * exposure + 0.15 * neighbor_mean_x
    return {
        "adjacency": adjacency,
        "x": x,
        "observed_treatment": treatment,
        "observed_exposure": exposure,
        "degree": degree,
        "y_obs": y,
    }


def _graph_inputs(batch: dict[str, torch.Tensor], index: int = 0) -> dict[str, torch.Tensor]:
    return {
        "adjacency": batch["adjacency"][index],
        "x": batch["x"][index],
        "observed_treatment": batch["observed_treatment"][index],
        "observed_exposure": batch["observed_exposure"][index],
        "degree": batch["degree"][index],
        "y_obs": batch["y_obs"][index],
    }


def _fit(batch: dict[str, torch.Tensor], config: NSIConfig):
    return fit_predict_nsi_ite(
        **_graph_inputs(batch),
        treatment_prob=0.5,
        config=config,
    )


def _config(**overrides: object) -> NSIConfig:
    values = {
        "pcr_rank_max": 6,
        "min_effective_donors": 3.0,
        "max_bandwidth": 1.0,
        "min_support_mass": 0.99,
        "svd_rcond": 1.0e-6,
    }
    values.update(overrides)
    return NSIConfig(**values)


def test_nsi_feature_map_uses_only_own_x_degree_and_exposure() -> None:
    x = np.asarray([-2.0, 0.5, 3.0], dtype=np.float64)
    degree = np.asarray([1, 2, 1], dtype=np.int64)
    exposure = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    signatures = _unit_signatures(x, degree)
    features = _psi_matrix(signatures, exposure)

    expected = np.column_stack(
        [
            np.ones(3, dtype=np.float64),
            x,
            degree.astype(np.float64) / 2.0,
            exposure,
        ]
    )
    np.testing.assert_allclose(features, expected, rtol=0.0, atol=0.0)
    assert features.shape == (3, 4)


def test_nsi_query_feature_matches_same_four_feature_definition() -> None:
    x = np.asarray([-2.0, 0.5, 3.0], dtype=np.float64)
    degree = np.asarray([1, 2, 1], dtype=np.int64)
    signatures = _unit_signatures(x, degree)

    query = _psi_query(signatures[1], exposure=0.25)

    np.testing.assert_allclose(
        query,
        np.asarray([1.0, 0.5, 1.0, 0.25], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_nsi_rejects_asymmetric_adjacency() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["adjacency"] = inputs["adjacency"].clone()
    inputs["adjacency"][0, 1] = 0.0

    with pytest.raises(ValueError, match="adjacency must be symmetric"):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_rejects_adjacency_self_loops() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["adjacency"] = inputs["adjacency"].clone()
    inputs["adjacency"][0, 0] = 1.0

    with pytest.raises(ValueError, match="adjacency diagonal must be zero"):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_rejects_nonbinary_adjacency() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["adjacency"] = inputs["adjacency"].clone()
    inputs["adjacency"][0, 1] = 0.5
    inputs["adjacency"][1, 0] = 0.5

    with pytest.raises(ValueError, match="adjacency must be binary"):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_rejects_degree_inconsistent_with_adjacency() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["degree"] = inputs["degree"].clone()
    inputs["degree"][0] += 1.0

    with pytest.raises(ValueError, match="degree must equal adjacency row sums"):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_rejects_exposure_inconsistent_with_graph_and_treatment() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["observed_exposure"] = inputs["observed_exposure"].clone()
    inputs["observed_exposure"][0] = 0.25

    with pytest.raises(
        ValueError,
        match="observed_exposure must equal treated-neighbor count divided by degree",
    ):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_rejects_stale_exposure_after_treatment_changes() -> None:
    batch = _factual_batch()
    inputs = _graph_inputs(batch)
    inputs["observed_treatment"] = inputs["observed_treatment"].clone()
    inputs["observed_treatment"][0] = 1.0 - inputs["observed_treatment"][0]

    with pytest.raises(
        ValueError,
        match="observed_exposure must equal treated-neighbor count divided by degree",
    ):
        fit_predict_nsi_ite(
            **inputs,
            treatment_prob=0.5,
            config=_config(),
        )


def test_nsi_is_deterministic_and_preserves_total_identity() -> None:
    batch = _factual_batch()
    config = _config()

    first = _fit(batch, config)
    second = _fit(batch, config)

    assert first.node_effects.shape == (24, 3)
    assert first.supported.shape == (24, 3)
    assert first.effective_sample_size.shape == (24, 3)
    assert first.local_radius.shape == (24, 3)
    np.testing.assert_allclose(first.node_effects, second.node_effects, equal_nan=True)
    np.testing.assert_array_equal(first.supported, second.supported)
    np.testing.assert_allclose(first.effective_sample_size, second.effective_sample_size, equal_nan=True)
    np.testing.assert_allclose(first.local_radius, second.local_radius, equal_nan=True)

    supported_all = first.supported[..., 0] & first.supported[..., 1] & first.supported[..., 2]
    assert bool(np.any(supported_all))
    np.testing.assert_allclose(
        first.node_effects[..., 2][supported_all],
        (first.node_effects[..., 0] + first.node_effects[..., 1])[supported_all],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.isfinite(first.effective_sample_size[first.supported]).all()
    assert np.isfinite(first.local_radius[first.supported]).all()


def test_nsi_excludes_target_nodes_factual_outcome_from_its_own_prediction() -> None:
    batch = _factual_batch()
    config = _config()
    baseline = _fit(batch, config)

    perturbed = {key: value.clone() for key, value in batch.items()}
    perturbed["y_obs"][0, 0] += 1000.0
    changed = _fit(perturbed, config)

    np.testing.assert_allclose(
        baseline.node_effects[0], changed.node_effects[0], equal_nan=True
    )
    np.testing.assert_array_equal(baseline.supported[0], changed.supported[0])


def test_nsi_support_failure_returns_nan_and_false_support() -> None:
    batch = _factual_batch(n_units=12)
    result = _fit(batch, _config(min_effective_donors=100.0))

    assert not result.supported.any()
    assert np.isnan(result.node_effects).all()


def test_nsi_ignores_oracle_and_counterfactual_batch_fields() -> None:
    batch = _factual_batch()
    batch_with_oracle = copy.deepcopy(batch)
    batch_with_oracle["oracle_ite"] = torch.randn((1, 24, 3))
    batch_with_oracle["oracle_ate"] = torch.randn((1, 3))
    batch_with_oracle["tau"] = torch.randn((1, 24))
    batch_with_oracle["gamma"] = torch.randn((1, 24))
    batch_with_oracle["eta"] = torch.randn((1, 24))
    batch_with_oracle["outcome_a"] = torch.randn((1, 24, 3))
    batch_with_oracle["outcome_b"] = torch.randn((1, 24, 3))

    reference = _fit(batch_with_oracle, _config())

    for key in ("oracle_ite", "oracle_ate", "tau", "gamma", "eta", "outcome_a", "outcome_b"):
        batch_with_oracle[key] = torch.randn_like(batch_with_oracle[key]) * 100.0
    changed = _fit(batch_with_oracle, _config())

    np.testing.assert_allclose(reference.node_effects, changed.node_effects, equal_nan=True)
    np.testing.assert_array_equal(reference.supported, changed.supported)


def test_nsi_batch_evaluator_requires_only_factual_keys() -> None:
    batch = _factual_batch(n_units=16)
    evaluated = evaluate_nsi_ite_on_batch(
        batch=batch,
        treatment_prob=0.5,
        config=_config(min_effective_donors=2.0),
    )

    assert evaluated["method"] == NSI_METHOD_NAME
    assert len(evaluated["unit_effects"]) == 16 * 3
    assert set(evaluated["required_batch_keys"]) == {
        "adjacency",
        "x",
        "observed_treatment",
        "observed_exposure",
        "degree",
        "y_obs",
    }
    assert all(row["method"] == NSI_METHOD_NAME for row in evaluated["unit_effects"])
    assert all(row["effect"] in {"direct", "spillover", "total"} for row in evaluated["unit_effects"])


def test_nsi_config_rejects_invalid_locality_and_rank_settings() -> None:
    with pytest.raises(ValueError):
        NSIConfig(pcr_rank_max=0)
    with pytest.raises(ValueError):
        NSIConfig(min_effective_donors=0.0)
    with pytest.raises(ValueError):
        NSIConfig(max_bandwidth=-0.1)
    with pytest.raises(ValueError):
        NSIConfig(min_support_mass=1.01)
    with pytest.raises(ValueError):
        NSIConfig(svd_rcond=0.0)


def test_nsi_total_support_does_not_require_low_treated_arm() -> None:
    """Total needs only mu(1, high) and mu(0, low), not mu(1, low)."""

    n_units = 8
    adjacency = torch.zeros((n_units, n_units), dtype=torch.float32)
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0

    # Controls are paired only with controls, treated units only with treated units.
    # Hence factual support contains (T=0,E=0) and (T=1,E=1), but no (T=1,E=0).
    treatment = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
    )
    degree = adjacency.sum(dim=-1)
    exposure = (adjacency @ treatment) / degree
    x = torch.linspace(-1.0, 1.0, n_units, dtype=torch.float32)
    y = 0.3 + 0.7 * x + 0.5 * treatment + 0.25 * exposure

    result = fit_predict_nsi_ite(
        adjacency=adjacency,
        x=x,
        observed_treatment=treatment,
        observed_exposure=exposure,
        degree=degree,
        y_obs=y,
        treatment_prob=0.5,
        config=_config(
            min_effective_donors=2.0,
            max_bandwidth=0.0,
            min_support_mass=0.99,
        ),
    )

    # Target 0 has supported mu(0,low) and mu(1,high), but unsupported mu(1,low).
    assert not bool(result.supported[0, 0])
    assert not bool(result.supported[0, 1])
    assert bool(result.supported[0, 2])
    assert np.isfinite(result.node_effects[0, 2])
    assert np.isnan(result.node_effects[0, 0])
    assert np.isnan(result.node_effects[0, 1])
