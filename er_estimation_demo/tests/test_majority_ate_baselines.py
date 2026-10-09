from __future__ import annotations

import numpy as np
import pytest
import torch

import train_local_network_interference as demo
from baselines.tnet import TNET_METHOD_NAME, TNetConfig
from baselines.runner import (
    ATE_METHOD_NAMES,
    MAIN_ATE_METHOD_NAMES,
    majority_arm_propensities,
    run_ate_baselines_on_batch,
    _majority_state_codes,
)


def test_ate_method_names_are_paper_labels_only() -> None:
    assert ATE_METHOD_NAMES == (
        "HT",
        "Haj",
        "F",
        "L",
        "reg-net",
        TNET_METHOD_NAME,
    )
    assert MAIN_ATE_METHOD_NAMES == ("HT", "Haj", "F", "L", "reg-net", TNET_METHOD_NAME)


def test_majority_arm_propensities_are_positive_and_sum_to_one() -> None:
    degree = np.asarray([1, 2, 6, 60], dtype=np.int64)
    propensity = majority_arm_propensities(degree, treatment_prob=0.5)

    assert propensity.shape == (4, 4)
    assert np.all(propensity > 0.0)
    np.testing.assert_allclose(propensity.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
    # Fixed paper-facing state order is 00, 10, 01, 11.
    # Thus own-treatment=0 occupies columns 00 and 01; own-treatment=1 occupies 10 and 11.
    np.testing.assert_allclose(propensity[:, [0, 2]].sum(axis=1), 0.5)
    np.testing.assert_allclose(propensity[:, [1, 3]].sum(axis=1), 0.5)


def test_majority_state_codes_follow_documented_00_10_01_11_order() -> None:
    treatment = np.asarray([0, 1, 0, 1], dtype=np.int64)
    degree = np.asarray([2, 2, 2, 2], dtype=np.int64)
    neighbor_counts = np.asarray([0, 0, 2, 2], dtype=np.int64)

    codes = _majority_state_codes(treatment, neighbor_counts, degree)

    np.testing.assert_array_equal(codes, np.asarray([0, 1, 2, 3]))


def test_ate_baselines_use_graph_level_oracle_truth_and_common_contrasts() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=80, er_edge_probability=0.10),
        batch_size=2,
        seed=1234,
        device=torch.device("cpu"),
    )
    report = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=2,
        ridge=1.0e-6,
        tnet_config=TNetConfig(hidden_dim=8, grid_size=4, spline_basis=7, epochs=2, seed=7),
    )

    assert report["methods"] == ATE_METHOD_NAMES
    assert len(report["graph_effects"]) == 2 * 3
    for row in report["graph_effects"]:
        effect_index = ("direct", "spillover", "total").index(row["effect"])
        assert row["truth"] == pytest.approx(
            float(batch["oracle_ate"][row["dataset_id"] - 1, effect_index])
        )
        for method in ATE_METHOD_NAMES:
            assert method in row
            if row[f"{method}_supported"]:
                assert np.isfinite(float(row[method]))
            else:
                # The unprojected reg-net objective need not have a finite
                # minimum on a finite sample; do not substitute a fake score.
                assert method == "reg-net"
                assert np.isnan(float(row[method]))
    assert set(report["effects"]) == {"direct", "spillover", "total"}
    assert set(report["effects"]["direct"]) == set(ATE_METHOD_NAMES)


def test_negative_spillover_truth_is_not_clipped_to_positive() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(
            n_units=60,
            er_edge_probability=0.12,
            gamma_mean=-0.3,
            gamma_sd=0.0,
            eta_mean=0.0,
            eta_sd=0.0,
        ),
        batch_size=1,
        seed=91,
        device=torch.device("cpu"),
    )
    report = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=2,
        ridge=1.0e-6,
        tnet_config=TNetConfig(hidden_dim=8, grid_size=4, spline_basis=7, epochs=2, seed=7),
    )
    spill = next(row for row in report["graph_effects"] if row["effect"] == "spillover")
    assert spill["truth"] < 0.0


def test_ate_report_metadata_uses_documented_arm_code_order() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=24, er_edge_probability=0.20),
        batch_size=1,
        seed=2026,
        device=torch.device("cpu"),
    )
    report = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=2,
        ridge=1.0e-6,
        tnet_config=TNetConfig(hidden_dim=8, grid_size=4, spline_basis=7, epochs=2, seed=7),
    )

    assert report["arm_codes"] == {"00": 0, "10": 1, "01": 2, "11": 3}


def test_oracle_semiparametric_anchor_is_reported_separately_from_methods() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=40, er_edge_probability=0.10, noise_sd=0.0),
        batch_size=2,
        seed=4242,
        device=torch.device("cpu"),
    )
    report = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=2,
        ridge=1.0e-6,
        tnet_config=TNetConfig(hidden_dim=8, grid_size=4, spline_basis=7, epochs=2, seed=9),
    )

    assert "Oracle-DR" not in report["methods"]
    anchor = report["semiparametric_anchor"]
    assert len(anchor) == 2 * 3
    assert {row["effect"] for row in anchor} == {"direct", "spillover", "total"}
    assert all(float(row["variance"]) >= 0.0 for row in anchor)
    assert all(float(row["standard_error"]) >= 0.0 for row in anchor)
