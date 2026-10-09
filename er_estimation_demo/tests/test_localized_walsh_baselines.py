from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.localized_walsh import (
    ITE_METHOD_NAME,
    LOCALIZED_METHOD_NAMES,
    _localized_weights,
    build_exact_count_configuration,
    build_radius_one_configuration,
    evaluate_localized_dr_lasso_majority_ite,
    rooted_configuration_distance,
    summarize_effect_rows,
)
import train_local_network_interference as demo


def _adjacency(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=np.uint8)
    for left, right in edges:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    return adjacency


def test_paper_facing_ite_method_name_is_exact() -> None:
    assert ITE_METHOD_NAME == "Localized DR-Lasso"
    assert LOCALIZED_METHOD_NAMES == ("Localized DR-Lasso",)


def test_radius_one_distance_uses_rooted_structure_and_treatment_marks() -> None:
    adjacency = _adjacency(4, [(0, 1), (0, 2), (0, 3), (1, 2)])
    factual = np.asarray([0, 1, 0, 0], dtype=np.uint8)
    moved = np.asarray([0, 0, 0, 1], dtype=np.uint8)
    first = build_radius_one_configuration(adjacency, factual, root=0)
    second = build_radius_one_configuration(
        adjacency, factual, root=0, neighbor_treatment=moved
    )
    assert first.neighbor_exposure == pytest.approx(1.0 / 3.0)
    assert second.neighbor_exposure == pytest.approx(1.0 / 3.0)
    assert rooted_configuration_distance(first, second, radius=1) == pytest.approx(0.125)


def test_exact_count_configuration_realizes_requested_neighbor_count() -> None:
    adjacency = _adjacency(6, [(0, i) for i in range(1, 6)])
    treatment = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.uint8)
    config = build_exact_count_configuration(
        adjacency,
        treatment,
        root=0,
        treated_neighbor_count=3,
        seed=123,
    )
    assert config.neighbor_exposure == pytest.approx(3.0 / 5.0)
    assert config.treatment_marks[0] in (-1, 1)


def test_kernel_localization_does_not_silently_expand_to_knn() -> None:
    distances = np.asarray([0.0, 0.25, 0.25], dtype=np.float64)
    kernel = _localized_weights(
        distances, bandwidth=0.24, min_neighbors=2, mode="kernel"
    )
    knn = _localized_weights(
        distances, bandwidth=0.24, min_neighbors=2, mode="knn"
    )
    assert kernel.n_positive == 1
    assert kernel.used_adaptive_knn is False
    assert knn.n_positive == 2
    assert knn.used_adaptive_knn is True


def test_summary_reports_signed_bias_and_support() -> None:
    rows = [
        {"effect": "direct", "truth": 1.0, ITE_METHOD_NAME: 1.5},
        {"effect": "direct", "truth": 3.0, ITE_METHOD_NAME: 2.5},
    ]
    summary = summarize_effect_rows(rows, methods=(ITE_METHOD_NAME,))
    direct = summary["direct"][ITE_METHOD_NAME]
    assert direct["rmse"] == 0.5
    assert direct["mae"] == 0.5
    assert direct["bias"] == 0.0
    assert direct["support_rate"] == 1.0


def test_majority_ite_evaluator_uses_oracle_truth_and_only_paper_method() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=12, er_edge_probability=0.30),
        batch_size=1,
        seed=303,
        device=torch.device("cpu"),
    )
    report = evaluate_localized_dr_lasso_majority_ite(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=0.30,
        min_neighbors=3,
        localization_mode="knn",
        lasso_alpha=0.0,
        nuisance_ridge=0.0,
        cross_fit_folds=2,
        arm_samples=2,
        seed=99,
    )
    assert report["methods"] == (ITE_METHOD_NAME,)
    assert len(report["query_units"]) == 12 * 3
    first = report["query_units"][0]
    effect_index = ("direct", "spillover", "total").index(first["effect"])
    assert first["truth"] == pytest.approx(
        float(batch["oracle_ite"][0, first["query_unit_id"] - 1, effect_index])
    )
    assert "effective_sample_size" in first
    assert "local_radius" in first
    assert "supported" in first
    assert "Rooted-WLasso" not in first
    assert "Rooted-DRLasso" not in first


def test_strict_kernel_support_can_return_nan_without_forcing_knn() -> None:
    # A heterogeneous ER graph with a tiny bandwidth makes many query-specific
    # rooted configurations unsupported; this is a valid estimator outcome.
    batch = demo.generate_batch(
        demo.DataConfig(n_units=18, er_edge_probability=0.55),
        batch_size=1,
        seed=404,
        device=torch.device("cpu"),
    )
    report = evaluate_localized_dr_lasso_majority_ite(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=1.0e-9,
        min_neighbors=4,
        localization_mode="kernel",
        lasso_alpha=0.0,
        nuisance_ridge=0.0,
        cross_fit_folds=2,
        arm_samples=2,
        seed=7,
    )
    rows = report["query_units"]
    assert any(not row["supported"] for row in rows)
    assert all(
        np.isnan(row[ITE_METHOD_NAME])
        for row in rows
        if not row["supported"]
    )
