from __future__ import annotations

import pytest
import torch

import pfn_interference.model as demo
from pfn_interference.baselines.runner import ATE_METHOD_NAMES, MAIN_ATE_METHOD_NAMES, EFFECT_NAMES, run_ate_baselines_on_batch
from pfn_interference.baselines.tnet import TNET_METHOD_NAME, TNetConfig


def test_paper_facing_ate_runner_has_only_approved_method_names() -> None:
    assert ATE_METHOD_NAMES == (
        "HT",
        "Haj",
        "F",
        "L",
        "reg-net",
        TNET_METHOD_NAME,
    )
    assert MAIN_ATE_METHOD_NAMES == ("HT", "Haj", "F", "L", "reg-net", TNET_METHOD_NAME)
    assert EFFECT_NAMES == ("direct", "spillover", "total")


def test_ate_runner_uses_same_graph_oracle_truth() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=20, er_edge_probability=0.25),
        batch_size=2,
        seed=123,
        device=torch.device("cpu"),
    )
    report = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        bandwidth=2,
        ridge=1e-6,
        tnet_config=TNetConfig(hidden_dim=8, grid_size=4, spline_basis=7, epochs=2, seed=3),
    )
    assert report["methods"] == ATE_METHOD_NAMES
    assert len(report["graph_effects"]) == 2 * 3
    for row in report["graph_effects"]:
        effect_index = EFFECT_NAMES.index(row["effect"])
        expected = float(batch["oracle_ate"][row["dataset_id"] - 1, effect_index])
        assert row["truth"] == pytest.approx(expected)
        for method in ATE_METHOD_NAMES:
            assert method in row


def test_removed_diagnostic_is_not_computed_or_emitted():
    batch = demo.generate_batch(demo.DataConfig(n_units=20, er_edge_probability=0.3),
                                batch_size=1, seed=2026, device=torch.device("cpu"))
    report = run_ate_baselines_on_batch(batch=batch, treatment_prob=.5, bandwidth=2,
        ridge=1e-6, tnet_config=TNetConfig(hidden_dim=8, epochs=1, seed=2026))
    assert set(report["methods"]) == {"HT", "Haj", "F", "L", "reg-net", "Tnet"}
    assert all(not any("AIPW" in key for key in row) for row in report["graph_effects"])


def test_f_and_l_do_not_inherit_network_adjustment_ridge():
    batch = demo.generate_batch(demo.DataConfig(n_units=100, er_edge_probability=.3),
                                batch_size=1, seed=2026, device=torch.device("cpu"))
    reports = [run_ate_baselines_on_batch(batch=batch, treatment_prob=.5,
        bandwidth=2, ridge=ridge, tnet_config=TNetConfig(hidden_dim=4, epochs=1))
        for ridge in (0., 100.)]
    for first, second in zip(reports[0]["graph_effects"], reports[1]["graph_effects"]):
        for method in ("F", "L"):
            assert first[method + "_supported"] and second[method + "_supported"]
            assert first[method] == pytest.approx(second[method], abs=1e-12)
