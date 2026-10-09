from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pfn_interference.model as demo
from pfn_interference.baselines.ns_pcr import NSI_METHOD_NAME
from pfn_interference.baselines.tnet import TNET_METHOD_NAME
from pfn_interference.metrics.unified import (
    MAIN_ATE_METHODS,
    MAIN_ITE_METHODS,
    evaluate_unified_benchmark,
    predict_pfn_majority_ite,
    save_unified_benchmark,
)


class LinearOraclePFN(torch.nn.Module):
    """One-component GMM with Y(t,e) contrast = tau*t + gamma*e."""

    def __init__(self, tau: float, gamma: float):
        super().__init__()
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.training = False

    def forward(self, tokens, queries, adjacency):
        from pfn_interference.estimands import arm_mean_exposure
        t,s=queries.unbind(-1)
        d=adjacency.sum(-1).cpu().numpy().astype(int)
        low=torch.tensor([[arm_mean_exposure(int(v),.5,0) for v in row] for row in d],device=t.device)
        high=torch.tensor([[arm_mean_exposure(int(v),.5,1) for v in row] for row in d],device=t.device)
        mean=self.tau*t+self.gamma*torch.where(s==0,low[...,None],high[...,None])
        return {
            "gmm_pi": torch.ones(*mean.shape, 1, device=mean.device),
            "gmm_mu": mean.unsqueeze(-1),
            "gmm_sigma": torch.ones(*mean.shape, 1, device=mean.device) * 0.1,
        }


class FakeCEPOSurface:
    model_config = {}
    def eval(self):
        return self

    def predict_cepo(
        self,
        *,
        X_context,
        t_context,
        y_context,
        X_query,
        t_query,
        temperature,
        n_samples=None,
    ):
        del X_context, t_context, y_context, temperature, n_samples
        x = X_query[..., 0]
        e = X_query[..., -1]
        value = 0.1 * x + 0.2 * e + t_query * (0.5 + 0.1 * e)
        return value.unsqueeze(1)


def test_main_method_sets_are_intentionally_small() -> None:
    assert MAIN_ITE_METHODS == ("PFN with interference", "Localized DR-Lasso", NSI_METHOD_NAME, "HyperSCI", TNET_METHOD_NAME, "causalPFN")
    assert MAIN_ATE_METHODS == ("PFN with interference", "HT", "Haj", "F", "L", "reg-net", TNET_METHOD_NAME, "causalPFN")


def test_design_weight_tensors_support_real_64_by_300_batch_shape() -> None:
    # Regression coverage for the real evaluation shape that exposed a MUSA
    # advanced-indexing incompatibility: 64 graphs x 300 units.
    from pfn_interference.metrics.unified import _design_weight_tensors

    degree = torch.arange(1, 20, dtype=torch.float32).repeat(64 * 300 // 19 + 1)
    degree = degree[: 64 * 300].reshape(64, 300)
    low, high = _design_weight_tensors(degree, 0.5)

    assert low.shape == (64, 300, 20)
    assert high.shape == (64, 300, 20)
    torch.testing.assert_close(low.sum(dim=-1), torch.ones((64, 300)))
    torch.testing.assert_close(high.sum(dim=-1), torch.ones((64, 300)))
    counts = torch.arange(20, dtype=torch.float32).view(1, 1, -1)
    impossible = counts > degree.unsqueeze(-1)
    assert torch.count_nonzero(low.masked_select(impossible)) == 0
    assert torch.count_nonzero(high.masked_select(impossible)) == 0


def test_pfn_arm_integration_recovers_linear_oracle_truth() -> None:
    config = demo.DataConfig(
        n_units=12,
        er_edge_probability=0.30,
        dgp_version="tnet_linear_neighbor_covariate",
        tau_mean=0.7,
        tau_episode_sd=0.0,
        tau_unit_sd=0.0,
        tau_own_score_slope=0.0,
        tau_neighbor_score_slope=0.0,
        gamma_mean=-0.25,
        gamma_sd=0.0,
        gamma_own_score_slope=0.0,
        gamma_neighbor_score_slope=0.0,
        eta_mean=0.0,
        eta_sd=0.0,
        noise_sd=0.0,
    )
    batch = demo.generate_batch(
        config, batch_size=2, seed=11, device=torch.device("cpu")
    )
    model = LinearOraclePFN(tau=0.7, gamma=-0.25)

    estimate = predict_pfn_majority_ite(
        model=model,
        batch=batch,
        treatment_prob=0.5,
    )

    torch.testing.assert_close(estimate, batch["oracle_ite"], atol=1e-5, rtol=1e-5)


def test_unified_benchmark_uses_same_graph_truth_for_ite_and_ate() -> None:
    config = demo.DataConfig(
        n_units=16,
        er_edge_probability=0.25,
        dgp_version="tnet_linear_neighbor_covariate",
        tau_mean=0.5,
        tau_episode_sd=0.0,
        tau_unit_sd=0.0,
        tau_own_score_slope=0.0,
        tau_neighbor_score_slope=0.0,
        gamma_mean=0.3,
        gamma_sd=0.0,
        gamma_own_score_slope=0.0,
        gamma_neighbor_score_slope=0.0,
        eta_mean=0.0,
        eta_sd=0.0,
        noise_sd=0.0,
    )
    batch = demo.generate_batch(
        config, batch_size=2, seed=22, device=torch.device("cpu")
    )
    model = LinearOraclePFN(tau=0.5, gamma=0.3)
    report = evaluate_unified_benchmark(
        model=model,
        batch=batch,
        treatment_prob=0.5,
        ate_bandwidth=2,
        ate_ridge=1e-6,
        localized_bandwidth=0.30,
        localized_min_neighbors=3,
        localized_mode="knn",
        localized_lasso_alpha=0.0,
        localized_nuisance_ridge=0.0,
        localized_cross_fit_folds=2,
        localized_arm_samples=1,
        standard_ite_epochs=5,
        standard_ite_hidden_dim=12,
        tnet_epochs=2,
        tnet_hidden_dim=8,
        tnet_grid_size=4,
        tnet_spline_basis=7,
        causalpfn_model=FakeCEPOSurface(),
        causalpfn_query_chunk_size=32,
        seed=99,
    )

    assert tuple(report["main_ite_methods"]) == MAIN_ITE_METHODS
    assert tuple(report["main_ate_methods"]) == MAIN_ATE_METHODS
    assert len(report["ite_unit_results"]) == 2 * 16 * 3 * 5 + 2 * 16
    assert len(report["ate_graph_results"]) == 2 * 3 * 7 + 2
    pfn_main = next(row for row in report["main_ite_table"] if row["method"] == "PFN with interference")
    ldr_main = next(
        row for row in report["main_ite_table"]
        if row["method"] == "Localized DR-Lasso"
    )
    assert pfn_main["evaluation_subset"] == "all_targets"
    assert ldr_main["evaluation_subset"] == "supported_targets"
    nsi_main = next(row for row in report["main_ite_table"] if row["method"] == NSI_METHOD_NAME)
    assert nsi_main["evaluation_subset"] == "supported_targets"
    assert "spillover_mean_effective_sample_size" in nsi_main
    assert "spillover_mean_local_radius" in nsi_main
    assert "spillover_mean_effective_sample_size" in ldr_main
    assert "spillover_mean_local_radius" in ldr_main
    assert "spillover_mean_truth_all_targets" in ldr_main
    for method in ("HyperSCI",):
        row = next(item for item in report["main_ite_table"] if item["method"] == method)
        assert row["evaluation_subset"] == "all_targets"
        method_rows = [
            item for item in report["ite_unit_results"] if item["method"] == method
        ]
        assert method_rows
        assert all(np.isfinite(item["estimate"]) for item in method_rows)

    cpf_ite = [
        row for row in report["ite_unit_results"]
        if row["method"] == "causalPFN"
    ]
    assert len(cpf_ite) == 2 * 16
    assert {row["effect"] for row in cpf_ite} == {"direct"}
    assert all(row["supported"] for row in cpf_ite)
    cpf_ate = [
        row for row in report["ate_graph_results"]
        if row["method"] == "causalPFN"
    ]
    assert len(cpf_ate) == 2
    assert {row["effect"] for row in cpf_ate} == {"direct"}
    cpf_main = next(
        row for row in report["main_ite_table"]
        if row["method"] == "causalPFN"
    )
    assert np.isfinite(cpf_main["direct_rmse"])
    assert np.isnan(cpf_main["spillover_rmse"])
    assert np.isnan(cpf_main["total_rmse"])

    pfn_ite = [
        row for row in report["ite_unit_results"]
        if row["method"] == "PFN with interference"
    ]
    assert all(abs(row["error"]) < 1e-5 for row in pfn_ite)
    assert all(row["degree"] > 0 for row in pfn_ite)
    pfn_ate = [
        row for row in report["ate_graph_results"]
        if row["method"] == "PFN with interference"
    ]
    assert all(abs(row["error"]) < 1e-5 for row in pfn_ate)
    assert all(row["average_degree"] > 0 for row in pfn_ate)
    tnet_ite = [
        row for row in report["ite_unit_results"]
        if row["method"] == TNET_METHOD_NAME
    ]
    assert len(tnet_ite) == 2 * 16 * 3
    assert all(row["dr_guarantee"] is False for row in tnet_ite)
    for method in ("Haj", "F"):
        method_ate = [
            row for row in report["ate_graph_results"]
            if row["method"] == method
        ]
        assert len(method_ate) == 2 * 3
        assert any(row["effect"] == "direct" and row["supported"] for row in method_ate)
        assert all(bool(np.isfinite(row["estimate"])) == bool(row["supported"]) for row in method_ate)

    tnet_ate = [
        row for row in report["ate_graph_results"]
        if row["method"] == TNET_METHOD_NAME
    ]
    assert len(tnet_ate) == 2 * 3
    anchor = report["oracle_semiparametric_bound"]
    assert len(anchor) == 2 * 3
    assert all(row["variance"] >= 0.0 for row in anchor)
    bound_summary = report["oracle_semiparametric_bound_summary"]
    assert len(bound_summary) == 3
    assert {row["effect"] for row in bound_summary} == {"direct", "spillover", "total"}
    assert all(row["rmse_bound"] >= 0.0 for row in bound_summary)
    assert all("pfn_to_bound_ratio" in row for row in bound_summary)
    assert all("tnet_to_bound_ratio" in row for row in bound_summary)


def test_save_unified_benchmark_writes_expected_files_and_columns(tmp_path: Path) -> None:
    report = {
        "main_ite_methods": list(MAIN_ITE_METHODS),
        "main_ate_methods": list(MAIN_ATE_METHODS),
        "ite_unit_results": [
            {
                "dataset_id": 1, "unit_id": 1, "effect": "direct",
                "method": "PFN with interference", "truth": 0.5, "estimate": 0.6,
                "error": 0.1, "supported": True,
                "effective_sample_size": np.nan, "local_radius": np.nan,
            }
        ],
        "ite_summary": [
            {
                "effect": "direct", "method": "PFN with interference", "evaluation_subset": "all_targets",
                "mean_truth": 0.5, "mean_estimate": 0.6, "bias": 0.1,
                "mae": 0.1, "rmse": 0.1, "n_total": 1,
                "n_supported": 1, "support_rate": 1.0,
            }
        ],
        "ate_graph_results": [
            {
                "dataset_id": 1, "effect": "direct", "method": "PFN with interference",
                "truth": 0.5, "estimate": 0.6, "error": 0.1,
                "supported": True,
            }
        ],
        "ate_summary": [
            {
                "effect": "direct", "method": "PFN with interference", "mean_truth": 0.5,
                "mean_estimate": 0.6, "bias": 0.1, "mae": 0.1,
                "rmse": 0.1, "n_total": 1, "n_supported": 1,
                "support_rate": 1.0,
            }
        ],
        "oracle_semiparametric_bound": [
            {"dataset_id": 1, "effect": "direct", "truth": 0.5, "variance": 0.01, "standard_error": 0.1}
        ],
        "oracle_semiparametric_bound_summary": [
            {"effect": "direct", "mean_variance": 0.01, "rmse_bound": 0.1, "pfn_rmse": 0.1, "pfn_to_bound_ratio": 1.0, "tnet_rmse": 0.2, "tnet_to_bound_ratio": 2.0}
        ],
    }
    paths = save_unified_benchmark(report, tmp_path)
    expected = {
        "ite_unit_results.csv", "ite_summary.csv", "ate_graph_results.csv",
        "ate_summary.csv", "main_ite_table.csv", "main_ate_table.csv",
        "oracle_semiparametric_bound.csv",
        "benchmark_summary.json", "benchmark_readable.txt",
    }
    assert {path.name for path in paths.values()} == expected
    assert all(path.exists() for path in paths.values())

    with (tmp_path / "benchmark_summary.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["main_ite_methods"] == list(MAIN_ITE_METHODS)
    assert saved["main_ate_methods"] == list(MAIN_ATE_METHODS)
    with (tmp_path / "ite_unit_results.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert {"truth", "estimate", "error", "supported"}.issubset(row)


def test_summary_retains_truth_even_when_method_is_unsupported() -> None:
    from pfn_interference.metrics.unified import _metric_rows

    rows = [
        {
            "dataset_id": 1,
            "unit_id": 1,
            "effect": "spillover",
            "method": "Localized DR-Lasso",
            "truth": -0.2,
            "estimate": float("nan"),
            "effective_sample_size": 1.0,
            "local_radius": 0.8,
        },
        {
            "dataset_id": 1,
            "unit_id": 2,
            "effect": "spillover",
            "method": "Localized DR-Lasso",
            "truth": 0.4,
            "estimate": 0.5,
            "effective_sample_size": 4.0,
            "local_radius": 0.2,
        },
    ]
    summary = _metric_rows(
        rows,
        methods=("Localized DR-Lasso",),
        effects=("spillover",),
        subset_name="all_targets",
    )[0]

    assert summary["mean_truth_all_targets"] == pytest.approx(0.1)
    assert summary["mean_truth"] == pytest.approx(0.4)
    assert summary["mean_estimate"] == pytest.approx(0.5)
    assert summary["support_rate"] == pytest.approx(0.5)
    assert summary["mean_effective_sample_size"] == pytest.approx(4.0)
    assert summary["median_effective_sample_size"] == pytest.approx(4.0)
    assert summary["mean_local_radius"] == pytest.approx(0.2)

class SharedMuOraclePFN(torch.nn.Module):
    def forward(self,tokens,queries,adjacency):
        d,s=queries.unbind(-1)
        mean=2+3*d-s
        return dict(gmm_pi=torch.ones_like(mean[...,None]),gmm_mu=mean[...,None],gmm_sigma=torch.ones_like(mean[...,None]))


def test_pfn_uses_binary_arm_mu_queries_and_shared_differences():
    batch={'tokens':torch.zeros(1,3,5),'adjacency':torch.zeros(1,3,3)}
    got=predict_pfn_majority_ite(model=SharedMuOraclePFN(),batch=batch,treatment_prob=.5)
    torch.testing.assert_close(got,torch.tensor([[[3.,-1.,2.]]]).expand(1,3,3))


def test_summary_carries_network_density_diagnostic() -> None:
    from pfn_interference.metrics.unified import _metric_rows

    rows = [
        {"dataset_id": 1, "unit_id": 1, "effect": "direct", "method": "PFN with interference", "truth": 0.5, "estimate": 0.5, "degree": 6},
        {"dataset_id": 1, "unit_id": 2, "effect": "direct", "method": "PFN with interference", "truth": 0.5, "estimate": 0.5, "degree": 10},
    ]
    summary = _metric_rows(rows, methods=("PFN with interference",), effects=("direct",), subset_name="all_targets")[0]
    assert summary["mean_degree"] == pytest.approx(8.0)


def test_ite_summary_uses_explicit_all_supported_and_common_subset_labels() -> None:
    config = demo.DataConfig(
        n_units=12,
        er_edge_probability=0.30,
        tau_episode_sd=0.0,
        tau_unit_sd=0.0,
        gamma_sd=0.0,
        eta_mean=0.0,
        eta_sd=0.0,
        noise_sd=0.0,
    )
    batch = demo.generate_batch(config, batch_size=1, seed=88, device=torch.device("cpu"))
    model = LinearOraclePFN(tau=config.tau_mean, gamma=config.gamma_mean)
    report = evaluate_unified_benchmark(
        model=model,
        batch=batch,
        treatment_prob=0.5,
        ate_bandwidth=2,
        ate_ridge=1e-6,
        localized_bandwidth=0.30,
        localized_min_neighbors=3,
        localized_mode="knn",
        localized_lasso_alpha=0.0,
        localized_nuisance_ridge=0.0,
        localized_cross_fit_folds=2,
        localized_arm_samples=1,
        standard_ite_epochs=5,
        standard_ite_hidden_dim=12,
        tnet_epochs=2,
        tnet_hidden_dim=8,
        tnet_grid_size=4,
        tnet_spline_basis=7,
        seed=99,
    )
    labels = {(row["method"], row["evaluation_subset"]) for row in report["ite_summary"]}
    assert ("PFN with interference", "all_targets") in labels
    assert ("Localized DR-Lasso", "supported_targets") in labels
    assert (NSI_METHOD_NAME, "supported_targets") in labels
    assert ("PFN with interference", "common_support") in labels
    assert ("Localized DR-Lasso", "common_support") in labels
    assert not any(label in {"all-target", "common-support"} for _, label in labels)


class RejectOutOfRangeExposurePFN(torch.nn.Module):
    """Return NaN whenever the evaluator sends any exposure outside [0, 1]."""

    def __init__(self):
        super().__init__()
        self.training = False

    def forward(self, tokens, queries, adjacency):
        del tokens, adjacency
        t,s=queries.unbind(-1)
        mean=.5*t+.3*s
        invalid=((t!=0)&(t!=1))|((s!=0)&(s!=1))
        mean = torch.where(invalid, torch.full_like(mean, float("nan")), mean)
        return {
            "gmm_pi": torch.ones(*mean.shape, 1, device=mean.device),
            "gmm_mu": mean.unsqueeze(-1),
            "gmm_sigma": torch.ones(*mean.shape, 1, device=mean.device) * 0.1,
        }


def test_pfn_arm_integration_never_sends_impossible_count_exposures() -> None:
    # Mixed degrees force the global count support to contain counts that are
    # impossible for the lower-degree unit. Those positions must be replaced
    # by a legal placeholder before the query reaches the model; their design
    # weights remain exactly zero for that unit.
    batch = {
        "tokens": torch.zeros((1, 2, 4), dtype=torch.float32),
        "adjacency": torch.zeros((1, 2, 2), dtype=torch.float32),
        "degree": torch.tensor([[2.0, 6.0]], dtype=torch.float32),
    }
    estimate = predict_pfn_majority_ite(
        model=RejectOutOfRangeExposurePFN(),
        batch=batch,
        treatment_prob=0.5,
        query_chunk_size=64,
    )

    assert torch.isfinite(estimate).all()
