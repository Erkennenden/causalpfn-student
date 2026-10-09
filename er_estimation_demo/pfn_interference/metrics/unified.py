"""Unified ITE/ATE evaluation on the same synthetic graph episodes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from pfn_interference.baselines.causalpfn_direct import (
    CAUSALPFN_METHOD_NAME,
    predict_causalpfn_direct_on_batch,
)
from pfn_interference.baselines.localized_walsh import (
    ITE_METHOD_NAME,
    evaluate_localized_dr_lasso_majority_ite,
)
from pfn_interference.baselines.ns_pcr import (
    NSIConfig,
    NSI_METHOD_NAME,
    evaluate_nsi_ite_on_batch,
)
from pfn_interference.baselines.ite_standard import (
    LITERATURE_ITE_METHOD_NAMES,
    evaluate_literature_ite_baselines_on_batch,
)
from pfn_interference.baselines.tnet import TNET_METHOD_NAME, TNetConfig
from pfn_interference.baselines.runner import (
    ATE_METHOD_NAMES,
    MAIN_ATE_METHOD_NAMES,
    EFFECT_NAMES,
    run_ate_baselines_on_batch,
)
from pfn_interference.estimands import conditional_count_weights
from pfn_interference.baselines.localized_config import LOCALIZED_PROTOCOL


BASE_MAIN_ITE_METHODS = ("PFN with interference", ITE_METHOD_NAME, NSI_METHOD_NAME) + LITERATURE_ITE_METHOD_NAMES + (TNET_METHOD_NAME,)
BASE_MAIN_ATE_METHODS = ("PFN with interference",) + MAIN_ATE_METHOD_NAMES
MAIN_ITE_METHODS = BASE_MAIN_ITE_METHODS + (CAUSALPFN_METHOD_NAME,)
MAIN_ATE_METHODS = BASE_MAIN_ATE_METHODS + (CAUSALPFN_METHOD_NAME,)


def _posterior_mean(predictions: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (predictions["gmm_pi"] * predictions["gmm_mu"]).sum(dim=-1)


def _design_weight_tensors(
    degree: torch.Tensor,
    treatment_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    degree_cpu = torch.round(degree.detach().cpu()).to(torch.int64)
    max_degree = int(degree_cpu.max().item())
    shape = (*degree_cpu.shape, max_degree + 1)
    low = torch.zeros(shape, dtype=torch.float32)
    high = torch.zeros(shape, dtype=torch.float32)
    # Fill via flattened [B*N, K] views rather than indexing a 3-D tensor
    # with a 2-D boolean mask.  The latter works on the CPU PyTorch build used
    # by our unit tests, but MUSA PyTorch rejects it for real [B,N] batches.
    # Flattening makes the indexing semantics explicit and backend-independent.
    flat_degree = degree_cpu.reshape(-1)
    flat_low = low.reshape(-1, max_degree + 1)
    flat_high = high.reshape(-1, max_degree + 1)
    for value in torch.unique(flat_degree):
        d = int(value.item())
        flat_mask = flat_degree == d
        low_values = torch.tensor(
            conditional_count_weights(d, treatment_prob, 0), dtype=torch.float32
        )
        high_values = torch.tensor(
            conditional_count_weights(d, treatment_prob, 1), dtype=torch.float32
        )
        flat_low[flat_mask, : d + 1] = low_values
        flat_high[flat_mask, : d + 1] = high_values
    return low.to(degree.device), high.to(degree.device)


def predict_pfn_majority_ite(*, model, batch, treatment_prob, query_chunk_size=64):
    """Four exact-arm CEPO queries, followed by three shared-mean differences.

    query_chunk_size is retained for callers; there is no exposure-grid
    integration because the model is trained directly on design-averaged mu.
    """
    from pfn_interference.cepo import predict_mu_distributions, effects_from_mu
    if not 0 < treatment_prob < 1 or query_chunk_size <= 0:
        raise ValueError("Invalid treatment probability or query chunk size")
    return effects_from_mu(_posterior_mean(predict_mu_distributions(model, batch)))


def _metric_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str],
    effects: Sequence[str],
    subset_name: str,
    subset_keys: set[tuple[int, int, str]] | None = None,
) -> list[dict]:
    result: list[dict] = []
    for effect in effects:
        effect_rows = [row for row in rows if row["effect"] == effect]
        if subset_keys is not None:
            effect_rows = [
                row
                for row in effect_rows
                if (int(row["dataset_id"]), int(row.get("unit_id", 0)), effect)
                in subset_keys
            ]
        for method in methods:
            selected = [row for row in effect_rows if row["method"] == method]
            all_truths = np.asarray(
                [float(row["truth"]) for row in selected], dtype=np.float64
            )
            mean_truth_all_targets = (
                float(all_truths.mean()) if all_truths.size else float("nan")
            )
            degree_values = []
            for row in selected:
                value = row.get("degree", row.get("average_degree", float("nan")))
                value = float(value)
                if math.isfinite(value):
                    degree_values.append(value)
            mean_degree = (
                float(np.mean(degree_values)) if degree_values else float("nan")
            )
            truths: list[float] = []
            estimates: list[float] = []
            effective_sample_sizes: list[float] = []
            local_radii: list[float] = []
            for row in selected:
                estimate = float(row["estimate"])
                if not math.isfinite(estimate):
                    continue
                truths.append(float(row["truth"]))
                estimates.append(estimate)
                neff = float(row.get("effective_sample_size", float("nan")))
                radius = float(row.get("local_radius", float("nan")))
                if math.isfinite(neff):
                    effective_sample_sizes.append(neff)
                if math.isfinite(radius):
                    local_radii.append(radius)
            n_total = len(selected)
            n_supported = len(estimates)
            if n_supported:
                truth_a = np.asarray(truths, dtype=np.float64)
                estimate_a = np.asarray(estimates, dtype=np.float64)
                error = estimate_a - truth_a
                metrics = {
                    "mean_truth": float(truth_a.mean()),
                    "mean_estimate": float(estimate_a.mean()),
                    "bias": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                }
            else:
                metrics = {
                    "mean_truth": float("nan"),
                    "mean_estimate": float("nan"),
                    "bias": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                }
            result.append(
                {
                    "effect": effect,
                    "method": method,
                    "evaluation_subset": subset_name,
                    "mean_truth_all_targets": mean_truth_all_targets,
                    "mean_degree": mean_degree,
                    **metrics,
                    "n_total": n_total,
                    "n_supported": n_supported,
                    "support_rate": n_supported / max(n_total, 1),
                    "mean_effective_sample_size": (
                        float(np.mean(effective_sample_sizes))
                        if effective_sample_sizes else float("nan")
                    ),
                    "median_effective_sample_size": (
                        float(np.median(effective_sample_sizes))
                        if effective_sample_sizes else float("nan")
                    ),
                    "mean_local_radius": (
                        float(np.mean(local_radii))
                        if local_radii else float("nan")
                    ),
                }
            )
    return result


def _main_table(
    summary: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    *,
    subset: str = "all_targets",
    method_subsets: Mapping[str, str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        method_subset = (
            method_subsets.get(method, subset) if method_subsets is not None else subset
        )
        row: dict[str, object] = {
            "method": method,
            "evaluation_subset": method_subset,
        }
        for effect in EFFECT_NAMES:
            matches = [
                item
                for item in summary
                if item["method"] == method
                and item["effect"] == effect
                and item.get("evaluation_subset", "all_targets") == method_subset
            ]
            if not matches:
                row[f"{effect}_rmse"] = float("nan")
                row[f"{effect}_mae"] = float("nan")
                row[f"{effect}_bias"] = float("nan")
                row[f"{effect}_support_rate"] = 0.0
                continue
            item = matches[0]
            for metric in (
                "mean_truth_all_targets",
                "mean_truth",
                "mean_estimate",
                "bias",
                "mae",
                "rmse",
                "support_rate",
                "mean_effective_sample_size",
                "median_effective_sample_size",
                "mean_local_radius",
            ):
                if metric in item:
                    row[f"{effect}_{metric}"] = item[metric]
        rows.append(row)
    return rows


def evaluate_unified_benchmark(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    treatment_prob: float,
    ate_bandwidth: int,
    ate_ridge: float,
    localized_bandwidth: float,
    localized_min_neighbors: int,
    localized_mode: str,
    localized_lasso_alpha: float,
    localized_nuisance_ridge: float,
    localized_cross_fit_folds: int,
    localized_arm_samples: int,
    seed: int,
    nsi_pcr_rank_max: int = 6,
    nsi_min_effective_donors: float = 12.0,
    nsi_max_bandwidth: float = 0.50,
    nsi_min_support_mass: float = 0.99,
    nsi_svd_rcond: float = 1.0e-6,
    standard_ite_gps_ridge: float = 1.0e-3,
    standard_ite_epochs: int = 500,
    standard_ite_learning_rate: float = 1.0e-3,
    standard_ite_balance_weight: float = 1.0e-2,
    standard_ite_hidden_dim: int = 32,
    hypersci_arm_samples: int = 4096,
    tnet_epochs: int = 160,
    tnet_hidden_dim: int = 64,
    tnet_grid_size: int = 20,
    tnet_spline_basis: int = 12,
    tnet_learning_rate_1step: float = 1.0e-4,
    tnet_learning_rate_2step: float = 1.0e-2,
    causalpfn_model: object | None = None,
    causalpfn_query_chunk_size: int = 512,
) -> dict:
    from pfn_interference.cepo import (predict_mu_distributions, effects_from_mu, mu_prediction_rows,
                      PREDICTION_PROTOCOL)
    pfn_mu_distribution = predict_mu_distributions(model, batch)
    pfn_mu = _posterior_mean(pfn_mu_distribution)
    pfn_ite = effects_from_mu(pfn_mu)
    mu_rows = mu_prediction_rows(pfn_mu_distribution, batch, treatment_prob)
    oracle_ite = batch["oracle_ite"].detach().cpu()
    degree_cpu = batch["degree"].detach().cpu()
    pfn_cpu = pfn_ite.detach().cpu()
    num_graphs, n_units = pfn_cpu.shape[:2]

    localized = evaluate_localized_dr_lasso_majority_ite(
        batch=batch,
        treatment_prob=treatment_prob,
        bandwidth=localized_bandwidth,
        min_neighbors=localized_min_neighbors,
        localization_mode=localized_mode,
        lasso_alpha=localized_lasso_alpha,
        nuisance_ridge=localized_nuisance_ridge,
        cross_fit_folds=localized_cross_fit_folds,
        arm_samples=localized_arm_samples,
        seed=seed,
    )
    localized_map = {
        (int(row["dataset_id"]), int(row["query_unit_id"]), str(row["effect"])): row
        for row in localized["query_units"]
    }
    nsi_config = NSIConfig(
        pcr_rank_max=nsi_pcr_rank_max,
        min_effective_donors=nsi_min_effective_donors,
        max_bandwidth=nsi_max_bandwidth,
        min_support_mass=nsi_min_support_mass,
        svd_rcond=nsi_svd_rcond,
    )
    nsi = evaluate_nsi_ite_on_batch(
        batch=batch,
        treatment_prob=treatment_prob,
        config=nsi_config,
    )
    nsi_map = {
        (int(row["dataset_id"]), int(row["unit_id"]), str(row["effect"])): row
        for row in nsi["unit_effects"]
    }
    literature = evaluate_literature_ite_baselines_on_batch(
        batch=batch,
        treatment_prob=treatment_prob,
        gps_ridge=standard_ite_gps_ridge,
        neural_epochs=standard_ite_epochs,
        neural_learning_rate=standard_ite_learning_rate,
        neural_balance_weight=standard_ite_balance_weight,
        neural_hidden_dim=standard_ite_hidden_dim,
        arm_samples=hypersci_arm_samples,
        seed=seed,
    )
    literature_map = {
        (
            int(row["dataset_id"]),
            int(row["unit_id"]),
            str(row["effect"]),
            str(row["method"]),
        ): row
        for row in literature["unit_effects"]
    }

    causalpfn = None
    if causalpfn_model is not None:
        causalpfn = predict_causalpfn_direct_on_batch(
            model=causalpfn_model,
            batch=batch,
            treatment_prob=treatment_prob,
            query_chunk_size=causalpfn_query_chunk_size,
        )

    ate_baselines = run_ate_baselines_on_batch(
        batch=batch,
        treatment_prob=treatment_prob,
        bandwidth=ate_bandwidth,
        ridge=ate_ridge,
        tnet_config=TNetConfig(
            hidden_dim=tnet_hidden_dim,
            grid_size=tnet_grid_size,
            spline_basis=tnet_spline_basis,
            epochs=tnet_epochs,
            learning_rate_1step=tnet_learning_rate_1step,
            learning_rate_2step=tnet_learning_rate_2step,
            seed=seed,
        ),
    )
    tnet_map = {
        (int(row["dataset_id"]), int(row["unit_id"]), str(row["effect"])): row
        for row in ate_baselines["tnet_unit_effects"]
    }

    ite_rows: list[dict] = []
    for dataset in range(num_graphs):
        for unit in range(n_units):
            for effect_index, effect in enumerate(EFFECT_NAMES):
                truth = float(oracle_ite[dataset, unit, effect_index])
                estimate = float(pfn_cpu[dataset, unit, effect_index])
                ite_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "unit_id": unit + 1,
                        "effect": effect,
                        "method": "PFN with interference",
                        "truth": truth,
                        "estimate": estimate,
                        "error": estimate - truth,
                        "supported": True,
                        "effective_sample_size": float("nan"),
                        "local_radius": float("nan"),
                        "dr_guarantee": None,
                        "degree": int(round(float(degree_cpu[dataset, unit]))),
                    }
                )
                source = localized_map[(dataset + 1, unit + 1, effect)]
                estimate_ldr = float(source[ITE_METHOD_NAME])
                ite_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "unit_id": unit + 1,
                        "effect": effect,
                        "method": ITE_METHOD_NAME,
                        "truth": truth,
                        "estimate": estimate_ldr,
                        "error": estimate_ldr - truth if math.isfinite(estimate_ldr) else float("nan"),
                        "supported": bool(source["supported"]),
                        "effective_sample_size": float(source["effective_sample_size"]),
                        "local_radius": float(source["local_radius"]),
                        "dr_guarantee": None,
                        "degree": int(source.get("degree", round(float(degree_cpu[dataset, unit])))),
                    }
                )
                nsi_source = nsi_map[(dataset + 1, unit + 1, effect)]
                nsi_estimate = float(nsi_source["estimate"])
                ite_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "unit_id": unit + 1,
                        "effect": effect,
                        "method": NSI_METHOD_NAME,
                        "truth": truth,
                        "estimate": nsi_estimate,
                        "error": nsi_estimate - truth if math.isfinite(nsi_estimate) else float("nan"),
                        "supported": bool(nsi_source["supported"]),
                        "effective_sample_size": float(nsi_source["effective_sample_size"]),
                        "local_radius": float(nsi_source["local_radius"]),
                        "dr_guarantee": None,
                        "degree": int(nsi_source["degree"]),
                    }
                )
                for method in LITERATURE_ITE_METHOD_NAMES:
                    literature_source = literature_map[
                        (dataset + 1, unit + 1, effect, method)
                    ]
                    literature_estimate = float(literature_source["estimate"])
                    ite_rows.append(
                        {
                            "dataset_id": dataset + 1,
                            "unit_id": unit + 1,
                            "effect": effect,
                            "method": method,
                            "truth": truth,
                            "estimate": literature_estimate,
                            "error": (
                                literature_estimate - truth
                                if math.isfinite(literature_estimate)
                                else float("nan")
                            ),
                            "supported": bool(literature_source["supported"]),
                            "effective_sample_size": float("nan"),
                            "local_radius": float("nan"),
                            "dr_guarantee": None,
                            "degree": int(literature_source["degree"]),
                        }
                    )
                tnet_source = tnet_map[(dataset + 1, unit + 1, effect)]
                tnet_estimate = float(tnet_source["estimate"])
                ite_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "unit_id": unit + 1,
                        "effect": effect,
                        "method": TNET_METHOD_NAME,
                        "truth": truth,
                        "estimate": tnet_estimate,
                        "error": tnet_estimate - truth if math.isfinite(tnet_estimate) else float("nan"),
                        "supported": bool(math.isfinite(tnet_estimate)),
                        "effective_sample_size": float("nan"),
                        "local_radius": float("nan"),
                        "dr_guarantee": False,
                        "degree": int(round(float(degree_cpu[dataset, unit]))),
                    }
                )

    if causalpfn is not None:
        causalpfn_ite = causalpfn.ite_direct.detach().cpu()
        for dataset in range(num_graphs):
            for unit in range(n_units):
                truth = float(oracle_ite[dataset, unit, 0])
                estimate = float(causalpfn_ite[dataset, unit])
                ite_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "unit_id": unit + 1,
                        "effect": "direct",
                        "method": CAUSALPFN_METHOD_NAME,
                        "truth": truth,
                        "estimate": estimate,
                        "error": estimate - truth,
                        "supported": bool(math.isfinite(estimate)),
                        "effective_sample_size": float("nan"),
                        "local_radius": float("nan"),
                        "dr_guarantee": None,
                        "degree": int(round(float(degree_cpu[dataset, unit]))),
                    }
                )

    # Primary ITE metrics use method-appropriate target sets: PFN and the
    # literature outcome-surface baselines are evaluated on every target, while
    # Localized DR-Lasso and NSI are evaluated only where their support rules hold.
    # Common-support rows still provide the apples-to-apples comparison on the
    # exact targets supported by Localized DR-Lasso.
    ite_summary = _metric_rows(
        ite_rows,
        methods=("PFN with interference",) + LITERATURE_ITE_METHOD_NAMES + (TNET_METHOD_NAME,),
        effects=EFFECT_NAMES,
        subset_name="all_targets",
    )
    ite_summary.extend(
        _metric_rows(
            ite_rows,
            methods=(ITE_METHOD_NAME, NSI_METHOD_NAME),
            effects=EFFECT_NAMES,
            subset_name="supported_targets",
        )
    )
    if causalpfn is not None:
        ite_summary.extend(
            _metric_rows(
                ite_rows,
                methods=(CAUSALPFN_METHOD_NAME,),
                effects=("direct",),
                subset_name="all_targets",
            )
        )
    common_keys = {
        (int(row["dataset_id"]), int(row["unit_id"]), str(row["effect"]))
        for row in ite_rows
        if row["method"] == ITE_METHOD_NAME and bool(row["supported"])
    }
    ite_summary.extend(
        _metric_rows(
            ite_rows,
            methods=BASE_MAIN_ITE_METHODS,
            effects=EFFECT_NAMES,
            subset_name="common_support",
            subset_keys=common_keys,
        )
    )

    baseline_map = {
        (int(row["dataset_id"]), str(row["effect"])): row
        for row in ate_baselines["graph_effects"]
    }
    pfn_ate = pfn_cpu.mean(dim=1)
    oracle_ate = batch["oracle_ate"].detach().cpu()
    ate_rows: list[dict] = []
    for dataset in range(num_graphs):
        for effect_index, effect in enumerate(EFFECT_NAMES):
            truth = float(oracle_ate[dataset, effect_index])
            pfn_estimate = float(pfn_ate[dataset, effect_index])
            ate_rows.append(
                {
                    "dataset_id": dataset + 1,
                    "effect": effect,
                    "method": "PFN with interference",
                    "truth": truth,
                    "estimate": pfn_estimate,
                    "error": pfn_estimate - truth,
                    "supported": True,
                    "average_degree": float(degree_cpu[dataset].float().mean()),
                }
            )
            source = baseline_map[(dataset + 1, effect)]
            for method in ATE_METHOD_NAMES:
                estimate = float(source[method])
                ate_rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "effect": effect,
                        "method": method,
                        "truth": truth,
                        "estimate": estimate,
                        "error": estimate - truth if math.isfinite(estimate) else float("nan"),
                        "supported": bool(math.isfinite(estimate)),
                        "average_degree": float(source.get("average_degree", degree_cpu[dataset].float().mean())),
                    }
                )
    if causalpfn is not None:
        causalpfn_ate = causalpfn.ate_direct.detach().cpu()
        for dataset in range(num_graphs):
            truth = float(oracle_ate[dataset, 0])
            estimate = float(causalpfn_ate[dataset])
            ate_rows.append(
                {
                    "dataset_id": dataset + 1,
                    "effect": "direct",
                    "method": CAUSALPFN_METHOD_NAME,
                    "truth": truth,
                    "estimate": estimate,
                    "error": estimate - truth,
                    "supported": bool(math.isfinite(estimate)),
                    "average_degree": float(degree_cpu[dataset].float().mean()),
                }
            )

    ate_summary = _metric_rows(
        ate_rows,
        methods=("PFN with interference",) + ATE_METHOD_NAMES,
        effects=EFFECT_NAMES,
        subset_name="all_targets",
    )
    if causalpfn is not None:
        ate_summary.extend(
            _metric_rows(
                ate_rows,
                methods=(CAUSALPFN_METHOD_NAME,),
                effects=("direct",),
                subset_name="all_targets",
            )
        )
    for row in ate_summary:
        row.pop("evaluation_subset", None)

    active_ite_methods = MAIN_ITE_METHODS if causalpfn is not None else BASE_MAIN_ITE_METHODS
    active_ate_methods = MAIN_ATE_METHODS if causalpfn is not None else BASE_MAIN_ATE_METHODS
    oracle_bound_rows = list(ate_baselines["semiparametric_anchor"])
    oracle_bound_summary = _semiparametric_bound_summary(oracle_bound_rows, ate_summary)

    return {
        "prediction_protocol": PREDICTION_PROTOCOL,
        "pfn_prediction_target": "mu00,mu01,mu10,mu11; exact majority-arm conditional expected outcomes",
        "pfn_effect_uncertainty": "not computed: marginal CEPOs do not specify a joint posterior",
        "pfn_additivity_max_abs": float((pfn_cpu[...,2]-pfn_cpu[...,0]-pfn_cpu[...,1]).abs().max()),
        "pfn_mu_predictions": mu_rows,
        "main_ite_methods": list(active_ite_methods),
        "main_ate_methods": list(active_ate_methods),
        "ite_unit_results": ite_rows,
        "ite_summary": ite_summary,
        "ate_graph_results": ate_rows,
        "ate_summary": ate_summary,
        "oracle_semiparametric_bound": oracle_bound_rows,
        "oracle_semiparametric_bound_summary": oracle_bound_summary,
        "main_ite_table": _main_table(
            ite_summary,
            active_ite_methods,
            method_subsets={
                "PFN with interference": "all_targets",
                ITE_METHOD_NAME: "supported_targets",
                NSI_METHOD_NAME: "supported_targets",
                **{method: "all_targets" for method in LITERATURE_ITE_METHOD_NAMES},
                TNET_METHOD_NAME: "all_targets",
                **({CAUSALPFN_METHOD_NAME: "all_targets"} if causalpfn is not None else {}),
            },
        ),
        "main_ate_table": _main_table(ate_summary, active_ate_methods),
        "causalpfn_diagnostics": {
            "enabled": causalpfn is not None,
            "query_chunk_size": int(causalpfn_query_chunk_size),
            "context": "factual retrieval within each graph; covariates=[X]",
            "effect": "direct only",
        },
        "localized_diagnostics": {
            "method": ITE_METHOD_NAME,
            "implementation_protocol": LOCALIZED_PROTOCOL,
            "inference_scope": "ITE only; empirical point estimates",
            "estimand_bridge": localized["estimand_bridge"],
            "lasso_alpha": localized_lasso_alpha,
            "nuisance_shrinkage": localized_nuisance_ridge,
            "cross_fit_folds": localized_cross_fit_folds,
            "nuisance_mean_scope": "training fold only during cross-fitting",
            "arm_samples": localized_arm_samples,
            "localization_mode": localized_mode,
            "bandwidth": localized_bandwidth,
            "min_neighbors": localized_min_neighbors,
            "localization_geometry": localized["localization_geometry"],
            "localization_warnings": localized["localization_warnings"],
        },
        "nsi_diagnostics": {
            **nsi["config"],
            "method": NSI_METHOD_NAME,
            "positioning": nsi["positioning"],
            "primary_subset": "supported_targets",
        },
        "literature_ite_diagnostics": literature["training"],
        "hypersci_integration_diagnostics": [
            {key: row[key] for key in ("dataset_id", "unit_id", "effect", "integration_mcse")}
            for row in literature["unit_effects"]
        ],
        "tnet_diagnostics": {
            "method": TNET_METHOD_NAME,
            "double_robustness": "average effects only; no individual-effect DR guarantee",
            "epochs": int(tnet_epochs),
            "hidden_dim": int(tnet_hidden_dim),
            "grid_size": int(tnet_grid_size),
            "spline_basis": int(tnet_spline_basis),
            "training": ate_baselines["tnet_training"],
        },
        "ate_diagnostics": {
            **ate_baselines["regression_config"],
            "oracle_anchor": "true-nuisance EIF with network-HAC variance; excluded from method rankings",
        },
        "estimands": {
            "direct": "mu_i(1,0)-mu_i(0,0)",
            "spillover": "mu_i(1,1)-mu_i(1,0)",
            "total": "mu_i(1,1)-mu_i(0,0)",
            "majority_arm": "S_i=1{K_i>floor(d_i/2)}",
        },
    }


def _semiparametric_bound_summary(
    anchor_rows: Sequence[Mapping[str, object]],
    ate_summary: Sequence[Mapping[str, object]],
) -> list[dict]:
    rows: list[dict] = []
    for effect in EFFECT_NAMES:
        selected = [row for row in anchor_rows if str(row["effect"]) == effect]
        variances = np.asarray([float(row["variance"]) for row in selected], dtype=np.float64)
        mean_variance = float(np.mean(variances)) if variances.size else float("nan")
        rmse_bound = float(np.sqrt(mean_variance)) if math.isfinite(mean_variance) else float("nan")

        def method_rmse(method: str) -> float:
            matches = [
                row for row in ate_summary
                if str(row.get("method")) == method and str(row.get("effect")) == effect
            ]
            return float(matches[0]["rmse"]) if matches else float("nan")

        pfn_rmse = method_rmse("PFN with interference")
        tnet_rmse = method_rmse(TNET_METHOD_NAME)
        def ratio(value: float) -> float:
            if not math.isfinite(value) or not math.isfinite(rmse_bound) or rmse_bound <= 0.0:
                return float("nan")
            return value / rmse_bound

        rows.append(
            {
                "effect": effect,
                "mean_variance": mean_variance,
                "rmse_bound": rmse_bound,
                "pfn_rmse": pfn_rmse,
                "pfn_to_bound_ratio": ratio(pfn_rmse),
                "tnet_rmse": tnet_rmse,
                "tnet_to_bound_ratio": ratio(tnet_rmse),
                "interpretation": "oracle true-nuisance EIF with network-HAC variance; non-ranked semiparametric anchor",
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _format(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.6f}"


def _readable(report: Mapping[str, object]) -> str:
    lines = [
        "UNIFIED MAJORITY-ARM ITE + ATE BENCHMARK",
        "ITE methods: " + ", ".join(str(v) for v in report.get("main_ite_methods", [])),
        "ATE methods: " + ", ".join(str(v) for v in report.get("main_ate_methods", [])),
        "Spillover estimates retain their signed values; no positivity clipping is applied.",
        "",
        "ITE MAIN TABLE",
    ]
    for row in report.get("main_ite_table", []):
        lines.append(f"[{row['method']}]")
        for effect in EFFECT_NAMES:
            lines.append(
                f"  {effect}: truth_all={_format(row.get(f'{effect}_mean_truth_all_targets'))}, "
                f"truth_supported={_format(row.get(f'{effect}_mean_truth'))}, "
                f"estimate={_format(row.get(f'{effect}_mean_estimate'))}, "
                f"bias={_format(row.get(f'{effect}_bias'))}, "
                f"MAE={_format(row.get(f'{effect}_mae'))}, "
                f"RMSE={_format(row.get(f'{effect}_rmse'))}, "
                f"support={_format(row.get(f'{effect}_support_rate'))}, "
                f"mean_n_eff={_format(row.get(f'{effect}_mean_effective_sample_size'))}"
            )
    lines.extend(["", "ATE MAIN TABLE"])
    for row in report.get("main_ate_table", []):
        lines.append(f"[{row['method']}]")
        for effect in EFFECT_NAMES:
            lines.append(
                f"  {effect}: truth_all={_format(row.get(f'{effect}_mean_truth_all_targets'))}, "
                f"truth_supported={_format(row.get(f'{effect}_mean_truth'))}, "
                f"estimate={_format(row.get(f'{effect}_mean_estimate'))}, "
                f"bias={_format(row.get(f'{effect}_bias'))}, "
                f"MAE={_format(row.get(f'{effect}_mae'))}, "
                f"RMSE={_format(row.get(f'{effect}_rmse'))}, "
                f"support={_format(row.get(f'{effect}_support_rate'))}"
            )
    lines.extend(["", "ORACLE SEMIPARAMETRIC EIF/HAC ANCHOR"])
    for row in report.get("oracle_semiparametric_bound_summary", []):
        lines.append(
            f"  {row['effect']}: RMSE_bound={_format(row.get('rmse_bound'))}, "
            f"PFN with interference RMSE={_format(row.get('pfn_rmse'))}, PFN with interference/bound={_format(row.get('pfn_to_bound_ratio'))}, "
            f"Tnet_RMSE={_format(row.get('tnet_rmse'))}, Tnet/bound={_format(row.get('tnet_to_bound_ratio'))}"
        )
    return "\n".join(lines) + "\n"


def save_unified_benchmark(
    report: Mapping[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ite_unit_results": output_dir / "ite_unit_results.csv",
        "ite_summary": output_dir / "ite_summary.csv",
        "ate_graph_results": output_dir / "ate_graph_results.csv",
        "ate_summary": output_dir / "ate_summary.csv",
        "main_ite_table": output_dir / "main_ite_table.csv",
        "main_ate_table": output_dir / "main_ate_table.csv",
        "oracle_semiparametric_bound": output_dir / "oracle_semiparametric_bound.csv",
        "benchmark_summary": output_dir / "benchmark_summary.json",
        "benchmark_readable": output_dir / "benchmark_readable.txt",
    }
    if report.get("pfn_mu_predictions"):
        paths["pfn_mu_predictions"] = output_dir / "pfn_mu_predictions.csv"
        _write_csv(report["pfn_mu_predictions"], paths["pfn_mu_predictions"])
    if report.get("hypersci_integration_diagnostics"):
        paths["hypersci_integration_diagnostics"] = output_dir / "hypersci_integration_diagnostics.csv"
        _write_csv(report["hypersci_integration_diagnostics"], paths["hypersci_integration_diagnostics"])
    _write_csv(report["ite_unit_results"], paths["ite_unit_results"])
    _write_csv(report["ite_summary"], paths["ite_summary"])
    _write_csv(report["ate_graph_results"], paths["ate_graph_results"])
    _write_csv(report["ate_summary"], paths["ate_summary"])
    main_ite = report.get("main_ite_table") or _main_table(
        report["ite_summary"], MAIN_ITE_METHODS
    )
    main_ate = report.get("main_ate_table") or _main_table(
        report["ate_summary"], MAIN_ATE_METHODS
    )
    _write_csv(main_ite, paths["main_ite_table"])
    _write_csv(main_ate, paths["main_ate_table"])
    bound_rows = report.get("oracle_semiparametric_bound_summary") or report.get("oracle_semiparametric_bound")
    if not bound_rows:
        raise ValueError("oracle semiparametric bound rows are required.")
    _write_csv(bound_rows, paths["oracle_semiparametric_bound"])
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"ite_unit_results", "ate_graph_results", "hypersci_integration_diagnostics", "pfn_mu_predictions"}
    }
    with paths["benchmark_summary"].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    paths["benchmark_readable"].write_text(_readable({**report, "main_ite_table": main_ite, "main_ate_table": main_ate}), encoding="utf-8")
    return paths
