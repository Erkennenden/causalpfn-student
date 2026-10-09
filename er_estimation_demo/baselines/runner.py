"""Paper-facing graph-level ATE baselines for the unified majority-arm target."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

from estimands import arm_mean_exposure, arm_probability
from .auxiliary_adjustment.estimators import (
    estimate_ht_hajek,
    estimate_weighted_regression,
    solve_network_adjustment,
)
from .auxiliary_adjustment.features import build_majority_phi0_g2_exact
from .semiparametric import (
    oracle_majority_arm_nuisance,
    oracle_semiparametric_anchor,
)
from .tnet import (
    TNET_METHOD_NAME,
    TNetConfig,
    fit_tnet_graph,
    predict_tnet_majority_effects,
)


EFFECT_NAMES = ("direct", "spillover", "total")
REG_NET_METHOD_NAME = "reg-net"
ATE_METHOD_NAMES = ("HT", "Haj", "F", "L", REG_NET_METHOD_NAME, TNET_METHOD_NAME)
MAIN_ATE_METHOD_NAMES = ("HT", "Haj", "F", "L", REG_NET_METHOD_NAME, TNET_METHOD_NAME)


def majority_arm_propensities(
    degree: np.ndarray,
    *,
    treatment_prob: float,
) -> np.ndarray:
    """Return P((D_i,S_i)=arm) for fixed codes 00,10,01,11."""

    degree = np.asarray(degree)
    if degree.ndim != 1:
        raise ValueError("degree must be one-dimensional.")
    if np.any(degree <= 0) or not np.allclose(degree, np.rint(degree)):
        raise ValueError("degree must contain positive integers.")
    p = float(treatment_prob)
    if not 0.0 < p < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")
    result = np.empty((degree.size, 4), dtype=np.float64)
    for index, value in enumerate(degree.astype(np.int64).tolist()):
        low = arm_probability(value, p, 0)
        high = arm_probability(value, p, 1)
        result[index] = [
            (1.0 - p) * low,   # 00
            p * low,           # 10
            (1.0 - p) * high,  # 01
            p * high,          # 11
        ]
    return result


def _majority_state_codes(
    treatment: np.ndarray,
    neighbor_counts: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=np.int64)
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.int64)
    degree = np.asarray(degree, dtype=np.int64)
    threshold = degree // 2
    majority = (neighbor_counts > threshold).astype(np.int64)
    return majority * 2 + treatment


def _majority_hac_kernel(adjacency: np.ndarray, bandwidth: int) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    distances = shortest_path(
        np.asarray(adjacency, dtype=np.float64),
        directed=False,
        unweighted=True,
    )
    return (distances < int(bandwidth)).astype(np.float64)


def _ate_metric(rows: list[dict], method: str, effect: str) -> dict:
    selected = [row for row in rows if row["effect"] == effect]
    truth: list[float] = []
    estimate: list[float] = []
    for row in selected:
        value = float(row.get(method, float("nan")))
        if not np.isfinite(value):
            continue
        truth.append(float(row["truth"]))
        estimate.append(value)
    n_total = len(selected)
    n_supported = len(estimate)
    if not estimate:
        return {
            "mean_true": float("nan"),
            "mean_estimate": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_total": n_total,
            "n_supported": 0,
            "support_rate": 0.0 if n_total else float("nan"),
        }
    truth_a = np.asarray(truth, dtype=np.float64)
    estimate_a = np.asarray(estimate, dtype=np.float64)
    error = estimate_a - truth_a
    return {
        "mean_true": float(truth_a.mean()),
        "mean_estimate": float(estimate_a.mean()),
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "n_total": n_total,
        "n_supported": n_supported,
        "support_rate": n_supported / max(n_total, 1),
    }


def run_ate_baselines_on_batch(
    *,
    batch: Mapping[str, torch.Tensor],
    treatment_prob: float,
    bandwidth: int,
    ridge: float,
    tnet_config: TNetConfig | None = None,
) -> Dict[str, object]:
    """Estimate majority-arm ATE contrasts with HT, Hajek, Fisher, Lin, reg-net, and TNet."""

    required = {
        "adjacency",
        "observed_treatment",
        "observed_exposure",
        "x",
        "y_obs",
        "oracle_ate",
    }
    # Nonlinear generators supply exact integrated arm means; never pretend
    # that their outcomes are linear in exposure using fake tau/gamma values.
    required.update({'oracle_arm_means'} if 'oracle_arm_means' in batch else
                    {'structural_baseline', 'tau', 'gamma', 'eta'})
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"shared test batch is missing keys: {sorted(missing)}")
    arrays = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in batch.items()
        if key in required
    }
    adjacency = arrays["adjacency"].astype(np.uint8, copy=False)
    treatment = arrays["observed_treatment"].astype(np.int64, copy=False)
    exposure = arrays["observed_exposure"].astype(np.float64, copy=False)
    x = arrays["x"].astype(np.float64, copy=False)
    outcome = arrays["y_obs"].astype(np.float64, copy=False)
    oracle = arrays["oracle_ate"].astype(np.float64, copy=False)
    num_graphs, n_units = treatment.shape
    if oracle.shape != (num_graphs, 3):
        raise ValueError("oracle_ate must have shape [graphs,3].")

    contrast_codes = {
        "direct": (1, 0),      # 10 - 00
        "spillover": (3, 1),   # 11 - 10
        "total": (3, 0),       # 11 - 00
    }
    rows: list[dict] = []
    semiparametric_rows: list[dict] = []
    tnet_unit_effects: list[dict] = []
    tnet_training: list[dict] = []
    for graph_index in range(num_graphs):
        adj = adjacency[graph_index]
        degree = adj.sum(axis=1).astype(np.int64)
        counts = np.rint(exposure[graph_index] * degree).astype(np.int64)
        state_codes = _majority_state_codes(treatment[graph_index], counts, degree)
        arm_propensity = majority_arm_propensities(
            degree, treatment_prob=treatment_prob
        )
        observed_propensity = arm_propensity[np.arange(n_units), state_codes]
        node_x = x[graph_index]
        neighbor_x = (adj @ node_x) / degree
        hac = _majority_hac_kernel(adj, bandwidth)

        if 'oracle_arm_means' in arrays:
            all_mu = arrays['oracle_arm_means']
            if all_mu.shape != (num_graphs,n_units,2,2) or not np.isfinite(all_mu).all():
                raise ValueError('oracle_arm_means must be finite [graphs,nodes,t,s]')
            mu = all_mu[graph_index]
            oracle_arm_means = np.stack((mu[:,0,0],mu[:,1,0],mu[:,0,1],mu[:,1,1]),-1)
        else:
            oracle_arm_means = oracle_majority_arm_nuisance(
                **{key: arrays[key][graph_index].astype(np.float64,copy=False)
                   for key in ('structural_baseline','tau','gamma','eta')},
                degree=degree, treatment_prob=treatment_prob)

        base_tnet_config = tnet_config or TNetConfig()
        graph_tnet_config = replace(
            base_tnet_config, seed=int(base_tnet_config.seed) + graph_index
        )
        fitted_tnet = fit_tnet_graph(
            adjacency=adj,
            covariates=node_x[:, None],
            treatment=treatment[graph_index],
            exposure=exposure[graph_index],
            outcomes=outcome[graph_index],
            config=graph_tnet_config,
        )
        tnet_training.append({"dataset_id": graph_index + 1,
                              **fitted_tnet.training_config,
                              "final_losses": fitted_tnet.final_losses})
        tnet_prediction = predict_tnet_majority_effects(
            fitted_tnet.model,
            adjacency=adj,
            covariates=node_x[:, None],
            treatment_prob=treatment_prob,
        )
        tnet_graph_effect = tnet_prediction.node_effects.mean(axis=0)
        for unit_index in range(n_units):
            for effect_index, effect_name in enumerate(EFFECT_NAMES):
                tnet_unit_effects.append(
                    {
                        "dataset_id": graph_index + 1,
                        "unit_id": unit_index + 1,
                        "effect": effect_name,
                        "estimate": float(tnet_prediction.node_effects[unit_index, effect_index]),
                        "dr_guarantee": False,
                    }
                )

        for effect_index, effect_name in enumerate(EFFECT_NAMES):
            code_a, code_b = contrast_codes[effect_name]
            indicator_a = state_codes == code_a
            indicator_b = state_codes == code_b
            propensity_a = arm_propensity[:, code_a]
            propensity_b = arm_propensity[:, code_b]

            ht_haj = estimate_ht_hajek(
                outcomes=outcome[graph_index],
                indicator_a=indicator_a,
                indicator_b=indicator_b,
                propensity_a=propensity_a,
                propensity_b=propensity_b,
            )
            ht = ht_haj["HT"]
            haj = ht_haj["Haj"]
            fisher = estimate_weighted_regression(
                outcomes=outcome[graph_index],
                state_codes=state_codes,
                observed_propensity=observed_propensity,
                covariates=x[graph_index, :, None],
                target_a_code=code_a,
                target_b_code=code_b,
                interacted=False,
            )
            lin = estimate_weighted_regression(
                outcomes=outcome[graph_index],
                state_codes=state_codes,
                observed_propensity=observed_propensity,
                covariates=x[graph_index, :, None],
                target_a_code=code_a,
                target_b_code=code_b,
                interacted=True,
            )
            phi0_g2 = build_majority_phi0_g2_exact(
                degree=degree,
                treatment=treatment[graph_index],
                neighbor_counts=counts,
                x=node_x[:, None],
                neighbor_x=neighbor_x[:, None],
                treatment_prob=treatment_prob,
                target_a_code=code_a,
                target_b_code=code_b,
            )
            reg_net = solve_network_adjustment(
                outcomes=outcome[graph_index],
                features=phi0_g2.normalized,
                indicator_a=indicator_a,
                indicator_b=indicator_b,
                propensity_a=propensity_a,
                propensity_b=propensity_b,
                hac_kernel=hac,
                ridge=0.0,
            )
            tnet_estimate = float(tnet_graph_effect[effect_index])
            anchor = oracle_semiparametric_anchor(
                outcomes=outcome[graph_index],
                state_codes=state_codes,
                arm_propensity=arm_propensity,
                oracle_arm_outcome_means=oracle_arm_means,
                code_a=code_a,
                code_b=code_b,
                kernel=hac,
            )
            rows.append(
                {
                    "dataset_id": graph_index + 1,
                    "effect": effect_name,
                    "truth": float(oracle[graph_index, effect_index]),
                    "HT": float(ht.estimate) if ht.supported else float("nan"),
                    "Haj": float(haj.estimate) if haj.supported else float("nan"),
                    "F": float(fisher.estimate) if fisher.supported else float("nan"),
                    "L": float(lin.estimate) if lin.supported else float("nan"),
                    REG_NET_METHOD_NAME: float(reg_net.estimate) if reg_net.supported else float("nan"),
                    TNET_METHOD_NAME: tnet_estimate,
                    "HT_supported": bool(ht.supported),
                    "Haj_supported": bool(haj.supported),
                    "F_supported": bool(fisher.supported),
                    "L_supported": bool(lin.supported),
                    f"{REG_NET_METHOD_NAME}_supported": bool(reg_net.supported),
                    f"{TNET_METHOD_NAME}_supported": bool(np.isfinite(tnet_estimate)),
                    "average_degree": float(degree.mean()),
                }
            )
            semiparametric_rows.append(
                {
                    "dataset_id": graph_index + 1,
                    "effect": effect_name,
                    "truth": float(oracle[graph_index, effect_index]),
                    "eif_truth": float(anchor.truth),
                    "variance": float(anchor.variance),
                    "standard_error": float(anchor.standard_error),
                    "average_degree": float(degree.mean()),
                }
            )

    effects: Dict[str, Dict[str, dict]] = {}
    for effect_name in EFFECT_NAMES:
        effects[effect_name] = {
            method: _ate_metric(rows, method, effect_name)
            for method in ATE_METHOD_NAMES
        }
    return {
        "evaluation": "graph-level majority-arm average causal effects",
        "methods": ATE_METHOD_NAMES,
        "num_datasets": num_graphs,
        "num_units_per_dataset": n_units,
        "arm_codes": {"00": 0, "10": 1, "01": 2, "11": 3},
        "contrasts": contrast_codes,
        "effects": effects,
        "graph_effects": rows,
        "tnet_unit_effects": tnet_unit_effects,
        "tnet_training": tnet_training,
        "regression_config": {"F_ridge": 0.0, "L_ridge": 0.0,
                              "reg_net_ridge": 0.0,
                              "reg_net_protocol": "paper_raw_ipw_hac",
                              "reg_net_psd_projection": False},
        "semiparametric_anchor": semiparametric_rows,
    }
