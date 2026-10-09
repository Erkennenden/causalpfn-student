"""HyperSCI batch evaluation and shared exact exposure integration utility.

HyperSCI uses neighbor treatment configurations, not a scalar exposure input.
Its majority-arm integration is Monte Carlo; integrate_majority_ite below is
an exact scalar-exposure helper retained for other baselines such as TNet.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pfn_interference.estimands import conditional_count_weights


LITERATURE_ITE_METHOD_NAMES = ("HyperSCI",)
EFFECT_NAMES = ("direct", "spillover", "total")
_EPS = 1.0e-12


def _as_vector(values: np.ndarray, *, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n is not None and array.size != n:
        raise ValueError(f"{name} must have length {n}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_graph(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square.")
    if not np.allclose(adj, adj.T):
        raise ValueError("adjacency must be symmetric.")
    if not np.allclose(np.diag(adj), 0.0):
        raise ValueError("adjacency must have zero diagonal.")
    degree = adj.sum(axis=1)
    if np.any(degree <= 0) or not np.allclose(degree, np.rint(degree)):
        raise ValueError("all graph nodes must have positive integer degree.")
    return adj, degree.astype(np.int64)


def integrate_majority_ite(
    *,
    predictor: Callable[[int, np.ndarray], np.ndarray],
    degree: np.ndarray,
    treatment_prob: float,
) -> np.ndarray:
    """Integrate a node-wise outcome surface over exact majority-arm support.

    ``predictor(t, z)`` receives a scalar own-treatment value and an exposure
    vector of length N and must return one conditional outcome prediction per
    node.  The result columns are direct, spillover, and total ITE.
    """

    degree_a = np.asarray(degree)
    if degree_a.ndim != 1 or np.any(degree_a <= 0):
        raise ValueError("degree must be a positive one-dimensional array.")
    if not np.allclose(degree_a, np.rint(degree_a)):
        raise ValueError("degree must be integer-valued.")
    degree_i = np.rint(degree_a).astype(np.int64)
    p = float(treatment_prob)
    if not 0.0 < p < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")
    n = degree_i.size
    max_degree = int(degree_i.max())

    weight_by_arm = []
    for arm in (0, 1):
        weights = np.zeros((n, max_degree + 1), dtype=np.float64)
        for d in np.unique(degree_i):
            mask = degree_i == d
            values = conditional_count_weights(int(d), p, arm)
            weights[mask, : int(d) + 1] = values
        weight_by_arm.append(weights)

    arm_means = np.zeros((2, 2, n), dtype=np.float64)
    for arm in (0, 1):
        weights = weight_by_arm[arm]
        for count in range(max_degree + 1):
            weight = weights[:, count]
            if not np.any(weight > 0.0):
                continue
            valid_count = np.minimum(count, degree_i)
            exposure = valid_count.astype(np.float64) / degree_i
            for treatment in (0, 1):
                prediction = np.asarray(
                    predictor(treatment, exposure), dtype=np.float64
                )
                if prediction.shape != (n,):
                    raise ValueError("predictor must return shape [N].")
                if not np.isfinite(prediction).all():
                    raise ValueError("predictor returned non-finite values.")
                arm_means[treatment, arm] += weight * prediction

    direct = arm_means[1, 0] - arm_means[0, 0]
    spillover = arm_means[1, 1] - arm_means[1, 0]
    total = arm_means[1, 1] - arm_means[0, 0]
    return np.column_stack([direct, spillover, total])


from .hypersci import fit_predict_hypersci_ite


def evaluate_literature_ite_baselines_on_batch(
    *,
    batch: dict | object,
    treatment_prob: float,
    gps_ridge: float = 1.0e-3,
    neural_epochs: int = 500,
    neural_learning_rate: float = 1.0e-3,
    neural_balance_weight: float = 1.0e-2,
    neural_hidden_dim: int = 32,
    seed: int = 0,
    arm_samples: int = 4096,
) -> dict[str, object]:
    """Fit HyperSCI separately on each factual graph."""

    # Accepted for compatibility with the unchanged training CLI; unused.
    del gps_ridge
    if not hasattr(batch, "keys"):
        raise ValueError("batch must be a mapping-like object.")
    required = {
        "adjacency",
        "x",
        "observed_treatment",
        "y_obs",
    }
    missing = required.difference(batch.keys())
    if missing:
        raise ValueError(f"batch is missing factual ITE keys: {sorted(missing)}")

    def as_numpy(name: str) -> np.ndarray:
        value = batch[name]
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    adjacency = as_numpy("adjacency")
    x = as_numpy("x")
    treatment = as_numpy("observed_treatment")
    outcomes = as_numpy("y_obs")
    if treatment.ndim != 2:
        raise ValueError("observed_treatment must have shape [graphs, units].")
    num_graphs, n_units = treatment.shape
    if adjacency.shape != (num_graphs, n_units, n_units):
        raise ValueError("adjacency shape is incompatible with treatment.")

    rows: list[dict[str, object]] = []
    for graph_index in range(num_graphs):
        graph_kwargs = dict(
            adjacency=adjacency[graph_index],
            x=x[graph_index],
            treatment=treatment[graph_index],
            outcomes=outcomes[graph_index],
            treatment_prob=float(treatment_prob),
        )
        estimate, mcse = fit_predict_hypersci_ite(
            **graph_kwargs, epochs=int(neural_epochs),
            learning_rate=float(neural_learning_rate),
            balance_weight=float(neural_balance_weight),
            hidden_dim=int(neural_hidden_dim), arm_samples=int(arm_samples),
            seed=int(seed) + 1000 * graph_index,
        )
        estimates = {"HyperSCI": estimate}
        degree = np.asarray(adjacency[graph_index]).sum(axis=1).astype(np.int64)
        for method in LITERATURE_ITE_METHOD_NAMES:
            estimate = np.asarray(estimates[method], dtype=np.float64)
            if estimate.shape != (n_units, 3):
                raise RuntimeError(f"{method} returned an invalid ITE shape.")
            for unit_index in range(n_units):
                for effect_index, effect in enumerate(EFFECT_NAMES):
                    value = float(estimate[unit_index, effect_index])
                    rows.append(
                        {
                            "dataset_id": graph_index + 1,
                            "unit_id": unit_index + 1,
                            "effect": effect,
                            "method": method,
                            "estimate": value,
                            "supported": bool(np.isfinite(value)),
                            "degree": int(degree[unit_index]),
                            "integration_mcse": float(mcse[unit_index,effect_index]),
                        }
                    )
    return {
        "evaluation": "node-level majority-arm ITE from factual graph fits",
        "methods": LITERATURE_ITE_METHOD_NAMES,
        "num_datasets": num_graphs,
        "num_units_per_dataset": n_units,
        "unit_effects": rows,
        "training": {
            "method": "HyperSCI",
            "adaptation": "pair-edge hypergraph; conditional assignment Monte Carlo",
            "loss": "standardized factual MSE + balance_weight * upstream Wasserstein(phi)",
            "arm_samples": int(arm_samples),
            "weight_decay": .01,
            "dropout": .5,
            "encoder": "one-layer attention; 2 heads; skip=123; n_out=0",
            "device": "cpu",
            "seed_rule": "evaluation seed + 1000 * graph_index",
            "neural_epochs": int(neural_epochs),
            "neural_learning_rate": float(neural_learning_rate),
            "neural_balance_weight": float(neural_balance_weight),
            "neural_hidden_dim": int(neural_hidden_dim),
        },
    }
