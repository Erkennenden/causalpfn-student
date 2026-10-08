"""Leung (2022) HT estimation and GitHub-aligned network-HAC inference.

The computational details in this module follow the public replication function
``make_Zs`` and ``network_SE`` from ``mpleung/ANI/inference_module.py``.  The
only interface adaptation is that this project passes a NumPy adjacency matrix
rather than a NetworkX graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path


_GITHUB_SOURCE = "https://github.com/mpleung/ANI/blob/main/inference_module.py"
_EXCLUDED_FILL_VALUE = -1000.0


@dataclass(frozen=True)
class LeungEstimate:
    estimate: float
    contributions: np.ndarray
    supported: bool
    standard_error: float = float("nan")


@dataclass(frozen=True)
class NetworkHACResult:
    """Diagnostics returned by the GitHub-aligned ``network_SE`` adapter."""

    standard_error: float
    average_path_length: float
    bandwidth: int
    psd_fallback_used: bool
    asymptotic_variance: float


def _as_eligible_mask(size: int, eligible: np.ndarray | None) -> np.ndarray:
    if eligible is None:
        return np.ones(size, dtype=bool)
    mask = np.asarray(eligible, dtype=bool)
    if mask.shape != (size,):
        raise ValueError("eligible must have one entry per unit.")
    return mask


def _validate_ht_inputs(
    outcomes: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    indicator_a = np.asarray(indicator_a, dtype=bool)
    indicator_b = np.asarray(indicator_b, dtype=bool)
    propensity_a = np.asarray(propensity_a, dtype=np.float64)
    propensity_b = np.asarray(propensity_b, dtype=np.float64)
    if outcomes.ndim != 1:
        raise ValueError("estimator inputs must be one-dimensional.")
    if any(
        value.shape != outcomes.shape
        for value in (indicator_a, indicator_b, propensity_a, propensity_b)
    ):
        raise ValueError("all estimator inputs must have matching shapes.")
    return outcomes, indicator_a, indicator_b, propensity_a, propensity_b


def make_zs(
    *,
    outcomes: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Translate the official ``make_Zs`` routine to the project's array API.

    As in the GitHub implementation, excluded entries receive the sentinel value
    ``-1000`` and are ignored by subsequent subsample calculations.
    """

    outcomes, indicator_a, indicator_b, propensity_a, propensity_b = (
        _validate_ht_inputs(
            outcomes,
            indicator_a,
            indicator_b,
            propensity_a,
            propensity_b,
        )
    )
    mask = _as_eligible_mask(outcomes.size, eligible)
    if np.any(indicator_a[mask] & (propensity_a[mask] <= 0.0)) or np.any(
        indicator_b[mask] & (propensity_b[mask] <= 0.0)
    ):
        raise ValueError("an observed target exposure has nonpositive propensity.")

    weight_a = indicator_a[mask].astype(np.float64)
    weight_b = indicator_b[mask].astype(np.float64)
    selected_pa = propensity_a[mask]
    selected_pb = propensity_b[mask]
    positive_a = weight_a == 1.0
    positive_b = weight_b == 1.0
    weight_a[positive_a] /= selected_pa[positive_a]
    weight_b[positive_b] /= selected_pb[positive_b]

    contributions = np.full(outcomes.shape, _EXCLUDED_FILL_VALUE, dtype=np.float64)
    contributions[mask] = outcomes[mask] * (weight_a - weight_b)
    return contributions


def estimate_ht(
    *,
    outcomes: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
    eligible: np.ndarray | None = None,
) -> LeungEstimate:
    """Estimate the average exposure contrast using Leung's HT contributions."""

    outcomes, indicator_a, indicator_b, propensity_a, propensity_b = (
        _validate_ht_inputs(
            outcomes,
            indicator_a,
            indicator_b,
            propensity_a,
            propensity_b,
        )
    )
    mask = _as_eligible_mask(outcomes.size, eligible)
    if not np.any(mask):
        return LeungEstimate(
            float("nan"),
            np.full(outcomes.shape, np.nan, dtype=np.float64),
            False,
        )
    # The paper's overlap condition is target-state specific.  Treat any zero
    # target propensity in the included population as unsupported, even if the
    # state was not realized in this particular assignment.
    if np.any(propensity_a[mask] <= 0.0) or np.any(propensity_b[mask] <= 0.0):
        return LeungEstimate(
            float("nan"),
            np.full(outcomes.shape, np.nan, dtype=np.float64),
            False,
        )
    contributions = make_zs(
        outcomes=outcomes,
        indicator_a=indicator_a,
        indicator_b=indicator_b,
        propensity_a=propensity_a,
        propensity_b=propensity_b,
        eligible=mask,
    )
    return LeungEstimate(float(contributions[mask].mean()), contributions, True)


def _validate_adjacency(adjacency: np.ndarray, size: int | None = None) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square.")
    if size is not None and adjacency.shape != (size, size):
        raise ValueError("adjacency and contributions use different sizes.")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("Leung's network_SE assumes an undirected graph.")
    return adjacency


def _giant_component_average_path_length(
    adjacency: np.ndarray,
    distances: np.ndarray,
) -> float:
    """Match the official code's APL calculation on the largest component."""

    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    if n_components == 0 or adjacency.shape[0] == 0:
        return 0.0
    component_sizes = np.bincount(labels, minlength=n_components)
    giant_label = int(np.argmax(component_sizes))
    giant = np.flatnonzero(labels == giant_label)
    if giant.size <= 1:
        return 0.0
    selected = distances[np.ix_(giant, giant)]
    return float(selected.sum() / giant.size / (giant.size - 1))


def network_hac_kernel(
    adjacency: np.ndarray,
    *,
    bandwidth: int,
) -> np.ndarray:
    """Return the official uniform kernel ``1{distance <= bandwidth}``."""

    adjacency = _validate_adjacency(adjacency)
    if bandwidth < 0:
        raise ValueError("bandwidth must be nonnegative for a fixed kernel.")
    distances = shortest_path(adjacency, directed=False, unweighted=True)
    return (distances <= bandwidth).astype(np.float64)


def network_hac(
    *,
    contributions: np.ndarray,
    adjacency: np.ndarray,
    bandwidth: int = -1,
    eligible: np.ndarray | None = None,
    k_neighborhood: int = 0,
    exponential_neighborhood_growth: bool = True,
) -> NetworkHACResult:
    """GitHub-aligned adapter of ``network_SE`` for one contribution vector.

    ``bandwidth < 0`` activates Leung's average-path-length rule.  A nonpositive
    uniform-kernel variance is replaced by the positive-semidefinite normalized
    neighborhood Gram estimator used in the public replication code.
    """

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("contributions must be one-dimensional.")
    adjacency = _validate_adjacency(adjacency, values.size)
    mask = _as_eligible_mask(values.size, eligible)
    n = int(mask.sum())
    if n == 0:
        return NetworkHACResult(float("nan"), float("nan"), int(bandwidth), False, float("nan"))
    if k_neighborhood < 0:
        raise ValueError("k_neighborhood must be nonnegative.")

    if bandwidth == 0:
        se = float(values[mask].std() / math.sqrt(n))
        return NetworkHACResult(
            standard_error=se,
            average_path_length=0.0,
            bandwidth=0,
            psd_fallback_used=False,
            asymptotic_variance=float(values[mask].var()),
        )

    distances = shortest_path(adjacency, directed=False, unweighted=True)
    average_path_length = _giant_component_average_path_length(adjacency, distances)
    selected_bandwidth = int(bandwidth)
    if selected_bandwidth < 0:
        recommended = (
            round(average_path_length / 2.0)
            if exponential_neighborhood_growth
            else round(average_path_length ** (1.0 / 3.0))
        )
        selected_bandwidth = int(recommended)
    selected_bandwidth = max(2 * int(k_neighborhood), selected_bandwidth)

    weights = distances <= selected_bandwidth
    centered = values[mask] - values[mask].mean()
    selected_weights = weights[np.ix_(mask, mask)].astype(np.float64)
    asymptotic_variance = float(centered @ selected_weights @ centered / n)
    psd_fallback_used = asymptotic_variance <= 0.0

    if psd_fallback_used:
        # This follows the fallback branch in the official GitHub function.
        fallback_bandwidth = max(int(k_neighborhood), selected_bandwidth)
        neighbors = (distances <= fallback_bandwidth).astype(np.float64)
        row_sums = neighbors.sum(axis=1)
        normalized = np.divide(
            neighbors,
            np.sqrt(row_sums)[:, None],
            out=np.zeros_like(neighbors),
            where=row_sums[:, None] > 0.0,
        )
        psd_weights = normalized @ normalized.T
        selected_psd = psd_weights[np.ix_(mask, mask)]
        asymptotic_variance = float(centered @ selected_psd @ centered / n)

    standard_error = float(math.sqrt(max(asymptotic_variance, 0.0) / n))
    return NetworkHACResult(
        standard_error=standard_error,
        average_path_length=average_path_length,
        bandwidth=selected_bandwidth,
        psd_fallback_used=psd_fallback_used,
        asymptotic_variance=asymptotic_variance,
    )


def network_hac_se(
    *,
    contributions: np.ndarray,
    adjacency: np.ndarray,
    bandwidth: int,
    eligible: np.ndarray | None = None,
    k_neighborhood: int = 0,
    exponential_neighborhood_growth: bool = True,
) -> float:
    """Backward-compatible scalar wrapper around :func:`network_hac`."""

    return network_hac(
        contributions=contributions,
        adjacency=adjacency,
        bandwidth=bandwidth,
        eligible=eligible,
        k_neighborhood=k_neighborhood,
        exponential_neighborhood_growth=exponential_neighborhood_growth,
    ).standard_error
