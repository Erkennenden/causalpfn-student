"""Auxiliary variables and the paper's exact phi0 normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from estimands import arm_probability


@dataclass(frozen=True)
class Phi0Result:
    gamma: np.ndarray
    normalized_by_assignment: np.ndarray


@dataclass(frozen=True)
class ObservedPhi0Result:
    gamma: np.ndarray
    normalized: np.ndarray


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return values


def build_g1(
    *,
    treatment: np.ndarray,
    exposure: np.ndarray,
    x: np.ndarray,
    neighbor_x: np.ndarray,
) -> np.ndarray:
    """Construct G1=(D, treated-neighbor proportion, X, neighbor-mean X)."""

    treatment = _as_feature_matrix(treatment, name="treatment")
    exposure = _as_feature_matrix(exposure, name="exposure")
    x = _as_feature_matrix(x, name="x")
    neighbor_x = _as_feature_matrix(neighbor_x, name="neighbor_x")
    n = treatment.shape[0]
    if any(matrix.shape[0] != n for matrix in (exposure, x, neighbor_x)):
        raise ValueError("all G1 inputs must contain the same number of units.")
    return np.concatenate([treatment, exposure, x, neighbor_x], axis=1)


def build_g2(
    g1: np.ndarray,
    *,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
) -> np.ndarray:
    """Construct G2=(G1 1(T=t), G1 1(T=t'))."""

    g1 = _as_feature_matrix(g1, name="g1")
    indicator_a = np.asarray(indicator_a, dtype=bool)
    indicator_b = np.asarray(indicator_b, dtype=bool)
    if indicator_a.shape != (g1.shape[0],) or indicator_b.shape != (
        g1.shape[0],
    ):
        raise ValueError("G2 indicators must have one entry per unit.")
    return np.concatenate(
        [g1 * indicator_a[:, None], g1 * indicator_b[:, None]], axis=1
    )


def normalize_phi0(
    *,
    feature_by_assignment: np.ndarray,
    ht_weight_by_assignment: np.ndarray,
    probabilities: np.ndarray,
) -> Phi0Result:
    """Apply phi0(G_i)=G_i-gamma_i w_HT,i by exact design expectation."""

    features = np.asarray(feature_by_assignment, dtype=np.float64)
    weights = np.asarray(ht_weight_by_assignment, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("feature_by_assignment must have shape [states, units, q].")
    if weights.shape != features.shape[:2]:
        raise ValueError("ht weights must have shape [states, units].")
    if probabilities.shape != (features.shape[0],):
        raise ValueError("probabilities must have one entry per assignment.")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to one.")

    numerator = np.einsum("s,sn,snq->nq", probabilities, weights, features)
    denominator = np.einsum("s,sn->n", probabilities, np.square(weights))
    gamma = np.divide(
        numerator,
        denominator[:, None],
        out=np.zeros_like(numerator),
        where=denominator[:, None] > 0.0,
    )
    normalized = features - weights[:, :, None] * gamma[None, :, :]
    return Phi0Result(gamma=gamma, normalized_by_assignment=normalized)


def build_g2_full(
    g1: np.ndarray,
    *,
    state_codes: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Construct G1 interacted with every state in a finite exposure mapping."""

    g1 = _as_feature_matrix(g1, name="g1")
    state_codes = np.asarray(state_codes, dtype=np.int64)
    if state_codes.shape != (g1.shape[0],):
        raise ValueError("state_codes must have one entry per unit.")
    if n_states <= 0 or np.any(state_codes < 0) or np.any(state_codes >= n_states):
        raise ValueError("state_codes must lie in [0, n_states).")
    one_hot = np.eye(int(n_states), dtype=np.float64)[state_codes]
    return (one_hot[:, :, None] * g1[:, None, :]).reshape(g1.shape[0], -1)


def build_majority_phi0_g2_exact(
    *,
    degree: np.ndarray,
    treatment: np.ndarray,
    neighbor_counts: np.ndarray,
    x: np.ndarray,
    neighbor_x: np.ndarray,
    treatment_prob: float,
    target_a_code: int,
    target_b_code: int,
) -> ObservedPhi0Result:
    """Return observed phi0(G2) for the four majority-arm states by exact design expectation.

    The expectation in phi0 is evaluated over own treatment D_i and the treated-neighbor
    count K_i.  Under iid Bernoulli assignment this is exact because the benchmark's G1
    depends on treatment only through D_i and K_i/d_i.
    """

    degree = np.asarray(degree, dtype=np.int64)
    treatment = np.asarray(treatment, dtype=np.int64)
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.int64)
    x = _as_feature_matrix(x, name="x")
    neighbor_x = _as_feature_matrix(neighbor_x, name="neighbor_x")
    n = degree.size
    if degree.shape != (n,) or treatment.shape != (n,) or neighbor_counts.shape != (n,):
        raise ValueError("degree, treatment, and neighbor_counts must be one-dimensional.")
    if x.shape[0] != n or neighbor_x.shape[0] != n:
        raise ValueError("x and neighbor_x must contain one row per unit.")
    if x.shape[1] != neighbor_x.shape[1]:
        raise ValueError("x and neighbor_x must have matching feature dimensions.")
    if np.any(degree <= 0) or np.any(neighbor_counts < 0) or np.any(neighbor_counts > degree):
        raise ValueError("degree/count inputs are invalid.")
    if np.any((treatment != 0) & (treatment != 1)):
        raise ValueError("treatment must be binary.")
    p = float(treatment_prob)
    if not 0.0 < p < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")
    if target_a_code == target_b_code or target_a_code not in range(4) or target_b_code not in range(4):
        raise ValueError("target codes must be distinct members of {0,1,2,3}.")

    observed_majority = (neighbor_counts > (degree // 2)).astype(np.int64)
    observed_state = 2 * observed_majority + treatment
    observed_exposure = neighbor_counts.astype(np.float64) / degree
    observed_g1 = build_g1(
        treatment=treatment,
        exposure=observed_exposure,
        x=x,
        neighbor_x=neighbor_x,
    )
    observed_g2 = build_g2_full(observed_g1, state_codes=observed_state, n_states=4)

    base_dim = observed_g1.shape[1]
    gamma = np.zeros((n, 4 * base_dim), dtype=np.float64)
    observed_weight = np.zeros(n, dtype=np.float64)

    for i in range(n):
        d = int(degree[i])
        low_prob = arm_probability(d, p, 0)
        high_prob = arm_probability(d, p, 1)
        arm_probs = np.array(
            [(1.0 - p) * low_prob, p * low_prob, (1.0 - p) * high_prob, p * high_prob],
            dtype=np.float64,
        )
        propensity_a = arm_probs[target_a_code]
        propensity_b = arm_probs[target_b_code]
        numerator = np.zeros(4 * base_dim, dtype=np.float64)
        denominator = 0.0

        for own_treatment in (0, 1):
            own_probability = p if own_treatment == 1 else 1.0 - p
            for count in range(d + 1):
                count_probability = (
                    math.comb(d, count)
                    * p**count
                    * (1.0 - p) ** (d - count)
                )
                probability = own_probability * count_probability
                majority = int(count > d // 2)
                state_code = 2 * majority + own_treatment
                weight = (1.0 if state_code == target_a_code else 0.0) / propensity_a - (
                    1.0 if state_code == target_b_code else 0.0
                ) / propensity_b
                if weight == 0.0:
                    continue
                g1 = np.concatenate(
                    [
                        np.array([float(own_treatment), count / d], dtype=np.float64),
                        x[i],
                        neighbor_x[i],
                    ]
                )
                g2 = np.zeros((4, base_dim), dtype=np.float64)
                g2[state_code] = g1
                g2 = g2.reshape(-1)
                numerator += probability * weight * g2
                denominator += probability * weight * weight

        if denominator > 0.0:
            gamma[i] = numerator / denominator
        state_code = int(observed_state[i])
        observed_weight[i] = (1.0 if state_code == target_a_code else 0.0) / propensity_a - (
            1.0 if state_code == target_b_code else 0.0
        ) / propensity_b

    normalized = observed_g2 - observed_weight[:, None] * gamma
    return ObservedPhi0Result(gamma=gamma, normalized=normalized)
