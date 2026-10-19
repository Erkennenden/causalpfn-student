"""共同约定：四个处理—暴露状态、条件分配概率及效应定义。"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


# ========================================================================
# contracts
# ========================================================================


def _validate_degree(degree: int) -> int:
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be a positive integer.")
    return degree


def _validate_treatment_prob(treatment_prob: float) -> float:
    treatment_prob = float(treatment_prob)
    if not 0.0 < treatment_prob < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")
    return treatment_prob


@lru_cache(maxsize=None)
def _conditional_count_weights_cached(
    degree: int, treatment_prob: float, arm: int
) -> tuple[float, ...]:
    degree = _validate_degree(degree)
    treatment_prob = _validate_treatment_prob(treatment_prob)
    if arm not in (0, 1):
        raise ValueError("arm must be 0 (majority-low) or 1 (majority-high).")
    threshold = degree // 2
    counts = range(degree + 1)
    log_p = math.log(treatment_prob)
    log_q = math.log1p(-treatment_prob)
    log_weights = np.full(degree + 1, -np.inf, dtype=np.float64)
    for count in counts:
        in_arm = count <= threshold if arm == 0 else count > threshold
        if not in_arm:
            continue
        log_choose = (
            math.lgamma(degree + 1)
            - math.lgamma(count + 1)
            - math.lgamma(degree - count + 1)
        )
        log_weights[count] = log_choose + count * log_p + (degree - count) * log_q
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("majority arm has empty exact-count support.")
    shift = float(np.max(log_weights[finite]))
    weights = np.zeros(degree + 1, dtype=np.float64)
    weights[finite] = np.exp(log_weights[finite] - shift)
    weights /= weights.sum()
    return tuple((float(value) for value in weights))


def conditional_count_weights(
    degree: int, treatment_prob: float, arm: int
) -> np.ndarray:
    """P(K=k | majority arm) for K ~ Binomial(degree, treatment_prob)."""
    return np.asarray(
        _conditional_count_weights_cached(int(degree), float(treatment_prob), int(arm)),
        dtype=np.float64,
    )


def arm_probability(degree: int, treatment_prob: float, arm: int) -> float:
    """Return P(S=arm) under iid Bernoulli neighbor treatment."""
    degree = _validate_degree(degree)
    treatment_prob = _validate_treatment_prob(treatment_prob)
    threshold = degree // 2
    probability = 0.0
    for count in range(degree + 1):
        in_arm = count <= threshold if arm == 0 else count > threshold
        if not in_arm:
            continue
        probability += (
            math.comb(degree, count)
            * treatment_prob**count
            * (1.0 - treatment_prob) ** (degree - count)
        )
    return float(probability)


def arm_mean_exposure(degree: int, treatment_prob: float, arm: int) -> float:
    """Return E[K/degree | S=arm]."""
    degree = _validate_degree(degree)
    weights = conditional_count_weights(degree, treatment_prob, arm)
    counts = np.arange(degree + 1, dtype=np.float64)
    return float(np.dot(weights, counts / degree))
