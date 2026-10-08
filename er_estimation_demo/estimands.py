"""Common majority-arm causal estimands for PFN and paper baselines.

The simulator retains the exact neighborhood exposure E_i = K_i / d_i.  The
benchmark additionally defines a coarse majority arm S_i = 1{K_i > floor(d_i/2)}.
All ITE/ATE targets in the unified benchmark are design averages over the exact
count support inside S_i=0 or S_i=1.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


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


def majority_partition(degree: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the largest low-arm count and smallest high-arm count."""

    if torch.any(degree <= 0):
        raise ValueError("degree must be positive.")
    rounded = torch.round(degree)
    if not torch.allclose(degree, rounded):
        raise ValueError("degree must be integer-valued.")
    low_max = torch.floor(rounded / 2).to(torch.int64)
    high_min = low_max + 1
    return low_max, high_min


@lru_cache(maxsize=None)
def _conditional_count_weights_cached(
    degree: int,
    treatment_prob: float,
    arm: int,
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
        log_weights[count] = (
            log_choose + count * log_p + (degree - count) * log_q
        )
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("majority arm has empty exact-count support.")
    shift = float(np.max(log_weights[finite]))
    weights = np.zeros(degree + 1, dtype=np.float64)
    weights[finite] = np.exp(log_weights[finite] - shift)
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def conditional_count_weights(
    degree: int,
    treatment_prob: float,
    arm: int,
) -> np.ndarray:
    """P(K=k | majority arm) for K ~ Binomial(degree, treatment_prob)."""

    return np.asarray(
        _conditional_count_weights_cached(
            int(degree), float(treatment_prob), int(arm)
        ),
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
        probability += math.comb(degree, count) * treatment_prob**count * (
            1.0 - treatment_prob
        ) ** (degree - count)
    return float(probability)


def arm_mean_exposure(degree: int, treatment_prob: float, arm: int) -> float:
    """Return E[K/degree | S=arm]."""

    degree = _validate_degree(degree)
    weights = conditional_count_weights(degree, treatment_prob, arm)
    counts = np.arange(degree + 1, dtype=np.float64)
    return float(np.dot(weights, counts / degree))


def _arm_mean_exposure_tensor(
    degree: torch.Tensor,
    treatment_prob: float,
    arm: int,
) -> torch.Tensor:
    original_shape = degree.shape
    flat = degree.detach().cpu().reshape(-1)
    values = torch.empty(flat.numel(), dtype=torch.float64)
    for value in torch.unique(flat):
        degree_value = int(round(float(value.item())))
        mean = arm_mean_exposure(degree_value, treatment_prob, arm)
        values[flat == value] = mean
    return values.reshape(original_shape).to(
        device=degree.device,
        dtype=degree.dtype if degree.dtype.is_floating_point else torch.float32,
    )


def oracle_ite_from_parameters(
    *,
    degree: torch.Tensor,
    tau: torch.Tensor,
    gamma: torch.Tensor,
    eta: torch.Tensor,
    treatment_prob: float,
) -> torch.Tensor:
    """Return direct, spillover, total arm-level ITE for every node.

    This keeps the majority-arm estimand unchanged, but deliberately avoids
    ``torch.unique`` during CPU data generation.  Some accelerator-enabled
    PyTorch builds have very slow CPU kernels for that operation.  The arm
    means depend only on the integer degree, so a small NumPy lookup is both
    simpler and equivalent.
    """

    degree, tau, gamma, eta = torch.broadcast_tensors(degree, tau, gamma, eta)
    rounded = torch.round(degree)
    if torch.any(rounded <= 0) or not torch.allclose(degree, rounded):
        raise ValueError("degree must contain positive integer values.")

    degree_np = rounded.detach().cpu().numpy().astype(np.int64, copy=False)
    low_np = np.empty(degree_np.shape, dtype=np.float64)
    high_np = np.empty(degree_np.shape, dtype=np.float64)
    for degree_value in np.unique(degree_np):
        mask = degree_np == degree_value
        d = int(degree_value)
        low_np[mask] = arm_mean_exposure(d, treatment_prob, 0)
        high_np[mask] = arm_mean_exposure(d, treatment_prob, 1)

    out_dtype = degree.dtype if degree.dtype.is_floating_point else torch.float32
    low = torch.from_numpy(low_np).to(device=degree.device, dtype=out_dtype)
    high = torch.from_numpy(high_np).to(device=degree.device, dtype=out_dtype)
    delta = high - low
    direct = tau + eta * low
    spillover = (gamma + eta) * delta
    total = tau + gamma * delta + eta * high
    return torch.stack([direct, spillover, total], dim=-1)


def oracle_ate(ite: torch.Tensor) -> torch.Tensor:
    """Average node-level ITE over the unit dimension (second-to-last axis)."""

    if ite.ndim < 2 or ite.shape[-1] != 3:
        raise ValueError("ite must end in a three-effect dimension.")
    return ite.mean(dim=-2)


def sample_majority_exposures(
    degree: torch.Tensor,
    *,
    treatment_prob: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample exact low/high exposures from conditional Binomial design weights.

    The distribution is the same as before.  Sampling uses one pair of uniform
    draws plus cached conditional-Binomial CDFs instead of ``torch.unique`` and
    ``torch.multinomial``.  This mirrors the older simulator's lightweight CPU
    generation path and avoids accelerator-build CPU-kernel slowdowns.
    """

    if torch.any(degree <= 0):
        raise ValueError("degree must be positive.")
    rounded = torch.round(degree)
    if not torch.allclose(degree, rounded):
        raise ValueError("degree must be integer-valued.")
    treatment_prob = _validate_treatment_prob(treatment_prob)

    degree_np = rounded.detach().cpu().numpy().astype(np.int64, copy=False)
    low_counts = np.empty(degree_np.shape, dtype=np.int64)
    high_counts = np.empty(degree_np.shape, dtype=np.int64)

    # Keep all randomness tied to the supplied torch.Generator.  Only the
    # inverse-CDF mapping is done in NumPy.
    low_u = torch.rand(
        degree.shape, generator=generator, dtype=torch.float64, device="cpu"
    ).numpy()
    high_u = torch.rand(
        degree.shape, generator=generator, dtype=torch.float64, device="cpu"
    ).numpy()

    for degree_value in np.unique(degree_np):
        d = int(degree_value)
        mask = degree_np == d
        low_cdf = np.cumsum(conditional_count_weights(d, treatment_prob, 0))
        high_cdf = np.cumsum(conditional_count_weights(d, treatment_prob, 1))
        low_counts[mask] = np.searchsorted(low_cdf, low_u[mask], side="right")
        high_counts[mask] = np.searchsorted(high_cdf, high_u[mask], side="right")

    degree_float = degree_np.astype(np.float32, copy=False)
    low = torch.from_numpy(low_counts.astype(np.float32) / degree_float)
    high = torch.from_numpy(high_counts.astype(np.float32) / degree_float)
    return low.to(degree.device), high.to(degree.device)
