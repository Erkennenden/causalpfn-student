"""Oracle influence-function diagnostics for the majority-arm ATE benchmark.

This module is intentionally separate from fitted estimators.  It plugs
simulator-known outcome nuisances and known randomized arm propensities into
an AIPW/EIC representation and estimates the long-run variance with the
benchmark's network-HAC kernel.  The resulting quantity is used as an oracle
semiparametric anchor, not as a ranked fitted method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfn_interference.estimands import arm_mean_exposure
from .dr import network_hac_variance


@dataclass(frozen=True)
class OracleEIFResult:
    contribution: np.ndarray
    truth: float


@dataclass(frozen=True)
class OracleSemiparametricAnchor:
    variance: float
    standard_error: float
    truth: float
    contribution: np.ndarray


def oracle_majority_arm_nuisance(
    *,
    structural_baseline: np.ndarray,
    tau: np.ndarray,
    gamma: np.ndarray,
    eta: np.ndarray,
    degree: np.ndarray,
    treatment_prob: float,
) -> np.ndarray:
    """Return true node-level E[Y(d,E)|S=s] for codes 00,10,01,11.

    The current simulator has structural mean
    m_i + tau_i*d + gamma_i*e + eta_i*d*e, so conditional integration over
    the majority arm depends only on E[e|S=s].
    """

    baseline = np.asarray(structural_baseline, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    degree = np.asarray(degree, dtype=np.int64)
    n = baseline.size
    if any(value.shape != (n,) for value in (tau, gamma, eta, degree)):
        raise ValueError("all nuisance inputs must have one value per node.")
    if np.any(degree <= 0):
        raise ValueError("degree must be positive.")

    low = np.asarray(
        [arm_mean_exposure(int(d), treatment_prob, 0) for d in degree],
        dtype=np.float64,
    )
    high = np.asarray(
        [arm_mean_exposure(int(d), treatment_prob, 1) for d in degree],
        dtype=np.float64,
    )
    result = np.empty((n, 4), dtype=np.float64)
    for code, (own_treatment, majority_state) in enumerate(
        ((0, 0), (1, 0), (0, 1), (1, 1))
    ):
        mean_exposure = low if majority_state == 0 else high
        result[:, code] = (
            baseline
            + tau * own_treatment
            + gamma * mean_exposure
            + eta * own_treatment * mean_exposure
        )
    return result


def oracle_effect_eif(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    arm_propensity: np.ndarray,
    oracle_arm_outcome_means: np.ndarray,
    code_a: int,
    code_b: int,
) -> OracleEIFResult:
    """Plug true nuisances into the arm EIC and return a contrast contribution."""

    outcomes = np.asarray(outcomes, dtype=np.float64)
    state_codes = np.asarray(state_codes, dtype=np.int64)
    propensity = np.asarray(arm_propensity, dtype=np.float64)
    nuisance = np.asarray(oracle_arm_outcome_means, dtype=np.float64)
    n = outcomes.size
    if state_codes.shape != (n,) or propensity.shape != (n, 4) or nuisance.shape != (n, 4):
        raise ValueError("state, propensity, and nuisance shapes must align with outcomes.")
    if not (0 <= int(code_a) < 4 and 0 <= int(code_b) < 4):
        raise ValueError("arm codes must lie in 0 through 3.")
    if np.any(propensity <= 0.0) or np.any(~np.isfinite(propensity)):
        raise ValueError("arm propensities must be finite and positive.")

    psi = nuisance.mean(axis=0)

    def arm_phi(code: int) -> np.ndarray:
        indicator = (state_codes == code).astype(np.float64)
        return (
            indicator / propensity[:, code] * (outcomes - nuisance[:, code])
            + nuisance[:, code]
            - psi[code]
        )

    contribution = arm_phi(int(code_a)) - arm_phi(int(code_b))
    return OracleEIFResult(
        contribution=contribution,
        truth=float(psi[int(code_a)] - psi[int(code_b)]),
    )


def oracle_semiparametric_anchor(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    arm_propensity: np.ndarray,
    oracle_arm_outcome_means: np.ndarray,
    code_a: int,
    code_b: int,
    kernel: np.ndarray,
) -> OracleSemiparametricAnchor:
    """Return oracle EIF/HAC variance and standard-error anchor for a contrast."""

    eif = oracle_effect_eif(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=arm_propensity,
        oracle_arm_outcome_means=oracle_arm_outcome_means,
        code_a=code_a,
        code_b=code_b,
    )
    variance = network_hac_variance(eif.contribution, kernel)
    return OracleSemiparametricAnchor(
        variance=float(variance),
        standard_error=float(np.sqrt(variance)),
        truth=eif.truth,
        contribution=eif.contribution,
    )
