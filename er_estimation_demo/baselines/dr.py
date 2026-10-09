"""Shared AIPW algebra and network-HAC helpers for oracle diagnostics.

No fitted ridge baseline is defined here.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    arm_propensity: np.ndarray,
    outcome_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    state_codes = np.asarray(state_codes, dtype=np.int64)
    arm_propensity = np.asarray(arm_propensity, dtype=np.float64)
    outcome_predictions = np.asarray(outcome_predictions, dtype=np.float64)
    if outcomes.ndim != 1:
        raise ValueError("outcomes must be one-dimensional.")
    n = outcomes.size
    if state_codes.shape != (n,):
        raise ValueError("state_codes must have one entry per outcome.")
    if arm_propensity.shape != (n, 4):
        raise ValueError("arm_propensity must have shape [units,4].")
    if outcome_predictions.shape != (n, 4):
        raise ValueError("outcome_predictions must have shape [units,4].")
    if np.any((state_codes < 0) | (state_codes >= 4)):
        raise ValueError("state_codes must lie in 0 through 3.")
    if np.any(~np.isfinite(arm_propensity)) or np.any(arm_propensity <= 0.0):
        raise ValueError("all arm propensities must be finite and positive.")
    if np.any(~np.isfinite(outcomes)) or np.any(~np.isfinite(outcome_predictions)):
        raise ValueError("outcomes and outcome predictions must be finite.")
    return outcomes, state_codes, arm_propensity, outcome_predictions


def aipw_arm_means(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    arm_propensity: np.ndarray,
    outcome_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return four AIPW arm means and node-level pseudo outcomes."""

    outcomes, state_codes, arm_propensity, outcome_predictions = _validate_inputs(
        outcomes, state_codes, arm_propensity, outcome_predictions
    )
    pseudo = outcome_predictions.copy()
    rows = np.arange(outcomes.size)
    observed_prediction = outcome_predictions[rows, state_codes]
    pseudo[rows, state_codes] += (
        outcomes - observed_prediction
    ) / arm_propensity[rows, state_codes]
    return pseudo.mean(axis=0), pseudo


def estimate_oracle_dr_arm_means(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    arm_propensity: np.ndarray,
    oracle_arm_outcome_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use DGP-known outcome nuisances in the same AIPW estimating equation."""

    return aipw_arm_means(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=arm_propensity,
        outcome_predictions=oracle_arm_outcome_means,
    )


def network_hac_variance(
    contribution: np.ndarray,
    kernel: np.ndarray,
) -> float:
    """Estimate Var(mean(contribution)) with a supplied network-HAC kernel."""

    contribution = np.asarray(contribution, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if contribution.ndim != 1:
        raise ValueError("contribution must be one-dimensional.")
    n = contribution.size
    if kernel.shape != (n, n):
        raise ValueError("kernel must be square with one row per contribution.")
    centered = contribution - contribution.mean()
    value = float(centered @ kernel @ centered / (n * n))
    return max(value, 0.0)
