"""Regression and network-dependent estimators from the adjustment paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class AdjustmentEstimate:
    estimate: float
    supported: bool
    standard_error: float = float("nan")
    psd_stabilized: bool = False


@dataclass(frozen=True)
class NetworkAdjustmentResult:
    estimate: float
    beta: np.ndarray
    supported: bool
    objective_before: float
    objective_after: float
    standard_error: float
    psd_stabilized: bool


def _validate_contrast_inputs(
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
        raise ValueError("outcomes must be one-dimensional.")
    if any(
        array.shape != outcomes.shape
        for array in (indicator_a, indicator_b, propensity_a, propensity_b)
    ):
        raise ValueError("contrast arrays must have matching shapes.")
    return outcomes, indicator_a, indicator_b, propensity_a, propensity_b


def _hajek_weights(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if np.any(propensity_a <= 0.0) or np.any(propensity_b <= 0.0):
        return None
    raw_a = indicator_a.astype(np.float64) / propensity_a
    raw_b = indicator_b.astype(np.float64) / propensity_b
    denominator_a = raw_a.mean()
    denominator_b = raw_b.mean()
    if denominator_a <= 0.0 or denominator_b <= 0.0:
        return None
    a = raw_a / denominator_a
    b = raw_b / denominator_b
    return a - b, a, b


def estimate_ht_hajek(
    *,
    outcomes: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
) -> Dict[str, AdjustmentEstimate]:
    outcomes, indicator_a, indicator_b, propensity_a, propensity_b = (
        _validate_contrast_inputs(
            outcomes,
            indicator_a,
            indicator_b,
            propensity_a,
            propensity_b,
        )
    )
    if np.any(propensity_a <= 0.0) or np.any(propensity_b <= 0.0):
        unsupported = AdjustmentEstimate(float("nan"), False)
        return {"HT": unsupported, "Haj": unsupported}
    ht_weights = (
        indicator_a.astype(np.float64) / propensity_a
        - indicator_b.astype(np.float64) / propensity_b
    )
    ht = AdjustmentEstimate(float(np.mean(ht_weights * outcomes)), True)
    hajek = _hajek_weights(indicator_a, indicator_b, propensity_a, propensity_b)
    if hajek is None:
        haj = AdjustmentEstimate(float("nan"), False)
    else:
        weights, _, _ = hajek
        haj = AdjustmentEstimate(float(np.mean(weights * outcomes)), True)
    return {"HT": ht, "Haj": haj}


def estimate_weighted_regression(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    observed_propensity: np.ndarray,
    covariates: np.ndarray,
    target_a_code: int,
    target_b_code: int,
    interacted: bool,
    ridge: float = 0.0,
) -> AdjustmentEstimate:
    """Unpenalized Fisher/Lin WLS with exposure-state intercepts.

    ``ridge=0`` is retained for callers using the former API; a nonzero
    penalty is rejected rather than silently changing the F/L estimator.
    """

    outcomes = np.asarray(outcomes, dtype=np.float64)
    state_codes = np.asarray(state_codes, dtype=np.int64)
    observed_propensity = np.asarray(observed_propensity, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    n = outcomes.size
    if outcomes.shape != (n,) or state_codes.shape != (n,) or (
        observed_propensity.shape != (n,)
    ) or covariates.shape[0] != n:
        raise ValueError("weighted-regression inputs have incompatible shapes.")
    if ridge != 0.0:
        raise ValueError("F/L use unpenalized WLS; ridge must be zero.")
    if np.any(observed_propensity <= 0.0):
        return AdjustmentEstimate(float("nan"), False)
    if not np.any(state_codes == target_a_code) or not np.any(
        state_codes == target_b_code
    ):
        return AdjustmentEstimate(float("nan"), False)

    n_states = max(
        int(state_codes.max(initial=0)), target_a_code, target_b_code
    ) + 1
    one_hot = np.eye(n_states, dtype=np.float64)[state_codes]
    centered = covariates - covariates.mean(axis=0, keepdims=True)
    if interacted:
        slopes = (one_hot[:, :, None] * centered[:, None, :]).reshape(n, -1)
    else:
        slopes = centered
    design = np.concatenate([one_hot, slopes], axis=1)
    sqrt_w = np.sqrt(1.0 / observed_propensity)
    weighted_design = design * sqrt_w[:, None]
    # SVD least squares avoids forming the squared-condition-number Gram
    # matrix. Redundant columns are allowed when the contrast is estimable.
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, outcomes * sqrt_w, rcond=None
    )
    if rank < design.shape[1]:
        contrast = np.zeros(design.shape[1])
        contrast[target_a_code], contrast[target_b_code] = 1.0, -1.0
        representation = np.linalg.lstsq(weighted_design.T, contrast, rcond=None)[0]
        if not np.allclose(weighted_design.T @ representation, contrast,
                           rtol=1e-8, atol=1e-10):
            return AdjustmentEstimate(float("nan"), False)
    estimate = float(coefficients[target_a_code] - coefficients[target_b_code])
    return AdjustmentEstimate(estimate, True)


def _hajek_influence_linear_parts(
    outcomes: np.ndarray,
    features: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
    *,
    normalize_influence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    hajek = _hajek_weights(indicator_a, indicator_b, propensity_a, propensity_b)
    if hajek is None:
        return None
    weights, a, b = hajek
    if not normalize_influence:
        # Lu et al.'s empirical V_Haj uses raw I/pi multiplying residuals.
        # Only the within-arm means and the final point estimate are Hajek
        # normalized. The legacy stabilized protocol used normalized a/b here.
        a = indicator_a.astype(np.float64) / propensity_a
        b = indicator_b.astype(np.float64) / propensity_b
    sum_a = a.sum()
    sum_b = b.sum()
    mu_y_a = float(a @ outcomes / sum_a)
    mu_y_b = float(b @ outcomes / sum_b)
    mu_f_a = (a[:, None] * features).sum(axis=0) / sum_a
    mu_f_b = (b[:, None] * features).sum(axis=0) / sum_b
    v0 = a * (outcomes - mu_y_a) - b * (outcomes - mu_y_b)
    matrix = a[:, None] * (features - mu_f_a) - b[:, None] * (
        features - mu_f_b
    )
    return weights, v0, matrix


def _project_psd_kernel(
    hac_kernel: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Project the finite-sample HAC kernel onto the PSD cone.

    The projection is performed at the kernel level so the quadratic and
    linear terms used to optimize beta are induced by the same objective.
    This avoids the inconsistent finite-sample stabilization that results
    from clipping only M.T @ H @ M while leaving M.T @ H @ v unchanged.
    """

    symmetric = 0.5 * (hac_kernel + hac_kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    stabilized = bool(np.any(eigenvalues < -1e-10))
    clipped = np.maximum(eigenvalues, 0.0)
    psd = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return 0.5 * (psd + psd.T), stabilized


def solve_network_adjustment(
    *,
    outcomes: np.ndarray,
    features: np.ndarray,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
    hac_kernel: np.ndarray,
    ridge: float = 0.0,
    mode: str = "paper",
) -> NetworkAdjustmentResult:
    """Minimize raw empirical V_Haj; legacy stabilization is explicit only.

    An indefinite quadratic or incompatible null-space linear term has no
    finite minimum; report unsupported rather than changing the paper objective.
    """

    outcomes, indicator_a, indicator_b, propensity_a, propensity_b = (
        _validate_contrast_inputs(
            outcomes,
            indicator_a,
            indicator_b,
            propensity_a,
            propensity_b,
        )
    )
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    if features.shape[0] != outcomes.size:
        raise ValueError("features must contain one row per outcome.")
    hac_kernel = np.asarray(hac_kernel, dtype=np.float64)
    if hac_kernel.shape != (outcomes.size, outcomes.size):
        raise ValueError("hac_kernel must be square with one row per unit.")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative.")
    if mode not in ("paper", "stabilized"):
        raise ValueError("mode must be paper or stabilized.")
    if mode == "paper" and ridge != 0.0:
        raise ValueError("The paper reg-net protocol has no ridge penalty.")

    parts = _hajek_influence_linear_parts(
        outcomes,
        features,
        indicator_a,
        indicator_b,
        propensity_a,
        propensity_b,
        normalize_influence=mode == "stabilized",
    )
    if parts is None:
        return NetworkAdjustmentResult(
            float("nan"),
            np.full(features.shape[1], np.nan),
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            False,
        )
    weights, v0, matrix = parts
    n = outcomes.size
    if mode == "stabilized":
        effective_kernel, psd_stabilized = _project_psd_kernel(hac_kernel)
    else:
        effective_kernel = 0.5 * (hac_kernel + hac_kernel.T)
        psd_stabilized = False
    objective_before = float(v0 @ effective_kernel @ v0 / n)
    if features.shape[1] == 0:
        beta = np.empty(0, dtype=np.float64)
        residual_influence = v0
    else:
        quadratic = matrix.T @ effective_kernel @ matrix / n
        linear = matrix.T @ effective_kernel @ v0 / n
        quadratic = 0.5 * (quadratic + quadratic.T)
        if quadratic.size:
            eigenvalues = np.linalg.eigvalsh(quadratic)
            scale = max(1.0, float(np.max(np.abs(eigenvalues))))
            if mode == "paper" and float(np.min(eigenvalues)) < -1e-10 * scale:
                return NetworkAdjustmentResult(float("nan"),
                    np.full(features.shape[1], np.nan), False, objective_before,
                    float("nan"), float("nan"), False)
            if mode == "stabilized":
                quadratic = quadratic + ridge * scale * np.eye(quadratic.shape[0])
        beta = np.linalg.pinv(quadratic, rcond=1e-12) @ linear
        if mode == "paper" and not np.allclose(quadratic @ beta, linear,
                                               rtol=1e-7, atol=1e-10):
            return NetworkAdjustmentResult(float("nan"),
                np.full(features.shape[1], np.nan), False, objective_before,
                float("nan"), float("nan"), False)
        residual_influence = v0 - matrix @ beta
    objective_after = float(
        residual_influence @ effective_kernel @ residual_influence / n
    )
    estimate = float(np.mean(weights * (outcomes - features @ beta)))
    standard_error = (float(np.sqrt(objective_after / n))
                      if objective_after >= 0.0 else float("nan"))
    return NetworkAdjustmentResult(
        estimate=estimate,
        beta=beta,
        supported=True,
        objective_before=objective_before,
        objective_after=objective_after,
        standard_error=standard_error,
        psd_stabilized=psd_stabilized,
    )


def estimate_auxiliary_methods(
    *,
    outcomes: np.ndarray,
    state_codes: np.ndarray,
    observed_propensity: np.ndarray,
    target_a_code: int,
    target_b_code: int,
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    propensity_a: np.ndarray,
    propensity_b: np.ndarray,
    x: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    phi0_g1: np.ndarray,
    phi0_g2: np.ndarray,
    hac_kernel: np.ndarray,
    ridge: float,
) -> Dict[str, AdjustmentEstimate]:
    """Compute every method reported in the paper's main simulation table."""

    base = estimate_ht_hajek(
        outcomes=outcomes,
        indicator_a=indicator_a,
        indicator_b=indicator_b,
        propensity_a=propensity_a,
        propensity_b=propensity_b,
    )
    result: Dict[str, AdjustmentEstimate] = dict(base)
    result["F"] = estimate_weighted_regression(
        outcomes=outcomes,
        state_codes=state_codes,
        observed_propensity=observed_propensity,
        covariates=x,
        target_a_code=target_a_code,
        target_b_code=target_b_code,
        interacted=False,
    )
    result["L"] = estimate_weighted_regression(
        outcomes=outcomes,
        state_codes=state_codes,
        observed_propensity=observed_propensity,
        covariates=x,
        target_a_code=target_a_code,
        target_b_code=target_b_code,
        interacted=True,
    )
    result["F-phi0(G1)"] = estimate_weighted_regression(
        outcomes=outcomes,
        state_codes=state_codes,
        observed_propensity=observed_propensity,
        covariates=phi0_g1,
        target_a_code=target_a_code,
        target_b_code=target_b_code,
        interacted=False,
    )
    result["F-phi0(G2)"] = estimate_weighted_regression(
        outcomes=outcomes,
        state_codes=state_codes,
        observed_propensity=observed_propensity,
        covariates=phi0_g2,
        target_a_code=target_a_code,
        target_b_code=target_b_code,
        interacted=False,
    )

    n_states = int(max(np.max(state_codes), target_a_code, target_b_code)) + 1
    one_hot = np.eye(n_states)[state_codes]
    centered_x = np.asarray(x, dtype=np.float64)
    if centered_x.ndim == 1:
        centered_x = centered_x[:, None]
    centered_x = centered_x - centered_x.mean(axis=0, keepdims=True)
    lin_features = (one_hot[:, :, None] * centered_x[:, None, :]).reshape(
        outcomes.size, -1
    )
    feature_sets: Mapping[str, np.ndarray] = {
        "ND-F": centered_x,
        "ND-phi0(G1)": phi0_g1,
        "ND-G1": g1,
        "ND-L": lin_features,
        "reg-net": phi0_g2,
        "ND-G2": g2,
    }
    for name, features in feature_sets.items():
        nd = solve_network_adjustment(
            outcomes=outcomes,
            features=features,
            indicator_a=indicator_a,
            indicator_b=indicator_b,
            propensity_a=propensity_a,
            propensity_b=propensity_b,
            hac_kernel=hac_kernel,
            ridge=0.0,
        )
        result[name] = AdjustmentEstimate(
            estimate=nd.estimate,
            supported=nd.supported,
            standard_error=nd.standard_error,
            psd_stabilized=nd.psd_stabilized,
        )
    return result
