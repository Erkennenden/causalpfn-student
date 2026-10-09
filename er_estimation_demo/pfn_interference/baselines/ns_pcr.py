"""Cross-sectional NSI adaptation for node-level network-interference ITEs.

The original Network Synthetic Interventions estimator is a panel-data method
that learns synthetic donor weights from repeated outcome trajectories.  This
benchmark has one factual measurement per node, so this adaptation instead:

1. localizes donors by factual own treatment and exact neighborhood exposure;
2. represents each node with its own observed covariate, normalized degree,
   and factual/query exposure;
3. fits a localized weighted principal-component regression (PCR) from donor
   features to factual donor outcomes; and
4. integrates exact-exposure predictions over the benchmark's majority-low and
   majority-high conditional design distributions.

Only factual graph data are accepted by the public estimator.  No simulator
counterfactual outcomes or structural DGP parameters are estimator inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from pfn_interference.estimands import conditional_count_weights


NSI_METHOD_NAME = "NSI"
NSI_EFFECT_NAMES = ("direct", "spillover", "total")
NSI_REQUIRED_BATCH_KEYS = (
    "adjacency",
    "x",
    "observed_treatment",
    "observed_exposure",
    "degree",
    "y_obs",
)


@dataclass(frozen=True)
class NSIConfig:
    """Configuration for the cross-sectional NSI adaptation."""

    pcr_rank_max: int = 6
    min_effective_donors: float = 12.0
    max_bandwidth: float = 0.50
    min_support_mass: float = 0.99
    svd_rcond: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.pcr_rank_max < 1:
            raise ValueError("pcr_rank_max must be at least 1.")
        if not np.isfinite(self.min_effective_donors) or self.min_effective_donors <= 0.0:
            raise ValueError("min_effective_donors must be positive and finite.")
        if not np.isfinite(self.max_bandwidth) or self.max_bandwidth < 0.0:
            raise ValueError("max_bandwidth must be non-negative and finite.")
        if not np.isfinite(self.min_support_mass) or not 0.0 <= self.min_support_mass <= 1.0:
            raise ValueError("min_support_mass must lie in [0, 1].")
        if not np.isfinite(self.svd_rcond) or self.svd_rcond <= 0.0:
            raise ValueError("svd_rcond must be positive and finite.")


@dataclass(frozen=True)
class NSIResult:
    """Per-node Direct, Spillover, and Total estimates for one graph."""

    node_effects: np.ndarray
    supported: np.ndarray
    effective_sample_size: np.ndarray
    local_radius: np.ndarray


@dataclass(frozen=True)
class _QueryFit:
    prediction: float
    supported: bool
    effective_sample_size: float
    radius: float


@dataclass(frozen=True)
class _ArmFit:
    mean: float
    supported: bool
    effective_sample_size: float
    radius: float
    supported_mass: float


def _as_numpy(value: object, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_graph_inputs(
    *,
    adjacency: object,
    x: object,
    observed_treatment: object,
    observed_exposure: object,
    degree: object,
    y_obs: object,
    treatment_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjacency_np = _as_numpy(adjacency, name="adjacency")
    x_np = _as_numpy(x, name="x")
    treatment_np = _as_numpy(observed_treatment, name="observed_treatment")
    exposure_np = _as_numpy(observed_exposure, name="observed_exposure")
    degree_np = _as_numpy(degree, name="degree")
    y_np = _as_numpy(y_obs, name="y_obs")

    if adjacency_np.ndim != 2 or adjacency_np.shape[0] != adjacency_np.shape[1]:
        raise ValueError("adjacency must have shape [N, N].")
    n_units = adjacency_np.shape[0]
    for name, array in (
        ("x", x_np),
        ("observed_treatment", treatment_np),
        ("observed_exposure", exposure_np),
        ("degree", degree_np),
        ("y_obs", y_np),
    ):
        if array.shape != (n_units,):
            raise ValueError(f"{name} must have shape [N].")
    if n_units < 2:
        raise ValueError("NSI requires at least two units.")
    if not np.allclose(adjacency_np, adjacency_np.T, rtol=0.0, atol=1.0e-7):
        raise ValueError("adjacency must be symmetric.")
    if np.max(np.abs(np.diag(adjacency_np))) > 1.0e-7:
        raise ValueError("adjacency diagonal must be zero.")
    rounded_adjacency = np.rint(adjacency_np)
    if np.max(np.abs(adjacency_np - rounded_adjacency)) > 1.0e-7 or np.any(
        (rounded_adjacency < 0.0) | (rounded_adjacency > 1.0)
    ):
        raise ValueError("adjacency must be binary.")
    if np.any(degree_np <= 0.0):
        raise ValueError("degree must be strictly positive for every node.")
    if not np.isfinite(treatment_prob) or not 0.0 < float(treatment_prob) < 1.0:
        raise ValueError("treatment_prob must lie strictly between 0 and 1.")
    if np.any((exposure_np < -1.0e-7) | (exposure_np > 1.0 + 1.0e-7)):
        raise ValueError("observed_exposure must lie in [0, 1].")

    rounded_treatment = np.rint(treatment_np)
    if np.max(np.abs(treatment_np - rounded_treatment)) > 1.0e-6 or np.any(
        (rounded_treatment < 0.0) | (rounded_treatment > 1.0)
    ):
        raise ValueError("observed_treatment must be binary.")

    rounded_degree = np.rint(degree_np)
    if np.max(np.abs(degree_np - rounded_degree)) > 1.0e-5:
        raise ValueError("degree must contain integer-valued counts.")

    graph_degree = rounded_adjacency.sum(axis=1)
    if np.any(graph_degree <= 0.0):
        raise ValueError("adjacency must not contain isolated nodes.")
    if not np.array_equal(rounded_degree, graph_degree):
        raise ValueError("degree must equal adjacency row sums.")

    # A and the factual treatment are the source of truth.  This validation is
    # deliberately separate from donor construction: donors are still pooled
    # across degrees and localized by exposure proportions, not exact counts.
    treated_count = rounded_adjacency @ rounded_treatment
    exact_exposure = treated_count / graph_degree
    if not np.allclose(exposure_np, exact_exposure, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            "observed_exposure must equal treated-neighbor count divided by degree."
        )

    return (
        rounded_adjacency,
        x_np,
        rounded_treatment.astype(np.int64),
        exact_exposure,
        graph_degree.astype(np.int64),
        y_np,
    )


def _unit_signatures(
    x: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    """Return the minimal cross-sectional node representation.

    Neighbor covariates are deliberately excluded.  Network treatment
    information enters through exposure localization/query exposure, while the
    representation itself keeps only an intercept, the node's own covariate,
    and normalized degree.
    """

    n_units = x.shape[0]
    degree_norm = degree.astype(np.float64) / float(n_units - 1)
    return np.column_stack(
        [
            np.ones(n_units, dtype=np.float64),
            x,
            degree_norm,
        ]
    )


def _psi_matrix(
    h: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    """Append factual exposure to the minimal node representation."""

    return np.column_stack([h, exposure])


def _psi_query(
    h_row: np.ndarray,
    exposure: float,
) -> np.ndarray:
    """Construct the same four-feature representation for a query exposure."""

    return np.concatenate(
        [h_row, np.asarray([exposure], dtype=np.float64)]
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    square_total = float(np.sum(weights**2))
    if total <= 0.0 or square_total <= 0.0:
        return 0.0
    return total * total / square_total


def _candidate_radii(distances: np.ndarray, max_bandwidth: float) -> np.ndarray:
    within = distances[(distances > 0.0) & (distances <= max_bandwidth)]
    candidates = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            within.astype(np.float64, copy=False),
            np.asarray([max_bandwidth], dtype=np.float64),
        ]
    )
    return np.unique(candidates)


def _weighted_design(
    donor_features: np.ndarray,
    target_feature: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight_sum = float(np.sum(weights))
    feature_mean = np.sum(weights[:, None] * donor_features, axis=0) / weight_sum
    centered = donor_features - feature_mean
    variance = np.sum(weights[:, None] * centered**2, axis=0) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    # Degenerate columns contain no centered information.  Assigning scale one
    # keeps them exactly zero instead of introducing an arbitrary large value.
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = centered / scale
    target_standardized = (target_feature - feature_mean) / scale
    weighted = np.sqrt(weights)[:, None] * standardized
    return weighted, target_standardized


def _svd_components(
    weighted_design: np.ndarray,
    *,
    svd_rcond: float,
    pcr_rank_max: int,
    n_positive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    try:
        u, singular_values, vt = np.linalg.svd(weighted_design, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        return None
    largest = float(singular_values[0])
    if largest <= 0.0:
        return None
    numerical_rank = int(np.sum(singular_values > svd_rcond * largest))
    kappa = min(int(pcr_rank_max), numerical_rank, int(n_positive) - 1)
    if kappa < 1:
        return None
    return u, singular_values, vt, kappa


def _fit_query(
    *,
    target_index: int,
    target_treatment: int,
    target_exposure: float,
    treatment: np.ndarray,
    factual_exposure: np.ndarray,
    factual_features: np.ndarray,
    h: np.ndarray,
    y_obs: np.ndarray,
    config: NSIConfig,
) -> _QueryFit:
    eligible = (treatment == int(target_treatment))
    eligible[target_index] = False
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return _QueryFit(float("nan"), False, float("nan"), float("nan"))

    distances = np.abs(factual_exposure[eligible_indices] - float(target_exposure))
    target_feature = _psi_query(
        h[target_index],
        float(target_exposure),
    )

    selected = None
    for radius in _candidate_radii(distances, config.max_bandwidth):
        if radius == 0.0:
            local_mask = distances == 0.0
        else:
            local_mask = distances <= radius
        if not np.any(local_mask):
            continue
        donor_indices = eligible_indices[local_mask]
        local_distances = distances[local_mask]
        if radius == 0.0:
            weights = np.ones(donor_indices.size, dtype=np.float64)
        else:
            weights = np.exp(-0.5 * (local_distances / radius) ** 2)
        n_positive = int(np.sum(weights > 0.0))
        n_eff = _effective_sample_size(weights)
        if n_eff + 1.0e-12 < config.min_effective_donors:
            continue

        weighted_design, target_standardized = _weighted_design(
            factual_features[donor_indices], target_feature, weights
        )
        decomposition = _svd_components(
            weighted_design,
            svd_rcond=config.svd_rcond,
            pcr_rank_max=config.pcr_rank_max,
            n_positive=n_positive,
        )
        if decomposition is None:
            continue
        u, singular_values, vt, kappa = decomposition
        selected = (
            donor_indices,
            weights,
            target_standardized,
            u,
            vt,
            singular_values,
            kappa,
            float(n_eff),
            float(radius),
        )
        break

    if selected is None:
        return _QueryFit(float("nan"), False, float("nan"), float("nan"))

    (
        donor_indices,
        weights,
        target_standardized,
        u,
        vt,
        singular_values,
        kappa,
        n_eff,
        radius,
    ) = selected
    weight_sum = float(np.sum(weights))
    donor_outcomes = y_obs[donor_indices]
    outcome_mean = float(np.sum(weights * donor_outcomes) / weight_sum)
    centered_outcome = donor_outcomes - outcome_mean
    weighted_outcome = np.sqrt(weights) * centered_outcome

    left_scores = u[:, :kappa].T @ weighted_outcome
    pcr_scores = left_scores / singular_values[:kappa]
    target_scores = target_standardized @ vt[:kappa].T
    prediction = outcome_mean + float(target_scores @ pcr_scores)
    if not np.isfinite(prediction):
        return _QueryFit(float("nan"), False, float("nan"), float("nan"))
    return _QueryFit(prediction, True, n_eff, radius)


def _fit_arm(
    *,
    target_index: int,
    target_treatment: int,
    arm: int,
    treatment_prob: float,
    degree: np.ndarray,
    query_cache: dict[tuple[int, int], _QueryFit],
    query_factory,
    config: NSIConfig,
) -> _ArmFit:
    d = int(degree[target_index])
    conditional_weights = np.asarray(
        conditional_count_weights(d, treatment_prob, arm), dtype=np.float64
    )

    supported_mass = 0.0
    weighted_prediction = 0.0
    supported_neff: list[float] = []
    supported_radius: list[float] = []
    for count, probability in enumerate(conditional_weights):
        probability = float(probability)
        if probability <= 0.0:
            continue
        key = (int(target_treatment), int(count))
        fit = query_cache.get(key)
        if fit is None:
            fit = query_factory(int(target_treatment), count / d)
            query_cache[key] = fit
        if not fit.supported:
            continue
        supported_mass += probability
        weighted_prediction += probability * fit.prediction
        supported_neff.append(float(fit.effective_sample_size))
        supported_radius.append(float(fit.radius))

    if supported_mass + 1.0e-12 < config.min_support_mass or supported_mass <= 0.0:
        return _ArmFit(
            float("nan"), False, float("nan"), float("nan"), float(supported_mass)
        )
    arm_mean = weighted_prediction / supported_mass
    if not np.isfinite(arm_mean):
        return _ArmFit(
            float("nan"), False, float("nan"), float("nan"), float(supported_mass)
        )
    return _ArmFit(
        float(arm_mean),
        True,
        float(min(supported_neff)),
        float(max(supported_radius)),
        float(supported_mass),
    )


def fit_predict_nsi_ite(
    *,
    adjacency: object,
    x: object,
    observed_treatment: object,
    observed_exposure: object,
    degree: object,
    y_obs: object,
    treatment_prob: float,
    config: NSIConfig = NSIConfig(),
) -> NSIResult:
    """Estimate Direct, Spillover, and Total ITEs for one factual graph."""

    (
        adjacency_np,
        x_np,
        treatment_np,
        exposure_np,
        degree_np,
        y_np,
    ) = _validate_graph_inputs(
        adjacency=adjacency,
        x=x,
        observed_treatment=observed_treatment,
        observed_exposure=observed_exposure,
        degree=degree,
        y_obs=y_obs,
        treatment_prob=treatment_prob,
    )

    h = _unit_signatures(x_np, degree_np)
    factual_features = _psi_matrix(h, exposure_np)
    n_units = x_np.shape[0]
    node_effects = np.full((n_units, 3), np.nan, dtype=np.float64)
    supported = np.zeros((n_units, 3), dtype=bool)
    effective_sample_size = np.full((n_units, 3), np.nan, dtype=np.float64)
    local_radius = np.full((n_units, 3), np.nan, dtype=np.float64)

    for target_index in range(n_units):
        query_cache: dict[tuple[int, int], _QueryFit] = {}

        def query_factory(target_treatment: int, target_exposure: float) -> _QueryFit:
            return _fit_query(
                target_index=target_index,
                target_treatment=target_treatment,
                target_exposure=target_exposure,
                treatment=treatment_np,
                factual_exposure=exposure_np,
                factual_features=factual_features,
                h=h,
                y_obs=y_np,
                config=config,
            )

        arm_00 = _fit_arm(
            target_index=target_index,
            target_treatment=0,
            arm=0,
            treatment_prob=treatment_prob,
            degree=degree_np,
            query_cache=query_cache,
            query_factory=query_factory,
            config=config,
        )
        arm_10 = _fit_arm(
            target_index=target_index,
            target_treatment=1,
            arm=0,
            treatment_prob=treatment_prob,
            degree=degree_np,
            query_cache=query_cache,
            query_factory=query_factory,
            config=config,
        )
        arm_11 = _fit_arm(
            target_index=target_index,
            target_treatment=1,
            arm=1,
            treatment_prob=treatment_prob,
            degree=degree_np,
            query_cache=query_cache,
            query_factory=query_factory,
            config=config,
        )

        direct_supported = arm_10.supported and arm_00.supported
        spillover_supported = arm_11.supported and arm_10.supported
        total_supported = arm_11.supported and arm_00.supported

        if direct_supported:
            node_effects[target_index, 0] = arm_10.mean - arm_00.mean
            supported[target_index, 0] = True
            effective_sample_size[target_index, 0] = min(
                arm_10.effective_sample_size, arm_00.effective_sample_size
            )
            local_radius[target_index, 0] = max(arm_10.radius, arm_00.radius)
        if spillover_supported:
            node_effects[target_index, 1] = arm_11.mean - arm_10.mean
            supported[target_index, 1] = True
            effective_sample_size[target_index, 1] = min(
                arm_11.effective_sample_size, arm_10.effective_sample_size
            )
            local_radius[target_index, 1] = max(arm_11.radius, arm_10.radius)
        if total_supported:
            if direct_supported and spillover_supported:
                # When all three arms are available, preserve the benchmark
                # identity exactly up to floating-point addition.
                node_effects[target_index, 2] = (
                    node_effects[target_index, 0] + node_effects[target_index, 1]
                )
            else:
                # Total itself only contrasts mu(1, high) with mu(0, low), so
                # mu(1, low) is not required for support or estimation.
                node_effects[target_index, 2] = arm_11.mean - arm_00.mean
            supported[target_index, 2] = True
            effective_sample_size[target_index, 2] = min(
                arm_00.effective_sample_size,
                arm_11.effective_sample_size,
            )
            local_radius[target_index, 2] = max(arm_00.radius, arm_11.radius)

    return NSIResult(
        node_effects=node_effects,
        supported=supported,
        effective_sample_size=effective_sample_size,
        local_radius=local_radius,
    )


def evaluate_nsi_ite_on_batch(
    *,
    batch: Mapping[str, object],
    treatment_prob: float,
    config: NSIConfig = NSIConfig(),
) -> dict[str, object]:
    """Evaluate NSI on every graph using only the required factual batch keys."""

    missing = set(NSI_REQUIRED_BATCH_KEYS).difference(batch)
    if missing:
        raise ValueError(f"batch is missing NSI factual keys: {sorted(missing)}")

    factual = {
        key: batch[key].detach().cpu() if isinstance(batch[key], torch.Tensor) else np.asarray(batch[key])
        for key in NSI_REQUIRED_BATCH_KEYS
    }
    adjacency = factual["adjacency"]
    if adjacency.ndim != 3:
        raise ValueError("batch adjacency must have shape [B, N, N].")
    batch_size, n_units, _ = adjacency.shape
    rows: list[dict[str, object]] = []

    for dataset_index in range(batch_size):
        result = fit_predict_nsi_ite(
            adjacency=factual["adjacency"][dataset_index],
            x=factual["x"][dataset_index],
            observed_treatment=factual["observed_treatment"][dataset_index],
            observed_exposure=factual["observed_exposure"][dataset_index],
            degree=factual["degree"][dataset_index],
            y_obs=factual["y_obs"][dataset_index],
            treatment_prob=treatment_prob,
            config=config,
        )
        for unit_index in range(n_units):
            for effect_index, effect in enumerate(NSI_EFFECT_NAMES):
                rows.append(
                    {
                        "dataset_id": dataset_index + 1,
                        "unit_id": unit_index + 1,
                        "effect": effect,
                        "method": NSI_METHOD_NAME,
                        "estimate": float(result.node_effects[unit_index, effect_index]),
                        "supported": bool(result.supported[unit_index, effect_index]),
                        "effective_sample_size": float(
                            result.effective_sample_size[unit_index, effect_index]
                        ),
                        "local_radius": float(result.local_radius[unit_index, effect_index]),
                        "degree": int(round(float(_as_numpy(factual["degree"][dataset_index], name="degree")[unit_index]))),
                    }
                )

    return {
        "method": NSI_METHOD_NAME,
        "unit_effects": rows,
        "required_batch_keys": list(NSI_REQUIRED_BATCH_KEYS),
        "config": {
            "pcr_rank_max": int(config.pcr_rank_max),
            "min_effective_donors": float(config.min_effective_donors),
            "max_bandwidth": float(config.max_bandwidth),
            "min_support_mass": float(config.min_support_mass),
            "svd_rcond": float(config.svd_rcond),
        },
        "positioning": (
            "NSI (cross-sectional adaptation): factual treatment/exposure localization "
            "and localized PCR over own covariate, normalized degree, and exposure; "
            "factual graph data only."
        ),
    }
