"""Exact treatment-design and exposure-state utilities.

The demo has only ten units, so all treatment assignments can be enumerated.
This avoids Monte Carlo error in propensity scores and in the paper's phi0
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TreatmentDesign:
    assignments: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.assignments.ndim != 2:
            raise ValueError("assignments must have shape [states, units].")
        if self.probabilities.shape != (self.assignments.shape[0],):
            raise ValueError("probabilities must have one entry per assignment.")
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("treatment-design probabilities must sum to one.")


@dataclass(frozen=True)
class DegreeContrast:
    degree: int
    k_low: int
    k_high: int
    e_low: float
    e_high: float


def enumerate_treatment_design(
    n_units: int,
    treatment_prob: float,
    *,
    reject_all_equal: bool = True,
) -> TreatmentDesign:
    """Enumerate the Bernoulli design used by the simulator exactly."""

    if n_units <= 0:
        raise ValueError("n_units must be positive.")
    if not 0.0 < treatment_prob < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")

    assignments = np.asarray(list(product((0, 1), repeat=n_units)), dtype=np.int64)
    if reject_all_equal:
        treated = assignments.sum(axis=1)
        keep = (treated > 0) & (treated < n_units)
        assignments = assignments[keep]
    treated = assignments.sum(axis=1)
    probabilities = (
        treatment_prob**treated
        * (1.0 - treatment_prob) ** (n_units - treated)
    ).astype(np.float64)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("treatment design has zero total probability.")
    probabilities /= total
    return TreatmentDesign(assignments=assignments, probabilities=probabilities)


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric.")
    if not np.allclose(np.diag(adjacency), 0.0):
        raise ValueError("adjacency must have a zero diagonal.")
    return adjacency


def treated_neighbor_counts(
    adjacency: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Return exact treated-neighbor counts for each assignment and node."""

    adjacency = _validate_adjacency(adjacency)
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim == 1:
        assignments = assignments[None, :]
    if assignments.ndim != 2 or assignments.shape[1] != adjacency.shape[0]:
        raise ValueError("assignments must have shape [states, units].")
    return np.rint(assignments @ adjacency.T).astype(np.int64)


def build_degree_contrasts(degrees: np.ndarray) -> Dict[int, DegreeContrast]:
    """Choose adjacent central support points separately for every degree."""

    degrees = np.asarray(degrees)
    if degrees.ndim != 1:
        raise ValueError("degrees must be one-dimensional.")
    if np.any(degrees <= 0) or not np.allclose(degrees, np.rint(degrees)):
        raise ValueError("degrees must be positive integers.")

    result: Dict[int, DegreeContrast] = {}
    for degree in sorted({int(value) for value in degrees.tolist()}):
        k_low = 0 if degree == 1 else (degree - 1) // 2
        k_high = k_low + 1
        result[degree] = DegreeContrast(
            degree=degree,
            k_low=k_low,
            k_high=k_high,
            e_low=k_low / degree,
            e_high=k_high / degree,
        )
    return result


def state_indicators(
    treatment: np.ndarray,
    neighbor_counts: np.ndarray,
    *,
    own_treatment: int,
    treated_neighbor_count: int,
) -> np.ndarray:
    treatment = np.asarray(treatment)
    neighbor_counts = np.asarray(neighbor_counts)
    if treatment.shape != neighbor_counts.shape:
        raise ValueError("treatment and neighbor_counts must have matching shapes.")
    return (treatment == own_treatment) & (
        neighbor_counts == treated_neighbor_count
    )


def state_propensities(
    adjacency: np.ndarray,
    design: TreatmentDesign,
    *,
    own_treatment: int,
    treated_neighbor_count: int,
) -> np.ndarray:
    """Compute P(D_i=t, sum_j A_ij D_j=k) for every node."""

    adjacency = _validate_adjacency(adjacency)
    if design.assignments.shape[1] != adjacency.shape[0]:
        raise ValueError("design and adjacency use different numbers of units.")
    counts = treated_neighbor_counts(adjacency, design.assignments)
    indicators = state_indicators(
        design.assignments,
        counts,
        own_treatment=own_treatment,
        treated_neighbor_count=treated_neighbor_count,
    )
    return indicators.astype(np.float64).T @ design.probabilities
