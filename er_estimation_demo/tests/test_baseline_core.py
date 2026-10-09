from __future__ import annotations

import numpy as np

from pfn_interference.baselines.core import (
    build_degree_contrasts,
    enumerate_treatment_design,
    state_propensities,
    treated_neighbor_counts,
)


def test_rejected_extreme_bernoulli_design_is_exactly_normalized() -> None:
    design = enumerate_treatment_design(
        n_units=3,
        treatment_prob=0.5,
        reject_all_equal=True,
    )

    assert design.assignments.shape == (6, 3)
    assert np.all(design.assignments.sum(axis=1) > 0)
    assert np.all(design.assignments.sum(axis=1) < 3)
    assert np.isclose(design.probabilities.sum(), 1.0)
    assert np.allclose(design.probabilities, np.full(6, 1.0 / 6.0))


def test_treated_neighbor_counts_use_the_supplied_graph() -> None:
    adjacency = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    assignments = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=int,
    )

    counts = treated_neighbor_counts(adjacency, assignments)

    assert np.array_equal(counts, np.array([[1, 1, 1], [2, 0, 0]]))


def test_degree_contrasts_use_exact_reachable_central_support_points() -> None:
    contrasts = build_degree_contrasts(np.array([1, 1, 9, 9]))

    assert contrasts[1].k_low == 0
    assert contrasts[1].k_high == 1
    assert contrasts[1].e_low == 0.0
    assert contrasts[1].e_high == 1.0
    assert contrasts[9].k_low == 4
    assert contrasts[9].k_high == 5
    assert np.isclose(contrasts[9].e_low, 4.0 / 9.0)
    assert np.isclose(contrasts[9].e_high, 5.0 / 9.0)


def test_state_propensity_matches_hand_count_under_rejected_design() -> None:
    adjacency = np.ones((3, 3), dtype=float) - np.eye(3)
    design = enumerate_treatment_design(
        n_units=3,
        treatment_prob=0.5,
        reject_all_equal=True,
    )

    propensity = state_propensities(
        adjacency,
        design,
        own_treatment=1,
        treated_neighbor_count=1,
    )

    assert np.allclose(propensity, np.full(3, 1.0 / 3.0))
