from __future__ import annotations

import numpy as np

from pfn_interference.baselines.leung_ani.estimator import estimate_ht, network_hac_se


def test_ht_estimator_matches_hand_computation() -> None:
    result = estimate_ht(
        outcomes=np.array([1.0, 2.0]),
        indicator_a=np.array([True, False]),
        indicator_b=np.array([False, True]),
        propensity_a=np.array([0.5, 0.5]),
        propensity_b=np.array([0.5, 0.5]),
    )

    assert result.supported
    assert np.isclose(result.estimate, -1.0)
    assert np.allclose(result.contributions, np.array([2.0, -4.0]))


def test_ht_estimator_rejects_zero_target_propensity() -> None:
    result = estimate_ht(
        outcomes=np.array([1.0]),
        indicator_a=np.array([True]),
        indicator_b=np.array([False]),
        propensity_a=np.array([0.0]),
        propensity_b=np.array([0.5]),
    )

    assert not result.supported
    assert np.isnan(result.estimate)


def test_network_hac_keeps_disconnected_components_independent() -> None:
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    se = network_hac_se(
        contributions=np.array([0.0, 2.0, 10.0, 12.0]),
        adjacency=adjacency,
        bandwidth=1,
    )

    assert np.isclose(se, np.sqrt(12.5))
