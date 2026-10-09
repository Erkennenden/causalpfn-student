from __future__ import annotations

import numpy as np

from pfn_interference.baselines.dr import (
    aipw_arm_means,
    estimate_oracle_dr_arm_means,
    network_hac_variance,
)


def test_aipw_arm_means_match_hand_computation() -> None:
    outcomes = np.asarray([1.0, 2.0, 3.0, 4.0])
    state_codes = np.asarray([0, 1, 2, 3])
    propensity = np.full((4, 4), 0.25)
    predictions = np.asarray(
        [
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
        ]
    )

    means, pseudo = aipw_arm_means(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=propensity,
        outcome_predictions=predictions,
    )

    expected = np.asarray([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(means, expected)
    np.testing.assert_allclose(pseudo.mean(axis=0), expected)


def test_oracle_dr_uses_same_aipw_equation() -> None:
    outcomes = np.asarray([1.0, 2.0, 3.0, 4.0])
    state_codes = np.asarray([0, 1, 2, 3])
    propensity = np.full((4, 4), 0.25)
    oracle_predictions = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), (4, 1))

    means, pseudo = estimate_oracle_dr_arm_means(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=propensity,
        oracle_arm_outcome_means=oracle_predictions,
    )

    np.testing.assert_allclose(means, np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert pseudo.shape == (4, 4)


def test_network_hac_variance_is_nonnegative_and_zero_for_constant_contribution() -> None:
    kernel = np.eye(4)
    assert network_hac_variance(np.ones(4), kernel) == 0.0
    value = network_hac_variance(np.asarray([0.0, 1.0, 2.0, 3.0]), kernel)
    assert np.isfinite(value)
    assert value >= 0.0


