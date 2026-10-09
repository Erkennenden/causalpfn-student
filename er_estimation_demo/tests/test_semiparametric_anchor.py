import numpy as np

from pfn_interference.baselines.semiparametric import (
    oracle_effect_eif,
    oracle_majority_arm_nuisance,
    oracle_semiparametric_anchor,
)


def test_oracle_majority_arm_nuisance_matches_linear_structural_mean():
    baseline = np.array([1.0, 2.0])
    tau = np.array([0.4, 0.6])
    gamma = np.array([0.2, 0.3])
    eta = np.array([0.0, 0.0])
    degree = np.array([2, 2])
    got = oracle_majority_arm_nuisance(
        structural_baseline=baseline,
        tau=tau,
        gamma=gamma,
        eta=eta,
        degree=degree,
        treatment_prob=0.5,
    )
    # degree=2: low K in {0,1} with conditional mean E=1/3; high is K=2 => E=1
    expected = np.array(
        [
            [1.0 + 0.2 / 3.0, 1.0 + 0.4 + 0.2 / 3.0, 1.0 + 0.2, 1.0 + 0.4 + 0.2],
            [2.0 + 0.3 / 3.0, 2.0 + 0.6 + 0.3 / 3.0, 2.0 + 0.3, 2.0 + 0.6 + 0.3],
        ]
    )
    np.testing.assert_allclose(got, expected)


def test_oracle_effect_eif_matches_hand_calculation():
    outcomes = np.array([1.5, 3.0])
    state_codes = np.array([1, 0])
    propensity = np.full((2, 4), 0.25)
    nuisance = np.array(
        [
            [1.0, 2.0, 1.5, 2.5],
            [2.0, 4.0, 2.5, 4.5],
        ]
    )
    result = oracle_effect_eif(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=propensity,
        oracle_arm_outcome_means=nuisance,
        code_a=1,
        code_b=0,
    )
    # psi_10=3, psi_00=1.5. For node 1: 4*(1.5-2)+(2-3) - (1-1.5) = -2.5
    # node 2: (4-3) - [4*(3-2)+(2-1.5)] = -3.5
    np.testing.assert_allclose(result.contribution, np.array([-2.5, -3.5]))
    assert result.truth == 1.5


def test_oracle_semiparametric_anchor_is_nonnegative():
    outcomes = np.array([1.5, 3.0])
    state_codes = np.array([1, 0])
    propensity = np.full((2, 4), 0.25)
    nuisance = np.array(
        [
            [1.0, 2.0, 1.5, 2.5],
            [2.0, 4.0, 2.5, 4.5],
        ]
    )
    kernel = np.eye(2)
    anchor = oracle_semiparametric_anchor(
        outcomes=outcomes,
        state_codes=state_codes,
        arm_propensity=propensity,
        oracle_arm_outcome_means=nuisance,
        code_a=1,
        code_b=0,
        kernel=kernel,
    )
    assert anchor.variance >= 0.0
    assert anchor.standard_error >= 0.0
    assert np.isclose(anchor.standard_error, np.sqrt(anchor.variance))
