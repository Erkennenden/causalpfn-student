from __future__ import annotations

import numpy as np

from pfn_interference.baselines.auxiliary_adjustment.features import (
    build_g1,
    build_g2,
    normalize_phi0,
)
from pfn_interference.baselines.auxiliary_adjustment.estimators import (
    estimate_weighted_regression,
    solve_network_adjustment,
)


def test_g1_and_g2_follow_the_paper_feature_definitions() -> None:
    g1 = build_g1(
        treatment=np.array([0.0, 1.0]),
        exposure=np.array([0.25, 0.75]),
        x=np.array([2.0, 4.0]),
        neighbor_x=np.array([3.0, 5.0]),
    )
    g2 = build_g2(
        g1,
        indicator_a=np.array([True, False]),
        indicator_b=np.array([False, True]),
    )

    assert g1.shape == (2, 4)
    assert np.allclose(g1[0], np.array([0.0, 0.25, 2.0, 3.0]))
    assert g2.shape == (2, 8)
    assert np.allclose(g2[0, :4], g1[0])
    assert np.allclose(g2[0, 4:], 0.0)
    assert np.allclose(g2[1, :4], 0.0)
    assert np.allclose(g2[1, 4:], g1[1])


def test_phi0_exactly_decorrelates_features_from_ht_weights() -> None:
    probabilities = np.full(4, 0.25)
    weights = np.array(
        [
            [2.0, -2.0],
            [2.0, 0.0],
            [0.0, -2.0],
            [0.0, 0.0],
        ]
    )
    features = np.array(
        [
            [[1.0], [4.0]],
            [[2.0], [3.0]],
            [[3.0], [2.0]],
            [[4.0], [1.0]],
        ]
    )

    result = normalize_phi0(
        feature_by_assignment=features,
        ht_weight_by_assignment=weights,
        probabilities=probabilities,
    )
    correlation = np.einsum(
        "s,sn,snq->nq",
        probabilities,
        weights,
        result.normalized_by_assignment,
    )

    assert np.allclose(correlation, 0.0, atol=1e-12)


def test_weighted_regression_recovers_target_intercept_difference() -> None:
    estimate = estimate_weighted_regression(
        outcomes=np.array([1.0, 1.0, 3.0, 3.0]),
        state_codes=np.array([0, 0, 1, 1]),
        observed_propensity=np.full(4, 0.5),
        covariates=np.zeros((4, 1)),
        target_a_code=1,
        target_b_code=0,
        interacted=False,
        ridge=0.0,
    )

    assert estimate.supported
    assert np.isclose(estimate.estimate, 2.0, atol=1e-8)


def test_unpenalized_regression_handles_redundant_covariates():
    # Duplicated columns should not require adding a penalty to the estimand.
    for interacted in (False, True):
        x = np.array([-2., -1., 1., 2., -1., 0., 2., 3.])
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        xc = x - x.mean()
        y = 2. + 4. * t + (3. + (t if interacted else 0.)) * xc
        out = estimate_weighted_regression(outcomes=y, state_codes=t,
            observed_propensity=np.linspace(.2, .7, 8), covariates=np.column_stack([x, 2*x]),
            target_a_code=1, target_b_code=0, interacted=interacted)
        assert out.supported
        assert np.isclose(out.estimate, 4.0, atol=1e-10)


def test_unidentifiable_unpenalized_contrast_is_not_silently_regularized():
    out = estimate_weighted_regression(outcomes=np.array([1., 1., 3., 3.]),
        state_codes=np.array([0, 0, 1, 1]), observed_propensity=np.full(4, .5),
        covariates=np.array([0., 0., 1., 1.]), target_a_code=1, target_b_code=0,
        interacted=False)
    assert not out.supported
    assert np.isnan(out.estimate)


def test_network_adjustment_never_increases_paper_hac_objective() -> None:
    outcomes = np.array([0.0, 1.0, 3.0, 4.0])
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    indicator_a = np.array([False, False, True, True])
    indicator_b = np.array([True, True, False, False])
    propensity = np.full(4, 0.5)
    kernel = np.eye(4)

    result = solve_network_adjustment(
        outcomes=outcomes,
        features=features,
        indicator_a=indicator_a,
        indicator_b=indicator_b,
        propensity_a=propensity,
        propensity_b=propensity,
        hac_kernel=kernel,
        ridge=0.0,
    )

    assert result.supported
    assert result.objective_after <= result.objective_before + 1e-10


def test_reg_net_matches_raw_ipw_hajek_variance_formula():
    # Hand-derived from the paper's raw I/pi influence terms and identity HAC.
    out = solve_network_adjustment(outcomes=np.array([0., 2., 1., 7.]),
        features=np.array([0., 1., 2., 4.]),
        indicator_a=np.array([True, True, False, False]),
        indicator_b=np.array([False, False, True, True]),
        propensity_a=np.array([.2, .4, .3, .4]),
        propensity_b=np.array([.3, .2, .5, .25]),
        hac_kernel=np.eye(4), ridge=0.0)
    assert out.supported
    assert np.isclose(out.objective_before, 338./9.)
    assert np.isclose(out.beta[0], 242./89.)
    assert np.isclose(out.estimate, 1021./267.)


def test_majority_phi0_g2_matches_exact_design_expectation() -> None:
    from pfn_interference.baselines.auxiliary_adjustment import features as feature_module

    assert hasattr(feature_module, "build_majority_phi0_g2_exact")

    degree = np.array([2], dtype=np.int64)
    treatment = np.array([1], dtype=np.int64)
    neighbor_counts = np.array([0], dtype=np.int64)
    x = np.array([[2.0]])
    neighbor_x = np.array([[3.0]])
    result = feature_module.build_majority_phi0_g2_exact(
        degree=degree,
        treatment=treatment,
        neighbor_counts=neighbor_counts,
        x=x,
        neighbor_x=neighbor_x,
        treatment_prob=0.5,
        target_a_code=1,
        target_b_code=0,
    )

    probabilities = []
    weights = []
    g2_rows = []
    for own_treatment in (0, 1):
        for count, count_probability in enumerate((0.25, 0.5, 0.25)):
            probability = 0.5 * count_probability
            majority = int(count > 1)
            state_code = 2 * majority + own_treatment
            g1 = np.array([own_treatment, count / 2.0, 2.0, 3.0])
            one_hot = np.eye(4)[state_code]
            g2 = (one_hot[:, None] * g1[None, :]).reshape(-1)
            weight = (1.0 if state_code == 1 else 0.0) / 0.375 - (
                1.0 if state_code == 0 else 0.0
            ) / 0.375
            probabilities.append(probability)
            weights.append(weight)
            g2_rows.append(g2)

    probabilities = np.asarray(probabilities)
    weights = np.asarray(weights)
    g2_rows = np.asarray(g2_rows)
    expected_gamma = (
        probabilities[:, None] * weights[:, None] * g2_rows
    ).sum(axis=0) / (probabilities * weights**2).sum()

    observed_g1 = np.array([1.0, 0.0, 2.0, 3.0])
    observed_g2 = (np.eye(4)[1][:, None] * observed_g1[None, :]).reshape(-1)
    observed_weight = 1.0 / 0.375
    expected_normalized = observed_g2 - expected_gamma * observed_weight

    np.testing.assert_allclose(result.gamma[0], expected_gamma, atol=1e-12)
    np.testing.assert_allclose(result.normalized[0], expected_normalized, atol=1e-12)


def test_network_adjustment_projects_indefinite_hac_kernel_consistently() -> None:
    outcomes = np.array([
        0.1257302210933933, -0.1321048632913019, 0.6404226504432821,
        0.10490011715303971, -0.535669373161111, 0.36159505490948474,
    ])
    features = np.array([
        [1.3040000451301372, 0.9470809631292422],
        [-0.7037352358069926, -1.2654214710460525],
        [-0.6232744625373522, 0.0413259793472436],
        [-2.3250307746388343, -0.21879166393254573],
        [-1.2459109472530652, -0.7322673547034516],
        [-0.5442589828573099, -0.31630015636915454],
    ])
    indicator_a = np.array([True, True, True, False, False, False])
    indicator_b = ~indicator_a
    propensity = np.full(6, 0.5)
    kernel = np.array([
        [1.0, 0.972991775547243, -0.5690764232413851, 0.3563174305656732, 0.3403403751478455, 1.0765724699795725],
        [0.972991775547243, 1.0, -0.4763374121127607, -0.5256695049755944, 0.16179278751654416, 0.7676494441022097],
        [-0.5690764232413851, -0.4763374121127607, 1.0, 0.6624105273735685, 0.2395573764178222, 0.3563765598494387],
        [0.3563174305656732, -0.5256695049755944, 0.6624105273735685, 1.0, -0.7864941733202739, 0.15280257122844554],
        [0.3403403751478455, 0.16179278751654416, 0.2395573764178222, -0.7864941733202739, 1.0, 0.9779020916649408],
        [1.0765724699795725, 0.7676494441022097, 0.3563765598494387, 0.15280257122844554, 0.9779020916649408, 1.0],
    ])

    result = solve_network_adjustment(
        outcomes=outcomes,
        features=features,
        indicator_a=indicator_a,
        indicator_b=indicator_b,
        propensity_a=propensity,
        propensity_b=propensity,
        hac_kernel=kernel,
        ridge=1e-6,
        mode="stabilized",
    )

    assert result.supported
    assert result.psd_stabilized
    assert np.linalg.norm(result.beta) < 100.0
    assert result.objective_before >= -1e-10
    assert result.objective_after >= -1e-10
    assert result.objective_after <= result.objective_before + 1e-10


def test_paper_reg_net_reports_unbounded_quadratic_as_unsupported():
    out = solve_network_adjustment(outcomes=np.array([0., 2., 1., 7.]),
        features=np.array([0., 1., 2., 4.]),
        indicator_a=np.array([True, True, False, False]),
        indicator_b=np.array([False, False, True, True]),
        propensity_a=np.full(4, .5), propensity_b=np.full(4, .5),
        hac_kernel=-np.eye(4))
    assert not out.supported
    assert np.isnan(out.estimate)
    assert not out.psd_stabilized
