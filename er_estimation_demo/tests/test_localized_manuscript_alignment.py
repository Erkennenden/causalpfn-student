"""Regression checks for the manuscript-aligned binary-treatment adapter."""
import numpy as np
import pytest
import torch

import pfn_interference.model as demo
import pfn_interference.experiments.evaluate_matched_old_sampler as matched
from pfn_interference.baselines.localized_walsh import (
    ITE_METHOD_NAME, PaperLocalizedWalshEstimator, _localized_weights,
    evaluate_localized_dr_lasso_majority_ite,
)


def test_heldout_outcomes_cannot_enter_nuisance_shrinkage_mean():
    n = 10
    adjacency = np.ones((n, n))-np.eye(n)
    treatment = np.arange(n) % 2
    y = np.arange(n, dtype=float)
    def predict_nuisance(outcome):
        fitted = PaperLocalizedWalshEstimator.fit(
            adjacency=adjacency, treatment=treatment, x=np.zeros(n), outcome=outcome,
            bandwidth=2., min_neighbors=20, localization_mode='kernel',
            lasso_alpha=1e-4, nuisance_ridge=.5, cross_fit_folds=2,
        )
        return outcome-fitted.outcome_residual
    before = predict_nuisance(y)
    changed = y.copy(); changed[::2] += 10000
    after = predict_nuisance(changed)
    np.testing.assert_allclose(after[::2], before[::2], atol=1e-10)


def test_explicit_compact_bandwidth_does_not_pool_nonisomorphic_configurations(monkeypatch):
    monkeypatch.setattr('sys.argv', ['pfn_interference/model.py', '--localized-bandwidth', '.24'])
    args = demo.parse_args()
    result = _localized_weights(np.array([0., .25, .25]), bandwidth=args.localized_bandwidth,
                                min_neighbors=args.localized_min_neighbors, mode=args.localized_mode)
    # Only the identical configuration is locally supported. Do not silently
    # give every structurally unrelated donor almost the same weight.
    np.testing.assert_array_equal(result.weights, [1., 0., 0.])
    assert result.effective_sample_size == 1.


def test_matched_cli_exposes_same_configuration_and_preserves_explicit_overrides():
    parser = matched.build_arg_parser()
    args = parser.parse_args(['--checkpoint', 'model.pt', '--output-dir', 'out',
                             '--localized-bandwidth', '.24', '--localized-arm-samples', '3',
                             '--localized-nuisance-shrinkage', '.02'])
    assert args.localized_bandwidth == .24
    assert args.localized_arm_samples == 3
    assert args.localized_nuisance_ridge == .02


def test_majority_adapter_recovers_binary_effect_and_never_fits_oracle_outcomes():
    batch = demo.generate_batch(demo.DataConfig(n_units=12, er_edge_probability=.5),
                               batch_size=1, seed=313, device=torch.device('cpu'))
    batch['x'] = torch.zeros_like(batch['x'])
    batch['y_obs'] = 1.+3.*batch['observed_treatment']
    kwargs = dict(treatment_prob=.5, bandwidth=2., min_neighbors=20,
                  localization_mode='kernel', lasso_alpha=0., nuisance_ridge=.001,
                  cross_fit_folds=5, arm_samples=16, seed=135)
    report = evaluate_localized_dr_lasso_majority_ite(batch=batch, **kwargs)
    changed = dict(batch, oracle_ite=batch['oracle_ite']+9999.)
    again = evaluate_localized_dr_lasso_majority_ite(batch=changed, **kwargs)
    assert len(report['query_units']) == 36
    assert report['methods'] == ('Localized DR-Lasso',)
    for row, repeated in zip(report['query_units'], again['query_units']):
        assert row['supported']
        expected = 0. if row['effect']=='spillover' else 3.
        assert row[ITE_METHOD_NAME] == pytest.approx(expected, abs=1e-10)
        assert repeated[ITE_METHOD_NAME] == row[ITE_METHOD_NAME]


def test_broad_kernel_reports_nearly_uniform_localization_without_changing_predictions():
    batch = demo.generate_batch(demo.DataConfig(n_units=12, er_edge_probability=.5),
                               batch_size=1, seed=313, device=torch.device('cpu'))
    report = evaluate_localized_dr_lasso_majority_ite(
        batch=batch,treatment_prob=.5,bandwidth=2.,min_neighbors=20,
        localization_mode='kernel',lasso_alpha=1e-4,nuisance_ridge=.001,
        cross_fit_folds=5,arm_samples=2,seed=135)
    assert 'localization_geometry' in report, 'A broad kernel must disclose its loss of locality.'
    geometry = report['localization_geometry'][0]
    assert geometry['maximum_graph_distance'] == .25
    assert geometry['minimum_to_maximum_raw_kernel_weight'] == pytest.approx(.984375)
    assert geometry['near_uniform_kernel'] is True
    assert report['localization_warnings']
