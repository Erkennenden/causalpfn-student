"""Restore the reference X law and express the SCM directly in observed X."""
from dataclasses import asdict

import pytest
import torch

import pfn_interference.model as demo
import pfn_interference.simulation.csv_tasks as adapter
import pfn_interference.prior as prior


def config():
    return demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')


def test_training_x_has_reference_standard_normal_law_without_task_location_or_scale():
    batch = adapter.generate_random_batch(config(), task_ids=range(256), seed=713,
                                          stream='x_reference_restore')
    # Independent population check catches the old shifted/scaled mixture.
    x = batch['x'].double()
    assert abs(float(x.mean())) < .05
    assert .95 < float(x.std()) < 1.05
    assert abs(float((x.abs() > 1).double().mean()) - .31731050786291415) < .035
    assert not {'z', 'neighbor_z', 'prior_location', 'prior_x_scale',
                'prior_covariate_family'} & batch.keys()


def test_scm_functions_use_observed_x_and_neighbor_x_in_the_expanded_formula():
    cfg = config()
    stream, seed = 'expanded_x_scm', 912
    ids = [0, 1, 2, 3, 7, 19]
    batch = adapter.generate_random_batch(cfg, task_ids=ids, seed=seed, stream=stream)
    families = adapter.prior_from_data(cfg).function_families
    for j, task_id in enumerate(ids):
        functions = []
        for role in ('baseline', 'neighbor', 'direct'):
            g = prior._task_generator(seed, stream, task_id, f'functions/{role}/family')
            family = families[int(torch.randint(0, len(families), (), generator=g))]
            functions.append(prior.sample_function(family,
                prior._task_generator(seed, stream, task_id, f'functions/{role}/parameters')))
        x, t = batch['x'][j], batch['observed_treatment'][j]
        neighbor_x = batch['adjacency'][j] @ x / batch['degree'][j]
        expected = (batch['prior_baseline_intercept'][j]
            + batch['prior_baseline_amplitude'][j] * functions[0](x)
            + batch['prior_neighbor_coefficient'][j] * functions[1](neighbor_x)
            + (batch['prior_tau_level'][j]
               + batch['prior_tau_amplitude'][j] * functions[2](x)) * t
            + batch['prior_gamma'][j] * (batch['adjacency'][j] @ t / batch['degree'][j])
            + batch['noise'][j])
        torch.testing.assert_close(batch['y_obs'][j], expected, atol=3e-6, rtol=3e-6)


def test_old_random_checkpoint_config_is_not_silently_reinterpreted_as_new_x_law():
    old = {**asdict(config()), 'random_prior_version': 'c_prior_v1_gh256'}
    with pytest.raises(ValueError, match='random_prior_version'):
        demo.data_config_from_mapping(old)


def test_matched_test_uses_the_same_direct_standard_normal_x_law():
    from pfn_interference.experiments.evaluate_matched_old_sampler import generate_matched_test_batch
    batch = generate_matched_test_batch(config(), range(16), seed=2026)
    expected = torch.stack([torch.randn(24, generator=prior._task_generator(
        2026, adapter.BENCHMARK_STREAM, task_id, 'data/normal')) for task_id in range(16)])
    torch.testing.assert_close(batch['x'], expected, atol=0, rtol=0)
