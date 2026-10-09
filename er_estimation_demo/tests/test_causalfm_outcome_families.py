"""Regression checks for source-faithful f primitives and no-U network adapters."""
import inspect
import itertools
import json

import numpy as np
import pytest
import torch

from causalfm_dgp_v1 import generate_task, sample_outcome_function

FAMILIES = ("standard", "dense", "composition")


def sample(seed, family):
    assert "family" in inspect.signature(sample_outcome_function).parameters, "f families are missing"
    return sample_outcome_function(seed, family=family)


def source_reference(seed, family, rows):
    """Use original node-by-node forward calls; no vectorized adapter helpers."""
    from causalfm_dgp_v1._upstream_outcome import OutcomeGenerator
    from causalfm_dgp_v1._upstream_base import BaseMLPGenerator
    from causalfm_dgp_v1._upstream_frontdoor import MediatorGenerator, OutcomeGeneratorWithMediator

    def dense(generator, n_inputs):
        layers = max(3, generator.prior_layers())
        width = generator.prior_hidden_size()
        generator.network = generator._construct_network(layers, width, n_inputs, 1)
        generator._sample_network_parameters()
        return generator._forward_propagate

    def sparse(generator, n_inputs):
        layers = max(3, generator.prior_layers())
        width = generator.prior_hidden_size()
        generator.outcome_network = generator.construct_outcome_network(layers, width, n_inputs)
        generator.sample_outcome_network_parameters()
        noise = dict.fromkeys(generator.outcome_network, 0.)
        return lambda row: generator.forward_propagate(row, noise)[generator.output_node]

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        priors = dict(prior_layers=lambda: np.random.randint(3, 6),
                      prior_hidden_size=lambda: np.random.randint(10, 25))
        if family == "standard":
            f = sparse(OutcomeGenerator(**priors, edge_drop_prob=.4), 4)
        elif family == "dense":
            f = dense(BaseMLPGenerator(**priors, use_layer_noise=False), 4)
        else:
            original = MediatorGenerator(prior_layers=lambda: np.random.randint(3, 5),
                                         prior_hidden_size=lambda: np.random.randint(10, 25))
            layers, width = original.prior_layers(), original.prior_hidden_size()
            original.mediator_network = original.construct_mediator_network(layers, width, 4)
            original.sample_mediator_network_parameters()
            noise = dict.fromkeys(original.mediator_network, 0.)
            g = sparse(OutcomeGeneratorWithMediator(**priors, edge_drop_prob=.4), 3)
            def f(row):
                h = original.forward_propagate(row, noise)[original.output_node]
                return g(np.array([row[0], row[1], h]))
        return np.array([f(row) for row in rows])
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_all_adapters_match_original_primitives_without_internal_noise(family, seed):
    # Catches resampling parameters, wrong input placement, activation or composition.
    function = sample(seed, family)
    rows = np.random.default_rng(17).normal(size=(24, 4))
    rows[:, 2] = np.arange(24) % 2
    rows[:, 3] = np.linspace(0, 1, 24)
    expected = source_reference(seed, family, rows)
    np.testing.assert_allclose(function(rows.reshape(2, 3, 4, 4)), expected.reshape(2, 3, 4), atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
def test_family_is_fixed_for_whole_task_with_external_noise_only(family):
    # Catches noise or parameters resampled per evaluation, and incorrect CEPO labels.
    function = sample(27, family)
    graph = torch.ones(4, 4) - torch.eye(4)
    clean = generate_task(graph, seed=7, task_id=2, noise_sd=0, outcome_family=family)
    noisy = generate_task(graph, seed=7, task_id=2, noise_sd=4, outcome_family=family)
    assert clean.metadata["outcome_family"] == family
    assert clean.outcome.to_dict()["input_names"] == ["x", "neighbor_x", "t", "e"]
    for name in ("x", "observed_treatment", "cepo_target", "oracle_ite"):
        torch.testing.assert_close(clean.batch[name], noisy.batch[name], rtol=0, atol=0)
    torch.testing.assert_close(noisy.batch["y_obs"] - clean.batch["y_obs"], noisy.batch["noise"])
    z = np.array([[.2, -.8, 1., .3], [0., 1., 0., 1.]])
    np.testing.assert_array_equal(function(z), function(z))
    json.dumps(function.to_dict(), allow_nan=False)
    for i in range(4):
        x, nx = float(clean.batch["x"][0, i]), float(clean.batch["neighbor_x"][0, i])
        means = []
        for t, arm in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            values = [clean.outcome(np.array([x, nx, t, sum(a)/3]))
                      for a in itertools.product((0, 1), repeat=3) if int(sum(a) > 1) == arm]
            means.append(np.mean(values))
        np.testing.assert_allclose(clean.batch["cepo_target"][0, i], means, atol=1e-6, rtol=1e-6)


def test_default_training_tasks_sample_all_families_and_keep_covariate_streams():
    # A sampler hard-wired to the old f must fail, while X/T keep their old stream.
    assert "outcome_family" in inspect.signature(generate_task).parameters, "task family selector missing"
    graph = torch.ones(3, 3) - torch.eye(3)
    seen = set()
    for task_id in range(32):
        task = generate_task(graph, task_id=task_id)
        seen.add(task.metadata["outcome_family"])
        old = generate_task(graph, task_id=task_id, outcome_family="standard")
        for key in ("x", "neighbor_x", "observed_treatment", "noise"):
            torch.testing.assert_close(task.batch[key], old.batch[key], rtol=0, atol=0)
    assert seen == set(FAMILIES)


def test_sampling_and_evaluation_preserve_global_rng():
    state = np.random.get_state()
    for family in (*FAMILIES, "mixture"):
        f = sample(125, family)
        f(np.ones((8, 4)))
        after = np.random.get_state()
        assert state[0] == after[0] and state[2:] == after[2:]
        np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("family", ["unknown", "additive"])
def test_invalid_or_removed_family_rejected_before_generation(family):
    with pytest.raises(ValueError, match="family"):
        sample(0, family)

