"""C prior contracts: task isolation, SCM truths and fixed-reference functions."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pytest
import torch

from random_lpe_prior import (
    FUNCTION_FAMILIES, PriorConfig, generate_tasks, original_formula_reference,
    sample_function, standard_normal_reference,
)


def small(**kwargs):
    return PriorConfig(n_units=24, **kwargs)


def test_task_draws_are_bitwise_independent_of_order_chunk_and_rank_partition():
    cfg = small()
    ids = [7, 2, 15, 0, 21, 9]
    full = generate_tasks(cfg, ids, seed=531)
    regrouped = [generate_tasks(cfg, ids[rank::2], seed=531) for rank in range(2)]
    for key, expected in full.items():
        actual = torch.empty_like(expected)
        for rank in range(2):
            actual[rank::2] = regrouped[rank][key]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    single = generate_tasks(cfg, [15], seed=531)
    for key in full:
        torch.testing.assert_close(single[key][0], full[key][2], atol=0, rtol=0)


def test_balanced_strata_and_spline_shift_change_only_direct_mechanism():
    cfg = small()
    usual = generate_tasks(cfg, range(16), seed=51, stream="development", balanced=True)
    shifted = generate_tasks(cfg, range(16), seed=51, stream="development", balanced=True,
                             function_shift=True)
    assert usual["regime"].tolist() == [0, 3, 0, 3] * 4
    assert usual["graph_type"].tolist() == [2]*4 + [3]*4 + [4]*4 + [5]*4
    assert (usual["prior_function_family"] < 5).all()
    assert (shifted["prior_function_family"][:, 2] == 5).all()
    for key in ("x", "adjacency", "structural_baseline", "noise", "gamma",
                "observed_treatment", "queries", "prior_tau_amplitude", "prior_tau_level"):
        torch.testing.assert_close(usual[key], shifted[key], atol=0, rtol=0)
    assert not torch.equal(usual["tau"], shifted["tau"])


def test_noise_free_baseline_observed_token_contract_and_exact_causal_truths():
    batch = generate_tasks(small(), range(16), balanced=True)
    assert batch["tokens"].shape == (16, 24, 5)
    assert batch["queries"].shape == (16, 24, 3, 4)
    for i, key in enumerate(("x", "observed_treatment", "y_obs", "observed_exposure")):
        torch.testing.assert_close(batch["tokens"][..., i], batch[key], atol=0, rtol=0)
    expected_y = (batch["structural_baseline"] + batch["tau"]*batch["observed_treatment"]
                  + batch["gamma"]*batch["observed_exposure"] + batch["noise"])
    torch.testing.assert_close(batch["y_obs"], expected_y)
    ta, ea, tb, eb = batch["queries"].unbind(-1)
    expected_query = batch["tau"][..., None]*(ta-tb) + batch["gamma"][..., None]*(ea-eb)
    torch.testing.assert_close(batch["query_effect"], expected_query, atol=0, rtol=0)
    null = batch["regime"] < 2
    assert batch["oracle_ite"][null, :, 1].count_nonzero() == 0
    torch.testing.assert_close(batch["oracle_ite"][null, :, 0], batch["oracle_ite"][null, :, 2], atol=0, rtol=0)
    torch.testing.assert_close(batch["oracle_ite"][..., 0], batch["tau"], atol=0, rtol=0)
    torch.testing.assert_close(batch["oracle_ite"][..., 2], batch["oracle_ite"][..., 0]+batch["oracle_ite"][..., 1])
    torch.testing.assert_close(batch["neighbor_x"], torch.bmm(batch["adjacency"], batch["x"][..., None]).squeeze(-1)/batch["degree"])
    assert not batch["eta"].count_nonzero()


def test_graph_and_gate_overrides_preserve_independent_task_draws():
    cfg = small(graph_family="er", nointerference_prob=0)
    active = generate_tasks(cfg, range(8))
    null = generate_tasks(replace(cfg, nointerference_prob=1), range(8))
    for key in ("x", "tau", "noise", "adjacency", "observed_treatment", "queries"):
        torch.testing.assert_close(active[key], null[key], atol=0, rtol=0)
    assert (active["prior_neighbor_coefficient"] != 0).all()
    assert (null["prior_neighbor_coefficient"] == 0).all()
    unrelated = replace(cfg, nointerference_prob=1)
    er = generate_tasks(unrelated, range(8))
    sbm = generate_tasks(replace(unrelated, graph_family="sbm"), range(8))
    assert not torch.equal(er["adjacency"], sbm["adjacency"])
    for key in ("x", "tau", "noise", "y_obs", "structural_baseline", "observed_treatment"):
        torch.testing.assert_close(er[key], sbm[key], atol=0, rtol=0)
    another_stream = generate_tasks(cfg, range(8), stream="validation")
    assert not torch.equal(active["x"], another_stream["x"])
    assert not torch.equal(active["observed_treatment"], another_stream["observed_treatment"])


def test_concurrent_prefetch_has_no_global_rng_side_effects():
    before = torch.random.get_rng_state().clone()
    cfg = small()
    expected = generate_tasks(cfg, [5, 13, 29, 41])
    with ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(lambda task_id: generate_tasks(cfg, [task_id]), [5, 13, 29, 41]))
    for key in expected:
        torch.testing.assert_close(torch.cat([batch[key] for batch in batches]), expected[key], atol=0, rtol=0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_iid_treatment_keeps_empty_realized_arms():
    # Fixed task identities with all controls / all treated; no overlap redraw.
    batch = generate_tasks(PriorConfig(n_units=12), [58, 5496], seed=1729, stream="prior_audit")
    assert batch["observed_treatment"].sum(1).tolist() == [0, 12]
    assert torch.isfinite(batch["tokens"]).all()


def test_prior_population_has_coupled_balanced_gates_and_centered_unit_variance_noise():
    batch = generate_tasks(PriorConfig(n_units=12), range(1024), seed=1729, stream="prior_audit")
    proportion = torch.bincount(batch["regime"], minlength=4)/1024
    torch.testing.assert_close(proportion, torch.tensor([.5, 0., 0., .5]), atol=0, rtol=0)
    # Every graph family must see both complete regimes.
    for key, values in (("graph_type", range(2, 6)),):
        for value in values:
            subset = batch["regime"][batch[key] == value]
            assert set(subset.tolist()) == {0, 3}
            assert abs(float((subset < 2).float().mean())-.5) < .1
    for column in range(3):
        frequencies = torch.bincount(batch["prior_function_family"][:, column], minlength=5)/1024
        torch.testing.assert_close(frequencies, torch.full((5,), .2), atol=.055, rtol=0)
    epsilon = batch["noise"]/batch["prior_noise_sd"][:, None]
    assert abs(float(epsilon.mean())) < .035
    assert .97 < float(epsilon.std()) < 1.03
    assert abs(float(batch["observed_treatment"].mean())-.5) < .025
    assert (batch["prior_gamma"][batch["regime"] >= 2] < 0).all()
    assert (batch["prior_gamma"] >= -2).all()
    assert (batch["prior_tau_amplitude"] >= 0).all()
    assert (batch["prior_tau_amplitude"] <= 4).all()
    # a_tau bounds the normalized multiplier, not the resulting signed effect.
    assert float(batch["tau"].min()) < -8 and float(batch["tau"].max()) > 8


@pytest.mark.parametrize("family", [*FUNCTION_FAMILIES, "spline"])
def test_functions_share_values_and_have_unit_gaussian_reference_variance(family):
    nodes, weights = standard_normal_reference()
    # Independent, denser quadrature catches accidentally normalizing on actual X.
    dense_x, dense_w = np.polynomial.hermite.hermgauss(320)
    dense_x = torch.from_numpy(dense_x*math.sqrt(2))
    dense_w = torch.from_numpy(dense_w/math.sqrt(math.pi))
    for seed in range(8):
        fn = sample_function(family, torch.Generator().manual_seed(seed))
        values = fn(nodes)
        assert torch.isfinite(values).all()
        assert abs(float(weights @ values)) < 1e-10
        assert abs(float(weights @ values.square())-1) < 1e-10
        fine = fn(dense_x)
        tolerance = .035 if family == "spline" else .004
        assert abs(float(dense_w @ fine)) < tolerance
        assert abs(float(dense_w @ fine.square())-1) < tolerance
        repeated = fn(torch.tensor([-2., 0., 1., 0., -2.], dtype=torch.float64))
        assert repeated[0] == repeated[4] and repeated[1] == repeated[3]
        assert torch.isfinite(fn(torch.linspace(-8, 8, 101, dtype=torch.float64))).all()


def test_original_reference_matches_hand_computed_three_node_path():
    x = torch.tensor([[0., math.log(2), 0.]])
    graph = torch.tensor([[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]])
    t = torch.tensor([[0., 1., 0.]])
    noise = torch.tensor([[1., -2., 3.]])
    batch = original_formula_reference(x=x, adjacency=graph, treatment=t, noise=noise)
    torch.testing.assert_close(batch["tau"], torch.tensor([[6.2, 6.4, 6.2]]))
    torch.testing.assert_close(batch["structural_baseline"], torch.tensor([[1-3*math.log(2), 1-math.log(2), 1-3*math.log(2)]]))
    # Degree1: high-low=1; degree2: E[K/d|K>1]-E[K/d|K<=1]=2/3.
    torch.testing.assert_close(batch["oracle_ite"][..., 1], torch.tensor([[-.9, -.6, -.9]]))
    torch.testing.assert_close(batch["oracle_ite"][..., 2], torch.tensor([[5.3, 5.8, 5.3]]))
    expected_y = torch.tensor([[1-3*math.log(2)-.9+1, 1-math.log(2)+6.4-2, 1-3*math.log(2)-.9+3]])
    torch.testing.assert_close(batch["y_obs"], expected_y)


def test_all_graph_families_are_valid_in_explicit_small_smoke_configuration():
    batch = generate_tasks(PriorConfig(n_units=12), range(16), balanced=True)
    assert (batch["degree"] > 0).all()
    assert batch["adjacency"].diagonal(dim1=-2, dim2=-1).count_nonzero() == 0
    torch.testing.assert_close(batch["adjacency"], batch["adjacency"].transpose(-1, -2), atol=0, rtol=0)
    assert (batch["prior_target_mean_degree"] >= 4).all()
    assert (batch["prior_target_mean_degree"] <= 4.4).all()


@pytest.mark.parametrize("kwargs", [
    {"n_units": 11}, {"nointerference_prob": float("nan")}, {"nointerference_prob": 1.1},
    {"noise_min": 0}, {"noise_max": float("inf")}, {"noise_min": 7},
    {"target_mean_degree_min": 0}, {"target_mean_degree_max": 3},
    {"function_families": ("spline",)}, {"treatment_prob": .3},
    {"config_version": "unknown"},
])
def test_invalid_prior_config_is_rejected_before_generation(kwargs):
    with pytest.raises(ValueError):
        PriorConfig(**kwargs)


@pytest.mark.parametrize("ids", [[], [-1], [0, 0], [1.5]])
def test_task_identity_rejects_empty_negative_duplicate_or_noninteger_ids(ids):
    with pytest.raises(ValueError):
        generate_tasks(small(), ids)
