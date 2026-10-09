"""Network DGP checks: analytic truth, upstream fidelity, and model compatibility."""
import importlib
import math

import numpy as np
import pytest
import torch


def api():
    # Keep a missing implementation an explicit failed requirement during RED.
    assert importlib.util.find_spec("causalfm_dgp_v1") is not None, "network DGP is not implemented"
    return importlib.import_module("causalfm_dgp_v1")


def triangle():
    return torch.ones(3, 3) - torch.eye(3)


def nonlinear(z):
    x, neighbor_x, t, e = np.moveaxis(z, -1, 0)
    return x + 2 * neighbor_x + 3 * t + 5 * e**2 + 7 * t * e**2


@pytest.mark.parametrize("p,expected", [
    (.5, [25/6, 10, 85/6]),
    (.25, [3.7, 10.8, 14.5]),
])
def test_nonlinear_truth_uses_full_support_not_mean_exposure(p, expected):
    # Replacing E[f(e)] with f(E[e]), or swapping effect order, breaks these literals.
    result = api().aggregate_effects(nonlinear, x=np.zeros(2), neighbor_x=np.zeros(2),
                                   degree=np.array([2, 2]), treatment_prob=p)
    np.testing.assert_allclose(result["oracle_ite"], [expected, expected], atol=1e-12)
    np.testing.assert_allclose(result["oracle_ate"], expected, atol=1e-12)


def test_mixed_degrees_match_exhaustive_neighbor_assignment_enumeration():
    # An independent enumeration catches wrong degree bins or arm normalization.
    import itertools
    degrees = [1, 2, 3, 4]
    p = .3
    result = api().aggregate_effects(nonlinear, x=np.zeros(4), neighbor_x=np.zeros(4),
                                   degree=np.array(degrees), treatment_prob=p)
    expected = np.zeros((4, 2, 2))
    for i, d in enumerate(degrees):
        mass = np.zeros(2)
        for assignment in itertools.product((0, 1), repeat=d):
            k = sum(assignment)
            s = int(k > d // 2)
            weight = p**k * (1-p)**(d-k)
            mass[s] += weight
            for t in (0, 1):
                expected[i, t, s] += weight * (3*t + (5+7*t)*(k/d)**2)
        expected[i] /= mass
    np.testing.assert_allclose(result["oracle_arm_means"], expected, atol=1e-12)


def test_same_task_is_reproducible_and_noise_does_not_change_truth():
    dgp = api()
    first = dgp.generate_task(triangle(), seed=19, task_id=4, noise_sd=0)
    again = dgp.generate_task(triangle(), seed=19, task_id=4, noise_sd=0)
    noisy = dgp.generate_task(triangle(), seed=19, task_id=4, noise_sd=4)
    for key in first.batch:
        torch.testing.assert_close(first.batch[key], again.batch[key], rtol=0, atol=0)
    for key in ("oracle_ite", "oracle_ate", "cepo_target", "queries", "x", "observed_treatment"):
        torch.testing.assert_close(first.batch[key], noisy.batch[key], rtol=0, atol=0)
    torch.testing.assert_close(noisy.batch["y_obs"]-first.batch["y_obs"], noisy.batch["noise"])
    assert not torch.equal(first.batch["y_obs"], noisy.batch["y_obs"])
    x = noisy.batch["x"][0].numpy()
    nx = noisy.batch["neighbor_x"][0].numpy()
    t = noisy.batch["observed_treatment"][0].numpy()
    e = noisy.batch["observed_exposure"][0].numpy()
    mean = noisy.outcome(np.stack((x, nx, t, e), axis=-1))
    np.testing.assert_allclose(noisy.batch["y_obs"][0], mean + noisy.batch["noise"][0].numpy(), rtol=1e-6, atol=1e-6)
    assert dgp.generate_task(triangle(), seed=19, task_id=5).metadata["outcome_seed"] != first.metadata["outcome_seed"]
    assert dgp.generate_task(triangle(), seed=19, task_id=4, stream="test").metadata["outcome_seed"] != first.metadata["outcome_seed"]


def test_vectorized_function_matches_original_forward_and_keeps_rng_state():
    dgp = api()
    from causalfm_dgp_v1._upstream_outcome import OutcomeGenerator
    state = np.random.get_state()
    function = dgp.sample_outcome_function(2026, family="standard")
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    try:
        np.random.seed(2026)
        layers, width = np.random.randint(3, 6), np.random.randint(10, 25)
        source = OutcomeGenerator(edge_drop_prob=.4)
        source.outcome_network = source.construct_outcome_network(layers, width, 4)
        source.sample_outcome_network_parameters()
        noise = dict.fromkeys(source.outcome_network.nodes(), 0.)
        z = np.array([[.2, -.7, 0., .25], [-.3, .1, 1., .8]])
        expected = [source.forward_propagate(row, noise)[source.output_node] for row in z]
        np.testing.assert_allclose(function(z), expected, atol=1e-12)
    finally:
        np.random.set_state(state)


def test_task_query_labels_and_aggregated_labels_use_same_function():
    task = api().generate_task(triangle(), outcome=nonlinear, seed=7, noise_sd=2)
    batch = task.batch
    torch.testing.assert_close(batch["oracle_ite"][0], torch.tensor([[25/6, 10., 85/6]]).expand(3, -1))
    base = batch['x'][0] + 2 * batch['neighbor_x'][0]
    expected = torch.stack([base + 5/6, base + 5, base + 5, base + 15], -1)
    torch.testing.assert_close(batch['cepo_target'][0], expected)
    expected_queries = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]]).expand(1,3,4,2)
    torch.testing.assert_close(batch['queries'], expected_queries)
    assert batch["tokens"].shape == (1, 3, 5)
    torch.testing.assert_close(batch["tokens"][..., 2], batch["y_obs"])
    torch.testing.assert_close(batch["oracle_ite"][..., 2], batch["oracle_ite"][..., 0]+batch["oracle_ite"][..., 1])


def test_batch_runs_existing_transformer_backward():
    from train_local_network_interference import LocalNetworkQueryTransformer, ModelConfig, compute_training_losses
    batch = api().generate_task(triangle(), seed=17).batch
    model = LocalNetworkQueryTransformer(ModelConfig(d_model=16, num_heads=4, num_layers=1, ffn_dim=32))
    predictions = model(batch["tokens"], batch["queries"], batch["adjacency"])
    loss = compute_training_losses(model, predictions, batch)["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())


@pytest.mark.parametrize("graph", [torch.zeros(3, 3), torch.ones(3, 3),
                                     torch.tensor([[0., 1.], [0., 0.]]),
                                     torch.tensor([[0., .5], [.5, 0.]])])
def test_rejects_graphs_incompatible_with_existing_count_exposure(graph):
    with pytest.raises(ValueError):
        api().generate_task(graph)


@pytest.mark.parametrize("kwargs", [{"noise_sd": -1}, {"noise_sd": math.nan},
                                      {"treatment_prob": 0}, {"treatment_prob": 1},
                                      {"task_id": -1}, {"task_id": 1.5}, {"stream": ""}])
def test_rejects_invalid_task_settings(kwargs):
    with pytest.raises(ValueError):
        api().generate_task(triangle(), **kwargs)


def test_demo_exports_factual_data_truth_and_reusable_batch(tmp_path):
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path
    from fixed_er_graph import adjacency_sha256

    assert importlib.util.find_spec("causalfm_dgp_v1.demo") is not None, "demo is not implemented"
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(dict(n_units=3, upper_hex="e0", adjacency_sha256=adjacency_sha256(triangle()))))
    output = tmp_path / "output"
    subprocess.run([sys.executable, "-m", "causalfm_dgp_v1.demo", "--graph-file", str(graph_file),
                    "--tasks", "2", "--output", str(output)], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    batch = torch.load(output / "batch.pt", weights_only=True)
    assert batch["tokens"].shape == (2, 3, 5)
    with (output / "nodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    np.testing.assert_allclose([float(row["direct_effect"]) for row in rows], batch["oracle_ite"][..., 0].flatten())
    assert len(json.loads((output / "mechanisms.json").read_text())) == 2
