"""Behavioral contracts for the requested fixed A-group outcome mechanism."""
from dataclasses import replace
import json
import math

import pytest
import torch

import train_local_network_interference as demo


def config():
    return demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='a_group_fixed_v1')


@pytest.mark.parametrize('balanced', [False, True])
def test_a_formula_noise_and_three_effects(balanced):
    from csv_random_dgp import generate_random_batch
    b = generate_random_batch(config(), task_ids=range(16), seed=2026, balanced=balanced)
    x, t, e = b['x'], b['observed_treatment'], b['observed_exposure']
    tau = 6 + .2 * x.exp()
    expected = 1 - x - 3*b['neighbor_x'] + tau*t - .9*e + b['noise']
    torch.testing.assert_close(b['y_obs'], expected)
    torch.testing.assert_close(b['tau'], tau)
    torch.testing.assert_close(b['gamma'], torch.full_like(x, -.9))
    assert b['eta'].count_nonzero() == 0
    assert set(b['regime'].tolist()) == {3}
    assert b['prior_noise_sd'].unique().tolist() == [4.]
    # Independently enumerate the binomial distribution for the majority contrast.
    gaps = {}
    for d in b['degree'].unique().int().tolist():
        lo = range(d//2 + 1)
        hi = range(d//2 + 1, d + 1)
        mean = lambda ks: sum(k*math.comb(d, k) for k in ks)/sum(math.comb(d, k) for k in ks)/d
        gaps[d] = mean(hi) - mean(lo)
    delta = torch.tensor([[gaps[int(d)] for d in row] for row in b['degree']])
    truth = torch.stack((tau, -.9*delta, tau - .9*delta), -1)
    torch.testing.assert_close(b['oracle_ite'], truth)
    torch.testing.assert_close(b['oracle_ate'], truth.mean(1))
    ta, ea, tb, eb = b['queries'].unbind(-1)
    torch.testing.assert_close(b['query_effect'], tau[..., None]*(ta-tb) - .9*(ea-eb))
    torch.testing.assert_close(b['outcome_a']-b['outcome_b'], b['query_effect'], atol=4e-6, rtol=1e-5)


@pytest.mark.parametrize('balanced', [False, True])
def test_only_outcome_changes_and_chunking_is_stable(balanced):
    from csv_random_dgp import generate_random_batch
    a = generate_random_batch(config(), task_ids=range(16), seed=41, balanced=balanced)
    old = generate_random_batch(replace(config(), dgp_version='random_lpe_v1'),
                                task_ids=range(16), seed=41, balanced=balanced)
    for key in ('adjacency', 'graph_type', 'x', 'observed_treatment', 'observed_exposure',
                'degree', 'queries', 'task_seed'):
        assert torch.equal(a[key], old[key]), key
    assert torch.equal(a['noise'], 4*old['noise'])
    parts = [generate_random_batch(config(), task_ids=range(i, i+4), seed=41, balanced=balanced)
             for i in range(0, 16, 4)]
    for key in a:
        assert torch.equal(a[key], torch.cat([p[key] for p in parts])), key
    if balanced:
        assert a['graph_type'].bincount().tolist() == old['graph_type'].bincount().tolist()


def test_a_csv_roundtrip_and_rejects_old_bank(tmp_path):
    path = tmp_path/'a.csv'
    args = dict(data_csv=path, num_datasets=8, data_seed=42,
                generation_batch_size=3, validation_fraction=.25, split_seed=7)
    demo.prepare_episode_csv_files(config(), **args)
    manifest = json.loads(path.with_suffix('.csv.manifest.json').read_text())
    assert manifest['data_config']['dgp_version'] == 'a_group_fixed_v1'
    assert manifest['identity']['outcome_dgp']['noise_sd'] == 4.
    assert {r['regime'] for r in manifest['task_metadata']} == {3}
    loaded, ids = demo.load_episode_batch_csv(path, device=torch.device('cpu'))
    original = demo.generate_batch(config(), 8, seed=42, device=torch.device('cpu'))
    assert ids == list(range(8))
    for key in demo.CSV_TENSOR_KEYS:
        assert torch.equal(loaded[key], original[key].to(loaded[key].dtype)), key
    before = path.read_bytes()
    with pytest.raises(ValueError, match='identity|config'):
        demo.prepare_episode_csv_files(replace(config(), dgp_version='random_lpe_v1'), **args)
    assert path.read_bytes() == before
    old_path = tmp_path/'old.csv'
    demo.prepare_episode_csv_files(replace(config(), dgp_version='random_lpe_v1'), **{**args, 'data_csv': old_path})
    with pytest.raises(ValueError, match='identity|config'):
        demo.prepare_episode_csv_files(config(), **{**args, 'data_csv': old_path})


def test_normal_test_uses_a_formula_on_all_four_families():
    from evaluate_matched_old_sampler import generate_matched_test_batch
    from csv_random_dgp import BENCHMARK_STREAM, generate_random_batch
    actual = generate_matched_test_batch(config(), range(16), seed=2026)
    expected = generate_random_batch(config(), task_ids=range(16), seed=2026,
                                     stream=BENCHMARK_STREAM, balanced=True)
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
    assert actual['regime'].unique().tolist() == [3]
    assert actual['graph_type'].unique(return_counts=True)[1].tolist() == [4, 4, 4, 4]


def test_reports_accept_a_group_without_inventing_null_tasks(tmp_path):
    from dual_sampler_reporting import write_family_reports
    from evaluation.unified import _write_csv
    result = dict(manifest=dict(graphs=16, balanced=True, seed=2026,
        data_config=dict(graph_family='mixed', dgp_version='a_group_fixed_v1'),
        task_ids=list(range(16)), checkpoint_sha256='test'),
        input_hashes=[dict(task_id=i, regime=3) for i in range(16)])
    (tmp_path/'results.json').write_text(json.dumps(result))
    rows = [dict(dataset_id=i, unit_id=1, method=method, effect='direct',
                 truth=6.2, estimate=6.3, supported=True)
            for i in range(1, 17) for method in ('PFN with interference', 'baseline')]
    for filename in ('ite_unit_results.csv', 'ate_graph_results.csv'):
        _write_csv(rows, tmp_path/filename)
    write_family_reports(tmp_path, sampler='matched_old_sampler')
    regimes = json.loads((tmp_path/'regime_results.json').read_text())
    assert regimes['regime_graph_counts'] == {'3': 16}
    assert not (tmp_path/'regime_tables/regime_0').exists()
    assert (tmp_path/'family_tables/configuration/ite_table.tex').exists()


@pytest.mark.parametrize('entry', ['evaluate_dual_sampler', 'evaluate_random_csv_tables'])
def test_old_test_entry_names_execute_only_normal_evaluator(entry, monkeypatch):
    import importlib
    import evaluate_matched_old_sampler
    calls = []
    monkeypatch.setattr(evaluate_matched_old_sampler, 'main', lambda argv: calls.append(argv))
    argv = ['--checkpoint', 'a.pt', '--output-dir', 'normal-results']
    importlib.import_module(entry).main(argv)
    assert calls == [argv]
