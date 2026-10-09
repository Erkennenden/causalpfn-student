"""Fixed-ER evaluation must count repeated data, never invent more networks."""
import json
import math

import pytest
import torch


def _fixed_config():
    from train_local_network_interference import DataConfig
    return DataConfig(n_units=24, graph_family='er', graph_protocol='fixed_er_v1',
                      fixed_graph_seed=731, er_edge_probability=.25,
                      dgp_version='a_group_fixed_v1')


def test_test_stream_reuses_training_network_but_refreshes_data():
    from csv_random_dgp import TRAIN_STREAM, generate_random_batch
    from evaluate_matched_old_sampler import generate_matched_test_batch

    config = _fixed_config()
    train = generate_random_batch(config, task_ids=[1, 20, 53], seed=17, stream=TRAIN_STREAM)
    test = generate_matched_test_batch(config, [0, 1, 2], seed=2026)
    assert torch.equal(test['adjacency'], train['adjacency'])
    assert torch.equal(test['degree'], train['degree'])
    assert not torch.equal(test['x'], train['x'])
    assert not torch.equal(test['observed_treatment'], train['observed_treatment'])
    assert not torch.equal(test['noise'], train['noise'])
    assert test['graph_type'].unique().tolist() == [2]  # ER_GRAPH in the existing format.
    assert test['regime'].unique().tolist() == [3]


@pytest.mark.parametrize('change', ['edge', 'fractional'])
def test_evaluator_rejects_changed_adjacency_before_scoring(change):
    from evaluate_matched_old_sampler import validate_fixed_graph_batch
    from csv_random_dgp import generate_random_batch

    config = _fixed_config()
    batch = generate_random_batch(config, task_ids=[0, 1], seed=2026)
    if change == 'edge':
        batch['adjacency'][1, 0, 1] = 1. - batch['adjacency'][1, 0, 1]
    else:
        batch['adjacency'][1, 0, 0] = .5
    with pytest.raises(ValueError, match='adjacency'):
        validate_fixed_graph_batch(config, batch)


def test_cli_accepts_episode_alias_before_loading_checkpoint(tmp_path):
    from evaluate_matched_old_sampler import main

    with pytest.raises(FileNotFoundError):
        main(['--device', 'cpu', '--episodes', '1', '--checkpoint', str(tmp_path/'missing.pt'),
              '--output-dir', str(tmp_path/'out')])


def _write_result(output_dir, *, episodes=3, graph_hashes=None):
    from evaluation.unified import _write_csv

    graph_hashes = graph_hashes or ['fixed-network'] * episodes
    result = dict(
        manifest=dict(
            graphs=episodes, episodes=episodes, balanced=False, seed=2026,
            unique_graph_count=1, checkpoint_sha256='checkpoint',
            fixed_graph=dict(protocol='fixed_er_v1', adjacency_sha256='fixed-network',
                             unique_graph_count=1),
            data_config=dict(graph_family='er', graph_protocol='fixed_er_v1',
                             dgp_version='a_group_fixed_v1'),
            task_ids=list(range(episodes))),
        input_hashes=[dict(task_id=i, regime=3, adjacency_sha256=graph_hashes[i])
                      for i in range(episodes)])
    (output_dir/'results.json').write_text(json.dumps(result))
    rows = [dict(dataset_id=i, unit_id=1, method=method, effect='direct',
                 truth=0., estimate=float(i) if method == 'PFN with interference' else 99.,
                 supported=(method == 'PFN with interference' or i != 2))
            for i in range(1, episodes + 1) for method in ('PFN with interference', 'baseline')]
    for filename in ('ite_unit_results.csv', 'ate_graph_results.csv'):
        _write_csv(rows, output_dir/filename)


def test_fixed_er_reports_count_repeats_without_fabricated_families(tmp_path):
    from dual_sampler_reporting import write_family_reports

    _write_result(tmp_path)
    report = write_family_reports(tmp_path, sampler='matched_old_sampler')
    assert report['unique_graph_count'] == 1
    assert report['episode_count'] == 3
    assert report['graph_family_counts'] == {'er': 3}
    assert set(p.name for p in (tmp_path/'family_tables').iterdir()) == {'all', 'er'}
    rows = report['summaries']['ite']
    assert {r['graph_family'] for r in rows} == {'all', 'er'}
    er_pfn = next(r for r in rows if r['graph_family'] == 'er' and r['method'] == 'PFN with interference')
    assert er_pfn['n_total'] == 3
    assert er_pfn['n_episodes'] == 3
    assert er_pfn['unique_graph_count'] == 1
    assert er_pfn['rmse'] == pytest.approx(math.sqrt(14/3))
    er_baseline = next(r for r in rows if r['graph_family'] == 'er' and r['method'] == 'baseline')
    assert er_baseline['n_supported'] == 2
    paired = next(r for r in report['paired_comparisons']['ite'] if r['graph_family'] == 'er')
    assert paired['n_paired'] == 2
    assert paired['pfn_rmse'] == pytest.approx(math.sqrt(5))
    regimes = json.loads((tmp_path/'regime_results.json').read_text())
    assert regimes['regime_graph_counts'] == {'3': 3}
    assert {r['graph_family'] for r in regimes['summaries']['ite']} == {'all', 'er'}
    assert set(p.name for p in (tmp_path/'regime_tables/regime_3').iterdir()) == {'all', 'er'}
    assert not (tmp_path/'regime_tables/regime_0').exists()


def test_fixed_er_reporting_rejects_a_changed_network(tmp_path):
    from dual_sampler_reporting import write_family_reports

    _write_result(tmp_path, graph_hashes=['fixed-network', 'other-network', 'fixed-network'])
    with pytest.raises(ValueError, match='adjacency'):
        write_family_reports(tmp_path, sampler='matched_old_sampler')


@pytest.mark.parametrize('episodes', [1, 3, 64])
def test_fixed_er_allows_any_positive_episode_count(episodes):
    from evaluate_dual_sampler import validate_graph_count

    validate_graph_count(episodes, graph_protocol='fixed_er_v1')


def test_fixed_er_rejects_empty_evaluation():
    from evaluate_dual_sampler import validate_graph_count

    with pytest.raises(ValueError, match='positive'):
        validate_graph_count(0, graph_protocol='fixed_er_v1')


@pytest.mark.parametrize("support_mass", [None, .99])
@pytest.mark.parametrize("bandwidth", [None, .24])
def test_main_evaluator_keeps_other_baselines_and_forwards_localized_settings(tmp_path, monkeypatch, support_mass, bandwidth):
    from dataclasses import asdict
    import pytest
    import torch
    import train_local_network_interference as demo
    import baselines.causalpfn_direct as causal
    import evaluation.unified as unified
    import evaluate_matched_old_sampler as entry
    cfg = demo.DataConfig(n_units=24, er_edge_probability=.25, graph_protocol='fixed_er_v1',
                          dgp_version='a_group_fixed_v1')
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'placeholder for mocked loader')
    model = torch.nn.Linear(1, 1)
    metadata = dict(data_config=asdict(cfg), model_config={}, epoch=1, monitor_name='point', prediction_protocol=demo.PREDICTION_PROTOCOL)
    monkeypatch.setattr(demo, 'load_checkpoint', lambda **kwargs: (model, None, metadata))
    monkeypatch.setattr(causal, 'load_causalpfn_checkpoint', lambda *args: model)
    class Called(Exception):
        pass
    def inspect(**kwargs):
        assert kwargs['batch']['tokens'].shape == (1,24,5)
        assert kwargs['localized_arm_samples'] == 16
        assert kwargs['localized_bandwidth'] == (2. if bandwidth is None else bandwidth)
        assert kwargs['nsi_min_support_mass'] == (1. if support_mass is None else support_mass)
        assert kwargs['standard_ite_epochs'] == 500
        assert kwargs['tnet_epochs'] == 160
        assert kwargs['causalpfn_model'] is model
        raise Called
    monkeypatch.setattr(unified, 'evaluate_unified_benchmark', inspect)
    with pytest.raises(Called):
        entry.main(['--checkpoint',str(checkpoint),'--output-dir',str(tmp_path/'results'),
                    '--device','cpu','--graphs','1','--causalpfn-checkpoint',str(checkpoint)] +
                   ([] if support_mass is None else ['--nsi-min-support-mass', str(support_mass)]) +
                   ([] if bandwidth is None else ['--localized-bandwidth', str(bandwidth)]))
