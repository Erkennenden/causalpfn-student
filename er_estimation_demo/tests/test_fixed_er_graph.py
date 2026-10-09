"""The experimental unit is a data realization on one immutable ER network."""
from dataclasses import asdict, replace
import json

import pytest
import torch

import train_local_network_interference as demo
from csv_random_dgp import generate_random_batch


def fixed_config(**overrides):
    values = dict(n_units=24, graph_family='er', dgp_version='a_group_fixed_v1',
                  graph_protocol='fixed_er_v1', er_edge_probability=.25,
                  fixed_graph_seed=731)
    values.update(overrides)
    return demo.DataConfig(**values)


def test_one_network_across_tasks_seeds_streams_and_chunks():
    cfg = fixed_config()
    train = generate_random_batch(cfg, task_ids=[0, 1, 2], seed=12, stream='train')
    test = generate_random_batch(cfg, task_ids=[40, 7], seed=2026, stream='test', balanced=True)
    graph = train['adjacency'][0]
    for batch in (train, test):
        assert torch.equal(batch['adjacency'], graph.expand_as(batch['adjacency']))
        assert torch.equal(batch['degree'], graph.sum(-1).expand_as(batch['degree']))
        assert batch['graph_type'].unique().tolist() == [demo.ER_GRAPH]
    assert not torch.equal(train['x'][0], train['x'][1])
    assert not torch.equal(train['observed_treatment'][0], train['observed_treatment'][1])
    assert not torch.equal(train['noise'][0], test['noise'][0])
    parts = [generate_random_batch(cfg, task_ids=[i], seed=12, stream='train') for i in range(3)]
    for key in train:
        assert torch.equal(train[key], torch.cat([part[key] for part in parts])), key
    assert torch.equal(graph, graph.T)
    assert not graph.diagonal().any()
    assert (graph.sum(-1) > 0).all()


def test_a_equation_and_nongraph_random_streams_are_unchanged():
    cfg = fixed_config()
    fixed = generate_random_batch(cfg, task_ids=range(3), seed=39)
    old = generate_random_batch(demo.DataConfig(n_units=24, graph_family='er',
                                dgp_version='a_group_fixed_v1'), task_ids=range(3), seed=39)
    for key in ('x', 'observed_treatment', 'noise', 'tau'):
        assert torch.equal(fixed[key], old[key]), key
    x, t = fixed['x'], fixed['observed_treatment']
    neighbor_x = torch.einsum('bij,bj->bi', fixed['adjacency'], x) / fixed['degree']
    exposure = torch.einsum('bij,bj->bi', fixed['adjacency'], t) / fixed['degree']
    expected = 1 - x - 3*neighbor_x + (6 + .2*x.exp())*t - .9*exposure + fixed['noise']
    torch.testing.assert_close(fixed['y_obs'], expected)
    assert fixed['prior_noise_sd'].unique().tolist() == [4.]
    assert fixed['regime'].unique().tolist() == [3]
    ta, ea, tb, eb = fixed['queries'].unbind(-1)
    torch.testing.assert_close(fixed['query_effect'],
                              (6 + .2*x.exp())[..., None]*(ta-tb) - .9*(ea-eb))


def test_serialized_config_restores_graph_without_sampling(monkeypatch):
    cfg = fixed_config()
    expected = generate_random_batch(cfg, task_ids=[0], seed=1)['adjacency']
    saved = json.loads(json.dumps(asdict(cfg)))
    assert saved['fixed_graph_upper_hex']
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved graphs must be decoded, never sampled again')
    monkeypatch.setattr(demo, '_generate_er_adjacency', forbidden)
    restored = demo.data_config_from_mapping(saved)
    actual = generate_random_batch(restored, task_ids=[100], seed=999, stream='benchmark')['adjacency']
    assert torch.equal(actual, expected)


def test_graph_file_is_reused_and_rejects_mismatched_parameters(tmp_path, monkeypatch):
    from fixed_er_graph import load_or_create_graph
    path = tmp_path/'graph.json'
    first = load_or_create_graph(path, n_units=24, edge_probability=.25, seed=731)
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Existing graph file must be reused')
    monkeypatch.setattr(demo, '_generate_er_adjacency', forbidden)
    assert load_or_create_graph(path, n_units=24, edge_probability=.25, seed=731) == first
    with pytest.raises(ValueError, match='match|parameters'):
        load_or_create_graph(path, n_units=24, edge_probability=.25, seed=732)
    assert path.read_bytes() == before


def test_saved_fixed_config_cannot_silently_regenerate_a_missing_graph():
    saved = asdict(fixed_config())
    saved.pop('fixed_graph_upper_hex')
    with pytest.raises(ValueError, match='saved|adjacency|graph'):
        demo.data_config_from_mapping(saved)


def test_fixed_csv_split_roundtrip_and_graph_identity(tmp_path):
    cfg = fixed_config()
    path = tmp_path/'data.csv'
    args = dict(data_csv=path, num_datasets=6, data_seed=42,
                generation_batch_size=2, validation_fraction=1/3, split_seed=7)
    demo.prepare_episode_csv_files(cfg, **args)
    manifest = json.loads(path.with_suffix('.csv.manifest.json').read_text())
    assert manifest['identity']['fixed_graph']['unique_graph_count'] == 1
    graph = generate_random_batch(cfg, task_ids=[0], seed=42)['adjacency'][0]
    for csv_path in (path, *demo.episode_split_paths(path)):
        loaded, _ = demo.load_episode_batch_csv(csv_path, device=torch.device('cpu'))
        assert torch.equal(loaded['adjacency'], graph.expand_as(loaded['adjacency']))
    before = path.read_bytes()
    other = fixed_config(fixed_graph_seed=732)
    assert other.fixed_graph_upper_hex != cfg.fixed_graph_upper_hex
    with pytest.raises(ValueError, match='identity|config'):
        demo.prepare_episode_csv_files(other, **args)
    assert path.read_bytes() == before


def test_fixed_protocol_rejects_other_families_and_invalid_graphs():
    with pytest.raises(ValueError, match='ER|er'):
        fixed_config(graph_family='mixed')
    with pytest.raises(ValueError, match='graph|hex|isolat'):
        fixed_config(fixed_graph_upper_hex='00')
    with pytest.raises(ValueError, match='ER|er'):
        generate_random_batch(fixed_config(), task_ids=[0], seed=1,
                              graph_types=torch.tensor([demo.SBM_GRAPH]))


def test_release_network_has_1000_nodes_and_survives_train_test_stream_changes(monkeypatch):
    from pathlib import Path
    from fixed_er_graph import load_or_create_graph, adjacency_sha256
    path = Path(__file__).resolve().parents[1]/'data/fixed_er_N1000_p0.02_seed12345.json'
    def forbidden(*args, **kwargs):
        raise AssertionError('The release network must not be resampled')
    monkeypatch.setattr(demo, '_generate_er_adjacency', forbidden)
    record = load_or_create_graph(path, n_units=1000, edge_probability=.02, seed=12345)
    cfg = demo.DataConfig(n_units=1000, graph_family='er', graph_protocol='fixed_er_v1',
                          dgp_version='a_group_fixed_v1', er_edge_probability=.02,
                          fixed_graph_seed=12345, fixed_graph_upper_hex=record['upper_hex'])
    for stream, seed in [('train',12345), ('benchmark',2026)]:
        batch = generate_random_batch(cfg, task_ids=[0,9], seed=seed, stream=stream)
        assert batch['tokens'].shape == (2,1000,5)
        assert {adjacency_sha256(a) for a in batch['adjacency']} == {record['adjacency_sha256']}
        assert record['edge_count'] == 9914
        assert batch['prior_noise_sd'].unique().tolist() == [4.]
        assert batch['regime'].unique().tolist() == [3]
