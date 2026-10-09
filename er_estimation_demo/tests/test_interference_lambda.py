"""One graph, matched DGP strength, separate fits for the three experiments."""
from dataclasses import asdict, replace
import json

import pytest
import torch

import train_local_network_interference as demo
from csv_random_dgp import generate_random_batch
from evaluate_matched_old_sampler import generate_matched_test_batch, build_arg_parser


def config(**kwargs):
    return demo.DataConfig(n_units=24, er_edge_probability=.25,
        graph_protocol='fixed_er_v1', dgp_version='a_group_fixed_v1', **kwargs)


@pytest.mark.parametrize('strength', [0., 1., 2., 3.])
def test_scaled_outcomes_and_effects_on_identical_random_inputs(strength):
    cfg = config(interference_lambda=strength)
    base = generate_matched_test_batch(replace(cfg, interference_lambda=1.), [0])
    actual = generate_matched_test_batch(cfg, [0])
    for key in ('adjacency', 'x', 'neighbor_x', 'observed_treatment',
                'observed_exposure', 'noise', 'queries', 'tau'):
        assert torch.equal(actual[key], base[key]), key
    x, t, e = (actual[k] for k in ('x', 'observed_treatment', 'observed_exposure'))
    expected = 1-x-3*strength*actual['neighbor_x']+(6+.2*x.exp())*t-.9*strength*e+actual['noise']
    torch.testing.assert_close(actual['y_obs'], expected)
    torch.testing.assert_close(actual['oracle_ite'][..., 0], base['oracle_ite'][..., 0])
    torch.testing.assert_close(actual['oracle_ite'][..., 1], strength*base['oracle_ite'][..., 1])
    torch.testing.assert_close(actual['oracle_ite'][..., 2],
        actual['oracle_ite'][..., 0]+actual['oracle_ite'][..., 1])
    torch.testing.assert_close(actual['oracle_ate'], actual['oracle_ite'].mean(1))
    ta, ea, tb, eb = actual['queries'].unbind(-1)
    torch.testing.assert_close(actual['query_effect'], actual['tau'][..., None]*(ta-tb)-.9*strength*(ea-eb))
    # The training stream must receive the same coefficient change.
    train = generate_random_batch(cfg, task_ids=[0], seed=12345)
    torch.testing.assert_close(train['gamma'], torch.full_like(train['x'], -.9*strength))
    assert not torch.equal(train['x'], actual['x'])


def test_serialization_defaults_and_test_count():
    cfg = config(interference_lambda=3.)
    saved = json.loads(json.dumps(asdict(cfg)))
    restored = demo.data_config_from_mapping(saved)
    assert restored == cfg
    saved.pop('interference_lambda')
    assert demo.data_config_from_mapping(saved).interference_lambda == 1.
    assert build_arg_parser().parse_args(['--checkpoint', 'x', '--output-dir', 'y']).graphs == 1


@pytest.mark.parametrize('strength', [-1., float('nan'), float('inf')])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError, match='interference_lambda'):
        config(interference_lambda=strength)


def test_lambda_cannot_silently_affect_other_dgps():
    with pytest.raises(ValueError, match='interference_lambda'):
        demo.DataConfig(interference_lambda=2.)


def test_csv_roundtrip_and_cache_separation(tmp_path):
    cfg = config(interference_lambda=2.)
    path = tmp_path/'lambda2.csv'
    args = dict(data_csv=path, num_datasets=4, data_seed=17,
                generation_batch_size=2, validation_fraction=.25, split_seed=0)
    demo.prepare_episode_csv_files(cfg, **args)
    manifest = json.loads(path.with_suffix('.csv.manifest.json').read_text())
    spec = manifest['identity']['outcome_dgp']
    assert spec['interference_lambda'] == 2.
    assert spec['neighbor_coefficient'] == -6.
    assert spec['spillover_coefficient'] == -1.8
    loaded, ids = demo.load_episode_batch_csv(path, device=torch.device('cpu'))
    original = generate_random_batch(cfg, task_ids=ids, seed=17)
    for key in demo.CSV_TENSOR_KEYS:
        assert torch.equal(loaded[key], original[key].to(loaded[key].dtype)), key
    before = path.read_bytes()
    with pytest.raises(ValueError, match='identity|config'):
        demo.prepare_episode_csv_files(replace(cfg, interference_lambda=3.), **args)
    assert path.read_bytes() == before


def test_test_lambda_override_preserves_training_config_and_graph():
    from evaluate_matched_old_sampler import evaluation_config
    train = config(interference_lambda=1.)
    for strength in (1., 2., 3.):
        test = evaluation_config(train, strength)
        assert train.interference_lambda == 1.
        assert test.interference_lambda == strength
        assert test.fixed_graph_upper_hex == train.fixed_graph_upper_hex
        assert {k: v for k, v in asdict(test).items() if k != 'interference_lambda'} == {
            k: v for k, v in asdict(train).items() if k != 'interference_lambda'}
    assert evaluation_config(train, None) == train
    args = build_arg_parser().parse_args(['--checkpoint', 'x', '--output-dir', 'y', '--test-lambda', '3'])
    assert args.test_lambda == 3.
