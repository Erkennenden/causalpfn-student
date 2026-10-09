"""Catch fixed/per-node gamma, accidental RNG shifts, and stale dataset reuse."""
from dataclasses import asdict, replace
import json

import pytest
import torch

import train_local_network_interference as demo
from csv_random_dgp import generate_random_batch
from evaluate_matched_old_sampler import evaluation_config, generate_matched_test_batch, build_arg_parser


def config(**kwargs):
    return demo.DataConfig(n_units=24, er_edge_probability=.25,
        graph_protocol='fixed_er_v1', dgp_version='a_group_random_gamma_v1', **kwargs)


def test_gamma_shared_within_task_varies_across_tasks_without_rng_drift():
    cfg = config()
    ids = list(range(32))
    actual = generate_random_batch(cfg, task_ids=ids, seed=17)
    old = generate_random_batch(replace(cfg, dgp_version='a_group_fixed_v1'), task_ids=ids, seed=17)
    gamma = actual['gamma'][:, 0]
    assert gamma.min() >= -3.6 and gamma.max() <= 0
    assert gamma.unique().numel() == len(ids)
    assert gamma.min() < -2.7 and gamma.max() > -.9
    assert torch.equal(actual['gamma'], gamma[:, None].expand_as(actual['gamma']))
    for key in ('adjacency', 'x', 'neighbor_x', 'observed_treatment', 'observed_exposure', 'noise', 'queries', 'tau', 'structural_baseline'):
        assert torch.equal(actual[key], old[key]), key
    x, t, e = (actual[k] for k in ('x', 'observed_treatment', 'observed_exposure'))
    expected = 1-x-3*actual['neighbor_x']+(6+.2*x.exp())*t+gamma[:, None]*e+actual['noise']
    torch.testing.assert_close(actual['y_obs'], expected)
    assert actual['tokens'].shape[-1] == 5
    assert torch.equal(actual['tokens'][..., 2], actual['y_obs'])
    assert torch.equal(actual['prior_gamma'], gamma)
    torch.testing.assert_close(actual['oracle_ite'][..., 1], old['oracle_ite'][..., 1]*(gamma[:, None]/-.9))
    torch.testing.assert_close(actual['oracle_ite'][..., 0], old['oracle_ite'][..., 0])
    torch.testing.assert_close(actual['oracle_ate'], actual['oracle_ite'].mean(1))


def test_gamma_reproducible_under_batching_order_and_data_split():
    cfg = config()
    all_tasks = generate_random_batch(cfg, task_ids=[4, 1, 9], seed=81)
    for offset, task_id in enumerate([4, 1, 9]):
        single = generate_random_batch(cfg, task_ids=[task_id], seed=81)
        for key in all_tasks:
            assert torch.equal(single[key][0], all_tasks[key][offset]), key
    other_seed = generate_random_batch(cfg, task_ids=[4, 1, 9], seed=82)
    assert not torch.equal(other_seed['gamma'], all_tasks['gamma'])


@pytest.mark.parametrize('gamma', [-.9, -1.8, -2.7, 0.])
def test_fixed_gamma_tests_change_y_and_truth_only(gamma):
    train = config()
    cfg = evaluation_config(train, test_gamma=gamma)
    actual = generate_matched_test_batch(cfg, [0])
    old = generate_matched_test_batch(replace(train, dgp_version='a_group_fixed_v1'), [0])
    for key in ('adjacency', 'x', 'observed_treatment', 'observed_exposure', 'noise', 'queries', 'structural_baseline'):
        assert torch.equal(actual[key], old[key]), key
    torch.testing.assert_close(actual['y_obs']-old['y_obs'], (gamma+.9)*actual['observed_exposure'], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual['gamma'], torch.full_like(actual['x'], gamma))
    torch.testing.assert_close(actual['oracle_ite'][..., 1], old['oracle_ite'][..., 1]*(gamma/-.9))
    torch.testing.assert_close(actual['oracle_ite'][..., 2], actual['oracle_ite'][..., 0]+actual['oracle_ite'][..., 1])
    assert train.spillover_gamma is None
    assert cfg.fixed_graph_upper_hex == train.fixed_graph_upper_hex


@pytest.mark.parametrize('kwargs', [
    {'spillover_gamma_min': 0., 'spillover_gamma_max': 0.},
    {'spillover_gamma_min': 1., 'spillover_gamma_max': 0.},
    {'spillover_gamma_min': float('nan')},
    {'spillover_gamma_max': float('inf')},
    {'spillover_gamma': float('nan')},
    {'interference_lambda': 2.},
])
def test_invalid_prior_or_mixed_lambda_rejected(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_gamma_evaluation_cli_and_checkpoint_roundtrip():
    cfg = config()
    assert demo.data_config_from_mapping(json.loads(json.dumps(asdict(cfg)))) == cfg
    args = build_arg_parser().parse_args(['--checkpoint', 'x', '--output-dir', 'y', '--test-gamma', '-1.8'])
    assert args.test_gamma == -1.8
    with pytest.raises(ValueError):
        evaluation_config(cfg, test_lambda=2., test_gamma=-1.8)
    with pytest.raises(ValueError):
        evaluation_config(cfg, test_lambda=2.)
    # Old checkpoint can be evaluated under the new test DGP without altering metadata.
    old = replace(cfg, dgp_version='a_group_fixed_v1')
    assert evaluation_config(old, test_gamma=-2.7).spillover_gamma == -2.7
    assert old.dgp_version == 'a_group_fixed_v1'


def test_csv_gamma_roundtrip_and_prior_identity(tmp_path):
    cfg = config()
    path = tmp_path/'random_gamma.csv'
    args = dict(data_csv=path, num_datasets=4, data_seed=17,
                generation_batch_size=2, validation_fraction=.25, split_seed=0)
    demo.prepare_episode_csv_files(cfg, **args)
    loaded, ids = demo.load_episode_batch_csv(path, device=torch.device('cpu'))
    original = generate_random_batch(cfg, task_ids=ids, seed=17)
    for key in demo.CSV_TENSOR_KEYS:
        assert torch.equal(loaded[key], original[key].to(loaded[key].dtype)), key
    identity = json.loads(path.with_suffix('.csv.manifest.json').read_text())['identity']
    assert identity['outcome_dgp']['spillover_distribution'] == {'family': 'uniform', 'low': -3.6, 'high': 0.}
    assert identity['outcome_dgp']['neighbor_coefficient'] == -3.
    before = path.read_bytes()
    for changed in (replace(cfg, spillover_gamma_min=-4.), replace(cfg, spillover_gamma=-.9), replace(cfg, dgp_version='a_group_fixed_v1')):
        with pytest.raises(ValueError, match='identity|config'):
            demo.prepare_episode_csv_files(changed, **args)
        assert path.read_bytes() == before


def test_prepare_only_cli_and_ddp_shards_preserve_random_gamma(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    bank=tmp_path/'gamma.csv'
    cmd=[sys.executable,str(root/'train_local_network_interference.py'),
         '--prepare-data-only','--device','cpu','--dgp-version','a_group_random_gamma_v1',
         '--n-units','24','--er-edge-probability','.25','--fixed-graph-file',str(tmp_path/'graph.json'),
         '--data-csv',str(bank),'--csv-datasets','8','--validation-fraction','.25',
         '--data-seed','42','--spillover-gamma-min','-3.6','--spillover-gamma-max','0']
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
    meta=json.loads(Path(str(bank)+'.manifest.json').read_text())
    cfg=demo.data_config_from_mapping(meta['identity']['data_config'])
    shard_dir=tmp_path/'shards'
    demo.prepare_ddp_train_shards(bank,shard_dir=shard_dir,world_size=2)
    seen=[]
    for rank in range(2):
        loaded,ids=demo.load_episode_batch_csv(shard_dir/f'train_rank{rank}.csv',device=torch.device('cpu'))
        seen.extend(ids)
        expected=generate_random_batch(cfg,task_ids=ids,seed=42)
        for key in ('tokens','gamma','query_effect','oracle_ite'):
            assert torch.equal(loaded[key],expected[key].to(loaded[key].dtype)),key
    assert sorted(seen)==sorted(meta['training_dataset_ids'])
    assert not set(seen).intersection(meta['validation_dataset_ids'])
