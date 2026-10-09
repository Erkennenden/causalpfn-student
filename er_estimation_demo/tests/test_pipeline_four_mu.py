"""Fixed ER1000 random-a pipeline plus analytic K10 regression fixtures for four CEPO GMMs."""
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pfn_interference.cepo import cepo_queries, cepo_targets, compute_cepo_losses, effects_from_mu
from pfn_interference.data import TaskBank
from pfn_interference.training import RunConfig, train, load_model
from pfn_interference.evaluation import evaluate, pfn_rows
from pfn_interference.graph import adjacency_sha256
from pfn_interference.model import ModelConfig, LocalNetworkQueryTransformer


def cached_a(bank,split,task_id):
    payload=torch.load(bank.directory/split/f'task_{task_id:08d}.pt',weights_only=True)
    return payload['metadata']['a']


def complete_bank(tmp_path):
    graph = torch.ones(10, 10) - torch.eye(10)
    path = tmp_path / 'graph.json'
    path.write_text(json.dumps(dict(protocol='fixed_complete_v1', n_units=10,
        edge_probability=1., seed=0, upper_hex='fffffffffff8',
        adjacency_sha256=adjacency_sha256(graph), edge_count=45,
        realized_mean_degree=9., no_isolates=True, unique_graph_count=1)))
    return TaskBank(path, tmp_path / 'cache', data_seed=21)


def test_default_pipeline_uses_one_frozen_er_1000_node_graph(tmp_path, monkeypatch):
    bank = TaskBank(RunConfig().graph_file, tmp_path / 'cache')
    assert bank.graph.shape == (1000, 1000)
    assert bank.record['seed'] == 12345 and bank.record['edge_probability'] == .02
    # Replay the existing no-isolates rejection sampler independently.
    rng=torch.Generator().manual_seed(12345)
    i,j=torch.triu_indices(1000,1000,offset=1)
    for attempt in range(1000):
        edges=torch.rand(499500,generator=rng)<.02
        expected=torch.zeros(1000,1000)
        expected[i,j]=edges.float();expected[j,i]=edges.float()
        if bool((expected.sum(-1)>0).all()): break
    else: raise AssertionError('No valid graph drawn')
    torch.testing.assert_close(bank.graph,expected)
    def forbidden(*args, **kwargs):
        raise AssertionError('A saved graph must never be redrawn for a data task')
    monkeypatch.setattr('pfn_interference.model._generate_er_adjacency', forbidden)
    for seed in (12345,2025,2029):
        other = TaskBank(RunConfig().graph_file, tmp_path / 'cache', data_seed=seed)
        for split in ('train','validation','test'):
            batch = other.batch(split,[0,1])
            torch.testing.assert_close(batch['adjacency'],expected.expand(2,1000,1000))
            assert not torch.equal(batch['x'][0],batch['x'][1])


def test_er_b_labels_use_each_nodes_degree(tmp_path):
    bank = TaskBank(RunConfig().graph_file, tmp_path / 'cache', data_seed=12345,noise_sd=0)
    assert bank.graph.shape == (1000,1000)
    b = bank.batch('test',[0])
    assert torch.unique(b['degree']).numel()>1
    a = cached_a(bank,'test',0)
    expected=torch.empty(1000,4)
    for i in range(1000):
        d=int(b['degree'][0,i]); moments=[]
        for arm in (0,1):
            counts=[k for k in range(d+1) if int(k>d//2)==arm]
            moments.append(sum(k*math.comb(d,k) for k in counts)/(d*sum(math.comb(d,k) for k in counts)))
        x=float(b['x'][0,i]); nx=float(b['neighbor_x'][0,i])
        base=1-x-3*nx; tau=a+.2*math.exp(x)
        expected[i]=torch.tensor([base-.9*moments[0],base-.9*moments[1],
                                  base+tau-.9*moments[0],base+tau-.9*moments[1]])
    torch.testing.assert_close(b['cepo_target'][0],expected,atol=2e-6,rtol=1e-6)
    assert torch.unique(b['oracle_ite'][0,:,1]).numel()>1


def test_b_dgp_factual_data_and_four_exact_noise_free_targets(tmp_path):
    bank = complete_bank(tmp_path)
    b = bank.batch('train', [0, 1])
    x = b['x']
    baseline = 1 - x - 3 * b['neighbor_x']
    coefficients=torch.tensor([cached_a(bank,'train',i) for i in (0,1)])[:,None]
    tau = coefficients + .2 * torch.exp(x)
    # For Binomial(9, 1/2), conditional E[K/9] is exactly 93/256 or 163/256.
    low, high = 93 / 256, 163 / 256
    expected = torch.stack((baseline - .9 * low, baseline - .9 * high,
                            baseline + tau - .9 * low, baseline + tau - .9 * high), -1)
    assert b['cepo_target'].shape == (2, 10, 4)
    torch.testing.assert_close(b['cepo_target'], expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(b['oracle_arm_means'].flatten(-2), expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(b['oracle_ite'][..., 0], tau)
    torch.testing.assert_close(b['oracle_ite'][..., 1], torch.full_like(x, -63 / 256))
    quiet = TaskBank(bank.graph_file, tmp_path / 'cache', data_seed=21, noise_sd=0).batch('train', [0, 1])
    torch.testing.assert_close(quiet['cepo_target'], b['cepo_target'], rtol=0, atol=0)
    factual = baseline + tau * b['observed_treatment'] - .9 * b['observed_exposure']
    torch.testing.assert_close(quiet['y_obs'], factual, atol=2e-6, rtol=1e-6)
    assert not torch.equal(quiet['y_obs'], b['y_obs'])
    assert not torch.equal(b['x'][0], bank.batch('test', [0])['x'][0])


def test_mu01_has_equal_loss_weight_and_nonzero_gradient():
    b = dict(tokens=torch.zeros(1, 2, 5), degree=torch.ones(1, 2),
             structural_baseline=torch.zeros(1, 2), tau=torch.zeros(1, 2),
             gamma=torch.zeros(1, 2), design_treatment_prob=torch.tensor([.5]))
    assert cepo_queries(b)[0, 0].tolist() == [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    target = cepo_targets(b)
    mean = torch.zeros(1, 2, 4, 1, requires_grad=True)
    pred = dict(gmm_pi=torch.ones_like(mean), gmm_mu=mean, gmm_sigma=torch.ones_like(mean))
    base = compute_cepo_losses(pred, {'cepo_target': target})['total_loss']
    shifted = target.clone()
    shifted[..., 1] = 2.
    losses = compute_cepo_losses(pred, {'cepo_target': shifted})
    assert losses['total_loss'].item() == pytest.approx(base.item() + .5)
    assert losses['mu01_gmm_nll'].item() == pytest.approx(.5 * math.log(2 * math.pi) + 2.)
    losses['total_loss'].backward()
    torch.testing.assert_close(mean.grad[..., 1, :], torch.full((1, 2, 1), -.25))
    assert torch.count_nonzero(mean.grad[..., [0, 2, 3], :]) == 0


def test_effect_contrasts_keep_original_definitions_with_four_arms():
    # Deliberately nonadditive mu01: catches accidental use of arm 01 in old effects.
    mu = torch.tensor([[[2., 100., 5., 9.]]])
    torch.testing.assert_close(effects_from_mu(mu), torch.tensor([[[3., 4., 7.]]]))


def test_real_model_and_pfn_export_preserve_all_gmm_parameters(tmp_path):
    b = complete_bank(tmp_path).batch('test', [0])
    model = LocalNetworkQueryTransformer(ModelConfig(d_model=8, num_heads=2,
        num_layers=2, ffn_dim=32)).eval()
    _, _, rows = pfn_rows(model, b)
    assert len(rows) == 40
    assert {r['arm'] for r in rows} == {'mu00', 'mu01', 'mu10', 'mu11'}
    for row in rows:
        pi, mean, sigma = [json.loads(row[k]) for k in ('gmm_pi', 'gmm_mu', 'gmm_sigma')]
        assert len(pi) == len(mean) == len(sigma) == 5
        assert sum(pi) == pytest.approx(1.) and min(pi) > 0 and min(sigma) > 0
        assert np.dot(pi, mean) == pytest.approx(row['estimate'], abs=1e-6)


def test_train_validation_and_ten_task_export_use_all_four_gmms(tmp_path):
    bank = TaskBank(RunConfig().graph_file,tmp_path/'cache')
    cfg = RunConfig(graph_file=str(bank.graph_file), cache_dir=str(tmp_path / 'cache'),
        device='cpu', train_tasks=4, validation_tasks=2, epochs=1, batch_size=2,
        num_layers=2, d_model=8, num_heads=2)
    path = train(cfg, tmp_path / 'training')
    model, state = load_model(path)
    assert len(model.encoder_layers) == 2
    history = json.loads((tmp_path / 'training/history.jsonl').read_text().splitlines()[-1])
    for split in ('train', 'validation'):
        metrics = history[split]
        values = [metrics[f'mu{arm}_gmm_nll'] for arm in ('00', '01', '10', '11')]
        assert metrics['total_loss'] == pytest.approx(sum(values) / 4)
    assert state['best_validation_nll'] == pytest.approx(history['validation']['total_loss'])
    assert (tmp_path / 'training/four_mu_loss_curves.png').is_file()
    with (tmp_path / 'training/validation_metrics.csv').open() as f:
        saved_metrics = list(csv.DictReader(f))
    assert float(saved_metrics[0]['validation_mu01_gmm_nll']) == pytest.approx(history['validation']['mu01_gmm_nll'])
    result = evaluate(path, tmp_path / 'evaluation', device='cpu', baselines=False)
    assert result['n_test_datasets'] == 10 and result['n_distinct_graphs'] == 1
    assert result['scenarios'] == ['linear_b_random_a']
    out = tmp_path / 'evaluation/linear_b_random_a'
    with (out / 'pfn_mu_predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40000
    assert {int(r['dataset_id']) for r in rows} == set(range(1, 11))
    for arm in ('00', '01', '10', '11'):
        with (out / f'pfn_mu{arm}_predictions.csv').open() as f:
            separated = list(csv.DictReader(f))
        assert len(separated) == 10000
        assert {r['arm'] for r in separated} == {f'mu{arm}'}
        assert all(len(json.loads(r['gmm_pi'])) == 5 for r in separated)
    with (out / 'pfn_mu_graph_means.csv').open() as f:
        graph_means = list(csv.DictReader(f))
    assert len(graph_means) == 10
    with (tmp_path/'evaluation/test_datasets.csv').open() as f:
        schedule=list(csv.DictReader(f))
    assert len({float(r['a']) for r in schedule})==10
    testbank=TaskBank(bank.graph_file,tmp_path/'cache',data_seed=2025)
    testbank.batch('test',[0])
    assert float(schedule[0]['a'])==cached_a(testbank,'test',0)
    assert float(rows[0]['a'])==float(schedule[0]['a'])
    with (out / 'pfn_mu_gmm_components.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 200000
    with (out / 'pfn_mu_summary.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 4


def test_three_arm_checkpoint_is_rejected(tmp_path):
    from pfn_interference.training import VERSION
    path = tmp_path / 'old.pt'
    torch.save(dict(version=VERSION, prediction_protocol='majority_cepo_gmm_v1'), path)
    with pytest.raises(ValueError, match='checkpoint|Checkpoint'):
        load_model(path)
