import math
import importlib.util
import numpy as np
import pytest
import torch
from pathlib import Path

from pfn_interference.simulation import generate_task
from pfn_interference.cepo import cepo_targets, effects_from_mu


def test_json_cache_identity_is_atomic_for_concurrent_rank_writers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import os
    import json
    from pfn_interference.data import write_json
    barrier = threading.Barrier(2)
    real_replace = os.replace
    def replace(source, target):
        barrier.wait(timeout=5)
        return real_replace(source, target)
    monkeypatch.setattr(os, 'replace', replace)
    monkeypatch.setattr(Path, 'replace', lambda self, target: os.replace(self, target))
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(write_json, tmp_path/'identity.json', {'rank': i}) for i in range(2)]
        for future in futures:
            future.result()
    assert json.loads((tmp_path/'identity.json').read_text())['rank'] in (0, 1)
    assert not list(tmp_path.glob('*.tmp'))


def test_docker_launcher_propagates_immediate_training_failure(tmp_path):
    import subprocess
    source = (Path(__file__).resolve().parents[1]/'scripts/launch_musa.sh').read_text()
    inside = source.split("<<'INSIDE'\n", 1)[1].rsplit('\nINSIDE', 1)[0]
    inside = inside.replace('/workspace/latest_causalfm_project.txt', str(tmp_path/'latest.txt'))
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/train.sh').write_text('exit 42\n')
    result = subprocess.run(['bash', '-s', '--', str(tmp_path)], input=inside, text=True, capture_output=True)
    assert result.returncode == 42, result.stdout + result.stderr
    assert (tmp_path/'.host_exit_code').read_text().strip() == '42'


def ring(n=12):
    a = torch.zeros(n, n)
    for i in range(n):
        a[i, (i+1) % n] = a[(i+1) % n, i] = 1
    return a


def nonlinear(z):
    return 2 + 3*z[..., 2] + (1+z[..., 2])*z[..., 3]**2


def test_nonlinear_cepo_integrates_outcomes_not_average_exposure():
    b = generate_task(ring(), seed=2026, outcome=nonlinear).batch
    assert 'cepo_target' in b, 'random-function tasks must carry exact CEPO labels'
    # Degree two: low-count weights are 1/3,2/3 at exposures 0,.5.
    expected = torch.tensor([13/6, 3., 16/3, 7.]).expand(1, 12, 4)
    torch.testing.assert_close(cepo_targets(b), expected)
    torch.testing.assert_close(b['oracle_ite'], torch.tensor([19/6, 5/3, 29/6]).expand(1, 12, 3))
    torch.testing.assert_close(effects_from_mu(cepo_targets(b)), b['oracle_ite'])
    assert b['queries'].shape == (1, 12, 4, 2)


def test_external_noise_never_enters_cepo_and_function_shared_with_factual_data():
    a = generate_task(ring(), seed=71, noise_sd=0).batch
    b = generate_task(ring(), seed=71, noise_sd=40).batch
    assert 'cepo_target' in a
    torch.testing.assert_close(a['cepo_target'], b['cepo_target'], rtol=0, atol=0)
    torch.testing.assert_close(a['y_obs']+b['noise'], b['y_obs'])
    task = generate_task(ring(), seed=71, noise_sd=0)
    inputs = torch.stack([a['x'], a['neighbor_x'], a['observed_treatment'], a['observed_exposure']], -1)
    np.testing.assert_allclose(task.outcome(inputs.numpy()), a['y_obs'].numpy(), atol=1e-6)
    other = generate_task(ring(), task_id=1, seed=71, noise_sd=0)
    assert task.metadata['outcome_seed'] != other.metadata['outcome_seed']


def test_ate_baselines_accept_exact_nonlinear_arm_means_without_fake_linear_parameters():
    from pfn_interference.baselines.runner import run_ate_baselines_on_batch
    from pfn_interference.baselines.tnet import TNetConfig
    b = generate_task(ring(24), seed=2026, outcome=nonlinear).batch
    result = run_ate_baselines_on_batch(batch=b, treatment_prob=.5, bandwidth=1, ridge=1e-3,
        tnet_config=TNetConfig(epochs=1, hidden_dim=4))
    assert len(result['graph_effects']) == 3
    assert [r['truth'] for r in result['graph_effects']] == pytest.approx([19/6, 5/3, 29/6])


def test_task_bank_preserves_cepo_and_split_reproducibility(tmp_path):
    assert importlib.util.find_spec('pfn_interference') is not None, 'new CEPO experiment entry is missing'
    from pfn_interference.data import TaskBank
    from pfn_interference.graph import load_or_create_graph
    graph = tmp_path/'graph.json'
    load_or_create_graph(graph, n_units=12, edge_probability=.4, seed=12)
    bank = TaskBank(graph, tmp_path/'cache', data_seed=5)
    a = bank.batch('train', [0,1])
    b = TaskBank(graph, tmp_path/'cache', data_seed=5).batch('train', [0,1])
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert a['cepo_target'].shape == (2,12,4)
    assert not torch.equal(a['cepo_target'][0], bank.batch('validation',[0])['cepo_target'][0])


def test_train_resume_and_checkpoint_inference_are_cepo_consistent(tmp_path):
    assert importlib.util.find_spec('pfn_interference.training') is not None, 'CEPO trainer missing'
    from pfn_interference.training import RunConfig, train, load_model
    from pfn_interference.graph import load_or_create_graph
    from dataclasses import replace
    graph=tmp_path/'g.json'
    load_or_create_graph(graph,n_units=12,edge_probability=.4,seed=12)
    cfg=RunConfig(graph_file=str(graph),cache_dir=str(tmp_path/'cache'),device='cpu',
        train_tasks=4,validation_tasks=2,epochs=1,batch_size=2,num_layers=1,d_model=8,num_heads=2)
    train(cfg,tmp_path/'run')
    first=torch.load(tmp_path/'run/model_last.pt',weights_only=True)
    assert first['prediction_protocol']=='majority_cepo_four_gmm_v2'
    assert first['task_presentations']==4
    train(replace(cfg,epochs=2),tmp_path/'run',resume=True)
    train(replace(cfg,epochs=2),tmp_path/'continuous')
    resumed=torch.load(tmp_path/'run/model_last.pt',weights_only=True)
    continuous=torch.load(tmp_path/'continuous/model_last.pt',weights_only=True)
    for k,v in resumed['model_state'].items():
        torch.testing.assert_close(v,continuous['model_state'][k],rtol=0,atol=0)
    model,state=load_model(tmp_path/'run/model_best.pt',device='cpu')
    assert state['epoch'] in (1,2)
    assert any(not torch.equal(first['model_state'][k],resumed['model_state'][k]) for k in first['model_state'])
    assert (tmp_path/'run/validation_curves.png').exists()


@pytest.mark.parametrize('entry', ['python', 'cli'])
@pytest.mark.parametrize('legacy', [False, True])
def test_evaluation_exports_same_test_datasets_for_ate_and_ite(tmp_path, entry, legacy):
    assert importlib.util.find_spec('pfn_interference.evaluation') is not None, 'new evaluator missing'
    from pfn_interference.training import RunConfig, train
    from pfn_interference.evaluation import evaluate
    from pfn_interference.data import TaskBank
    from pfn_interference.graph import load_or_create_graph
    graph=tmp_path/'g.json';load_or_create_graph(graph,n_units=12,edge_probability=.4,seed=12)
    cfg=RunConfig(graph_file=str(graph),cache_dir=str(tmp_path/'cache'),device='cpu',
        train_tasks=2,validation_tasks=1,epochs=1,batch_size=1,num_layers=1,d_model=8,num_heads=2)
    path=train(cfg,tmp_path/'run')
    seeds=[2026] if legacy else [2025,2026,2027,2028,2029]
    tasks=1 if legacy else 2
    count=len(seeds)*tasks
    if entry=='python':
        options=dict(test_seed=2026,test_tasks=1) if legacy else {}
        result=evaluate(path,tmp_path/'eval',device='cpu',baselines=False,**options)
    else:
        import subprocess, sys, json
        subprocess.run([sys.executable,'-m','pfn_interference','evaluate',
            '--checkpoint',str(path),'--output',str(tmp_path/'eval'),
            '--device','cpu','--pfn-only']+(['--test-seed','2026','--test-tasks','1'] if legacy else []),
            check=True,capture_output=True,text=True)
        result=json.loads((tmp_path/'eval/results.json').read_text())
    assert result['scenarios']==['linear_b_random_a']
    assert result['test_tasks']==tasks and result['test_seeds']==seeds
    assert result['n_test_datasets']==count
    assert len(result['summary'])==6
    assert not list((tmp_path/'eval').glob('lambda_*'))
    expected_truth={(seed,task):TaskBank(graph,tmp_path/'cache',data_seed=seed).batch('test',[task])['oracle_ite'][0]
                    for seed in seeds for task in range(tasks)}
    import csv
    with (tmp_path/'eval/linear_b_random_a/ite_unit_results.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==count*12*3
    assert {int(r['dataset_id']) for r in rows}==set(range(1,count+1))
    for r in rows:
        effect=('direct','spillover','total').index(r['effect'])
        expected=expected_truth[int(r['test_seed']),int(r['task_id'])]
        assert float(r['truth'])==pytest.approx(float(expected[int(r['unit_id'])-1,effect]))
    with (tmp_path/'eval/linear_b_random_a/ate_graph_results.csv').open() as f: ate=list(csv.DictReader(f))
    assert len(ate)==count*3
    with (tmp_path/'eval/test_datasets.csv').open() as f: manifest=list(csv.DictReader(f))
    assert [(int(r['test_seed']),int(r['task_id'])) for r in manifest]==list(expected_truth)
    with (tmp_path/'eval/linear_b_random_a/per_dataset_summary.csv').open() as f: per_dataset=list(csv.DictReader(f))
    assert len(per_dataset)==count*6
    for r in result['summary']:
        source=ate if r['level']=='ATE' else rows
        errors=np.array([float(x['estimate'])-float(x['truth']) for x in source if x['effect']==r['effect']])
        assert r['n_total']==count*(1 if r['level']=='ATE' else 12)
        assert r['n_datasets_total']==count
        assert r['rmse']==pytest.approx(float(np.sqrt(np.mean(errors**2))))
        if legacy and r['level']=='ATE':
            assert r['mae']==pytest.approx(abs(r['bias']))
            assert r['rmse']==pytest.approx(r['mae'])
