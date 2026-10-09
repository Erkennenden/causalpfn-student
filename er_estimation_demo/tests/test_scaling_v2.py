"""V2 contracts: controlled sweeps, real resume, and independent seed statistics."""
import csv
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scaling_v1.runner import RunConfig

ROOT = Path(__file__).resolve().parents[1]


def mod(name):
    assert importlib.util.find_spec('scaling_v2') is not None, 'V2 is not implemented'
    return importlib.import_module('scaling_v2.'+name)


@pytest.fixture
def cfg(tmp_path):
    from fixed_er_graph import adjacency_sha256
    graph = torch.ones(3,3)-torch.eye(3)
    path = tmp_path/'graph.json'
    path.write_text(json.dumps(dict(n_units=3,upper_hex='e0',adjacency_sha256=adjacency_sha256(graph))))
    return RunConfig(graph_file=str(path),cache_dir=str(tmp_path/'cache'),model_size='small',
        train_tasks=4,validation_tasks=1,test_tasks=1,steps=2,batch_size=1,
        validation_every=2,checkpoint_every=2,log_every=1,threads=1,device='cpu')


def test_axis_plan_changes_only_one_factor_and_deduplicates_anchor(cfg):
    jobs = mod('sweep').build_jobs(cfg,task_counts=[2,4],model_sizes=['small','medium'],
        step_budgets=[1,2],seeds=[11,12])
    triples = {(j.config.train_tasks,j.config.model_size,j.config.steps) for j in jobs}
    assert triples == {(2,'small',2),(4,'small',2),(4,'medium',2),(4,'small',1)}
    assert len(jobs)==8 and len({j.name for j in jobs})==8
    assert all(j.config.data_seed==12345 for j in jobs)


@pytest.mark.parametrize('change', [{'seeds':[3,3]}, {'task_counts':[0,4]},
    {'step_budgets':[]}, {'model_sizes':['unknown']}, {'task_counts':[2]}, {'seeds':[-1]}])
def test_invalid_plan_fails_before_creating_output(cfg,change):
    args=dict(task_counts=[2,4],model_sizes=['small'],step_budgets=[1,2],seeds=[11])
    args.update(change)
    with pytest.raises(ValueError):
        mod('sweep').build_jobs(cfg,**args)


def result_fixture(root,cfg,seed,value,*,device='cpu',identity='same-data',version=None):
    from scaling_v1 import VERSION
    cfg=replace(cfg,train_seed=seed)
    run=root/f'run_{len(list(root.glob("run_*")))}'
    run.mkdir(parents=True)
    result=dict(version=version or VERSION,status='complete',config=asdict(cfg),data_identity=identity,
        device=device,parameters=100,evaluated_step=1,best_validation_nll=value,
        unique_train_tasks_seen=2,task_presentations=2,test_seconds=1.,
        timing=dict(training_seconds=value,data_seconds=1.,validation_seconds=2.),
        effects=[dict(effect=e,ite_rmse=value,ite_mae=value,ate_rmse=value,ate_bias=value,
                      zero_predictor_ite_rmse=5.) for e in ('direct','spillover','total')])
    manifest=dict(version=result['version'],config=result['config'],data_identity=identity,
                  model_config=dict(d_model=32),torch_version='test-runtime')
    for name,payload in [('results.json',result),('manifest.json',manifest),
                         ('status.json',dict(status='complete',step=2,evaluated_step=1))]:
        (run/name).write_text(json.dumps(payload))
    return run


def test_seed_report_uses_sample_sd_and_keeps_checkpoint_coverage(cfg,tmp_path):
    root=tmp_path/'report'
    result_fixture(root,cfg,11,1.)
    result_fixture(root,cfg,12,3.)
    rows=mod('report').summarize(root)
    direct=next(r for r in rows if r['effect']=='direct')
    assert direct['ite_rmse_mean']==2.
    assert direct['ite_rmse_std']==pytest.approx(math.sqrt(2))
    assert direct['seed_count']==2
    assert direct['evaluated_unique_train_tasks_seen_mean']==1
    assert direct['unique_train_tasks_seen_mean']==2
    assert direct['evaluated_task_coverage_mean']==.25
    assert direct['observed_task_coverage_mean']==.5
    assert direct['compute_seconds_mean']==6.
    assert (root/'summary.csv').exists() and (root/'aggregate_summary.csv').exists()


def test_report_never_combines_different_data_devices_or_learning_rates(cfg,tmp_path):
    root=tmp_path/'report'
    result_fixture(root,cfg,11,1.)
    result_fixture(root,cfg,12,3.,identity='other-data')
    result_fixture(root,cfg,13,4.,device='musa:0')
    result_fixture(root,replace(cfg,learning_rate=.02),14,5.)
    rows=mod('report').summarize(root)
    assert len(rows)==12
    assert all(r['seed_count']==1 and r['ite_rmse_std'] is None for r in rows)


def test_duplicate_seed_cannot_be_counted_as_replication(cfg,tmp_path):
    root=tmp_path/'report'
    result_fixture(root,cfg,11,1.)
    result_fixture(root,cfg,11,3.)
    with pytest.raises(ValueError,match='Duplicate'):
        mod('report').summarize(root)


def test_stale_or_incompatible_results_are_excluded(cfg,tmp_path):
    root=tmp_path/'report'
    active=result_fixture(root,cfg,11,1.)
    (active/'status.json').write_text('{"status":"training"}')
    result_fixture(root,cfg,12,2.,version='old-effect-protocol')
    rows=mod('report').summarize(root)
    assert rows==[]
    assert json.loads((root/'progress.json').read_text())['completed_runs']==0


def test_sweep_resume_skips_finished_runs_and_rejects_changed_plan(cfg,tmp_path):
    sweep=mod('sweep')
    jobs=sweep.build_jobs(cfg,task_counts=[4],model_sizes=['small'],step_budgets=[1,2],seeds=[11,12])
    root=tmp_path/'sweep'
    sweep.run_sweep(jobs,root)
    saved={p:p.stat().st_mtime_ns for p in root.glob('*/model_last.pt')}
    assert len(saved)==4
    sweep.run_sweep(jobs,root,resume=True)
    assert saved=={p:p.stat().st_mtime_ns for p in saved}
    rows=list(csv.DictReader((root/'aggregate_summary.csv').open()))
    assert len(rows)==6 and {r['seed_count'] for r in rows}=={'2'}
    assert json.loads((root/'progress.json').read_text())['pending_runs']==[]
    different=sweep.build_jobs(cfg,task_counts=[4],model_sizes=['small'],step_budgets=[1,2,3],seeds=[11,12])
    with pytest.raises(ValueError,match='plan'):
        sweep.run_sweep(different,root,resume=True)


def test_cli_dry_run_has_no_training_side_effects(tmp_path):
    mod('sweep')
    output=tmp_path/'dry_run'
    r=subprocess.run([sys.executable,'-m','scaling_v2','sweep','--dry-run','--output',str(output)],
                     cwd=ROOT,capture_output=True,text=True,check=True)
    assert '27 runs' in r.stdout
    assert not output.exists()


def test_xlarge_trains_and_preserves_cepo_effect_identity(cfg,tmp_path):
    mod('sweep')
    from scaling_v1.runner import train
    result=train(replace(cfg,model_size='xlarge',steps=1),tmp_path/'xlarge')
    pred=torch.load(tmp_path/'xlarge/test_predictions.pt',weights_only=True)['predicted']
    torch.testing.assert_close(pred[...,0]+pred[...,1],pred[...,2])
    assert result['parameters']>1_107_648


def test_plot_exports_budget_and_compute_curves_with_replicates(cfg,tmp_path):
    root=tmp_path/'plots'
    for steps in (2,4):
        for seed,value in ((11,1.),(12,3.)):
            run=result_fixture(root,replace(cfg,steps=steps),seed,value)
            status=json.loads((run/'status.json').read_text())
            status['step']=steps
            (run/'status.json').write_text(json.dumps(status))
    mod('report').summarize(root,plot=True)
    assert list(root.glob('v2_steps_*.png'))
    assert list(root.glob('v2_compute_*.png'))
    assert all(p.stat().st_size>10000 for p in root.glob('v2_*.png'))


def test_plan_report_identifies_missing_seeds_and_ignores_foreign_runs(cfg,tmp_path):
    from scaling_v2 import VERSION
    root=tmp_path/'partial'
    a=result_fixture(root,cfg,11,1.)
    b=result_fixture(root,cfg,12,3.)
    result_fixture(root,cfg,99,900.)
    plan=dict(version=VERSION,data_identity='same-data',jobs=[
        dict(name=name,config=asdict(replace(cfg,train_seed=seed)))
        for name,seed in [(a.name,11),(b.name,12),('not_started',13)]])
    (root/'sweep_plan.json').write_text(json.dumps(plan))
    rows=mod('report').summarize(root)
    assert rows[0]['ite_rmse_mean']==2.
    assert rows[0]['seed_count']==2 and rows[0]['expected_seed_count']==3
    assert rows[0]['missing_seeds']=='[13]' and not rows[0]['replication_complete']
    progress=json.loads((root/'progress.json').read_text())
    assert progress['completed_runs']==2 and progress['pending_runs']==['not_started']


def test_resume_after_interruption_before_first_update_uses_initial_state(cfg,tmp_path,monkeypatch):
    sweep=mod('sweep')
    from scaling_v1.data import TaskBank
    jobs=sweep.build_jobs(cfg,task_counts=[4],model_sizes=['small'],step_budgets=[2],seeds=[11])
    output=tmp_path/'interrupted'
    def fail_batch(*args,**kwargs):
        raise RuntimeError('interrupted before first update')
    with monkeypatch.context() as patch:
        patch.setattr(TaskBank,'batch',fail_batch)
        with pytest.raises(RuntimeError,match='before first update'):
            sweep.run_sweep(jobs,output)
    assert (output/jobs[0].name/'model_initial.pt').exists()
    assert not (output/jobs[0].name/'model_last.pt').exists()
    sweep.run_sweep(jobs,output,resume=True)
    sweep.run_sweep(jobs,tmp_path/'full')
    a=torch.load(output/jobs[0].name/'model_last.pt',weights_only=True)
    b=torch.load(tmp_path/'full'/jobs[0].name/'model_last.pt',weights_only=True)
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)


@pytest.mark.parametrize('changed', ['threads','device','torch_version'])
def test_incomplete_resume_rejects_mixed_runtime_measurements(cfg,tmp_path,monkeypatch,changed):
    sweep=mod('sweep')
    from scaling_v1.data import TaskBank
    cfg=replace(cfg,steps=4)
    jobs=sweep.build_jobs(cfg,task_counts=[4],model_sizes=['small'],step_budgets=[4],seeds=[11])
    original_batch=TaskBank.batch
    calls=0
    def stop_after_checkpoint(self,split,ids):
        nonlocal calls
        if split=='train':
            calls+=1
            if calls==3:
                raise RuntimeError('stop after checkpoint')
        return original_batch(self,split,ids)
    output=tmp_path/'partial'
    with monkeypatch.context() as patch:
        patch.setattr(TaskBank,'batch',stop_after_checkpoint)
        with pytest.raises(RuntimeError,match='stop after checkpoint'):
            sweep.run_sweep(jobs,output)
    manifest_path=output/jobs[0].name/'manifest.json'
    original_manifest=manifest_path.read_text()
    if changed=='threads':
        retry=[replace(jobs[0],config=replace(jobs[0].config,threads=2))]
    else:
        manifest=json.loads(original_manifest)
        manifest[changed]='another-runtime'
        manifest_path.write_text(json.dumps(manifest))
        retry=jobs
    with pytest.raises(ValueError,match='runtime'):
        sweep.run_sweep(retry,output,resume=True)
    manifest_path.write_text(original_manifest)
    sweep.run_sweep(jobs,output,resume=True)
    assert json.loads((output/'progress.json').read_text())['completed_runs']==1
