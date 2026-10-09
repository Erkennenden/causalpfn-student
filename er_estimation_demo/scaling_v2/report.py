"""Per-run metrics and sample standard deviations across independent training seeds."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scaling_v1 import VERSION as TRAINING_VERSION
from scaling_v1.data import write_json
from . import VERSION

EFFECTS=('direct','spillover','total')
METRICS=('ite_rmse','ite_mae','ate_rmse','ate_bias','zero_predictor_ite_rmse',
    'best_validation_nll','evaluated_step','unique_train_tasks_seen','task_presentations',
    'evaluated_unique_train_tasks_seen','evaluated_task_presentations',
    'observed_task_coverage','evaluated_task_coverage','training_seconds','data_seconds',
    'validation_seconds','test_seconds','compute_seconds')


def collect_runs(directory):
    records=[]
    for path in sorted(Path(directory).rglob('results.json')):
        result=json.loads(path.read_text())
        if result.get('version')!=TRAINING_VERSION or result.get('status')!='complete':
            continue
        if not all((path.parent/name).exists() for name in ('status.json','manifest.json')):
            continue
        status=json.loads((path.parent/'status.json').read_text())
        manifest=json.loads((path.parent/'manifest.json').read_text())
        if (status.get('status')=='complete' and status.get('step')==result['config']['steps']
                and status.get('evaluated_step')==result.get('evaluated_step')
                and manifest.get('config')==result.get('config')
                and manifest.get('data_identity')==result.get('data_identity')
                and manifest.get('version')==result['version']):
            records.append((path.parent,result,manifest))
    return records


def experiment_key(config):
    ignored={'graph_file','cache_dir','device','threads','train_seed'}
    return json.dumps({k:v for k,v in config.items() if k not in ignored},sort_keys=True)


def condition_key(result,manifest):
    return json.dumps(dict(experiment=json.loads(experiment_key(result['config'])),
        data_identity=result['data_identity'],device=result['device'],threads=result['config']['threads'],
        model_config=manifest['model_config'],torch_version=manifest.get('torch_version'),
        training_version=result['version']),sort_keys=True)


def write_csv(path,rows,empty_fields):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory,*,plot=False):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    records=collect_runs(directory)
    plan_path=directory/'sweep_plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if plan and plan.get('version')!=VERSION:
        raise ValueError('Not a V2 sweep plan; report a single run or use its original report command.')
    if plan:
        from .sweep import config_identity
        planned_configs={j['name']:j['config'] for j in plan['jobs']}
        records=[(p,r,m) for p,r,m in records if str(p.relative_to(directory)) in planned_configs
            and config_identity(r['config'])==config_identity(planned_configs[str(p.relative_to(directory))])
            and r['data_identity']==plan.get('data_identity')]
    expected={}
    for job in plan.get('jobs',[]):
        expected.setdefault(experiment_key(job['config']),set()).add(job['config']['train_seed'])
    raw,groups=[],{}
    for path,result,manifest in records:
        cfg=result['config']
        key=condition_key(result,manifest)
        bucket=groups.setdefault(key,dict(records=[],seeds=set()))
        seed=cfg['train_seed']
        if seed in bucket['seeds']:
            raise ValueError(f'Duplicate training seed {seed} in equivalent runs; report a single experiment directory.')
        bucket['seeds'].add(seed)
        bucket['records'].append(result)
        selected_presentations=result['evaluated_step']*cfg['batch_size']
        selected_seen=min(selected_presentations,cfg['train_tasks'])
        condition=hashlib.sha256(key.encode()).hexdigest()[:12]
        counts=dict(evaluated_unique_train_tasks_seen=selected_seen,
            evaluated_task_presentations=selected_presentations,
            observed_task_coverage=result['unique_train_tasks_seen']/cfg['train_tasks'],
            evaluated_task_coverage=selected_seen/cfg['train_tasks'])
        if {e['effect'] for e in result['effects']}!=set(EFFECTS):
            raise ValueError(f'Missing or unknown effects: {path}')
        for effect in result['effects']:
            row=dict(condition_id=condition,run=str(path.relative_to(directory)),
                **{k:v for k,v in cfg.items() if k not in {'graph_file','cache_dir','device'}},
                parameters=result['parameters'],device=result['device'],data_identity=result['data_identity'],
                model_config=json.dumps(manifest['model_config'],sort_keys=True),
                torch_version=manifest.get('torch_version'),evaluated_step=result['evaluated_step'],
                best_validation_nll=result['best_validation_nll'],
                unique_train_tasks_seen=result['unique_train_tasks_seen'],
                task_presentations=result['task_presentations'],**counts,**result['timing'],
                test_seconds=result['test_seconds'],
                compute_seconds=sum(result['timing'].values())+result['test_seconds'],**effect)
            if not all(math.isfinite(row[k]) for k in METRICS):
                raise ValueError(f'Nonfinite metrics: {path}')
            raw.append(row)
    aggregates=[]
    for key,bucket in groups.items():
        sample=bucket['records'][0]
        condition=hashlib.sha256(key.encode()).hexdigest()[:12]
        expected_seeds=expected.get(experiment_key(sample['config']))
        seeds=sorted(bucket['seeds'])
        missing=sorted(expected_seeds-set(seeds)) if expected_seeds is not None else []
        for effect in EFFECTS:
            members=[r for r in raw if r['condition_id']==condition and r['effect']==effect]
            row={k:v for k,v in members[0].items() if k not in set(METRICS)|{'run','train_seed','effect'}}
            row.update(effect=effect,seed_count=len(seeds),seeds=json.dumps(seeds),
                expected_seed_count=len(expected_seeds) if expected_seeds is not None else None,
                missing_seeds=json.dumps(missing),replication_complete=not missing if expected_seeds is not None else None)
            for metric in METRICS:
                values=[m[metric] for m in members]
                row[metric+'_mean']=statistics.mean(values)
                row[metric+'_std']=statistics.stdev(values) if len(values)>1 else None
            aggregates.append(row)
    planned=[j['name'] for j in plan.get('jobs',[])]
    completed=[str(p.relative_to(directory)) for p,_,_ in records]
    write_csv(directory/'summary.csv',raw,['condition_id','run','effect','ite_rmse'])
    write_csv(directory/'aggregate_summary.csv',aggregates,['condition_id','effect','seed_count','ite_rmse_mean','ite_rmse_std'])
    write_json(directory/'progress.json',dict(version=VERSION,planned_runs=len(planned) if plan else None,
        completed_runs=len(records),pending_runs=[n for n in planned if n not in completed],
        uncertainty='sample SD across training seeds; fixed shared task bank and test set',
        timing='training + data + validation + test seconds; excludes checkpoint/report I/O'))
    if plot:
        from .plots import plot_summary
        plot_summary(directory,aggregates)
    print(f'V2 summary: {len(records)} completed runs, {len(aggregates)} effect rows -> {directory}',flush=True)
    return aggregates
