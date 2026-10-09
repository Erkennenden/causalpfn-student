"""Held-out ATE/ITE evaluation: five test seeds, two tasks per seed by default."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np
import torch
from .data import ROOT, TaskBank, write_json
from pfn_interference.dgp import SCENARIO, coefficient_metadata
from .training import load_model, to_device
from pfn_interference.cepo import ARM_NAMES, PREDICTION_PROTOCOL, effects_from_mu, predict_mu_distributions, mu_prediction_rows
from pfn_interference.model import resolve_device, load_causalpfn_checkpoint
from pfn_interference.baselines.localized_config import LOCALIZED_DEFAULTS

EFFECTS=('direct','spillover','total')
DEFAULT_TEST_SEEDS=(2025,2026,2027,2028,2029)


def resolve_test_seeds(test_seed, test_seeds, test_tasks):
    """Keep a single-seed replay option, and reject duplicate test datasets."""
    if type(test_tasks) is not int or test_tasks<1:
        raise ValueError('test_tasks must be a positive integer (tasks per seed)')
    if test_seed is not None and test_seeds is not None:
        raise ValueError('Use either test_seed or test_seeds, not both')
    seeds=list(test_seeds) if test_seeds is not None else (
        [test_seed] if test_seed is not None else list(DEFAULT_TEST_SEEDS))
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError('test_seeds must contain at least one integer seed')
    if len(set(seeds))!=len(seeds):
        raise ValueError('test_seeds must be unique; repeated seeds duplicate datasets')
    return seeds


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError(f'No rows for {path}')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def summarize_rows(rows,scenario,level):
    groups=defaultdict(list)
    for row in rows: groups[(row['method'],row['effect'])].append(row)
    out=[]
    for (method,effect),items in groups.items():
        supported=[r for r in items if r.get('supported',True) and math.isfinite(float(r['estimate']))]
        truth=np.array([r['truth'] for r in supported]);pred=np.array([r['estimate'] for r in supported]);err=pred-truth
        out.append(dict(scenario=scenario,level=level,method=method,effect=effect,
            mae=float(np.abs(err).mean()) if len(err) else float('nan'),
            rmse=float(np.sqrt(np.mean(err**2))) if len(err) else float('nan'),
            bias=float(err.mean()) if len(err) else float('nan'),
            mean_truth=float(truth.mean()) if len(err) else float('nan'),
            mean_pred=float(pred.mean()) if len(err) else float('nan'),
            n_supported=len(supported),n_total=len(items),
            n_datasets_supported=len({r['dataset_id'] for r in supported}),
            n_datasets_total=len({r['dataset_id'] for r in items}),
            mean_truth_all=float(np.mean([r['truth'] for r in items]))))
    return out


def pfn_rows(model,b):
    p=predict_mu_distributions(model,b)
    mu=(p['gmm_pi']*p['gmm_mu']).sum(-1).detach().cpu()
    pred=effects_from_mu(mu);truth=b['oracle_ite'].cpu();n=pred.shape[1]
    ite=[];ate=[];murows=mu_prediction_rows(p,b)
    for i in range(n):
        for j,name in enumerate(EFFECTS):
            ite.append(dict(dataset_id=1,unit_id=i+1,method='PFN with interference',effect=name,
                truth=float(truth[0,i,j]),estimate=float(pred[0,i,j]),supported=True))
    for j,name in enumerate(EFFECTS):
        ate.append(dict(dataset_id=1,method='PFN with interference',effect=name,
            truth=float(truth[0,:,j].double().mean()),estimate=float(pred[0,:,j].double().mean()),supported=True))
    return ite,ate,murows


def save_mu_outputs(target, rows):
    """Separate arm files, explicit component table and graph-level point means."""
    target=Path(target)
    write_csv(target/'pfn_mu_predictions.csv',rows)
    components=[];summary=[]
    graph_rows={}
    for arm in ARM_NAMES:
        selected=[r for r in rows if r['arm']==arm]
        write_csv(target/f'pfn_{arm}_predictions.csv',selected)
        truth=np.asarray([r['truth'] for r in selected],dtype=float)
        estimates=np.asarray([r['estimate'] for r in selected],dtype=float)
        error=estimates-truth
        summary.append(dict(arm=arm,mean_truth=float(truth.mean()),mean_pred=float(estimates.mean()),
            mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
            bias=float(error.mean()),n_nodes=len(selected),
            n_datasets=len({r['dataset_id'] for r in selected})))
        grouped=defaultdict(list)
        for row in selected:
            grouped[row['dataset_id']].append(row)
            provenance={key:row[key] for key in ('dataset_id','test_seed','task_id','unit_id','arm') if key in row}
            pi,mu,sigma=[json.loads(row[key]) for key in ('gmm_pi','gmm_mu','gmm_sigma')]
            for k,(weight,mean,sd) in enumerate(zip(pi,mu,sigma),1):
                components.append(dict(**provenance,component=k,weight=weight,mean=mean,std=sd))
        for dataset_id,items in grouped.items():
            graph_row=graph_rows.setdefault(dataset_id,{key:items[0][key]
                for key in ('dataset_id','test_seed','task_id','a','a_seed') if key in items[0]})
            graph_row[arm+'_truth']=float(np.mean([r['truth'] for r in items]))
            graph_row[arm+'_estimate']=float(np.mean([r['estimate'] for r in items]))
    write_csv(target/'pfn_mu_gmm_components.csv',components)
    write_csv(target/'pfn_mu_graph_means.csv',[graph_rows[k] for k in sorted(graph_rows)])
    write_csv(target/'pfn_mu_summary.csv',summary)
    return summary


def evaluate(checkpoint,output,*,test_tasks=2,test_seed=None,test_seeds=None,device='musa',baselines=True,
             baseline_epochs=500,tnet_epochs=160,hypersci_arm_samples=4096,cache_dir=None,
             localized_options=None):
    seeds=resolve_test_seeds(test_seed,test_seeds,test_tasks)
    if any(v<1 for v in (baseline_epochs,tnet_epochs,hypersci_arm_samples)):
        raise ValueError('Baseline budgets must be positive')
    localized_options = dict(localized_options or {})
    unknown = localized_options.keys() - LOCALIZED_DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown localized options: {sorted(unknown)}')
    localized_options = {**LOCALIZED_DEFAULTS, **localized_options}
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Evaluation output must be new or empty')
    actual_device=resolve_device(device)
    model,state=load_model(checkpoint,device=actual_device)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'fixed_graph.json',state['graph_record'])
    cfg=state['config']
    bank=TaskBank(output/'fixed_graph.json',cache_dir or cfg['cache_dir'],
        data_seed=cfg['data_seed'],noise_sd=cfg['noise_sd'])
    if bank.fingerprint!=state['data_identity']: raise ValueError('DGP/graph identity differs from checkpoint')
    test_banks={seed:TaskBank(output/'fixed_graph.json',cache_dir or cfg['cache_dir'],
        data_seed=seed,noise_sd=cfg['noise_sd']) for seed in seeds}
    schedule=[]
    for seed in seeds:
        for task_id in range(test_tasks):
            schedule.append(dict(dataset_id=len(schedule)+1,test_seed=seed,task_id=task_id,
                baseline_seed=seed+task_id,test_data_identity=test_banks[seed].fingerprint,
                graph_sha256=bank.record['adjacency_sha256'],
                **coefficient_metadata(seed=seed,stream='scaling_v1/test',task_id=task_id)))
    write_csv(output/'test_datasets.csv',schedule)
    causal_model=None
    if baselines:
        from pfn_interference.baselines.validation import validate_baseline_results
        from pfn_interference.metrics.unified import evaluate_unified_benchmark,save_unified_benchmark
        causal_model=load_causalpfn_checkpoint(ROOT/'checkpoints/causalpfn_v0.pt',device=torch.device('cpu'))
    scenarios=[SCENARIO]
    summary=[];scale_rows=[];mu_summary=[]
    for scenario in scenarios:
        target=output/scenario;target.mkdir()
        all_ite=[];all_ate=[];all_mu=[];per_dataset=[]
        count=len(schedule)
        for task in schedule:
            seed=task['test_seed'];task_id=task['task_id'];dataset_id=task['dataset_id']
            provenance={key:task[key] for key in ('dataset_id','test_seed','task_id','a','a_seed')}
            b=test_banks[seed].batch('test',[task_id])
            scale_rows.extend({**r,**provenance} for r in task_scales(b,scenario,task_id))
            task_output=target/f'task_{dataset_id-1:04d}'
            write_json(task_output/'test_task.json',task)
            b=to_device(b,actual_device)
            if baselines:
                report=evaluate_unified_benchmark(model=model,batch=b,treatment_prob=.5,
                    ate_bandwidth=2,ate_ridge=0.0,**localized_options,seed=task['baseline_seed'],
                    standard_ite_epochs=baseline_epochs,tnet_epochs=tnet_epochs,
                    hypersci_arm_samples=hypersci_arm_samples,causalpfn_model=causal_model)
                validate_baseline_results(report,n_units=bank.graph.shape[0])
                report['test_task']=task
                save_unified_benchmark(report,task_output)
                for message in report['localized_diagnostics']['localization_warnings']:
                    print('Localized diagnostic:',message,flush=True)
                ite,ate,mu=report['ite_unit_results'],report['ate_graph_results'],report['pfn_mu_predictions']
            else: ite,ate,mu=pfn_rows(model,b)
            ite=[{**r,**provenance} for r in ite]
            ate=[{**r,**provenance} for r in ate]
            mu=[{**r,**provenance} for r in mu]
            save_mu_outputs(task_output,mu)
            all_ite.extend(ite);all_ate.extend(ate);all_mu.extend(mu)
            per_dataset.extend({**r,**provenance} for r in
                summarize_rows(ite,scenario,'ITE')+summarize_rows(ate,scenario,'ATE'))
            print(f'test {scenario}: {dataset_id}/{count}; seed={seed} task_id={task_id}; all nodes complete',flush=True)
        for name,rows in (('ite_unit_results',all_ite),('ate_graph_results',all_ate)):
            write_csv(target/f'{name}.csv',rows)
        mu_summary.extend(dict(scenario=scenario,**r) for r in save_mu_outputs(target,all_mu))
        local=summarize_rows(all_ite,scenario,'ITE')+summarize_rows(all_ate,scenario,'ATE')
        write_csv(target/'per_dataset_summary.csv',per_dataset)
        write_csv(target/'summary.csv',local);summary.extend(local)
    write_csv(output/'all_results.csv',summary)
    write_csv(output/'test_effect_scales.csv',scale_rows)
    lines=['scenario | Level | Method | Effect | MAE | RMSE | Bias | Mean Truth | Mean Pred | support']
    for r in summary:
        lines.append(' | '.join([r['scenario'],r['level'],r['method'],r['effect'],
            *[f'{r[k]:.6f}' for k in ('mae','rmse','bias','mean_truth','mean_pred')],f"{r['n_supported']}/{r['n_total']}"]))
    lines.extend(['','Four CEPO GMM means (one marginal GMM per node/arm):',
                  'Arm | Mean Truth | Mean Pred | MAE | RMSE | Bias | Nodes'])
    for r in mu_summary:
        lines.append(' | '.join([r['arm'],*[f'{r[k]:.6f}' for k in
            ('mean_truth','mean_pred','mae','rmse','bias')],str(r['n_nodes'])]))
    (output/'all_results.txt').write_text('\n'.join(lines)+'\n')
    result=dict(status='complete',scenarios=scenarios,test_tasks=test_tasks,
        tasks_per_seed=test_tasks,test_seeds=seeds,test_seed=seeds[0] if len(seeds)==1 else None,
        n_test_datasets=len(schedule),n_distinct_graphs=1,test_datasets=schedule,
        baseline_comparison=baselines,checkpoint_epoch=state['epoch'],
        checkpoint=str(Path(checkpoint).resolve()), model_config=state['model_config'],
        a_distribution=bank.identity['a_distribution'],
        localized_options=localized_options if baselines else None,
        graph_sha256=bank.record['adjacency_sha256'],data_identity=bank.fingerprint,
        test_data_identity=test_banks[seeds[0]].fingerprint if len(seeds)==1 else None,
        test_data_identities={str(seed):test_banks[seed].fingerprint for seed in seeds},
        prediction_protocol=PREDICTION_PROTOCOL,cepo_order=list(ARM_NAMES),
        gmm_n_components=model.config.gmm_n_components,mu_summary=mu_summary,
        interpretation='linear_b_random_a: independent a~Uniform(4,8), X/T/noise datasets on one saved graph; not distinct topologies',
        aggregation=dict(ATE='pooled dataset-level errors over supported test datasets',
            ITE='pooled unit-level errors over supported nodes in all test datasets'),
        summary=summary)
    # Unsupported estimators use explicit null in JSON and blank/NaN in CSV.
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v): return None
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list): return [clean(x) for x in v]
        return v
    write_json(output/'results.json',clean(result))
    return result


def task_scales(batch,scenario,task_id):
    truth=batch['oracle_ite'].double().cpu().numpy()[0]
    mu=batch['cepo_target'].double().cpu().numpy()[0]
    rows=[]
    for j,effect in enumerate(EFFECTS):
        rows.append(dict(scenario=scenario,task_id=task_id,effect=effect,
            mean=float(truth[:,j].mean()),rms=float(np.sqrt(np.mean(truth[:,j]**2))),
            sd=float(truth[:,j].std()),cepo_min=float(mu.min()),cepo_max=float(mu.max())))
    return rows


def diagnose_prior(graph_file,cache_dir,output,*,tasks=128,data_seed=12345,noise_sd=4.):
    if tasks<1: raise ValueError('diagnostic tasks must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    bank=TaskBank(graph_file,cache_dir,data_seed=data_seed,noise_sd=noise_sd)
    rows=[];coefficients=[]
    for task_id in range(tasks):
        coefficients.append(dict(task_id=task_id,**coefficient_metadata(seed=data_seed,stream='scaling_v1/train',task_id=task_id)))
        rows.extend(task_scales(bank.batch('train',[task_id]),'training_prior',task_id))
    write_csv(output/'prior_effect_scales.csv',rows)
    write_csv(output/'prior_coefficients.csv',coefficients)
    stats={}
    for effect in EFFECTS:
        group=[r for r in rows if r['effect']==effect]
        stats[effect]={key:dict(zip(('min','q10','median','q90','max'),map(float,np.quantile([r[key] for r in group],[0,.1,.5,.9,1])))) for key in ('mean','rms','sd')}
    write_json(output/'prior_diagnostics.json',dict(tasks=tasks,noise_sd=noise_sd,
        graph_sha256=bank.record['adjacency_sha256'],effects=stats,
        a_distribution=bank.identity['a_distribution'],
        note='B DGP with one a~Uniform(4,8) per dataset; four exact CEPO labels exclude observation noise.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,effect,color in zip(axs,EFFECTS,('#2459a6','#e78524','#40946a')):
        ax.hist([r['rms'] for r in rows if r['effect']==effect],bins=min(20,tasks),color=color,alpha=.8)
        ax.set(title=effect.capitalize(),xlabel='True individual-effect RMS',ylabel='Tasks')
    fig.suptitle(f'Linear B with random a | {tasks} training tasks | noise SD={noise_sd:g}')
    fig.savefig(output/'prior_effect_scales.png',dpi=160);plt.close(fig)
    return stats
