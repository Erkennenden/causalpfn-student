"""Registered E1.2 size/mechanism/support screening with frozen predictors.

All changes are local diagnostics relative to the N200 reference distribution.
The checkpoint has no complete training-distribution manifest: no claim that
each condition is outside its training support. No calibration/refitting here.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark import configuration as frozen_configuration,load_methods,predict_method
from diagnose_e12_oracle import oracle_posterior
from e12_acceptance_contract import support_diagnostics,estimate_for_request,ordinary_plugin_values
from evaluate import evaluate_context
from make_table import write_csv,EFFECTS
from pfn_social.e12_configuration import FAMILY_BASE_IDS,configuration,generate_mechanism
from pfn_social.configuration_graph import degree_statistics
from pfn_social.phase3_hierarchical import coefficient_prior,_components,structural_outcome,_hash_arrays
from pfn_social.phase5_large_network import generate_mechanism as parent_mechanism,rng_for,make_repeat
from pfn_social.phase6_limited_outcomes import mask_rng
from pfn_social.intervals import predict_intervals
from pfn_social.phase6_sparse_classical import pack
from scripts.phase5_strong_controls import design_from_h

ROOT=Path(__file__).resolve().parent
TARGET_ID='local-e12-proposal:finite-network-majority-p0.5-mu-v1'
METHODS=('j','ht','kernel-new','oracle-known-dgp')
CONDITIONS=('reference_n200','size_n50','size_n100','direct_coeff_x2',
            'spill_main_coeff_x2','noise_sd2','drop_observed_11')
_MODELS=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models():
    global _MODELS
    if _MODELS is None:
        _MODELS=load_methods(['j','ht','kernel-new'],frozen_configuration())
    return _MODELS


def specification():
    return dict(name='e12_acceptance_screening_v1',families=list(FAMILY_BASE_IDS),
        mechanism_ids=list(range(4000,4012)),repeats=[0,1,2],requested_observed=50,
        conditions=list(CONDITIONS),nodes=[50,100,200],nominal_level=.95,
        target_id=TARGET_ID,target_status='POINT is a local simulation request, not an upstream identification decision',
        perturbations=dict(direct_coeff_x2='beta[7:11]*2; main direct coefficients only',
            spill_main_coeff_x2='beta[11:15] and beta[19:22] *2; main exposure/neighbor coefficients',
            noise_sd2='noise SD 2 instead of 1; same standard-noise RNG',
            drop_observed_11='Remove state11 from original 50 outcome rows; no replacement; treatment and target unchanged'),
        pairing='At N200, variants reuse graph, X, T, mask and standard noise for each mechanism/repeat.',
        methods=list(METHODS),oracle='True h, correct condition-specific coefficient prior and noise; fits observed Y only.',
        support='Direct unit-state observations are a conservative diagnostic, not a certified universal empirical_support definition.',
        ood='Shifts relative to E1.2 reference; original training-support membership unknown.',
        limitation='12 independent mechanisms per family/condition; screening, not precise population coverage certification.',
        source_sha256={p:sha(ROOT/p) for p in ['run_e12_acceptance.py','e12_acceptance_contract.py',
            'diagnose_e12_oracle.py','pfn_social/phase3_hierarchical.py','pfn_social/phase5_large_network.py',
            'pfn_social/configuration_graph.py','pfn_social/intervals.py']},
        frozen_assets=frozen_configuration()['frozen_assets_sha256'])


def point_metrics(prediction,truth,common):
    ite=np.stack((truth[:,2]-truth[:,0],truth[:,3]-truth[:,2],truth[:,3]-truth[:,0]),1)
    result=[]
    for estimand,key,target,labels,support in (
        ('mu','arm_means',truth,('00','01','10','11'),'arm_supported'),
        ('ITE','ite_effects',ite,EFFECTS,'ite_supported'),
        ('ATE','ate_effects',ite.mean(0)[None],EFFECTS,'ate_supported')):
        values=np.asarray(prediction[key]).reshape(target.shape)
        for i,label in enumerate(labels):
            valid=bool(prediction[support][i]) and np.isfinite(values[:,i]).all()
            error=values[:,i]-target[:,i]
            result.append(dict(**common,estimand=estimand,effect=label,valid=valid,
                mse=float(np.mean(error**2)) if valid else np.nan,
                bias=float(np.mean(error)) if valid else np.nan))
    return result


def summarize_points(rows):
    groups=defaultdict(list)
    keys=('condition','family','mechanism','method','estimand','effect')
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    tasks=[]
    for identity,part in sorted(groups.items()):
        good=all(r['valid'] for r in part)
        tasks.append(dict(zip(keys,identity),repeats=len(part),complete=good,
            rmse=float(np.sqrt(np.mean([r['mse'] for r in part]))) if good else np.nan,
            signed_bias=float(np.mean([r['bias'] for r in part])) if good else np.nan,
            error_variance=float(np.var([r['bias'] for r in part],ddof=1)) if good and len(part)>1 else np.nan))
    groups=defaultdict(list);keys=('condition','family','method','estimand','effect')
    for row in tasks:groups[tuple(row[k] for k in keys)].append(row)
    summary=[]
    for identity,part in sorted(groups.items()):
        good=[r for r in part if r['complete']]
        rmse=np.array([r['rmse'] for r in good]);bias=np.array([r['signed_bias'] for r in good])
        summary.append(dict(zip(keys,identity),tasks_expected=len(part),tasks_complete=len(good),
            mean_task_rmse=float(rmse.mean()) if len(good) else np.nan,
            sd_task_rmse=float(rmse.std(ddof=1)) if len(good)>1 else np.nan,
            pooled_rmse=float(np.sqrt(np.mean(rmse**2))) if len(good) else np.nan,
            signed_bias=float(bias.mean()) if len(good) else np.nan,
            absolute_conditional_bias=float(np.mean(abs(bias))) if len(good) else np.nan,
            mean_within_task_error_variance=float(np.mean([r['error_variance'] for r in good])) if len(good) else np.nan))
    return tasks,summary


def task(job):
    family,mid,destination=job;destination=Path(destination)/f'f{family}_m{mid}'
    destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    points,intervals,supports,graphs,times=[],[],[],[],[]
    mean,cov=coefficient_prior()
    with threadpool_limits(limits=1):
        estimators=models()
        for n in (200,50,100):
            cfg=replace(configuration(family),n=n)
            mechanism=parent_mechanism(cfg,mid)
            a=mechanism.adjacency;h=mechanism.h;x=mechanism.x
            graphstat=degree_statistics(a)
            graphs.append(dict(family=family,mechanism=mid,config=json.loads(cfg.canonical_json()),
                               mechanism_sha256=mechanism.mechanism_sha256,**graphstat))
            ids=mask_rng(family,mid).permutation(n)[:50]
            variants=[c for c in CONDITIONS if (n==200 and not c.startswith('size_')) or c==f'size_n{n}']
            for repeat in range(3):
                raw=make_repeat(mechanism,repeat)
                treatment=raw['observed_treatment'].numpy()
                standard_noise=rng_for(cfg,mid,'noise',repeat).normal(size=n)
                factual,query=design_from_h(h,a,treatment)
                np.testing.assert_allclose(query@mechanism.coefficients,mechanism.truth,atol=1e-10)
                for condition in variants:
                    folder=destination/condition;folder.mkdir(exist_ok=True)
                    scale=np.ones(25);noise_sd=1.
                    if condition=='direct_coeff_x2':scale[7:11]=2
                    if condition=='spill_main_coeff_x2':scale[11:15]=2;scale[19:22]=2
                    if condition=='noise_sd2':noise_sd=2.
                    beta=mechanism.coefficients*scale
                    changed=replace(mechanism,coefficients=beta,components=_components(a,h,beta))
                    truth=query@beta
                    # Shared original DGP, with explicitly registered coefficients/noise.
                    y=(structural_outcome(changed,treatment)+noise_sd*standard_noise).astype(np.float32)
                    if condition in ('reference_n200','size_n50','size_n100'):
                        np.testing.assert_array_equal(y,raw['y_obs'].numpy())
                    selected=ids.copy()
                    if condition=='drop_observed_11':
                        states=support_diagnostics(a,treatment,selected)['state_indices']
                        selected=selected[states[selected]!=3]
                    if not len(selected):raise ValueError('No observed outcomes after support deletion')
                    inputs=dict(x=x,adjacency=a.astype(np.float32),treatment=treatment,
                                observed_indices=selected,outcomes_observed=y[selected].copy())
                    context=f'f{family}_m{mid}_r{repeat}_observed{len(selected)}'
                    np.savez_compressed(folder/f'{context}__factual.npz',**inputs)
                    np.savez_compressed(folder/f'{context}__truth.npz',arm_means=truth)
                    common=dict(condition=condition,family=family,mechanism=mid,repeat=repeat,
                        n=n,m_observed=len(selected),requested_observed=50,context_id=context)
                    sd=support_diagnostics(a,treatment,selected)
                    support=dict(**common,direct_supported_queries=int(sd['direct_observation_mask'].sum()),
                        query_count=4*n,globally_empty_arms=int(np.sum(sd['observed_arm_counts']==0)),
                        state11_observed=int(sd['observed_arm_counts'][1,1]),
                        population_positive_all_states=True,target_unchanged=True)
                    for method in METHODS:
                        began=time.perf_counter()
                        if method=='oracle-known-dgp':
                            oracle=oracle_posterior(factual[selected],query,y[selected],mean*scale,
                                                    cov*scale[:,None]*scale[None,:],noise_sd**2)
                            pred=pack(oracle['arm_means'],n=n)
                            interval=dict(ate_lower=oracle['ate_lower'],ate_upper=oracle['ate_upper'],
                                interval_level=np.array(.95),interval_kind=np.array('simulation_oracle_known_condition'))
                        else:
                            pred=predict_method(method,estimators[method],inputs,mid*100+repeat)
                            interval=dict(ate_lower=np.full(3,np.nan),ate_upper=np.full(3,np.nan),
                                interval_level=np.array(.95),interval_kind=np.array('unsupported'))
                            if method=='j':interval=predict_intervals(estimators[method],**inputs,level=.95)
                        elapsed=time.perf_counter()-began
                        times.append(dict(**common,method=method,seconds=elapsed))
                        pred.update(interval)
                        np.savez_compressed(folder/f'{context}__{method}.npz',
                            **{k:v for k,v in pred.items() if k!='diagnostics'})
                        points.extend(point_metrics(pred,truth,dict(**common,method=method)))
                        receipt=dict(context_id=context,method=method,ate_supported=pred['ate_supported'],
                            ate_interval_supported=[method in ('j','oracle-known-dgp')]*3,interval_level=.95)
                        intervals.extend(dict(r,condition=condition,n=n,requested_observed=50)
                                         for r in evaluate_context(truth,pred,receipt))
                        if method=='j':
                            bundle=estimate_for_request(lambda **kw:pred,inputs,target_id=TARGET_ID,status='POINT',
                                node_ids=np.arange(n),empirical_support_mask=sd['direct_observation_mask'],
                                population_support_mask=np.ones((n,2,2),dtype=bool))
                            blocked=0
                            for node in range(n):
                                try:ordinary_plugin_values(bundle,[(node,1,1)])
                                except ValueError:blocked+=1
                            support['blocked_state11_queries']=blocked
                            support['allowed_observed_queries']=len(ordinary_plugin_values(bundle,
                                [(int(i),int(treatment[i]),int(sd['state_indices'][i]%2)) for i in selected]))
                            np.savez_compressed(folder/f'{context}__support.npz',node_ids=np.arange(n),
                                direct_observation_counts=sd['direct_observation_counts'],
                                direct_observation_mask=sd['direct_observation_mask'],
                                population_support_mask=np.ones((n,2,2),dtype=bool),
                                target_id=np.array(TARGET_ID),status=np.array('POINT'))
                    if condition=='drop_observed_11':
                        assert support['state11_observed']==0 and support['blocked_state11_queries']==n
                    supports.append(support)
    for name,rows in [('point_contexts',points),('interval_contexts',intervals),('support',supports),('runtime',times)]:
        write_csv(destination/(name+'.csv'),rows)
    (destination/'graphs.json').write_text(json.dumps(graphs,indent=2))
    return dict(points=points,intervals=intervals,support=supports,runtime=times,graphs=graphs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    args.output.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    spec=specification()
    (args.output/'protocol.json').write_text(json.dumps(dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                                                          specification=spec),indent=2))
    combined={k:[] for k in ('points','intervals','support','runtime','graphs')}
    jobs=[(f,m,str(args.output/'shards')) for f in FAMILY_BASE_IDS for m in range(4000,4012)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(task,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            for key in combined:combined[key].extend(result[key])
            if i%6==0:print(f'Acceptance screening {i}/{len(jobs)} mechanisms complete',flush=True)
    for key in ('points','intervals','support','runtime'):write_csv(args.output/(key+'.csv'),combined[key])
    tasks,summary=summarize_points(combined['points'])
    write_csv(args.output/'task_metrics.csv',tasks);write_csv(args.output/'point_summary.csv',summary)
    (args.output/'graph_statistics.json').write_text(json.dumps(combined['graphs'],indent=2))
    assert spec==specification()
    result=dict(contexts=len(combined['support']),method_contexts=len(combined['runtime']),
        graphs=len(combined['graphs']),independent_mechanisms_per_family_condition=12,
        supported_observed_queries_passed=True,deleted_state11_queries_all_blocked=True,
        source_and_frozen_assets_unchanged=True,oracle_truth_check=True,
        target_id_local_proposal=True,upstream_interface_integrated=False)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
