"""Supplement the saved held-out test with the fixed simulation-only oracle.

Uses saved factual arrays; reconstructs the true basis with the known shared
projection. Reconstructed true coefficients are ONLY a truth audit, never fit
inputs. No graph regeneration and no J/baseline/calibration modification.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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

from audit_e12_evidence import task_summaries
from diagnose_e12_oracle import oracle_posterior
from evaluate import evaluate_context,summarize
from make_table import read_results,write_csv
from pfn_social.e12_configuration import FAMILY_BASE_IDS,configuration
from pfn_social.phase3_hierarchical import hidden_features,coefficient_prior
from pfn_social.phase5_large_network import common_projection,rng_for
from pfn_social.phase6_sparse_classical import pack
from run_e12_acceptance import point_metrics
from scripts.phase5_strong_controls import design_from_h


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task(job):
    family,mids,source,output=job;source=Path(source);output=Path(output)
    torch.set_num_threads(1)
    prior,cov=coefficient_prior();cfg=configuration(family)
    rows,points,receipts,checks=[],[],[],[]
    with threadpool_limits(limits=1):
        for mid in mids:
            with np.load(source/'truth'/f'f{family}_m{mid}.npz') as saved:truth=saved['arm_means']
            # Simulator-side audit only; excluded from oracle_posterior arguments.
            true_beta=rng_for(cfg,mid,'coefficient').multivariate_normal(prior,cov)
            for repeat in range(3):
                for budget in (20,50,100):
                    context=f'f{family}_m{mid}_r{repeat}_observed{budget}'
                    with np.load(source/'factual'/f'{context}.npz') as saved:
                        x,a,t,ids,y=[saved[k] for k in ('x','adjacency','treatment','observed_indices','outcomes_observed')]
                    h=hidden_features(x,common_projection(),cfg.setting.representation)
                    factual,query=design_from_h(h,a,t)
                    if repeat==0 and budget==20:
                        error=float(np.max(abs(query@true_beta-truth)))
                        assert error<1e-10
                        checks.append(dict(family=family,mechanism=mid,max_truth_reconstruction_error=error))
                    started=time.perf_counter()
                    oracle=oracle_posterior(factual[ids],query,y,prior,cov,cfg.setting.noise_sd**2)
                    pred=pack(oracle['arm_means'],n=len(x))
                    pred.update(ate_lower=oracle['ate_lower'],ate_upper=oracle['ate_upper'],
                        interval_level=np.array(.95),interval_kind=np.array('simulation_oracle_true_basis_prior_noise'),
                        ate_interval_supported=np.ones(3,dtype=bool))
                    elapsed=time.perf_counter()-started
                    np.savez_compressed(output/'predictions'/f'{context}__oracle-known-dgp.npz',
                        **{k:v for k,v in pred.items() if k!='diagnostics'})
                    receipt=dict(context_id=context,method='oracle-known-dgp',online_seconds=elapsed,
                        interval_seconds=0.,timing='Combined point and interval time in online_seconds',
                        arm_supported=pred['arm_supported'].tolist(),ite_supported=pred['ite_supported'].tolist(),
                        ate_supported=pred['ate_supported'].tolist(),ate_interval_supported=[True]*3,
                        failure=None,interval_failure=None,interval_level=.95)
                    receipts.append(receipt)
                    rows.extend(evaluate_context(truth,pred,receipt))
                    common=dict(family=family,mechanism=mid,repeat=repeat,m_observed=budget,context_id=context)
                    points.extend(point_metrics(pred,truth,dict(common,method='oracle-known-dgp')))
                    for method in ('j','ht','kernel-new'):
                        with np.load(source/'predictions'/f'{context}__{method}.npz') as saved:
                            original={key:saved[key] for key in saved.files}
                        points.extend(point_metrics(original,truth,dict(common,method=method)))
    return rows,points,receipts,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'predictions').mkdir();(args.output/'evaluation').mkdir()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    hashes={p:sha(args.input/p) for p in ('execution.json','evaluation/context_metrics.csv')}
    protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        method='oracle-known-dgp',simulation_only=True,families=list(FAMILY_BASE_IDS),
        mechanism_ids=list(range(3000,3200)),repeat_ids=[0,1,2],observed=[20,50,100],
        input_hashes=hashes,script_sha256=sha(__file__),
        oracle_function_sha256=sha(Path(__file__).parent/'diagnose_e12_oracle.py'),
        knows=['true shared hidden basis','coefficient prior','noise SD'],
        fits_only='observed factual design and observed outcomes',
        true_coefficients='truth-reconstruction audit only; never passed to fitting',
        no_J_or_calibration_changes=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows,points,receipts,checks=[],[],[],[]
    jobs=[(f,list(range(start,start+25)),str(args.input),str(args.output))
          for f in FAMILY_BASE_IDS for start in range(3000,3200,25)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(task,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            r,p,rc,c=future.result();rows.extend(r);points.extend(p);receipts.extend(rc);checks.extend(c)
            if i%4==0:print(f'Oracle comparison {i}/{len(jobs)} shards complete',flush=True)
    original_rows,_=read_results(args.input)
    for row in original_rows:row['width_sum']=float(row['width_sum'])
    combined=original_rows+rows
    manifest=json.loads((args.input/'execution.json').read_text())
    manifest.update(methods=manifest['methods']+['oracle-known-dgp'],receipts=manifest['receipts']+receipts,
        interval_methods=manifest['interval_methods']+['oracle-known-dgp'],
        evaluation_view_only=True,source_run=str(args.input),
        raw_prediction_locations='Original methods remain in source_run; only new oracle predictions are stored here.')
    (args.output/'execution.json').write_text(json.dumps(manifest,indent=2))
    write_csv(args.output/'oracle_context_metrics.csv',rows)
    write_csv(args.output/'point_context_metrics.csv',points)
    tasks,summary=task_summaries(points)
    write_csv(args.output/'point_task_metrics.csv',tasks);write_csv(args.output/'point_task_mean_sd.csv',summary)
    write_csv(args.output/'truth_checks.csv',checks)
    for name,data in [('context_metrics',combined),('summary',summarize(combined)),
                     ('summary_by_family',summarize(combined,('family',))),
                     ('summary_by_mechanism',summarize(combined,('family','mechanism')))]:
        write_csv(args.output/'evaluation'/(name+'.csv'),data)
    assert all(sha(args.input/p)==value for p,value in hashes.items())
    result=dict(oracle_contexts=len(receipts),truth_checked_mechanisms=len(checks),
        max_truth_reconstruction_error=max(c['max_truth_reconstruction_error'] for c in checks),
        original_sources_unchanged=True,point_metric_rows=len(points),
        evaluation_view_only=True,no_parameter_selection=True)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
