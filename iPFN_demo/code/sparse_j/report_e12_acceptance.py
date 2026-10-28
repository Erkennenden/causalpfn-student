"""Build the teacher-requirement evidence tables for local screening."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate import _cluster_se
from make_table import write_csv,EFFECTS
from run_e12_acceptance import CONDITIONS


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def summarize_intervals(rows,by_family):
    groups=defaultdict(list)
    keys=('condition','family','method','effect') if by_family else ('condition','method','effect')
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for identity,members in sorted(groups.items()):
        requested=sum(r['interval_requested'] for r in members)
        valid=sum(r['interval_valid'] for r in members);hits=sum(r['covered'] for r in members)
        result.append(dict(zip(keys,identity),contexts=len(members),
            mechanisms=len({(r['family'],r['mechanism']) for r in members}),
            interval_requested=requested,interval_valid=valid,covered=hits,
            coverage=hits/requested if requested else np.nan,
            coverage_cluster_se=_cluster_se(members,'interval_requested') if requested else np.nan,
            mean_width=sum(r['width_sum'] for r in members)/valid if valid else np.nan))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=read(args.input/'intervals.csv')
    for row in rows:
        for k in ('family','mechanism','interval_requested','interval_valid','covered'):row[k]=int(row[k])
        row['width_sum']=float(row['width_sum'])
    pooled=summarize_intervals(rows,False);families=summarize_intervals(rows,True)
    write_csv(args.output/'coverage_by_condition.csv',pooled)
    write_csv(args.output/'coverage_by_condition_family.csv',families)
    tasks=read(args.input/'task_metrics.csv');groups=defaultdict(list)
    for r in tasks:groups[r['condition'],r['method'],r['estimand'],r['effect']].append(r)
    summaries=[]
    for (condition,method,estimand,effect),members in sorted(groups.items()):
        good=[r for r in members if r['complete']=='True']
        rmse=np.array([float(r['rmse']) for r in good]);bias=np.array([float(r['signed_bias']) for r in good])
        summaries.append(dict(condition=condition,method=method,estimand=estimand,effect=effect,
            tasks_expected=len(members),tasks_complete=len(good),
            mean_task_rmse=float(rmse.mean()) if len(good) else np.nan,
            sd_task_rmse=float(rmse.std(ddof=1)) if len(good)>1 else np.nan,
            pooled_rmse=float(np.sqrt(np.mean(rmse**2))) if len(good) else np.nan,
            signed_bias=float(bias.mean()) if len(good) else np.nan,
            absolute_conditional_bias=float(np.mean(abs(bias))) if len(good) else np.nan,
            mean_within_task_error_variance=float(np.mean([float(r['error_variance']) for r in good])) if len(good) else np.nan))
    write_csv(args.output/'point_by_condition.csv',summaries)
    support=read(args.input/'support.csv');summaries=[]
    for condition in CONDITIONS:
        part=[r for r in support if r['condition']==condition]
        summaries.append(dict(condition=condition,contexts=len(part),
            actual_observed_min=min(int(r['m_observed']) for r in part),actual_observed_max=max(int(r['m_observed']) for r in part),
            zero_state11_contexts=sum(int(r['state11_observed'])==0 for r in part),
            state11_queries=sum(int(r['n']) for r in part),
            state11_blocked=sum(int(r['blocked_state11_queries']) for r in part),
            direct_unit_state_support_fraction=sum(int(r['direct_supported_queries']) for r in part)/sum(int(r['query_count']) for r in part),
            all_observed_queries_allowed=all(int(r['allowed_observed_queries'])==int(r['m_observed']) for r in part)))
    write_csv(args.output/'support_summary.csv',summaries)
    timing=read(args.input/'runtime.csv');groups=defaultdict(list)
    for r in timing:groups[r['condition'],r['method']].append(float(r['seconds']))
    write_csv(args.output/'runtime_summary.csv',[dict(condition=k[0],method=k[1],contexts=len(v),
        mean_ms=float(np.mean(v)*1000),median_ms=float(np.median(v)*1000),
        p95_ms=float(np.quantile(v,.95)*1000),
        note='J/oracle include interval computation; HT/kernel-new are point-only. Four parallel processes, one thread each.') for k,v in sorted(groups.items())])
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    labels=['Reference N200','N50','N100','Direct coeff x2','Spill coeff x2','Noise SD2','Drop observed 11']
    for ax,effect in zip(axes,EFFECTS):
        for method,offset,color,label in [('j',-.12,'#2563eb','Frozen J'),('oracle-known-dgp',.12,'#d97706','Known-DGP oracle')]:
            selected=[next(r for r in pooled if (r['condition'],r['method'],r['effect'])==(c,method,effect)) for c in CONDITIONS]
            cov=np.array([r['coverage'] for r in selected]);se=np.array([r['coverage_cluster_se'] for r in selected])
            ax.errorbar(cov,np.arange(7)+offset,xerr=1.96*se,fmt='o',capsize=3,color=color,label=label)
        ax.axvline(.9,color='gray',ls='--');ax.set(yticks=range(7),yticklabels=labels if effect=='direct' else [],
            title=effect.capitalize(),xlabel='ATE 95% interval empirical coverage',xlim=(.60,1.02))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Acceptance screening | 72 independent mechanisms per condition, 3 repeats\nBars: approximate 1.96 x mechanism-cluster SE; no OOD calibration guarantee')
    fig.savefig(args.output/'screening_coverage.png',dpi=180);fig.savefig(args.output/'screening_coverage.pdf')
    plt.close(fig)
    for name in ('protocol.json','verification.json','point_summary.csv','support.csv','graph_statistics.json'):
        shutil.copyfile(args.input/name,args.output/name)
    print(args.output,flush=True)


if __name__=='__main__':main()
