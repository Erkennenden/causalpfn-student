"""Paired cluster bootstrap, coverage uncertainty, and effect scale diagnostics."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from make_table import read_results,write_csv,ate_table
from pfn_social.contracts import arm_mean_exposure

EFFECTS=('direct','spillover','total')

def bootstrap_difference(strata,root=True,replicates=10000,seed=2026102734):
    """Whole-cluster rows: sum_A,count_A,sum_B,count_B; same weights for A/B."""
    rng=np.random.default_rng(seed); total=np.zeros(4); sampled=np.zeros((replicates,4))
    for block in strata:
        block=np.asarray(block,float)
        if block.ndim!=2 or block.shape[1]!=4 or not len(block):raise ValueError('Four-column clusters required')
        total+=block.sum(0)
        weights=rng.multinomial(len(block),np.full(len(block),1/len(block)),size=replicates)
        sampled+=weights@block
    if min(total[1],total[3])<=0:
        return dict(difference=np.nan,lower=np.nan,upper=np.nan,valid_bootstraps=0)
    valid=(sampled[:,1]>0)&(sampled[:,3]>0)
    f=np.sqrt if root else lambda x:x
    values=f(sampled[valid,0]/sampled[valid,1])-f(sampled[valid,2]/sampled[valid,3])
    lo,hi=np.quantile(values,[.025,.975]) if len(values) else [np.nan,np.nan]
    return dict(difference=float(f(total[0]/total[1])-f(total[2]/total[3])),lower=float(lo),upper=float(hi),valid_bootstraps=int(valid.sum()))

def baseline_comparisons(rows,run):
    lookup={(r['context_id'],r['effect'],r['method']):r for r in rows}
    groups=defaultdict(list)
    for r in rows:
        if r['method']=='j':groups[r['m_observed'],r['effect']].append(r)
    methods=sorted({r['method'] for r in rows}-{'j'}); out=[]
    for (budget,effect),group in sorted(groups.items()):
        families=sorted({r['family'] for r in group})
        for family in (['all']+families if run!='matched' else families):
            js=[r for r in group if family=='all' or r['family']==family]
            for method in methods:
                clusters=defaultdict(lambda:np.zeros(4))
                for j in js:
                    key=(j['family'],j['mechanism']);clusters[key]
                    b=lookup[j['context_id'],effect,method]
                    if j['point_valid']==b['point_valid']==1:
                        clusters[key]+=[j['squared_error_sum'],1,b['squared_error_sum'],1]
                strata=defaultdict(list)
                for (fam,mid),value in sorted(clusters.items()):strata[fam].append(value)
                totals=np.sum(list(clusters.values()),0);n=totals[1]
                out.append(dict(run=run,family=family,m_observed=budget,effect=effect,baseline=method,
                    shared_contexts=int(n),expected_contexts=len(js),independent_clusters=len(clusters),
                    j_rmse=float(np.sqrt(totals[0]/n)) if n else np.nan,
                    baseline_rmse=float(np.sqrt(totals[2]/n)) if n else np.nan,
                    **bootstrap_difference(list(strata.values()))))
    return out

def coverage_summary(rows,run):
    groups=defaultdict(list);out=[]
    for r in rows:
        if r['method']=='j':groups[r['family'],r['m_observed'],r['effect']].append(r)
    for (family,budget,effect),group in sorted(groups.items()):
        clusters=defaultdict(lambda:np.zeros(4))
        for r in group:clusters[r['mechanism']]+=[r['covered'],r['interval_requested'],0,r['interval_requested']]
        stat=bootstrap_difference([list(clusters.values())],root=False)
        valid=sum(r['interval_valid'] for r in group)
        out.append(dict(run=run,family=family,m_observed=budget,effect=effect,mechanisms=len(clusters),
            covered=sum(r['covered'] for r in group),requested=sum(r['interval_requested'] for r in group),
            interval_valid=valid,coverage=stat['difference'],lower=stat['lower'],upper=stat['upper'],
            meets_90=stat['difference']>=.9,mean_width=sum(float(r['width_sum']) for r in group)/valid if valid else np.nan))
    return out

def spatial_comparisons(rows):
    lookup={(r['family'],r['mechanism'],r['repeat'],r['m_observed'],r['effect'],r['method']):r for r in rows}
    groups=defaultdict(list);out=[]
    for r in rows:
        if r['family']==13:groups[r['m_observed'],r['effect'],r['method']].append(r)
    for (budget,effect,method),group in sorted(groups.items()):
        clusters=defaultdict(lambda:np.zeros(4))
        for u in group:
            c=lookup[14,u['mechanism'],u['repeat'],budget,effect,method];clusters[u['mechanism']]
            if u['point_valid']==c['point_valid']==1:clusters[u['mechanism']]+=[c['squared_error_sum'],1,u['squared_error_sum'],1]
        totals=np.sum(list(clusters.values()),0);n=totals[1]
        out.append(dict(m_observed=budget,effect=effect,method=method,independent_pairs=len(clusters),
            shared_context_pairs=int(n),expected_context_pairs=len(group),
            clustered_rmse=float(np.sqrt(totals[0]/n)) if n else np.nan,
            uniform_rmse=float(np.sqrt(totals[2]/n)) if n else np.nan,
            **bootstrap_difference([list(clusters.values())])))
    return out

def graph_scales(directory,rows,run):
    with (directory/'graph_diagnostics.csv').open() as f:graphs={(int(r['family']),int(r['mechanism'])):r for r in csv.DictReader(f)}
    truths={}; topology=[];scales=[]
    for (family,mid),graph in graphs.items():
        with np.load(directory/'truth'/f'f{family}_m{mid}.npz') as z:mu=z['arm_means']
        ite=np.stack((mu[:,2]-mu[:,0],mu[:,3]-mu[:,2],mu[:,3]-mu[:,0]),-1)
        with np.load(directory/'graphs'/f'f{family}_m{mid}.npz') as z:degree=z['adjacency'].sum(1)
        contrast=np.mean([arm_mean_exposure(int(d),.5,1)-arm_mean_exposure(int(d),.5,0) for d in degree])
        truths[family,mid]=(ite.mean(0),np.mean(ite**2,axis=0),contrast)
    for family in sorted({f for f,m in graphs}):
        group=[v for (f,m),v in graphs.items() if f==family]
        out=dict(run=run,family=family,distribution=group[0]['distribution'],mechanisms=len(group))
        for key in ('radius','mean_degree','density','clustering','components','attempts'):
            values=[float(r[key]) for r in group]
            out.update({key+'_mean':float(np.mean(values)),key+'_min':min(values),key+'_max':max(values)})
        topology.append(out)
    groups=defaultdict(list)
    for r in rows:
        if r['method']=='j':groups[r['family'],r['m_observed'],r['effect']].append(r)
    for (family,budget,effect),group in sorted(groups.items()):
        k=EFFECTS.index(effect);keys=sorted({(family,r['mechanism']) for r in group})
        scale=float(np.sqrt(np.mean([truths[key][0][k]**2 for key in keys])))
        valid=[r for r in group if r['point_valid']]; nci=sum(r['interval_valid'] for r in group)
        rmse=float(np.sqrt(np.mean([r['squared_error_sum'] for r in valid]))) if valid else np.nan
        scales.append(dict(run=run,family=family,m_observed=budget,effect=effect,
            radius_mean=np.mean([float(graphs[key]['radius']) for key in keys]),
            mean_degree=np.mean([float(graphs[key]['mean_degree']) for key in keys]),
            exposure_contrast=np.mean([truths[key][2] for key in keys]),true_ate_rms=scale,
            true_ite_rms=np.sqrt(np.mean([truths[key][1][k] for key in keys])),rmse=rmse,
            normalized_rmse=rmse/scale if scale>0 else np.nan,
            signed_bias=np.mean([r['error_sum'] for r in valid]) if valid else np.nan,
            mean_absolute_error=np.mean([abs(r['error_sum']) for r in valid]) if valid else np.nan,
            mean_width=sum(float(r['width_sum']) for r in group)/nci if nci else np.nan))
    return topology,scales

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','replication','matched','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    contrasts=[];coverage=[];scales=[];topology=[];tables=[]
    with threadpool_limits(limits=1):
        for run,directory in [('original',a.original),('replication',a.replication),('matched',a.matched)]:
            rows,digest=read_results(directory)
            contrasts+=baseline_comparisons(rows,run);coverage+=coverage_summary(rows,run)
            t,s=graph_scales(directory,rows,run);topology+=t;scales+=s
            for family in sorted({r['family'] for r in rows}):
                for budget in (20,50,100):tables += [dict(run=run,family=family,**r) for r in ate_table([r for r in rows if r['family']==family],budget)]
            (a.output/(run+'_source.json')).write_text(json.dumps(dict(context_metrics_sha256=digest)))
            if run=='matched':write_csv(a.output/'paired_spatial_rmse.csv',spatial_comparisons(rows))
    for name,data in [('paired_baseline_bootstrap',contrasts),('coverage_bootstrap',coverage),('effect_scales',scales),('topology_summary',topology),('ate_by_scenario',tables)]:write_csv(a.output/(name+'.csv'),data)
    (a.output/'analysis_protocol.json').write_text(json.dumps(dict(bootstrap_replicates=10000,seed=2026102734,
        interval='marginal percentile 95%, exploratory; no multiplicity correction',
        accuracy='pooled shared valid contexts; bootstrap entire mechanisms stratified by scenario or entire matched pairs',
        coverage='all requested intervals including failures',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(a.output)

if __name__=='__main__':main()
