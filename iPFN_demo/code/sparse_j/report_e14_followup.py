"""Draw follow-up figures and reuse the upstream figure/table templates."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_figures import build_figure
from make_table import read_results,write_csv,ate_table,render_table,render_coverage_table

def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def val(r,k):return float(r[k])

def templates(directory):
    rows,digest=read_results(directory);out=directory/'templates';out.mkdir(exist_ok=False)
    for family in sorted({r['family'] for r in rows}):
        group=[r for r in rows if r['family']==family]
        for budget in (20,50,100):
            records=ate_table(group,budget)
            write_csv(out/f'f{family}_ate_m{budget}.csv',records)
            (out/f'f{family}_ate_m{budget}.tex').write_text(render_table(records,budget)+'\n'+render_coverage_table(records,budget),encoding='utf-8')
        fig,records=build_figure([r for r in group if r['m_observed']==50])
        for suffix in ('png','pdf'):fig.savefig(out/f'f{family}_ate_m50.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig);write_csv(out/f'f{family}_ate_m50_source.csv',records)
    (out/'provenance.json').write_text(json.dumps(dict(context_metrics_sha256=digest,
        implementation='Unchanged make_figures.build_figure and make_table functions',
        grouping='Matched families plotted separately; independent mechanism counts remain correct'),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--replication',type=Path,required=True)
    p.add_argument('--matched',type=Path,required=True)
    p.add_argument('--skip-templates',action='store_true',help='Refresh diagnostic figures without rebuilding existing template directories')
    a=p.parse_args();out=a.input
    coverage=read(out/'coverage_bootstrap.csv');baselines=read(out/'paired_baseline_bootstrap.csv')
    spatial=read(out/'paired_spatial_rmse.csv');scales=read(out/'effect_scales.csv')
    primary=[next(r for r in coverage if (r['run'],r['family'],r['m_observed'],r['effect'])==(run,'9','50','spillover')) for run in ('original','replication')]
    kernels=[r for r in baselines if r['run']=='original' and r['family']=='all' and r['m_observed']=='50' and r['baseline'].startswith('kernel')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,caption):
        for suffix in ('png','pdf'):fig.savefig(out/(name+'.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig);figures.append((name,caption))
    fig,ax=plt.subplots(figsize=(8,3.7),constrained_layout=True)
    for i,r in enumerate(primary):
        y=val(r,'coverage');ax.errorbar(i,y,yerr=[[y-val(r,'lower')],[val(r,'upper')-y]],fmt='o',capsize=7,color=('#778394','#087e8b')[i])
        ax.annotate(f"{y:.2%}\n{r['covered']}/{r['requested']}",(i,y),xytext=(15,0),textcoords='offset points')
    ax.axhline(.90,color='#c65b37',ls='--',label='Empirical target: 90%');ax.axhline(.95,color='#657083',ls=':',label='Nominal interval level: 95%')
    ax.set(xticks=[0,1],xticklabels=['Initial: 20 mechanisms','New: 100 mechanisms'],xlim=(-.5,1.65),ylabel='Coverage',title='Uniform r=0.24 | M=50 | Treated spillover');ax.legend(loc='lower right',fontsize=8)
    save(fig,'replication_coverage','原始与新增样本分开报告；误差条为按整个机制重抽样得到的覆盖率95%区间。')
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,effect in zip(axes,('direct','spillover','total')):
        group=[r for r in kernels if r['effect']==effect]
        for i,r in enumerate(group):
            v=val(r,'difference');ax.errorbar(v,i,xerr=[[max(0,v-val(r,'lower'))],[max(0,val(r,'upper')-v)]],fmt='o',capsize=4,color='#087e8b')
        ax.axvline(0,color='#667788',ls='--');ax.set(yticks=range(len(group)),yticklabels=[r['baseline'] for r in group],ylim=(-.5,1.5),title=effect.title(),xlabel='RMSE(J) - RMSE(baseline)')
    save(fig,'kernel_comparison','原始六场景M=50，按场景分层、机制成组的配对bootstrap；负值有利于J，跨0不据此判断优劣。全部baseline见CSV。')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,effect in zip(axes,('direct','spillover','total')):
        group=[r for r in spatial if r['effect']==effect and r['method']=='j']
        for i,r in enumerate(group):
            v=val(r,'difference');ax.errorbar(i,v,yerr=[[max(0,v-val(r,'lower'))],[max(0,val(r,'upper')-v)]],fmt='o',capsize=5,color='#c26034')
        ax.axhline(0,color='#667788',ls='--');ax.set(xticks=range(3),xticklabels=[r['m_observed'] for r in group],title=effect.title(),xlabel='Observed outcomes M',ylabel='J RMSE(clustered) - RMSE(uniform)')
    save(fig,'spatial_comparison','50对独立机制，平均度均为18；每次重抽样保留整对机制。')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,(distribution,families) in enumerate([('Uniform',('7','8','9')),('Clustered',('10','11','12'))]):
        group=[r for r in scales if r['run']=='original' and r['family'] in families and r['m_observed']=='50' and r['effect']=='spillover']
        x=[val(r,'radius_mean') for r in group]
        for key,label in [('true_ate_rms','True ATE RMS'),('rmse','J RMSE'),('mean_width','J interval width')]:axes[row,0].plot(x,[val(r,key) for r in group],'o-',label=label)
        axes[row,0].legend(fontsize=8)
        axes[row,1].plot(x,[val(r,'normalized_rmse') for r in group],'o-',color='#087e8b')
        axes[row,2].plot(x,[val(r,'exposure_contrast') for r in group],'o-',color='#c26034')
        for col,label in enumerate(('Absolute scale','RMSE / true ATE RMS','Mean exposure contrast')):axes[row,col].set(title=distribution+' / '+label,xlabel='Radius',xticks=[.12,.18,.24])
    save(fig,'effect_scales','溢出效应M=50：区分绝对误差、真实效应尺度、区间宽度与majority exposure对比。')
    if not a.skip_templates:
        templates(a.replication);templates(a.matched)
    print(out)

if __name__=='__main__':main()
