"""Plot actual per-epoch held-out metrics; does not use final test results."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

EFFECTS=('direct','spillover','total')
ARMS=('mu00','mu01','mu10','mu11')


def read_history(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        item=json.loads(line)
        epoch=int(item['epoch']); v=item['validation']; t=item.get('train',{})
        if epoch<1 or (rows and epoch<=rows[-1]['epoch']):
            raise ValueError('History epochs must be positive and strictly increasing.')
        row=dict(epoch=epoch,train_gmm_nll=t.get('total_loss',float('nan')),
            validation_gmm_nll=v['validation_gmm_nll'],
            validation_rmse_macro=v['effect_rmse_macro'],lr=item.get('lr',float('nan')))
        for effect in EFFECTS:
            for metric in ('rmse','mae','bias'):
                row[f'validation_{effect}_{metric}']=v[f'{effect}_effect_{metric}']
        for arm in ARMS:
            for split,metrics in (('train',t),('validation',v)):
                key=arm+'_gmm_nll'
                if key in metrics:
                    row[f'{split}_{key}']=metrics[key]
        if not all(math.isfinite(float(value)) for key,value in row.items() if key.startswith('validation_')):
            raise ValueError('Cannot plot nonfinite validation metrics.')
        rows.append(row)
    if not rows: raise ValueError('No completed validation epochs in history.')
    return rows


def plot_history(history_path,output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_history(history_path)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    epochs=[r['epoch'] for r in rows]
    marker='o' if len(rows)==1 else None
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('PFN with interference | CEPO supervision, derived effects',fontsize=15)
    ax=axs[0,0]
    ax.plot(epochs,[r['train_gmm_nll'] for r in rows],label='Training',color='#9a9a9a',lw=1.6,marker=marker)
    ax.plot(epochs,[r['validation_gmm_nll'] for r in rows],label='Validation',color='#2459a6',lw=2,marker=marker)
    ax.set(title='CEPO GMM negative log-likelihood',ylabel='NLL')
    colors=('#2459a6','#e78524','#40946a')
    for ax,metric in zip((axs[0,1],axs[1,0],axs[1,1]),('rmse','mae','bias')):
        for effect,color in zip(EFFECTS,colors):
            ax.plot(epochs,[r[f'validation_{effect}_{metric}'] for r in rows],label=effect.capitalize(),color=color,lw=1.8,marker=marker)
        ax.set(title='Validation '+metric.upper(),ylabel=metric.upper())
        if metric=='bias':ax.axhline(0,color='#666666',lw=.8,ls='--')
    best=min(rows,key=lambda r:r['validation_gmm_nll'])['epoch']
    axs[0,0].axvline(best,color='#777777',lw=1,ls=':',label=f'Best CEPO NLL: epoch {best}')
    for ax in axs.flat:
        if len(rows)==1: ax.set_xticks(epochs)
        ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    for extension in ('png','pdf'):
        path=out/f'validation_curves.{extension}'
        temp=path.with_name(path.name+'.building')
        fig.savefig(temp,format=extension,dpi=160)
        temp.replace(path)
    plt.close(fig)
    if all(f'{split}_{arm}_gmm_nll' in r for r in rows for arm in ARMS for split in ('train','validation')):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        fig.suptitle('Four CEPO GMM losses | equal weight 1/4',fontsize=15)
        for ax,arm in zip(axs.flat,ARMS):
            ax.plot(epochs,[r[f'train_{arm}_gmm_nll'] for r in rows],label='Training',color='#9a9a9a',marker=marker)
            ax.plot(epochs,[r[f'validation_{arm}_gmm_nll'] for r in rows],label='Validation',color='#2459a6',marker=marker)
            ax.axvline(best,color='#777777',lw=1,ls=':',label=f'Best four-arm NLL: epoch {best}')
            ax.set(title=arm,xlabel='Epoch',ylabel='GMM NLL')
            if len(rows)==1: ax.set_xticks(epochs)
            ax.grid(alpha=.2);ax.legend(fontsize=9)
            ax.spines[['top','right']].set_visible(False)
        for extension in ('png','pdf'):
            path=out/f'four_mu_loss_curves.{extension}'
            temp=path.with_name(path.name+'.building')
            fig.savefig(temp,format=extension,dpi=160)
            temp.replace(path)
        plt.close(fig)
    path=out/'validation_metrics.csv';temp=path.with_suffix('.csv.building')
    with temp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();rows=plot_history(a.history,a.output_dir)
    print(f'Validation curves written: {a.output_dir}; completed epochs={len(rows)}',flush=True)


if __name__=='__main__':main()
