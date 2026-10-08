"""Plots of compatible replicate summaries; bars represent sample SD, not confidence intervals."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from scaling_v1.runner import RunConfig
from .report import EFFECTS


def grouping_key(row,varying):
    ignore=set(varying)|{'graph_file','cache_dir','train_seed'}
    identity={f.name:row[f.name] for f in fields(RunConfig) if f.name not in ignore}
    identity.update(data_identity=row['data_identity'],torch_version=row['torch_version'],
                    seeds=row['seeds'])
    if 'model_size' not in varying:
        identity['model_config']=row['model_config']
    return json.dumps(identity,sort_keys=True)


def plot_summary(directory,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    # Only replace files produced by this report, including now-obsolete cohorts.
    for old in directory.glob('v2_*.png'):
        old.unlink()
    settings=(('tasks','train_tasks',{'train_tasks'},'Training task bank size'),
        ('parameters','parameters',{'model_size'},'PFN parameters'),
        ('steps','steps',{'steps'},'Requested optimizer updates'),
        ('compute','compute_seconds_mean',{'model_size','train_tasks','steps'},'Measured runtime (seconds)'))
    colors=dict(small='tab:blue',medium='tab:orange',large='tab:green',xlarge='tab:red')
    for name,xfield,varying,xlabel in settings:
        cohorts={}
        for row in rows:
            cohorts.setdefault(grouping_key(row,varying),[]).append(row)
        for identity,cohort in cohorts.items():
            direct=[r for r in cohort if r['effect']=='direct']
            if name!='compute' and len({r[xfield] for r in direct})<2:
                continue
            fig,axes=plt.subplots(1,3,figsize=(13,4.6))
            fig.subplots_adjust(bottom=.22,top=.78,wspace=.3)
            for effect,axis in zip(EFFECTS,axes):
                selected=sorted([r for r in cohort if r['effect']==effect],key=lambda r:r[xfield])
                if name=='compute':
                    labelled=set()
                    for r in selected:
                        model=r['model_size']
                        axis.errorbar(r[xfield],r['ite_rmse_mean'],yerr=r['ite_rmse_std'],
                            fmt='o',color=colors.get(model,'black'),capsize=3,
                            label=model if model not in labelled else None)
                        labelled.add(model)
                    axis.legend(fontsize=8)
                else:
                    deviation=[r['ite_rmse_std'] for r in selected]
                    axis.errorbar([r[xfield] for r in selected],[r['ite_rmse_mean'] for r in selected],
                        yerr=deviation if all(v is not None for v in deviation) else None,fmt='o-',capsize=4)
                    axis.set_xscale('log',base=2)
                axis.set(xlabel=xlabel,ylabel='Aggregated ITE RMSE',title=effect)
                axis.grid(alpha=.25)
            sample=direct[0]
            fixed=[]
            for field,label in [('model_size','model'),('train_tasks','D'),('steps','updates')]:
                if field not in varying:
                    fixed.append(f'{label}={sample[field]}')
            fig.suptitle(f'{name} | '+', '.join(fixed)+f" | seeds={sample['seeds']} | {sample['device']}",fontsize=11)
            partial=any(r['evaluated_task_coverage_mean']<1 or r['replication_complete'] is False for r in direct)
            note='Bars: sample SD across training seeds; n=1 has no bar. Validation-selected checkpoints.'
            if partial:
                note+='\nPARTIAL: some checkpoints have not seen the full task bank, or planned seeds are missing.'
            if name=='compute':
                note+='\nRuntime includes data, training, validation and test; excludes checkpoint/report I/O.'
            fig.text(.5,.035,note,ha='center',fontsize=8)
            suffix=hashlib.sha256(identity.encode()).hexdigest()[:10]
            fig.savefig(directory/f'v2_{name}_{suffix}.png',dpi=180)
            plt.close(fig)
