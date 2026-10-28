"""Audit and report the frozen, independently calibrated E1.2 evaluation."""
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

from benchmark import configuration as check_frozen_assets
from make_table import read_results, write_csv, EFFECTS
from report_e12 import family_tables, LABELS
from run_e12_calibration import digest, source_fingerprints, CALIBRATION_IDS, TEST_IDS, BUDGETS
from pfn_social.e12_configuration import FAMILY_BASE_IDS, configuration
from pfn_social.e12_interval_calibration import METHOD, conformal_factor


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def bootstrap_pair(original,calibrated):
    """Whole-mechanism resampling; repeats remain clustered. Descriptive only."""
    original,calibrated=np.asarray(original),np.asarray(calibrated)
    rng=np.random.default_rng(20261028)
    indices=rng.integers(len(original),size=(10000,len(original)))
    values=np.stack((original[indices].mean(1),calibrated[indices].mean(1)),1)
    result={}
    for i,name in enumerate(('original','calibrated')):
        result[name+'_bootstrap_low'],result[name+'_bootstrap_high']=map(float,np.quantile(values[:,i],[.025,.975]))
    result['gain_bootstrap_low'],result['gain_bootstrap_high']=map(float,np.quantile(values[:,1]-values[:,0],[.025,.975]))
    return result


def audit(run):
    protocol=json.loads((run/'protocol.json').read_text())
    artifact=json.loads((run/'frozen_calibration.json').read_text())
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    frozen=next(i for i,r in enumerate(events) if r['stage']=='calibration_frozen')
    tests=[i for i,r in enumerate(events) if r['stage']=='stage_started' and r.get('role')=='test']
    assert tests and min(tests)>frozen
    assert all(events[i]['frozen_calibration_sha256']==digest(run/'frozen_calibration.json') for i in tests)
    assert events[frozen]['sha256']==digest(run/'frozen_calibration.json')
    assert artifact['registered_protocol_sha256']==digest(run/'protocol.json')
    assert artifact['calibration_execution_sha256']==digest(run/'calibration/merged/execution.json')
    assert protocol['specification']['source_sha256']==source_fingerprints()
    assert not set(CALIBRATION_IDS)&set(TEST_IDS)
    check_frozen_assets()
    scores=read(run/'calibration_scores.csv')
    for factor in artifact['factors']:
        selected=[r for r in scores if (int(r['family']),int(r['m_observed']),r['effect'])==
                  (factor['family'],factor['m_observed'],factor['effect'])]
        assert len(selected)==100 and {int(r['mechanism']) for r in selected}==set(CALIBRATION_IDS)
        assert {int(r['repeat']) for r in selected}=={0}
        expected=conformal_factor([float(r['score']) for r in selected])
        assert expected['factor']==factor['factor'] and factor['order_statistic']==96
    directory=run/'test/merged'
    manifest=json.loads((directory/'execution.json').read_text())
    graphs=json.loads((directory/'graph_statistics.json').read_text())
    assert len(graphs)==1200
    assert {(g['family'],g['mechanism']) for g in graphs}=={(f,m) for f in FAMILY_BASE_IDS for m in TEST_IDS}
    for graph in graphs:
        setting=configuration(graph['family']).setting
        assert graph['connected'] and graph['degree_mean']==setting.degree
        assert graph['degree_sd']==setting.degree_spread
    original=[r for r in manifest['receipts'] if r['method']=='j']
    assert len(original)==10800
    checked=0
    for receipt in original:
        context=receipt['context_id']
        with np.load(directory/'predictions'/f'{context}__j.npz') as a, np.load(directory/'predictions'/f'{context}__{METHOD}.npz') as b:
            for key in ('arm_means','ite_effects','ate_effects','arm_supported','ite_supported','ate_supported'):
                np.testing.assert_array_equal(a[key],b[key])
            if not receipt['interval_failure']:
                assert np.all(b['ate_lower']<=a['ate_lower']) and np.all(b['ate_upper']>=a['ate_upper'])
        checked+=1
    return dict(protocol_sha256=digest(run/'protocol.json'),calibration_sha256=digest(run/'frozen_calibration.json'),
        all_calibration_frozen_before_first_test_generation=True,registered_source_hashes_unchanged=True,
        no_calibration_test_mechanism_overlap=True,all_54_factors_reproduced_from_calibration_only=True,
        all_frozen_asset_hashes_match=True,verified_test_graphs=1200,verified_point_identical_contexts=checked,
        original_J_intervals_contained_in_calibrated=True,
        prediction_failure_receipts=sum(bool(r['failure']) for r in manifest['receipts']),
        interval_failure_receipts=sum(bool(r['interval_failure']) for r in manifest['receipts']))


def build_tables(run,out):
    directory=run/'test/merged'
    rows,metric_hash=read_results(directory)
    summary=read(directory/'evaluation/summary_by_family.csv')
    lookup={(int(r['family']),r['method'],int(r['m_observed']),r['effect']):r for r in summary}
    records=family_tables(rows,summary)
    for record in records:
        source=lookup[record['family'],record['method'],record['m_observed'],record['effect']]
        record['mean_interval_width']=float(source['mean_width'])
    write_csv(out/'all_methods_by_family.csv',records)
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['family'],row['method'],row['m_observed'],row['effect']].append(row)
    factors=json.loads((run/'frozen_calibration.json').read_text())['factors']
    factor_lookup={(r['family'],r['m_observed'],r['effect']):r['factor'] for r in factors}
    comparison=[]
    task_rows=[]
    for (family,method,budget,effect),members in sorted(grouped.items()):
        tasks=defaultdict(list)
        for r in members:
            tasks[r['mechanism']].append(r)
        rmse=[];bias=[]
        for mid,task in tasks.items():
            valid=all(r['point_valid']==1 for r in task) and len(task)==3
            if valid:
                rmse.append(float(np.sqrt(np.mean([r['squared_error_sum'] for r in task]))))
                bias.append(float(np.mean([r['error_sum'] for r in task])))
            task_rows.append(dict(family=family,mechanism=mid,method=method,m_observed=budget,effect=effect,
                complete=valid,rmse=rmse[-1] if valid else np.nan,signed_bias=bias[-1] if valid else np.nan))
        record=next(r for r in records if (r['family'],r['method'],r['m_observed'],r['effect'])==(family,method,budget,effect))
        record.update(mean_task_rmse=float(np.mean(rmse)) if rmse else np.nan,
                      sd_task_rmse=float(np.std(rmse,ddof=1)) if len(rmse)>1 else np.nan,
                      sd_task_signed_bias=float(np.std(bias,ddof=1)) if len(bias)>1 else np.nan)
    write_csv(out/'all_methods_by_family.csv',records)
    write_csv(out/'task_metrics.csv',task_rows)
    for family in FAMILY_BASE_IDS:
        for budget in BUDGETS:
            for effect in EFFECTS:
                key=(family,budget,effect)
                a=lookup[family,'j',budget,effect];b=lookup[family,METHOD,budget,effect]
                j=grouped[family,'j',budget,effect];cal=grouped[family,METHOD,budget,effect]
                by_context={r['context_id']:r for r in cal}
                for r in j:
                    v=by_context[r['context_id']]
                    assert (r['error_sum'],r['squared_error_sum'],r['point_valid'])==(
                        v['error_sum'],v['squared_error_sum'],v['point_valid'])
                jm={mid:np.mean([r['covered'] for r in j if r['mechanism']==mid]) for mid in TEST_IDS}
                cm={mid:np.mean([r['covered'] for r in cal if r['mechanism']==mid]) for mid in TEST_IDS}
                width_a=float(a['mean_width']);width_b=float(b['mean_width'])
                factor=factor_lookup[key]
                np.testing.assert_allclose(width_b/width_a,factor,rtol=1e-12)
                raw=float(a['coverage_including_failures']);scaled=float(b['coverage_including_failures'])
                first_a=[r for r in j if r['repeat']==0];first_b=[r for r in cal if r['repeat']==0]
                comparison.append(dict(family=family,graph_label=LABELS[family],m_observed=budget,effect=effect,
                    mechanisms=200,repeats=3,contexts=len(j),frozen_factor=factor,
                    original_covered=int(a['covered']),calibrated_covered=int(b['covered']),
                    original_coverage=raw,calibrated_coverage=scaled,coverage_gain=scaled-raw,
                    original_mean_width=width_a,calibrated_mean_width=width_b,width_ratio=width_b/width_a,
                    original_target_met=raw>=.90,calibrated_target_met=scaled>=.90,
                    original_repeat0_coverage=float(np.mean([r['covered'] for r in first_a])),
                    calibrated_repeat0_coverage=float(np.mean([r['covered'] for r in first_b])),
                    **bootstrap_pair(list(jm.values()),list(cm.values()))))
    write_csv(out/'coverage_comparison.csv',comparison)
    failures=[r for r in comparison if not r['calibrated_target_met']]
    if failures:
        write_csv(out/'failed_calibrated_cells.csv',failures)
    else:
        (out/'failed_calibrated_cells.csv').write_text('family,m_observed,effect,calibrated_coverage\n')
    return comparison,records,metric_hash


def plots(out,comparison):
    columns=[(b,e) for b in BUDGETS for e in EFFECTS]
    fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    minimum=min(.85,min(r['original_coverage'] for r in comparison))
    for ax,key,title in zip(axes,('original_coverage','calibrated_coverage'),('Frozen J','J + independent calibration')):
        grid=np.array([[next(r[key] for r in comparison if (r['family'],r['m_observed'],r['effect'])==(f,b,e))
                        for b,e in columns] for f in FAMILY_BASE_IDS])
        heat=ax.imshow(grid,vmin=minimum,vmax=1,cmap='RdYlGn',aspect='auto')
        ax.set(yticks=range(6),yticklabels=[LABELS[f] for f in FAMILY_BASE_IDS],
               xticks=range(9),xticklabels=[f'm={b}\n{e}' for b,e in columns],title=title)
        for i,j in np.ndindex(grid.shape):
            ax.text(j,i,f'{grid[i,j]*100:.1f}%'+('*' if grid[i,j]<.90 else ''),ha='center',va='center',fontsize=9)
    fig.colorbar(heat,ax=axes,fraction=.02,pad=.02,label='Held-out coverage')
    fig.suptitle('Nominal 95% ATE intervals | 200 held-out mechanisms per family x 3 repeats\n* below 90% target; all factors frozen before testing')
    fig.savefig(out/'coverage_before_after.png',dpi=180)
    fig.savefig(out/'coverage_before_after.pdf')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
    grid=np.array([[next(r['width_ratio'] for r in comparison if (r['family'],r['m_observed'],r['effect'])==(f,b,e))
                    for b,e in columns] for f in FAMILY_BASE_IDS])
    heat=ax.imshow(grid,vmin=1,vmax=max(1.01,grid.max()),cmap='Blues',aspect='auto')
    ax.set(yticks=range(6),yticklabels=[LABELS[f] for f in FAMILY_BASE_IDS],xticks=range(9),
           xticklabels=[f'm={b}\n{e}' for b,e in columns],title='Interval width multiplier | calibration-only choice; point estimates unchanged')
    for i,j in np.ndindex(grid.shape):
        ax.text(j,i,f'{grid[i,j]:.2f}x',ha='center',va='center',color='white' if grid[i,j]>1+(grid.max()-1)*.6 else 'black')
    fig.colorbar(heat,ax=ax,fraction=.025,label='Calibrated / original width')
    fig.savefig(out/'width_cost.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    verification=audit(args.input)
    comparison,records,metric_hash=build_tables(args.input,args.output)
    plots(args.output,comparison)
    verification.update(context_metrics_sha256=metric_hash,cells=54,
        original_cells_meeting_90=sum(r['original_target_met'] for r in comparison),
        calibrated_cells_meeting_90=sum(r['calibrated_target_met'] for r in comparison),
        min_original_coverage=min(r['original_coverage'] for r in comparison),
        min_calibrated_coverage=min(r['calibrated_coverage'] for r in comparison),
        min_width_multiplier=min(r['width_ratio'] for r in comparison),
        max_width_multiplier=max(r['width_ratio'] for r in comparison),
        mean_width_multiplier=float(np.mean([r['width_ratio'] for r in comparison])),
        point_metrics_identical=True,no_test_set_retuning=True,
        bootstrap='10000 whole-mechanism paired resamples, seed 20261028; descriptive 95% bands, not simultaneous guarantees.',
        oracle_or_prior_strength_adjustment_performed=False)
    (args.output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    for name in ('protocol.json','events.jsonl','frozen_calibration.json','calibration_factors.csv','calibration_scores.csv'):
        shutil.copyfile(args.input/name,args.output/name)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    main()
