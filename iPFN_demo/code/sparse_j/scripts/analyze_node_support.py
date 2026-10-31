"""Postprocess saved E1.5 predictions; never import a predictor or DGP module.

Only four unchanged functions are extracted from the hash-verified source ZIP:
effects, score, support_reason, summarize. Bootstrap is the frozen statistical
mechanism bootstrap, not a new experimental draw. All writes are new files.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np

METHODS = ('j', 'kernel-old', 'kernel-new')
ARMS = ('00', '01', '10', '11')
EFFECTS = ('direct', 'spillover', 'total')
REASONS = ('conditional_structural_zero', 'assignment_zero',
           'observation_sampling_zero', 'covered')
IDENTITY = ('context_id', 'mechanism_id', 'pair_repeat_id', 'center_t', 'method')
ERROR_FIELDS = ('true_mean', 'pred_mean', 'bias', 'mae', 'mse', 'rmse')
DIFFERENCE_FIELDS = ('center_minus_leaf_bias', 'center_minus_leaf_mae',
                     'center_minus_leaf_rmse')
PROTOCOL = 'e1_5_star_saved_node_support_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, message='Frozen metric disagreement'):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    require(bool(rows), 'Refuse empty CSV')
    fields = list(rows[0])
    require(all(set(r) == set(fields) for r in rows), 'CSV schema differs')
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_frozen(run):
    """Compile selected immutable bodies, not source-module imports/top levels."""
    manifest = json.loads((Path(run)/'run_manifest.json').read_text(encoding='utf-8'))
    specifications = {
        'iPFN_demo/code/sparse_j/scripts/e1_5_star.py':
            (('effects', 'score'), ('ARMS', 'EFFECTS', 'N')),
        'iPFN_demo/code/sparse_j/scripts/e1_5_star_analysis.py':
            (('support_reason', 'summarize'),
             ('BOOTSTRAP_SEED', 'BOOTSTRAP_DRAWS', 'STRUCTURAL', 'ASSIGNMENT', 'SAMPLING', 'COVERED')),
    }
    namespace = {'np': np, '__name__': 'frozen_saved_postprocessing'}
    provenance = {}
    with zipfile.ZipFile(Path(run)/'source_snapshot.zip') as archive:
        for name, (functions, constants) in specifications.items():
            raw = archive.read(name)
            require(hashlib.sha256(raw).hexdigest() == manifest['source_sha256'][name],
                    'Frozen source hash mismatch: '+name)
            source = raw.decode('utf-8')
            tree = ast.parse(source)
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
                elif isinstance(target, ast.Tuple):
                    values = ast.literal_eval(node.value)
                    for key, value in zip(target.elts, values):
                        if isinstance(key, ast.Name) and key.id in constants:
                            namespace[key.id] = value
            chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
            require({n.name for n in chosen} == set(functions), 'Frozen functions missing')
            for node in chosen:
                require(not node.decorator_list, 'Unexpected decorator')
                provenance[node.name] = dict(source=name,
                    source_sha256=manifest['source_sha256'][name],
                    function_text_sha256=hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest())
            exec(compile(ast.Module(body=chosen, type_ignores=[]), 'snapshot:'+name, 'exec'), namespace)
    require(namespace['N'] == 200 and namespace['ARMS'] == ARMS
            and namespace['EFFECTS'] == EFFECTS, 'Frozen state definition changed')
    require(namespace['BOOTSTRAP_SEED'] == 2026102502
            and namespace['BOOTSTRAP_DRAWS'] == 10000, 'Bootstrap definition changed')
    return SimpleNamespace(**namespace), provenance


def context_id(mid, rid, center):
    return f'f0_m{mid}_r{2*rid+center}_observed50'


def check_task_grid(contexts, receipts, mids, rids, methods=METHODS):
    expected_contexts = {context_id(m, r, c) for m in mids for r in rids for c in (0, 1)}
    ids = [r['context_id'] for r in contexts]
    keys = [(r['context_id'], r['method']) for r in receipts]
    require(len(ids) == len(set(ids)) and set(ids) == expected_contexts,
            'Missing/duplicate/unexpected context')
    require(len(keys) == len(set(keys)) and set(keys) ==
            {(c, method) for c in expected_contexts for method in methods},
            'Missing/duplicate/unexpected method-context')


def error_statistics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    require(truth.shape == prediction.shape and truth.size > 0
            and np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Invalid error arrays')
    error = prediction-truth
    mse = float(np.mean(error**2))
    return dict(true_mean=float(truth.mean()), pred_mean=float(prediction.mean()),
                bias=float(error.mean()), mae=float(np.abs(error).mean()), mse=mse,
                rmse=math.sqrt(mse))


def score_context(frozen, inputs, truth, prediction, identity, model_arm_supported,
                  model_effect_supported):
    require(truth.shape == prediction.shape == (200, 4), 'Require truth/prediction[200,4]')
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Nonfinite point arrays')
    mid, rid, center = (identity[k] for k in ('mechanism_id', 'pair_repeat_id', 'center_t'))
    native_metrics, native_support = frozen.score(inputs, truth, prediction, mid, rid, center)
    true_effects, pred_effects = frozen.effects(truth), frozen.effects(prediction)
    response, effects = [], []
    for category, nodes in (('center', np.arange(1)), ('leaf', np.arange(1, 200))):
        common = dict(identity, center_observed=bool(0 in inputs['observed_indices']),
                      category=category, n_nodes=len(nodes))
        native = next(r for r in native_metrics if r['category'] == category)
        for j, arm in enumerate(ARMS):
            row = dict(common, arm=arm, **error_statistics(truth[nodes, j], prediction[nodes, j]))
            close(row['rmse'], next(r['arm_rmse'] for r in native_support
                                  if r['category'] == category and r['arm'] == arm))
            response.append(row)
        close(math.sqrt(np.mean([r['mse'] for r in response if r['category'] == category])),
              native['arm_rmse'])
        for j, effect in enumerate(EFFECTS):
            row = dict(common, effect=effect, model_effect_supported=bool(model_effect_supported[j]),
                       **error_statistics(true_effects[nodes, j], pred_effects[nodes, j]))
            for field, suffix in (('true_mean', 'ate_true'), ('pred_mean', 'ate_pred'),
                                  ('bias', 'bias'), ('rmse', 'rmse')):
                close(row[field], native[effect+'_'+suffix])
            effects.append(row)
    support = []
    for row in native_support:
        support.append(dict(identity, **{k: v for k, v in row.items() if k not in identity},
            structurally_allowed=row['support_reason'] != frozen.STRUCTURAL,
            assignment_produced=row['full_graph_count'] > 0,
            observed_covered=row['observed_count'] > 0,
            model_arm_supported=bool(model_arm_supported[ARMS.index(row['arm'])])))
    differences = []
    for effect in EFFECTS:
        center_row = next(r for r in effects if r['category'] == 'center' and r['effect'] == effect)
        leaf_row = next(r for r in effects if r['category'] == 'leaf' and r['effect'] == effect)
        stats = error_statistics(np.array([center_row['true_mean']-leaf_row['true_mean']]),
                                 np.array([center_row['pred_mean']-leaf_row['pred_mean']]))
        delta = {f'center_minus_leaf_{k}':center_row[k]-leaf_row[k] for k in ('bias','mae','rmse')}
        close(stats['bias'], delta['center_minus_leaf_bias'])
        differences.append(dict(identity, center_observed=bool(0 in inputs['observed_indices']),
            category='center_minus_leaf', n_nodes=1, effect=effect, **stats, **delta))
    return response, effects, differences, support, native_metrics, native_support


def error_aggregate(rows, target, level, mids, rids, methods=METHODS,
                    categories=('center', 'leaf'), extra=()):
    """Macro RMSE is frozen mean(context RMSE); pooled_rmse is separate."""
    keys = ['method', 'center_t', 'category', target]
    if level == 'mechanism':
        keys.insert(1, 'mechanism_id')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    targets = ARMS if target == 'arm' else EFFECTS
    count = len(methods)*2*len(categories)*len(targets)*(len(mids) if level == 'mechanism' else 1)
    require(len(groups) == count, 'Incomplete aggregation group grid')
    output = []
    for key, members in sorted(groups.items()):
        label = dict(zip(keys, key))
        require(label['method'] in methods and label['category'] in categories
                and label['center_t'] in (0, 1) and label[target] in targets, 'Unexpected group')
        if level == 'mechanism':
            require(label['mechanism_id'] in mids and len(members) == len(rids)
                    and {r['pair_repeat_id'] for r in members} == set(rids), 'Incomplete repeat group')
            contexts = len(members)
        else:
            require(len(members) == len(mids) and {r['mechanism_id'] for r in members} == set(mids),
                    'Incomplete independent mechanism group')
            contexts = sum(r['contexts'] for r in members)
        require(len({r['n_nodes'] for r in members}) == 1, 'Mixed center/leaf group')
        stats = {f:float(np.mean([r[f] for r in members])) for f in (*ERROR_FIELDS, *extra)}
        require(np.isfinite(list(stats.values())).all(), 'Nonfinite aggregate')
        output.append(dict(label, n_nodes=members[0]['n_nodes'], contexts=contexts,
            mechanisms=1 if level == 'mechanism' else len(mids), repeats_per_mechanism=len(rids),
            **stats, mean_context_rmse=stats['rmse'], pooled_rmse=math.sqrt(stats['mse'])))
    return output


def support_aggregate(rows, level, mids, rids, methods=METHODS):
    keys = ['method', 'center_t', 'category', 'arm']
    if level == 'mechanism':
        keys.insert(1, 'mechanism_id')
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    require(len(groups) == len(methods)*2*2*4*(len(mids) if level == 'mechanism' else 1),
            'Incomplete Support group grid')
    flags = ('structurally_allowed','assignment_produced','observed_covered','model_supported')
    output = []
    for key, members in sorted(groups.items()):
        label = dict(zip(keys, key))
        if level == 'mechanism':
            require(len(members)==len(rids) and {r['pair_repeat_id'] for r in members}==set(rids),
                    'Incomplete Support repeats')
            counts={reason+'_count':sum(r['support_reason']==reason for r in members) for reason in REASONS}
            counts.update({f+'_count':sum(bool(r['model_arm_supported'] if f=='model_supported' else r[f])
                                        for r in members) for f in flags})
            contexts=len(members)
            means={k:float(np.mean([r[k] for r in members])) for k in
                   ('full_graph_count','observed_count','arm_rmse','center_observed')}
            means={('mean_'+k if k!='center_observed' else 'center_observed_rate'):v for k,v in means.items()}
            rates={k[:-6]+'_rate':v/contexts for k,v in counts.items()}
        else:
            require(len(members)==len(mids) and {r['mechanism_id'] for r in members}==set(mids),
                    'Incomplete Support mechanisms')
            count_keys=[r+'_count' for r in REASONS]+[f+'_count' for f in flags]
            counts={k:sum(r[k] for r in members) for k in count_keys}
            contexts=sum(r['contexts'] for r in members)
            average_keys=[k for k in members[0] if k.startswith('mean_') or k.endswith('_rate')]
            rates={k:float(np.mean([r[k] for r in members])) for k in average_keys}
            means={}
        require(sum(counts[r+'_count'] for r in REASONS)==contexts, 'Support reason counts differ')
        output.append(dict(label, contexts=contexts, mechanisms=1 if level=='mechanism' else len(mids),
                           **counts, **means, **rates))
    return output


def verify_input(run):
    run=Path(run).resolve();m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    e=json.loads((run/'execution.json').read_text(encoding='utf-8'));p=m['plan']
    require(m['status']=='complete' and m['protocol']=='e1_5_star_ate_v1', 'Not a completed formal input')
    require(p['cohort']=='formal' and p['mechanism_ids']==list(range(1000,1020))
            and p['pair_repeat_ids']==list(range(10)) and p['center_order']==[0,1]
            and p['seed']==20261024 and p['n']==200 and p['m_observed']==50, 'Frozen plan mismatch')
    require(e['plan']==p and e['methods']==list(METHODS), 'Execution mismatch')
    check_task_grid(e['contexts'],e['receipts'],p['mechanism_ids'],p['pair_repeat_ids'])
    require(not any(r['failure'] is not None or r['interval_failure'] is not None for r in e['receipts']),
            'Failed formal record: stop instead of dropping it')
    files={f.relative_to(run).as_posix() for f in run.rglob('*') if f.is_file()}
    require(files-{'run_manifest.json','acceptance_report.json'}==set(m['output_sha256']), 'Input inventory differs')
    hashes={name:digest(run/name) for name in sorted(files)}
    require(all(hashes[name]==h for name,h in m['output_sha256'].items()), 'Input hash mismatch')
    return m,e,hashes


def validate_arrays(inputs, truth, point):
    require(set(inputs)=={'x','adjacency','treatment','observed_indices','outcomes_observed'}, 'Wrong factual keys')
    for k,shape in [('x',(200,20)),('adjacency',(200,200)),('treatment',(200,)),
                    ('observed_indices',(50,)),('outcomes_observed',(50,))]:
        require(inputs[k].shape==shape and np.isfinite(inputs[k]).all(), 'Factual shape/nonfinite: '+k)
    idx=inputs['observed_indices'];require(idx.dtype==np.int64 and len(np.unique(idx))==50
        and idx.min()>=0 and idx.max()<200, 'Invalid observed indices')
    a=inputs['adjacency'];require(np.array_equal(a,a.T) and a[0].sum()==199
        and np.all(a[1:].sum(1)==1) and np.all(np.diag(a)==0)
        and np.isin(a,[0,1]).all(), 'Not the saved star graph')
    require(np.isin(inputs['treatment'],[0,1]).all(), 'Treatment must be binary')
    require(truth.shape==(200,4) and np.isfinite(truth).all(), 'Invalid truth')
    for k,shape in [('arm_means',(200,4)),('ite_effects',(200,3)),('ate_effects',(3,)),
                    ('arm_supported',(4,)),('ite_supported',(3,)),('ate_supported',(3,))]:
        require(point[k].shape==shape and np.isfinite(point[k]).all(), 'Point shape/nonfinite: '+k)
    require(all(point[k].all() for k in ('arm_supported','ite_supported','ate_supported')), 'Incomplete model capability')


def load_npz(path):
    with np.load(path,allow_pickle=False) as saved:
        return {k:saved[k] for k in saved.files}


def analyze(run):
    manifest,execution,hashes=verify_input(run);frozen,functions=load_frozen(run)
    mids=manifest['plan']['mechanism_ids'];rids=manifest['plan']['pair_repeat_ids'];run=Path(run)
    response=[];effects=[];difference=[];support=[];native_metrics={m:[] for m in METHODS};native_support={m:[] for m in METHODS};native_rows=[]
    contexts={r['context_id']:r for r in execution['contexts']};receipts={(r['context_id'],r['method']):r for r in execution['receipts']}
    shared_support={}
    for mid in mids:
        truth=load_npz(run/'truth'/f'f0_m{mid}.npz')['arm_means']
        for rid in rids:
            pair=[]
            for center in (0,1):
                cid=context_id(mid,rid,center);inputs=load_npz(run/'factual'/(cid+'.npz'));pair.append(inputs);info=contexts[cid]
                require((info['mechanism_id'],info['pair_repeat_id'],info['center_t'])==(mid,rid,center), 'Context identity mismatch')
                require(inputs['treatment'][0]==center and info['observed_indices']==inputs['observed_indices'].tolist()
                        and info['center_observed']==bool(0 in inputs['observed_indices']), 'Observation identity mismatch')
                entropy=20261024+mid if rid==0 else [20261024,2026102501,mid,rid]
                require(info['rng_entropy']==entropy and info['bit_generator']=='PCG64','Recorded seed mismatch')
                for method in METHODS:
                    receipt=receipts[cid,method]
                    require(receipt['factual_sha256']==hashes['factual/'+cid+'.npz']
                        and receipt['truth_sha256']==hashes[f'truth/f0_m{mid}.npz'], 'Method input hash mismatch')
                    point=load_npz(run/'predictions'/(cid+'__'+method+'.npz'));validate_arrays(inputs,truth,point)
                    np.testing.assert_allclose(point['ite_effects'],frozen.effects(point['arm_means']),rtol=2e-6,atol=2e-6)
                    np.testing.assert_allclose(point['ate_effects'],point['ite_effects'].mean(0),rtol=2e-6,atol=2e-6)
                    identity=dict(context_id=cid,mechanism_id=mid,pair_repeat_id=rid,center_t=center,method=method)
                    rr,ee,dd,ss,nm,ns=score_context(frozen,inputs,truth,point['arm_means'],identity,
                                                  point['arm_supported'],point['ite_supported'])
                    response+=rr;effects+=ee;difference+=dd;support+=ss
                    native_metrics[method]+=nm;native_support[method]+=ns
                    native_rows.extend(dict(identity,**{k:v for k,v in row.items() if k not in identity}) for row in nm)
                    support_identity=[{k:v for k,v in row.items() if k not in ('method','arm_rmse')} for row in ss]
                    if cid in shared_support:require(support_identity==shared_support[cid],'Methods have different empirical Support')
                    else:shared_support[cid]=support_identity
            for key in ('x','adjacency','observed_indices'):require(np.array_equal(pair[0][key],pair[1][key]),'Pair input differs')
            require(np.array_equal(pair[0]['treatment'][1:],pair[1]['treatment'][1:]),'Paired leaf treatment differs')
    tables={'response_context':response,'effects_context':effects,'effect_center_leaf_context':difference,
            'support_context':support,'frozen_metrics_context':native_rows}
    for name,rows,target,cats,extra in [('response',response,'arm',('center','leaf'),()),
            ('effects',effects,'effect',('center','leaf'),()),
            ('effect_center_leaf',difference,'effect',('center_minus_leaf',),DIFFERENCE_FIELDS)]:
        mech=error_aggregate(rows,target,'mechanism',mids,rids,categories=cats,extra=extra)
        tables[name+'_mechanism']=mech
        tables[name+'_overall_by_center']=error_aggregate(mech,target,'overall',mids,rids,categories=cats,extra=extra)
    tables['support_mechanism']=support_aggregate(support,'mechanism',mids,rids)
    tables['support_overall_by_center']=support_aggregate(tables['support_mechanism'],'overall',mids,rids)
    paired=[];primary=[];secondary=[];summaries=[];summary_csv=[]
    for method in METHODS:
        # Never pass method/context strings into the frozen numeric secondary loop.
        cc,mm,ss,summary=frozen.summarize(native_support[method],native_metrics[method],mids,rids)
        paired.extend(dict(method=method,**r) for r in cc);primary.extend(dict(method=method,**r) for r in mm)
        secondary.extend(dict(method=method,**r) for r in ss);summaries.append(dict(method=method,summary=summary))
        summary_csv.append(dict(method=method,mean_delta=summary['mean_delta'],ci95_lower=summary['ci95'][0],
            ci95_upper=summary['ci95'][1],independent_units=summary['independent_units'],
            bootstrap_seed=summary['bootstrap_seed'],bootstrap_draws=summary['bootstrap_draws'],
            bootstrap_unit=summary['bootstrap_unit'],interpretation=summary['interpretation']))
    tables.update(support_paired_arm_contrasts=paired,support_primary_by_mechanism=primary,
                  frozen_metrics_mechanism=secondary,support_primary_summary=summary_csv)
    lookup={(r['method'],r['mechanism_id'],r['center_t'],r['category'],r['effect']):r
            for r in tables['effects_mechanism']}
    for r in secondary:
        for effect in EFFECTS:
            wanted=lookup[r['method'],r['mechanism_id'],r['center_t'],r['category'],effect]
            for a,b in [('bias','bias'),('rmse','rmse'),('true_mean','ate_true'),('pred_mean','ate_pred')]:
                close(wanted[a],r[effect+'_'+b])
    reason_rows=[]
    for method in METHODS:
        for center in (0,1):
            for category in ('center','leaf'):
                members=[r for r in support if r['method']==method and r['center_t']==center and r['category']==category]
                for reason in REASONS:
                    per_mechanism=[sum(r['support_reason']==reason for r in members if r['mechanism_id']==mid)/(len(rids)*4) for mid in mids]
                    reason_rows.append(dict(method=method,center_t=center,category=category,support_reason=reason,
                        context_state_cases=len(members),count=sum(r['support_reason']==reason for r in members),
                        mechanism_equal_rate=float(np.mean(per_mechanism)),mechanisms=len(mids)))
    tables['support_reason_counts_by_center']=reason_rows
    expected_counts={'response_context':9600,'effects_context':7200,'effect_center_leaf_context':3600,
        'support_context':9600,'frozen_metrics_context':2400,'response_mechanism':960,'effects_mechanism':720,
        'effect_center_leaf_mechanism':360,'response_overall_by_center':48,'effects_overall_by_center':36,
        'effect_center_leaf_overall_by_center':18,'support_mechanism':960,'support_overall_by_center':48,
        'support_paired_arm_contrasts':2400,'support_primary_by_mechanism':60,'frozen_metrics_mechanism':240,
        'support_primary_summary':3,'support_reason_counts_by_center':48}
    require(set(tables)==set(expected_counts),'Output table inventory differs')
    for name,count in expected_counts.items():
        require(len(tables[name])==count,'Output row count differs: '+name)
        for row in tables[name]:
            require(all(np.isfinite(v) for v in row.values() if isinstance(v,(float,np.floating))),
                    'Nonfinite output: '+name)
    require(all(digest(run/name)==h for name,h in hashes.items()),'Input changed during computation')
    return manifest,execution,hashes,functions,tables,summaries


def markdown_report(tables):
    lines=['# E1.5 星图：保存预测的节点误差与 Support 补充分析','',
        '全部结果来自已经保存的真值、事实输入和预测；没有模型调用、训练、DGP生成或 --validate。冻结函数以只含四个函数的 AST 提取方式执行，没有导入项目推断模块。','',
        '## 指标与汇总','',
        '- 中心为节点0，叶子为节点1–199；每个上下文先在类别内部平均，未用199个叶子扩充机制权重。',
        '- 四状态为00、01、10、11；Direct=10−00，Spillover=11−10，Total=11−00。',
        '- bias=mean(pred−truth)，MAE=mean(abs(pred−truth))，MSE=mean((pred−truth)^2)，上下文RMSE=sqrt(MSE)。',
        '- 先平均同机制的10次重复，再等权平均20机制；中心T=0/1分别报告。汇总列rmse（亦列为mean_context_rmse）是上下文RMSE的平均，保持冻结summarize语义；pooled_rmse是sqrt(平均MSE)，是另一个指标。',
        '- center_minus_leaf文件的true_mean/pred_mean是中心平均效应减叶子平均效应；bias/MAE/RMSE描述该差异估计的误差。额外center_minus_leaf_bias/mae/rmse列是两类误差指标的差，不能与差异估计的MAE/RMSE混称。',
        '- Support保留conditional_structural_zero、assignment_zero、observation_sampling_zero、covered四个冻结名称；另列结构允许、实际产生、实际观测。center_observed是独立字段。',
        '- model_arm_supported是模型能力标记，不是经验Support；没有把全体ATE或ATE区间当作中心/叶子效应或节点区间。','',
        '## 中心/叶子误差','',
        '以下为冻结四状态合并响应RMSE的上下文均值，先按机制平均；不是四个状态RMSE的简单平均。逐状态均值、bias、MAE和RMSE见response_*.csv。','',
        '| 中心T | 方法 | 中心响应RMSE | 叶子响应RMSE |','|---:|---|---:|---:|']
    for center in (0,1):
        for method in METHODS:
            selected=[r for r in tables['frozen_metrics_mechanism'] if r['method']==method and r['center_t']==center]
            values={cat:np.mean([r['arm_rmse'] for r in selected if r['category']==cat]) for cat in ('center','leaf')}
            lines.append(f"| {center} | {method} | {values['center']:.6f} | {values['leaf']:.6f} |")
    lines+=['','节点效应误差（rmse为上下文RMSE的机制等权均值）：','',
            '| 中心T | 方法 | 效应 | 中心RMSE | 叶子RMSE |','|---:|---|---|---:|---:|']
    for center in (0,1):
        for method in METHODS:
            for effect in EFFECTS:
                selected=[r for r in tables['effects_overall_by_center'] if r['method']==method and r['center_t']==center and r['effect']==effect]
                values={r['category']:r['rmse'] for r in selected}
                lines.append(f"| {center} | {method} | {effect} | {values['center']:.6f} | {values['leaf']:.6f} |")
    lines+=['','## Support 原因与误差关联','',
        '支持原因计数以一个上下文×一个类别×一个状态为一例（每中心状态800例），不是800个独立样本。三方法共享事实输入，因此计数相同，仅列J避免三倍计数。','',
        '| 中心T | 类别 | 原因 | 例数/800 |','|---:|---|---|---:|']
    for r in tables['support_reason_counts_by_center']:
        if r['method']=='j':lines.append(f"| {r['center_t']} | {r['category']} | {r['support_reason']} | {r['count']}/800 |")
    lines+=['','冻结主差值为叶子四状态“条件性结构不可覆盖中心分配下RMSE − 允许覆盖中心分配下RMSE”，先平均四状态、再平均同机制重复、再平均机制。允许覆盖不等于实际已观测。','',
        '| 方法 | 平均差值 | 机制Bootstrap 95%区间 |','|---|---:|---|']
    for r in tables['support_primary_summary']:
        lines.append(f"| {r['method']} | {r['mean_delta']:.6f} | [{r['ci95_lower']:.6f}, {r['ci95_upper']:.6f}] |")
    lines+=['','## 局限及复用','',
        'Bootstrap完全沿用冻结summarize：种子2026102502，10000次整机制重抽样，独立单位20。它只用于既有保存误差的统计不确定性，不是新增实验样本；不是ATE后验区间。frozen_support_summary.json原样保留原函数的legacy协议标签与ci_note，补充分析协议另记为e1_5_star_saved_node_support_v1。',
        'Support结果仅为条件关联，中心处理改变同时改变事实输入和模型条件化，不能解释为Support的因果效应。不合并中心状态估计独立重复方差，不筛选异常机制。原正式覆盖率不足的结论保持不变，本分析不改进或校准区间。',
        '三方法分别汇总；全部400上下文、1200方法—上下文均保留。没有新增节点后验区间。输出仅在本新目录，不自动修改或加入PR。',
        '复用：在已有兼容Python环境中运行 analyze_node_support.py --input <原正式目录> --output <全新目录>。不安装依赖；输出已存在时拒绝覆盖（仅允许首次准备目录中含本脚本、测试和准备记录的情况）。详见input_provenance.json、analysis_manifest.json和acceptance_report.json。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.input.resolve();out=args.output.resolve();started=time.perf_counter()
    require(out!=run and run not in out.parents,'Never write in the original run')
    allowed={'analyze_node_support.py','test_node_support.py','preparation_protection.json','test_report.json'}
    if out.exists():
        require(out==Path(__file__).resolve().parent and {p.name for p in out.iterdir()}<=allowed,
                'Output exists: no overwrite/resume')
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name in ('analyze_node_support.py','test_node_support.py'):
            shutil.copyfile(Path(__file__).parent/name,out/name)
    m,e,hashes,functions,tables,summaries=analyze(run)
    require(not any(name=='pfn_social' or name.startswith('pfn_social.') for name in sys.modules),
            'Unexpected project module import')
    for name,rows in tables.items():write_csv(out/(name+'.csv'),rows)
    write_json(out/'frozen_support_summary.json',summaries)
    write_json(out/'input_provenance.json',dict(input_directory=str(run),input_file_sha256=hashes,
        input_protocol=m['protocol'],plan=m['plan'],frozen_functions=functions,
        model_asset_hashes_recorded_not_loaded=m['assets_sha256']))
    with (out/'ANALYSIS.md').open('x',encoding='utf-8',newline='\n') as f:f.write(markdown_report(tables))
    csv_rows={}
    for name,expected in tables.items():
        with (out/(name+'.csv')).open(encoding='utf-8',newline='') as f:actual=list(csv.DictReader(f))
        require(len(actual)==len(expected),'Saved CSV count mismatch')
        for a,b in zip(actual,expected):
            for key,v in b.items():
                if isinstance(v,(float,np.floating)):close(float(a[key]),v,'CSV numeric mismatch')
                else:require(a[key]==str(v),'CSV value mismatch')
        csv_rows[name]=len(actual)
    require(all(digest(run/n)==h for n,h in hashes.items()),'Source input changed before acceptance')
    acceptance=dict(passed=True,protocol=PROTOCOL,mechanisms=20,pairs=200,contexts=400,
        method_contexts=1200,methods=list(METHODS),csv_rows=csv_rows,input_hashes_verified=len(hashes),
        frozen_metric_cross_checks=True,paired_inputs_shared=True,missing_or_failed_records_dropped=0,
        no_predictor_dgp_or_validate_called=True,project_modules_imported=[],
        bootstrap_seed=2026102502,bootstrap_draws=10000,bootstrap_unit='whole mechanism',
        wall_seconds=time.perf_counter()-started)
    write_json(out/'acceptance_report.json',acceptance)
    inventory={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'analysis_manifest.json',dict(status='complete',protocol=PROTOCOL,
        input_manifest_sha256=hashes['run_manifest.json'],python=platform.python_version(),
        numpy=np.__version__,command_line=sys.argv,frozen_functions=functions,
        output_sha256=inventory,acceptance=acceptance))
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':
    main()
