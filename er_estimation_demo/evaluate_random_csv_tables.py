#!/usr/bin/env python3
"""Shared support/table helpers and a compatibility entry for normal evaluation.

The historical C-sampler evaluation is no longer run by this module.
"""
from __future__ import annotations

import math

import numpy as np

EFFECTS=('direct','spillover','total')


def supported(row):
    return bool(row.get('supported',True)) and math.isfinite(float(row['estimate']))


def paired_comparisons(rows):
    def key(r): return int(r['dataset_id']),int(r.get('unit_id',0)),r['effect']
    ref={key(r):r for r in rows if r['method']=='PFN with interference'}
    result=[]
    for name in dict.fromkeys(r['method'] for r in rows if r['method']!='PFN with interference'):
        for effect in EFFECTS:
            candidates=[r for r in rows if r['method']==name and r['effect']==effect]
            if not candidates: continue
            pairs=[(ref[key(r)],r) for r in candidates if key(r) in ref and supported(r) and supported(ref[key(r)])]
            if any(not math.isclose(float(a['truth']),float(b['truth']),abs_tol=1e-6,rel_tol=1e-6) for a,b in pairs):
                raise ValueError('Pairwise comparison contains mismatched causal targets.')
            a=np.array([float(x['estimate'])-float(x['truth']) for x,y in pairs])
            b=np.array([float(y['estimate'])-float(y['truth']) for x,y in pairs])
            result.append(dict(method=name,reference_method='PFN with interference',effect=effect,
                evaluation_subset='pairwise_common_support',n_total=len(candidates),n_paired=len(pairs),
                pfn_rmse=float(np.sqrt(np.mean(a*a))) if len(a) else float('nan'),
                method_rmse=float(np.sqrt(np.mean(b*b))) if len(b) else float('nan'),
                method_minus_pfn_mse=float(np.mean(b*b-a*a)) if len(a) else float('nan')))
    return result


def _table(rows,level):
    lines=[r'\begin{table}[H]',r'\centering',
        r'\caption{Random LPE, fixed-CSV PFN with interference: '+level.upper()+r'}',
        r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Method & Effect & MAE & RMSE & Bias & Mean Truth & Mean Pred \\',r'\midrule']
    for r in rows:
        vals=[f'{float(r[k]):.6f}' if math.isfinite(float(r[k])) else '---'
              for k in ('mae','rmse','bias','mean_truth','mean_estimate')]
        lines.append(' & '.join([r['method'].replace('_',r'\_'),r['effect'].title(),*vals])+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'


def main(argv=None):
    """Compatibility entry: C evaluation is removed; use the normal sampler."""
    from evaluate_matched_old_sampler import main as normal_main
    return normal_main(argv)


if __name__ == '__main__':
    main()
