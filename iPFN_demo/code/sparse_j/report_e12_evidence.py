"""Render evidence tables and plots without changing predictions or intervals."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_table import write_csv


def read(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    output = root/'figures'
    output.mkdir(exist_ok=True)
    j = read(root/'metrics/j_interval_contexts.csv')
    oracle = read(root/'oracle/interval_contexts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, effect in zip(axes, ('spillover', 'total')):
        for source, name, offset, color in ((j,'Frozen J',-.12,'#2563eb'),
                                           (oracle,'Known-basis oracle',.12,'#d97706')):
            rows = sorted((r for r in source if r['family']=='103' and r['mechanism']=='2'
                           and r['m_observed']=='50' and r['effect']==effect), key=lambda r:int(r['repeat']))
            x = np.array([int(r['repeat']) for r in rows])+offset
            center = np.array([(float(r['lower'])+float(r['upper']))/2 for r in rows])
            half = np.array([float(r['half_width']) for r in rows])
            ax.errorbar(x, center, yerr=half, fmt='o', markersize=4, capsize=2,
                        color=color, label=name, alpha=.85)
        ax.axhline(float(rows[0]['truth']), color='black', linestyle='--', label='True ATE')
        ax.set(title=effect.capitalize(), xlabel='Repeat within the same mechanism', ylabel='ATE and nominal 95% interval', xticks=range(12))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle('Configuration k=8, CV=0 | mechanism 2 | 50 observed outcomes\nPost-hoc diagnostic: both estimators show conditional shrinkage bias')
    fig.savefig(output/'difficult_mechanism_intervals.png',dpi=180)
    fig.savefig(output/'difficult_mechanism_intervals.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, source, title in ((axes[0], j, 'Frozen J'), (axes[1],oracle,'Known-basis oracle')):
        grid = np.empty((8, 3))
        for mid in range(8):
            for col,effect in enumerate(('direct','spillover','total')):
                rows = [r for r in source if r['family']=='103' and int(r['mechanism'])==mid
                        and r['m_observed']=='50' and r['effect']==effect]
                grid[mid,col] = sum(int(r['covered']) for r in rows)
        ax.imshow(grid, vmin=0, vmax=12, cmap='YlGnBu',aspect='auto')
        for row,col in np.ndindex(grid.shape):
            ax.text(col,row,f'{int(grid[row,col])}/12',ha='center',va='center',
                    color='white' if grid[row,col]>8 else 'black')
        ax.set(title=title,xticks=range(3),xticklabels=['Direct','Spillover','Total'],
               yticks=range(8),ylabel='Independent mechanism')
    fig.suptitle('Covered repeats within each fixed mechanism | k=8, CV=0, observed=50')
    fig.savefig(output/'coverage_by_mechanism.png',dpi=180)
    plt.close(fig)

    tables = read(root/'metrics/task_mean_sd.csv')+read(root/'oracle/task_mean_sd.csv')
    write_csv(root/'comparison_f103_observed50.csv', [r for r in tables if r['family']=='103'
              and r['m_observed']=='50'])
    if (root/'replication_metrics/j_interval_by_family.csv').exists():
        original = read(root/'metrics/j_interval_by_family.csv')
        replication = read(root/'replication_metrics/j_interval_by_family.csv')
        fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
        for rows,label,offset,color in ((original,'Original: 8 mechanisms x 12 repeats',-.1,'#d97706'),
                                       (replication,'Replication: 100 mechanisms x 3 repeats',.1,'#2563eb')):
            selected=[next(r for r in rows if r['family']=='103' and r['m_observed']=='50' and r['effect']==effect)
                      for effect in ('direct','spillover','total')]
            cov=np.array([float(r['coverage']) for r in selected])
            lo=np.array([float(r['bootstrap_low']) for r in selected])
            hi=np.array([float(r['bootstrap_high']) for r in selected])
            ax.errorbar(np.arange(3)+offset,cov,yerr=[cov-lo,hi-cov],fmt='o',capsize=5,color=color,label=label)
        ax.axhline(.9,color='black',linestyle='--',label='90% target')
        ax.axhline(.95,color='gray',linestyle=':',label='95% nominal')
        ax.set(xticks=range(3),xticklabels=['Direct','Spillover','Total'],ylabel='Empirical coverage',
               ylim=(.60,1.01),title='Frozen J: same graph family, new independent mechanisms')
        ax.legend(loc='lower left',fontsize=9)
        ax.text(.99,.02,'Bars: descriptive 95% whole-mechanism bootstrap bands',transform=ax.transAxes,
                ha='right',fontsize=8)
        fig.savefig(output/'independent_replication.png',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
