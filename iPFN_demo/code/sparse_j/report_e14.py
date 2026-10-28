"""Supplement unchanged make_figures.py / make_table.py with RGG diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from make_table import ate_table, read_results, write_csv


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def supplement(directory):
    directory = Path(directory)
    output = directory/'diagnostics'
    output.mkdir(exist_ok=False)
    rows, _ = read_results(directory)
    graph_rows = read_csv(directory/'graph_diagnostics.csv')
    protocol = json.loads((directory/'protocol.json').read_text())
    scenarios = protocol['scenarios']
    by_family = []
    for scenario in scenarios:
        family = scenario['setting_id']
        for observed in (20, 50, 100):
            for row in ate_table([r for r in rows if r['family'] == family], observed):
                by_family.append(dict(family=family, distribution=scenario['distribution'], radius=scenario['radius'], **row))
    write_csv(output/'ate_by_scenario.csv', by_family)

    # A coverage check for each setting, budget and effect, not just a pooled rate.
    summaries = read_csv(directory/'evaluation'/'summary_by_family.csv')
    coverage = []
    for row in summaries:
        if row['method'] != 'j':
            continue
        rate = float(row['coverage_including_failures'])
        coverage.append(dict(family=int(row['family']), m_observed=int(row['m_observed']), effect=row['effect'],
            covered=int(row['covered']), requested=int(row['interval_requested']), coverage=rate,
            mechanism_cluster_se=float(row['coverage_including_failures_se']),
            mean_width=float(row['mean_width']), meets_90_percent=bool(rate >= .90)))
    write_csv(output/'coverage_check.csv', coverage)

    ite = read_csv(directory/'evaluation'/'ite_context_metrics.csv')
    groups = defaultdict(list)
    for row in ite:
        groups[int(row['family']), int(row['m_observed']), row['method'], row['effect']].append(row)
    ite_summary = []
    for (family, observed, method, effect), group in sorted(groups.items()):
        valid = [r for r in group if r['supported'] == 'True']
        n = sum(int(r['node_count']) for r in valid)
        ite_summary.append(dict(family=family, m_observed=observed, method=method, effect=effect,
            valid_contexts=len(valid), contexts=len(group),
            rmse=float(np.sqrt(sum(float(r['squared_error_sum']) for r in valid)/n)) if n else np.nan,
            signed_bias=sum(float(r['error_sum']) for r in valid)/n if n else np.nan))
    write_csv(output/'ite_summary.csv', ite_summary)

    oracle = read_csv(directory/'evaluation'/'oracle_context_metrics.csv')
    groups = defaultdict(list)
    for row in oracle:
        groups[int(row['family']), int(row['m_observed']), row['effect']].append(row)
    oracle_summary = []
    for (family, observed, effect), group in sorted(groups.items()):
        errors = np.array([float(r['ate_error']) for r in group])
        oracle_summary.append(dict(family=family, m_observed=observed, effect=effect,
            contexts=len(group), rmse=float(np.sqrt(np.mean(errors**2))), signed_bias=float(errors.mean()),
            coverage=float(np.mean([r['covered'] == 'True' for r in group])),
            mean_width=float(np.mean([float(r['width']) for r in group])),
            ite_rmse=float(np.sqrt(np.mean([float(r['ite_mse']) for r in group])))))
    write_csv(output/'oracle_summary.csv', oracle_summary)

    # Fair comparisons restricted to the intersection of valid contexts for J
    # and each baseline; support counts make selection explicit.
    lookup = {(r['family'], r['context_id'], r['effect'], r['method']): r for r in rows}
    paired = []
    for family in sorted({r['family'] for r in rows}):
        for observed in (20, 50, 100):
            for effect in ('direct', 'spillover', 'total'):
                js = [r for r in rows if r['family'] == family and r['m_observed'] == observed
                      and r['effect'] == effect and r['method'] == 'j']
                for method in protocol['methods']:
                    if method == 'j':
                        continue
                    pairs = [(r, lookup[family, r['context_id'], effect, method]) for r in js]
                    pairs = [(j, b) for j, b in pairs if j['point_valid'] == b['point_valid'] == 1]
                    paired.append(dict(family=family, m_observed=observed, effect=effect, baseline=method,
                        shared_contexts=len(pairs), expected_contexts=len(js),
                        j_rmse=float(np.sqrt(np.mean([j['squared_error_sum'] for j, b in pairs]))) if pairs else np.nan,
                        baseline_rmse=float(np.sqrt(np.mean([b['squared_error_sum'] for j, b in pairs]))) if pairs else np.nan))
    write_csv(output/'paired_baseline_comparison.csv', paired)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        family = scenario['setting_id']
        with np.load(directory/'graphs'/f'f{family}_m0.npz') as data:
            graph = nx.from_numpy_array(data['adjacency'])
            pos = data['positions']
            nx.draw_networkx_edges(graph, pos, ax=ax, width=.3, alpha=.16)
            ax.scatter(*pos.T, s=7, c='#197f8a')
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal',
                title=f"{scenario['distribution'].title()}, r={scenario['radius']:.2f}\n"
                      f"mean degree={np.mean(data['adjacency'].sum(1)):.1f}")
    fig.suptitle('RGG topology examples (mechanism 0; no added edges)')
    save_figure(fig, output, 'rgg_examples')

    effects = ('direct', 'spillover', 'total')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, effect in zip(axes, effects):
        matrix = np.array([[next(r['coverage'] for r in coverage if
            (r['family'], r['m_observed'], r['effect']) == (s['setting_id'], m, effect))
            for m in (20, 50, 100)] for s in scenarios])
        im = ax.imshow(matrix, vmin=.8, vmax=1., cmap='RdYlGn', aspect='auto')
        for i in range(len(scenarios)):
            for j in range(3):
                ax.text(j, i, f'{100*matrix[i,j]:.1f}%', ha='center', va='center', color='#17202a')
        ax.set(xticks=range(3), xticklabels=[20, 50, 100], xlabel='Observed outcomes M',
            yticks=range(len(scenarios)), yticklabels=[f"{s['distribution']}, r={s['radius']:.2f}" for s in scenarios],
            title=effect.title())
    fig.colorbar(im, ax=list(axes), shrink=.75, label='Empirical coverage')
    fig.suptitle('Frozen J: nominal 95% ATE intervals (target: at least 90%)')
    save_figure(fig, output, 'coverage_by_scenario')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for d, distribution in enumerate(('uniform', 'clustered')):
        for k, effect in enumerate(effects):
            ax = axes[d, k]
            for method, style in [('j', 'o-'), ('ht', 's--'), ('kernel-new', '^--'), ('kernel-old', 'v:')]:
                if method not in protocol['methods']:
                    continue
                values = [r for r in by_family if r['distribution'] == distribution and
                          r['m_observed'] == 50 and r['effect'] == effect and r['method'] == method]
                ax.plot([r['radius'] for r in values], [r['rmse'] for r in values], style, label=method)
            ax.set(title=f'{distribution.title()} / {effect}', xlabel='Radius', ylabel='ATE RMSE', xticks=[.12,.18,.24])
            ax.legend(fontsize=8)
    fig.suptitle('Radius sensitivity at M=50; identical factual inputs across methods')
    save_figure(fig, output, 'radius_sensitivity')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    supplement(parser.parse_args().input)
