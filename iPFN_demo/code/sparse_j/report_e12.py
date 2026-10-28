"""Family-specific E1.2 tables and coverage plot, using existing report metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_table import read_results, ate_table, write_csv, EFFECTS
from pfn_social.e12_configuration import FAMILY_BASE_IDS, PROTOCOL

LABELS = {0: 'WS k4', 2: 'WS k8', 3: 'ER k4', 4: 'SBM k4',
          100: 'Config k4 CV0', 101: 'Config k4 CV0.25', 102: 'Config k4 CV0.5',
          103: 'Config k8 CV0', 104: 'Config k8 CV0.25', 105: 'Config k8 CV0.5'}


def family_tables(rows, cluster_rows):
    lookup = {(int(r['family']), r['method'], int(r['m_observed']), r['effect']): r
              for r in cluster_rows}
    records = []
    for family in sorted({r['family'] for r in rows}):
        subset = [r for r in rows if r['family'] == family]
        for budget in sorted({r['m_observed'] for r in subset}):
            for record in ate_table(subset, budget):
                extra = lookup[family, record['method'], budget, record['effect']]
                records.append(dict(family=family, graph_label=LABELS[family], **record,
                    signed_bias=float(extra['bias']),
                    coverage_cluster_se=float(extra['coverage_including_failures_se'])))
    return records


def coverage_figure(records):
    rows = [r for r in records if r['method'] == 'j']
    families = sorted({r['family'] for r in rows})
    budgets = sorted({r['m_observed'] for r in rows})
    values = np.array([[next(r['coverage_including_failures'] for r in rows
                       if (r['family'], r['m_observed'], r['effect']) == (f, m, e))
                       for m in budgets for e in EFFECTS] for f in families])
    fig, ax = plt.subplots(figsize=(10, max(3, len(families)*.43+1.7)))
    heat = ax.imshow(values, vmin=.75, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_yticks(range(len(families)), [LABELS[f] for f in families])
    ax.set_xticks(range(3*len(budgets)), [f'm={m}\n{e}' for m in budgets for e in EFFECTS])
    for i in range(len(families)):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f'{100*v:.1f}%' + ('*' if v < .90 else ''), ha='center', va='center', fontsize=8)
    ax.set_title('Frozen iPFN-J: empirical coverage of 95% ATE intervals\n* below 90% target; requested failures count as misses', pad=12)
    fig.colorbar(heat, ax=ax, fraction=.025, pad=.03, label='Coverage')
    fig.tight_layout()
    return fig


def accuracy_figure(records):
    rows = [r for r in records if r['family'] in FAMILY_BASE_IDS]
    if not rows:
        return None
    budgets = sorted({r['m_observed'] for r in rows})
    budget = 50 if 50 in budgets else budgets[0]
    rows = [r for r in rows if r['m_observed'] == budget]
    from make_table import method_order, METHOD_NAMES
    methods = method_order(rows)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for i, bases in enumerate(((100, 101, 102), (103, 104, 105))):
        for j, effect in enumerate(EFFECTS):
            ax = axes[i, j]
            for method in methods:
                points = [(f, next((r['rmse'] if r['ranked'] else np.nan for r in rows if
                    (r['family'], r['method'], r['effect']) == (f, method, effect)), np.nan))
                    for f in bases]
                ax.plot([0, .25, .5], [v for _, v in points], marker='o', markersize=3,
                        label=METHOD_NAMES.get(method, method), linewidth=1.3)
            ax.set_title(f'Mean degree {4 if i == 0 else 8}: {effect}')
            ax.set_xticks([0, .25, .5])
            ax.set_xlabel('Degree CV')
            ax.set_ylabel('ATE RMSE')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(4, len(methods)), fontsize=8)
    fig.suptitle(f'Configuration graph comparison, m={budget}\nIndependent mechanisms per family; incomplete-support cells omitted (see CSV)', fontsize=11)
    fig.tight_layout(rect=(0, .09, 1, .91))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input/'execution.json').read_text())
    protocol = manifest['extension_protocol']
    if protocol['name'] != PROTOCOL or manifest['interval_level'] != .95:
        raise ValueError('This report requires E1.2 v1 and nominal 95% intervals')
    rows, digest = read_results(args.input)
    # Verify the registered Cartesian grid, including failed method receipts.
    expected = {(f'f{f}_m{m}_r{r}_observed{k}', method)
                for f in protocol['families'] for m in protocol['mechanism_ids']
                for r in protocol['repeat_ids'] for k in protocol['observed_outcomes']
                for method in manifest['methods']}
    actual = {(r['context_id'], r['method']) for r in manifest['receipts']}
    if expected != actual or len(actual) != len(manifest['receipts']):
        raise ValueError('Missing or duplicate registered method/context receipts')
    with (args.input/'evaluation'/'summary_by_family.csv').open(newline='') as handle:
        records = family_tables(rows, list(csv.DictReader(handle)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'by_family.csv', records)
    configuration_rows = [r for r in rows if r['family'] in FAMILY_BASE_IDS]
    for budget in sorted({r['m_observed'] for r in configuration_rows}):
        write_csv(args.output/f'configuration_pooled_m{budget}.csv', ate_table(configuration_rows, budget))
    j_rows = [r for r in records if r['method'] == 'j' and r['family'] in FAMILY_BASE_IDS]
    assessment = dict(context_metrics_sha256=digest, method_contexts=len(expected),
        prediction_failures=sum(bool(r['failure']) for r in manifest['receipts']),
        interval_failures=sum(bool(r['interval_failure']) for r in manifest['receipts']),
        target=.90, nominal=.95, evaluated_configuration_cells=len(j_rows),
        cells_meeting_target=sum(r['coverage_including_failures'] >= .90 for r in j_rows),
        all_configuration_cells_meet_target=(all(r['coverage_including_failures'] >= .90 for r in j_rows)
                                             if j_rows else None),
        caveat='Empirical rates only; 8 mechanisms per full-run family, repeats share mechanisms; not a coverage guarantee.')
    (args.output/'assessment.json').write_text(json.dumps(assessment, indent=2))
    if any(r['method'] == 'j' for r in records):
        fig = coverage_figure(records)
        fig.savefig(args.output/'coverage.png', dpi=200, bbox_inches='tight')
        fig.savefig(args.output/'coverage.svg', bbox_inches='tight')
        plt.close(fig)
    fig = accuracy_figure(records)
    if fig is not None:
        fig.savefig(args.output/'configuration_accuracy.png', dpi=200, bbox_inches='tight')
        fig.savefig(args.output/'configuration_accuracy.svg', bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == '__main__':
    main()
