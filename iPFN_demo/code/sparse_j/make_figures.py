"""Plot paper-style RMSE and interval coverage from benchmark evaluation records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np

from make_table import EFFECTS, METHOD_NAMES, method_order, read_results, write_csv

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': .6, 'ytick.major.width': .6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300,
})
BLUE, GREY, PURPLE, RED = '#2166AC', '#999999', '#762A83', '#B2182B'


def aggregate(rows, metric):
    if metric == 'rmse':
        count = sum(r['point_valid'] for r in rows)
        return float(np.sqrt(sum(r['squared_error_sum'] for r in rows) / count)) if count else np.nan
    count = sum(r['interval_requested'] for r in rows)
    return sum(r['covered'] for r in rows) / count if count else np.nan


def build_figure(rows, metric):
    methods = method_order(rows)
    level = rows[0]['interval_level']
    budgets = sorted({r['m_observed'] for r in rows})
    contexts = {r['context_id'] for r in rows}
    mechanisms = {(r['family'], r['mechanism']) for r in rows}
    mechanism_contexts = defaultdict(set)
    for row in rows:
        mechanism_contexts[row['family'], row['mechanism']].add(row['context_id'])
    height = max(135, 95 + 13 * len(methods))
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, height / 25.4), sharey=True)
    exports = []
    prefix = 'point' if metric == 'rmse' else 'interval'
    for li, estimand in enumerate(('ITE', 'ATE')):
        for ei, effect in enumerate(EFFECTS):
            ax = axes[li, ei]
            all_values = []
            for j, method in enumerate(methods):
                members = [r for r in rows if (r['method'], r['estimand'], r['effect'])
                           == (method, estimand, effect)]
                groups = defaultdict(list)
                for row in members:
                    groups[row['family'], row['mechanism']].append(row)
                estimate = aggregate(members, metric)
                requested = sum(r[prefix + '_requested'] for r in members)
                valid = sum(r[prefix + '_valid'] for r in members)
                partial = requested > 0 and (valid < requested
                    or any(r[prefix + '_requested'] == 0 for r in members)
                    or {r['context_id'] for r in members} != contexts)
                base = dict(method=method, estimand=estimand, effect=effect, metric=metric)
                exports.append(dict(**base, scope='pooled', family='', mechanism='', estimate=estimate,
                    requested=requested, valid=valid, contexts=len(members), partial_support=partial))
                values = []
                for (family, mechanism), group in sorted(groups.items()):
                    value = aggregate(group, metric)
                    if np.isfinite(value):
                        values.append(value)
                    group_requested = sum(r[prefix + '_requested'] for r in group)
                    group_valid = sum(r[prefix + '_valid'] for r in group)
                    exports.append(dict(**base, scope='mechanism', family=family, mechanism=mechanism,
                        estimate=value, requested=group_requested, valid=group_valid, contexts=len(group),
                        partial_support=group_requested > 0 and (group_valid < group_requested
                            or any(r[prefix + '_requested'] == 0 for r in group)
                            or {r['context_id'] for r in group} != mechanism_contexts[family, mechanism])))
                y = len(methods) - 1 - j
                color = BLUE if method == 'j' else PURPLE if method in ('hinet', 'hypersci', 'tnet') else GREY
                if np.isfinite(estimate):
                    offsets = np.linspace(-.15, .15, len(values)) if len(values) > 1 else np.zeros(len(values))
                    ax.scatter(values, y + offsets, s=7, color=color, alpha=.35, linewidths=0)
                    ax.scatter([estimate], [y], marker='D', s=25,
                               facecolors='none' if partial else color, edgecolors=color, linewidths=.8, zorder=3)
                    all_values.extend(values + [estimate])
                else:
                    ax.text(.5, y, 'Unsupported' if not requested else 'No valid estimate',
                            transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=6)
            if metric == 'rmse':
                # A perfect estimate is kept at zero on a linear axis, never clipped or shifted.
                use_log = bool(all_values) and min(all_values) > 0
                ax.set_xscale('log' if use_log else 'linear')
                if use_log:
                    limits = (min(all_values) / 1.15, max(all_values) * 1.15)
                    ax.set_xlim(*limits)
                    ax.set_xticks(np.geomspace(*limits, num=3))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.set_xlim(left=0)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.3g}'))
                ax.set_xlabel('RMSE (log scale)' if use_log else 'RMSE')
            else:
                ax.axvline(.90, color=RED, linestyle='--', linewidth=.8, zorder=0)
                ax.axvline(level, color=GREY, linestyle=':', linewidth=.8, zorder=0)
                ax.set_xlim(-.04, 1.04)
                ax.set_xticks([0, .5, 1], ['0%', '50%', '100%'])
                ax.set_xlabel('Coverage')
            ax.set_ylim(-.6, len(methods) - .4)
            ax.set_title(f'{chr(97 + li * 3 + ei)}  {estimand} / {effect.title()}', loc='left', weight='bold')
            ax.set_yticks(range(len(methods)), [METHOD_NAMES.get(m, m) for m in methods[::-1]], fontsize=6.3)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.235, right=.985, top=.86, bottom=.26, wspace=.30, hspace=.55)
    title = 'Accuracy across recorded contexts' if metric == 'rmse' else f'{100 * level:g}% interval coverage'
    fig.text(.02, .97, title, weight='bold', fontsize=10)
    fig.text(.02, .925, f'{len(contexts)} contexts; {len(mechanisms)} independent mechanisms; '
             'observed outcomes m = ' + '/'.join(map(str, budgets)) + '.', fontsize=6.5)
    handles = [Line2D([], [], marker='o', color=GREY, linestyle='', markersize=3, label='Mechanism'),
               Line2D([], [], marker='D', color=BLUE, linestyle='', markersize=4, label='Pooled'),
               Line2D([], [], marker='D', markerfacecolor='none', color=GREY, linestyle='', markersize=4,
                      label='Incomplete support')]
    if metric != 'rmse':
        handles += [Line2D([], [], color=RED, linestyle='--', linewidth=.8, label='90% threshold'),
                    Line2D([], [], color=GREY, linestyle=':', linewidth=.8, label=f'{100 * level:g}% nominal')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3, fontsize=6.3)
    note = ('RMSE pools squared errors across recorded nodes and repeats before taking the square root.'
            if metric == 'rmse' else 'Coverage includes failed requested intervals in the denominator; unsupported intervals are excluded.')
    fig.text(.02, .075, note, fontsize=6)
    fig.text(.02, .035, 'Small points show mechanism summaries, not confidence limits. Counts and missing support are exported to CSV.', fontsize=6)
    return fig, exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Benchmark output directory')
    parser.add_argument('--output', type=Path, required=True, help='New directory for figures and source CSV')
    parser.add_argument('--observed', type=int, help='Optional observed-outcome budget')
    args = parser.parse_args()
    rows, digest = read_results(args.input)
    if args.observed is not None:
        rows = [r for r in rows if r['m_observed'] == args.observed]
        if not rows:
            parser.error('The requested observed-outcome budget is absent.')
    args.output.mkdir(parents=True, exist_ok=False)
    for metric, filename in (('rmse', 'accuracy'), ('coverage', 'coverage')):
        fig, records = build_figure(rows, metric)
        fig.savefig(args.output / f'{filename}.pdf', bbox_inches='tight')
        fig.savefig(args.output / f'{filename}.svg', bbox_inches='tight')
        fig.savefig(args.output / f'{filename}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        write_csv(args.output / f'{filename}_source.csv', records)
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        context_metrics_sha256=digest, observed_budgets=sorted({r['m_observed'] for r in rows}),
        interval_level=rows[0]['interval_level'], coverage_denominator='all requested intervals, including failures',
        accuracy='Pooled valid squared errors; hollow diamonds mark incomplete support.',
        points='Mechanism summaries, not confidence limits.'), indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
