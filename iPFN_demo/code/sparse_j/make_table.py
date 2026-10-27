"""Build paper-style ATE tables from benchmark evaluation records."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

EFFECTS = ('direct', 'spillover', 'total')
METHOD_NAMES = {
    'j': 'iPFN-J', 'ht': 'HT', 'hajek': 'Hajek', 'f': 'F (raw20)',
    'lin': 'Lin (raw20)', 'f-z2': 'F-z2', 'lin-z2': 'Lin-z2',
    'non-interference': 'Non-interference OLS', 'kernel-old': 'Kernel prior',
    'kernel-new': 'Calibrated kernel', 'causalpfn': 'CausalPFN',
    'hinet': 'HINet', 'tnet': 'Stabilized TNet', 'hypersci': 'HyperSCI',
}
INTEGER_FIELDS = ('family', 'mechanism', 'repeat', 'm_observed', 'point_requested',
                  'point_valid', 'interval_requested', 'interval_valid', 'covered')
FLOAT_FIELDS = ('error_sum', 'squared_error_sum', 'interval_level')
METRICS = ('absolute_conditional_bias', 'variance_unbiased', 'rmse')


def read_results(directory):
    """Read three ATE rows per receipt, ignoring ITE rows in older reports."""
    directory = Path(directory)
    path = directory / 'evaluation' / 'context_metrics.csv'
    manifest = json.loads((directory / 'execution.json').read_text(encoding='utf-8'))
    with path.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['estimand'] != 'ITE']
    expected = {(r['context_id'], r['method'], 'ATE', effect)
                for r in manifest['receipts'] for effect in EFFECTS}
    identities = [(r['context_id'], r['method'], r['estimand'], r['effect']) for r in rows]
    if not rows or len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError('Evaluation rows are missing or duplicated; rerun evaluate.py.')
    for row in rows:
        for key in INTEGER_FIELDS:
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
        if row['prediction_failure'] not in ('True', 'False'):
            raise ValueError('Invalid prediction_failure flag; rerun evaluate.py.')
        row['prediction_failure'] = row['prediction_failure'] == 'True'
        if not (0 <= row['point_valid'] <= row['point_requested']
                and 0 <= row['covered'] <= row['interval_valid'] <= row['interval_requested']
                and np.isfinite([row[k] for k in FLOAT_FIELDS]).all()
                and row['squared_error_sum'] >= 0 and 0 < row['interval_level'] < 1):
            raise ValueError('Invalid evaluation counts or values; rerun evaluate.py.')
        if row['estimand'] == 'ATE' and row['point_requested'] not in (0, 1):
            raise ValueError('ATE evaluation must have at most one point per context.')
    if len({r['interval_level'] for r in rows}) != 1:
        raise ValueError('Use one nominal interval level per report.')
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_order(rows):
    methods = {row['method'] for row in rows}
    return [m for m in METHOD_NAMES if m in methods] + sorted(methods - METHOD_NAMES.keys())


def ate_table(rows, observed):
    """Use complete mechanisms; partial support is explicit and never ranked.

    Bias averages |mean(error)| within each mechanism. Variance averages
    sample error variance (R-1 divisor). RMSE pools squared errors before
    taking the root. Mechanisms receive equal weight, as in the paper table.
    Coverage retains every requested interval, including failed intervals.
    """
    selected = [r for r in rows if r['estimand'] == 'ATE' and r['m_observed'] == observed]
    expected = defaultdict(set)
    for row in selected:
        expected[row['family'], row['mechanism']].add(row['context_id'])
    result = []
    for method in method_order(selected):
        for effect in EFFECTS:
            members = [r for r in selected if r['method'] == method and r['effect'] == effect]
            groups = defaultdict(list)
            for row in members:
                groups[row['family'], row['mechanism']].append(row)
            complete = [group for key, group in groups.items()
                        if {r['context_id'] for r in group} == expected[key]
                        and all(r['point_requested'] == r['point_valid'] == 1 for r in group)]
            errors = [np.array([r['error_sum'] for r in group]) for group in complete]
            requested = sum(r['interval_requested'] for r in members)
            valid = sum(r['interval_valid'] for r in members)
            hits = sum(r['covered'] for r in members)
            result.append(dict(method=method, m_observed=observed, effect=effect,
                absolute_conditional_bias=float(np.mean([abs(e.mean()) for e in errors])) if errors else np.nan,
                variance_unbiased=float(np.mean([e.var(ddof=1) for e in errors]))
                    if errors and all(len(e) > 1 for e in errors) else np.nan,
                rmse=float(np.sqrt(np.mean([np.mean([r['squared_error_sum'] for r in g])
                                            for g in complete]))) if complete else np.nan,
                mechanisms_used=len(complete), mechanisms_expected=len(expected),
                contexts_used=sum(map(len, complete)), contexts_expected=sum(map(len, expected.values())),
                ranked=bool(complete) and len(complete) == len(expected),
                interval_level=members[0]['interval_level'], interval_requested=requested,
                interval_valid=valid, covered=hits,
                coverage=hits / valid if valid else np.nan,
                coverage_including_failures=hits / requested if requested else np.nan))
    return result


def latex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in value)


def render_table(rows, observed):
    lines = [r'\begin{table}[tb]', r'\centering',
        rf'\caption{{ATE accuracy with {observed} observed outcomes.}}',
        r'\begingroup\footnotesize', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l*{9}{r}@{}}', r'\toprule',
        r'&\multicolumn{3}{c}{Direct}&\multicolumn{3}{c}{Treated spillover}&\multicolumn{3}{c}{Total}\\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
        r'Method&$|\mathrm{Bias}|$&Variance&RMSE&$|\mathrm{Bias}|$&Variance&RMSE&$|\mathrm{Bias}|$&Variance&RMSE\\',
        r'\midrule']
    for method in method_order(rows):
        cells = []
        for effect in EFFECTS:
            row = next(r for r in rows if r['method'] == method and r['effect'] == effect)
            for metric in METRICS:
                if not np.isfinite(row[metric]):
                    cells.append(r'\multicolumn{1}{c}{--}')
                    continue
                value = f'{row[metric]:.4f}'
                if not row['ranked']:
                    value += r'$^{\dagger}$'
                else:
                    competitors = [r[metric] for r in rows if r['effect'] == effect
                                   and r['ranked'] and np.isfinite(r[metric])]
                    rank = 1 + sum(v < row[metric] for v in competitors)
                    shade = {1: 24, 2: 17, 3: 11, 4: 7, 5: 4, 6: 2}.get(rank, 1)
                    if rank == 1:
                        value = r'\textbf{' + value + '}'
                    value = r'\cellcolor{blue!' + str(shade) + '}' + value
                cells.append(value)
        lines.append(latex_escape(METHOD_NAMES.get(method, method)) + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}\endgroup', r'\vspace{3pt}',
        r'\begin{minipage}{\linewidth}\scriptsize',
        r'Bias is mean absolute conditional bias; variance uses the $R-1$ divisor within each mechanism.',
        r'Mechanisms are equally weighted; RMSE takes the root after pooling squared errors.',
        r'$^{\dagger}$Incomplete support: values condition on mechanisms with all recorded repeats valid and are unranked.',
        r'-- indicates no complete supported mechanism, or fewer than two repeats for variance.',
        r'Numeric support counts and interval coverage are provided in the accompanying CSV.',
        r'\end{minipage}', r'\end{table}'])
    return '\n'.join(lines) + '\n'


def render_coverage_table(rows, observed):
    level = 100 * rows[0]['interval_level']
    lines = [r'\begin{table}[tb]', r'\centering',
        rf'\caption{{ATE interval coverage at nominal {level:g}\%, with {observed} observed outcomes.}}',
        r'\begin{tabular}{lrrr}', r'\toprule',
        r'Method & Direct & Treated spillover & Total \\', r'\midrule']
    for method in method_order(rows):
        cells = []
        for effect in EFFECTS:
            row = next(r for r in rows if r['method'] == method and r['effect'] == effect)
            value = row['coverage_including_failures']
            cells.append(f'{100 * value:.2f}' + r'\%' if np.isfinite(value) else '--')
        lines.append(latex_escape(METHOD_NAMES.get(method, method)) + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\begin{minipage}{\linewidth}\scriptsize',
        r'Coverage includes failed requested intervals in the denominator. -- means intervals are unsupported.',
        r'These are marginal posterior intervals; nominal coverage is not a frequentist guarantee.',
        r'\end{minipage}', r'\end{table}'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Benchmark output directory')
    parser.add_argument('--output', type=Path, required=True, help='New directory for CSV and LaTeX tables')
    parser.add_argument('--observed', type=int, help='Optional observed-outcome budget')
    args = parser.parse_args()
    rows, digest = read_results(args.input)
    budgets = sorted({r['m_observed'] for r in rows})
    if args.observed is not None:
        if args.observed not in budgets:
            parser.error('The requested observed-outcome budget is absent.')
        budgets = [args.observed]
    args.output.mkdir(parents=True, exist_ok=False)
    for observed in budgets:
        records = ate_table(rows, observed)
        write_csv(args.output / f'ate_m{observed}.csv', records)
        (args.output / f'ate_m{observed}.tex').write_text(
            render_table(records, observed) + '\n' + render_coverage_table(records, observed), encoding='utf-8')
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        context_metrics_sha256=digest, observed_budgets=budgets,
        metrics='Equal mechanism weighting; complete mechanisms only; R-1 sample variance.',
        coverage='Covered / valid intervals; failed intervals excluded.',
        coverage_including_failures='Covered / all requested intervals, including failures; unsupported intervals excluded.'), indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
