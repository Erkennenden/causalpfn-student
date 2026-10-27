"""Score saved causal effects and marginal intervals against separate simulation truth."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

EFFECTS = ('direct', 'spillover', 'total')
GROUP_KEYS = ('method', 'm_observed', 'estimand', 'effect', 'interval_level')
SUM_KEYS = ('point_requested', 'point_valid', 'error_sum', 'squared_error_sum',
            'interval_requested', 'interval_valid', 'covered', 'width_sum')


def context_metadata(identity):
    match = re.fullmatch(r'f(\d+)_m(\d+)_r(\d+)_observed(\d+)', identity)
    if not match:
        raise ValueError(f'Unrecognized context_id: {identity}')
    return dict(zip(('family', 'mechanism', 'repeat', 'm_observed'), map(int, match.groups())))


def _array(prediction, key, shape):
    try:
        value = np.asarray(prediction.get(key, np.full(shape, np.nan)), dtype=float)
    except (TypeError, ValueError):
        return np.full(shape, np.nan)
    return value if value.shape == shape else np.full(shape, np.nan)


def _metrics(row):
    def ratio(numerator, denominator):
        return row[numerator] / row[denominator] if row[denominator] else float('nan')
    row.update(bias=ratio('error_sum', 'point_valid'),
               rmse=np.sqrt(ratio('squared_error_sum', 'point_valid')),
               coverage=ratio('covered', 'interval_valid'),
               coverage_including_failures=ratio('covered', 'interval_requested'),
               mean_width=ratio('width_sum', 'interval_valid'),
               interval_success_rate=ratio('interval_valid', 'interval_requested'))
    return row


def evaluate_context(truth, prediction, receipt):
    """Six rows per context; invalid supported intervals remain in denominators.

    Coverage is marginal across nodes for ITE, across simulation draws for ATE.
    Unsupported estimands are explicitly excluded, never treated as zero error.
    """
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 4 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError('Require finite truth arm_means[N,4]')
    effects = np.stack((truth[:, 2]-truth[:, 0], truth[:, 3]-truth[:, 2], truth[:, 3]-truth[:, 0]), -1)
    metadata = context_metadata(receipt['context_id'])
    level = float(receipt.get('interval_level', prediction.get('interval_level', .95)))
    if not 0 < level < 1:
        raise ValueError('Require 0 < interval_level < 1')
    if 'interval_level' in prediction and not np.isclose(float(prediction['interval_level']), level, rtol=0, atol=1e-12):
        raise ValueError(f"Interval level disagrees with execution.json: {receipt['context_id']} / {receipt['method']}")
    rows = []
    for estimand, target in (('ITE', effects), ('ATE', effects.mean(0, keepdims=True))):
        prefix = estimand.lower()
        shape = effects.shape if estimand == 'ITE' else (3,)
        point = _array(prediction, prefix+'_effects', shape).reshape(-1, 3)
        lo = _array(prediction, prefix+'_lower', shape).reshape(-1, 3)
        hi = _array(prediction, prefix+'_upper', shape).reshape(-1, 3)
        point_support = receipt.get(prefix+'_supported', prediction.get(prefix+'_supported', [False]*3))
        support = receipt.get(prefix+'_interval_supported', prediction.get(prefix+'_interval_supported', [False]*3))
        for index, effect in enumerate(EFFECTS):
            valid_point = np.isfinite(point[:, index]) & bool(point_support[index]) & (not bool(receipt.get('failure')))
            valid = (np.isfinite(lo[:, index]) & np.isfinite(hi[:, index])
                     & (lo[:, index] <= hi[:, index]) & bool(support[index])
                     & (not bool(receipt.get('interval_failure'))))
            covered = valid & (lo[:, index] <= target[:, index]) & (target[:, index] <= hi[:, index])
            error = point[valid_point, index]-target[valid_point, index]
            requested = len(target) if support[index] else 0
            row = dict(**metadata, context_id=receipt['context_id'], method=receipt['method'],
                       estimand=estimand, effect=effect, interval_level=level,
                       point_requested=len(target) if point_support[index] else 0,
                       point_valid=int(valid_point.sum()), error_sum=float(error.sum()),
                       squared_error_sum=float(np.square(error).sum()),
                       interval_requested=requested, interval_valid=int(valid.sum()), covered=int(covered.sum()),
                       width_sum=float((hi[valid, index]-lo[valid, index]).sum()),
                       prediction_failure=bool(receipt.get('failure')),
                       interval_status=('unsupported' if not requested else 'ok' if valid.all() else 'failed_or_invalid'))
            rows.append(_metrics(row))
    return rows


def _cluster_se(rows, denominator):
    """Cluster ratio SE: independent mechanisms, dependent nodes/repeats within.

    This is an approximate Monte Carlo SE, not a node-wise binomial SE.
    Fewer than two contributing mechanisms yields NaN.
    """
    clusters = defaultdict(lambda: np.zeros(2))
    for row in rows:
        if row[denominator]:
            clusters[(row['family'], row['mechanism'])] += [row['covered'], row[denominator]]
    if len(clusters) < 2:
        return float('nan')
    values = np.array(list(clusters.values()))
    hits, counts = values.T
    rate = hits.sum()/counts.sum()
    k = len(counts)
    return float(np.sqrt(k/(k-1)*np.square(hits-rate*counts).sum())/counts.sum())


def summarize(rows, extra_keys=()):
    keys = (*extra_keys, *GROUP_KEYS)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, members in sorted(groups.items()):
        summary = dict(zip(keys, identity))
        summary.update({key: sum(row[key] for row in members) for key in SUM_KEYS})
        summary.update(contexts=len(members), mechanisms=len({(r['family'], r['mechanism']) for r in members}),
                       prediction_failure_contexts=sum(r['prediction_failure'] for r in members),
                       interval_failure_contexts=sum(r['interval_status']=='failed_or_invalid' for r in members),
                       interval_unsupported_contexts=sum(r['interval_status']=='unsupported' for r in members),
                       coverage_se=_cluster_se(members, 'interval_valid'),
                       coverage_including_failures_se=_cluster_se(members, 'interval_requested'))
        result.append(_metrics(summary))
    return result


def _write_csv(path, rows):
    if not rows:
        raise ValueError('No evaluation rows')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'execution.json').read_text())
    rows, seen = [], set()
    for receipt in manifest['receipts']:
        identity = (receipt['context_id'], receipt['method'])
        if identity in seen:
            raise ValueError(f'Duplicate receipt: {identity}')
        seen.add(identity)
        metadata = context_metadata(receipt['context_id'])
        with np.load(directory/'truth'/f"f{metadata['family']}_m{metadata['mechanism']}.npz", allow_pickle=False) as saved:
            truth = saved['arm_means']
        path = directory/'predictions'/('__'.join(identity)+'.npz')
        try:
            with np.load(path, allow_pickle=False) as saved:
                prediction = {key: saved[key] for key in saved.files}
        except (OSError, ValueError, BadZipFile, EOFError) as error:
            prediction = {}
            receipt = dict(receipt, failure={'type': type(error).__name__, 'message': str(error)},
                           interval_failure={'type': type(error).__name__, 'message': str(error)})
        rows.extend(evaluate_context(truth, prediction, receipt))
    output = directory/'evaluation'
    output.mkdir(exist_ok=True)
    for name, data in (('context_metrics', rows), ('summary', summarize(rows)),
                       ('summary_by_family', summarize(rows, ('family',))),
                       ('summary_by_mechanism', summarize(rows, ('family', 'mechanism')))):
        _write_csv(output/(name+'.csv'), data)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Benchmark output directory')
    args = parser.parse_args()
    print(evaluate_run(args.input))


if __name__ == '__main__':
    main()
