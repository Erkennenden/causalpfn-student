"""Read-only evidence audit of a saved E1.2 run; no model or interval tuning."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import context_metadata
from make_table import read_results, write_csv, EFFECTS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def task_summaries(rows):
    """Each independent (family, mechanism) is a task; repeats stay within."""
    keys = ('family', 'mechanism', 'method', 'm_observed', 'estimand', 'effect')
    tasks = []
    for identity, members in sorted(group_rows(rows, keys).items()):
        complete = all(r['valid'] for r in members)
        tasks.append(dict(zip(keys, identity), repeats_expected=len(members),
            repeats_valid=sum(r['valid'] for r in members), complete=complete,
            rmse=float(np.sqrt(np.mean([r['mse'] for r in members]))) if complete else np.nan,
            signed_bias=float(np.mean([r['bias'] for r in members])) if complete else np.nan))
    keys = ('family', 'method', 'm_observed', 'estimand', 'effect')
    summaries = []
    for identity, members in sorted(group_rows(tasks, keys).items()):
        valid = [r for r in members if r['complete']]
        rmse = np.array([r['rmse'] for r in valid])
        bias = np.array([r['signed_bias'] for r in valid])
        summaries.append(dict(zip(keys, identity), tasks_expected=len(members),
            tasks_complete=len(valid), complete_support=len(valid) == len(members),
            mean_task_rmse=float(rmse.mean()) if len(valid) else np.nan,
            sd_task_rmse=float(rmse.std(ddof=1)) if len(valid) > 1 else np.nan,
            pooled_rmse=float(np.sqrt(np.square(rmse).mean())) if len(valid) else np.nan,
            mean_task_bias=float(bias.mean()) if len(valid) else np.nan,
            sd_task_bias=float(bias.std(ddof=1)) if len(valid) > 1 else np.nan))
    return tasks, summaries


def raw_context(job):
    directory, receipt, truth = job
    name, context = receipt['method'], receipt['context_id']
    meta = context_metadata(context)
    point_rows, intervals = [], []
    path = directory/'predictions'/f'{context}__{name}.npz'
    true_ite = np.column_stack((truth[:, 2]-truth[:, 0], truth[:, 3]-truth[:, 2], truth[:, 3]-truth[:, 0]))
    with np.load(path, allow_pickle=False) as saved:
        for estimand, key, target, labels, support in (
            ('mu', 'arm_means', truth, ('00', '01', '10', '11'), receipt['arm_supported']),
            ('ITE', 'ite_effects', true_ite, EFFECTS, receipt['ite_supported'])):
            estimate = saved[key]
            if estimate.shape != target.shape:
                raise ValueError(f'Unexpected prediction shape: {path}, {key}')
            for i, label in enumerate(labels):
                valid = bool(support[i]) and not receipt['failure'] and np.isfinite(estimate[:, i]).all()
                error = estimate[:, i]-target[:, i]
                point_rows.append(dict(**meta, context_id=context, method=name, estimand=estimand,
                    effect=label, valid=valid, mse=float(np.mean(error**2)) if valid else np.nan,
                    bias=float(error.mean()) if valid else np.nan))
        if name == 'j':
            point, lo, hi = (saved[k] for k in ('ate_effects', 'ate_lower', 'ate_upper'))
            with np.load(directory/'factual'/f'{context}.npz', allow_pickle=False) as fact:
                a, t, ids = (fact[k] for k in ('adjacency', 'treatment', 'observed_indices'))
            states = 2*t.astype(int)+(a@t > a.sum(1)/2).astype(int)
            counts = np.bincount(states[ids], minlength=4)
            for i, effect in enumerate(EFFECTS):
                true = float(true_ite[:, i].mean())
                center, half = float((lo[i]+hi[i])/2), float((hi[i]-lo[i])/2)
                if half <= 0 or not np.isfinite([point[i], lo[i], hi[i], true]).all():
                    raise ValueError('Invalid J interval in evidence audit')
                arms = ((0, 2), (2, 3), (0, 3))[i]
                intervals.append(dict(**meta, context_id=context, method=name, effect=effect,
                    truth=true, point=float(point[i]), lower=float(lo[i]), upper=float(hi[i]),
                    point_error=float(point[i]-true), center_error=center-true,
                    half_width=half, width=2*half, score=abs(center-true)/half,
                    covered=int(lo[i] <= true <= hi[i]), truth_below=int(true < lo[i]),
                    truth_above=int(true > hi[i]), empty_target_arm=int(any(counts[k] == 0 for k in arms)),
                    **{f'observed_arm_{k:02b}': int(counts[k]) for k in range(4)}))
    return point_rows, intervals


def interval_summaries(rows):
    keys = ('family', 'mechanism', 'm_observed', 'effect')
    mechanisms = []
    for identity, members in sorted(group_rows(rows, keys).items()):
        error = np.array([r['point_error'] for r in members])
        mechanisms.append(dict(zip(keys, identity), repeats=len(members),
            covered=sum(r['covered'] for r in members), coverage=np.mean([r['covered'] for r in members]),
            truth=members[0]['truth'], signed_bias=error.mean(), rmse=np.sqrt(np.mean(error**2)),
            error_sd=error.std(ddof=1) if len(error)>1 else np.nan,
            center_bias=np.mean([r['center_error'] for r in members]),
            mean_width=np.mean([r['width'] for r in members]),
            max_score=max(r['score'] for r in members),
            truth_below=sum(r['truth_below'] for r in members),
            truth_above=sum(r['truth_above'] for r in members),
            empty_target_arm_contexts=sum(r['empty_target_arm'] for r in members)))
    keys = ('family', 'm_observed', 'effect')
    families = []
    for identity, members in sorted(group_rows(mechanisms, keys).items()):
        coverage = np.array([r['coverage'] for r in members])
        misses = np.array([r['repeats']-r['covered'] for r in members])
        rng = np.random.default_rng(20261028)
        samples = coverage[rng.integers(len(members), size=(10000, len(members)))].mean(1)
        families.append(dict(zip(keys, identity), mechanisms=len(members),
            coverage=coverage.mean(), cluster_se=coverage.std(ddof=1)/np.sqrt(len(members)) if len(members)>1 else np.nan,
            bootstrap_low=np.quantile(samples,.025), bootstrap_high=np.quantile(samples,.975),
            total_misses=misses.sum(), worst_mechanism=members[int(misses.argmax())]['mechanism'],
            worst_mechanism_misses=misses.max(), worst_share_of_misses=misses.max()/misses.sum() if misses.sum() else 0.,
            mean_width=np.mean([r['mean_width'] for r in members]),
            mean_absolute_conditional_bias=np.mean([abs(r['signed_bias']) for r in members]),
            mean_within_mechanism_error_sd=np.mean([r['error_sd'] for r in members]),
            truth_below=sum(r['truth_below'] for r in members),
            truth_above=sum(r['truth_above'] for r in members),
            empty_target_arm_contexts=sum(r['empty_target_arm_contexts'] for r in members)))
    return mechanisms, families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_metrics = args.input/'evaluation/context_metrics.csv'
    hashes = {str(p): digest(p) for p in (args.input/'execution.json', source_metrics)}
    manifest = json.loads((args.input/'execution.json').read_text())
    raw_ate, _ = read_results(args.input)
    rows = [dict(family=r['family'], mechanism=r['mechanism'], repeat=r['repeat'],
        m_observed=r['m_observed'], context_id=r['context_id'], method=r['method'],
        estimand='ATE', effect=r['effect'], valid=r['point_valid']==1,
        mse=r['squared_error_sum'] if r['point_valid'] else np.nan,
        bias=r['error_sum'] if r['point_valid'] else np.nan) for r in raw_ate]
    truths = {}
    jobs = []
    unsupported = []
    for receipt in manifest['receipts']:
        meta = context_metadata(receipt['context_id'])
        key = (meta['family'], meta['mechanism'])
        if not any(receipt['arm_supported']) and not any(receipt['ite_supported']):
            unsupported.append(dict(**meta, method=receipt['method'], reason='mu_and_ITE_not_supported'))
            continue
        if key not in truths:
            with np.load(args.input/'truth'/f'f{key[0]}_m{key[1]}.npz') as saved:
                truths[key] = saved['arm_means']
        jobs.append((args.input, receipt, truths[key]))
    intervals = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (points, interval_rows) in enumerate(pool.map(raw_context, jobs), 1):
            rows.extend(points)
            intervals.extend(interval_rows)
            if i % 1000 == 0:
                print(f'Audited {i}/{len(jobs)} supported mu/ITE predictions', flush=True)
    write_csv(args.output/'context_point_metrics.csv', rows)
    # A J-only replication has no unsupported mu/ITE rows. Keep a valid header.
    with (args.output/'unsupported_mu_ite.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'family','mechanism','repeat','m_observed','method','reason'))
        writer.writeheader()
        writer.writerows(unsupported)
    tasks, summary = task_summaries(rows)
    write_csv(args.output/'task_metrics.csv', tasks)
    write_csv(args.output/'task_mean_sd.csv', summary)
    write_csv(args.output/'j_interval_contexts.csv', intervals)
    mechanisms, families = interval_summaries(intervals)
    write_csv(args.output/'j_interval_by_mechanism.csv', mechanisms)
    write_csv(args.output/'j_interval_by_family.csv', families)
    # Audit support/hits against the independently produced original evaluator.
    expected = {(r['context_id'],r['effect']):r for r in raw_ate if r['method']=='j'}
    assert len(expected) == len(intervals)
    for r in intervals:
        original = expected[r['context_id'],r['effect']]
        assert r['covered']==original['covered']
        np.testing.assert_allclose(r['point_error'],original['error_sum'],rtol=0,atol=1e-12)
    timing = []
    keys = ('method','m_observed')
    receipts = [dict(r, **context_metadata(r['context_id'])) for r in manifest['receipts']]
    for identity, members in sorted(group_rows(receipts, keys).items()):
        online = np.array([r['online_seconds'] for r in members])
        interval = np.array([r['interval_seconds'] for r in members])
        task_times = [np.mean([r['online_seconds'] for r in g]) for g in group_rows(members,('family','mechanism')).values()]
        timing.append(dict(zip(keys,identity), contexts=len(members),
            online_mean_ms=1000*online.mean(), online_median_ms=1000*np.median(online),
            online_p95_ms=1000*np.quantile(online,.95),
            task_mean_online_sd_ms=1000*np.std(task_times,ddof=1),
            interval_mean_ms=1000*interval.mean() if identity[0] in manifest['interval_methods'] else np.nan,
            interval_supported=identity[0] in manifest['interval_methods']))
    write_csv(args.output/'runtime.csv',timing)
    for p, value in hashes.items():
        assert digest(p)==value, f'Original source changed: {p}'
    record = dict(source_sha256=hashes, script_sha256=digest(__file__), original_sources_unchanged=True,
        supported_mu_ite_contexts=len(jobs), context_metric_rows=len(rows),
        interval_context_effect_rows=len(intervals), original_ATE_hits_and_errors_match=True,
        model_load_seconds=manifest['model_load_seconds'],
        timing='Original receipts; excludes model loading, graph generation, disk IO; method caches/order retained. Unsupported interval timing omitted.',
        tasks='Independent (family, mechanism); repeats nested. SD is across task RMSE/bias, not within-mechanism variance.',
        bootstrap='10000 whole-mechanism resamples, seed 20261028; percentile intervals descriptive, especially unstable for the original 8-mechanism groups.',
        support='Global observed arm counts are not per-node empirical positivity certificates.',
        no_calibration_performed=True)
    (args.output/'provenance.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
