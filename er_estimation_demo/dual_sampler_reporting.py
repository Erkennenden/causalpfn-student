"""Separate within-sampler metrics, pooled over all graphs and each family."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from evaluate_random_csv_tables import EFFECTS, paired_comparisons, _table
from evaluation.unified import _metric_rows, _write_csv, _json_safe

FAMILIES = ('er', 'configuration', 'rgg', 'sbm')
SAMPLER_TITLES = {'matched_old_sampler': 'Matched old-sampler test',
                  'historical_c': 'C-sampler test (X~N(0,1); new benchmark)'}
REGIME_TITLES = {0: 'No neighbor effects', 3: 'Both neighbor effects'}


def family_for_dataset(dataset_id):
    """Four tasks per graph-family block, with two null and two active tasks."""
    if int(dataset_id) < 1:
        raise ValueError('dataset_id must be one-based.')
    return FAMILIES[((int(dataset_id)-1)//4) % 4]


def _supported(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Invalid support flag: {value!r}')


def normalize_row(source):
    row = dict(source)
    row['dataset_id'] = int(row['dataset_id'])
    if 'unit_id' in row:
        row['unit_id'] = int(row['unit_id'])
    row['truth'] = float(row['truth'])
    row['estimate'] = float(row['estimate'])
    if not math.isfinite(row['truth']):
        raise ValueError('Benchmark truth must be finite.')
    row['supported'] = _supported(row.get('supported', True)) and math.isfinite(row['estimate'])
    if not row['supported']:
        row['estimate'] = float('nan')
    return row


def summarize_rows(rows, *, graph_count, level, sampler, fixed_er=False):
    if graph_count <= 0 or (not fixed_er and graph_count % 16):
        raise ValueError('Balanced family reports require a positive multiple of 16 graphs.')
    if level not in ('ite', 'ate') or sampler not in SAMPLER_TITLES:
        raise ValueError('Unknown level or sampler.')
    rows = [normalize_row(r) for r in rows]
    if any(not 1 <= r['dataset_id'] <= graph_count for r in rows):
        raise ValueError('Row contains a dataset_id outside the completed benchmark.')
    summaries, paired = [], []
    for family in ('all', 'er') if fixed_er else ('all', *FAMILIES):
        selected = [r for r in rows if fixed_er or family == 'all' or family_for_dataset(r['dataset_id']) == family]
        methods = tuple(dict.fromkeys(r['method'] for r in selected))
        prefix = dict(sampler=sampler, estimand_level=level, graph_family=family,
                      n_graphs=graph_count if fixed_er or family == 'all' else graph_count//4)
        if fixed_er:
            prefix.update(n_episodes=graph_count, unique_graph_count=1)
        metrics = _metric_rows(selected, methods=methods, effects=EFFECTS,
                              subset_name='method_specific_support')
        summaries.extend({**prefix, **r} for r in metrics if r['n_total'])
        # Comparisons always pair PFN and a baseline within THIS sampler.
        paired.extend({**prefix, **r} for r in paired_comparisons(selected))
    return summaries, paired


def _read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def summarize_regime_rows(rows, *, regime_by_dataset, level, sampler, fixed_er=False):
    """Use recorded mechanism metadata, never inferred outcome values or ID parity."""
    rows = [normalize_row(r) for r in rows]
    summaries, paired = [], []
    for regime, title in REGIME_TITLES.items():
        for family in ('all', 'er') if fixed_er else ('all', *FAMILIES):
            ids = {i for i, r in regime_by_dataset.items() if r == regime
                   and (fixed_er or family == 'all' or family_for_dataset(i) == family)}
            selected = [r for r in rows if r['dataset_id'] in ids]
            if not ids:
                continue
            methods = tuple(dict.fromkeys(r['method'] for r in selected))
            prefix = dict(sampler=sampler, estimand_level=level, regime=regime,
                          regime_name=title, graph_family=family, n_graphs=len(ids))
            if fixed_er:
                prefix.update(n_episodes=len(ids), unique_graph_count=1)
            metrics = _metric_rows(selected, methods=methods, effects=EFFECTS,
                                   subset_name='method_specific_support')
            summaries.extend({**prefix, **r} for r in metrics if r['n_total'])
            paired.extend({**prefix, **r} for r in paired_comparisons(selected))
    return summaries, paired


def write_regime_reports(output_dir, *, sampler, result):
    regime_by_dataset = {int(r['task_id']) + 1: int(r['regime'])
                         for r in result['input_hashes']}
    a_group = result['manifest']['data_config'].get('dgp_version') in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1')
    fixed_er = result['manifest']['data_config'].get('graph_protocol') == 'fixed_er_v1'
    title = 'Fixed ER graph; independent data repeats' if fixed_er else SAMPLER_TITLES[sampler]
    expected_regimes = {3} if a_group else set(REGIME_TITLES)
    if a_group:
        spec = result['manifest'].get('outcome_dgp') or {}
        saved = result['manifest']['data_config']
        zero = (spec.get('neighbor_coefficient') == spec.get('spillover_coefficient') == 0
                or (saved.get('dgp_version') == 'a_group_fixed_v1'
                    and saved.get('interference_lambda', 1.) == 0))
        if zero:
            expected_regimes = {0}
    if set(regime_by_dataset.values()) != expected_regimes:
        raise ValueError(f'Regime reports require exactly {sorted(expected_regimes)} for this DGP.')
    summaries, pairs = {}, {}
    lines = [title,
             'Regime 0: neighbor-X coefficient=0 and treatment-spillover coefficient=0.',
             'Regime 3: both neighbor coefficients are active.',
             'Each regime is scored separately; unknown/unsupported predictions retain their counts.']
    if a_group:
        from a_group_dgp import outcome_description, outcome_spec
        spec = result['manifest'].get('outcome_dgp')
        if spec is None:
            spec = outcome_spec(result['manifest']['data_config'].get('interference_lambda', 1.0))
        lines = [title, 'A group: ' + outcome_description(spec) + '.']
    for level, filename in (('ite', 'ite_unit_results.csv'), ('ate', 'ate_graph_results.csv')):
        summaries[level], pairs[level] = summarize_regime_rows(_read_rows(output_dir/filename),
            regime_by_dataset=regime_by_dataset, level=level, sampler=sampler, fixed_er=fixed_er)
        _write_csv(summaries[level], output_dir/f'{level}_by_regime.csv')
        _write_csv(pairs[level], output_dir/f'{level}_paired_by_regime.csv')
        for regime, title in REGIME_TITLES.items():
            if regime not in expected_regimes:
                continue
            for family in ('all', 'er') if fixed_er else ('all', *FAMILIES):
                selected = [r for r in summaries[level]
                            if r['regime'] == regime and r['graph_family'] == family]
                folder = output_dir/'regime_tables'/f'regime_{regime}'/family
                folder.mkdir(parents=True, exist_ok=True)
                _write_csv(selected, folder/f'{level}_summary.csv')
                table = _table(selected, level).replace('Random LPE, fixed-CSV PFN with interference:',
                    f'{"Fixed ER graph" if fixed_er else SAMPLER_TITLES[sampler]}; {title}; {family}:')
                (folder/f'{level}_table.tex').write_text(table)
            for row in summaries[level]:
                if row['regime'] == regime and row['graph_family'] == 'all':
                    lines.append(f'{level.upper()} | {title} | {row["method"]} | '
                        f'{row["effect"]} | RMSE={row["rmse"]:.6f} | '
                        f'Support={row["n_supported"]}/{row["n_total"]}')
    report = dict(sampler=sampler, checkpoint_sha256=result['manifest']['checkpoint_sha256'],
        regime_graph_counts={str(r): sum(v == r for v in regime_by_dataset.values())
                             for r in sorted(expected_regimes)}, summaries=summaries, paired_comparisons=pairs)
    if fixed_er:
        report.update(unique_graph_count=1, regime_episode_counts=report['regime_graph_counts'])
    (output_dir/'regime_results.json').write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
    (output_dir/'regime_summary.txt').write_text('\n'.join(lines)+'\n')
    return report


def _family_table(rows, level, sampler, family, *, fixed_er=False):
    caption = f'{"Fixed ER graph; independent data repeats" if fixed_er else SAMPLER_TITLES[sampler]}; {family.title()}'
    return _table(rows, level).replace('Random LPE, fixed-CSV PFN with interference:', caption+':')


def write_family_reports(output_dir, *, sampler):
    """Add reports to a completed evaluator output without changing its tables."""
    output_dir = Path(output_dir)
    result = json.loads((output_dir/'results.json').read_text())
    manifest = result['manifest']
    graph_count = int(manifest['graphs'])
    fixed_er = manifest['data_config'].get('graph_protocol') == 'fixed_er_v1'
    if graph_count <= 0 or manifest.get('task_ids') != list(range(graph_count)):
        raise ValueError('Reports require a positive count and complete ordered task IDs.')
    if fixed_er:
        if manifest['data_config']['graph_family'] != 'er':
            raise ValueError('Fixed-ER reports require ER graph metadata.')
        expected_hash = manifest.get('fixed_graph', {}).get('adjacency_sha256')
        if not expected_hash or any(r.get('adjacency_sha256') != expected_hash for r in result['input_hashes']):
            raise ValueError('Fixed-ER adjacency hashes must all match the checkpoint graph.')
    elif (manifest.get('balanced') is not True or manifest['data_config']['graph_family'] != 'mixed'):
        raise ValueError('Family reports require the frozen balanced task-id design.')
    if [r['task_id'] for r in result['input_hashes']] != list(range(graph_count)):
        raise ValueError('Missing or reordered per-graph input hashes.')
    summaries, pairs = {}, {}
    readable = ['Fixed ER graph; independent data repeats' if fixed_er else SAMPLER_TITLES[sampler],
                f'Checkpoint SHA256: {manifest["checkpoint_sha256"]}',
                (f'Episodes: {graph_count}; unique networks: 1; n_graphs columns count episodes.'
                 if fixed_er else f'All graphs: {graph_count}; Configuration graphs: {graph_count//4}'),
                'RMSE is pooled over target errors; ATE uses saved whole-graph ATE values.',
                'PFN with interference/baseline comparisons use common support within this sampler.']
    for level, filename in (('ite', 'ite_unit_results.csv'), ('ate', 'ate_graph_results.csv')):
        rows = _read_rows(output_dir/filename)
        if {int(r['dataset_id']) for r in rows} != set(range(1,graph_count+1)):
            raise ValueError(f'Incomplete {level.upper()} graph results.')
        summaries[level], pairs[level] = summarize_rows(rows, graph_count=graph_count,
                                                        level=level, sampler=sampler, fixed_er=fixed_er)
        _write_csv(summaries[level], output_dir/f'{level}_by_graph_family.csv')
        if pairs[level]:
            _write_csv(pairs[level], output_dir/f'{level}_paired_by_graph_family.csv')
        for family in ('all', 'er') if fixed_er else ('all', *FAMILIES):
            local = [r for r in summaries[level] if r['graph_family'] == family]
            folder = output_dir/'family_tables'/family
            folder.mkdir(parents=True, exist_ok=True)
            _write_csv(local, folder/f'{level}_summary.csv')
            (folder/f'{level}_table.tex').write_text(_family_table(local, level, sampler, family, fixed_er=fixed_er))
            readable.append(f'\n{level.upper()} / {family} | Method | Effect | RMSE | MAE | Support')
            for row in local:
                readable.append(f'{row["method"]} | {row["effect"]} | {row["rmse"]:.6f} | '
                                f'{row["mae"]:.6f} | {row["n_supported"]}/{row["n_total"]}')
    report = dict(sampler=sampler, checkpoint_sha256=manifest['checkpoint_sha256'],
                  seed=manifest['seed'], graph_count=graph_count,
                  graph_family_counts={'er': graph_count} if fixed_er else {name:graph_count//4 for name in FAMILIES},
                  summaries=summaries, paired_comparisons=pairs)
    if fixed_er:
        report.update(episode_count=graph_count, unique_graph_count=1,
                      adjacency_sha256=expected_hash, graph_family_episode_counts={'er': graph_count})
    (output_dir/'graph_family_results.json').write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
    (output_dir/'graph_family_summary.txt').write_text('\n'.join(readable)+'\n')
    write_regime_reports(output_dir, sampler=sampler, result=result)
    return report


def validate_pair_manifests(old, historical):
    for key in ('checkpoint_sha256', 'model_config', 'data_config', 'prior_config',
                'graphs', 'seed', 'stream', 'balanced', 'function_shift'):
        if old.get(key) != historical.get(key):
            raise ValueError(f'Dual-test manifests disagree on {key}.')
    if old['causalpfn']['checkpoint_sha256'] != historical['causalpfn']['checkpoint_sha256']:
        raise ValueError('Dual tests used different CausalPFN weights.')


def combine_reports(output_dir):
    """Keep separate columns for each benchmark, including its own baselines."""
    output_dir = Path(output_dir)
    completed = {name:json.loads((output_dir/name/'results.json').read_text()) for name in SAMPLER_TITLES}
    validate_pair_manifests(completed['matched_old_sampler']['manifest'], completed['historical_c']['manifest'])
    reports = {name:write_family_reports(output_dir/name,sampler=name) for name in SAMPLER_TITLES}
    indexed = {}
    for name, report in reports.items():
        indexed[name] = {(level,r['graph_family'],r['method'],r['effect']):r
                         for level,rows in report['summaries'].items() for r in rows}
    if indexed['matched_old_sampler'].keys() != indexed['historical_c'].keys():
        raise ValueError('The two benchmarks did not run the same methods and effects.')
    comparisons = []
    for key, old in indexed['matched_old_sampler'].items():
        historical = indexed['historical_c'][key]
        row = dict(estimand_level=key[0],graph_family=key[1],method=key[2],effect=key[3],n_graphs=old['n_graphs'])
        for name, values in (('matched_old_sampler',old),('historical_c',historical)):
            for metric in ('rmse','mae','bias','n_supported','n_total'):
                row[f'{name}_{metric}'] = values[metric]
        comparisons.append(row)
    _write_csv(comparisons,output_dir/'dual_rmse_comparison.csv')
    _write_csv([r for r in comparisons if r['graph_family'] in ('all','configuration')],
               output_dir/'all_and_configuration_comparison.csv')
    regime_reports = {name: json.loads((output_dir/name/'regime_results.json').read_text())
                      for name in SAMPLER_TITLES}
    by_regime = {name: {(level, r['regime'], r['graph_family'], r['method'], r['effect']): r
                       for level, rows in report['summaries'].items() for r in rows}
                 for name, report in regime_reports.items()}
    if by_regime['matched_old_sampler'].keys() != by_regime['historical_c'].keys():
        raise ValueError('Two-regime benchmarks did not evaluate the same targets.')
    regime_comparisons = []
    for key, old in by_regime['matched_old_sampler'].items():
        row = dict(estimand_level=key[0], regime=key[1], regime_name=REGIME_TITLES[key[1]],
                   graph_family=key[2], method=key[3], effect=key[4], n_graphs=old['n_graphs'])
        for name in SAMPLER_TITLES:
            for metric in ('rmse','mae','bias','n_supported','n_total'):
                row[f'{name}_{metric}'] = by_regime[name][key][metric]
        regime_comparisons.append(row)
    _write_csv(regime_comparisons, output_dir/'dual_regime_comparison.csv')
    old_hashes = completed['matched_old_sampler']['input_hashes']
    historical_hashes = completed['historical_c']['input_hashes']
    if len(old_hashes) != len(historical_hashes):
        raise ValueError('Dual benchmarks have different graph counts.')
    input_comparison=[]
    for old,historical in zip(old_hashes,historical_hashes):
        if old['task_id'] != historical['task_id']:
            raise ValueError('Dual benchmark task IDs are not aligned.')
        family=family_for_dataset(old['task_id']+1)
        identical=old['sha256']==historical['sha256']
        if family!='configuration' and not identical:
            raise ValueError(f'Unexpected non-Configuration input shift at task {old["task_id"]}.')
        input_comparison.append(dict(task_id=old['task_id'],graph_family=family,
            matched_old_sampler_sha256=old['sha256'],historical_c_sha256=historical['sha256'],identical=identical))
    _write_csv(input_comparison,output_dir/'input_hash_comparison.csv')
    lines=['Same checkpoint; two separately evaluated benchmarks.',
           'Each sampler has freshly evaluated baselines; historical reference scores are not substituted.',
           'RMSE/support columns use each method\'s own support. For PFN with interference versus a baseline, see each sampler\'s paired-by-family CSV.',
           'Level | Family | Method | Effect | Matched old RMSE | C-sampler X-normal RMSE']
    for r in comparisons:
        if r['graph_family'] in ('all','configuration'):
            lines.append(f'{r["estimand_level"].upper()} | {r["graph_family"]} | {r["method"]} | {r["effect"]} | '
                         f'{r["matched_old_sampler_rmse"]:.6f} | {r["historical_c_rmse"]:.6f}')
    (output_dir/'dual_summary.txt').write_text('\n'.join(lines)+'\n')
    return dict(comparisons=comparisons,input_hash_comparison=input_comparison)
