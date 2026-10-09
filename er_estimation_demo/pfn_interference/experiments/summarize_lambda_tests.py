"""Verify paired fixed-lambda tests and collect all ITE/ATE baseline summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PAIRED_KEYS = ('adjacency', 'x', 'observed_treatment', 'observed_exposure', 'noise', 'queries')
LAMBDAS = (0., 1., 2., 3.)
METRICS = ('mae', 'rmse', 'bias', 'mean_truth', 'mean_estimate', 'n_supported', 'n_total')


def summarize(directories, output_dir):
    if len(directories) != 4:
        raise ValueError('Need exactly four lambda test directories.')
    results = [json.loads((Path(d)/'results.json').read_text()) for d in directories]
    base = results[0]
    hashes = []
    rows = []
    for expected, result in zip(LAMBDAS, results):
        manifest = result['manifest']
        if manifest['test_lambda'] != expected:
            raise ValueError(f'Expected test lambda {expected}.')
        if manifest['checkpoint_sha256'] != base['manifest']['checkpoint_sha256']:
            raise ValueError('Lambda tests used different checkpoints.')
        if manifest['training_outcome_dgp'] != base['manifest']['training_outcome_dgp']:
            raise ValueError('Training priors differ across tests.')
        if manifest['graphs'] != 1 or len(result['input_hashes']) != 1:
            raise ValueError('Need one paired data episode per lambda.')
        if manifest['seed'] != base['manifest']['seed']:
            raise ValueError('Paired test seeds differ.')
        record = result['input_hashes'][0]
        if record['task_id'] != base['input_hashes'][0]['task_id']:
            raise ValueError('Paired task IDs differ.')
        for name, value in (('beta', -3*expected), ('gamma', -.9*expected)):
            if not math.isclose(record[name], value, abs_tol=1e-6):
                raise ValueError(f'Generated {name} differs from requested lambda.')
            if not math.isclose(manifest['test_coefficients'][name], value, abs_tol=1e-6):
                raise ValueError(f'Manifest {name} differs from requested lambda.')
        for key in PAIRED_KEYS:
            if record['component_sha256'][key] != base['input_hashes'][0]['component_sha256'][key]:
                raise ValueError(f'Inputs are not paired: {key}.')
        hashes.append(record['component_sha256'])
        for level in ('ite', 'ate'):
            for row in result['summaries'][level]:
                rows.append(dict(**{'lambda':expected}, beta=-3*expected, gamma=-.9*expected, level=level, method=row['method'], effect=row['effect'],
                                 **{key: row[key] for key in METRICS}))
    for key in ('y_obs', 'gamma', 'oracle_ite', 'prior_neighbor_coefficient'):
        if len({h[key] for h in hashes}) != 4:
            raise ValueError(f'Outcome inputs did not change across all lambda tests: {key}.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'lambda_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('lambda','beta','gamma','level','method','effect',*METRICS))
        writer.writeheader(); writer.writerows(rows)
    audit = dict(paired=True, checkpoint_sha256=base['manifest']['checkpoint_sha256'],
                 test_lambdas=LAMBDAS, seed=base['manifest']['seed'], episodes_per_lambda=1,
                 training_outcome_dgp=base['manifest']['training_outcome_dgp'],
                 shared_component_sha256={key: hashes[0][key] for key in PAIRED_KEYS},
                 outcome_component_sha256=[{key: h[key] for key in ('y_obs','gamma','oracle_ite','prior_neighbor_coefficient')} for h in hashes])
    (out/'pairing_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['One trained PFN with interference; four paired lambda tests; all existing pfn_interference.baselines.',
             'Pairing and checkpoint consistency: PASSED',
             'lambda | beta | gamma | Level | Method | Effect | MAE | RMSE | Bias | Mean Truth | Mean Pred | Support']
    for row in rows:
        numbers = ' | '.join('NaN' if row[key] is None else f'{float(row[key]):.6f}'
                             for key in METRICS[:5])
        lines.append(f'{row["lambda"]:g} | {row["beta"]:g} | {row["gamma"]:g} | {row["level"].upper()} | {row["method"]} | {row["effect"]} | {numbers} | {row["n_supported"]}/{row["n_total"]}')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('directories', nargs=4, type=Path)
    args = parser.parse_args()
    summarize(args.directories, args.output_dir)
    print((args.output_dir/'summary.txt').read_text(), end='')


if __name__ == '__main__':
    main()
