"""Verify paired fixed-gamma tests and collect all ITE/ATE baseline summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PAIRED_KEYS = ('adjacency', 'x', 'observed_treatment', 'observed_exposure', 'noise', 'queries')
GAMMAS = (-.9, -1.8, -2.7)
METRICS = ('mae', 'rmse', 'bias', 'mean_truth', 'mean_estimate', 'n_supported', 'n_total')


def summarize(directories, output_dir):
    if len(directories) != 3:
        raise ValueError('Need exactly three gamma test directories.')
    results = [json.loads((Path(d)/'results.json').read_text()) for d in directories]
    base = results[0]
    hashes = []
    rows = []
    for expected, result in zip(GAMMAS, results):
        manifest = result['manifest']
        if manifest['test_gamma'] != expected:
            raise ValueError(f'Expected test gamma {expected}.')
        if manifest['checkpoint_sha256'] != base['manifest']['checkpoint_sha256']:
            raise ValueError('Gamma tests used different checkpoints.')
        if manifest['training_outcome_dgp'] != base['manifest']['training_outcome_dgp']:
            raise ValueError('Training priors differ across tests.')
        if manifest['graphs'] != 1 or len(result['input_hashes']) != 1:
            raise ValueError('Need one paired data episode per gamma.')
        if manifest['seed'] != base['manifest']['seed']:
            raise ValueError('Paired test seeds differ.')
        record = result['input_hashes'][0]
        if record['task_id'] != base['input_hashes'][0]['task_id']:
            raise ValueError('Paired task IDs differ.')
        if not math.isclose(record['gamma'], expected, abs_tol=1e-6):
            raise ValueError('Generated gamma differs from requested test gamma.')
        for key in PAIRED_KEYS:
            if record['component_sha256'][key] != base['input_hashes'][0]['component_sha256'][key]:
                raise ValueError(f'Inputs are not paired: {key}.')
        hashes.append(record['component_sha256'])
        for level in ('ite', 'ate'):
            for row in result['summaries'][level]:
                rows.append(dict(gamma=expected, level=level, method=row['method'], effect=row['effect'],
                                 **{key: row[key] for key in METRICS}))
    for key in ('y_obs', 'gamma', 'oracle_ite'):
        if len({h[key] for h in hashes}) != 3:
            raise ValueError(f'Outcome inputs did not change across all gamma tests: {key}.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'gamma_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('gamma','level','method','effect',*METRICS))
        writer.writeheader(); writer.writerows(rows)
    audit = dict(paired=True, checkpoint_sha256=base['manifest']['checkpoint_sha256'],
                 test_gammas=GAMMAS, seed=base['manifest']['seed'], episodes_per_gamma=1,
                 training_outcome_dgp=base['manifest']['training_outcome_dgp'],
                 shared_component_sha256={key: hashes[0][key] for key in PAIRED_KEYS},
                 outcome_component_sha256=[{key: h[key] for key in ('y_obs','gamma','oracle_ite')} for h in hashes])
    (out/'pairing_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['One trained PFN with interference; three paired gamma tests; all existing baselines.',
             'Pairing and checkpoint consistency: PASSED',
             'gamma | Level | Method | Effect | MAE | RMSE | Bias | Mean Truth | Mean Pred | Support']
    for row in rows:
        numbers = ' | '.join('NaN' if row[key] is None else f'{float(row[key]):.6f}'
                             for key in METRICS[:5])
        lines.append(f'{row["gamma"]:g} | {row["level"].upper()} | {row["method"]} | {row["effect"]} | {numbers} | {row["n_supported"]}/{row["n_total"]}')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('directories', nargs=3, type=Path)
    args = parser.parse_args()
    summarize(args.directories, args.output_dir)
    print((args.output_dir/'summary.txt').read_text(), end='')


if __name__ == '__main__':
    main()
