"""Frozen E1.5 Star ATE comparisons, exported for the upstream report tools.

No network training, calibration, seed search or automatic retries. Smoke is
one mechanism/pair; pilot and formal require explicit CLI selection.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from benchmark import configuration, load_methods, predict_method, source_directory
from evaluate import evaluate_context, summarize
from make_table import read_results
from pfn_social.intervals import predict_intervals
from pfn_social.phase6_sparse_classical import pack
from pfn_social.sparse_frozen_gaussian import validate_factual_inputs
from scripts.e1_5_star import (N, M, DEFAULT_SEED, effects, build_world, make_pair,
                               verify_assets, source_snapshot, git_record, sha256,
                               write_json, write_csv)

METHODS = ('j', 'kernel-old', 'kernel-new')
LEVEL = .95
PROTOCOL = 'e1_5_star_ate_v1'
POINT_KEYS = ('arm_means', 'ite_effects', 'ate_effects', 'arm_supported', 'ite_supported', 'ate_supported')
INTERVAL_KEYS = ('ate_lower', 'ate_upper', 'interval_level', 'interval_kind', 'ate_interval_supported')


def frozen_plan(cohort):
    mids, rids = {'smoke': ([900], [0]),
                  'pilot': (list(range(900, 904)), list(range(5))),
                  'pilot_fresh': (list(range(904, 908)), list(range(5))),
                  'formal': (list(range(1000, 1020)), list(range(10)))}[cohort]
    return dict(cohort=cohort, mechanism_ids=mids, pair_repeat_ids=rids, seed=DEFAULT_SEED,
                n=N, m_observed=M, family=0, graph='star', center_node=0,
                center_order=[0, 1], methods=list(METHODS), interval_level=LEVEL)


def context_id(mid, rid, center):
    return f'f0_m{mid}_r{2*rid+center}_observed50'


def ate_truth(mu):
    mu = np.asarray(mu)
    if mu.shape != (N, 4) or not np.isfinite(mu).all():
        raise ValueError('Require finite all-node truth[200,4]')
    return effects(mu).mean(0)


class InputMutationError(RuntimeError):
    pass


def call_on_copy(function, inputs):
    copied = {k: v.copy() for k, v in inputs.items()}
    try:
        return function(copied)
    finally:
        if any(not np.array_equal(copied[k], v) for k, v in inputs.items()):
            raise InputMutationError('Predictor mutated its factual input copy')


def check_point(point):
    for key, shape in [('arm_means', (N, 4)), ('ite_effects', (N, 3)), ('ate_effects', (3,))]:
        value = np.asarray(point[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid point prediction: {key}')
    np.testing.assert_allclose(point['ite_effects'], effects(point['arm_means']), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(point['ate_effects'], np.asarray(point['ite_effects']).mean(0), rtol=2e-6, atol=2e-6)
    for key, size in [('arm_supported', 4), ('ite_supported', 3), ('ate_supported', 3)]:
        if np.asarray(point[key]).shape != (size,) or not np.asarray(point[key]).all():
            raise ValueError(f'Unexpected missing support for frozen mean model: {key}')


def predict_record(name, model, inputs, cid, mid, rid, center, *, predictor=None, interval_predictor=None):
    """Requested capability stays true on failure; validity is separate."""
    predictor = predictor or predict_method
    interval_predictor = interval_predictor or predict_intervals
    fit_seed = 651013 + 100*mid + 2*rid + center  # recorded; unused by these frozen models
    failure = interval_failure = None
    began = time.perf_counter()
    try:
        point = call_on_copy(lambda data: predictor(name, model, data, fit_seed), inputs)
        check_point(point)
    except InputMutationError:
        raise
    except Exception as exc:
        point = pack(np.full((N, 4), np.nan), n=N)
        failure = dict(type=type(exc).__name__, message=str(exc))
    point_seconds = time.perf_counter()-began
    # ATE capability means requested target, not observed numerical finiteness.
    payload = {key: np.asarray(point[key]) for key in POINT_KEYS}
    payload['ate_supported'] = np.ones(3, dtype=bool)
    payload.update(ate_lower=np.full(3, np.nan), ate_upper=np.full(3, np.nan),
                   interval_level=np.array(LEVEL),
                   interval_kind=np.array('frozen_J_marginal_posterior_credible' if name == 'j' else 'unsupported'),
                   ate_interval_supported=np.full(3, name == 'j'))
    began = time.perf_counter()
    if name == 'j':
        try:
            result = call_on_copy(lambda data: interval_predictor(model, **data, level=LEVEL), inputs)
            low, high = np.asarray(result['ate_lower']), np.asarray(result['ate_upper'])
            if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                    or np.any(low > high) or float(result['interval_level']) != LEVEL
                    or str(result['interval_kind']) != 'frozen_J_marginal_posterior_credible'):
                raise ValueError('Invalid J ATE interval fields')
            payload['ate_lower'], payload['ate_upper'] = low, high
        except InputMutationError:
            raise
        except Exception as exc:
            interval_failure = dict(type=type(exc).__name__, message=str(exc))
    receipt = dict(context_id=cid, method=name, mechanism_id=mid, pair_repeat_id=rid, center_t=center,
                   fit_seed=fit_seed, fit_seed_used=False, online_seconds=point_seconds,
                   interval_seconds=time.perf_counter()-began, interval_level=LEVEL,
                   failure=failure, interval_failure=interval_failure,
                   ate_supported=[True]*3, ate_interval_supported=[name == 'j']*3,
                   arm_supported=payload['arm_supported'].tolist(), ite_supported=payload['ite_supported'].tolist())
    return payload, receipt


def receipt_metadata(receipt):
    """Use explicit pair metadata; the ID repeat encodes both repeat and center."""
    for field in ('mechanism_id', 'pair_repeat_id', 'center_t'):
        value = receipt.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid or missing receipt {field}: {value!r}')
    mid, rid, center = (receipt[k] for k in ('mechanism_id', 'pair_repeat_id', 'center_t'))
    if center not in (0, 1):
        raise ValueError(f'Invalid receipt center_t: {center!r}')
    if receipt.get('context_id') != context_id(mid, rid, center):
        raise ValueError('Receipt context_id disagrees with mechanism_id/pair_repeat_id/center_t')
    if receipt.get('method') not in METHODS:
        raise ValueError('Invalid or missing receipt method')
    return dict(repeat=rid, pair_repeat_id=rid, center_t=center)


def score_saved(directory, receipts):
    rows = []
    for receipt in receipts:
        pair_metadata = receipt_metadata(receipt)
        mid, cid, method = receipt['mechanism_id'], receipt['context_id'], receipt['method']
        with np.load(directory/'truth'/f'f0_m{mid}.npz', allow_pickle=False) as saved:
            truth = saved['arm_means']
        with np.load(directory/'predictions'/f'{cid}__{method}.npz', allow_pickle=False) as saved:
            prediction = {k: saved[k] for k in saved.files}
        rows.extend(dict(row, **pair_metadata) for row in evaluate_context(truth, prediction, receipt))
    return rows


def evaluation_tables(rows):
    return dict(context_metrics=rows, summary=summarize(rows),
                summary_by_family=summarize(rows, ('family',)),
                summary_by_mechanism=summarize(rows, ('family', 'mechanism')))


def write_views(directory, execution):
    """Export overall and center views from the same receipt-corrected rows."""
    rows = score_saved(directory, execution['receipts'])
    (directory/'evaluation').mkdir(exist_ok=False)
    for name, members in evaluation_tables(rows).items():
        write_csv(directory/'evaluation'/(name+'.csv'), members)
    for center in (0, 1):
        view = directory/'report_views'/f'center_t{center}'
        (view/'evaluation').mkdir(parents=True, exist_ok=False)
        receipts = [r for r in execution['receipts'] if r['center_t'] == center]
        ids = {r['context_id'] for r in receipts}
        write_json(view/'execution.json', dict(execution, receipts=receipts,
                   report_view=dict(center_t=center, parent_execution_sha256=sha256(directory/'execution.json'),
                                    purpose='Reports only; raw predictions remain in the parent run')))
        for name, members in evaluation_tables([r for r in rows if r['context_id'] in ids]).items():
            write_csv(view/'evaluation'/(name+'.csv'), members)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_csv(path, expected):
    with path.open(encoding='utf-8', newline='') as handle:
        stored = list(csv.DictReader(handle))
    _require(len(stored) == len(expected), f'CSV row count mismatch: {path}')
    for number, (row, wanted) in enumerate(zip(stored, expected)):
        _require(set(row) == set(wanted), f'CSV fields mismatch: {path}')
        for key, value in wanted.items():
            if isinstance(value, (float, np.floating)):
                _require(np.isclose(float(row[key]), value, rtol=1e-10, atol=1e-10, equal_nan=True),
                         f'CSV metric mismatch: {path.name} row={number} field={key}')
            else:
                _require(row[key] == str(value), f'CSV value mismatch: {path.name} row={number} field={key}')


def validate_result(directory, *, candidate=False):
    """Replay factual streams and score artifacts, never re-run a predictor."""
    directory = Path(directory)
    began = time.perf_counter()
    manifest = json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
    _require(manifest['status'] == ('validating' if candidate else 'complete'), 'Incomplete run')
    plan = manifest['plan']
    _require(plan == frozen_plan(plan['cohort']), 'Frozen protocol mismatch')
    _require(manifest['protocol'] == PROTOCOL and manifest['assets_sha256'] == verify_assets(), 'Protocol/asset mismatch')
    repo = ROOT.parents[2]
    with zipfile.ZipFile(directory/'source_snapshot.zip') as archive:
        _require(set(archive.namelist()) == set(manifest['source_sha256']), 'Source snapshot membership mismatch')
        for name, digest in manifest['source_sha256'].items():
            _require(sha256(repo/name) == digest and hashlib.sha256(archive.read(name)).hexdigest() == digest,
                     f'Source changed: {name}')
    files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    _require(files-{'run_manifest.json', 'acceptance_report.json'} == set(manifest['output_sha256']), 'Output inventory mismatch')
    for name, digest in manifest['output_sha256'].items():
        _require(sha256(directory/name) == digest, f'Output hash mismatch: {name}')
    execution = json.loads((directory/'execution.json').read_text(encoding='utf-8'))
    _require(execution['plan'] == plan and execution['methods'] == list(METHODS)
             and execution['interval_level'] == LEVEL and execution['protocol'] == PROTOCOL
             and execution['interval_methods'] == ['j'] and execution['torch_threads'] == execution['BLAS_threads'] == 1,
             'Execution protocol mismatch')
    expected = {(context_id(m, r, c), name) for m in plan['mechanism_ids']
                for r in plan['pair_repeat_ids'] for c in (0, 1) for name in METHODS}
    receipts = execution['receipts']
    for receipt in receipts:
        receipt_metadata(receipt)
    index = {(r['context_id'], r['method']): r for r in receipts}
    _require(len(index) == len(receipts) and set(index) == expected, 'Missing/duplicate/unexpected method-contexts')
    contexts = execution['contexts']
    context_index = {r['context_id']: r for r in contexts}
    _require(len(contexts) == len(context_index) and set(context_index) == {c for c, _ in expected}, 'Context inventory mismatch')
    mechanism_index = {r['mechanism_id']: r for r in execution['mechanisms']}
    _require(len(mechanism_index) == len(execution['mechanisms']) and set(mechanism_index) == set(plan['mechanism_ids']),
             'Mechanism inventory mismatch')
    for mid in plan['mechanism_ids']:
        base, world, truth = build_world(mid)
        _require(mechanism_index[mid]['config'] == asdict(base.config)
                 and mechanism_index[mid]['parent_mechanism_sha256'] == base.mechanism_sha256
                 and np.array_equal(mechanism_index[mid]['ate_truth'], ate_truth(truth)), 'Mechanism provenance/ATE mismatch')
        truth_file = directory/'truth'/f'f0_m{mid}.npz'
        with np.load(truth_file, allow_pickle=False) as saved:
            _require(set(saved.files) == {'arm_means'} and np.array_equal(saved['arm_means'], truth), 'Mechanism truth mismatch')
        for rid in plan['pair_repeat_ids']:
            pair, entropy = make_pair(base, world, mid, rid, plan['seed'])
            for center, canonical in enumerate(pair):
                cid = context_id(mid, rid, center)
                info = context_index[cid]
                _require(info['rng_entropy'] == entropy and info['center_t'] == center
                         and info['pair_repeat_id'] == rid and info['mechanism_id'] == mid
                         and info['bit_generator'] == 'PCG64', 'Pair random stream mismatch')
                factual_file = directory/'factual'/(cid+'.npz')
                with np.load(factual_file, allow_pickle=False) as saved:
                    _require(set(saved.files) == set(canonical), 'Wrong factual keys')
                    inputs = {k: saved[k] for k in saved.files}
                validate_factual_inputs(**inputs)
                for key, value in canonical.items():
                    _require(inputs[key].dtype == value.dtype and np.array_equal(inputs[key], value), f'Factual/pair mismatch: {cid} {key}')
                _require(info['observed_indices'] == inputs['observed_indices'].tolist()
                         and info['center_observed'] == bool(0 in inputs['observed_indices']), 'Observation receipt mismatch')
                for name in METHODS:
                    receipt = index[cid, name]
                    _require(receipt['factual_sha256'] == sha256(factual_file) and receipt['truth_sha256'] == sha256(truth_file), 'Method input/truth mismatch')
                    _require(receipt['mechanism_id'] == mid and receipt['pair_repeat_id'] == rid and receipt['center_t'] == center
                             and receipt['fit_seed'] == 651013+100*mid+2*rid+center and receipt['fit_seed_used'] is False,
                             'Method receipt identity/seed mismatch')
                    with np.load(directory/'predictions'/f'{cid}__{name}.npz', allow_pickle=False) as saved:
                        _require(set(saved.files) == set(POINT_KEYS+INTERVAL_KEYS), 'Prediction field mismatch')
                        point = {k: saved[k] for k in saved.files}
                    for key, shape in [('arm_means', (N, 4)), ('ite_effects', (N, 3)), ('ate_effects', (3,)),
                                       ('arm_supported', (4,)), ('ite_supported', (3,)), ('ate_supported', (3,))]:
                        _require(point[key].shape == shape, f'Prediction shape mismatch: {key}')
                    _require(receipt['ate_supported'] == [True]*3 and point['ate_supported'].tolist() == [True]*3, 'Point requests lost')
                    if receipt['failure'] is None:
                        check_point(point)
                    else:
                        for key in ('arm_means', 'ite_effects', 'ate_effects'):
                            _require(np.isnan(point[key]).all(), 'Failed point must remain NaN, not zero')
                    supports = [name == 'j']*3
                    _require(point['interval_level'].shape == () and
                             receipt['ate_interval_supported'] == supports and point['ate_interval_supported'].tolist() == supports
                             and float(point['interval_level']) == LEVEL and receipt['interval_level'] == LEVEL, 'Interval requests/level mismatch')
                    lo, hi = point['ate_lower'], point['ate_upper']
                    _require(lo.shape == hi.shape == (3,), 'Wrong ATE interval shape')
                    if name == 'j' and receipt['interval_failure'] is None:
                        _require(np.isfinite([lo, hi]).all() and np.all(lo <= hi), 'Invalid J interval without failure receipt')
                    else:
                        _require(np.isnan([lo, hi]).all(), 'Unsupported/failed intervals must stay NaN')
                    _require(str(point['interval_kind']) == ('frozen_J_marginal_posterior_credible' if name == 'j' else 'unsupported'), 'Wrong interval kind')
                    if name != 'j':
                        _require(receipt['interval_failure'] is None, 'Unsupported baseline interval is not a failed request')
    rows = score_saved(directory, receipts)
    for name, wanted in evaluation_tables(rows).items():
        _check_csv(directory/'evaluation'/(name+'.csv'), wanted)
    read_results(directory)  # actual upstream consumer, not a substitute parser
    for center in (0, 1):
        view = directory/'report_views'/f'center_t{center}'
        selected = [r for r in receipts if r['center_t'] == center]
        ve = json.loads((view/'execution.json').read_text(encoding='utf-8'))
        _require(ve['receipts'] == selected and ve['report_view']['center_t'] == center
                 and ve['report_view']['parent_execution_sha256'] == sha256(directory/'execution.json'), 'Report view provenance mismatch')
        ids = {r['context_id'] for r in selected}
        for name, wanted in evaluation_tables([r for r in rows if r['context_id'] in ids]).items():
            _check_csv(view/'evaluation'/(name+'.csv'), wanted)
        read_results(view)
    return dict(passed=True, contexts=len(contexts), method_contexts=len(receipts), evaluation_rows=len(rows),
                point_failure_calls=sum(r['failure'] is not None for r in receipts),
                interval_failure_calls=sum(r['interval_failure'] is not None for r in receipts),
                baseline_intervals='N/A; not requested', seconds=time.perf_counter()-began)


def run_experiment(output, cohort='smoke'):
    output = Path(output).resolve()  # source_directory() temporarily changes cwd
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; never overwrite or resume')
    plan = frozen_plan(cohort)
    started = time.perf_counter()
    assets, config = verify_assets(), configuration()
    torch.set_num_threads(1)
    with source_directory(), threadpool_limits(limits=1):
        load_started = time.perf_counter()
        models = load_methods(list(METHODS), config)
        load_seconds = time.perf_counter()-load_started
        output.mkdir(parents=True, exist_ok=False)
        for child in ('factual', 'truth', 'predictions'):
            (output/child).mkdir()
        manifest = dict(protocol=PROTOCOL, status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                        plan=plan, assets_sha256=assets, source_sha256=source_snapshot(output), git=git_record(),
                        python=platform.python_version(), platform=platform.platform(), executable=sys.executable, command_line=sys.argv,
                        versions={p: metadata.version(p) for p in ('numpy', 'torch', 'scipy', 'threadpoolctl', 'matplotlib')},
                        cpu_threads=1, model_load_seconds=load_seconds)
        write_json(output/'run_manifest.json', manifest)
        receipts, contexts, mechanisms, generation_times, pair_times = [], [], [], [], []
        for mid in plan['mechanism_ids']:
            began = time.perf_counter()
            base, world, truth = build_world(mid)
            generation_times.append(time.perf_counter()-began)
            truth_file = output/'truth'/f'f0_m{mid}.npz'
            np.savez_compressed(truth_file, arm_means=truth)
            mechanisms.append(dict(mechanism_id=mid, config=asdict(base.config),
                                   parent_mechanism_sha256=base.mechanism_sha256, ate_truth=ate_truth(truth).tolist()))
            for rid in plan['pair_repeat_ids']:
                began = time.perf_counter()
                pair, entropy = make_pair(base, world, mid, rid, plan['seed'])
                for center, inputs in enumerate(pair):
                    cid = context_id(mid, rid, center)
                    validate_factual_inputs(**inputs)
                    factual_file = output/'factual'/(cid+'.npz')
                    np.savez_compressed(factual_file, **inputs)
                    contexts.append(dict(context_id=cid, mechanism_id=mid, pair_repeat_id=rid, center_t=center,
                                         rng_entropy=entropy, bit_generator='PCG64',
                                         observed_indices=inputs['observed_indices'].tolist(), center_observed=bool(0 in inputs['observed_indices'])))
                    for name in METHODS:
                        prediction, receipt = predict_record(name, models[name], inputs, cid, mid, rid, center)
                        receipt.update(factual_sha256=sha256(factual_file), truth_sha256=sha256(truth_file))
                        np.savez_compressed(output/'predictions'/f'{cid}__{name}.npz', **prediction)
                        receipts.append(receipt)
                pair_times.append(time.perf_counter()-began)
        execution = dict(protocol=PROTOCOL, plan=plan, methods=list(METHODS), interval_level=LEVEL,
                         interval_methods=['j'], config=config, cohort=cohort, mechanisms=mechanisms,
                         contexts=contexts, receipts=receipts, torch_threads=1, BLAS_threads=1,
                         point_support_semantics='ATE declared capability; numerical validity and failure are separate',
                         no_training_of_pretrained_J_or_kernel_parameters=True, no_hidden_outcomes_supplied=True)
        write_json(output/'execution.json', execution)
        began = time.perf_counter()
        write_views(output, execution)
        evaluation_seconds = time.perf_counter()-began
        manifest['timing'] = dict(generation_seconds=generation_times, pair_seconds=pair_times,
                                  evaluation_seconds=evaluation_seconds)
        manifest['output_sha256'] = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*'))
                                     if p.is_file() and p.name != 'run_manifest.json'}
        manifest['status'] = 'validating'
        write_json(output/'run_manifest.json', manifest)
        try:
            acceptance = validate_result(output, candidate=True)
        except Exception as exc:
            manifest.update(status='validation_failed', error=str(exc))
            write_json(output/'run_manifest.json', manifest)
            raise
        manifest.update(status='complete', wall_seconds_after_import=time.perf_counter()-started, acceptance=acceptance)
        fixed = max(0., manifest['wall_seconds_after_import']-sum(generation_times)-sum(pair_times)
                    -evaluation_seconds-acceptance['seconds'])
        manifest['estimate_400_seconds'] = fixed+20*np.mean(generation_times)+200*np.mean(pair_times)+(400/len(contexts))*(evaluation_seconds+acceptance['seconds'])
        write_json(output/'run_manifest.json', manifest)
        write_json(output/'acceptance_report.json', acceptance)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    parser.add_argument('--cohort', choices=('smoke', 'pilot', 'pilot_fresh', 'formal'), default='smoke')
    parser.add_argument('--validate', action='store_true', help='Read-only validation of an existing run; no predictions')
    args = parser.parse_args()
    result = validate_result(args.output) if args.validate else run_experiment(args.output, args.cohort)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
