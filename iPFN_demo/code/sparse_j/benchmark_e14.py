"""Run E1.4 RGG scenarios using unchanged frozen models and report formats."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark import (ROOT, ALL_METHODS, DEFAULT_METHODS, configuration,
                       load_methods, predict_method, sha, source_directory)
from evaluate import evaluate_run
from make_table import write_csv
from pfn_social.e14_rgg import SCENARIOS, generate_rgg
from pfn_social.e14_oracle import predict_oracle
from pfn_social.e14_followup import REPLICATION, MATCHED, generate_matched
from pfn_social.intervals import predict_intervals
from pfn_social.phase6_limited_outcomes import make_observed_draw
from pfn_social.phase6_sparse_classical import pack


def effects(mu):
    return np.stack((mu[:, 2]-mu[:, 0], mu[:, 3]-mu[:, 2], mu[:, 3]-mu[:, 0]), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mechanisms', type=int, default=20)
    parser.add_argument('--design', choices=('original','replication','degree-matched'), default='original')
    parser.add_argument('--mechanism-start', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--families', type=int, nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=list(DEFAULT_METHODS))
    args = parser.parse_args()
    if args.mechanisms < 1 or args.repeats < 1 or args.mechanism_start < 0:
        parser.error('Positive mechanism/repeat counts required')
    if len(set(args.methods)) != len(args.methods) or (args.families and len(set(args.families)) != len(args.families)):
        parser.error('Duplicate methods/families')
    available = {'original':SCENARIOS, 'replication':REPLICATION, 'degree-matched':MATCHED}[args.design]
    if args.families and not set(args.families) <= {s.setting_id for s in available}:
        parser.error('Families do not belong to design')
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('truth', 'factual', 'predictions', 'graphs'):
        (destination/name).mkdir()
    config = configuration()
    scenarios = [s for s in available if not args.families or s.setting_id in args.families]
    protocol = dict(task='E1.4', design=args.design, mechanism_start=args.mechanism_start, scenarios=[asdict(s) for s in scenarios],
        mechanisms_per_scenario=args.mechanisms, assignment_noise_repeats=args.repeats,
        observed_outcomes=[20, 50, 100], methods=args.methods, interval_level=.95,
        no_isolates='whole point cloud rejection; no added edges; disconnected graphs retained',
        outcome_mask='independent SRS without replacement, fixed within mechanism and nested across M',
        training='none; existing frozen J/kernel assets unchanged',
        rng='separate SeedSequence channels; independent family/mechanism seeds',
        estimand='finite-network design-average direct10-00, treated spillover11-10, total11-00',
        baseline_intervals='unsupported by upstream implementation; do not interpret as zero coverage',
        coverage_target=.90, bias='also report signed bias; template uses mean absolute within-mechanism bias')
    if args.design == 'degree-matched':
        for spec in protocol['scenarios']:
            spec.update(radius=None, target_degree=18, radius_rule='midpoint of distance ranks 1800 and 1801', shared_dgp_family=90)
    protocol['pairing'] = 'Shared X,beta,T,noise,masks within pairs' if args.design == 'degree-matched' else 'Independent mechanisms'
    # Save the protocol before outcomes or scores are generated.
    (destination/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    torch.set_num_threads(1)
    receipts, graph_rows, ite_rows, arm_rows, oracle_rows = [], [], [], [], []
    start = time.perf_counter()
    with source_directory(), threadpool_limits(limits=1):
        methods = load_methods(args.methods, config)
        load_seconds = time.perf_counter() - start
        for scenario in scenarios:
            for mid in range(args.mechanism_start, args.mechanism_start+args.mechanisms):
                generator = generate_matched if args.design == 'degree-matched' else generate_rgg
                mechanism, positions, graph_info = generator(scenario, mid)
                family, truth = scenario.setting_id, mechanism.parent.truth
                true_ite = effects(truth)
                graph_rows.append(graph_info)
                np.savez_compressed(destination/'graphs'/f'f{family}_m{mid}.npz',
                    positions=positions, adjacency=mechanism.parent.adjacency,
                    observed_permutation=mechanism.observation_permutation)
                np.savez_compressed(destination/'truth'/f'f{family}_m{mid}.npz', arm_means=truth)
                for repeat in range(args.repeats):
                    rows, _ = make_observed_draw(mechanism, repeat)
                    for row in rows:
                        identity = f"f{family}_m{mid}_r{repeat}_observed{row['m_observed']}"
                        inputs = {k: v.numpy().copy() for k, v in row['batch'].items()}
                        np.savez_compressed(destination/'factual'/(identity+'.npz'), **inputs)
                        a, t, ids = inputs['adjacency'], inputs['treatment'], inputs['observed_indices']
                        state = 2*t.astype(int)+(a@t > a.sum(1)//2).astype(int)
                        arm_rows.append(dict(context_id=identity, family=family, mechanism=mid, repeat=repeat,
                            m_observed=row['m_observed'], **{f'observed_{s:02b}': int((state[ids] == s).sum()) for s in range(4)}))
                        oracle = predict_oracle(true_h=mechanism.parent.h, **inputs)
                        oracle_ite = effects(oracle['arm_means'])
                        for k, effect in enumerate(('direct', 'spillover', 'total')):
                            truth_ate = true_ite[:, k].mean()
                            oracle_rows.append(dict(context_id=identity, family=family, mechanism=mid, repeat=repeat,
                                m_observed=row['m_observed'], effect=effect,
                                ate_error=float(oracle['ate_effects'][k]-truth_ate),
                                covered=bool(oracle['ate_lower'][k] <= truth_ate <= oracle['ate_upper'][k]),
                                width=float(oracle['ate_upper'][k]-oracle['ate_lower'][k]),
                                ite_mse=float(np.mean((oracle_ite[:, k]-true_ite[:, k])**2))))
                        for name, model in methods.items():
                            seed = 651013+1000*family+100*mid+repeat
                            began = time.perf_counter()
                            failure = interval_failure = None
                            try:
                                prediction = predict_method(name, model, inputs, seed)
                            except Exception as error:
                                prediction = pack(n=len(t))
                                failure = dict(type=type(error).__name__, message=str(error))
                            elapsed = time.perf_counter()-began
                            intervals = dict(ate_lower=np.full(3, np.nan), ate_upper=np.full(3, np.nan),
                                interval_level=np.array(.95), interval_kind=np.array('unsupported'),
                                ate_interval_supported=np.full(3, name == 'j'))
                            began = time.perf_counter()
                            if name == 'j':
                                try:
                                    intervals.update(predict_intervals(model, **inputs, level=.95))
                                except Exception as error:
                                    interval_failure = dict(type=type(error).__name__, message=str(error))
                            np.savez_compressed(destination/'predictions'/(identity+'__'+name+'.npz'),
                                **{k: prediction[k] for k in ('arm_means', 'ite_effects', 'ate_effects',
                                    'arm_supported', 'ite_supported', 'ate_supported')}, **intervals)
                            receipts.append(dict(context_id=identity, method=name, fit_seed=seed,
                                online_seconds=elapsed, arm_supported=prediction['arm_supported'].tolist(),
                                ite_supported=prediction['ite_supported'].tolist(),
                                ate_supported=prediction['ate_supported'].tolist(), failure=failure,
                                interval_level=.95, interval_seconds=time.perf_counter()-began,
                                interval_failure=interval_failure, ate_interval_supported=intervals['ate_interval_supported'].tolist()))
                            for k, effect in enumerate(('direct', 'spillover', 'total')):
                                supported = bool(prediction['ite_supported'][k]) and failure is None
                                error = prediction['ite_effects'][:, k]-true_ite[:, k]
                                ite_rows.append(dict(context_id=identity, family=family, mechanism=mid, repeat=repeat,
                                    m_observed=row['m_observed'], method=name, effect=effect, supported=supported,
                                    node_count=len(t), squared_error_sum=float(error@error) if supported else np.nan,
                                    error_sum=float(error.sum()) if supported else np.nan))
                # Checkpoint receipts after each completed mechanism.
                manifest = dict(config=config, protocol=protocol, methods=args.methods, cohort='e14_rgg',
                    model_load_seconds=load_seconds, torch_threads=1, BLAS_threads=1, receipts=receipts,
                    interval_level=.95, interval_methods=['j'] if 'j' in methods else [],
                    no_training_of_pretrained_J_or_kernel_parameters=True, no_hidden_outcomes_supplied=True)
                (destination/'execution.json').write_text(json.dumps(manifest), encoding='utf-8')
                print(f'family {family}: mechanism {mid+1}/{args.mechanisms}; '
                      f'{len(receipts)} method-contexts; {time.perf_counter()-start:.1f}s', flush=True)
    configuration()  # Detect any changed model asset after execution.
    write_csv(destination/'graph_diagnostics.csv', graph_rows)
    write_csv(destination/'arm_diagnostics.csv', arm_rows)
    evaluate_run(destination)
    write_csv(destination/'evaluation'/'ite_context_metrics.csv', ite_rows)
    write_csv(destination/'evaluation'/'oracle_context_metrics.csv', oracle_rows)
    provenance = dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
        for name in ('numpy', 'torch', 'scipy', 'networkx', 'matplotlib', 'threadpoolctl')},
        elapsed_seconds=time.perf_counter()-start, source_sha256={str(p.relative_to(ROOT)): sha(p)
            for p in [ROOT/'benchmark_e14.py', ROOT/'pfn_social/e14_rgg.py', ROOT/'pfn_social/e14_oracle.py', ROOT/'pfn_social/e14_followup.py', ROOT/'evaluate.py',
                      ROOT/'make_figures.py', ROOT/'make_table.py']})
    (destination/'runtime.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(destination, flush=True)


if __name__ == '__main__':
    main()
