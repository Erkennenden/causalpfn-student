#!/usr/bin/env python3
"""Evaluate a frozen checkpoint under its saved training graph protocol.

Fixed-ER checkpoints reuse one saved network for independent data repeats.
Legacy resampled checkpoints retain their balanced four-family benchmark.
A-group checkpoints use epsilon SD=4; --test-lambda changes only test outcomes.
Localized DR-Lasso uses the manuscript-aligned adapted implementation and pilot defaults.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np

from evaluate_random_csv_tables import EFFECTS, supported, paired_comparisons, _table
from baseline_completeness import validate_baseline_results, ITE_METHODS, ATE_METHODS
from baselines.localized_config import add_localized_arguments, localized_benchmark_kwargs, LOCALIZED_PROTOCOL


def generate_matched_test_batch(config, task_ids, *, seed=2026, device='cpu'):
    """Use an independent data stream and preserve the checkpoint graph protocol."""
    from csv_random_dgp import BENCHMARK_STREAM, generate_random_batch
    fixed_er = getattr(config, 'graph_protocol', 'resampled') == 'fixed_er_v1'
    if not fixed_er and config.graph_family != 'mixed':
        raise ValueError('Matched four-family test requires mixed-graph training config.')
    batch = generate_random_batch(config, task_ids=task_ids, seed=seed,
        stream=BENCHMARK_STREAM, device=device, balanced=not fixed_er)
    if fixed_er:
        validate_fixed_graph_batch(config, batch)
    return batch


def validate_fixed_graph_batch(config, batch):
    """Reject a different network before any predictions or metrics are computed."""
    import torch
    from fixed_er_graph import adjacency_from_config, graph_metadata, adjacency_sha256
    expected_adjacency = adjacency_from_config(config)
    expected_hash = graph_metadata(config)['adjacency_sha256']
    adjacency = batch['adjacency']
    if adjacency.ndim != 3 or adjacency.shape[0] == 0 or tuple(adjacency.shape[1:]) != (config.n_units, config.n_units):
        raise ValueError('Fixed-ER adjacency batch has an invalid shape.')
    if any(not torch.equal(graph.detach().cpu(), expected_adjacency)
           or adjacency_sha256(graph) != expected_hash for graph in adjacency):
        raise ValueError('Test adjacency does not match the fixed graph saved in the checkpoint.')
    return expected_hash


def evaluation_config(training_config, test_lambda=None, test_gamma=None):
    """Override only outcome strength, preserving saved training graph and metadata."""
    if test_gamma is not None:
        if test_lambda is not None:
            raise ValueError('Choose --test-gamma or --test-lambda, not both.')
        if training_config.dgp_version not in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
            raise ValueError('--test-gamma requires an A-group DGP.')
        if training_config.dgp_version == 'a_group_random_coeff_v1':
            raise ValueError('Use --test-lambda to fix both coefficients after random-coefficient training.')
        return replace(training_config, dgp_version='a_group_random_gamma_v1',
                       interference_lambda=1., spillover_gamma=test_gamma)
    if test_lambda is None:
        return training_config
    if not math.isfinite(test_lambda) or test_lambda < 0:
        raise ValueError('test_lambda must be nonnegative and finite.')
    if training_config.dgp_version == 'a_group_random_coeff_v1':
        return replace(training_config, neighbor_beta=-3*test_lambda, spillover_gamma=-.9*test_lambda)
    if training_config.dgp_version != 'a_group_fixed_v1':
        raise ValueError('--test-lambda requires the A-group DGP.')
    return replace(training_config, interference_lambda=test_lambda)


def build_arg_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='musa:0')
    parser.add_argument('--graphs','--episodes',dest='graphs',type=int,default=1,
                        help='Independent data repeats; fixed ER uses one saved network for every repeat.')
    outcome = parser.add_mutually_exclusive_group()
    outcome.add_argument('--test-lambda',type=float,default=None,
                        help='Fix beta=-3*lambda and gamma=-.9*lambda; omit to use the saved training DGP.')
    outcome.add_argument('--test-gamma',type=float,default=None,
                        help='Fixed exposure coefficient; neighbor-X coefficient stays -3.')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--standard-ite-epochs','--hypersci-epochs',type=int,default=500)
    parser.add_argument('--hypersci-arm-samples',type=int,default=4096)
    parser.add_argument('--tnet-epochs',type=int,default=160)
    parser.add_argument('--nsi-min-support-mass',type=float,default=1.,
                        help='Keep matched protocol at 1.0; legacy all-methods wrapper uses 0.99.')
    parser.add_argument('--causalpfn-checkpoint',type=Path,default=Path(__file__).parent/'checkpoints/causalpfn_v0.pt')
    add_localized_arguments(parser)
    return parser


def main(argv=None):
    import torch
    from train_local_network_interference import load_checkpoint,data_config_from_mapping,resolve_device
    from csv_random_dgp import prior_from_data,BENCHMARK_STREAM,file_sha256
    from baselines.causalpfn_direct import load_causalpfn_checkpoint
    from evaluation.unified import evaluate_unified_benchmark,save_unified_benchmark,_metric_rows,_write_csv,_json_safe
    parser=build_arg_parser()
    args=parser.parse_args(argv)
    if args.graphs <= 0 or args.seed<0 or min(args.standard_ite_epochs,args.tnet_epochs)<=0 or args.hypersci_arm_samples<2:
        parser.error('Need positive episodes, a nonnegative seed, and positive baseline epochs.')
    if not math.isfinite(args.nsi_min_support_mass) or not 0 < args.nsi_min_support_mass <= 1:
        parser.error('NSI minimum support mass must be in (0,1].')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; never overwrite a previous benchmark.')
    torch.set_num_threads(1)
    device=resolve_device(args.device)
    model,optimizer,meta=load_checkpoint(path=args.checkpoint,device=device)
    del optimizer
    model.eval().requires_grad_(False)
    if (meta['data_config'].get('graph_protocol') == 'fixed_er_v1'
            and not meta['data_config'].get('fixed_graph_upper_hex')):
        parser.error('Fixed-ER checkpoint is missing its saved adjacency; refusing to resample it.')
    training_config=data_config_from_mapping(meta['data_config'])
    try:
        config=evaluation_config(training_config, args.test_lambda, args.test_gamma)
    except ValueError as exc:
        parser.error(str(exc))
    prior=prior_from_data(config)
    fixed_er = config.graph_protocol == 'fixed_er_v1'
    from evaluate_dual_sampler import validate_graph_count
    try:
        validate_graph_count(args.graphs, graph_protocol=config.graph_protocol)
    except ValueError as exc:
        parser.error(str(exc))
    if not fixed_er and config.graph_family!='mixed':
        parser.error('Matched four-family benchmark requires mixed-graph training config.')
    causal_device=torch.device('cpu') if device.type=='musa' else device
    causal=load_causalpfn_checkpoint(args.causalpfn_checkpoint,causal_device)
    causal.eval().requires_grad_(False)
    ck_hash=file_sha256(args.checkpoint)
    root=Path(__file__).parent
    source_files=[Path(__file__),root/'baseline_completeness.py',root/'evaluate_dual_sampler.py',root/'evaluate_random_csv_tables.py',root/'dual_sampler_reporting.py',root/'train_local_network_interference.py',root/'cepo.py',root/'random_lpe_prior.py',root/'a_group_dgp.py',root/'csv_random_dgp.py',root/'estimands.py',*(root/'baselines').rglob('*.py'),*(root/'evaluation').rglob('*.py'),*(root/'benchmark_reference').glob('*.py')]
    if fixed_er:
        source_files.append(root/'fixed_er_graph.py')
    manifest=dict(baseline_config=dict(standard_ite_epochs=args.standard_ite_epochs,tnet_epochs=args.tnet_epochs,
        hypersci=dict(arm_samples=args.hypersci_arm_samples,learning_rate=.001,balance_weight=.01,weight_decay=.01,
            hidden_dim=32,dropout=.5,encoder="gat",heads=2,layers=1,skip="123",device="cpu",pyg_version="2.6.1",
            adaptation="pair-edge; majority conditional assignment integration"),
        localized_protocol=LOCALIZED_PROTOCOL,
        **localized_benchmark_kwargs(args),nsi_min_support_mass=args.nsi_min_support_mass),required_ite_methods=ITE_METHODS,required_ate_methods=ATE_METHODS,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=ck_hash,
        prediction_protocol=meta['prediction_protocol'],checkpoint_epoch=meta['epoch'],checkpoint_role=meta['monitor_name'],
        training_data_config=asdict(training_config),
        training_lambda=(None if training_config.dgp_version == 'a_group_random_coeff_v1' else training_config.interference_lambda),
        test_lambda=(args.test_lambda if args.test_lambda is not None else
                     None if training_config.dgp_version == 'a_group_random_coeff_v1' else config.interference_lambda),
        outcome_shift=config != training_config,
        test_gamma=args.test_gamma,
        data_config=asdict(config),prior_config=asdict(prior),model_config=meta['model_config'],
        stream=BENCHMARK_STREAM,seed=args.seed,graphs=args.graphs,task_ids=list(range(args.graphs)),
        balanced=not fixed_er,function_shift=False,covariate_law='X iid N(0,1)',historical_score_comparable=False,sampler='matched_old_sampler',torch_version=torch.__version__,device=str(device),
        estimand='Exact-majority Direct/Spillover/Total; whole-graph ATE = average node ITE',
        training='frozen CSV, equal-weight mu00/mu10/mu11 GMM NLL; shared-mean contrasts; frozen PFN with interference',
        support='Method-specific support plus pairwise common-support summaries; unsupported stays NaN',
        causalpfn=dict(method='causalPFN',features=['X'],algorithm='S',effects=['direct'],checkpoint_sha256=file_sha256(args.causalpfn_checkpoint),device=str(causal_device)),
        source_sha256={str(f.relative_to(root)):file_sha256(f) for f in source_files})
    if fixed_er:
        from fixed_er_graph import graph_metadata
        manifest.update(fixed_graph=graph_metadata(config), episodes=args.graphs,
            unique_graph_count=1, graph_count_semantics='independent data episodes on one fixed network',
            graph_reference='same saved ER adjacency as training and validation; independent X/T/noise test stream')
    else:
        manifest.update(configuration_edge_swaps='1x edge_count',
            graph_reference='same OLD graph samplers as fixed-CSV training; independent balanced test stream')
    if config.dgp_version in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        from a_group_dgp import outcome_spec_for_config
        manifest['training_outcome_dgp'] = outcome_spec_for_config(training_config)
        manifest['outcome_dgp'] = outcome_spec_for_config(config)
        manifest['regime_protocol'] = 'A-group outcome; lambda=0 disables both neighbor-X and treatment-spillover terms'
    if training_config.dgp_version == 'a_group_random_coeff_v1':
        manifest['test_coefficients'] = dict(beta=config.neighbor_beta, gamma=config.spillover_gamma)
        if config.neighbor_beta is not None:
            manifest['within_training_coefficient_bounds'] = dict(
                beta=training_config.neighbor_beta_min <= config.neighbor_beta <= training_config.neighbor_beta_max,
                gamma=training_config.spillover_gamma_min <= config.spillover_gamma <= training_config.spillover_gamma_max)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'run_manifest.json').write_text(json.dumps(_json_safe(manifest),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pooled={'ite':[],'ate':[]}; input_hashes=[]; coverage=[]
    label = 'FIXED ER: one saved network' if fixed_er else 'MATCHED OLD-SAMPLER RAW'
    if config.dgp_version in ('a_group_random_gamma_v1', 'a_group_random_coeff_v1'):
        from a_group_dgp import outcome_description
        description = ('train: ' + outcome_description(manifest['training_outcome_dgp']) +
                       '; test: ' + outcome_description(manifest['outcome_dgp']))
    else:
        description = f'train lambda={training_config.interference_lambda:g}; test lambda={config.interference_lambda:g}'
    caption_description = description.replace('b_n', 'neighbor-X coefficient').replace('~', ' drawn from ')
    print(f'{label}; {description}; episodes={args.graphs}; checkpoint epoch={meta["epoch"]}; monitor={meta["monitor_name"]}; frozen test seed={args.seed}',flush=True)
    for task_id in range(args.graphs):
        random.seed(args.seed+task_id); np.random.seed(args.seed+task_id); torch.manual_seed(args.seed+task_id)
        b=generate_matched_test_batch(config,[task_id],seed=args.seed,device=device)
        h=hashlib.sha256()
        for k in ('tokens','adjacency','oracle_ite'):
            a=b[k].detach().cpu().contiguous().numpy()
            h.update(json.dumps([k,a.shape,str(a.dtype)]).encode());h.update(a.tobytes())
        print(f'{"episode" if fixed_er else "graph"} {task_id+1}/{args.graphs}, regime={int(b["regime"][0])}',flush=True)
        report=evaluate_unified_benchmark(model=model,batch=b,treatment_prob=.5,
            ate_bandwidth=3,ate_ridge=0.0,**localized_benchmark_kwargs(args),
            nsi_min_effective_donors=12.,nsi_min_support_mass=args.nsi_min_support_mass,
            standard_ite_epochs=args.standard_ite_epochs,standard_ite_hidden_dim=32,
            hypersci_arm_samples=args.hypersci_arm_samples,
            tnet_epochs=args.tnet_epochs,tnet_hidden_dim=64,
            causalpfn_model=causal,causalpfn_query_chunk_size=512,seed=args.seed+task_id)
        inventory = validate_baseline_results(report, n_units=config.n_units)
        coverage.extend(dict(dataset_id=task_id+1, **row) for row in inventory)
        save_unified_benchmark(report,args.output_dir/'graphs'/f'graph_{task_id:04d}')
        for level,raw in [('ite','ite_unit_results'),('ate','ate_graph_results')]:
            pooled[level].extend({**r,'dataset_id':task_id+1} for r in report[raw])
        input_record = dict(task_id=task_id,regime=int(b['regime'][0]),sha256=h.hexdigest())
        # Per-component hashes certify pairing without hiding outcome changes in one digest.
        input_record['component_sha256'] = {}
        for key in ('adjacency', 'x', 'observed_treatment', 'observed_exposure', 'noise', 'queries', 'y_obs', 'gamma', 'oracle_ite', 'prior_neighbor_coefficient'):
            array = b[key].detach().cpu().contiguous().numpy()
            digest = hashlib.sha256(json.dumps([array.shape, str(array.dtype)]).encode()+array.tobytes()).hexdigest()
            input_record['component_sha256'][key] = digest
        input_record['gamma'] = float(b['gamma'][0, 0])
        input_record['beta'] = float(b['prior_neighbor_coefficient'][0])
        if fixed_er:
            from fixed_er_graph import adjacency_sha256
            input_record.update(graph_family='er', adjacency_sha256=adjacency_sha256(b['adjacency'][0]))
        input_hashes.append(input_record)
    _write_csv(coverage, args.output_dir/'baseline_coverage.csv')
    summaries={};pairs={};tables=[]; readable=[]
    for level in ('ite','ate'):
        # Preserve total target counts but exclude explicitly unsupported predictions.
        rows=[r if supported(r) else {**r,'estimate':float('nan')} for r in pooled[level]]
        methods=tuple(dict.fromkeys(r['method'] for r in rows))
        summaries[level]=[r for r in _metric_rows(rows,methods=methods,effects=EFFECTS,
                         subset_name='method_specific_support') if r['n_total']]
        pairs[level]=paired_comparisons(rows)
        _write_csv(rows,args.output_dir/(level+'_unit_results.csv' if level=='ite' else 'ate_graph_results.csv'))
        _write_csv(summaries[level],args.output_dir/f'{level}_summary.csv')
        _write_csv(pairs[level],args.output_dir/f'{level}_paired_comparisons.csv')
        table=_table(summaries[level],level).replace('Random LPE, fixed-CSV PFN with interference:',
            (f'A group, {caption_description}' if config.dgp_version in ('a_group_fixed_v1', 'a_group_random_gamma_v1', 'a_group_random_coeff_v1') else 'Random LPE')+
            ('; fixed ER network, independent data repeats:' if fixed_er else '; normal matched-old sampler:')); tables.append(table)
        (args.output_dir/f'{level}_table.tex').write_text(table)
        readable.extend(['',f'===== {level.upper()} RESULTS =====','Method | Effect | MAE | RMSE | Bias | Mean Truth | Mean Pred | Support'])
        for r in summaries[level]:
            nums=[f'{float(r[k]):.6f}' if math.isfinite(float(r[k])) else 'NaN' for k in ('mae','rmse','bias','mean_truth','mean_estimate')]
            readable.append(' | '.join([r['method'],r['effect'],*nums,f'{r["n_supported"]}/{r["n_total"]}']))
    if file_sha256(args.checkpoint)!=ck_hash:
        raise RuntimeError('Checkpoint changed during test.')
    (args.output_dir/'results.json').write_text(json.dumps(_json_safe(dict(manifest=manifest,input_hashes=input_hashes,summaries=summaries,paired_comparisons=pairs)),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'tables.tex').write_text('\n'.join(tables))
    text='\n'.join(readable)+'\n'; (args.output_dir/'summary.txt').write_text(text)
    print(text,flush=True)
    from dual_sampler_reporting import write_family_reports
    write_family_reports(args.output_dir, sampler='matched_old_sampler')
    print(('Fixed ER 测试完成：' if fixed_er else 'Matched old-sampler 测试完成：')+str(args.output_dir.resolve()),flush=True)


if __name__=='__main__':
    main()
