"""Sampler comparisons with explicitly versioned, reviewed DGP updates."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import train_local_network_interference as demo

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    assert importlib.util.find_spec(name) is not None, f'Missing dual-test component: {name}'
    return __import__(name)


def test_matched_test_uses_training_sampler_and_isolates_configuration_shift():
    matched = module('evaluate_matched_old_sampler')
    from csv_random_dgp import BENCHMARK_STREAM, TRAIN_STREAM, generate_random_batch, prior_from_data
    from benchmark_reference.prior import PriorConfig, generate_tasks
    from dataclasses import asdict
    config = demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')
    old = matched.generate_matched_test_batch(config, range(16), seed=2026, device='cpu')
    training_sampler = generate_random_batch(config, task_ids=range(16), seed=2026,
        stream=BENCHMARK_STREAM, balanced=True)
    historical = generate_tasks(PriorConfig(**asdict(prior_from_data(config))), range(16),
        seed=2026, stream=BENCHMARK_STREAM, balanced=True)
    for key in old:
        assert torch.equal(old[key], training_sampler[key]), key
    is_configuration = old['graph_type'] == demo.CONFIGURATION_GRAPH
    assert is_configuration.sum().item() == 4
    for key in old:
        assert torch.equal(old[key][~is_configuration], historical[key][~is_configuration]), key
    assert not torch.equal(old['adjacency'][is_configuration], historical['adjacency'][is_configuration])
    # Original failed-swap handling can redraw degrees in the old sampler;
    # each suite must retain its own degree-dependent majority-effect truth.
    for key in ('x', 'tau', 'gamma', 'noise', 'observed_treatment'):
        assert torch.equal(old[key], historical[key]), key
    train = generate_random_batch(config, task_ids=range(16), seed=2026, stream=TRAIN_STREAM)
    assert not torch.equal(old['x'], train['x'])


def test_family_summaries_pool_errors_and_keep_unsupported_targets():
    reporting = module('dual_sampler_reporting')
    rows=[]
    for dataset in range(1,65):
        configuration=(dataset-1)//4 % 4 == 1
        for method in ('PFN with interference', 'B'):
            for unit in (1,2):
                estimate=2. if configuration else 1.
                if method=='B': estimate*=2
                supported=not (configuration and method=='B' and unit==2)
                rows.append(dict(dataset_id=dataset,unit_id=unit,method=method,effect='direct',
                    truth=0.,estimate=estimate if supported else 999.,supported=supported))
    summaries,pairs=reporting.summarize_rows(rows,graph_count=64,level='ite',sampler='matched_old_sampler')
    index={(r['graph_family'],r['method']):r for r in summaries}
    assert index['all','PFN with interference']['n_total']==128
    assert index['all','PFN with interference']['rmse']==pytest.approx(math.sqrt(1.75))
    assert index['configuration','PFN with interference']['rmse']==2.
    assert index['configuration','PFN with interference']['n_graphs']==16
    assert index['configuration','B']['n_total']==32
    assert index['configuration','B']['n_supported']==16
    assert index['configuration','B']['rmse']==4.
    pair=next(r for r in pairs if r['graph_family']=='configuration' and r['method']=='B')
    assert pair['n_paired']==16 and pair['pfn_rmse']==2. and pair['method_rmse']==4.


def test_family_ate_summary_uses_saved_whole_graph_ate():
    reporting = module('dual_sampler_reporting')
    rows=[dict(dataset_id=i,method='PFN with interference',effect='direct',truth=0.,estimate=float(i),supported=True)
          for i in range(1,17)]
    summaries,_=reporting.summarize_rows(rows,graph_count=16,level='ate',sampler='historical_c')
    config=next(r for r in summaries if r['graph_family']=='configuration')
    assert config['n_total']==4
    assert config['rmse']==pytest.approx(math.sqrt((25+36+49+64)/4))


@pytest.mark.parametrize('count',[0,4,63,65])
def test_dual_evaluation_rejects_unbalanced_graph_counts(count):
    dual = module('evaluate_dual_sampler')
    with pytest.raises(ValueError,match='16'):
        dual.validate_graph_count(count)






def test_dual_manifest_comparison_rejects_different_checkpoints():
    reporting=module('dual_sampler_reporting')
    old=dict(checkpoint_sha256='checkpoint-A',model_config={},data_config={},prior_config={},
        graphs=64,seed=2026,stream='test-stream',balanced=True,function_shift=False,
        causalpfn={'checkpoint_sha256':'causal-A'})
    historical={**old,'checkpoint_sha256':'checkpoint-B'}
    with pytest.raises(ValueError,match='checkpoint_sha256'):
        reporting.validate_pair_manifests(old,historical)
    historical={**old,'causalpfn':{'checkpoint_sha256':'causal-B'}}
    with pytest.raises(ValueError,match='CausalPFN'):
        reporting.validate_pair_manifests(old,historical)


