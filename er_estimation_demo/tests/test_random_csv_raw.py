"""Regression contracts for old raw-output trainer + frozen random-DGP CSV."""
import ast
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import pfn_interference.model as demo

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw):
    values = dict(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')
    values.update(kw)
    return demo.DataConfig(**values)


def make_csv(tmp_path, **kw):
    params = dict(data_csv=tmp_path/'episodes.csv', num_datasets=8, data_seed=42,
                  generation_batch_size=3, validation_fraction=.25, split_seed=7)
    params.update(kw)
    return demo.prepare_episode_csv_files(cfg(), **params)


def stable_ast_fingerprint(node):
    """Match historical hashes independent of ast.dump's empty-list policy.

    Normalize a copy so inspection never changes the caller's syntax tree.
    Nonempty fields, constants and executable statements stay protected.
    """
    normalized = copy.deepcopy(node)
    for item in ast.walk(normalized):
        item._fields = tuple(name for name in item._fields
                             if getattr(item, name, None) != [])
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def test_stable_ast_fingerprint_ignores_empty_fields_but_detects_changed_code():
    original = ast.parse('def f():\n    return 1\n').body[0]
    before = ast.dump(original)
    without_empty_fields = copy.deepcopy(original)
    for item in ast.walk(without_empty_fields):
        item._fields = tuple(name for name in item._fields
                             if getattr(item, name, None) != [])
    changed = ast.parse('def f():\n    return 2\n').body[0]
    assert stable_ast_fingerprint(original) == stable_ast_fingerprint(without_empty_fields)
    assert stable_ast_fingerprint(original) != stable_ast_fingerprint(changed)
    assert ast.dump(original) == before




def test_random_dgp_has_correct_raw_inputs_and_effects():
    b = demo.generate_batch(cfg(), 16, seed=31, device=torch.device('cpu'))
    assert b['tokens'].shape == (16, 24, 5)
    for j,key in enumerate(('x','observed_treatment','y_obs','observed_exposure')):
        assert torch.equal(b['tokens'][...,j],b[key])
    assert b['prior_tau_level'].unique().numel() > 1
    assert set(b['prior_function_family'].flatten().tolist()) <= {0,1,4}
    assert b['prior_noise_sd'].unique().tolist() == [1.0]
    assert torch.equal(b['oracle_ite'][...,0],b['tau'])
    assert b['eta'].count_nonzero() == 0
    t,e,tt,ee=b['queries'].unbind(-1)
    torch.testing.assert_close(b['query_effect'],b['tau'][...,None]*(t-tt)+b['gamma'][...,None]*(e-ee))
    torch.testing.assert_close(b['query_effect'],b['outcome_a']-b['outcome_b'],atol=5e-6,rtol=5e-6)
    torch.testing.assert_close(b['y_obs'],b['structural_baseline']+b['tau']*b['observed_treatment']+b['gamma']*b['observed_exposure']+b['noise'])
    assert (b['adjacency'].sum(-1)>0).all()
    assert torch.equal(b['adjacency'],b['adjacency'].transpose(-1,-2))


def test_fixed_task_ids_do_not_depend_on_generation_chunk():
    import pfn_interference.simulation.csv_tasks as adapter
    one=adapter.generate_random_batch(cfg(),task_ids=range(9),seed=77,stream='test')
    parts=[adapter.generate_random_batch(cfg(),task_ids=range(i,min(i+4,9)),seed=77,stream='test') for i in range(0,9,4)]
    for key in one:
        assert torch.equal(one[key],torch.cat([p[key] for p in parts],0)),key


def test_csv_roundtrip_preserves_all_observed_and_label_tensors(tmp_path):
    b=demo.generate_batch(cfg(),2,seed=73,device=torch.device('cpu'))
    demo.save_episode_batch_csv(b,tmp_path/'a.csv')
    c, ids=demo.load_episode_batch_csv(tmp_path/'a.csv',device=torch.device('cpu'))
    assert ids == [0,1]
    for key in demo.CSV_TENSOR_KEYS:
        assert torch.equal(b[key].to(c[key].dtype),c[key]), key
    assert not any(k.startswith('prior_') for k in c)  # metadata is never model input


def test_random_csv_manifest_roundtrip_and_reuse_without_generation(tmp_path,monkeypatch):
    prepared=make_csv(tmp_path)
    import pfn_interference.simulation.csv_tasks as adapter
    mp=adapter.manifest_path(tmp_path/'episodes.csv')
    m=json.loads(mp.read_text())
    assert m['data_config']['dgp_version']=='random_lpe_v1'
    assert m['prior_config']['function_families']==['linear','polynomial','exponential']
    assert m['prior_config']['nointerference_prob']==.5
    assert m['complete'] is True
    assert len(m['training_dataset_ids'])==6 and len(m['validation_dataset_ids'])==2
    assert not set(m['training_dataset_ids']) & set(m['validation_dataset_ids'])
    assert len(m['task_metadata'])==8
    before=(tmp_path/'episodes.csv').read_bytes()
    def fail(*a,**k): raise AssertionError('CSV must be reused, not regenerated')
    monkeypatch.setattr(demo,'generate_batch',fail)
    second=make_csv(tmp_path,generation_batch_size=7)
    assert second['summary']==prepared['summary']
    assert before==(tmp_path/'episodes.csv').read_bytes()


@pytest.mark.parametrize('changed', ['seed','count','config'])
def test_random_csv_rejects_mismatched_identity(tmp_path,changed):
    make_csv(tmp_path)
    newcfg=cfg(random_nointerference_prob=.2) if changed=='config' else cfg()
    with pytest.raises(ValueError,match='manifest|identity|config'):
        demo.prepare_episode_csv_files(newcfg,data_csv=tmp_path/'episodes.csv',
            num_datasets=9 if changed=='count' else 8,data_seed=43 if changed=='seed' else 42,
            generation_batch_size=3,validation_fraction=.25,split_seed=7)


def test_random_csv_rejects_missing_or_truncated_manifest(tmp_path):
    make_csv(tmp_path)
    import pfn_interference.simulation.csv_tasks as adapter
    mp=adapter.manifest_path(tmp_path/'episodes.csv')
    mp.unlink()
    with pytest.raises(ValueError,match='manifest'):
        make_csv(tmp_path)


def test_ddp_checks_prepared_random_dgp_identity(tmp_path):
    make_csv(tmp_path)
    args=SimpleNamespace(data_csv=tmp_path/'episodes.csv',csv_datasets=8,data_seed=42,
        validation_fraction=.25,split_seed=7,overwrite_data_csv=False)
    runtime=demo.DistributedRuntime(rank=1,world_size=2)
    with pytest.raises(ValueError,match='manifest|identity|config'):
        demo.load_or_prepare_training_data(args,cfg(random_nointerference_prob=.2),runtime)
    b=demo.load_or_prepare_training_data(args,cfg(),runtime)
    assert b['train_batch']['tokens'].shape[0]==6


def test_training_visits_fixed_csv_no_generator_call(tmp_path,monkeypatch):
    make_csv(tmp_path)
    p=demo.load_existing_episode_csv_splits(tmp_path/'episodes.csv')
    model=demo.LocalNetworkQueryTransformer(demo.ModelConfig(d_model=16,num_layers=1,ffn_dim=32))
    opt=demo.create_optimizer(model.parameters())
    p['train_batch']['cepo_target']=demo.cepo_targets(p['train_batch'],treatment_prob=cfg().treatment_prob)
    before=p['train_batch']['tokens'].clone()
    def fail(*a,**k): raise AssertionError('Do not draw new training graphs inside epochs')
    monkeypatch.setattr(demo,'generate_batch',fail)
    for epoch in (1,2):
        loss=demo.train_epoch(model,opt,p['train_batch'],batch_size=2,training_seed=0,
                              epoch=epoch,device=torch.device('cpu'))
        assert torch.isfinite(torch.tensor(loss))
    assert torch.equal(before,p['train_batch']['tokens'])
    assert model.self_encoder[0].weight.grad is not None


def test_random_raw_loss_is_cepo_nll():
    b=demo.generate_batch(cfg(),2,seed=11,device=torch.device('cpu'))
    model=demo.LocalNetworkQueryTransformer(demo.ModelConfig(d_model=16,num_layers=1,ffn_dim=32))
    pred=model(b['tokens'],demo.cepo_queries(b),b['adjacency'])
    assert not any('residual' in k or 'outcome_scale' in k for k in pred)
    losses=demo.compute_training_losses(model,pred,b)
    expected=torch.stack([demo.gmm_nll_loss(pred['gmm_pi'][...,k,:],pred['gmm_mu'][...,k,:],
                         pred['gmm_sigma'][...,k,:],demo.cepo_targets(b)[...,k]) for k in range(4)]).mean()
    torch.testing.assert_close(losses['total_loss'],expected)


def test_legacy_checkpoint_configuration_does_not_become_random():
    old=demo.data_config_from_mapping({'n_units':24,'er_edge_probability':.3,'dgp_version':'gao_ding_design2_outcome'})
    assert old.dgp_version=='gao_ding_design2_outcome'
    b=demo.generate_batch(old,2,seed=1,device=torch.device('cpu'))
    torch.testing.assert_close(b['tau'],6+.2*torch.exp(b['x']))


def test_cli_exposes_dgp_and_uses_requested_ten_layer_default(monkeypatch):
    monkeypatch.setattr('sys.argv',['train','--dgp-version','random_lpe_v1','--nointerference-prob','.5'])
    args=demo.parse_args()
    assert args.dgp_version=='random_lpe_v1'
    assert args.d_model==128 and args.num_layers==10


def test_benchmark_paired_errors_use_same_supported_units():
    import importlib.util
    assert importlib.util.find_spec('pfn_interference.experiments.evaluate_random_csv_tables') is not None, 'Missing frozen raw-model evaluation entry'
    from pfn_interference.experiments.evaluate_random_csv_tables import paired_comparisons
    rows=[dict(dataset_id=1,unit_id=i,effect='direct',method=m,truth=0.,estimate=x,supported=s)
          for i,m,x,s in [(1,'PFN with interference',1.,True),(2,'PFN with interference',2.,True),
                          (1,'B',3.,True),(2,'B',999.,False)]]
    r=paired_comparisons(rows)[0]
    assert r['n_paired']==1 and r['pfn_rmse']==1. and r['method_rmse']==3.






def test_legacy_mode_cannot_accidentally_load_random_csv(tmp_path):
    make_csv(tmp_path)
    with pytest.raises(ValueError,match='random_lpe_v1|Random'):
        demo.prepare_episode_csv_files(demo.DataConfig(n_units=24,er_edge_probability=.3),
            data_csv=tmp_path/'episodes.csv',num_datasets=8,data_seed=42)
    args=SimpleNamespace(data_csv=tmp_path/'episodes.csv',csv_datasets=8,data_seed=42,
        validation_fraction=.25,split_seed=7,overwrite_data_csv=False)
    with pytest.raises(ValueError,match='random_lpe_v1|Random'):
        demo.load_or_prepare_training_data(args,demo.DataConfig(n_units=24,er_edge_probability=.3),
                                          demo.DistributedRuntime(rank=1,world_size=2))

