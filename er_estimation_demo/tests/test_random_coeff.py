"""Two task-level coefficients, paired lambda tests and immutable caches."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch
import pfn_interference.model as demo
from pfn_interference.simulation.csv_tasks import generate_random_batch, select_two_regime_split_ids
from pfn_interference.experiments.evaluate_matched_old_sampler import evaluation_config, generate_matched_test_batch


def config(**kwargs):
    return demo.DataConfig(n_units=24, er_edge_probability=.25,
        graph_protocol='fixed_er_v1', dgp_version='a_group_random_coeff_v1', **kwargs)


def test_two_independent_task_coefficients_preserve_input_streams():
    cfg=config()
    b=generate_random_batch(cfg, task_ids=list(range(64)), seed=17)
    old=generate_random_batch(replace(cfg,dgp_version='a_group_random_gamma_v1'),task_ids=list(range(64)),seed=17)
    beta=b['prior_neighbor_coefficient']; gamma=b['gamma'][:,0]
    assert beta.min()>=-6 and beta.max()<=0 and beta.unique().numel()==64
    assert gamma.min()>=-3.6 and gamma.max()<=0
    assert not torch.allclose(beta/6, gamma/3.6)
    assert abs(np.corrcoef(beta.numpy(),gamma.numpy())[0,1])<.4
    for key in ('x','adjacency','noise','observed_treatment','observed_exposure','queries','gamma'):
        assert torch.equal(b[key],old[key]),key
    x,t,e=(b[k] for k in ('x','observed_treatment','observed_exposure'))
    torch.testing.assert_close(b['y_obs'],1-x+beta[:,None]*b['neighbor_x']+(6+.2*x.exp())*t+gamma[:,None]*e+b['noise'])
    torch.testing.assert_close(b['oracle_ite'],old['oracle_ite'])
    assert b['tokens'].shape[-1]==5
    for offset in (1,9,31):
        single=generate_random_batch(cfg,task_ids=[offset],seed=17)
        for key in b: assert torch.equal(b[key][offset],single[key][0]),key


@pytest.mark.parametrize('strength',[0.,1.,2.,3.])
def test_lambda_changes_both_coefficients_and_preserves_pairing(strength):
    train=config()
    cfg=evaluation_config(train,test_lambda=strength)
    assert cfg.neighbor_beta == -3*strength
    assert cfg.spillover_gamma == -.9*strength
    assert train.neighbor_beta is None and train.spillover_gamma is None
    b=generate_matched_test_batch(cfg,[0])
    base=generate_matched_test_batch(evaluation_config(train,test_lambda=1),[0])
    for key in ('x','adjacency','noise','observed_treatment','observed_exposure','queries'):
        assert torch.equal(b[key],base[key]),key
    x,t,e=(b[k] for k in ('x','observed_treatment','observed_exposure'))
    torch.testing.assert_close(b['y_obs'],1-x-3*strength*b['neighbor_x']+(6+.2*x.exp())*t-.9*strength*e+b['noise'])
    torch.testing.assert_close(b['oracle_ite'][...,0],base['oracle_ite'][...,0])
    torch.testing.assert_close(b['oracle_ite'][...,1],strength*base['oracle_ite'][...,1])
    torch.testing.assert_close(b['oracle_ite'][...,2],b['oracle_ite'][...,0]+b['oracle_ite'][...,1])
    if strength==0: assert torch.count_nonzero(b['oracle_ite'][...,1])==0
    assert demo.data_config_from_mapping(json.loads(json.dumps(asdict(cfg))))==cfg


@pytest.mark.parametrize('kwargs',[
    {'neighbor_beta_min':0,'neighbor_beta_max':-1}, {'neighbor_beta_min':float('nan')},
    {'neighbor_beta_max':float('inf')}, {'neighbor_beta':float('nan')},
    {'neighbor_beta':-3}, {'spillover_gamma':-.9}])
def test_invalid_or_half_fixed_coefficients_are_rejected(kwargs):
    with pytest.raises(ValueError): config(**kwargs)


def test_csv_and_ddp_keep_random_coefficients_and_reject_stale_bank(tmp_path):
    cfg=config();path=tmp_path/'coeff.csv'
    kwargs=dict(data_csv=path,num_datasets=8,data_seed=17,generation_batch_size=2,validation_fraction=.25,split_seed=0)
    demo.prepare_episode_csv_files(cfg,**kwargs)
    loaded,ids=demo.load_episode_batch_csv(path,device=torch.device('cpu'))
    expected=generate_random_batch(cfg,task_ids=ids,seed=17)
    for key in demo.CSV_TENSOR_KEYS: assert torch.equal(loaded[key],expected[key].to(loaded[key].dtype)),key
    identity=json.loads(path.with_suffix('.csv.manifest.json').read_text())['identity']
    assert identity['outcome_dgp']['neighbor_distribution']=={'family':'uniform','low':-6.,'high':0.}
    before=path.read_bytes()
    with pytest.raises(ValueError,match='identity|config'):
        demo.prepare_episode_csv_files(replace(cfg,neighbor_beta_min=-9),**kwargs)
    assert path.read_bytes()==before
    shards=tmp_path/'shards';demo.prepare_ddp_train_shards(path,shard_dir=shards,world_size=2)
    for rank in range(2):
        b,ids=demo.load_episode_batch_csv(shards/f'train_rank{rank}.csv',device=torch.device('cpu'))
        expected=generate_random_batch(cfg,task_ids=ids,seed=17)
        for key in ('tokens','structural_baseline','gamma','oracle_ite'):
            assert torch.equal(b[key],expected[key].to(b[key].dtype)),key


def test_exact_40960_training_5120_validation_split():
    metadata=[dict(task_id=i,regime=3,graph_type=0) for i in range(46080)]
    train,val,summary=select_two_regime_split_ids(metadata,validation_fraction=5120/46080,split_seed=0)
    assert len(train)==40960 and len(val)==5120
    assert not set(train)&set(val)
    assert set(train)|set(val)==set(range(46080))


@pytest.mark.parametrize('strength',[None,0,3])
def test_evaluator_manifest_does_not_label_random_coefficients_as_lambda_one(tmp_path,monkeypatch,strength):
    import pfn_interference.baselines.causalpfn_direct as causal
    import pfn_interference.metrics.unified as unified
    import pfn_interference.experiments.evaluate_matched_old_sampler as entry
    cfg=config();model=torch.nn.Linear(1,1)
    ckpt=tmp_path/'model.pt';ckpt.write_bytes(b'loader boundary fixture')
    metadata=dict(data_config=asdict(cfg),model_config={},epoch=1,monitor_name='point',prediction_protocol=demo.PREDICTION_PROTOCOL)
    monkeypatch.setattr(demo,'load_checkpoint',lambda **kwargs:(model,None,metadata))
    monkeypatch.setattr(causal,'load_causalpfn_checkpoint',lambda *args:model)
    class BeforeFit(Exception):pass
    def stop(**kwargs):raise BeforeFit
    monkeypatch.setattr(unified,'evaluate_unified_benchmark',stop)
    args=['--checkpoint',str(ckpt),'--output-dir',str(tmp_path/'result'),'--device','cpu','--causalpfn-checkpoint',str(ckpt)]
    if strength is not None:args+=['--test-lambda',str(strength)]
    with pytest.raises(BeforeFit):entry.main(args)
    manifest=json.loads((tmp_path/'result/run_manifest.json').read_text())
    assert manifest['test_lambda'] is None if strength is None else manifest['test_lambda']==strength
    assert manifest['test_coefficients']['beta']==(None if strength is None else -3*strength)
