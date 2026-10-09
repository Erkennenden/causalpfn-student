import math
from dataclasses import asdict
import pytest
import torch
import pfn_interference.model as m

def fixture():
    return {"tokens":torch.zeros(1,2,5), "design_treatment_prob":torch.tensor([.5]), "degree":torch.tensor([[1.,2.]]),
            "structural_baseline":torch.full((1,2),2.), "tau":torch.full((1,2),3.),
            "gamma":torch.full((1,2),-1.), "eta":torch.zeros(1,2),
            "query_effect":torch.full((1,2,3),999.), "y_obs":torch.full((1,2),999.)}

def test_cepo_labels_are_exact_arm_means_not_noisy_outcomes_or_sampled_contrasts():
    b=fixture()
    assert hasattr(m,"cepo_targets"), "CEPO supervision is missing"
    torch.testing.assert_close(m.cepo_targets(b),torch.tensor([[[2.,1.,5.,4.],[5/3,1.,14/3,4.]]]))
    b['query_effect'].zero_(); b['y_obs'].zero_()
    torch.testing.assert_close(m.cepo_targets(b),torch.tensor([[[2.,1.,5.,4.],[5/3,1.,14/3,4.]]]))
    b["design_treatment_prob"]=torch.tensor([.25])
    torch.testing.assert_close(m.cepo_targets(b,treatment_prob=.25),torch.tensor([[[2.,1.,5.,4.],[1.8,1.,4.8,4.]]]))

def test_mu_nll_uses_absolute_cepo_levels_and_not_effect_labels():
    model=m.LocalNetworkQueryTransformer(m.ModelConfig(d_model=8,num_heads=2,num_layers=1,ffn_dim=16,gmm_n_components=1))
    b=fixture(); target=torch.tensor([[[2.,1.,5.,4.],[5/3,1.,14/3,4.]]])
    mean=target.unsqueeze(-1).clone().requires_grad_()
    pred={'gmm_pi':torch.ones_like(mean),'gmm_mu':mean,'gmm_sigma':torch.ones_like(mean)}
    loss=m.compute_training_losses(model,pred,b)['total_loss']
    assert loss.item()==pytest.approx(.5*math.log(2*math.pi),abs=1e-6)
    b['structural_baseline']+=2
    shifted=m.compute_training_losses(model,pred,b)['total_loss']
    assert shifted.item()==pytest.approx(loss.item()+2.,abs=1e-6)
    shifted.backward(); assert torch.all(mean.grad<0)

def test_model_and_benchmark_reuse_four_mu_means_for_all_effects():
    from pfn_interference.metrics.unified import predict_pfn_majority_ite
    assert hasattr(m,'cepo_queries'), 'CEPO query interface is missing'
    b=m.generate_batch(m.DataConfig(n_units=10,er_edge_probability=.4),2,seed=91,device=torch.device('cpu'))
    model=m.LocalNetworkQueryTransformer(m.ModelConfig(d_model=8,num_heads=2,num_layers=1,ffn_dim=16)).eval()
    with torch.no_grad():
        p=model(b['tokens'],m.cepo_queries(b),b['adjacency'])
        mu=(p['gmm_pi']*p['gmm_mu']).sum(-1)
    got=predict_pfn_majority_ite(model=model,batch=b,treatment_prob=.5)
    torch.testing.assert_close(got[...,0],mu[...,2]-mu[...,0])
    torch.testing.assert_close(got[...,1],mu[...,3]-mu[...,2])
    torch.testing.assert_close(got[...,2],mu[...,3]-mu[...,0])
    torch.testing.assert_close(got[...,2],got[...,0]+got[...,1],atol=1e-6,rtol=1e-6)
    metrics=m.compute_query_metrics(model,p,b)
    assert 'cepo_gmm_nll_macro' in metrics
    assert 'direct_effect_gmm_nll' not in metrics
    assert 'direct_effect_coverage_90' not in metrics

def test_old_effect_checkpoint_is_rejected_before_state_loading(tmp_path):
    path=tmp_path/'old.pt'
    torch.save({'data_config':asdict(m.DataConfig()),'model_config':asdict(m.ModelConfig())},path)
    with pytest.raises(ValueError,match='CEPO|cepo|effect'):
        m.load_checkpoint(path=path,device=torch.device('cpu'))

@pytest.mark.parametrize('dgp_version,prob',[('gao_ding_design2_outcome',.25),('a_group_fixed_v1',.5)])
def test_generated_design_probability_controls_standalone_loss(dgp_version,prob):
    b=m.generate_batch(m.DataConfig(n_units=24,er_edge_probability=.4,
        treatment_prob=prob,dgp_version=dgp_version),2,seed=91,device=torch.device('cpu'))
    expected=m.cepo_targets(b,treatment_prob=prob)
    mean=expected.unsqueeze(-1)
    pred={'gmm_pi':torch.ones_like(mean),'gmm_mu':mean,'gmm_sigma':torch.ones_like(mean)}
    loss=m.compute_training_losses(None,pred,b)['total_loss']
    assert loss.item()==pytest.approx(.5*math.log(2*math.pi),abs=1e-6)
    with pytest.raises(ValueError,match='design|probability'):
        m.cepo_targets(b,treatment_prob=1-prob/2)


def test_missing_design_is_rejected_for_unlabelled_standalone_batch():
    b=fixture();b.pop('design_treatment_prob',None)
    with pytest.raises(ValueError,match='treatment_prob|design'):
        m.cepo_targets(b)
    target=m.cepo_targets(b,treatment_prob=.25)
    b['cepo_target']=target
    torch.testing.assert_close(m.cepo_targets(b),target)


def test_csv_smoke_entry_attaches_cepo_labels(tmp_path, monkeypatch):
    original=m.DataConfig
    monkeypatch.setattr(m,'N_UNITS',10)
    monkeypatch.setattr(m,'DataConfig',lambda **kwargs: original(**dict({"er_edge_probability":.4},**kwargs)))
    checkpoint=tmp_path/'cepo_smoke.pt'
    m.run_smoke_test(torch.device('cpu'),checkpoint)
    payload=torch.load(checkpoint,weights_only=False)
    assert payload['prediction_protocol']==m.PREDICTION_PROTOCOL


@pytest.mark.parametrize('prob', [.25, .5])
def test_default_dgp_cepo_levels_exclude_realized_observation_noise(prob):
    b=m.generate_batch(m.DataConfig(n_units=10,er_edge_probability=.4,
        treatment_prob=prob),16,seed=2026,device=torch.device('cpu'))
    x=b['x']; degree=b['degree']
    neighbor_x=torch.bmm(b['adjacency'],x.unsqueeze(-1)).squeeze(-1)/degree
    baseline=1-x-3*neighbor_x
    expected=torch.empty((*x.shape,4))
    # Enumerate the treatment counts independently of the CEPO label helper.
    for row in range(x.shape[0]):
        for node in range(x.shape[1]):
            d=int(degree[row,node]); moments=[]
            for high in (False,True):
                counts=[k for k in range(d+1) if (k>d//2)==high]
                weights=[math.comb(d,k)*prob**k*(1-prob)**(d-k) for k in counts]
                moments.append(sum(k/d*w for k,w in zip(counts,weights))/sum(weights))
            mu00=baseline[row,node]-.9*moments[0]
            mu01=baseline[row,node]-.9*moments[1]
            mu10=mu00+6+.2*torch.exp(x[row,node])
            mu11=baseline[row,node]+6+.2*torch.exp(x[row,node])-.9*moments[1]
            expected[row,node]=torch.stack((mu00,mu01,mu10,mu11))
    torch.testing.assert_close(m.cepo_targets(b),expected,atol=2e-6,rtol=1e-6)


def test_old_csv_and_its_matching_tensor_cache_cannot_bypass_label_validation(tmp_path):
    import hashlib
    b=m.generate_batch(m.DataConfig(n_units=10,er_edge_probability=.4),2,
        seed=2026,device=torch.device('cpu'))
    path=tmp_path/'old.csv'
    m.save_episode_batch_csv(b,path)
    path.write_text(path.read_text().replace(m.CSV_SCHEMA_VERSION,
        'majority_arm_v11_random_lpe_csv_raw'))
    identity={'version':1,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'size':path.stat().st_size,'expected_count':2}
    torch.save({'identity':identity,'batch':b,'ids':[0,1]},
        path.with_name(path.name+'.tensor_v1.pt'))
    with pytest.raises(ValueError,match='schema|Schema|version'):
        m.load_cached_episode_csv(path,2)


def test_old_default_cepo_checkpoint_with_noisy_labels_is_rejected(tmp_path):
    path=tmp_path/'old_default_cepo.pt'
    torch.save({'prediction_protocol':m.PREDICTION_PROTOCOL,
        'data_config':asdict(m.DataConfig()),'model_config':asdict(m.ModelConfig())},path)
    with pytest.raises(ValueError,match='noise|label|retrain'):
        m.load_checkpoint(path=path,device=torch.device('cpu'))
