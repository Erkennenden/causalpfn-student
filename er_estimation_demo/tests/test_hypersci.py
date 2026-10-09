import itertools
import numpy as np
import pytest
import torch

from baselines.hypersci import (pair_incidence, make_model, build_pair_cache,
                            predict_root_samples, conditional_assignments,
                            integrate_majority)


def fixture_graph():
    a=torch.zeros(5,5)
    for i,j in ((0,1),(0,2),(0,4),(1,2),(2,3),(3,4)):
        a[i,j]=a[j,i]=1
    return a


def test_incidence_preserves_exactly_original_pair_edges():
    a=fixture_graph()
    incidence,pairs=pair_incidence(a)
    assert pairs.shape==(6,2)
    assert incidence.shape==(2,12)
    assert torch.equal(torch.bincount(incidence[1]),torch.full((6,),2))
    reconstructed=torch.zeros_like(a)
    reconstructed[pairs[:,0],pairs[:,1]]=1
    reconstructed+=reconstructed.T.clone()
    assert torch.equal(a,reconstructed)


def test_local_counterfactual_matches_complete_official_forward():
    torch.manual_seed(2026)
    a=fixture_graph()
    x=torch.randn(5,2)
    model=make_model(2,hidden=8).eval()
    incidence,_=pair_incidence(a)
    cache=build_pair_cache(model,x,a)
    # Enumerate all assignments: ensures own-treatment intervention affects messages.
    for values in itertools.product((0.,1.),repeat=5):
        treatment=torch.tensor(values)
        with torch.no_grad():
            official=model(x,treatment,incidence)
        for i in range(5):
            neighbors=torch.where(a[i]>0)[0]
            mask=treatment[neighbors].numpy()[None,:]
            t=int(treatment[i])
            actual=predict_root_samples(model,cache,i,t,mask)
            expected=official['y1_pred' if t else 'y0_pred'][i]
            torch.testing.assert_close(torch.tensor(actual[0]),expected,atol=2e-6,rtol=2e-5)


def test_conditional_sampler_respects_design_and_is_reproducible():
    from scipy.stats import binom
    for arm in (0,1):
        draws=conditional_assignments(8,arm,30000,seed=2026)
        counts=draws.sum(1)
        assert np.all(counts<=4) if arm==0 else np.all(counts>4)
        k=np.arange(9)
        probs=binom.pmf(k,8,.5)*(k<=4 if arm==0 else k>4)
        probs/=probs.sum()
        np.testing.assert_allclose(np.bincount(counts.astype(int),minlength=9)/len(counts),probs,atol=.009)
        assert np.array_equal(draws,conditional_assignments(8,arm,30000,seed=2026))
        assert draws.shape==(30000,8)


def test_small_graph_majority_integration_matches_exact_enumeration():
    torch.manual_seed(14)
    a=fixture_graph();x=torch.randn(5,2)
    model=make_model(2,hidden=8).eval()
    cache=build_pair_cache(model,x,a)
    estimate,mcse=integrate_majority(model,cache,samples=50000,seed=2026)
    exact=np.zeros((5,3))
    for i in range(5):
        d=int(a[i].sum())
        masks=np.array(list(itertools.product((0.,1.),repeat=d)),dtype=np.float32)
        mu={}
        for s in (0,1):
            chosen=masks[(masks.sum(1)>d//2)==bool(s)]
            for t in (0,1):mu[t,s]=predict_root_samples(model,cache,i,t,chosen).mean()
        exact[i]=[mu[1,0]-mu[0,0],mu[1,1]-mu[1,0],mu[1,1]-mu[0,0]]
    assert np.all(np.abs(estimate-exact)<6*mcse+1e-5)
    np.testing.assert_allclose(estimate[:,2],estimate[:,0]+estimate[:,1],atol=1e-6)
    assert np.isfinite(mcse).all()


@pytest.mark.parametrize("probability", [.2, .8])
def test_sampler_uses_supplied_randomization_probability(probability):
    from scipy.stats import binom
    for arm in (0,1):
        draws=conditional_assignments(5,arm,30000,seed=123,treatment_prob=probability)
        counts=draws.sum(1).astype(int)
        k=np.arange(6)
        weights=binom.pmf(k,5,probability)*(k<=2 if arm==0 else k>2)
        weights/=weights.sum()
        np.testing.assert_allclose(np.bincount(counts,minlength=6)/len(counts), weights,atol=.01)


def test_fit_predict_is_deterministic_and_preserves_effect_identity():
    from baselines.hypersci import fit_predict_hypersci_ite
    a=fixture_graph().numpy()
    kwargs=dict(adjacency=a,x=np.arange(5,dtype=float),treatment=np.array([0,1,0,1,0]),
                outcomes=np.array([.3,2.,.1,1.,-.5]), treatment_prob=.3, epochs=2,
                hidden_dim=8,seed=21,arm_samples=128)
    first,se=fit_predict_hypersci_ite(**kwargs)
    second,se2=fit_predict_hypersci_ite(**kwargs)
    assert first.shape==se.shape==(5,3)
    assert np.isfinite(first).all() and np.isfinite(se).all()
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(se,se2)
    np.testing.assert_allclose(first[:,2],first[:,0]+first[:,1],atol=1e-8)


def test_single_member_treatment_group_has_finite_wasserstein_loss():
    from baselines.hypersci import fit_hyper
    a=fixture_graph().numpy()
    _,_,_,_,history=fit_hyper(a,np.arange(5,dtype=float),np.array([1,0,0,0,0]),
                            np.arange(5,dtype=float),epochs=2,hidden_dim=8)
    assert all(np.isfinite(item['loss']) for item in history)


def test_missing_treatment_group_is_rejected():
    from baselines.hypersci import fit_hyper
    with pytest.raises(ValueError,match='both treatment groups'):
        fit_hyper(fixture_graph(),np.arange(5),np.zeros(5),np.arange(5),epochs=1)
