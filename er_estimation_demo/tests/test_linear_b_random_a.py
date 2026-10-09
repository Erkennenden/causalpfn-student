"""Catch per-node draws, fixed coefficients, changed noise streams and stale checkpoints."""
import numpy as np
import pytest
import torch
from functools import partial
from pfn_interference.dgp import generate_task
from pfn_interference.simulation.dgp import generate_task as generic_task
from pfn_interference.training import load_model


def ring(n=12):
    graph=torch.zeros(n,n)
    for i in range(n):
        graph[i,(i+1)%n]=graph[(i+1)%n,i]=1
    return graph


def old_formula(z):
    x,nx,t,e=np.moveaxis(np.asarray(z),-1,0)
    return 1-x-3*nx+(6+.2*np.exp(x))*t-.9*e


def test_a_varies_by_dataset_and_is_shared_by_factual_and_four_mu():
    values=[]
    for task_id in range(6):
        task=generate_task(ring(),seed=21,task_id=task_id,stream='test')
        a=task.metadata.get('a',6.)
        values.append(a)
        b=task.batch
        assert 4 <= a < 8
        x=b['x'];base=1-x-3*b['neighbor_x'];tau=a+.2*x.exp()
        # Degree two: conditional exposure means are 1/3 and 1.
        expected=torch.stack([base-.3,base-.9,base+tau-.3,base+tau-.9],-1)
        torch.testing.assert_close(b['cepo_target'],expected,atol=3e-6,rtol=1e-6)
        torch.testing.assert_close(b['y_obs'],base+tau*b['observed_treatment']-.9*b['observed_exposure']+b['noise'],atol=3e-6,rtol=1e-6)
        inferred=b['oracle_ite'][...,0]-.2*x.exp()
        torch.testing.assert_close(inferred,torch.full_like(x,a))
        assert b['tokens'].shape[-1]==5  # The true coefficient is not a feature.
    assert len(set(values))==6,'Each task must draw its own a, not reuse the fixed 6'


def test_a_replays_across_cache_order_and_changes_across_splits():
    first=generate_task(ring(),seed=23,task_id=4,stream='train')
    repeated=generate_task(ring(),seed=23,task_id=4,stream='train')
    assert 'a' in first.metadata
    assert first.metadata['a']==repeated.metadata['a']
    for key in first.batch:
        torch.testing.assert_close(first.batch[key],repeated.batch[key],rtol=0,atol=0)
    variants=[generate_task(ring(),seed=s,task_id=i,stream=split).metadata['a']
              for s,i,split in [(23,4,'train'),(23,4,'validation'),(23,4,'test'),(24,4,'train'),(23,5,'train')]]
    assert len(set(variants))==5


def test_only_treated_outcomes_shift_and_all_other_random_streams_stay_identical():
    new=generate_task(ring(),seed=123,task_id=9,stream='train')
    old=generic_task(ring(),seed=123,task_id=9,stream='train',outcome=old_formula)
    a=new.metadata.get('a',6.)
    assert a != 6
    for key in ('x','neighbor_x','observed_treatment','observed_exposure','noise','adjacency'):
        torch.testing.assert_close(new.batch[key],old.batch[key],rtol=0,atol=0)
    shift=new.batch['cepo_target']-old.batch['cepo_target']
    torch.testing.assert_close(shift[...,:2],torch.zeros_like(shift[...,:2]),rtol=0,atol=0)
    torch.testing.assert_close(shift[...,2:],torch.full_like(shift[...,2:],a-6),atol=3e-6,rtol=1e-6)
    quiet=generate_task(ring(),seed=123,task_id=9,stream='train',noise_sd=0)
    assert quiet.metadata['a']==a
    torch.testing.assert_close(quiet.batch['cepo_target'],new.batch['cepo_target'],rtol=0,atol=0)


def test_fixed_b_checkpoint_is_rejected_before_model_loading(tmp_path):
    path=tmp_path/'old.pt'
    torch.save(dict(version='causalcfm_fixed_graph_linear_b_four_gmm_v2',
                    prediction_protocol='majority_cepo_four_gmm_v2'),path)
    with pytest.raises(ValueError,match='[Cc]heckpoint'):
        load_model(path)
