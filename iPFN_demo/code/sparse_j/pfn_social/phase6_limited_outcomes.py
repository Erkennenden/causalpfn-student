"""Select outcome subsets from complete synthetic network draws."""
from dataclasses import dataclass


import numpy as np


import torch


from pfn_social.phase5_large_network import LargeNetworkConfig, generate_mechanism as parent_mechanism, make_repeat as parent_repeat


N=200


OBSERVATION_COUNTS=(20,50,100)


BASE_IDS=(0,1,3,4,5,6,7)


SPLIT_SEED=331029


MASK_NAMESPACE=2026101606


MASK_CHANNEL=8


FACTUAL_KEYS=frozenset(('x','adjacency','treatment','observed_indices','outcomes_observed'))


def mask_rng(family_id,mechanism_id):
    return np.random.default_rng(np.random.SeedSequence([SPLIT_SEED,MASK_NAMESPACE,int(family_id),int(mechanism_id),MASK_CHANNEL]))


def configuration(family_id):
    if family_id not in range(7):raise ValueError('Seven unique full-network families only')
    return LargeNetworkConfig(BASE_IDS[family_id],N,family_id,'dev',SPLIT_SEED,'main')


@dataclass(frozen=True)
class LimitedOutcomeMechanism:
    parent:object
    observation_permutation:np.ndarray


def generate_mechanism(family_id,mechanism_id):
    # Draw the independent observation design before graph/X/f, and before T/Y.
    permutation=mask_rng(family_id,mechanism_id).permutation(N)
    mechanism=parent_mechanism(configuration(family_id),mechanism_id)
    return LimitedOutcomeMechanism(mechanism,permutation)


def make_observed_draw(mechanism,repeat_id,shared_arrays=None):
    m=mechanism.parent
    raw=parent_repeat(m,repeat_id) # Producer memory only; full Y never serialized.
    shared=shared_arrays or {'x':raw['x'].clone(),'adjacency':raw['adjacency'].clone()}
    treatment=raw['observed_treatment'].clone()
    rows=[]
    for mi,count in enumerate(OBSERVATION_COUNTS):
        idx=torch.tensor(mechanism.observation_permutation[:count].copy(),dtype=torch.int64)
        # Clone is essential: no tensor storage may expose an unobserved Y.
        observed=raw['y_obs'][idx].clone()
        batch=dict(x=shared['x'],adjacency=shared['adjacency'],treatment=treatment,
            observed_indices=idx,outcomes_observed=observed)
        validate_factual(batch)
        family=m.config.setting_id;mid=m.mechanism_id
        rows.append(dict(setting_id=3*family+mi,family_id=family,m_observed=count,n_units=N,
            mechanism_id=mid,repeat_id=repeat_id,draw_id=f'f{family}_m{mid}_r{repeat_id}',
            context_id=f'f{family}_m{mid}_r{repeat_id}_observed{count}',
            batch=batch))
    return rows,shared


def validate_factual(batch):
    if set(batch)!=FACTUAL_KEYS:raise ValueError('Exactly five factual arrays are required')
    x,a,t,idx,y=(batch[k] for k in ('x','adjacency','treatment','observed_indices','outcomes_observed'))
    count=len(idx)
    if count not in OBSERVATION_COUNTS:raise ValueError('Unregistered observation count')
    if x.shape!=(N,20) or a.shape!=(N,N) or t.shape!=(N,) or idx.shape!=(count,) or y.shape!=(count,):raise ValueError('Factual shape mismatch')
    if idx.dtype!=torch.int64 or len(torch.unique(idx))!=count or int(idx.min())<0 or int(idx.max())>=N:raise ValueError('Invalid observed indices')
    if any(not bool(torch.isfinite(v).all()) for v in (x,a,t,y)):raise ValueError('Nonfinite factual array')
    if not bool(torch.isin(t,torch.tensor([0.,1.])).all()):raise ValueError('Binary treatment required')
    if y.untyped_storage().nbytes()!=count*y.element_size():raise ValueError('Observed outcome storage exposes extra values')
    return True

