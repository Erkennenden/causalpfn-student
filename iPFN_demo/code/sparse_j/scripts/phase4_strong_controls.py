"""Feature and query designs for the first frozen kernel model."""
import numpy as np


from pfn_social.contracts import arm_mean_exposure


from scripts.phase3_kernel_controls import rbf


def feature(x,spec):
    z=(x-spec['xmean'])/spec['xscale']
    if spec['kind']=='linear':return z
    projected=z@spec['projection']
    return np.column_stack([z,rbf(projected,spec['centers'],spec['length'])@spec['kernel_basis']])


def design(x,a,t,spec):
    h=feature(x,spec);degree=a.sum(1);hb=a@h/degree[:,None];e=a@t/degree;j=a@(t[:,None]*h)/degree[:,None]
    base=np.column_stack([np.ones(len(x)),h,hb]);hetero=np.column_stack([np.ones(len(x)),h])
    def rows(own,exposure,identity):
        return np.column_stack([base,own[:,None]*hetero,exposure[:,None]*hetero,
            (own*exposure)[:,None]*hetero,identity,own[:,None]*identity])
    factual=rows(t,e,j);query=[]
    for own,arm in ((0,0),(0,1),(1,0),(1,1)):
        em=np.array([arm_mean_exposure(int(d),.5,arm) for d in degree])
        query.append(rows(np.full(len(x),float(own)),em,em[:,None]*hb))
    scale=spec.get('feature_scale',np.ones(factual.shape[1]))
    return factual/scale,np.stack(query,1)/scale

