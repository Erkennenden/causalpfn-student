"""Graph-environment feature and query designs for the calibrated kernel."""
import numpy as np


from pfn_social.contracts import arm_mean_exposure


from scripts.phase3_kernel_controls import rbf


def environment_features(x,a,transform):
    z=((x-transform['x_mean'])/transform['x_scale'])@transform['projection'];degree=a.sum(1)
    if np.any(degree<=0):raise ValueError('Declared Phase5 graphs have positive degree')
    wz=a@z/degree[:,None];w2z=a@wz/degree[:,None];u,v=z.T
    hermite=np.column_stack([u,v,(u*u-1)/np.sqrt(2),u*v,(v*v-1)/np.sqrt(2)])
    return np.column_stack([hermite,wz,w2z,np.broadcast_to(z.mean(0),z.shape),np.log1p(degree)])


def base_features(x,a,spec):
    if spec['input_kind']=='environment':
        tr={k:spec[k] for k in ('x_mean','x_scale','projection')}
        return (environment_features(x,a,tr)-spec['environment_mean'])/spec['environment_scale']
    return (x-spec['xmean'])/spec['xscale']


def feature(x,a,spec):
    if spec['kind']=='polynomial':
        z=((x-spec['x_mean'])/spec['x_scale'])@spec['projection'];u,v=z.T
        return np.column_stack([u,v,(u*u-1)/np.sqrt(2),u*v,(v*v-1)/np.sqrt(2)])
    base=base_features(x,a,spec)
    if spec['kind']=='linear':return base
    if spec.get('coordinate_mode')=='frozen_old':
        coordinate=((x-spec['x_mean'])/spec['x_scale'])@spec['projection']
    else:coordinate=base@spec['kernel_projection']
    k=rbf(coordinate,spec['centers'],spec['length'])@spec['kernel_basis']
    return np.column_stack([base,k]) if spec['include_raw'] else k


def design_from_h(h,a,t,scale=None):
    degree=a.sum(1);hb=a@h/degree[:,None];exposure=a@t/degree;identity=a@(t[:,None]*h)/degree[:,None]
    base=np.column_stack([np.ones(len(h)),h,hb]);hetero=np.column_stack([np.ones(len(h)),h])
    def rows(own,e,j):return np.column_stack([base,own[:,None]*hetero,e[:,None]*hetero,(own*e)[:,None]*hetero,j,own[:,None]*j])
    factual=rows(t,exposure,identity);query=[]
    for own,arm in ((0,0),(0,1),(1,0),(1,1)):
        em=np.array([arm_mean_exposure(int(d),.5,arm) for d in degree]);query.append(rows(np.full(len(h),float(own)),em,em[:,None]*hb))
    scale=np.ones(factual.shape[1]) if scale is None else scale
    return factual/scale,np.stack(query,1)/scale


def design(x,a,t,spec):return design_from_h(feature(x,a,spec),a,t,spec.get('feature_scale'))

