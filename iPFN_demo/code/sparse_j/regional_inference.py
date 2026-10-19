"""Fixed small-network calibration and weighted four-arm predictions.

For N < 50 and M == N, combine the three calibrated components.
Other inputs use the frozen J predictor."""
import numpy as np
import torch
from pfn_social.sparse_frozen_gaussian import graph_moments
from pfn_social.phase5_meta_bayes import design
from pfn_social.sparse_frozen_gaussian import validate_factual_inputs
from pfn_social.sparse_posterior_calibration import component_cache,cached_coefficients
from pfn_social.sparse_classical_calibration import FixedGaussian,design_function


def regional_gate(inputs):
    return len(inputs['x']) < 50 and len(inputs['observed_indices']) == len(inputs['x'])


def original_arm_means(name,parent,inputs):
    if name in ('j','kernel-new'):
        return parent.predict_arm_means(**inputs)
    return parent.predict(**inputs)['arm_means']


@torch.no_grad()
def make_cache(name,parent,inputs):
    x,a,t,indices,y=validate_factual_inputs(**inputs)
    if name=='j':
        model=parent.model
        xt=torch.tensor(x,dtype=torch.float32)[None]
        at=torch.tensor(a,dtype=torch.float32)[None]
        tt=torch.tensor(t,dtype=torch.float32)[None,:,None]
        et=torch.tensor(a@t/a.sum(1),dtype=torch.float32)[None,:,None]
        yt=torch.tensor(y,dtype=torch.float32)[None]
        degree=at.sum(-1,keepdim=True);h=model.encoder(xt,at);moments=graph_moments(at);cache=[]
        for representation,prior in zip((h,xt),model.priors):
            hb=at@representation/degree
            factual=design(representation,hb,tt,et,at@(tt*representation)/degree)
            queries=[]
            for own,arm in ((0,0),(0,1),(1,0),(1,1)):
                em=moments[...,arm:arm+1]
                queries.append(design(representation,hb,torch.full_like(tt,own),em,em*hb))
            cache.append(component_cache(factual[:,indices],torch.stack(queries,-2),yt,prior))
        state=dict(variance=model.noise_variance_grid.double()*model.log_grid_scale.clamp(-2,2).exp(),
            rep=torch.softmax(model.representation_logits.double(),-1),noise=torch.softmax(model.noise_logits.double(),-1),
            gamma=model.gamma().double(),covariance=torch.ones(2,dtype=torch.float64))
    elif name=='kernel-old':
        cache=[]
        for spec,prior,value,fn in parent.components:
            factual,queries=fn(x,a,t,spec)
            proper=FixedGaussian(prior['mean'],np.linalg.cholesky(prior['covariance']*value))
            cache.append(component_cache(torch.tensor(factual[indices])[None],torch.tensor(queries)[None],torch.tensor(y)[None],proper))
        state=dict(variance=torch.tensor(parent.selection['noise_variances'],dtype=torch.float64),
            rep=torch.tensor(parent.selection['representation_weights'],dtype=torch.float64),
            noise=torch.tensor(parent.selection['noise_weights'],dtype=torch.float64),gamma=torch.tensor(0.,dtype=torch.float64),
            covariance=torch.ones(2,dtype=torch.float64))
    elif name=='kernel-new':
        cache=[]
        for spec,kind,prior in zip(parent.specs,parent.design_kinds,parent.priors):
            factual,queries=design_function(kind)(x,a,t,spec)
            cache.append(component_cache(torch.tensor(factual[indices])[None],torch.tensor(queries)[None],torch.tensor(y)[None],prior))
        state=dict(variance=parent.noise_variance_grid*parent.log_grid_scale.clamp(-2,2).exp(),
            rep=torch.softmax(parent.representation_logits,-1),noise=torch.softmax(parent.noise_logits,-1),
            gamma=parent.gamma(),covariance=parent.covariance_scales())
    else:
        raise ValueError('Unknown frozen component identity')
    return cache,state


@torch.no_grad()
def calibrated_posterior(cache,state,setting):
    tau,nu,rho=[float(setting[k]) for k in ('covariance_scale','noise_variance_scale','nonlinear_representation_odds')]
    if not all(np.isfinite(v) and v>0 for v in (tau,nu,rho)):
        raise ValueError('Positive finite covariance/noise/representation scales required')
    variances=state['variance']*nu;rep=state['rep'].clone();rep[0]*=rho;rep/=rep.sum()
    predictions=[];evidences=[]
    for k,component in enumerate(cache):
        beta,evidence=cached_coefficients(component,state['covariance'][k]*tau,variances,state['gamma'])
        predictions.append(component['query_mean'][:,None]+torch.einsum('bnaq,bgq->bgna',component['query_lower'],beta))
        evidences.append(evidence)
    evidence=torch.stack(evidences,1);predictions=torch.stack(predictions,1)
    weights=torch.softmax((evidence+rep.log()[None,:,None]+state['noise'].log()[None,None]).flatten(1),-1).reshape(evidence.shape)
    mean=(predictions*weights[...,None,None]).sum((1,2))[0].numpy()
    if not np.isfinite(mean).all():raise FloatingPointError('Nonfinite regional arm means')
    return mean


class RegionalPosterior:
    def __init__(self,name,parent,setting):
        self.name,self.parent,self.setting=name,parent,dict(setting)
        self.identity=dict(component=name,regional_gate='N<50 and M=N',setting=dict(setting),no_context_parameter_fitting=True)

    def predict_arm_means(self,**inputs):
        if not regional_gate(inputs):return original_arm_means(self.name,self.parent,inputs)
        return calibrated_posterior(*make_cache(self.name,self.parent,inputs),self.setting)


class RegionalArmEnsemble:
    def __init__(self,parents,settings,weight):
        self.parents=dict(parents);self.components={name:RegionalPosterior(name,parents[name],settings[name]) for name in ('j','kernel-old','kernel-new')}
        self.weight=float(weight)
        if not 0<=self.weight<=1:raise ValueError('Convex weight required')
        self.identity=dict(regional_gate='N<50 and M=N',weight=self.weight,
            formula='w*J_regional+(1-w)*(kernel_old_regional+kernel_new_regional)/2',
            other_contexts='Exact original frozen J public predictor')

    def predict_arm_means(self,**inputs):
        if not regional_gate(inputs):return original_arm_means('j',self.parents['j'],inputs)
        means={name:component.predict_arm_means(**inputs) for name,component in self.components.items()}
        return self.weight*means['j']+(1-self.weight)*.5*(means['kernel-old']+means['kernel-new'])
