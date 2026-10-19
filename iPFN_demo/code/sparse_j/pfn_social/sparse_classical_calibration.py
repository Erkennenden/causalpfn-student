"""Load and query a frozen calibrated Gaussian kernel mixture."""
from pathlib import Path


import copy, math


import numpy as np


import torch


from torch import nn


from pfn_social.phase6_sparse_classical import validate


from pfn_social.sparse_posterior_calibration import component_cache, cached_coefficients


IDENTITIES={
 'population':('Population auxiliary-fitted kernel-linear Bayes mixture','Sparse-calibrated population kernel-linear Bayes mixture'),
 'phase5':('Phase5 auxiliary-fitted kernel-linear Bayes mixture','Sparse-calibrated Phase5 kernel-linear Bayes mixture')}


def design_function(kind):
    if kind=='phase4':
        from scripts.phase4_strong_controls import design
    elif kind=='phase5':
        from scripts.phase5_strong_controls import design
    else:raise ValueError('Unknown frozen observed-input feature dictionary')
    return design


def tree_numpy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
    if isinstance(value,dict):return {k:tree_numpy(v) for k,v in value.items()}
    if isinstance(value,list):return [tree_numpy(v) for v in value]
    return value


class FixedGaussian(nn.Module):
    def __init__(self,mean,lower):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float64).clone())
        self.register_buffer('factor',torch.as_tensor(lower,dtype=torch.float64).clone())
    def lower(self):return self.factor


class SparseClassicalCalibration(nn.Module):
    def __init__(self,identity,specs,design_kinds,means,lowers,representation_weights,noise_weights,noise_variances):
        super().__init__()
        if identity not in IDENTITIES or len(specs)!=2:raise ValueError('Exactly the registered two-component identity required')
        self.identity=identity;self.specs=copy.deepcopy(specs);self.design_kinds=list(design_kinds)
        self.priors=nn.ModuleList(FixedGaussian(mean,lower) for mean,lower in zip(means,lowers))
        self.register_buffer('noise_variance_grid',torch.tensor([.25,1.,4.,16.],dtype=torch.float64))
        ratio=np.asarray(noise_variances,dtype=float)/self.noise_variance_grid.numpy()
        if not np.allclose(ratio,ratio[0],rtol=1e-12,atol=1e-12):raise ValueError('Parent does not use the declared common scaled noise grid')
        initial_log_scale=float(np.log(ratio[0]))
        if not -2<=initial_log_scale<=2:raise ValueError('Parent noise scale outside shared calibration bounds')
        self.log_covariance_scales=nn.Parameter(torch.zeros(2,dtype=torch.float64))
        self.representation_logits=nn.Parameter(torch.tensor(np.log(representation_weights),dtype=torch.float64))
        self.noise_logits=nn.Parameter(torch.tensor(np.log(noise_weights),dtype=torch.float64))
        self.log_grid_scale=nn.Parameter(torch.tensor(initial_log_scale,dtype=torch.float64))
        self.log1p_gamma=nn.Parameter(torch.tensor(0.,dtype=torch.float64))

    def gamma(self):return self.log1p_gamma.clamp(0.,math.log(65.)).expm1()
    def covariance_scales(self):return self.log_covariance_scales.clamp(-math.log(4.),math.log(4.)).exp()


    @torch.no_grad()
    def make_cache(self,*,x,adjacency,treatment,observed_indices,outcomes_observed):
        x,a,t,idx,y=validate(x=x,adjacency=adjacency,treatment=treatment,observed_indices=observed_indices,outcomes_observed=outcomes_observed)
        result=[]
        for spec,kind,prior in zip(self.specs,self.design_kinds,self.priors):
            factual,queries=design_function(kind)(x,a,t,spec)
            result.append(component_cache(torch.from_numpy(factual[idx])[None],torch.from_numpy(queries)[None],torch.from_numpy(y)[None],prior))
        return result

    def forward_cached(self,cache,*,return_diagnostics=False):
        variance=self.noise_variance_grid*self.log_grid_scale.clamp(-2.,2.).exp()
        coefficients=[];evidence=[]
        for k,component in enumerate(cache):
            b,logev=cached_coefficients(component,self.covariance_scales()[k],variance,self.gamma())
            coefficients.append(b);evidence.append(logev)
        evidence=torch.stack(evidence,1)
        logprior=torch.log_softmax(self.representation_logits,-1)[None,:,None]+torch.log_softmax(self.noise_logits,-1)[None,None,:]
        weights=torch.softmax((evidence+logprior).flatten(1),-1).reshape(evidence.shape)
        mean=0.
        for k,(component,beta) in enumerate(zip(cache,coefficients)):
            beta=(beta*weights[:,k,:,None]).sum(1)
            mean=mean+component['query_mean']*weights[:,k].sum(-1)[:,None,None]+torch.einsum('bnaq,bq->bna',component['query_lower'],beta)
        if return_diagnostics:return mean,dict(joint_weights=weights,gamma=self.gamma(),covariance_scales=self.covariance_scales(),noise_variances=variance)
        return mean

    @torch.no_grad()
    def predict_arm_means(self,*,x,adjacency,treatment,observed_indices,outcomes_observed):
        self.eval();return self.forward_cached(self.make_cache(x=x,adjacency=adjacency,treatment=treatment,
            observed_indices=observed_indices,outcomes_observed=outcomes_observed))[0].numpy()


def frozen_state_sha(model):
    import hashlib
    h=hashlib.sha256()
    def update(value):
        if isinstance(value,np.ndarray):h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
        elif isinstance(value,torch.Tensor):update(value.detach().cpu().numpy())
        elif isinstance(value,dict):
            for key in sorted(value):h.update(str(key).encode());update(value[key])
        elif isinstance(value,(list,tuple)):
            for v in value:update(v)
        else:h.update(repr(value).encode())
    update(model.specs);update(model.design_kinds);update(model.priors.state_dict());update(model.noise_variance_grid)
    return h.hexdigest()


def load_model(path):
    value=torch.load(Path(path),weights_only=True,map_location='cpu')
    if value['format']!='sparse_classical_scalar_calibration_v1':raise ValueError('Unknown portable format')
    state=value['model_state'];specs=tree_numpy(value['specs'])
    model=SparseClassicalCalibration(value['identity'],specs,value['design_kinds'],
        [state[f'priors.{i}.mean'] for i in range(2)],[state[f'priors.{i}.factor'] for i in range(2)],
        [.5,.5],[.25]*4,[.25,1.,4.,16.])
    model.load_state_dict(state)
    if frozen_state_sha(model)!=value['frozen_state_sha256']:raise ValueError('Frozen dictionary/prior integrity mismatch')
    return model.eval(),value
