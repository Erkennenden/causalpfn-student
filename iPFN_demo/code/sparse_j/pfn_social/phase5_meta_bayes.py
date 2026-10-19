"""Frozen J model parameters and graph-environment encoder."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CoefficientPrior(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.dimension=dimension
        self.mean=nn.Parameter(torch.zeros(dimension))
        self.raw_cholesky=nn.Parameter(torch.eye(dimension)*float(np.log(np.expm1(.99))))

    def lower(self):
        return (torch.tril(self.raw_cholesky,-1)+torch.diag(F.softplus(torch.diagonal(self.raw_cholesky))+.01)).double()


def design(h,hbar,t,e,j):
    ones=torch.ones_like(t);hetero=torch.cat([ones,h],-1)
    return torch.cat([ones,h,hbar,t*hetero,e*hetero,t*e*hetero,j,t*j],-1)


def hermite_lift(z):
    """All two-dimensional linear and centered degree-two coordinates."""
    first,second=z.unbind(-1)
    root2=2.**.5
    return torch.stack((first,second,(first.square()-1)/root2,
                        first*second,(second.square()-1)/root2),-1)


class FixedQuadraticSubspaceEncoder(nn.Module):
    def __init__(self,mean,scale,projection,hidden=64):
        super().__init__()
        mean=np.asarray(mean,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64);projection=np.asarray(projection,dtype=np.float64)
        if mean.shape!=(20,) or scale.shape!=(20,) or projection.shape!=(20,2) or np.any(scale<=0):
            raise ValueError('RequirefixedTRAIN20Dmean/scaleand20x2projection')
        if not all(np.isfinite(v).all() for v in (mean,scale,projection)):
            raise ValueError('NonfiniteTRAINtransform')
        self.register_buffer('x_mean',torch.from_numpy(mean.copy()))
        self.register_buffer('x_scale',torch.from_numpy(scale.copy()))
        self.register_buffer('projection',torch.from_numpy(projection.copy()))
        self.network=nn.Sequential(nn.Linear(5,hidden),nn.Tanh(),nn.Linear(hidden,3),nn.Tanh())
        nn.init.normal_(self.network[2].weight,std=.1);nn.init.zeros_(self.network[2].bias)

    def project(self,x):return ((x.double()-self.x_mean)/self.x_scale)@self.projection
    def features(self,x):return hermite_lift(self.project(x))
    def forward(self,x):return self.network(self.features(x).to(x.dtype))


@dataclass(frozen=True)
class Phase5Config:
    architecture:str='J'
    raw_p:int=20
    hidden:int=64


def graph_environment_features(z,adjacency):
    """Exact shared 12-feature control interface, using pre-treatment X/A only.

    z is the fixed TRAIN-supervised two-dimensional raw-X transform. No
    context centering: [Hermite5(z),Wz,W²z,node-mean(z),log1p(degree)].
    """
    if z.shape[-1]!=2 or adjacency.shape[-2:]!=(z.shape[-2],z.shape[-2]):
        raise ValueError('Requirez[...,N,2]andmatchingadjacency')
    degree=adjacency.sum(-1,keepdim=True)
    if bool((degree.detach()<=0).any()):raise ValueError('Positivegraphdegreesrequired')
    w_z=adjacency.to(z.dtype)@z/degree.to(z.dtype)
    w2_z=adjacency.to(z.dtype)@w_z/degree.to(z.dtype)
    population=z.mean(-2,keepdim=True).expand_as(z)
    return torch.cat([hermite_lift(z),w_z,w2_z,population,degree.to(z.dtype).log1p()],-1)


class GraphEnvironmentEncoder(FixedQuadraticSubspaceEncoder):
    def __init__(self,mean,scale,projection,hidden=64):
        super().__init__(mean,scale,projection,hidden)
        self.network=nn.Sequential(nn.Linear(12,hidden),nn.Tanh(),nn.Linear(hidden,3),nn.Tanh())
        nn.init.normal_(self.network[2].weight,std=.1);nn.init.zeros_(self.network[2].bias)
    def features(self,x,adjacency):return graph_environment_features(self.project(x),adjacency)
    def forward(self,x,adjacency):return self.network(self.features(x,adjacency).to(x.dtype))


class Phase5TargetBayes(nn.Module):
    """Frozen parameter container for the J model; inference uses SparseFrozenGaussian."""
    def __init__(self,config=Phase5Config(),*,x_mean,x_scale,projection):
        super().__init__();self.config=config
        if config.raw_p!=20 or config.architecture!='J':raise ValueError('Require the J architecture with 20 raw features')
        self.encoder=GraphEnvironmentEncoder(x_mean,x_scale,projection,config.hidden)
        self.priors=nn.ModuleList([CoefficientPrior(25),CoefficientPrior(144)])
        self.representation_logits=nn.Parameter(torch.zeros(2));self.noise_logits=nn.Parameter(torch.zeros(4))
        self.log_grid_scale=nn.Parameter(torch.tensor(0.))
        self.register_buffer('noise_variance_grid',torch.tensor([.25,1.,4.,16.]))
        self.log1p_gamma=nn.Parameter(torch.tensor(math.log(2.),dtype=torch.float64))

    def gamma(self):return self.log1p_gamma.clamp(0.,math.log(65.)).expm1()


def load_model(path):
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    config=Phase5Config(**state['config']);weights=state['model_state']
    if config.architecture!='J':raise ValueError('This package loads the J architecture')
    transform={k:weights['encoder.'+k].numpy() for k in ('x_mean','x_scale','projection')}
    model=Phase5TargetBayes(config,**transform);model.load_state_dict(weights)
    return model.eval(),state
