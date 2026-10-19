"""Synthetic network mechanisms and repeatable treatment/outcome draws."""
from dataclasses import asdict,dataclass,replace
import hashlib,json
import numpy as np
import torch
from pfn_social.contracts import arm_mean_exposure
from pfn_social.phase3_hierarchical import (
    SETTINGS as BASE_SETTINGS,connected_graph,hidden_features,coefficient_prior,
    _components,_hash_arrays,structural_outcome,
)
from functools import lru_cache

P=20
REPRESENTATION_SEED=2026101604

@lru_cache(maxsize=1)
def common_projection():
    return np.linalg.qr(np.random.default_rng(REPRESENTATION_SEED).normal(size=(P,2)))[0]

ALLOWED_N=(50,100,200)
SEED_NAMESPACE=2026101605
CHANNELS={'graph':1,'x':2,'map':3,'coefficient':4,'treatment':5,'noise':6,'population':7}


@dataclass(frozen=True)
class LargeNetworkConfig:
    base_setting_id:int
    n:int
    setting_id:int
    split:str
    split_seed:int
    role:str='smoke'

    def __post_init__(self):
        if self.base_setting_id not in range(len(BASE_SETTINGS)):raise ValueError('UnknownStage4basesetting')
        if self.n not in ALLOWED_N:raise ValueError('Onlyexplicitn50/100/200configurationsarepermitted')
        if self.setting_id<0 or self.split_seed<0:raise ValueError('NonnegativeconfigurationID/seedrequired')
        if self.split not in ('smoke','train','dev'):raise ValueError('Finalsplitrequiresanewregisteredprotocol')
        if self.role not in ('main','learning_curve_diagnostic','smoke'):raise ValueError('Unknownconfigurationrole')

    @property
    def setting(self):
        return replace(BASE_SETTINGS[self.base_setting_id],n=self.n,name=f'phase5_base{self.base_setting_id}_n{self.n}')

    def canonical_json(self):return json.dumps(asdict(self),sort_keys=True,separators=(',',':'))


def seed_entropy(config,mechanism_id,channel,repeat_id=0):
    if channel not in CHANNELS or mechanism_id<0 or repeat_id<0:raise ValueError('InvalidRNGidentifier')
    split_code=int.from_bytes(hashlib.sha256(config.split.encode()).digest()[:4],'little')
    return (config.split_seed,SEED_NAMESPACE,split_code,config.setting_id,
            config.base_setting_id,config.n,int(mechanism_id),CHANNELS[channel],int(repeat_id))


def rng_for(config,mechanism_id,channel,repeat_id=0):
    return np.random.default_rng(np.random.SeedSequence(seed_entropy(config,mechanism_id,channel,repeat_id)))


@dataclass(frozen=True)
class LargeNetworkMechanism:
    config:LargeNetworkConfig
    mechanism_id:int
    adjacency:np.ndarray
    x:np.ndarray
    projection:np.ndarray
    h:np.ndarray
    coefficients:np.ndarray
    components:np.ndarray
    truth:np.ndarray
    population_mean:np.ndarray
    mechanism_sha256:str
    truth_sha256:str

    @property
    def setting(self):return self.config.setting
    @property
    def setting_id(self):return self.config.setting_id
    @property
    def split(self):return self.config.split


def generate_mechanism(config:LargeNetworkConfig,mechanism_id:int):
    s=config.setting;a=connected_graph(s,rng_for(config,mechanism_id,'graph'))
    population=rng_for(config,mechanism_id,'population').normal(0,.8,P)
    x=(population+.6*rng_for(config,mechanism_id,'x').normal(size=(s.n,P))).astype(np.float32)
    projection=common_projection().copy()
    if s.representation=='unshared_nonlinear':
        projection=np.linalg.qr(rng_for(config,mechanism_id,'map').normal(size=(P,2)))[0]
    h=hidden_features(x,projection,s.representation);mean,cov=coefficient_prior()
    beta=rng_for(config,mechanism_id,'coefficient').multivariate_normal(mean,cov)
    components=_components(a,h,beta);b,tau,gamma,eta=components.T
    degree=a.sum(1);hbar=a@h/degree[:,None];truth=np.empty((s.n,4),dtype=np.float64)
    for own,arm in ((0,0),(0,1),(1,0),(1,1)):
        em=np.array([arm_mean_exposure(int(d),.5,arm) for d in degree])
        truth[:,2*own+arm]=b+own*tau+em*(gamma+own*eta+hbar@(beta[19:22]+own*beta[22:25]))
    identity=np.frombuffer(config.canonical_json().encode(),dtype=np.uint8)
    return LargeNetworkMechanism(config,mechanism_id,a,x,projection,h,beta,components,truth,population,
        _hash_arrays(a,x,projection,beta,population,identity),_hash_arrays(truth))


def make_repeat(mechanism,repeat_id):
    m=mechanism;a=m.adjacency;n=m.setting.n
    assert a.shape==(n,n) and len(m.x)==n
    t=rng_for(m.config,m.mechanism_id,'treatment',repeat_id).binomial(1,.5,n).astype(np.float32)
    noise=rng_for(m.config,m.mechanism_id,'noise',repeat_id).normal(0,m.setting.noise_sd,n)
    y=(structural_outcome(m,t)+noise).astype(np.float32)
    f32=lambda value:torch.tensor(np.array(value,copy=True),dtype=torch.float32)
    return dict(adjacency=f32(a),x=f32(m.x),observed_treatment=f32(t),y_obs=f32(y))
