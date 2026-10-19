"""Design-based, linear, kernel and CausalPFN baseline adapters."""
import hashlib


import numpy as np


import torch


from scipy.special import softmax


from pfn_social.estimation import majority_arm_propensities


EFFECTS=('direct','spillover','total')


CONTRASTS=((1,0),(3,1),(3,0)) # state order00,10,01,11


def validate(*,x,adjacency,treatment,observed_indices,outcomes_observed):
    x,a,t,y=(np.asarray(v,dtype=float) for v in (x,adjacency,treatment,outcomes_observed))
    raw=np.asarray(observed_indices)
    if raw.ndim!=1 or raw.dtype.kind not in 'iu':raise ValueError('Integer observed indices required')
    ids=raw.astype(np.int64,copy=False);n=len(x)
    if x.shape!=(n,20) or a.shape!=(n,n) or t.shape!=(n,) or y.shape!=(len(ids),):raise ValueError('Require fullX20/A/T and ONLY M observedY')
    if len(ids)<2 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=n):raise ValueError('Distinct valid observed indices')
    if not all(np.isfinite(v).all() for v in (x,a,t,y)):raise ValueError('Finite observed inputs required')
    if not np.isin(t,[0,1]).all() or not np.isin(a,[0,1]).all() or not np.array_equal(a,a.T) or np.any(np.diag(a)) or np.any(a.sum(1)<=0):raise ValueError('BinaryT and positive-degree undirected graph required')
    return x,a,t,ids,y


def pack(mu=None,*,ite=None,ate=None,n,diagnostics=None):
    if mu is None:mu=np.full((n,4),np.nan)
    if ite is None:
        ite=np.column_stack([mu[:,2]-mu[:,0],mu[:,3]-mu[:,2],mu[:,3]-mu[:,0]])
    if ate is None:ate=ite.mean(0)
    return dict(arm_means=mu,ite_effects=ite,ate_effects=ate,
        arm_supported=np.isfinite(mu).all(0),ite_supported=np.isfinite(ite).all(0),
        ate_supported=np.isfinite(ate),diagnostics={} if diagnostics is None else diagnostics)


def cached_grid(f,q,y,mean,covariance,variances,cache,key,identity=False):
    """Algebraically identical Gaussian update, caching only X/A query terms."""
    if cache.get('key')!=key:
        cache.clear();cache.update(key=key,query_mean=q@mean,query_covariance=q if identity else q@covariance)
    cf=f.T if identity else covariance@f.T;kernel=f@cf
    eigen,vectors=np.linalg.eigh((kernel+kernel.T)/2);eigen=np.maximum(eigen,0)
    residual=y-f@mean;projection=vectors.T@residual
    denominator=eigen[:,None]+np.asarray(variances)[None]
    solutions=vectors@(projection[:,None]/denominator)
    cross=cache['query_covariance']@f.T
    means=cache['query_mean'][None]+np.einsum('nam,mg->gna',cross,solutions)
    evidence=-.5*(np.log(denominator).sum(0)+(projection[:,None]**2/denominator).sum(0)+len(y)*np.log(2*np.pi))
    return means,evidence


def query_key(x,a):return hashlib.sha256(x.tobytes()+a.tobytes()).digest()


class FrozenMixture:
    def __init__(self,components,selection):
        self.components,self.selection=components,selection;self.caches=[{} for _ in components]

    def predict(self,**kwargs):
        x,a,t,ids,y=validate(**kwargs);pred=[];evidence=[]
        rep=np.array(self.selection['representation_weights']);noise=np.array(self.selection['noise_weights'])
        for k,(spec,prior,value,fn) in enumerate(self.components):
            f,q=fn(x,a,t,spec);f=f[ids]
            means,logevs=cached_grid(f,q,y,prior['mean'],prior['covariance']*value,self.selection['noise_variances'],self.caches[k],query_key(x,a))
            for j,(mu,logev) in enumerate(zip(means,logevs)):
                pred.append(mu);evidence.append(logev+np.log(rep[k])+np.log(noise[j]))
        weights=softmax(evidence);mu=np.einsum('k,kna->na',weights,np.stack(pred))
        return pack(mu,n=len(x),diagnostics=dict(observed_rows=len(y),query_units=len(x),joint_weights=weights.tolist(),
            likelihood='Normalized M-dimensional Gaussian density; no division of log likelihood by M or N'))


def state_design(x,a,t,ids,auxiliary,interacted,rcond=1e-6):
    """Full-population centering, then restrict factual rows, never induced A."""
    n=len(x);degree=a.sum(1);count=a@t;state=(count>degree//2).astype(int)*2+t.astype(int)
    z=auxiliary-auxiliary.mean(0);scale=np.sqrt((z*z).mean(0));active=scale>0
    z=z[:,active]/scale[active]
    if z.shape[1]:
        u,s,_=np.linalg.svd(z,full_matrices=False);z=u[:,s>rcond*s[0]]*np.sqrt(n)
    one=np.eye(4)[state]
    slopes=(one[:,:,None]*z[:,None,:]).reshape(n,-1) if interacted else z
    design=np.column_stack([one,slopes])
    probabilities=majority_arm_propensities(degree,treatment_prob=.5)
    observed_probability=probabilities[np.arange(n),state]*(len(ids)/n)
    return design[ids],observed_probability[ids],state[ids],probabilities,z.shape[1]


class DesignBased:
    def __init__(self,method,transform=None):self.method,self.transform=method,transform

    def predict(self,**kwargs):
        x,a,t,ids,y=validate(**kwargs);n=len(x);method=self.method;ate=np.full(3,np.nan);diag={}
        if method in ('HT','Hajek'):
            degree=a.sum(1);states=(a@t>degree//2).astype(int)*2+t.astype(int)
            pi=majority_arm_propensities(degree,treatment_prob=.5);r=len(ids)/n
            for j,(left,right) in enumerate(CONTRASTS):
                wa=(states[ids]==left)/(pi[ids,left]*r);wb=(states[ids]==right)/(pi[ids,right]*r)
                if method=='HT':ate[j]=((wa-wb)*y).sum()/n
                elif wa.sum()>0 and wb.sum()>0:ate[j]=wa@y/wa.sum()-wb@y/wb.sum()
            diag=dict(observation_probability=r,observed_rows=len(ids),query_units=n,
                estimand='Finite-population fullN average joint-arm contrasts',weights='1/[P(T,S)*m/N]')
        elif method=='Non-interference OLS':
            f=np.column_stack([np.ones(n),x,t]);beta,_,_,_=np.linalg.lstsq(f[ids],y,rcond=1e-6)
            contrast=np.zeros(f.shape[1]);contrast[-1]=1
            weights=np.linalg.lstsq(f[ids].T,contrast,rcond=1e-6)[0]
            if np.linalg.norm(f[ids].T@weights-contrast)>1e-5:
                ite=np.full((n,3),np.nan);ite[:,1]=0.
                return pack(ite=ite,n=n,diagnostics=dict(status='own_treatment_contrast_not_estimable',
                    assumption='Explicit non-interference; spillover structurally0, direct/total unsupported',observed_rows=len(ids)))
            f0=f.copy();f0[:,-1]=0;f1=f.copy();f1[:,-1]=1
            mu0=f0@beta;mu1=f1@beta;mu=np.column_stack([mu0,mu0,mu1,mu1])
            return pack(mu,n=n,diagnostics=dict(assumption='Explicit non-interference model; spillover structurally0',observed_rows=len(ids)))
        else:
            aux=x
            if method.endswith('-z2'):
                tr=self.transform;aux=((x-tr['x_mean'])/tr['x_scale'])@tr['projection']
            interacted=method.startswith('Lin');f,prob,state,_,p=state_design(x,a,t,ids,aux,interacted)
            w=np.sqrt(1/prob);fw=f*w[:,None];beta,_,rank,s=np.linalg.lstsq(fw,y*w,rcond=1e-6)
            details=[]
            for j,(left,right) in enumerate(CONTRASTS):
                c=np.zeros(f.shape[1]);c[left]=1;c[right]=-1
                if not np.any(state==left) or not np.any(state==right):status='empty_target_arm'
                else:
                    representation=np.linalg.lstsq(fw.T,c,rcond=1e-6)[0]
                    status='ok' if np.linalg.norm(fw.T@representation-c)<=10e-6*np.linalg.norm(c) else 'contrast_not_estimable'
                if status=='ok':ate[j]=c@beta
                details.append(dict(effect=EFFECTS[j],status=status))
            diag=dict(observed_rows=len(ids),query_units=n,covariate_rank=p,design_rank=int(rank),design_columns=f.shape[1],
                condition_number=float(s[0]/s[-1]) if len(s) and s[-1]>0 else None,
                centering='FullN auxiliary mean, not observed-subset mean',weights='1/[P(T,S)*m/N]',contrasts=details)
        return pack(ite=np.full((n,3),np.nan),ate=ate,n=n,diagnostics=diag)


class CausalPFNReference:
    def __init__(self,model,adjusted):self.model,self.adjusted=model,adjusted

    @torch.inference_mode()
    def predict(self,**kwargs):
        x,a,t,ids,y=validate(**kwargs);n=len(x)
        if len(np.unique(t[ids]))!=2:return pack(ite=np.full((n,3),np.nan),ate=np.full(3,np.nan),n=n,diagnostics=dict(status='one_observed_treatment_group'))
        if self.adjusted:
            s=(a@t>a.sum(1)//2).astype(float);context=np.column_stack([x[ids],s[ids]]);query=np.column_stack([x,np.zeros(n)])
        else:context=x[ids];query=x
        config=self.model.model_config
        capacity=config['model']['max_num_features']-1
        if context.shape[1]>capacity:raise ValueError('Frozen CausalPFN feature cap exceeded')
        tx=torch.tensor(np.concatenate([query,query]),dtype=torch.float32)[None]
        tt=torch.tensor(np.r_[np.zeros(n),np.ones(n)],dtype=torch.float32)[None]
        result=self.model.predict_cepo(X_context=torch.tensor(context,dtype=torch.float32)[None],
            t_context=torch.tensor(t[ids],dtype=torch.float32)[None],y_context=torch.tensor(y,dtype=torch.float32)[None],
            X_query=tx,t_query=tt,temperature=torch.ones(1),n_samples=None)[0,0].numpy().astype(float)
        mu0,mu1=result[:n],result[n:];ite=np.full((n,3),np.nan);ite[:,0]=mu1-mu0
        mu=np.full((n,4),np.nan)
        if self.adjusted:mu[:,0]=mu0;mu[:,2]=mu1
        return pack(mu,ite=ite,n=n,diagnostics=dict(status='ok',observed_rows=len(ids),query_units=n,
            context_features=context.shape[1],feature_capacity=capacity,
            interface='Official frozen predict_cepo with ALL M available contexts; no new weight training/retrieval HPO',
            target='Own-treatment contrast queried at majorityS0' if self.adjusted else 'Marginal-own-treatment non-interference misspecification reference; not matched controlled direct'))

