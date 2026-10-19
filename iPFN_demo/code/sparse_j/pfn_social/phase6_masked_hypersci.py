"""Fit HyperSCI with an outcome mask and integrate majority-state queries."""
import numpy as np


import torch


from torch.nn import functional as F


from pfn_social.estimation import pair_incidence, seed_device, make_model, UTILS, build_pair_cache, predict_root_samples, conditional_assignments, fork_device_rng


EPOCHS=1601


LEARNING_RATE=.001


HIDDEN_DIM=32


BALANCE_WEIGHT=.01


ARM_SAMPLES=4096


def _arrays(x,adjacency,treatment,observed_indices,outcomes_observed):
    x=np.asarray(x);a=np.asarray(adjacency);t=np.asarray(treatment);idx=np.asarray(observed_indices);y=np.asarray(outcomes_observed)
    n=len(x)
    if x.ndim!=2 or a.shape!=(n,n) or t.shape!=(n,):raise ValueError('Full-network array shapes are incompatible')
    if idx.ndim!=1 or idx.dtype.kind not in 'iu' or len(idx)<2 or len(np.unique(idx))!=len(idx):raise ValueError('Unique observed integer indices required')
    if idx.min()<0 or idx.max()>=n or y.shape!=(len(idx),):raise ValueError('Exactly M observed outcomes required')
    if any(not np.isfinite(v).all() for v in (x,a,t,y)) or not np.isin(t,[0,1]).all() or len(np.unique(t))!=2:raise ValueError('Finite inputs and both binary treatment groups required')
    pair_incidence(a)
    return x,a,t,idx,y


def fit_masked_hyper(*,x,adjacency,treatment,observed_indices,outcomes_observed,
                     seed,epochs=EPOCHS,device='cpu'):
    if epochs<=0:raise ValueError('Positive epoch count required')
    x,a,t,idx,y=_arrays(x,adjacency,treatment,observed_indices,outcomes_observed)
    a=torch.as_tensor(a,dtype=torch.float32);index,_=pair_incidence(a);n=len(a)
    device=torch.device(device);seed_device(seed,device)
    features=np.column_stack([x,a.sum(1).numpy()/float(n-1)])
    std=features.std(0);std[std<1e-8]=1.
    features=torch.tensor((features-features.mean(0))/std,dtype=torch.float32,device=device)
    t=torch.as_tensor(t,dtype=torch.float32,device=device)
    # y is M-dimensional throughout fitting; no unknown-Y placeholders exist.
    mean=float(y.mean());scale=max(float(y.std()),1e-8)
    y=torch.tensor((y-mean)/scale,dtype=torch.float32,device=device)
    idx=torch.tensor(idx,dtype=torch.int64,device=device);index=index.to(device)
    model=make_model(features.shape[1],hidden=HIDDEN_DIM).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=.01)
    losses=[]
    for epoch in range(epochs):
        model.train();optimizer.zero_grad(set_to_none=True);result=model(features,t,index)
        prediction=torch.where(t>0,result['y1_pred'],result['y0_pred'])
        mse=F.mse_loss(prediction[idx],y)
        # All treatment statuses are observed, so this unchanged auxiliary loss
        # uses both full-network treatment groups, independently of Y masking.
        balance,_=UTILS.wasserstein(result['rep'][t>0],result['rep'][t==0],device,cuda=device.type!='cpu')
        loss=mse+BALANCE_WEIGHT*balance
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite masked HyperSCI loss at epoch{epoch+1}')
        loss.backward();optimizer.step()
        losses.append(dict(epoch=epoch+1,loss=float(loss.detach()),mse=float(mse.detach()),wasserstein=float(balance.detach())))
    model.eval().requires_grad_(False)
    return model,build_pair_cache(model,features,a),mean,scale,losses


@torch.no_grad()
def integrate_masked_hyper(model,cache,mean,scale,*,seed,samples=ARM_SAMPLES):
    if samples<2:raise ValueError('At least two conditional samples required')
    n=len(cache['degree']);arms=np.empty((n,4));mcse=np.empty((n,4))
    for node in range(n):
        degree=int(cache['degree'][node])
        for arm in (0,1):
            assignments=conditional_assignments(degree,arm,samples,
                seed=np.random.SeedSequence([seed,node,arm,7301]),treatment_prob=.5)
            for own in (0,1):
                values=predict_root_samples(model,cache,node,own,assignments).astype(np.float64)
                arms[node,2*own+arm]=mean+scale*values.mean()
                mcse[node,2*own+arm]=scale*values.std(ddof=1)/np.sqrt(samples)
    return arms,mcse


def fit_predict_masked_hypersci(*,x,adjacency,treatment,observed_indices,outcomes_observed,
                              seed,epochs=EPOCHS,samples=ARM_SAMPLES):
    with fork_device_rng('cpu'):
        model,cache,mean,scale,history=fit_masked_hyper(x=x,adjacency=adjacency,treatment=treatment,
            observed_indices=observed_indices,outcomes_observed=outcomes_observed,seed=seed,epochs=epochs)
        arms,mcse=integrate_masked_hyper(model,cache,mean,scale,seed=seed,samples=samples)
    return arms,dict(model=model,history=history,y_mean=mean,y_scale=scale,arm_mcse=mcse)

