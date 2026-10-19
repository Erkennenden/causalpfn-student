"""Fit HINet and TNet using selected observed outcomes."""
import math


from dataclasses import asdict


import numpy as np


import torch


from torch.nn import functional as F


from pfn_social.phase6_sparse_classical import validate, pack


from pfn_social.estimation import new_hinet, HINET_ALPHA_GRID, seed_device, integrate_hinet, TNetConfig, TNetModel, _build_tnet_optimizers, predict_tnet_majority_effects


def fit_hinet_masked(*,seed,epochs=500,**kwargs):
    x,a,t,ids,y=validate(**kwargs);device=torch.device('cpu')
    x,a,t=(torch.tensor(v,dtype=torch.float32) for v in (x,a,t));obs=torch.tensor(ids);yo=torch.tensor(y,dtype=torch.float32)
    edge=a.nonzero().T.contiguous();permutation=np.random.default_rng(seed).permutation(len(ids));cut=max(1,int(.8*len(ids)))
    train=torch.tensor(permutation[:cut]);valid=torch.tensor(permutation[cut:])
    def one(alpha,positions,initial):
        seed_device(initial,device);model=new_hinet(x.shape[1]);mean=yo[positions].mean();scale=yo[positions].std(unbiased=False).clamp_min(1e-6)
        target=(yo[positions]-mean)/scale;opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
        for _ in range(epochs):
            model.train();opt.zero_grad(set_to_none=True);tp,yp=model(x,t,None,edge)
            loss=F.mse_loss(yp.flatten()[obs[positions]],target)+alpha*F.binary_cross_entropy(tp.flatten(),t)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Masked HINet nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Masked HINet nonfinite gradient')
            opt.step()
        return model.eval(),float(mean),float(scale)
    scores=[]
    for alpha in HINET_ALPHA_GRID:
        values=[]
        for repeat in range(2):
            model,mean,scale=one(alpha,train,seed+1009*repeat)
            with torch.no_grad():pred=model(x,t,None,edge)[1].flatten()[obs[valid]]*scale+mean
            score=float((pred-yo[valid]).square().mean())
            if not math.isfinite(score):raise FloatingPointError('Masked HINet nonfinite factualvalidation')
            values.append(score)
        scores.append(dict(alpha=alpha,factual_validation_mse=float(np.mean(values))))
    threshold=1.1*scores[0]['factual_validation_mse'];alpha=max(r['alpha'] for r in scores if r['factual_validation_mse']<=threshold)
    model,mean,scale=one(alpha,torch.arange(len(ids)),seed+104729)
    official,od=integrate_hinet(model,x,a,samples=4096,seed=seed)
    mu,pd=integrate_hinet(model,x,a,samples=8192,seed=seed);mu=mu*scale+mean
    return pack(mu,n=len(x),diagnostics=dict(epochs=epochs,fits=13,hidden=16,learning_rate=.001,alpha_grid=list(HINET_ALPHA_GRID),alpha_repeats=2,
        selected_alpha=alpha,validation_scores=scores,y_mean=mean,y_scale=scale,
        outcome_loss='Observed TRAIN split only; validation observed labels only; final allM refit',
        treatment_loss='AllN known treatment labels; no withheld outcomes',observed_rows=len(ids),query_units=len(x),
        official4096=od,precision8192=pd,official_vs_precision_max_abs=float(np.max(np.abs(official*scale+mean-mu)))))


def fit_tnet_masked(*,seed,**kwargs):
    x,a,t,ids,y=validate(**kwargs);n=len(x);m=len(ids);device=torch.device('cpu')
    config=TNetConfig(epochs=160,learning_rate_2step=.001,seed=int(seed));seed_device(seed,device)
    z=a@t/a.sum(1);x,a,t,z=(torch.tensor(v,dtype=torch.float32) for v in (x,a,t,z));obs=torch.tensor(ids);yo=torch.tensor(y,dtype=torch.float32)
    model=TNetModel(input_dim=x.shape[1],config=config);location=yo.mean();scale=yo.std(unbiased=True)
    if not torch.isfinite(scale) or scale<=torch.finfo(yo.dtype).eps:scale=yo.new_tensor(1.)
    model.outcome_location.copy_(location);model.outcome_scale.copy_(scale);target=(yo-location)/scale
    # Same sample-size recipe, with M observed outcomes as the labeled sample.
    beta=20/math.sqrt(m) if config.beta is None else config.beta
    nuisance,targeted=_build_tnet_optimizers(model,config);base_parameters=[p for g in nuisance.param_groups for p in g['params']]
    def values():
        h=model.encode(a,x);mu=model.outcome_mean(h,t,z);p1=model.g1_head(h).clamp(1e-6,1-1e-6)
        g1=torch.where(t>=.5,p1,1-p1).clamp_min(config.propensity_floor);g2=model.g2_probability(h,z)
        corrected=mu+model.perturbation(t,z)/(g1.detach()*g2.detach()).clamp_min(config.propensity_floor)
        return mu,p1,g2,corrected
    history=[]
    for epoch in range(1,161):
        model.train();nuisance.zero_grad(set_to_none=True);mu,p1,g2,corrected=values()
        loss_mu=F.mse_loss(mu[obs],target);loss_g= config.alpha*F.binary_cross_entropy(p1,t)+config.gamma*(-torch.log(g2).mean())
        loss=loss_mu+loss_g+beta*F.mse_loss(corrected[obs],target)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Masked TNet nonfinite base loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(base_parameters,1.,error_if_nonfinite=True);nuisance.step()
        for _ in range(50):
            targeted.zero_grad(set_to_none=True);corrected=values()[3];targetloss=beta*F.mse_loss(corrected[obs],target)
            if not bool(torch.isfinite(targetloss)):raise FloatingPointError('Masked TNet nonfinite targeted loss')
            targetloss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);targeted.step()
        history.append(dict(epoch=epoch,base_loss=float(loss.detach()),targeted_loss=float(targetloss.detach())))
    model.eval();prediction=predict_tnet_majority_effects(model,adjacency=a.numpy(),covariates=x.numpy(),treatment_prob=.5)
    mu=prediction.arm_means[:,[0,2,1,3]]
    result=pack(mu,n=n,diagnostics=dict(config=asdict(config),effective_beta=beta,beta_sample_size=m,observed_rows=m,query_units=n,
        y_location=float(location),y_scale=float(scale),base_updates=160,targeted_updates=8000,
        outcome_loss='OnlyM observed outcome labels in everybase/targetedstep',
        treatment_exposure_losses='AllN knownT/exposure; fullgraph encoder retained',
        optimizer_stability='Same priorstabilizedrecipe: targetLR.001 andactivegradnorm1',history=history))
    return result

