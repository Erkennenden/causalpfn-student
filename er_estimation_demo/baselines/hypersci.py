"""HyperSCI on pair-edge hypergraphs, adapted to majority-arm ITE.

Fits only factual A, X, T and Y. Conditional assignment integration is Monte
Carlo; no true coefficient, outcome surface or effect is used to fit or tune.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from estimands import conditional_count_weights
from .hypersci_vendor import model as UPSTREAM
from .hypersci_vendor import utils as UTILS

def pair_incidence(adjacency):
    a=torch.as_tensor(adjacency)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not torch.equal(a,a.T):
        raise ValueError('Need an undirected square adjacency.')
    if torch.any(a.diag()!=0) or torch.any((a!=0)&(a!=1)) or torch.any(a.sum(1)<=0):
        raise ValueError('Need a binary graph without loops or isolated nodes.')
    pairs=torch.nonzero(torch.triu(a,diagonal=1),as_tuple=False)
    index=torch.stack([pairs.reshape(-1),torch.arange(len(pairs)).repeat_interleave(2)])
    return index,pairs


def make_model(input_dim,hidden=32):
    args=SimpleNamespace(h_dim=hidden,g_dim=hidden,n_out=0,dropout=.5,
        graph_model='hypergraph',encoder='gat',skip='123',num_gnn_layer=1,
        phi_layer=2,activate=1)
    return UPSTREAM.HyperSCI(args,input_dim)


@torch.no_grad()
def build_pair_cache(model,features,adjacency):
    """Exact decomposition of one PyG HypergraphConv layer on size-two edges.

    Attention is normalized inside each edge. Evaluate its four binary endpoint
    assignments once, then sum incident messages with the original node degree.
    Dropout must be disabled; treatment-dependent attention is recomputed in all
    four cases. This is an evaluation optimization, not an approximate model.
    """
    if model.training or model.num_gnn_layer!=1 or model.encoder!='gat' or model.n_out!=0:
        raise ValueError('Cache requires the one-layer attention evaluation configuration.')
    if model.hgnn.attention_mode!='node':
        raise ValueError('Only within-edge attention supports this decomposition.')
    a=torch.as_tensor(adjacency)
    index,pairs=pair_incidence(a)
    x=torch.as_tensor(features,dtype=torch.float32)
    phi=model.phi_x(x)
    edge_count=len(pairs)
    duplicated=phi[pairs.reshape(-1)]
    disjoint=torch.stack([torch.arange(2*edge_count),torch.arange(edge_count).repeat_interleave(2)])
    messages=torch.empty(edge_count,2,2,2,model.g_dim)
    bias=model.hgnn.bias
    for tu in (0,1):
        for tv in (0,1):
            treatments=torch.tensor([tu,tv],dtype=x.dtype).repeat(edge_count)
            masked=duplicated*treatments[:,None]
            attrs=masked.reshape(edge_count,2,-1).mean(1)
            pair_outputs=model.hgnn(masked,disjoint,hyperedge_attr=attrs)
            messages[:,:,tu,tv]=(pair_outputs-bias).reshape(edge_count,2,-1)
    degree=a.sum(1).long()
    neighbors=[torch.where(row>0)[0] for row in a]
    edge_lookup={tuple(pair):e for e,pair in enumerate(pairs.tolist())}
    base=torch.empty(len(x),2,model.g_dim)
    deltas=[]
    for i,adjacent in enumerate(neighbors):
        delta=torch.empty(2,len(adjacent),model.g_dim)
        for ti in (0,1):
            zero=[]
            for k,j in enumerate(adjacent.tolist()):
                u,v=sorted((i,j));e=edge_lookup[u,v]
                if i==u:
                    q0=messages[e,0,ti,0];q1=messages[e,0,ti,1]
                else:
                    q0=messages[e,1,0,ti];q1=messages[e,1,1,ti]
                zero.append(q0)
                delta[ti,k]=(q1-q0)/degree[i]
            base[i,ti]=torch.stack(zero).sum(0)/degree[i]+bias
        deltas.append(delta)
    return dict(features=x,phi=phi,base=base,deltas=deltas,degree=degree,neighbors=neighbors)


@torch.no_grad()
def predict_root_samples(model,cache,root,own_treatment,assignments):
    t=int(own_treatment)
    mask=torch.as_tensor(assignments,dtype=cache['features'].dtype)
    if mask.ndim!=2 or mask.shape[1]!=int(cache['degree'][root]):
        raise ValueError('Need one assignment per immediate neighbor.')
    interference=cache['base'][root,t]+mask@cache['deltas'][root][t]
    if model.activate:interference=F.relu(interference)
    n=len(mask)
    x=cache['features'][root].expand(n,-1)
    phi=cache['phi'][root].expand(n,-1)
    rep=torch.cat([x,phi if t else torch.zeros_like(phi),interference],dim=-1)
    head=model.out_t11 if t else model.out_t01
    return head(rep).squeeze(-1).numpy()


def conditional_assignments(degree,arm,samples,*,seed,treatment_prob=.5):
    rng=np.random.default_rng(seed)
    weights=conditional_count_weights(int(degree),float(treatment_prob),int(arm))
    counts=rng.choice(degree+1,size=samples,p=weights)
    # Random ranks give a uniform subset conditional on the drawn count.
    ranks=rng.random((samples,degree)).argsort(axis=1).argsort(axis=1)
    return (ranks<counts[:,None]).astype(np.float32)


def integrate_majority(model,cache,*,samples=4096,seed=2026,treatment_prob=.5,progress=None):
    if samples<2:raise ValueError('Need at least two draws to estimate Monte Carlo error.')
    n=len(cache['degree']);est=np.empty((n,3));se=np.empty((n,3))
    for i in range(n):
        d=int(cache['degree'][i])
        masks=[conditional_assignments(d,s,samples,seed=np.random.SeedSequence([seed,i,s,7301]),treatment_prob=treatment_prob)
               for s in (0,1)]
        low0=predict_root_samples(model,cache,i,0,masks[0]).astype(np.float64)
        low1=predict_root_samples(model,cache,i,1,masks[0]).astype(np.float64)
        high1=predict_root_samples(model,cache,i,1,masks[1]).astype(np.float64)
        de=low1-low0
        est[i]=[de.mean(),high1.mean()-low1.mean(),high1.mean()-low0.mean()]
        se[i]=[de.std(ddof=1)/np.sqrt(samples),
               np.sqrt((high1.var(ddof=1)+low1.var(ddof=1))/samples),
               np.sqrt((high1.var(ddof=1)+low0.var(ddof=1))/samples)]
        if progress is not None and (i+1)%100==0:progress(i+1,n)
    return est,se


def fit_hyper(adjacency,x,treatment,outcomes,*,seed=2026,epochs=500,
              learning_rate=.001,balance_weight=.01,hidden_dim=32,log=None):
    """One factual-graph fit. No DGP coefficient or causal truth accepted."""
    if epochs <= 0 or hidden_dim <= 0 or learning_rate <= 0 or balance_weight < 0:
        raise ValueError('Need positive epochs, hidden_dim, learning_rate and nonnegative balance_weight.')
    a=torch.as_tensor(adjacency,dtype=torch.float32)
    index,_=pair_incidence(a)
    n=len(a)
    for name, values in [('x',x), ('treatment',treatment), ('outcomes',outcomes)]:
        array=np.asarray(values)
        if array.shape!=(n,) or not np.isfinite(array).all():
            raise ValueError(f'{name} must be a finite length-N vector.')
    if not np.isin(treatment,[0,1]).all():
        raise ValueError('Treatment must be binary.')
    if len(np.unique(treatment))!=2:
        raise ValueError('HyperSCI requires observations from both treatment groups.')
    torch.manual_seed(seed)
    features=np.column_stack([np.asarray(x),a.sum(1).numpy()/float(n-1)])
    std=features.std(0);std[std<1e-8]=1.
    features=torch.tensor((features-features.mean(0))/std,dtype=torch.float32)
    treatment=torch.as_tensor(treatment,dtype=torch.float32)
    y=np.asarray(outcomes);mean=float(y.mean());scale=max(float(y.std()),1e-8)
    y=torch.tensor((y-mean)/scale,dtype=torch.float32)
    model=make_model(2,hidden=hidden_dim)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=.01)
    losses=[]
    for epoch in range(epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        result=model(features,treatment,index)
        prediction=torch.where(treatment>0,result['y1_pred'],result['y0_pred'])
        mse=F.mse_loss(prediction,y)
        treated=result['rep'][treatment>0];control=result['rep'][treatment==0]
        balance,_=UTILS.wasserstein(treated,control,torch.device('cpu'),cuda=False)
        loss=mse+balance_weight*balance
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at {epoch+1}.')
        loss.backward();optimizer.step()
        entry=dict(epoch=epoch+1,loss=float(loss.detach()),mse=float(mse.detach()),wasserstein=float(balance.detach()))
        losses.append(entry)
        if log is not None and (epoch==0 or (epoch+1)%50==0):log(entry)
    model.eval().requires_grad_(False)
    return model,build_pair_cache(model,features,a),mean,scale,losses


def fit_predict_hypersci_ite(*, adjacency, x, treatment, outcomes, treatment_prob,
                             epochs=500, learning_rate=.001, balance_weight=.01,
                             hidden_dim=32, seed=2026, arm_samples=4096):
    """Return (ITE, integration MC standard error), each [N,3] in Y units."""
    if not 0.0 < float(treatment_prob) < 1.0:
        raise ValueError('treatment_prob must be strictly between zero and one.')
    if arm_samples < 2:
        raise ValueError('arm_samples must be at least two.')
    with torch.random.fork_rng(devices=[]):
        model,cache,_,scale,_=fit_hyper(adjacency,x,treatment,outcomes,
            seed=seed,epochs=epochs,learning_rate=learning_rate,
            balance_weight=balance_weight,hidden_dim=hidden_dim)
        estimate,mcse=integrate_majority(model,cache,samples=arm_samples,seed=seed,
                                      treatment_prob=treatment_prob)
    return estimate*scale,mcse*scale
