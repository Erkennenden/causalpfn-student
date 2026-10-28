"""Local acceptance adapter proposal, not the team's upstream identification API.

Target/status/node IDs and empirical support must be supplied explicitly.
Direct observation counts are a conservative diagnostic, not a universal
definition of empirical support for models borrowing information across nodes.
"""
import numpy as np


def support_diagnostics(adjacency,treatment,observed_indices):
    a=np.asarray(adjacency);t=np.asarray(treatment);ids=np.asarray(observed_indices)
    n=len(t)
    if (a.shape!=(n,n) or t.shape!=(n,) or ids.ndim!=1
            or not np.issubdtype(ids.dtype,np.integer) or len(set(ids.tolist()))!=len(ids)
            or np.any(ids<0) or np.any(ids>=n) or not np.isin(t,[0,1]).all()
            or not np.array_equal(a,a.T) or np.any(np.diag(a)) or not np.isin(a,[0,1]).all()
            or np.any(a.sum(1)<=0)):
        raise ValueError('Require a simple positive-degree graph and unique observed indices')
    exposure=(a@t>a.sum(1)/2).astype(int)
    states=2*t.astype(int)+exposure
    counts=np.zeros((n,2,2),dtype=int)
    counts[ids,t[ids].astype(int),exposure[ids]]=1
    return dict(state_indices=states,direct_observation_counts=counts,
                direct_observation_mask=counts>0,
                observed_arm_counts=np.bincount(states[ids],minlength=4).reshape(2,2),
                note='One factual draw: each observed unit contributes to exactly one state. Cross-node borrowing is not certified here.')


def estimate_for_request(predictor,inputs,*,target_id,status,node_ids,
                         empirical_support_mask,population_support_mask):
    """Proposal wrapper: honor an external request; never infer identification."""
    if not isinstance(target_id,str) or not target_id:
        raise ValueError('An explicit upstream target_id is required')
    if status not in ('POINT','NOT_IDENTIFIED'):
        raise ValueError('This local proposal only implements POINT/NOT_IDENTIFIED')
    n=len(inputs['x']);nodes=np.asarray(node_ids)
    if nodes.shape!=(n,) or len(set(nodes.tolist()))!=n:
        raise ValueError('node_ids must uniquely match the supplied input row order')
    result=dict(target_id=target_id,status=status,node_ids=nodes.copy(),mu=None,
                sampling_uncertainty=None,empirical_support_mask=None,population_support_mask=None,
                diagnostics=dict(adapter='local_e12_proposal_not_upstream_integrated'))
    if status=='NOT_IDENTIFIED':
        result['diagnostics']['reason']='upstream_NOT_IDENTIFIED; predictor not called'
        return result
    masks=[]
    for mask in (empirical_support_mask,population_support_mask):
        value=np.asarray(mask)
        if value.shape!=(n,2,2) or value.dtype!=np.bool_:
            raise ValueError('Support masks must be explicitly supplied boolean[N,2,2]')
        masks.append(value.copy())
    prediction=predictor(**inputs)
    mu=np.asarray(prediction['arm_means'])
    if mu.shape!=(n,4):
        raise ValueError('Expected original arm order 00,01,10,11 in mu[N,4]')
    result.update(mu=mu.reshape(n,2,2).copy(),empirical_support_mask=masks[0],population_support_mask=masks[1])
    result['diagnostics']['extrapolation_mask']=~masks[0]
    if 'ate_lower' in prediction:
        result['sampling_uncertainty']=dict(ate_lower=np.array(prediction['ate_lower'],copy=True),
            ate_upper=np.array(prediction['ate_upper'],copy=True),kind=str(prediction.get('interval_kind','unspecified')))
    return result


def ordinary_plugin_values(bundle,queries):
    """A local decision boundary: block unsupported queries, never substitute 0."""
    if bundle['status']!='POINT' or bundle['mu'] is None:
        raise ValueError('No point-identified response surface')
    values=[]
    for node,t,s in queries:
        if (not isinstance(node,(int,np.integer)) or not 0<=node<len(bundle['node_ids'])
                or t not in (0,1) or s not in (0,1)):
            raise ValueError('Invalid query row/state')
        if not bundle['empirical_support_mask'][node,t,s] or not bundle['population_support_mask'][node,t,s]:
            raise ValueError('Unsupported query is extrapolation; ordinary plug-in blocked')
        value=bundle['mu'][node,t,s]
        if not np.isfinite(value):
            raise ValueError('Non-finite prediction')
        values.append(value)
    return np.asarray(values)
