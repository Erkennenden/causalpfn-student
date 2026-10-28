"""Simulation-only oracle diagnostic. Never used by the deployable J predictor.

Known true hidden basis, coefficient prior and noise variance; coefficients are
estimated from observed Y only. True coefficients/truth are audit-only inputs
outside oracle_posterior. This is not a claim of a team-approved oracle standard.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri
from threadpoolctl import threadpool_limits
import torch

from audit_e12_evidence import task_summaries, interval_summaries, digest
from make_table import write_csv, EFFECTS
from pfn_social.e12_configuration import generate_mechanism, FAMILY_BASE_IDS
from pfn_social.phase3_hierarchical import coefficient_prior, structural_outcome
from scripts.phase5_strong_controls import design_from_h


def oracle_posterior(factual, query, y, mean, covariance, noise_variance):
    """Conjugate Gaussian mean/interval; no true coefficient or target argument."""
    f, q, y, mean, covariance = [np.asarray(v,dtype=float) for v in (factual,query,y,mean,covariance)]
    if (f.ndim != 2 or q.ndim != 3 or q.shape[1] != 4 or q.shape[2] != f.shape[1]
            or y.shape != (len(f),) or mean.shape != (f.shape[1],)
            or covariance.shape != (f.shape[1],f.shape[1]) or not np.isfinite(noise_variance)
            or noise_variance <= 0):
        raise ValueError('Invalid oracle posterior dimensions or noise')
    if not all(np.isfinite(v).all() for v in (f,q,y,mean,covariance)):
        raise ValueError('Non-finite oracle input')
    factor = cho_factor(f@covariance@f.T+noise_variance*np.eye(len(f)),lower=True)
    gain = covariance@f.T
    posterior_mean = mean+gain@cho_solve(factor,y-f@mean)
    posterior_cov = covariance-gain@cho_solve(factor,gain.T)
    contrasts = np.stack((q[:,2]-q[:,0],q[:,3]-q[:,2],q[:,3]-q[:,0]),1).mean(0)
    ate = contrasts@posterior_mean
    effect_cov = contrasts@posterior_cov@contrasts.T
    variance = np.diag(effect_cov)
    if np.any(variance < -1e-10):
        raise FloatingPointError('Negative oracle posterior variance')
    half = ndtri(.975)*np.sqrt(np.maximum(variance,0))
    return dict(arm_means=q@posterior_mean, ate_effects=ate, ate_lower=ate-half,
                ate_upper=ate+half, effect_covariance=effect_cov)


def enumerate_node_truth(mechanism, node=0):
    """Independently average all neighbor assignments, Bernoulli(0.5)."""
    neighbors = np.flatnonzero(mechanism.adjacency[node])
    degree = len(neighbors)
    if degree > 12:
        raise ValueError('Enumeration is limited to degree<=12')
    patterns = ((np.arange(2**degree)[:,None] >> np.arange(degree)) & 1).astype(float)
    exposure = patterns.mean(1)
    state = exposure > .5
    identity = patterns@mechanism.h[neighbors]/degree
    beta = mechanism.coefficients
    b,tau,gamma,eta = mechanism.components[node]
    result = []
    for own,arm in ((0,0),(0,1),(1,0),(1,1)):
        response = b+own*tau+(gamma+own*eta)*exposure+identity@(beta[19:22]+own*beta[22:25])
        result.append(response[state==arm].mean())
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'predictions').mkdir()
    torch.set_num_threads(1)
    manifest = json.loads((args.input/'execution.json').read_text())
    meta = manifest['extension_protocol']
    lookup = {r['context_id']:r for r in manifest['receipts'] if r['method']=='j'}
    graph_rows = json.loads((args.input/'graph_statistics.json').read_text())
    graph_lookup = {(r['family'],r['mechanism']):r for r in graph_rows}
    records, intervals, runtime, audit = [], [], [], []
    prior_mean, prior_cov = coefficient_prior()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for family in [f for f in meta['families'] if f in FAMILY_BASE_IDS]:
            for mid in meta['mechanism_ids']:
                m = generate_mechanism(family,mid).parent
                expected = graph_lookup[family,mid]
                assert m.mechanism_sha256 == expected['mechanism_sha256']
                assert m.truth_sha256 == expected['truth_sha256']
                with np.load(args.input/'truth'/f'f{family}_m{mid}.npz') as saved:
                    np.testing.assert_array_equal(saved['arm_means'],m.truth)
                enum_error = float(np.max(abs(enumerate_node_truth(m)-m.truth[0])))
                assert enum_error < 1e-10
                audit.append(dict(family=family,mechanism=mid,truth_enumeration_max_error=enum_error,
                                  mechanism_and_truth_hash_match=True))
                truth_ite = np.column_stack((m.truth[:,2]-m.truth[:,0],m.truth[:,3]-m.truth[:,2],m.truth[:,3]-m.truth[:,0]))
                true_ate = truth_ite.mean(0)
                for repeat in meta['repeat_ids']:
                    for count in meta['observed_outcomes']:
                        context=f'f{family}_m{mid}_r{repeat}_observed{count}'
                        assert context in lookup
                        with np.load(args.input/'factual'/f'{context}.npz') as saved:
                            t,ids,y = (saved[k] for k in ('treatment','observed_indices','outcomes_observed'))
                            np.testing.assert_array_equal(saved['adjacency'],m.adjacency)
                            np.testing.assert_array_equal(saved['x'],m.x)
                        factual,query = design_from_h(m.h,m.adjacency,t)
                        # These two checks are simulator diagnostics, not fit inputs.
                        np.testing.assert_allclose(query@m.coefficients,m.truth,atol=1e-10)
                        np.testing.assert_allclose(factual@m.coefficients,structural_outcome(m,t),atol=1e-10)
                        began=time.perf_counter()
                        prediction=oracle_posterior(factual[ids],query,y,prior_mean,prior_cov,m.setting.noise_sd**2)
                        runtime.append(dict(context_id=context,seconds=time.perf_counter()-began))
                        np.savez_compressed(args.output/'predictions'/f'{context}.npz',**prediction)
                        mu=prediction['arm_means']
                        ite=np.column_stack((mu[:,2]-mu[:,0],mu[:,3]-mu[:,2],mu[:,3]-mu[:,0]))
                        common=dict(family=family,mechanism=mid,repeat=repeat,m_observed=count,
                                    context_id=context,method='oracle_true_basis_known_prior_noise')
                        for estimand,values,truth,labels in (
                            ('mu',mu,m.truth,('00','01','10','11')),('ITE',ite,truth_ite,EFFECTS),
                            ('ATE',prediction['ate_effects'][None],true_ate[None],EFFECTS)):
                            for i,effect in enumerate(labels):
                                error=values[:,i]-truth[:,i]
                                records.append(dict(**common,estimand=estimand,effect=effect,valid=True,
                                                    mse=float(np.mean(error**2)),bias=float(error.mean())))
                        degree=m.adjacency.sum(1)
                        states=2*t.astype(int)+(m.adjacency@t>degree/2).astype(int)
                        counts=np.bincount(states[ids],minlength=4)
                        for i,effect in enumerate(EFFECTS):
                            point=prediction['ate_effects'][i];lo=prediction['ate_lower'][i];hi=prediction['ate_upper'][i]
                            center=(lo+hi)/2;half=(hi-lo)/2
                            arms=((0,2),(2,3),(0,3))[i]
                            intervals.append(dict(**common,effect=effect,truth=true_ate[i],point=point,lower=lo,upper=hi,
                                point_error=point-true_ate[i],center_error=center-true_ate[i],half_width=half,
                                width=2*half,score=abs(center-true_ate[i])/half,covered=int(lo<=true_ate[i]<=hi),
                                truth_below=int(true_ate[i]<lo),truth_above=int(true_ate[i]>hi),
                                empty_target_arm=int(any(counts[k]==0 for k in arms))))
                contexts = len(meta['repeat_ids'])*len(meta['observed_outcomes'])
                print(f'Oracle diagnostic f{family} m{mid}: {contexts} contexts; truth enumeration verified',flush=True)
    write_csv(args.output/'context_point_metrics.csv',records)
    tasks,summary=task_summaries(records)
    write_csv(args.output/'task_metrics.csv',tasks)
    write_csv(args.output/'task_mean_sd.csv',summary)
    write_csv(args.output/'interval_contexts.csv',intervals)
    mechanisms,families=interval_summaries(intervals)
    write_csv(args.output/'interval_by_mechanism.csv',mechanisms)
    write_csv(args.output/'interval_by_family.csv',families)
    write_csv(args.output/'truth_checks.csv',audit)
    write_csv(args.output/'runtime.csv',runtime)
    provenance=dict(method='oracle_true_basis_known_prior_noise',simulation_only=True,
        knows=['true hidden basis h','coefficient prior distribution','noise variance'],
        fit_inputs='factual design restricted to observed_indices and outcomes_observed only',
        never_fit_with=['true coefficients','true mu/ITE/ATE','unobserved outcomes'],
        source_execution_sha256=digest(args.input/'execution.json'),script_sha256=digest(__file__),
        contexts=len(runtime),truth_checked_mechanisms=len(audit),nominal_interval_level=.95,
        elapsed_seconds=time.perf_counter()-start,no_J_or_baseline_parameters_modified=True,
        limitation='Diagnostic oracle, not a deployable method or a final team-approved oracle. Conditional coverage for each fixed mechanism is not guaranteed.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)


if __name__ == '__main__':
    main()
