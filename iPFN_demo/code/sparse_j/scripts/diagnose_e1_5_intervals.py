"""Independent deterministic audit of saved frozen-J ATE credible intervals.

Never imports project predictors, interval functions or DGP generators. Loads
raw checkpoint tensors, reconstructs the frozen encoder/design, and solves in
coefficient space (the production interval code solves in observation space).
Uses erfc + Brent roots, not production ndtr + 64-step bisection. No RNG calls.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.stats import binom
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parents[1]
EFFECTS = ('direct', 'spillover', 'total')
CONTRAST = np.array([[-1., 0., 1., 0.], [0., 0., -1., 1.], [-1., 0., 0., 1.]])
# Fixed audit tolerances; not fitted to pilot coverage or widened on failure.
CDF_TOL, ENDPOINT_TOL, MEAN_TOL = 1e-8, 1e-7, 2e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def solve_chol(chol, rhs):
    return solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)


def coefficient_posterior(factual, residual, prior_mean, prior_cov, noise):
    """Whitened primal precision; do not subtract two variances."""
    b = np.linalg.cholesky(prior_cov)
    h = factual @ b
    precision = np.eye(b.shape[0]) + h.T @ h / noise
    chol = np.linalg.cholesky(precision)
    rhs = h.T @ residual / noise
    latent_mean = solve_chol(chol, rhs)
    mean = prior_mean + b @ latent_mean
    cov = b @ solve_chol(chol, b.T)
    logdet = len(factual)*math.log(noise) + 2*np.log(np.diag(chol)).sum()
    quadratic = residual @ residual / noise - rhs @ latent_mean
    log_evidence = -.5*(logdet+quadratic+len(factual)*math.log(2*math.pi))
    return mean, cov, float(log_evidence)


def normal_cdf(z):
    return .5*math.erfc(-float(z)/math.sqrt(2))


def mixture_cdf(value, means, variances, weights):
    return math.fsum(float(w)*normal_cdf((value-m)/math.sqrt(v))
                     for m, v, w in zip(means, variances, weights))


def independent_quantile(probability, means, variances, weights):
    sd = np.sqrt(variances)
    left = float(np.min(means-12*sd))
    right = float(np.max(means+12*sd))
    return float(brentq(lambda x: mixture_cdf(x, means, variances, weights)-probability,
                        left, right, xtol=5e-13, rtol=5e-14))


def self_checks():
    mean, cov, _ = coefficient_posterior(np.ones((1, 1)), np.array([2.]),
                                         np.zeros(1), np.eye(1), 1.)
    np.testing.assert_allclose(mean, [1.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(cov, [[.5]], rtol=0, atol=1e-14)
    for p in (.025, .975):
        actual = independent_quantile(p, mean, cov.diagonal(), np.ones(1))
        expected = 1 + math.sqrt(.5)*statistics.NormalDist().inv_cdf(p)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    # Deterministic correlated 2D case: independent conditioning identity.
    prior = np.array([[2., .5], [.5, 1.]])
    y = np.array([1., -1.])
    mean, cov, _ = coefficient_posterior(np.eye(2), y, np.zeros(2), prior, .7)
    kernel_inverse = np.linalg.inv(prior+.7*np.eye(2))
    np.testing.assert_allclose(mean, prior @ kernel_inverse @ y, atol=1e-13)
    np.testing.assert_allclose(cov, prior-prior @ kernel_inverse @ prior, atol=1e-13)
    return dict(passed=True, cases=['scalar conjugate posterior', 'analytic normal quantiles',
                                  'correlated 2D conditioning identity'])


def log_probabilities(logits):
    values = np.asarray(logits, dtype=float)
    shift = float(values.max())
    return values-shift-math.log(float(np.exp(values-shift).sum()))


def design_rows(r, hbar, own, exposure, treated_neighbors):
    ones = torch.ones_like(own)
    hetero = torch.cat((ones, r), -1)
    return torch.cat((ones, r, hbar, own*hetero, exposure*hetero,
                      own*exposure*hetero, treated_neighbors, own*treated_neighbors), -1)


def representations_and_moments(inputs, state):
    # Match the frozen feature arithmetic's declared float32/double boundaries.
    x = torch.as_tensor(inputs['x'], dtype=torch.float32)
    a = torch.as_tensor(inputs['adjacency'], dtype=torch.float32)
    t = torch.as_tensor(inputs['treatment'], dtype=torch.float32)[:, None]
    degree = a.sum(1, keepdim=True)
    z = ((x.double()-state['encoder.x_mean'])/state['encoder.x_scale']) @ state['encoder.projection']
    u, v = z[:, 0], z[:, 1]
    lift = torch.stack((u, v, (u*u-1)/math.sqrt(2), u*v, (v*v-1)/math.sqrt(2)), -1)
    wz = a.double() @ z / degree.double()
    features = torch.cat((lift, wz, a.double() @ wz / degree.double(),
                          z.mean(0, keepdim=True).expand_as(z), degree.double().log1p()), -1).float()
    hidden = torch.tanh(F.linear(torch.tanh(F.linear(features, state['encoder.network.0.weight'],
                                                    state['encoder.network.0.bias'])),
                                 state['encoder.network.2.weight'], state['encoder.network.2.bias']))
    # Exact deterministic binomial integration, not the project's cached helper.
    moments = np.empty((len(x), 2))
    for d in np.unique(degree.numpy().astype(int)):
        counts = np.arange(d+1)
        probabilities = binom.pmf(counts, d, .5)
        for arm in (0, 1):
            mask = counts > d//2 if arm else counts <= d//2
            expectation = np.dot(probabilities[mask], counts[mask]/d)/probabilities[mask].sum()
            moments[degree.numpy().ravel() == d, arm] = expectation
    exposure = (np.asarray(inputs['adjacency'], dtype=float) @ np.asarray(inputs['treatment'], dtype=float)
                / np.asarray(inputs['adjacency'], dtype=float).sum(1))
    return (hidden, x), a, t, degree, torch.tensor(exposure, dtype=torch.float32)[:, None], torch.tensor(moments, dtype=torch.float32)


@torch.no_grad()
def audit_components(inputs, state):
    representations, a, t, degree, exposure, moments = representations_and_moments(inputs, state)
    indices = inputs['observed_indices']
    y = np.asarray(inputs['outcomes_observed'], dtype=np.float32).astype(float)
    gamma = float(state['log1p_gamma'].clamp(0., math.log(65.)).expm1())
    noises = (state['noise_variance_grid'].double()*state['log_grid_scale'].clamp(-2, 2).exp()).numpy()
    all_means, all_cov, evidence, labels, covariance_checks = [], [], [], [], []
    for rep_id, r in enumerate(representations):
        hbar = a @ r / degree
        factual = design_rows(r, hbar, t, exposure, a @ (t*r) / degree).double().numpy()[indices]
        queries = []
        for own, arm in ((0, 0), (0, 1), (1, 0), (1, 1)):
            em = moments[:, arm:arm+1]
            queries.append(design_rows(r, hbar, torch.full_like(t, own), em, em*hbar).double().numpy())
        q = np.stack(queries, axis=1)
        targets = np.einsum('ea,naq->neq', CONTRAST, q).mean(0)
        raw = state[f'priors.{rep_id}.raw_cholesky']
        lower = (torch.tril(raw, -1)+torch.diag(F.softplus(raw.diagonal())+.01)).double().numpy()
        prior_mean = state[f'priors.{rep_id}.mean'].double().numpy()
        original_cov = lower @ lower.T
        c = targets[:2]
        cross = original_cov @ c.T
        inflated = original_cov + gamma*cross @ np.linalg.solve(c @ original_cov @ c.T, cross.T)
        residual = y-factual @ prior_mean
        for noise_id, noise in enumerate(noises):
            mean, posterior_cov, ev = coefficient_posterior(factual, residual, prior_mean, inflated, float(noise))
            ate_cov = targets @ posterior_cov @ targets.T
            # Cross-state verification: average arms, then include their covariance.
            arm_cov = q.mean(0) @ posterior_cov @ q.mean(0).T
            via_arms = CONTRAST @ arm_cov @ CONTRAST.T
            # Cross-node verification: split 200 nodes into four deterministic
            # blocks and retain every block-pair covariance, including off-diagonal.
            block_rows = [np.einsum('ea,naq->neq', CONTRAST, block).mean(0)
                          for block in np.array_split(q, 4)]
            blocks = np.array([[b @ posterior_cov @ d.T for d in block_rows] for b in block_rows])
            via_blocks = blocks.sum((0, 1))/16
            wrong_block_independence = sum(blocks[i, i] for i in range(4))/16
            all_means.append(targets @ mean)
            all_cov.append(ate_cov)
            evidence.append(ev)
            labels.append(dict(representation='learned3' if rep_id == 0 else 'raw20', noise_index=noise_id, noise_variance=float(noise)))
            covariance_checks.append(dict(cross_arm_max_error=float(np.max(np.abs(ate_cov-via_arms))),
                 cross_node_block_sum_max_error=float(np.max(np.abs(ate_cov-via_blocks))),
                 total_variance_identity_error=float(abs(ate_cov[2, 2]-ate_cov[0, 0]-ate_cov[1, 1]-2*ate_cov[0, 1])),
                 correct_ate_variance=np.diag(ate_cov).tolist(),
                 incorrect_independent_block_variance=np.diag(wrong_block_independence).tolist()))
    prior_log = (log_probabilities(state['representation_logits'].double().numpy())[:, None]
                 + log_probabilities(state['noise_logits'].double().numpy())[None, :]).ravel()
    log_joint = np.asarray(evidence)+prior_log
    weights = np.exp(log_joint-log_joint.max())
    weights /= weights.sum()
    means, covariances = np.array(all_means), np.array(all_cov)
    variances = np.diagonal(covariances, axis1=1, axis2=2).copy()
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise ArithmeticError('Independent posterior has invalid variance; do not clip or repair')
    return means, variances, covariances, weights, labels, np.asarray(evidence), prior_log, covariance_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic directory only')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        parser.error('Use a new directory outside the immutable pilot run')
    started = time.perf_counter()
    manifest = json.loads((source/'run_manifest.json').read_text(encoding='utf-8'))
    execution = json.loads((source/'execution.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['plan']['cohort'] != 'pilot_fresh':
        raise ValueError('Require the completed pilot_fresh artifacts')
    for name, value in manifest['output_sha256'].items():
        if digest(source/name) != value:
            raise ValueError(f'Pilot hash mismatch: {name}')
    checkpoint = CODE/'models/iPFN_J.pt'
    if digest(checkpoint) != manifest['assets_sha256']['models/iPFN_J.pt']:
        raise ValueError('Frozen checkpoint hash mismatch')
    raw_checkpoint = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if raw_checkpoint['config'] != dict(architecture='J', raw_p=20, hidden=64):
        raise ValueError('Unexpected checkpoint structure')
    state = raw_checkpoint['model_state']
    selected = [f'f0_m905_r{r}_observed50' for r in range(10)] + [
        f'f0_m{m}_r{c}_observed50' for m in (904, 906, 907) for c in (0, 1)]
    receipts = {(r['context_id'], r['method']): r for r in execution['receipts']}
    torch.set_num_threads(1)
    rows, contexts = [], []
    output.mkdir(parents=True, exist_ok=False)
    (output/'components').mkdir()
    checks = self_checks()
    with threadpool_limits(limits=1):
        for cid in selected:
            receipt = receipts[cid, 'j']
            factual = source/'factual'/(cid+'.npz')
            if digest(factual) != receipt['factual_sha256'] or receipt['interval_failure'] is not None:
                raise ValueError(f'Invalid saved context: {cid}')
            with np.load(factual, allow_pickle=False) as saved:
                inputs = {k: saved[k] for k in saved.files}
            means, variances, covariances, weights, labels, log_evidence, log_prior, cov_checks = audit_components(inputs, state)
            with np.load(source/'predictions'/(cid+'__j.npz'), allow_pickle=False) as saved:
                original = {k: saved[k] for k in saved.files}
            mid = receipt['mechanism_id']
            with np.load(source/'truth'/f'f0_m{mid}.npz', allow_pickle=False) as saved:
                truth = CONTRAST @ saved['arm_means'].mean(0)
            mixture_mean = weights @ means
            for j, effect in enumerate(EFFECTS):
                low = independent_quantile(.025, means[:, j], variances[:, j], weights)
                high = independent_quantile(.975, means[:, j], variances[:, j], weights)
                saved_lo, saved_hi = float(original['ate_lower'][j]), float(original['ate_upper'][j])
                cdf_lo = mixture_cdf(saved_lo, means[:, j], variances[:, j], weights)
                cdf_hi = mixture_cdf(saved_hi, means[:, j], variances[:, j], weights)
                rows.append(dict(context_id=cid, mechanism_id=mid, center_t=receipt['center_t'], effect=effect,
                    saved_lower=saved_lo, independent_lower=low, lower_difference=low-saved_lo,
                    saved_upper=saved_hi, independent_upper=high, upper_difference=high-saved_hi,
                    cdf_at_saved_lower=cdf_lo, cdf_at_saved_upper=cdf_hi,
                    lower_cdf_residual=cdf_lo-.025, upper_cdf_residual=cdf_hi-.975,
                    saved_ate_mean=float(original['ate_effects'][j]), reconstructed_functional_mean=float(mixture_mean[j]),
                    mean_difference=float(mixture_mean[j]-original['ate_effects'][j]),
                    truth=float(truth[j]), cdf_at_truth=mixture_cdf(truth[j], means[:, j], variances[:, j], weights)))
            np.savez_compressed(output/'components'/(cid+'.npz'), component_means=means,
                                component_variances=variances, component_covariances=covariances,
                                weights=weights, log_evidence=log_evidence, log_prior=log_prior)
            contexts.append(dict(context_id=cid, weight_sum=float(weights.sum()), weights=weights.tolist(), labels=labels,
                                 minimum_component_variance=float(variances.min()), covariance_checks=cov_checks))
    summary = dict(contexts=len(selected), marginal_intervals=len(rows), component_count_per_context=8,
                   max_abs_endpoint_difference=max(max(abs(r['lower_difference']), abs(r['upper_difference'])) for r in rows),
                   max_abs_cdf_residual=max(max(abs(r['lower_cdf_residual']), abs(r['upper_cdf_residual'])) for r in rows),
                   max_abs_functional_mean_difference=max(abs(r['mean_difference']) for r in rows),
                   max_weight_normalization_error=max(abs(c['weight_sum']-1) for c in contexts),
                   minimum_component_variance=min(c['minimum_component_variance'] for c in contexts),
                   max_cross_arm_covariance_error=max(d['cross_arm_max_error'] for c in contexts for d in c['covariance_checks']),
                   max_cross_node_block_covariance_error=max(d['cross_node_block_sum_max_error'] for c in contexts for d in c['covariance_checks']),
                   max_total_variance_identity_error=max(d['total_variance_identity_error'] for c in contexts for d in c['covariance_checks']))
    passed = (summary['max_abs_endpoint_difference'] <= ENDPOINT_TOL and summary['max_abs_cdf_residual'] <= CDF_TOL
              and summary['max_abs_functional_mean_difference'] <= MEAN_TOL
              and max(summary['max_cross_arm_covariance_error'], summary['max_cross_node_block_covariance_error'],
                      summary['max_total_variance_identity_error']) < 1e-9)
    for name, value in manifest['output_sha256'].items():
        if digest(source/name) != value:
            raise ValueError(f'Pilot changed during audit: {name}')
    with (output/'endpoint_checks.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_write(output/'component_diagnostics.json', contexts)
    report = dict(passed=passed, summary=summary, self_checks=checks,
                  tolerances=dict(cdf=CDF_TOL, endpoints=ENDPOINT_TOL, functional_mean=MEAN_TOL),
                  method='Manual frozen encoder/design, whitened coefficient-space posterior; math.erfc CDF; scipy Brent roots',
                  no_point_prediction_api_called=True, no_dgp_or_random_sampling=True,
                  pilot_output_hashes_unchanged=True, checkpoint_sha256=digest(checkpoint),
                  diagnostic_sha256=digest(__file__), python=sys.version, wall_seconds=time.perf_counter()-started)
    json_write(output/'report.json', report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit('Independent audit differs from saved intervals; evidence preserved, no retry or tuning')


if __name__ == '__main__':
    main()
