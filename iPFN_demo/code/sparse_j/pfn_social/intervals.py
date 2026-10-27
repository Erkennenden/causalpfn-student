"""Marginal posterior intervals for frozen J's average treatment effects.

These are equal-tail Gaussian-mixture credible intervals, not a guarantee of
frequentist coverage. They condition on the frozen model and observed graph.
No new outcome noise is added. Empirical coverage requires separate truth.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, ndtri, softmax
import torch

from pfn_social.phase5_meta_bayes import design
from pfn_social.sparse_frozen_gaussian import (
    _geometry, graph_moments, validate_factual_inputs,
)


def component_effect_posterior(factual, query, y, prior_mean, prior_cov, noise_variances):
    """Return ATE component means/variances[G,3] and log evidence[G].

    Average contrast design rows BEFORE computing variances, preserving
    cross-arm/node covariance for the all-node ATE.
    Effect order is direct (10-00), spillover (11-10), total (11-00).
    """
    f, q, y, mu, cov, noises = [np.asarray(v, dtype=float) for v in
        (factual, query, y, prior_mean, prior_cov, noise_variances)]
    if (f.ndim != 2 or q.ndim != 3 or q.shape[1] != 4 or not len(f)
            or not len(q) or q.shape[-1] != f.shape[-1]
            or y.shape != (len(f),) or mu.shape != (f.shape[1],)
            or cov.shape != (f.shape[1], f.shape[1])
            or noises.ndim != 1 or not len(noises)):
        raise ValueError("Incompatible Gaussian component dimensions")
    if not all(np.isfinite(v).all() for v in (f, q, y, mu, cov, noises)) or np.any(noises <= 0):
        raise ValueError("Require finite inputs and positive noise variances")
    contrasts = np.stack((q[:, 2]-q[:, 0], q[:, 3]-q[:, 2], q[:, 3]-q[:, 0]), 1)
    targets = contrasts.mean(0)
    cross = targets @ cov @ f.T
    prior_var = np.einsum('ij,ij->i', targets @ cov, targets)
    kernel = f @ cov @ f.T
    residual = y - f @ mu
    means, variances, evidences = [], [], []
    for noise in noises:
        factor = cho_factor(kernel + noise * np.eye(len(f)), lower=True)
        solution = cho_solve(factor, residual)
        variance = prior_var - np.einsum('ij,ji->i', cross, cho_solve(factor, cross.T))
        tolerance = 1e-9 * np.maximum(1., np.abs(prior_var))
        if np.any(variance < -tolerance):
            raise FloatingPointError("Negative posterior variance")
        means.append(targets @ mu + cross @ solution)
        variances.append(np.maximum(variance, 0.))
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        evidences.append(-.5 * (logdet + residual @ solution + len(f)*math.log(2*math.pi)))
    return np.array(means), np.array(variances), np.array(evidences)


def mixture_interval(means, variances, weights, level=.95):
    """Invert the marginal mixture CDF; axis 0 indexes components."""
    means, variances, weights = [np.asarray(v, dtype=float) for v in (means, variances, weights)]
    if (means.ndim < 2 or means.shape != variances.shape
            or weights.shape != (len(means),) or not len(means)
            or not 0 < level < 1 or not np.isfinite(level)
            or not all(np.isfinite(v).all() for v in (means, variances, weights))
            or np.any(variances <= 0) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Require finite means, positive variances, nonnegative weights and 0 < level < 1")
    weights = weights / weights.sum()
    weights = weights.reshape((-1,) + (1,)*(means.ndim-1))
    sd = np.sqrt(variances)

    def quantile(probability):
        component_quantiles = means + ndtri(probability)*sd
        lo, hi = component_quantiles.min(0), component_quantiles.max(0)
        for _ in range(64):
            mid = lo + (hi-lo)/2
            cdf = (weights * ndtr((mid-means)/sd)).sum(0)
            lo, hi = np.where(cdf < probability, mid, lo), np.where(cdf >= probability, mid, hi)
        return lo + (hi-lo)/2

    return quantile((1-level)/2), quantile((1+level)/2)


@torch.no_grad()
def predict_intervals(adapter, *, x, adjacency, treatment, observed_indices,
                      outcomes_observed, level=.95):
    """Return ATE intervals using J's observed inputs and mixture weights."""
    x, a, t, indices, y = validate_factual_inputs(x=x, adjacency=adjacency,
        treatment=treatment, observed_indices=observed_indices, outcomes_observed=outcomes_observed)
    parent = adapter.model
    if parent.config.architecture != 'J':
        raise ValueError("Intervals require the frozen J model")
    xt, at = torch.tensor(x, dtype=torch.float32)[None], torch.tensor(a, dtype=torch.float32)[None]
    tt = torch.tensor(t, dtype=torch.float32)[None, :, None]
    et = torch.tensor(a @ t / a.sum(1), dtype=torch.float32)[None, :, None]
    yt = torch.tensor(y, dtype=torch.float32).double().numpy()
    moments, degree = graph_moments(at), at.sum(-1, keepdim=True)
    noise = (parent.noise_variance_grid.double() * parent.log_grid_scale.clamp(-2, 2).exp()).numpy()
    all_means, all_variances, all_evidences = [], [], []
    for representation, prior in zip((parent.encoder(xt, at), xt), parent.priors):
        hb = at @ representation / degree
        factual = design(representation, hb, tt, et, at @ (tt*representation) / degree)
        queries = []
        for own, arm in ((0, 0), (0, 1), (1, 0), (1, 1)):
            em = moments[..., arm:arm+1]
            queries.append(design(representation, hb, torch.full_like(tt, own), em, em*hb))
        query = torch.stack(queries, -2).double()
        lower = prior.lower().double()
        _, chol, sigma_c, gamma = _geometry(query, lower, parent.gamma())
        cov = lower @ lower.T + gamma * sigma_c @ torch.cholesky_solve(sigma_c.transpose(-1, -2), chol)
        mean, variance, evidence = component_effect_posterior(factual[0, indices].double().numpy(),
            query[0].numpy(), yt, prior.mean.double().numpy(), cov[0].numpy(), noise)
        all_means.append(mean)
        all_variances.append(variance)
        all_evidences.append(evidence)
    log_prior = (torch.log_softmax(parent.representation_logits.double(), -1)[:, None]
                 + torch.log_softmax(parent.noise_logits.double(), -1)[None, :]).numpy()
    weights = softmax((np.stack(all_evidences) + log_prior).ravel())
    means, variances = np.concatenate(all_means), np.concatenate(all_variances)
    lo, hi = mixture_interval(means, variances, weights, level)
    return dict(ate_lower=lo, ate_upper=hi, interval_level=np.array(level),
                interval_kind=np.array('frozen_J_marginal_posterior_credible'))
