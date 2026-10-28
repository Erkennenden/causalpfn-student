"""Privileged diagnostic: known true representation/prior/noise, unknown beta.

This is not a deployable baseline. It never receives the realized coefficients,
truth responses or unobserved outcomes, and is kept outside practical rankings.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri

from pfn_social.contracts import arm_mean_exposure
from pfn_social.phase3_hierarchical import coefficient_prior
from pfn_social.sparse_frozen_gaussian import validate_factual_inputs


def design(h, hbar, t, e, j):
    t, e = t[:, None], e[:, None]
    one = np.ones_like(t)
    hetero = np.column_stack((one, h))
    return np.column_stack((one, h, hbar, t*hetero, e*hetero, t*e*hetero, j, t*j))


def predict_oracle(*, true_h, noise_sd=1., **inputs):
    x, a, t, ids, y = validate_factual_inputs(**inputs)
    h = np.asarray(true_h, dtype=float)
    if h.shape != (len(x), 3) or not np.isfinite(h).all() or noise_sd <= 0:
        raise ValueError('Invalid privileged representation or known noise')
    degree = a.sum(1)
    hbar = a@h/degree[:, None]
    f = design(h, hbar, t, a@t/degree, a@(t[:, None]*h)/degree[:, None])[ids]
    queries = []
    for own, arm in ((0, 0), (0, 1), (1, 0), (1, 1)):
        e = np.array([arm_mean_exposure(int(d), .5, arm) for d in degree])
        queries.append(design(h, hbar, np.full(len(x), own), e, e[:, None]*hbar))
    q = np.stack(queries, 1)
    mean, cov = coefficient_prior()
    factor = cho_factor(f@cov@f.T + noise_sd**2*np.eye(len(ids)), lower=True)
    posterior_mean = mean + cov@f.T@cho_solve(factor, y-f@mean)
    posterior_cov = cov-cov@f.T@cho_solve(factor, f@cov)
    mu = q@posterior_mean
    c = np.stack((q[:, 2]-q[:, 0], q[:, 3]-q[:, 2], q[:, 3]-q[:, 0]), 1).mean(0)
    ate = c@posterior_mean
    variance = np.einsum('ij,ij->i', c@posterior_cov, c)
    if np.any(variance < -1e-10):
        raise FloatingPointError('Negative oracle posterior variance')
    half = ndtri(.975)*np.sqrt(np.maximum(variance, 0))
    return dict(arm_means=mu, ate_effects=ate, ate_lower=ate-half, ate_upper=ate+half)
