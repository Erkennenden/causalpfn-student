"""Cache Gaussian features and compute calibrated posterior coefficients."""
import math


import torch


from pfn_social.sparse_frozen_gaussian import ate_target_rows


DYNAMIC_KEYS = ('eigenvalues', 'f_lower_eigen', 'residual_projection', 'target_eigen')


@torch.no_grad()
def component_cache(factual, query, y_observed, prior):
    """Cache a fixed Gaussian component; no target/outcome-oracle input exists.

    f[B,M,Q], q[B or1,N,4,Q], y[B,M]. SVD uses all left vectors, retaining
    null-space residuals and the exact M-dimensional evidence normalizer.
    """
    f, q, y = factual.double(), query.double(), y_observed.double()
    if f.ndim != 3 or q.ndim != 4 or y.shape != f.shape[:2] or q.shape[-2] != 4:
        raise ValueError('Require f[B,M,Q], q[B or1,N,4,Q], y[B,M]')
    if q.shape[0] not in (1, f.shape[0]) or q.shape[-1] != f.shape[-1]:
        raise ValueError('Incompatible component dimensions')
    lower, mean = prior.lower().double(), prior.mean.double()
    fl = f @ lower
    c_lower = ate_target_rows(q) @ lower
    target_chol = torch.linalg.cholesky(c_lower @ c_lower.transpose(-1, -2))
    z = torch.linalg.solve_triangular(target_chol, c_lower, upper=False).transpose(-1, -2)
    u, singular, _ = torch.linalg.svd(fl, full_matrices=True)
    eigenvalues = torch.zeros(f.shape[:2], dtype=f.dtype, device=f.device)
    eigenvalues[:, :len(singular[0])] = singular.square()
    p = fl.transpose(-1, -2) @ u
    return dict(query_mean=q @ mean, query_lower=q @ lower, target_whitened=z,
        eigenvalues=eigenvalues, f_lower_eigen=p,
        residual_projection=(u.transpose(-1, -2) @ (y - f @ mean).unsqueeze(-1)).squeeze(-1),
        target_eigen=p.transpose(-1, -2) @ z)


def cached_coefficients(cache, covariance_scale, variances, gamma):
    """Whitened posterior coefficients/evidence for proper scale+rank2 priors.

    Sigma = tau L(I+gamma ZZ')L'. No eigenvalue/error clipping is used.
    Returns coefficient offsets[B,G,Q] and exact log evidence[B,G].
    """
    eigen, p, residual, re = (cache[k] for k in DYNAMIC_KEYS)
    z = cache['target_whitened']
    tau = torch.as_tensor(covariance_scale, dtype=eigen.dtype, device=eigen.device)
    gamma = torch.as_tensor(gamma, dtype=eigen.dtype, device=eigen.device)
    variances = torch.as_tensor(variances, dtype=eigen.dtype, device=eigen.device)
    if tau.numel() != 1 or gamma.numel() != 1 or bool(tau.detach() <= 0) or bool(gamma.detach() < 0):
        raise ValueError('Require positive global covariance scale and nonnegative global gamma')
    if variances.ndim != 1 or not bool(torch.isfinite(variances).all()) or bool((variances.detach() <= 0).any()):
        raise ValueError('Require positive finite noise variances')
    denominator = tau * eigen[:, None, :] + variances[None, :, None]
    r_solved = residual[:, None, :] / denominator
    re_solved = re[:, None, :, :] / denominator[..., None]
    b = torch.einsum('bmk,bgml->bgkl', re, re_solved)
    a = torch.einsum('bmk,bgm->bgk', re, r_solved)
    rank_scale = tau * gamma
    small = torch.eye(2, dtype=eigen.dtype, device=eigen.device) + rank_scale * b
    small_chol = torch.linalg.cholesky(small)
    update = rank_scale * torch.cholesky_solve(a.unsqueeze(-1), small_chol).squeeze(-1)
    offset0 = tau * torch.einsum('bqm,bgm->bgq', p, r_solved)
    gain = z[:, None] - tau * torch.einsum('bqm,bgmk->bgqk', p, re_solved)
    offset = offset0 + torch.einsum('bgqk,bgk->bgq', gain, update)
    quadratic = (residual[:, None, :] * r_solved).sum(-1) - (a * update).sum(-1)
    logdet = denominator.log().sum(-1) + 2 * torch.log(torch.diagonal(small_chol, dim1=-2, dim2=-1)).sum(-1)
    evidence = -.5 * (logdet + quadratic + residual.shape[-1] * math.log(2 * math.pi))
    return offset, evidence
