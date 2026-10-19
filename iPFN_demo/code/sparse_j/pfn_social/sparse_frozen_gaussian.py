"""Frozen Gaussian predictors with M observed outcomes and N graph queries.

The deployment API has no full-outcome argument or token buffer. All feature,
exposure and arm-query rows use the full graph; only the likelihood rows and
outcome vector are restricted to the observed indices. No parameters are fit.
"""
import math
import numpy as np
import torch
from pfn_social.phase5_meta_bayes import design
from functools import lru_cache
from pfn_social.contracts import arm_mean_exposure


@lru_cache(maxsize=None)
def _exposure(degree,arm):return float(arm_mean_exposure(degree,.5,arm))


def graph_moments(adjacency):
    degrees=adjacency.sum(-1).long();result=torch.empty((*degrees.shape,2),dtype=adjacency.dtype,device=adjacency.device)
    for degree in degrees.unique().tolist():
        for arm in (0,1):result[...,arm][degrees==degree]=_exposure(int(degree),arm)
    return result


def ate_target_rows(query):
    """query[B,N,4,Q], arms (00,01,10,11) -> C[B,2,Q]."""
    if query.ndim != 4 or query.shape[-2] != 4:
        raise ValueError("Require four-arm query[B,N,4,Q]")
    means = query.double().mean(1)
    return torch.stack((means[:, 2] - means[:, 0],
                        means[:, 3] - means[:, 2]), dim=1)


def _geometry(query, lower, gamma):
    c = ate_target_rows(query)
    if lower.ndim != 2 or lower.shape[0] != lower.shape[1] or lower.shape[0] != c.shape[-1]:
        raise ValueError("Require a shared square Gaussian prior factor")
    gamma = torch.as_tensor(gamma, dtype=lower.dtype, device=lower.device)
    if gamma.numel() != 1:
        raise ValueError("gamma must be one global scalar, not per-context weights")
    gamma = gamma.reshape(())
    if not bool(torch.isfinite(gamma.detach())) or bool(gamma.detach() < 0):
        raise ValueError("Require a finite nonnegative gamma")
    cl = c @ lower
    moment = cl @ cl.transpose(-1, -2)
    # Rank-two targets are required. Do not silently select/drop directions.
    moment_chol = torch.linalg.cholesky(moment)
    sigma_c = lower @ cl.transpose(-1, -2)
    return c, moment_chol, sigma_c, gamma


def gaussian_sparse_component(factual, query, y_observed, prior, variances,
                              gamma=0., *, force_primal=False):
    """Proper Gaussian posterior/evidence: f[B,M,Q], q[B,N,4,Q], y[B,M].

Prior inflation uses the two mean query contrasts over ALL N query nodes.
The evidence normalizer is M, never N. Finite gamma retains a proper prior.
"""
    f, q, y = factual.double(), query.double(), y_observed.double()
    if f.ndim != 3 or q.ndim != 4 or q.shape[-2] != 4:
        raise ValueError('Require factual[B,M,Q] and query[B,N,4,Q]')
    if y.shape != f.shape[:2] or q.shape[0] != f.shape[0] or q.shape[-1] != f.shape[-1]:
        raise ValueError('Batch/coefficient dimensions disagree; M and N may differ')
    if f.shape[1] < 1 or q.shape[1] < 1:
        raise ValueError('At least one observed and query node is required')
    lower, mean = prior.lower().double(), prior.mean.double()
    variances = torch.as_tensor(variances, dtype=f.dtype, device=f.device)
    if variances.ndim != 1 or not bool(torch.isfinite(variances).all()) or bool((variances <= 0).any()):
        raise ValueError('Require finite positive noise variances')
    c, target_chol, sigma_c, gamma = _geometry(q, lower, gamma)
    residual = y - f @ mean
    m, dimension = f.shape[1:]
    if m < dimension and not force_primal:
        fl = f @ lower
        fc = f @ sigma_c
        solved_fc = torch.cholesky_solve(fc.transpose(1, 2), target_chol)
        kernel = fl @ fl.transpose(1, 2) + gamma * (fc @ solved_fc)
        eye = torch.eye(m, dtype=f.dtype, device=f.device)
        chol = torch.linalg.cholesky(kernel[:, None] + variances[None, :, None, None] * eye)
        solution = torch.cholesky_solve(residual[:, None, :, None].expand(-1, len(variances), -1, -1), chol).squeeze(-1)
        cross = torch.einsum('bnaq,bmq->bnam', q @ lower, fl)
        qc = torch.einsum('bnaq,bqk->bnak', q, sigma_c)
        cross = cross + gamma * torch.einsum('bnak,bkm->bnam', qc, solved_fc)
        predictions = (q @ mean)[:, None] + torch.einsum('bnam,bgm->bgna', cross, solution)
        quadratic = (residual[:, None] * solution).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    else:
        base_precision = torch.cholesky_inverse(lower)
        target_precision = c.transpose(1, 2) @ torch.cholesky_solve(c, target_chol)
        precision = base_precision[None] - gamma / (1 + gamma) * target_precision
        rhs = f.transpose(1, 2) @ residual.unsqueeze(-1)
        chol = torch.linalg.cholesky(f.transpose(1, 2)[:, None] @ f[:, None] +
            variances[None, :, None, None] * precision[:, None])
        correction = torch.cholesky_solve(rhs[:, None].expand(-1, len(variances), -1, -1), chol).squeeze(-1)
        predictions = torch.einsum('bnaq,bgq->bgna', q, mean[None, None] + correction)
        quadratic = (residual.square().sum(-1)[:, None] -
            (rhs.squeeze(-1)[:, None] * correction).sum(-1)) / variances[None]
        logdet_prior = 2 * torch.log(torch.diagonal(lower)).sum() + 2 * torch.log1p(gamma)
        logdet = (m - dimension) * variances.log()[None] + logdet_prior + \
            2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return predictions, -.5 * (logdet + quadratic + m * math.log(2 * math.pi))


def validate_factual_inputs(*, x, adjacency, treatment, observed_indices,
                            outcomes_observed, raw_p=20):
    """Validate only the five declared observed arrays; no mapping/token input."""
    x, a, t, y = (np.asarray(z, dtype=float) for z in (x, adjacency, treatment, outcomes_observed))
    indices = np.asarray(observed_indices)
    n = len(x)
    if x.ndim != 2 or x.shape[1] != raw_p or n < 2 or a.shape != (n, n) or t.shape != (n,):
        raise ValueError('Require X[N,p], full A[N,N], and full treatment[N]')
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) < 1:
        raise ValueError('Require a nonempty integer observed_indices[M]')
    if y.shape != (len(indices),) or len(indices) > n:
        raise ValueError('Only outcomes_observed[M] is accepted; no full-Y placeholder')
    if (indices < 0).any() or (indices >= n).any() or len(np.unique(indices)) != len(indices):
        raise ValueError('Observed indices must be unique and in range')
    if not all(np.isfinite(z).all() for z in (x, a, t, y)):
        raise ValueError('Nonfinite observed inputs')
    if not np.isin(a, [0., 1.]).all() or not np.array_equal(a, a.T) or np.any(np.diag(a) != 0):
        raise ValueError('Require a full undirected loopless binary graph')
    if not np.isin(t, [0., 1.]).all() or np.any(a.sum(1) <= 0):
        raise ValueError('Require binary treatments and positive graph degrees')
    return x, a, t, indices.astype(np.int64, copy=False), y


def _mixture(parent, x, a, t, e, indices, y, moments, gamma):
    degree = a.sum(-1, keepdim=True)
    h = parent.encoder(x, a)
    variances = parent.noise_variance_grid.double() * parent.log_grid_scale.clamp(-2, 2).exp()
    predictions, evidences = [], []
    for representation, prior in zip((h, x), parent.priors):
        hb = a @ representation / degree
        # Build factual features on the entire graph before restricting rows.
        factual_full = design(representation, hb, t, e, a @ (t * representation) / degree)
        queries = []
        for own, arm in ((0, 0), (0, 1), (1, 0), (1, 1)):
            em = moments[..., arm:arm + 1]
            queries.append(design(representation, hb, torch.full_like(t, own), em, em * hb))
        mean, evidence = gaussian_sparse_component(factual_full[:, indices], torch.stack(queries, -2), y, prior, variances, gamma)
        predictions.append(mean)
        evidences.append(evidence)
    predictions, evidences = torch.stack(predictions, 1), torch.stack(evidences, 1)
    log_prior = torch.log_softmax(parent.representation_logits.double(), -1)[None, :, None] + \
        torch.log_softmax(parent.noise_logits.double(), -1)[None, None, :]
    weights = torch.softmax((evidences + log_prior).flatten(1), -1).reshape(len(x), 2, 4)
    return (predictions * weights[..., None, None]).sum((1, 2)), weights


class SparseFrozenGaussian:
    """Read-only adapter. No training method and no full-outcome input exist."""
    def __init__(self, model, *, identity=None):
        self.model = model.eval()
        self.identity = dict(identity or {})
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _predict(self, model, x, a, t, e, indices, y, moments):
        mean, weights = _mixture(model, x, a, t, e, indices, y, moments, model.gamma())
        return mean, {'joint_weights': weights}

    @torch.no_grad()
    def predict_arm_means(self, *, x, adjacency, treatment, observed_indices,
                          outcomes_observed, return_diagnostics=False):
        x, a, t, indices, y = validate_factual_inputs(x=x, adjacency=adjacency, treatment=treatment,
            observed_indices=observed_indices, outcomes_observed=outcomes_observed)
        # Float32 feature/exposure values match the sealed parent public API.
        # There is deliberately no N-vector outcome or token reconstruction.
        e = a @ t / a.sum(1)
        xt = torch.tensor(x, dtype=torch.float32)[None]
        at = torch.tensor(a, dtype=torch.float32)[None]
        tt = torch.tensor(t, dtype=torch.float32)[None, :, None]
        et = torch.tensor(e, dtype=torch.float32)[None, :, None]
        yt = torch.tensor(y, dtype=torch.float32)[None]
        mean, diagnostics = self._predict(self.model, xt, at, tt, et, indices, yt, graph_moments(at))
        result = mean[0].cpu().numpy()
        if result.shape != (len(x), 4) or not np.isfinite(result).all():
            raise FloatingPointError('Invalid all-node arm means')
        return (result, diagnostics) if return_diagnostics else result
