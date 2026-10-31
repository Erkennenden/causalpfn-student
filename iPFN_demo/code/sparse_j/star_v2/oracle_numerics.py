"""Approved 25-D Oracle Gaussian numerics; no ridge, clipping or inverse."""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

P = 25
EPS64 = np.finfo(np.float64).eps


def freeze(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite numerical result")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: immutable(item) for key,item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(immutable(item) for item in value)
    return value


def array64(value, shape, name):
    if type(value) is not np.ndarray or value.dtype != np.dtype("float64"):
        raise TypeError(name + " must be a plain float64 ndarray")
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + " has illegal shape or nonfinite data")
    return np.array(value, copy=True)


def noise_scale(noise_sd):
    if (type(noise_sd) not in (int, float) or isinstance(noise_sd, bool)
            or not np.isfinite(noise_sd) or noise_sd <= 0):
        raise ValueError("noise_sd must be finite and strictly positive")
    return float(noise_sd)


@dataclass(frozen=True, slots=True)
class CovarianceCheck:
    status: str
    covariance: np.ndarray | None
    diagnostics: Mapping
    failure_reason: str | None = None


def validate_covariance(covariance):
    """Check raw asymmetry BEFORE symmetrizing. Negative eigenvalues stay intact."""
    diag = dict(status="FAILED", symmetry_residual=None, tau_sym=None,
                minimum_eigenvalue=None, tau_psd=None,
                symmetry_status="NOT_CHECKED", psd_status="NOT_CHECKED")
    def failure(reason):
        return CovarianceCheck("FAILED", None, immutable(diag), reason)
    try:
        raw = array64(covariance, (P,P), "covariance")
        with np.errstate(over="ignore", invalid="ignore"):
            norm_inf = float(np.linalg.norm(raw, ord=np.inf))
            norm_2 = float(np.linalg.norm(raw, ord=2))
            residual = float(np.linalg.norm(raw-raw.T, ord=np.inf))
        if not np.isfinite([norm_inf,norm_2,residual]).all():
            return failure("Nonfinite covariance norm/residual")
        tau_sym = 10*P*EPS64*max(1.,norm_inf)
        tau_psd = 10*P*EPS64*max(1.,norm_2)
        diag.update(symmetry_residual=residual,tau_sym=tau_sym,tau_psd=tau_psd)
        if residual > tau_sym:
            diag["symmetry_status"] = "FAILED"
            return failure("Covariance asymmetry exceeds tau_sym")
        diag["symmetry_status"] = "OK"
        symmetric = (raw+raw.T)/2
        minimum = float(np.linalg.eigvalsh(symmetric)[0])
        diag["minimum_eigenvalue"] = minimum
        if not np.isfinite(minimum):
            return failure("Nonfinite covariance minimum eigenvalue")
        if minimum < -tau_psd:
            diag["psd_status"] = "FAILED"
            return failure("Covariance minimum eigenvalue is below -tau_psd")
        diag.update(psd_status="OK",status="OK")
        return CovarianceCheck("OK",freeze(symmetric),immutable(diag))
    except (TypeError,ValueError,np.linalg.LinAlgError) as exc:
        return failure(type(exc).__name__ + ": " + str(exc))


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    mean: np.ndarray | None
    covariance: np.ndarray | None
    mean_status: str
    covariance_status: str
    failure_reason: str | None
    diagnostics: Mapping


def gaussian_linear_posterior(*, design, outcomes, prior_mean, prior_covariance, noise_sd):
    """25-D prior whitening and Cholesky solves, no explicit matrix inverse."""
    if type(design) is not np.ndarray or design.ndim != 2 or design.shape[1] != P or not len(design):
        raise ValueError("Require nonempty factual design[M,25]")
    f = array64(design,design.shape,"design")
    y = array64(outcomes,(len(f),),"outcomes")
    mean = array64(prior_mean,(P,),"prior_mean")
    sigma = noise_scale(noise_sd)
    prior = validate_covariance(prior_covariance)
    diagnostics = {"prior_covariance":dict(prior.diagnostics)}
    posterior_mean = None
    def failure(reason):
        return GaussianPosterior(posterior_mean,None,
            "OK" if posterior_mean is not None else "FAILED","FAILED",reason,
            immutable(diagnostics))
    if prior.status != "OK":
        return failure("Prior covariance: "+prior.failure_reason)
    try:
        lower = np.linalg.cholesky(prior.covariance)
        diagnostics["prior_cholesky"] = "OK"
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            h = f @ lower / sigma
            r = (y-f@mean)/sigma
            q = np.eye(P)+h.T@h
        qcheck = validate_covariance(q)
        diagnostics["Q"] = dict(qcheck.diagnostics)
        if qcheck.status != "OK":
            return failure("Q: "+qcheck.failure_reason)
        factor = np.linalg.cholesky(qcheck.covariance)
        diagnostics["Q_cholesky"] = "OK"
        rhs = h.T@r
        mean_z = np.linalg.solve(factor.T,np.linalg.solve(factor,rhs))
        posterior_mean = freeze(mean+lower@mean_z)
        covariance_z = np.linalg.solve(factor.T,np.linalg.solve(factor,np.eye(P)))
        raw_covariance = lower@covariance_z@lower.T
        post = validate_covariance(raw_covariance)
        diagnostics["posterior_covariance"] = dict(post.diagnostics)
        if post.status != "OK":
            return failure("Posterior covariance: "+post.failure_reason)
        return GaussianPosterior(posterior_mean,post.covariance,"OK","OK",None,
                                 immutable(diagnostics))
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
        return failure(type(exc).__name__+": "+str(exc))
