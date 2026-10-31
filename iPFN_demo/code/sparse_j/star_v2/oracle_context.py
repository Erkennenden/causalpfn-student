"""Oracle-context only; explicit privileged inputs, analytic marginal intervals.

No simulator objects, true beta, truth, full Y, Pair inference or model loading.
This result is Oracle-specific; ordinary execute_contract is not bypassed.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .config import CONFIG
from .oracle_design import coefficient_prior, factual_design, query_design
from .oracle_numerics import (gaussian_linear_posterior, freeze, immutable,
                              array64, noise_scale)

EFFECT_ORDER = ("Direct","Spillover","Total")
Z_975 = 1.959963984540054
POINT_IDENTITY_ATOL = 1e-12  # Fixed before first deterministic test.


@dataclass(frozen=True, slots=True)
class OracleInterval:
    effect: str
    status: str
    lower: float | None = None
    upper: float | None = None
    variance: float | None = None
    failure_reason: str | None = None
    level: float = .95
    source: str = "model_posterior"
    statistical_unit: str = "context"


@dataclass(frozen=True, slots=True)
class OracleContextResult:
    posterior_mean: np.ndarray | None
    posterior_covariance: np.ndarray | None
    mu: np.ndarray | None
    ite: np.ndarray | None
    ate: np.ndarray | None
    intervals: tuple
    point_status: str
    posterior_covariance_status: str
    failure_reasons: Mapping
    covariance_diagnostics: Mapping
    noise_sd: float
    context_identity: Mapping
    method_id: str = "oracle-context"
    privileged: bool = True
    reference_only: bool = True


def _identity(value):
    if value is None:
        return immutable({})
    allowed = {"context_id","pair_id","request_id","target_id","protocol_hash"}
    if (not isinstance(value,Mapping) or not set(value).issubset(allowed)
            or "context_id" not in value
            or any(type(v) is not str or not v.strip() for v in value.values())):
        raise ValueError("Identity requires context_id and only explicit nonempty identity strings")
    if "protocol_hash" in value and value["protocol_hash"] != CONFIG.protocol_sha256:
        raise ValueError("Identity protocol_hash mismatch")
    return immutable(value)


def _effect_interval(effect, contrast, mean, covariance):
    variance = float(contrast@covariance@contrast)
    center = float(contrast@mean)
    if not np.isfinite(variance) or variance < 0:
        raise FloatingPointError("Invalid posterior variance for "+effect+": "+str(variance))
    if not np.isfinite(center):
        raise FloatingPointError("Nonfinite posterior interval center for "+effect)
    radius = Z_975*np.sqrt(variance)
    lower,upper = float(center-radius),float(center+radius)
    if not np.isfinite([lower,upper]).all():
        raise FloatingPointError("Nonfinite posterior endpoints for "+effect)
    return OracleInterval(effect,"OK",lower,upper,variance)


def gaussian_ate_intervals(*, contrasts, posterior_mean, posterior_covariance):
    """Three independent marginal intervals; negative variance is never clipped."""
    c = array64(contrasts,(3,25),"ATE contrasts")
    mean = array64(posterior_mean,(25,),"posterior_mean")
    covariance = array64(posterior_covariance,(25,25),"posterior_covariance")
    records = []
    for k,effect in enumerate(EFFECT_ORDER):
        try:
            with np.errstate(over="raise",invalid="raise"):
                record = _effect_interval(effect,c[k],mean,covariance)
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            record = OracleInterval(effect,"FAILED",
                failure_reason=type(exc).__name__+": "+str(exc))
        records.append(record)
    return tuple(records)


def estimate_oracle_context(*, true_h, noise_sd, x, adjacency, treatment,
                            observed_indices, outcomes_observed, context_identity=None):
    """Explicit five public arrays plus allowed h/noise; fixed prior internally.

    Small N is permitted for artificial references. This does not alter the
    production N=200/M=50 TaskSpec constraints or introduce Identification.
    """
    identity = _identity(context_identity)
    sigma = noise_scale(noise_sd)
    factual = factual_design(true_h=true_h,adjacency=adjacency,treatment=treatment)
    n = len(factual)
    if (type(x) is not np.ndarray or x.shape != (n,20)
            or x.dtype not in (np.dtype("float32"),np.dtype("float64"))
            or not np.isfinite(x).all()):
        raise ValueError("Require finite x[N,20] float32/float64")
    if (type(observed_indices) is not np.ndarray
            or observed_indices.dtype != np.dtype("int64")
            or observed_indices.ndim != 1 or not 1 <= len(observed_indices) <= n
            or np.any(observed_indices < 0) or np.any(observed_indices >= n)
            or len(np.unique(observed_indices)) != len(observed_indices)):
        raise ValueError("Require unique ordered int64 observed_indices within graph")
    y = array64(outcomes_observed,(len(observed_indices),),"outcomes_observed")
    indices = observed_indices.copy()
    mean,covariance = coefficient_prior()
    posterior = gaussian_linear_posterior(
        design=np.array(factual[indices],copy=True),outcomes=y,
        prior_mean=mean,prior_covariance=covariance,noise_sd=sigma)
    failures = {}
    if posterior.failure_reason:
        failures["posterior"] = posterior.failure_reason
    mu = ite = ate = contrasts = None
    point_status = "FAILED"
    if posterior.mean_status == "OK":
        try:
            query = query_design(true_h=true_h,adjacency=adjacency)
            with np.errstate(over="raise",invalid="raise"):
                raw_mu = query@posterior.mean
                raw_ite = np.column_stack((
                    raw_mu[:,1,0]-raw_mu[:,0,0],
                    raw_mu[:,1,1]-raw_mu[:,1,0],
                    raw_mu[:,1,1]-raw_mu[:,0,0]))
                raw_ate = raw_ite.mean(axis=0)
                if not np.allclose(raw_ite[:,2],raw_ite[:,0]+raw_ite[:,1],
                                   atol=POINT_IDENTITY_ATOL,rtol=0):
                    raise FloatingPointError("Total != Direct + Spillover")
                contrasts = np.stack((
                    query[:,1,0]-query[:,0,0],query[:,1,1]-query[:,1,0],
                    query[:,1,1]-query[:,0,0]),axis=1).mean(axis=0)
            mu,ite,ate = freeze(raw_mu),freeze(raw_ite),freeze(raw_ate)
            point_status = "OK"
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures["point"] = type(exc).__name__+": "+str(exc)
    else:
        failures["point"] = "Posterior mean unavailable: "+str(posterior.failure_reason)
    if point_status == "OK" and posterior.covariance_status == "OK":
        try:
            intervals = gaussian_ate_intervals(
                contrasts=contrasts,posterior_mean=posterior.mean,
                posterior_covariance=posterior.covariance)
        except (ValueError,TypeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            reason = type(exc).__name__+": "+str(exc)
            intervals = tuple(OracleInterval(effect,"FAILED",failure_reason=reason)
                              for effect in EFFECT_ORDER)
    else:
        reason = ("Point prediction unavailable" if point_status != "OK" else
                  "Posterior covariance unavailable: "+str(posterior.failure_reason))
        intervals = tuple(OracleInterval(effect,"FAILED",failure_reason=reason)
                          for effect in EFFECT_ORDER)
    interval_failures = {r.effect:r.failure_reason for r in intervals if r.status != "OK"}
    if interval_failures:
        failures["intervals"] = interval_failures
    return OracleContextResult(
        posterior.mean,posterior.covariance,mu,ite,ate,intervals,
        point_status,posterior.covariance_status,immutable(failures),
        posterior.diagnostics,sigma,identity)
