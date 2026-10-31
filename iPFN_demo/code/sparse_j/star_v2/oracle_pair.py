"""Star-specific shared-noise Oracle-pair; deterministic privileged reference."""
from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np

from .config import CONFIG
from .oracle_design import coefficient_prior, factual_design, query_design
from .oracle_numerics import (P, EPS64, freeze, immutable, array64, noise_scale,
                              validate_covariance)
from .oracle_context import (EFFECT_ORDER, POINT_IDENTITY_ATOL, OracleInterval,
                             gaussian_ate_intervals)


@dataclass(frozen=True, slots=True)
class ConstraintSolution:
    status: str
    particular: np.ndarray | None
    nullspace: np.ndarray | None
    singular_values: np.ndarray | None
    diagnostics: Mapping
    failure_reason: str | None = None


def solve_constraints(a, b):
    """Full SVD, frozen numerical rank and frozen orthogonal projection gate."""
    if type(a) is not np.ndarray or a.ndim != 2 or a.shape[1] != P or not len(a):
        raise ValueError("Require constraint matrix[M,25]")
    a = array64(a,a.shape,"constraint matrix")
    b = array64(b,(len(a),),"constraint rhs")
    diag = {"shape":tuple(a.shape),"rank":None,"tol_rank":None,
            "nullspace_dimension":None,"projection_residual":None,
            "constraint_tolerance":1e-10*max(1.,float(np.linalg.norm(b)))}
    singular = None
    def fail(reason):
        return ConstraintSolution("FAILED",None,None,singular,immutable(diag),reason)
    try:
        u,s,vt = np.linalg.svd(a,full_matrices=True)
        singular = freeze(s)
        if (not np.isfinite(u).all() or not np.isfinite(vt).all()
                or np.any(s < 0) or np.any(np.diff(s) > 0)):
            return fail("Invalid SVD factors")
        tol_rank = max(a.shape)*EPS64*float(s[0])
        rank = int(np.count_nonzero(s > tol_rank))
        ur,vr,v0 = u[:,:rank],vt[:rank].T,vt[rank:].T
        residual = float(np.linalg.norm(b-ur@(ur.T@b)))
        diag.update(rank=rank,tol_rank=tol_rank,nullspace_dimension=P-rank,
                    projection_residual=residual)
        ortho = float(np.linalg.norm(vt@vt.T-np.eye(P),ord=np.inf))
        null_residual = float(np.linalg.norm(a@v0))
        ortho_tol = 100*max(a.shape)*EPS64
        null_tol = 100*max(a.shape)*EPS64*max(1.,float(s[0]))
        diag.update(orthogonality_residual=ortho,orthogonality_tolerance=ortho_tol,
                    nullspace_residual=null_residual,nullspace_tolerance=null_tol)
        if ortho > ortho_tol or null_residual > null_tol:
            return fail("Abnormal SVD orthogonality or nullspace residual")
        if residual > diag["constraint_tolerance"]:
            return fail("Inconsistent exact constraints: projection residual exceeds frozen tolerance")
        zp = vr@((ur.T@b)/s[:rank]) if rank else np.zeros(P)
        particular_residual = float(np.linalg.norm(a@zp-b))
        diag["particular_residual"] = particular_residual
        if particular_residual > diag["constraint_tolerance"]:
            return fail("Particular solution violates frozen constraint tolerance")
        return ConstraintSolution("OK",freeze(zp),freeze(v0),singular,immutable(diag))
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
        return fail(type(exc).__name__+": "+str(exc))


@dataclass(frozen=True, slots=True)
class PairPosterior:
    status: str
    mean: np.ndarray | None
    covariance: np.ndarray | None
    covariance_status: str
    constraint: ConstraintSolution | None
    diagnostics: Mapping
    failure_reason: str | None = None


def shared_linear_posterior(*, design0, design1, outcomes0, outcomes1, noise_sd):
    """Numerical core accepts explicit rows only; always uses the fixed prior."""
    if type(design0) is not np.ndarray or design0.ndim != 2 or design0.shape[1] != P or not len(design0):
        raise ValueError("Require factual design[M,25]")
    f0 = array64(design0,design0.shape,"design0")
    f1 = array64(design1,f0.shape,"design1")
    y0 = array64(outcomes0,(len(f0),),"outcomes0")
    y1 = array64(outcomes1,(len(f0),),"outcomes1")
    sigma = noise_scale(noise_sd)
    mean,covariance = coefficient_prior()
    prior = validate_covariance(covariance)
    diag = {"prior_covariance":dict(prior.diagnostics),"average_noise_variance":sigma**2}
    constraint = None
    posterior_mean = None
    def fail(reason):
        return PairPosterior("FAILED",posterior_mean,None,"FAILED",constraint,
                             immutable(diag),reason)
    if prior.status != "OK":
        return fail("Prior covariance: "+prior.failure_reason)
    try:
        lower = np.linalg.cholesky(prior.covariance)
        diag["prior_cholesky"] = "OK"
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            df,dy = f1-f0,y1-y0
            fbar,ybar = (f0+f1)/2,(y0+y1)/2
            constraint = solve_constraints(df@lower,dy-df@mean)
            if constraint.status != "OK":
                return fail(constraint.failure_reason)
            zp,v0 = constraint.particular,constraint.nullspace
            base = mean+lower@zp
            residual = (ybar-fbar@base)/sigma
            diag["average_residual_norm"] = float(np.linalg.norm(residual))
            d = v0.shape[1]
            if d == 0:
                posterior_mean = freeze(base)
                raw_covariance = np.zeros((P,P))
                diag["nullspace_update"] = "ZERO_DIMENSION"
            else:
                lv0 = lower@v0
                h = fbar@lv0/sigma
                q = np.eye(d)+h.T@h
                if not np.isfinite(q).all():
                    return fail("Nonfinite nullspace precision")
                q_residual = float(np.linalg.norm(q-q.T,ord=np.inf))
                q_tol = 10*P*EPS64*max(1.,float(np.linalg.norm(q,ord=np.inf)))
                diag.update(Q_shape=tuple(q.shape),Q_symmetry_residual=q_residual,
                            Q_tau_sym=q_tol)
                if q_residual > q_tol:
                    return fail("Nullspace precision asymmetry exceeds tolerance")
                factor = np.linalg.cholesky((q+q.T)/2)
                mu_u = np.linalg.solve(factor.T,np.linalg.solve(factor,h.T@residual))
                cov_u = np.linalg.solve(factor.T,np.linalg.solve(factor,np.eye(d)))
                posterior_mean = freeze(base+lv0@mu_u)
                raw_covariance = lv0@cov_u@lv0.T
                diag["nullspace_update"] = "CHOLESKY_SOLVES"
        checked = validate_covariance(raw_covariance)
        diag["posterior_covariance"] = dict(checked.diagnostics)
        if checked.status != "OK":
            return fail("Posterior covariance: "+checked.failure_reason)
        return PairPosterior("OK",posterior_mean,checked.covariance,"OK",
                             constraint,immutable(diag))
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
        return fail(type(exc).__name__+": "+str(exc))


@dataclass(frozen=True, slots=True)
class OraclePairResult:
    status: str
    pair_identity: Mapping
    posterior_status: str
    posterior_mean: np.ndarray | None
    posterior_covariance: np.ndarray | None
    covariance_status: str
    constraint: ConstraintSolution | None
    point_status: str
    mu: np.ndarray | None
    ite: np.ndarray | None
    ate: np.ndarray | None
    intervals: tuple
    diagnostics: Mapping
    failure_reasons: Mapping
    noise_sd: float
    method_id: str = "oracle-pair"
    privileged: bool = True
    reference_only: bool = True


def _pair_identity(value):
    if value is None:
        return immutable({})
    allowed = {"pair_id","context_ids","request_id","target_id","protocol_hash"}
    if not isinstance(value,Mapping) or not set(value).issubset(allowed) or "pair_id" not in value:
        raise ValueError("Explicit pair identity required")
    for key,item in value.items():
        if key == "context_ids":
            if (type(item) is not tuple or len(item) != 2 or len(set(item)) != 2
                    or any(type(v) is not str or not v.strip() for v in item)):
                raise ValueError("Two ordered distinct context IDs required")
        elif type(item) is not str or not item.strip():
            raise ValueError("Nonempty identity strings required")
    if "protocol_hash" in value and value["protocol_hash"] != CONFIG.protocol_sha256:
        raise ValueError("Pair protocol_hash mismatch")
    return immutable(value)


def _indices(value,n):
    if (type(value) is not np.ndarray or value.dtype != np.dtype("int64")
            or value.ndim != 1 or not 1 <= len(value) <= n
            or np.any(value < 0) or np.any(value >= n)
            or len(np.unique(value)) != len(value)):
        raise ValueError("Unique ordered int64 observed indices required")
    return value.copy()


def _failed_intervals(reason):
    return tuple(OracleInterval(effect,"FAILED",failure_reason=reason,
                 source="joint_model_posterior",statistical_unit="pair")
                 for effect in EFFECT_ORDER)


def estimate_oracle_pair(*, true_h, noise_sd, adjacency, treatment0, treatment1,
                         observed_indices0, observed_indices1,
                         outcomes_observed0, outcomes_observed1, pair_identity=None):
    """Fixed-center Star pair, no matching/sorting or access to private objects."""
    identity = _pair_identity(pair_identity)
    sigma = noise_scale(noise_sd)
    f0 = factual_design(true_h=true_h,adjacency=adjacency,treatment=treatment0)
    f1 = factual_design(true_h=true_h,adjacency=adjacency,treatment=treatment1)
    n = len(f0)
    expected_a = np.zeros((n,n));expected_a[0,1:] = expected_a[1:,0] = 1.
    if not np.array_equal(adjacency,expected_a):
        raise ValueError("Oracle-pair requires node-0 Star topology")
    if (treatment0[0] != 0 or treatment1[0] != 1
            or not np.array_equal(treatment0[1:],treatment1[1:])):
        raise ValueError("Only center treatment may change, ordered 0 then 1")
    indices0,indices1 = _indices(observed_indices0,n),_indices(observed_indices1,n)
    if not np.array_equal(indices0,indices1):
        raise ValueError("Observed indices must match exactly, including order")
    y0 = array64(outcomes_observed0,(len(indices0),),"outcomes_observed0")
    y1 = array64(outcomes_observed1,(len(indices1),),"outcomes_observed1")
    posterior = shared_linear_posterior(
        design0=np.array(f0[indices0],copy=True),design1=np.array(f1[indices1],copy=True),
        outcomes0=y0,outcomes1=y1,noise_sd=sigma)
    failures = {}
    if posterior.failure_reason:
        failures["posterior"] = posterior.failure_reason
    mu = ite = ate = contrasts = None
    point_status = "FAILED"
    if posterior.mean is not None:
        try:
            q = query_design(true_h=true_h,adjacency=adjacency)
            raw_mu = q@posterior.mean
            raw_ite = np.column_stack((raw_mu[:,1,0]-raw_mu[:,0,0],
                raw_mu[:,1,1]-raw_mu[:,1,0],raw_mu[:,1,1]-raw_mu[:,0,0]))
            raw_ate = raw_ite.mean(axis=0)
            if not np.allclose(raw_ite[:,2],raw_ite[:,0]+raw_ite[:,1],atol=POINT_IDENTITY_ATOL,rtol=0):
                raise FloatingPointError("Total != Direct + Spillover")
            contrasts = np.stack((q[:,1,0]-q[:,0,0],q[:,1,1]-q[:,1,0],
                                  q[:,1,1]-q[:,0,0]),axis=1).mean(0)
            mu,ite,ate = freeze(raw_mu),freeze(raw_ite),freeze(raw_ate)
            point_status = "OK"
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures["point"] = type(exc).__name__+": "+str(exc)
    if point_status == "OK" and posterior.covariance_status == "OK":
        try:
            intervals = tuple(replace(r,source="joint_model_posterior",statistical_unit="pair")
                              for r in gaussian_ate_intervals(contrasts=contrasts,
                                  posterior_mean=posterior.mean,
                                  posterior_covariance=posterior.covariance))
        except (ValueError,TypeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            intervals = _failed_intervals(type(exc).__name__+": "+str(exc))
    else:
        intervals = _failed_intervals(posterior.failure_reason or "Point prediction unavailable")
    if any(r.status != "OK" for r in intervals):
        failures["intervals"] = {r.effect:r.failure_reason for r in intervals if r.status != "OK"}
    status = ("OK" if posterior.status == "OK" and point_status == "OK"
              and all(r.status == "OK" for r in intervals) else "FAILED")
    return OraclePairResult(status,identity,posterior.status,posterior.mean,posterior.covariance,
        posterior.covariance_status,posterior.constraint,point_status,mu,ite,ate,intervals,
        posterior.diagnostics,immutable(failures),sigma)
