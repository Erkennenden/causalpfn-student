"""S3.3 fixed-transform joint-pair regressions; ATE point outputs only.

Rank rcond=1e-6 and relative contrast residual <=10*rcond follow the fixed
phase6_sparse_classical.py implementation. No regularization, refitting of
projection, legacy propensity, model imports, intervals or standard errors.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np

from .contracts import PairTaskSpec, ContractError
from .estimation import execute_contract, EFFECT_ORDER
from .baselines_state_ht import StarPairDesign, _public_records, ARM_ORDER, DEPENDENCIES

TRANSFORM_SHA256 = "b261372fea1cb241e4acbd9734eab5b9506f6478e90d90c0f5f3e633b7a3a737"
TRANSFORM_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_transform.npz"
RCOND = 1e-6
ROW_SPACE_RTOL = 10 * RCOND


def _reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


def _freeze(a):
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _array(a, name, shape=None):
    if (type(a) is not np.ndarray or a.dtype != np.dtype("float64")
            or (shape is not None and a.shape != shape) or not np.isfinite(a).all()):
        _reject("ARRAY", "Require finite plain float64 array", field=name, shape=shape)


@dataclass(frozen=True, slots=True)
class FrozenTransform:
    x_mean: np.ndarray
    x_scale: np.ndarray
    projection: np.ndarray
    sha256: str

    def __post_init__(self):
        for name in ("x_mean", "x_scale", "projection"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def load_frozen_transform():
    """Verify the bytes actually loaded, with pickle disabled; never caches."""
    blob = TRANSFORM_PATH.read_bytes()
    identity = hashlib.sha256(blob).hexdigest()
    if identity != TRANSFORM_SHA256:
        _reject("ASSET_HASH", "Frozen feature transform hash mismatch",
                expected=TRANSFORM_SHA256, actual=identity)
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        if set(data.files) != {"x_mean", "x_scale", "projection"}:
            _reject("TRANSFORM", "Unexpected transform keys")
        mean, scale, projection = (data[k] for k in ("x_mean", "x_scale", "projection"))
    _array(mean, "x_mean", (20,))
    _array(scale, "x_scale", (20,))
    _array(projection, "projection", (20,2))
    if np.any(scale <= 0):
        _reject("TRANSFORM", "Require positive frozen scales")
    return FrozenTransform(mean, scale, projection, identity)


@dataclass(frozen=True, slots=True)
class FeatureBasis:
    raw: np.ndarray
    centered: np.ndarray
    rms: np.ndarray
    active: np.ndarray
    singular_values: np.ndarray
    z: np.ndarray
    asset_hash: str

    def __post_init__(self):
        for name in ("raw", "centered", "rms", "active", "singular_values", "z"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def rank(self):
        return self.z.shape[1]


def full_population_features(x):
    if (type(x) is not np.ndarray or x.shape != (200,20)
            or x.dtype != np.dtype("float32") or not np.isfinite(x).all()):
        _reject("X", "Require canonical float32 X[200,20]")
    transform = load_frozen_transform()
    raw = ((x.astype(np.float64)-transform.x_mean)/transform.x_scale) @ transform.projection
    if not np.isfinite(raw).all():
        _reject("NUMERIC", "Nonfinite transformed features")
    centered = raw-raw.mean(0)
    rms = np.sqrt(np.mean(centered*centered,axis=0))
    active = rms > 0
    standardized = centered[:,active]/rms[active]
    if standardized.shape[1]:
        u, singular, _ = np.linalg.svd(standardized,full_matrices=False)
        z = u[:,singular > RCOND*singular[0]]*np.sqrt(len(x))
    else:
        singular, z = np.empty(0), np.empty((len(x),0))
    return FeatureBasis(raw, centered, rms, active, singular, z, transform.sha256)


@dataclass(frozen=True, slots=True)
class RegressionDesign:
    method_id: str
    matrix: np.ndarray
    outcomes: np.ndarray
    contrasts: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    features: FeatureBasis

    def __post_init__(self):
        for name in ("matrix", "outcomes", "contrasts", "weights", "counts"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def _rows(z, codes, interacted):
    one = np.eye(4,dtype=np.float64)[codes]
    slopes = (one[:,:,None]*z[:,None,:]).reshape(len(z),4*z.shape[1]) if interacted else z
    return np.column_stack([one,slopes])


def build_regression_design(inputs, method_id):
    if method_id not in ("f-z2", "lin-z2"):
        _reject("METHOD", "Only F-z2/Lin-z2 are implemented")
    design = StarPairDesign(200,50)
    codes, observed, indices, _ = _public_records(inputs,design)
    features = full_population_features(inputs[0]["x"])
    z = features.z
    interacted = method_id == "lin-z2"
    matrix = np.vstack([_rows(z[indices],arm,interacted) for arm in codes])
    # Full-N average query rows, not average over the observed subset.
    query_means = np.stack([
        _rows(z,np.full(design.n,state,dtype=np.int64),interacted).mean(0)
        for state in range(4)])
    contrasts = np.stack([query_means[pos]-query_means[neg] for pos,neg in DEPENDENCIES])
    return RegressionDesign(method_id,matrix,observed.ravel(),contrasts,
                            np.full(2*design.m,design.weight),
                            np.bincount(codes.ravel(),minlength=4),features)


@dataclass(frozen=True, slots=True)
class ContrastFit:
    ate: np.ndarray
    output_status: tuple
    singular_values: np.ndarray
    rank: int
    columns: int
    condition_number: float | None
    effective_condition_number: float | None
    residuals: np.ndarray
    tolerances: np.ndarray

    def __post_init__(self):
        for name in ("ate", "singular_values", "residuals", "tolerances"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def fit_estimable_contrasts(matrix, outcomes, contrasts, weights):
    """One truncated-SVD row space defines rank, fit and every effect's gate.

    A minimum-norm coefficient representation is internal only. An effect is
    emitted solely when its full-N query contrast passes the row-space check.
    Constant WLS weights are required; no uncertainty is computed from rows.
    """
    _array(matrix,"design")
    if matrix.ndim != 2 or not min(matrix.shape):
        _reject("DESIGN", "Nonempty two-dimensional design required")
    rows, columns = matrix.shape
    _array(outcomes,"outcomes",(rows,))
    _array(contrasts,"contrasts",(3,columns))
    _array(weights,"weights",(rows,))
    if np.any(weights <= 0) or not np.all(weights == weights[0]):
        _reject("WEIGHTS", "This frozen pair design requires constant positive weights")
    root = np.sqrt(weights)
    weighted = matrix*root[:,None]
    response = outcomes*root
    if not np.isfinite(weighted).all() or not np.isfinite(response).all():
        _reject("NUMERIC", "Nonfinite weighted inputs")
    u, singular, vt = np.linalg.svd(weighted,full_matrices=False)
    keep = singular > RCOND*singular[0] if singular[0] > 0 else np.zeros_like(singular,dtype=bool)
    rank = int(keep.sum())
    basis = vt[keep]
    coefficient = basis.T @ ((u[:,keep].T@response)/singular[keep])
    represented = (contrasts@basis.T)@basis
    residuals = np.linalg.norm(contrasts-represented,axis=1)
    tolerances = ROW_SPACE_RTOL*np.linalg.norm(contrasts,axis=1)
    supported = residuals <= tolerances
    ate = np.full(3,np.nan)
    ate[supported] = contrasts[supported]@coefficient
    if not np.isfinite(ate[supported]).all():
        _reject("NUMERIC", "Nonfinite estimable contrast")
    effective = float(singular[0]/singular[keep][-1]) if rank else None
    condition = effective if rank == columns else None
    return ContrastFit(ate,tuple("OK" if v else "NOT_ESTIMABLE" for v in supported),
                       singular,rank,columns,condition,effective,residuals,tolerances)


def _compute(inputs, method_id):
    design = build_regression_design(inputs,method_id)
    fit = fit_estimable_contrasts(
        design.matrix,design.outcomes,design.contrasts,design.weights)
    failed = any(v != "OK" for v in fit.output_status)
    estimability = {
        effect: {"status": fit.output_status[k],
                 "reason": "contrast_in_row_space" if fit.output_status[k] == "OK"
                           else "contrast_not_in_row_space",
                 "row_space_residual": float(fit.residuals[k]),
                 "row_space_tolerance": float(fit.tolerances[k]),
                 "query_scope": "all_200_nodes"}
        for k,effect in enumerate(EFFECT_ORDER)}
    return {
        "output_status": {"ate": fit.output_status}, "ate": fit.ate,
        "arm_results": (None,None),
        "asset_hashes": {"feature_transform.npz": design.features.asset_hash},
        "estimability": estimability,
        "diagnostics": {
            "state_order": ARM_ORDER, "effect_order": EFFECT_ORDER,
            "state_counts": design.counts.tolist(),
            "design_rank": fit.rank, "design_columns": fit.columns,
            "singular_values": fit.singular_values.tolist(),
            "condition_number": fit.condition_number,
            "condition_number_status": "FINITE" if fit.condition_number is not None else "RANK_DEFICIENT",
            "effective_condition_number": fit.effective_condition_number,
            "rcond": RCOND, "row_space_relative_tolerance": ROW_SPACE_RTOL,
            "row_space_rule_source": "fixed phase6_sparse_classical.py",
            "covariate_rank": design.features.rank,
            "covariate_singular_values": design.features.singular_values.tolist(),
            "full_population_centering": True, "full_population_rms_scaling": True,
            "query_units": 200, "n_unique_observed_nodes": 50, "n_factual_records": 100,
            "observation_clusters": 50, "record_cluster_key": "node_id",
            "statistical_unit": "pair", "arms_independent": False,
            "r": .25, "q": .5, "weight": float(design.weights[0]),
            "standard_errors_reported": False},
        "failure_type": "CONTRAST_NOT_ESTIMABLE" if failed else None,
        "failure_message": "; ".join(
            effect+":contrast_not_in_row_space" for k,effect in enumerate(EFFECT_ORDER)
            if fit.output_status[k] != "OK") if failed else None,
        "failure_provenance": {"effects": {
            effect: estimability[effect] for k,effect in enumerate(EFFECT_ORDER)
            if fit.output_status[k] != "OK"}} if failed else {}}


def estimate_f_z2(request, pair_task):
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Require canonical N=200 PairTaskSpec")
    return execute_contract(request,pair_task,"f-z2",lambda inputs: _compute(inputs,"f-z2"))


def estimate_lin_z2(request, pair_task):
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Require canonical N=200 PairTaskSpec")
    return execute_contract(request,pair_task,"lin-z2",lambda inputs: _compute(inputs,"lin-z2"))
