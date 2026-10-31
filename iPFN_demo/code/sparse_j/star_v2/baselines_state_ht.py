"""S3.2: joint-pair State Mean and HT, using public factual inputs only.

Small designs enter the pure numeric cores, never the N=200 TaskSpec contract.
HT zero sums are estimates, not assertions of zero structural response.
"""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from .contracts import FACTUAL_KEYS, PairTaskSpec, ContractError
from .estimation import execute_contract, EFFECT_ORDER
from .support import REASONS

ARM_ORDER = ("00", "01", "10", "11")
DEPENDENCIES = ((2, 0), (3, 2), (3, 0))


def _reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


def _freeze(a):
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class StarPairDesign:
    n: int
    m: int
    leaf_assignment_probability: float = 0.5
    exposure_threshold: float = 0.5

    def __post_init__(self):
        if (type(self.n) is not int or self.n < 2 or self.n % 2
                or type(self.m) is not int or not 1 <= self.m <= self.n):
            _reject("DESIGN", "Require even N (odd center degree), 1 <= M <= N")
        if (type(self.leaf_assignment_probability) is not float
                or type(self.exposure_threshold) is not float
                or self.leaf_assignment_probability != 0.5 or self.exposure_threshold != 0.5):
            _reject("DESIGN", "Pair q uses iid fair leaves and strict majority >0.5")

    @property
    def r(self):
        return self.m / self.n

    @property
    def q(self):
        # Fair odd-size majority at center; fair own treatment for each leaf.
        # In a fixed-center 0/1 pair each included node-state occurs once.
        return self.leaf_assignment_probability

    @property
    def joint_inclusion_probability(self):
        return self.r * self.q

    @property
    def weight(self):
        return 1.0 / self.joint_inclusion_probability


def _array(a, shape, dtype, name):
    if (type(a) is not np.ndarray or a.shape != shape or a.dtype.str != dtype
            or not np.isfinite(a).all()):
        _reject("ARRAY", "Invalid plain array shape/dtype/value", field=name,
                required_shape=shape, required_dtype=dtype)


def _records(codes, outcomes, indices, design):
    if type(design) is not StarPairDesign:
        _reject("DESIGN", "Validated StarPairDesign required")
    _array(codes, (2, design.m), "<i8", "state_codes")
    _array(outcomes, (2, design.m), "<f8", "outcomes_observed")
    _array(indices, (design.m,), "<i8", "observed_indices")
    if (np.any(indices < 0) or np.any(indices >= design.n)
            or len(np.unique(indices)) != design.m):
        _reject("OBSERVATIONS", "Require shared ordered unique node sample")
    if not np.isin(codes, (0, 1, 2, 3)).all():
        _reject("STATE", "State code must be 2*t+s in 00,01,10,11 order")
    center = indices == 0
    if (np.any(codes[0, center] >= 2) or np.any(codes[1, center] < 2)
            or np.any(codes[0, center] % 2 != codes[1, center] % 2)
            or np.any(codes[0, ~center] % 2 != 0)
            or np.any(codes[1, ~center] % 2 != 1)
            or np.any(codes[0, ~center] // 2 != codes[1, ~center] // 2)
            or np.any(codes[0] == codes[1])):
        _reject("PAIR_STATES", "Ordered center arms/leaf sharing/state uniqueness violated")
    counts = np.bincount(codes.ravel(), minlength=4).astype(np.int64)
    sums = np.bincount(codes.ravel(), weights=outcomes.ravel(), minlength=4)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_sums = np.bincount(
            codes.ravel(), weights=(outcomes * design.weight).ravel(), minlength=4)
    if not np.isfinite(sums).all() or not np.isfinite(weighted_sums).all():
        _reject("NUMERIC", "Nonfinite state sum/weighted sum")
    hajek = np.full(4, np.nan)
    nonempty = counts > 0
    hajek[nonempty] = weighted_sums[nonempty] / (counts[nonempty] * design.weight)
    return counts, sums, weighted_sums, hajek


@dataclass(frozen=True, slots=True)
class StateEstimate:
    method_id: str
    counts: np.ndarray
    state_values: np.ndarray
    state_status: tuple
    state_reason: tuple
    empirical_state_support: np.ndarray
    hajek_diagnostic: np.ndarray
    ate: np.ndarray
    ate_status: tuple
    effect_reasons: tuple

    def __post_init__(self):
        for name in ("counts", "state_values", "empirical_state_support",
                     "hajek_diagnostic", "ate"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def _result(method, counts, means, hajek):
    finite = np.isfinite(means)
    if np.any(np.isinf(means)):
        _reject("NUMERIC", "Nonfinite state estimate")
    states = tuple("OK" if ok else "NOT_ESTIMABLE" for ok in finite)
    state_reasons = tuple(
        "observed_records" if count else
        "empty_sum_zero_contribution_not_zero_response" if method == "ht" else
        "empty_state_not_estimable" for count in counts)
    ate = np.full(3, np.nan)
    statuses, reasons = [], []
    for k, (positive, negative) in enumerate(DEPENDENCIES):
        missing = tuple(ARM_ORDER[z] for z in (positive, negative) if not finite[z])
        if missing:
            statuses.append("NOT_ESTIMABLE")
            reasons.append("empty_required_state:" + ",".join(missing))
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                value = means[positive] - means[negative]
            if not np.isfinite(value):
                _reject("NUMERIC", "Nonfinite effect contrast", effect=EFFECT_ORDER[k])
            ate[k] = value
            statuses.append("OK")
            reasons.append("valid_state_contrast")
    return StateEstimate(method, counts, means, states, state_reasons, counts > 0,
                         hajek, ate, tuple(statuses), tuple(reasons))


def state_mean_core(state_codes, outcomes_observed, observed_indices, design):
    """Pure joint-record core; no request, simulator or truth input."""
    counts, sums, _, hajek = _records(
        state_codes, outcomes_observed, observed_indices, design)
    means = np.full(4, np.nan)
    nonempty = counts > 0
    means[nonempty] = sums[nonempty] / counts[nonempty]
    return _result("state-mean", counts, means, hajek)


def pair_ht_core(state_codes, outcomes_observed, observed_indices, design):
    """Sum Y/(r*q) over both arms, divide once by N, never by arm count."""
    counts, _, weighted_sums, hajek = _records(
        state_codes, outcomes_observed, observed_indices, design)
    means = weighted_sums / design.n
    return _result("ht", counts, means, hajek)


def _public_records(inputs, design):
    """Validate five-array shared pair facts before deriving observed states."""
    if type(inputs) is not tuple or len(inputs) != 2:
        _reject("PAIR_INPUT", "Exactly two ordered five-array mappings required")
    for arm, payload in enumerate(inputs):
        if not isinstance(payload, Mapping) or set(payload) != set(FACTUAL_KEYS):
            _reject("INPUT_BUDGET", "Only the five canonical arrays are permitted", arm=arm)
        for name, shape, dtype in (
                ("x", (design.n, 20), "<f4"),
                ("adjacency", (design.n, design.n), "<f8"),
                ("treatment", (design.n,), "<f8"),
                ("observed_indices", (design.m,), "<i8"),
                ("outcomes_observed", (design.m,), "<f8")):
            _array(payload[name], shape, dtype, name)
    left, right = inputs
    expected = np.zeros((design.n, design.n))
    expected[0, 1:] = expected[1:, 0] = 1.
    if not all(np.array_equal(p["adjacency"], expected) for p in inputs):
        _reject("TOPOLOGY", "Require the canonical center-0 Star")
    if any(not np.array_equal(left[k], right[k]) for k in ("x", "adjacency", "observed_indices")):
        _reject("SHARED_INPUT", "X, graph and ordered observations must be shared")
    t0, t1 = (p["treatment"] for p in inputs)
    if (not np.isin(t0, [0, 1]).all() or not np.isin(t1, [0, 1]).all()
            or t0[0] != 0 or t1[0] != 1 or not np.array_equal(t0[1:], t1[1:])):
        _reject("SHARED_TREATMENT", "Require ordered centers 0/1 and shared binary leaves")
    indices = left["observed_indices"]
    # Observation validation precedes indexing.
    if (np.any(indices < 0) or np.any(indices >= design.n)
            or len(np.unique(indices)) != design.m):
        _reject("OBSERVATIONS", "Require M unique observations in canonical range")
    codes, full_counts, obs_counts, reason_counts = [], [], [], []
    observed_mask = np.zeros(design.n, dtype=bool)
    observed_mask[indices] = True
    center_s = int(t0[1:].mean() > design.exposure_threshold)
    for arm, treatment in enumerate((t0, t1)):
        full = 2 * treatment.astype(np.int64) + arm
        full[0] = 2 * arm + center_s
        codes.append(full[indices])
        full_counts.append(np.stack([np.bincount(full[:1], minlength=4),
                                     np.bincount(full[1:], minlength=4)]))
        obs_counts.append(np.stack([
            np.bincount(full[indices[indices == 0]], minlength=4),
            np.bincount(full[indices[indices != 0]], minlength=4)]))
        reasons = np.zeros((design.n, 4), dtype=np.int64)
        design_mask = np.zeros((design.n, 4), dtype=bool)
        design_mask[0, 2*arm:2*arm+2] = True
        design_mask[1:, arm::2] = True
        reasons[design_mask] = 1
        reasons[np.arange(design.n), full] = 2
        reasons[indices, full[indices]] = 3
        reason_counts.append(np.bincount(reasons.ravel(), minlength=4).tolist())
    return (np.asarray(codes, dtype=np.int64),
            np.stack([p["outcomes_observed"] for p in inputs]), indices,
            {"full_graph_count": np.asarray(full_counts).tolist(),
             "observed_count": np.asarray(obs_counts).tolist(),
             "support_reason_order": REASONS,
             "support_reason_counts": reason_counts,
             "center_observed": bool(observed_mask[0])})


def _compute(inputs, method):
    # Production N/M are fixed; small fixture callers use only the pure cores.
    design = StarPairDesign(200, 50)
    codes, outcomes, indices, support_diagnostics = _public_records(inputs, design)
    core = state_mean_core if method == "state-mean" else pair_ht_core
    result = core(codes, outcomes, indices, design)
    state_info = {
        ARM_ORDER[k]: {"count": int(result.counts[k]), "status": result.state_status[k],
                      "value": float(result.state_values[k]) if
                          np.isfinite(result.state_values[k]) else None,
                      "empirical_supported": bool(result.empirical_state_support[k]),
                      "reason": result.state_reason[k],
                      "hajek_diagnostic": float(result.hajek_diagnostic[k]) if
                          np.isfinite(result.hajek_diagnostic[k]) else None}
        for k in range(4)}
    effects = {
        effect: {"status": result.ate_status[k], "reason": result.effect_reasons[k],
                 "required_states": tuple(ARM_ORDER[z] for z in DEPENDENCIES[k])}
        for k, effect in enumerate(EFFECT_ORDER)}
    failed = any(state != "OK" for state in result.ate_status)
    diagnostics = {
        "state_order": ARM_ORDER, "effect_order": EFFECT_ORDER, "states": state_info,
        "design": {"N": design.n, "M": design.m, "r": design.r, "q": design.q,
                   "joint_inclusion_probability": design.joint_inclusion_probability,
                   "weight": design.weight},
        "n_unique_observed_nodes": design.m, "n_factual_records": 2 * design.m,
        "statistical_unit": "pair", "arms_independent": False,
        "hajek_role": "nonempty_state_equivalence_diagnostic_only",
        **support_diagnostics}
    return {"output_status": {"ate": result.ate_status}, "ate": result.ate,
            "arm_results": (None, None),
            "diagnostics": diagnostics, "estimability": effects,
            "failure_type": "EMPTY_REQUIRED_STATE" if failed else None,
            "failure_message": "; ".join(
                f"{effect}:{result.effect_reasons[k]}" for k, effect in enumerate(EFFECT_ORDER)
                if result.ate_status[k] != "OK") if failed else None,
            "failure_provenance": {"effects": {
                effect: result.effect_reasons[k] for k, effect in enumerate(EFFECT_ORDER)
                if result.ate_status[k] != "OK"}} if failed else {}}


def estimate_state_mean(request, pair_task):
    """Production PairEstimateBundle; NOT_IDENTIFIED never enters the core."""
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Production entry requires N=200 PairTaskSpec")
    return execute_contract(request, pair_task, "state-mean",
                            lambda inputs: _compute(inputs, "state-mean"))


def estimate_pair_ht(request, pair_task):
    """Production pair-level HT with validated shared facts and design weight."""
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Production entry requires N=200 PairTaskSpec")
    return execute_contract(request, pair_task, "ht", lambda inputs: _compute(inputs, "ht"))
