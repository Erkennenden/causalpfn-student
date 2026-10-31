"""S4.2 model-independent contracts, using injected predictors only.

No loaders, asset reads, real models, RNG or method dispatcher. Canonical
predictor arrays are 00,01,10,11; inputs remain raw float64 observed outcomes.
Request IDs are supplied by callers, never generated or rewritten here.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np

from .config import CONFIG
from .contracts import TaskSpec, PairTaskSpec, ContractError
from .estimation import (
    EstimationRequest, EstimateBundle, PairEstimateBundle, IntervalRecord,
    SamplingUncertainty, EFFECT_ORDER, execute_contract, validate_binding, metadata,
)

MODEL_METHODS = ("kernel-old", "kernel-new", "j")
_CONTRASTS = ((2, 0), (3, 2), (3, 0))


def _reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


def _method(method_id):
    if type(method_id) is not str or method_id not in MODEL_METHODS:
        _reject("MODEL_METHOD", "Only declared ordinary context models are accepted")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Optional point output envelope; diagnostics must be finite JSON-like."""
    arm_means: np.ndarray
    arm_order: tuple = CONFIG.arm_order
    diagnostics: Mapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelIntervals:
    """Injected context intervals only; no actual interval algorithm."""
    lower: np.ndarray
    upper: np.ndarray
    level: float = .95
    diagnostics: Mapping = field(default_factory=dict)


def _float_array(value, shape, name):
    if (type(value) is not np.ndarray or value.shape != shape
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        _reject("MODEL_OUTPUT", "Wrong plain floating array shape/dtype",
                field=name, expected_shape=shape)
    return value.astype(np.float64, copy=True)


def context_requests_from_parent(pair_task, parent, *, request_ids):
    """Explicit upstream IDs, inherited Identification; no TaskSpec mutation.

    Bound tasks may share a request correlation ID; task/context IDs distinguish
    their records. Callers must supply exactly the already-bound IDs in that
    case. Unbound tasks can associate distinct supplied child IDs.
    """
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Expected public PairTaskSpec")
    validate_binding(parent, pair_task)
    if type(request_ids) is not tuple or len(request_ids) != 2:
        _reject("REQUEST_IDS", "Exactly two explicit context request IDs required")
    return tuple(EstimationRequest.from_task(
        task, status=parent.status, target_id=parent.target_id,
        request_id=request_ids[k], parent_request_id=parent.request_id,
        requested_outputs=parent.requested_outputs,
        failure_type=parent.failure_type, failure_message=parent.failure_message,
        failure_provenance=parent.failure_provenance)
        for k, task in enumerate(pair_task.context_refs))


def _interval_outputs(payload, request, method_id, interval_predictor):
    if "interval" not in request.requested_outputs or method_id != "j":
        return {}, (), None, {}, {}
    diagnostics, provenance = {}, {}
    states = ["FAILED"] * 3
    records = []
    try:
        if not callable(interval_predictor):
            _reject("INTERVAL_CALLBACK", "Requested interval predictor is missing")
        result = interval_predictor(**payload)
        if type(result) is not ModelIntervals:
            _reject("INTERVAL_OUTPUT", "Expected typed context ModelIntervals")
        if result.level != .95:
            _reject("INTERVAL_LEVEL", "Frozen marginal level is .95")
        lower = _float_array(result.lower, (3,), "interval.lower")
        upper = _float_array(result.upper, (3,), "interval.upper")
        diagnostics = dict(metadata(result.diagnostics))
        for k, effect in enumerate(EFFECT_ORDER):
            valid = bool(np.isfinite([lower[k], upper[k]]).all() and lower[k] <= upper[k])
            states[k] = "OK" if valid else "FAILED"
            records.append(IntervalRecord(
                request.context_ids[0], effect, states[k],
                float(lower[k]) if valid else None, float(upper[k]) if valid else None,
                source="model_posterior" if valid else None))
            if not valid:
                provenance[effect] = {"type": "INVALID_INTERVAL", "message": "Nonfinite/reversed endpoints"}
    except Exception as exc:
        records = [IntervalRecord(request.context_ids[0], effect, "FAILED")
                   for effect in EFFECT_ORDER]
        states = ["FAILED"] * 3
        diagnostics = {}
        provenance = {effect: {"type": type(exc).__name__,
                              "message": str(exc) or type(exc).__name__}
                      for effect in EFFECT_ORDER}
    uncertainty = (SamplingUncertainty("OK", "model_posterior", "context", None)
                   if "OK" in states else SamplingUncertainty("FAILED"))
    return {"interval": tuple(states)}, tuple(records), uncertainty, diagnostics, provenance


def _context_output(payload, request, method_id, predictor, interval_predictor):
    # execute_contract handles whole-predictor exceptions: no partial success.
    result = predictor(**payload)
    diagnostics = {}
    if type(result) is ModelPrediction:
        if result.arm_order != CONFIG.arm_order:
            _reject("STATE_ORDER", "Require explicit 00,01,10,11 model axis")
        diagnostics = dict(metadata(result.diagnostics))
        result = result.arm_means
    points = _float_array(result, (200, 4), "arm_means")
    mu_ok = bool(np.isfinite(points).all())
    mu = points.reshape(200, 2, 2) if mu_ok else np.full((200, 2, 2), np.nan)
    ite = np.full((200, 3), np.nan)
    ate = np.full(3, np.nan)
    effect_states, effect_provenance = [], {}
    for k, (positive, negative) in enumerate(_CONTRASTS):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = points[:, positive] - points[:, negative]
            average = difference.mean()
        valid = bool(np.isfinite(difference).all() and np.isfinite(average))
        effect_states.append("OK" if valid else "FAILED")
        if valid:
            ite[:, k], ate[k] = difference, average
        else:
            effect_provenance[EFFECT_ORDER[k]] = {
                "type": "NONFINITE_EFFECT", "message": "Invalid necessary state/contrast/mean"}
    output_status = {}
    values = {}
    if "mu" in request.requested_outputs:
        output_status["mu"] = "OK" if mu_ok else "FAILED"
        values["mu"] = mu
    for name, array in (("ite", ite), ("ate", ate)):
        if name in request.requested_outputs:
            output_status[name] = tuple(effect_states)
            values[name] = array
    interval_status, intervals, uncertainty, interval_diag, interval_failures = _interval_outputs(
        payload, request, method_id, interval_predictor)
    output_status.update(interval_status)
    provenance = {}
    if not mu_ok and "mu" in request.requested_outputs:
        provenance["mu"] = {"type": "NONFINITE_STATE", "message": "Invalid state response; mu field unavailable"}
    if {"ite", "ate"} & set(request.requested_outputs):
        provenance["effects"] = effect_provenance
    if interval_failures:
        provenance["intervals"] = interval_failures
    failed = (("mu" in request.requested_outputs and not mu_ok)
              or bool({"ite", "ate"} & set(request.requested_outputs) and effect_provenance)
              or bool(interval_failures))
    return {
        **values, "output_status": output_status, "intervals": intervals,
        "sampling_uncertainty": uncertainty,
        "diagnostics": {"point": diagnostics, "interval": interval_diag,
                        "state_order": CONFIG.arm_order, "effect_order": EFFECT_ORDER,
                        "query_nodes": 200, "observed_nodes": 50},
        "estimability": {effect: {"status": effect_states[k],
                                "reason": "finite_full_graph_contrast" if effect_states[k] == "OK"
                                          else "invalid_full_graph_contrast"}
                        for k, effect in enumerate(EFFECT_ORDER)},
        "failure_type": "MODEL_OUTPUT_FAILURE" if failed else None,
        "failure_message": "Point field/effect or context interval unavailable" if failed else None,
        "failure_provenance": provenance,
    }


def estimate_context(task, request, method_id, predictor, *, interval_predictor=None):
    """Only five keyword arrays cross the injected predictor boundary."""
    if type(task) is not TaskSpec:
        _reject("CONTEXT_TASK", "Expected public TaskSpec, never simulator context")
    _method(method_id)
    validate_binding(request, task)
    if request.status == "POINT" and not callable(predictor):
        _reject("PREDICTOR", "A callable predictor must be explicitly injected")
    result = execute_contract(
        request, task, method_id,
        lambda payload: _context_output(payload, request, method_id, predictor, interval_predictor))
    # Detect externally-held-task tampering too; never repair/hash anew.
    validate_binding(request, task)
    return result


def _validate_children(pair_task, request, children):
    validate_binding(request, pair_task)
    if type(children) is not tuple or len(children) != 2:
        _reject("CHILD_REQUESTS", "Require two explicit context requests")
    for k, child in enumerate(children):
        validate_binding(child, pair_task.context_refs[k])
        if (child.parent_request_id != request.request_id
                or child.status != request.status or child.target_id != request.target_id
                or child.requested_outputs != request.requested_outputs
                or child.failure_type != request.failure_type
                or child.failure_message != request.failure_message
                or dict(child.failure_provenance) != dict(request.failure_provenance)):
            _reject("CHILD_REQUESTS", "Child must inherit upstream semantics and parent link", arm=k)


def compose_pair(pair_task, request, method_id, context_results):
    """Compose caller-provided context evidence; no model calls or retries."""
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Expected public PairTaskSpec")
    _method(method_id)
    if (type(context_results) is not tuple or len(context_results) != 2
            or any(type(arm) is not EstimateBundle for arm in context_results)):
        _reject("ARM_RESULTS", "Require exactly two context EstimateBundles")
    _validate_children(pair_task, request, tuple(arm.request for arm in context_results))
    if any(arm.method_id != method_id for arm in context_results):
        _reject("ARM_METHOD", "Both arm methods must agree")
    if "ate" not in request.requested_outputs:
        _reject("PAIR_REQUEST", "Pair composition requires explicitly requested ATE")
    ate = np.full(3, np.nan)
    statuses, evidence = [], {}
    for k, effect in enumerate(EFFECT_ORDER):
        states = tuple(arm.output_status["ate"][k] for arm in context_results)
        if request.status == "NOT_IDENTIFIED":
            status = "FAILED"  # output code; execution/status remain NOT_IDENTIFIED.
        elif all(state == "OK" for state in states):
            status = "OK"
            # Avoid overflow from adding two finite large values.
            ate[k] = sum(arm.ate[k] / 2 for arm in context_results)
            if not np.isfinite(ate[k]):
                _reject("PAIR_NUMERIC", "Nonfinite mean of valid context effects", effect=effect)
        else:
            status = "FAILED" if "FAILED" in states else "NOT_ESTIMABLE"
        statuses.append(status)
        evidence[effect] = {"status": status, "arm_statuses": states,
                            "reason": "two_context_mean" if status == "OK" else "necessary_arm_unavailable"}
    failure = any(state != "OK" for state in statuses)
    provenance = {"arms": tuple({
        "context_id": arm.context_ids[0], "request_id": arm.request_id,
        "execution_status": arm.execution_status, "output_status": dict(arm.output_status),
        "failure_type": arm.failure_type, "failure_message": arm.failure_message,
        "failure_provenance": arm.failure_provenance} for arm in context_results)}
    if request.status == "NOT_IDENTIFIED":
        failure_type, message, provenance = (
            request.failure_type, request.failure_message, request.failure_provenance)
    else:
        failure_type = "NECESSARY_ARM_FAILURE" if failure else None
        message = "One or more context ATE effects unavailable" if failure else None
    return PairEstimateBundle.from_outputs(
        request, pair_task, method_id, ate=ate, output_status={"ate": tuple(statuses)},
        arm_results=context_results, estimability=evidence,
        diagnostics={"aggregation": "mean_of_two_context_effects", "arms_independent": False,
                     "n_unique_observed_nodes": 50, "n_factual_records": 100,
                     "interval_scope": "context_only"},
        failure_type=failure_type, failure_message=message, failure_provenance=provenance)


def estimate_pair(pair_task, request, method_id, *, context_requests, predictors,
                  interval_predictors=(None, None)):
    """Public mock-ready orchestration, not a model loader or seven-method router."""
    if type(pair_task) is not PairTaskSpec:
        _reject("PAIR_TASK", "Expected public PairTaskSpec")
    _method(method_id)
    _validate_children(pair_task, request, context_requests)
    if "ate" not in request.requested_outputs:
        _reject("PAIR_REQUEST", "Pair adapter requires explicitly requested ATE")
    if (type(predictors) is not tuple or len(predictors) != 2
            or type(interval_predictors) is not tuple or len(interval_predictors) != 2):
        _reject("PREDICTORS", "Exactly two explicit predictor/interval slots required")
    if request.status == "POINT" and any(not callable(p) for p in predictors):
        _reject("PREDICTORS", "Both normal context predictors must be callable before execution")
    arms = tuple(estimate_context(
        task, context_requests[k], method_id, predictors[k],
        interval_predictor=interval_predictors[k])
        for k, task in enumerate(pair_task.context_refs))
    return compose_pair(pair_task, request, method_id, arms)
