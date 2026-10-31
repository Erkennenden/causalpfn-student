"""S3.1 contracts only: no Identification inference or numerical estimators.

Upstream status is mandatory. Unbound Stage-2 tasks are associated explicitly,
without rewriting their IDs, arrays or input hashes. Ordinary callbacks receive
only detached five-array mappings, never requests, bundles or diagnostics.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
import re

import numpy as np

from .config import CONFIG
from .contracts import (TaskSpec, PairTaskSpec, validate_task, validate_pair_task,
                        predictor_inputs, validate_predictor_inputs, ContractError)
from .support import (ContextSupport, PairSupport, method_capability,
                      support_for_task, support_for_pair)

EFFECT_ORDER = ("Direct", "Spillover", "Total")
OUTPUT_STATES = frozenset(("OK", "FAILED", "NOT_ESTIMABLE",
                          "UNSUPPORTED", "NOT_REQUESTED"))
EXECUTION_STATES = frozenset(("OK", "FAILED", "NOT_ESTIMABLE", "NOT_IDENTIFIED"))
REFERENCE = "iid Bernoulli(0.5)"
EXPOSURE = "neighbor treatment proportion > 0.5"
FUNCTIONAL = "mu_i(t,s) under reference assignment; Direct, Spillover, Total"
ATE_ONLY = frozenset(("state-mean", "ht", "f-z2", "lin-z2"))


def reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


def frozen_array(a):
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def text_id(value, name):
    if type(value) is not str or not value.strip():
        reject("IDENTITY", name + " must be a nonempty upstream string", field=name)


def digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def metadata(value):
    """Immutable JSON-like diagnostics; no simulator arrays or objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    if isinstance(value, Mapping):
        banned = {"truth", "beta", "h", "full_y", "full_outcomes", "noise",
                  "raw_noise", "coefficients", "evaluation_results"}
        if any(type(k) is not str or k.lower() in banned for k in value):
            reject("PRIVILEGED_METADATA", "Forbidden diagnostic key")
        return MappingProxyType({k: metadata(v) for k, v in value.items()})
    if type(value) in (tuple, list):
        return tuple(metadata(v) for v in value)
    reject("METADATA", "Only finite JSON-like metadata allowed")


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimationRequest:
    status: str
    target_id: str
    request_id: str
    scope: str
    pair_id: str
    context_ids: tuple
    input_hashes: tuple
    node_ids: np.ndarray
    identity: Mapping
    requested_outputs: tuple = ("mu", "ite", "ate")
    failure_type: str | None = None
    failure_message: str | None = None
    failure_provenance: Mapping = field(default_factory=dict)
    parent_request_id: str | None = None
    protocol_id: str = CONFIG.protocol_id
    protocol_version: str = CONFIG.protocol_version
    protocol_hash: str = CONFIG.protocol_sha256
    reference_assignment: str = REFERENCE
    exposure_definition: str = EXPOSURE
    target_functional: str = FUNCTIONAL
    arm_order: tuple = CONFIG.arm_order
    effect_order: tuple = EFFECT_ORDER

    def __post_init__(self):
        if self.status not in ("POINT", "NOT_IDENTIFIED"):
            reject("IDENTIFICATION", "Upstream POINT/NOT_IDENTIFIED is required")
        text_id(self.request_id, "request_id")
        if self.target_id != CONFIG.target_id:
            reject("TARGET", "Unregistered upstream target", target_id=self.target_id)
        if (self.protocol_id, self.protocol_version, self.protocol_hash,
            self.reference_assignment, self.exposure_definition,
            self.target_functional, self.arm_order, self.effect_order) != (
                CONFIG.protocol_id, CONFIG.protocol_version, CONFIG.protocol_sha256,
                REFERENCE, EXPOSURE, FUNCTIONAL, CONFIG.arm_order, EFFECT_ORDER):
            reject("PROTOCOL", "Request changes frozen scientific semantics")
        n = {"context": 1, "pair": 2}.get(self.scope)
        if (n is None or type(self.context_ids) is not tuple
                or type(self.input_hashes) is not tuple
                or len(self.context_ids) != n or len(set(self.context_ids)) != n
                or len(self.input_hashes) != n or not all(map(digest, self.input_hashes))):
            reject("IDENTITY", "Request scope/context/hash mismatch")
        text_id(self.pair_id, "pair_id")
        for value in self.context_ids:
            text_id(value, "context_id")
        if (type(self.node_ids) is not np.ndarray or self.node_ids.dtype.str != "<i8"
                or not np.array_equal(self.node_ids, np.arange(200))):
            reject("NODE_ORDER", "Require canonical int64 node_ids[200]")
        if (type(self.requested_outputs) is not tuple or not self.requested_outputs
                or len(set(self.requested_outputs)) != len(self.requested_outputs)
                or not set(self.requested_outputs) <= {"mu", "ite", "ate", "interval"}):
            reject("REQUEST", "Invalid requested output set")
        if self.parent_request_id is not None:
            text_id(self.parent_request_id, "parent_request_id")
        if self.status == "NOT_IDENTIFIED":
            text_id(self.failure_type, "upstream failure_type")
            text_id(self.failure_message, "upstream failure_message")
        object.__setattr__(self, "node_ids", frozen_array(self.node_ids))
        object.__setattr__(self, "identity", metadata(self.identity))
        object.__setattr__(self, "failure_provenance", metadata(self.failure_provenance))

    @property
    def population_support(self):
        return frozen_array(np.ones((200, 2, 2), dtype=bool))

    @property
    def task_id(self):
        return self.context_ids[0] if self.scope == "context" else self.pair_id

    @classmethod
    def from_task(cls, task, *, status, target_id, request_id, **kwargs):
        contexts = task_contexts(task)
        identity = {k: getattr(contexts[0], k) for k in (
            "cohort", "cohort_code", "mechanism_id", "pair_repeat_id", "rng_identity")}
        identity["center_t"] = tuple(t.center_t for t in contexts)
        result = cls(status=status, target_id=target_id, request_id=request_id,
                     scope="context" if type(task) is TaskSpec else "pair",
                     pair_id=task.pair_id,
                     context_ids=tuple(t.context_id for t in contexts),
                     input_hashes=tuple(t.input_hash for t in contexts),
                     node_ids=task.node_ids, identity=identity, **kwargs)
        validate_binding(result, task)
        return result


def task_contexts(task):
    if type(task) is TaskSpec:
        validate_task(task)
        return (task,)
    if type(task) is PairTaskSpec:
        validate_pair_task(task)
        return task.context_refs
    reject("TASK", "Expected Stage-2 TaskSpec or PairTaskSpec")


def validate_binding(request, task):
    if type(request) is not EstimationRequest:
        reject("REQUEST", "Expected an explicit EstimationRequest")
    contexts = task_contexts(task)
    scope = "context" if type(task) is TaskSpec else "pair"
    expected = {k: getattr(contexts[0], k) for k in (
        "cohort", "cohort_code", "mechanism_id", "pair_repeat_id", "rng_identity")}
    expected["center_t"] = tuple(t.center_t for t in contexts)
    if (request.scope != scope or request.pair_id != task.pair_id
            or request.context_ids != tuple(t.context_id for t in contexts)
            or request.input_hashes != tuple(t.input_hash for t in contexts)
            or dict(request.identity) != expected
            or not np.array_equal(request.node_ids, task.node_ids)):
        reject("BINDING", "Task/request identity conflict", request_id=request.request_id,
               context_ids=tuple(t.context_id for t in contexts))
    for t in contexts:
        if ((t.target_id is not None and t.target_id != request.target_id)
                or (t.request_id is not None and t.request_id != request.request_id)):
            reject("BINDING", "Existing upstream task link must remain unchanged",
                   context_id=t.context_id, request_id=t.request_id)


@dataclass(frozen=True, slots=True)
class SamplingUncertainty:
    status: str = "NOT_REQUESTED"
    source: str | None = None
    statistical_unit: str | None = None
    effective_sample_count: int | None = None

    def __post_init__(self):
        if self.status not in OUTPUT_STATES:
            reject("UNCERTAINTY", "Unknown uncertainty capability/output status")
        if (self.source is None) != (self.statistical_unit is None):
            reject("UNCERTAINTY", "Source and statistical unit must be provided together")
        if self.status == "OK":
            text_id(self.source, "uncertainty source")
            text_id(self.statistical_unit, "uncertainty statistical_unit")
            if self.source == "mechanism_bootstrap" and self.statistical_unit != "mechanism":
                reject("UNCERTAINTY", "Bootstrap unit must be mechanism")
        elif self.status == "FAILED" and self.source is not None:
            # Provenance is not an available uncertainty estimate. Method
            # eligibility is checked by EstimateBundle, not inferred here.
            text_id(self.source, "uncertainty source")
            text_id(self.statistical_unit, "uncertainty statistical_unit")
            if self.effective_sample_count is not None:
                reject("UNCERTAINTY", "Failed uncertainty has no effective sample count")
        elif any(v is not None for v in (
                self.source, self.statistical_unit, self.effective_sample_count)):
            reject("UNCERTAINTY", "Unavailable uncertainty must not fabricate metadata")
        n = self.effective_sample_count
        if n is not None and (type(n) is not int or n <= 0):
            reject("UNCERTAINTY", "Effective count must be a positive integer")


@dataclass(frozen=True, slots=True)
class IntervalRecord:
    context_id: str | None
    effect: str
    status: str
    lower: float | None = None
    upper: float | None = None
    level: float = 0.95
    source: str | None = None

    def __post_init__(self):
        if self.effect not in EFFECT_ORDER or self.status not in OUTPUT_STATES:
            reject("INTERVAL", "Invalid interval effect/status")
        if self.level != 0.95:
            reject("INTERVAL", "Frozen marginal interval level is 0.95")
        if self.status == "OK":
            if (type(self.lower) not in (float, int) or type(self.upper) not in (float, int)
                    or not np.isfinite([self.lower, self.upper]).all()
                    or self.lower > self.upper):
                reject("INTERVAL", "Finite ordered endpoints required")
            text_id(self.source, "interval source")
        elif self.lower is not None or self.upper is not None:
            reject("INTERVAL", "Unavailable interval must have no endpoints")


def normalized_status(request, capability, supplied):
    if not isinstance(supplied, Mapping) or not set(supplied) <= {"mu", "ite", "ate", "interval"}:
        reject("OUTPUT_STATUS", "Unknown output status field")
    result = {}
    for name in ("mu", "ite", "ate", "interval"):
        key = request.scope + "." + ("ate_interval" if name == "interval" else name)
        applicable = key in capability.outputs
        # ATE-only node outputs remain explicitly UNSUPPORTED, even if omitted.
        default = ("UNSUPPORTED" if name in ("mu", "ite") and capability.method_id in ATE_ONLY
                   else "NOT_REQUESTED" if name not in request.requested_outputs
                   else "UNSUPPORTED" if not applicable else None)
        value = supplied.get(name, default)
        values = (value,) if name == "mu" else value
        if type(values) is not tuple or len(values) != (1 if name == "mu" else 3):
            # Scalar defaults expand, explicit per-effect values must be tuples.
            if type(value) is str and name != "mu":
                values = (value,) * 3
            else:
                reject("OUTPUT_STATUS", "Three effect statuses required", field=name)
        for state in values:
            if state not in OUTPUT_STATES:
                reject("OUTPUT_STATUS", "Explicit output status required", field=name)
            if default is not None and state != default:
                reject("CAPABILITY", "Requested/capability status conflict", field=name)
            if default is None and state not in ("OK", "FAILED", "NOT_ESTIMABLE"):
                reject("OUTPUT_STATUS", "Applicable requested output needs execution evidence")
            if request.status == "NOT_IDENTIFIED" and state == "OK":
                reject("IDENTIFICATION", "Nonidentified request cannot have valid outputs")
        result[name] = values[0] if name == "mu" else tuple(values)
    return MappingProxyType(result)


def output_array(value, shape, statuses, field_name):
    states = (statuses,) if type(statuses) is str else statuses
    if value is None:
        a = np.full(shape, np.nan, dtype=np.float64)
    else:
        if type(value) is not np.ndarray or value.dtype.str != "<f8" or value.shape != shape:
            reject("OUTPUT_ARRAY", "Require canonical float64 shape", field=field_name, shape=shape)
        a = value
    for k, state in enumerate(states):
        part = a if len(states) == 1 else a[..., k]
        if state == "OK":
            if not np.isfinite(part).all():
                reject("OUTPUT_VALUE", "OK output must be finite", field=field_name, effect=k)
        elif not np.isnan(part).all():
            reject("OUTPUT_VALUE", "Non-OK output must be NaN, never a fake zero",
                   field=field_name, effect=k)
    return frozen_array(a)


def aggregate_execution(request, statuses):
    if request.status == "NOT_IDENTIFIED":
        return "NOT_IDENTIFIED"
    required = [v for name, value in statuses.items() for v in (
        (value,) if type(value) is str else value)
        if name in request.requested_outputs and v not in ("UNSUPPORTED", "NOT_REQUESTED")]
    if not required:
        reject("NO_APPLICABLE_OUTPUT", "Method has no applicable requested output")
    return "FAILED" if "FAILED" in required else (
        "NOT_ESTIMABLE" if "NOT_ESTIMABLE" in required else "OK")


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimateBundle:
    request: EstimationRequest
    method_id: str
    support: ContextSupport | PairSupport
    output_status: Mapping
    mu: np.ndarray | None = None
    ite: np.ndarray | None = None
    ate: np.ndarray | None = None
    intervals: tuple = ()
    sampling_uncertainty: SamplingUncertainty | None = None
    estimability: Mapping = field(default_factory=dict)
    diagnostics: Mapping = field(default_factory=dict)
    failure_type: str | None = None
    failure_message: str | None = None
    failure_provenance: Mapping = field(default_factory=dict)
    asset_hashes: Mapping = field(default_factory=dict)
    execution_status: str = field(init=False)

    def __post_init__(self):
        if type(self.request) is not EstimationRequest:
            reject("REQUEST", "Explicit upstream request required")
        pair = type(self) is PairEstimateBundle
        if self.request.scope != ("pair" if pair else "context"):
            reject("SCOPE", "Bundle scope mismatch")
        cap = method_capability(self.method_id)
        object.__setattr__(self, "method_id", cap.method_id)
        s = self.support
        if ((not pair and (type(s) is not ContextSupport
                           or s.context_id != self.context_ids[0]
                           or s.source_input_hash != self.input_hashes[0]))
                or (pair and (type(s) is not PairSupport or s.pair_id != self.pair_id
                    or tuple(c.context_id for c in s.context_supports) != self.context_ids
                    or tuple(c.source_input_hash for c in s.context_supports) != self.input_hashes))
                or not np.array_equal(s.node_ids, self.node_ids)):
            reject("SUPPORT_BINDING", "Support is not from these factual inputs")
        statuses = normalized_status(self.request, cap, self.output_status)
        object.__setattr__(self, "output_status", statuses)
        for name, shape in (("mu", (200, 2, 2)), ("ite", (200, 3)), ("ate", (3,))):
            object.__setattr__(self, name, output_array(
                getattr(self, name), shape, statuses[name], name))
        execution = aggregate_execution(self.request, statuses)
        object.__setattr__(self, "execution_status", execution)
        if execution == "NOT_IDENTIFIED":
            if ((self.failure_type, self.failure_message) != (
                    self.request.failure_type, self.request.failure_message)
                    or dict(self.failure_provenance) != dict(self.request.failure_provenance)):
                reject("FAILURE_PROVENANCE", "Preserve upstream identification failure verbatim")
        elif execution != "OK":
            text_id(self.failure_type, "failure_type")
            text_id(self.failure_message, "failure_message")
        for name in ("estimability", "diagnostics", "failure_provenance", "asset_hashes"):
            object.__setattr__(self, name, metadata(getattr(self, name)))
        if any(not digest(v) for v in self.asset_hashes.values()):
            reject("ASSET_HASH", "Assets require explicit SHA256 identities")
        uncertainty = self.sampling_uncertainty
        interval_key = self.request.scope + ".ate_interval"
        if uncertainty is None:
            interval_states = statuses["interval"]
            if "OK" in interval_states:
                reject("UNCERTAINTY", "Valid interval needs explicit uncertainty provenance")
            state = ("NOT_REQUESTED" if "interval" not in self.request.requested_outputs
                     else "UNSUPPORTED" if interval_key not in cap.outputs
                     else "FAILED" if "FAILED" in interval_states
                     else "NOT_ESTIMABLE" if "NOT_ESTIMABLE" in interval_states else None)
            if state is None:
                reject("UNCERTAINTY", "Valid interval needs explicit uncertainty provenance")
            object.__setattr__(self, "sampling_uncertainty", SamplingUncertainty(state))
        if type(self.sampling_uncertainty) is not SamplingUncertainty:
            reject("UNCERTAINTY", "Typed sampling uncertainty required")
        if ("interval" not in self.request.requested_outputs
                and self.sampling_uncertainty.status != "NOT_REQUESTED"):
            reject("UNCERTAINTY", "Uncertainty was not requested")
        if ("interval" in self.request.requested_outputs and interval_key not in cap.outputs
                and self.sampling_uncertainty.status != "UNSUPPORTED"):
            reject("UNCERTAINTY", "Unsupported uncertainty must be explicit")
        uncertainty = self.sampling_uncertainty
        if cap.method_id == "j":
            if uncertainty.effective_sample_count is not None:
                reject("UNCERTAINTY", "Analytic J uncertainty has no effective sample count")
            if (uncertainty.source is not None
                    and (uncertainty.source, uncertainty.statistical_unit)
                    != ("model_posterior", "context")):
                reject("UNCERTAINTY", "J provenance requires model_posterior/context")
            interval_states = statuses["interval"]
            if "OK" in interval_states and uncertainty.status != "OK":
                reject("UNCERTAINTY", "Partially valid J intervals retain OK uncertainty status")
            if interval_states == ("FAILED",) * 3 and uncertainty.status != "FAILED":
                reject("UNCERTAINTY", "All failed J intervals must retain FAILED status")
            if (uncertainty.status == "FAILED" and uncertainty.source is not None
                    and interval_states != ("FAILED",) * 3):
                reject("UNCERTAINTY", "Failed J provenance requires three failed intervals")
        elif uncertainty.status != "OK" and any(value is not None for value in (
                uncertainty.source, uncertainty.statistical_unit, uncertainty.effective_sample_count)):
            # Keep the original non-J unavailable-uncertainty rule.
            reject("UNCERTAINTY", "Unavailable uncertainty must not fabricate metadata")
        if self.sampling_uncertainty.status == "OK":
            expected_uncertainty = {
                "j": ("model_posterior", "context"),
                "oracle-context": ("model_posterior", "context"),
                "oracle-pair": ("joint_model_posterior", "pair"),
            }.get(cap.method_id)
            if (self.sampling_uncertainty.source,
                    self.sampling_uncertainty.statistical_unit) != expected_uncertainty:
                reject("UNCERTAINTY", "Source/unit conflicts with registered method capability")
            if "OK" not in statuses["interval"]:
                reject("UNCERTAINTY", "No valid interval evidence")
            if self.request.status != "POINT":
                reject("UNCERTAINTY", "No valid uncertainty for NOT_IDENTIFIED")
            if self.sampling_uncertainty.source == "mechanism_bootstrap":
                reject("UNCERTAINTY", "Evaluator bootstrap is not a single-task posterior")
            if "interval" not in self.request.requested_outputs:
                reject("UNCERTAINTY", "Unrequested uncertainty output")
            key = self.request.scope + ".ate_interval"
            if key not in cap.outputs:
                reject("UNCERTAINTY", "Method has no registered uncertainty capability")
        if type(self.intervals) is not tuple or any(type(i) is not IntervalRecord for i in self.intervals):
            reject("INTERVAL", "Typed immutable interval records required")
        if pair:
            self._validate_pair()
        else:
            self._validate_intervals(self.context_ids[0])

    def _validate_intervals(self, context_id):
        by_effect = {}
        for record in self.intervals:
            if record.context_id != context_id or record.effect in by_effect:
                reject("INTERVAL", "Duplicate/wrong interval identity")
            if record.status == "OK":
                expected_source = ("joint_model_posterior" if self.method_id == "oracle-pair"
                                   else "model_posterior")
                if record.source != expected_source:
                    reject("INTERVAL", "Interval source conflicts with method capability")
            by_effect[record.effect] = record
        for k, effect in enumerate(EFFECT_ORDER):
            state = self.output_status["interval"][k]
            record = by_effect.get(effect)
            if state in ("OK", "FAILED", "NOT_ESTIMABLE") and record is None:
                reject("INTERVAL", "Requested interval denominator needs a record", effect=effect)
            if record is not None and record.status != state:
                reject("INTERVAL", "Interval record/status mismatch", effect=effect)

    @classmethod
    def from_outputs(cls, request, task, method_id, **outputs):
        validate_binding(request, task)
        support = support_for_task(task) if type(task) is TaskSpec else support_for_pair(task)
        return cls(request=request, method_id=method_id, support=support, **outputs)

    @property
    def status(self): return self.request.status
    @property
    def target_id(self): return self.request.target_id
    @property
    def request_id(self): return self.request.request_id
    @property
    def pair_id(self): return self.request.pair_id
    @property
    def context_ids(self): return self.request.context_ids
    @property
    def node_ids(self): return self.request.node_ids
    @property
    def input_hashes(self): return self.request.input_hashes
    @property
    def input_hash(self): return self.input_hashes[0] if self.request.scope == "context" else None
    @property
    def task_id(self): return self.request.task_id
    @property
    def protocol_hash(self): return self.request.protocol_hash
    @property
    def protocol_version(self): return self.request.protocol_version
    @property
    def arm_order(self): return self.request.arm_order
    @property
    def effect_order(self): return EFFECT_ORDER
    @property
    def method_capability(self): return method_capability(self.method_id)
    @property
    def population_support(self): return self.support.population_support
    @property
    def design_support(self): return self.support.design_support
    @property
    def empirical_support_mask(self): return self.support.empirical_support_mask
    @property
    def extrapolation_flags(self):
        return MappingProxyType({
            "empirical": self.support.empirical_extrapolation,
            "context_design": (tuple(s.context_design_extrapolation for s in self.support.context_supports)
                               if type(self.support) is PairSupport
                               else self.support.context_design_extrapolation)})


@dataclass(frozen=True, slots=True, kw_only=True)
class PairEstimateBundle(EstimateBundle):
    # None execution evidence means not executed, NOT a second context estimate.
    arm_results: tuple = ()

    def _validate_pair(self):
        if type(self.arm_results) is not tuple or len(self.arm_results) != 2:
            reject("PAIR_ARMS", "Exactly two arm evidence entries are required")
        cap = self.method_capability
        joint = cap.method_id in ATE_ONLY or cap.method_id == "oracle-pair"
        if joint and any(v is not None for v in self.arm_results):
            reject("PAIR_ARMS", "Joint fit cannot invent independent context estimates")
        for k, arm in enumerate(self.arm_results):
            if arm is None:
                if not joint and self.status == "POINT":
                    reject("PAIR_ARMS", "Context method requires both arm result records")
                continue
            if (type(arm) is not EstimateBundle or arm.method_id != self.method_id
                    or arm.context_ids != (self.context_ids[k],)
                    or arm.input_hashes != (self.input_hashes[k],)
                    or arm.target_id != self.target_id
                    or arm.request.parent_request_id != self.request_id
                    or arm.pair_id != self.pair_id):
                reject("PAIR_ARMS", "Arm request/context identity mismatch", arm=k)
            for effect in range(3):
                if (self.output_status["ate"][effect] == "OK"
                        and arm.output_status["ate"][effect] != "OK"):
                    reject("PAIR_DEPENDENCY", "Failed necessary arm invalidates this effect",
                           arm=k, effect=effect)
        # J/kernel intervals live exclusively in their original context bundles.
        if self.method_id != "oracle-pair":
            if self.intervals:
                reject("PAIR_INTERVAL", "Do not synthesize a pair interval from context bounds")
        else:
            self._validate_intervals(None)

    @property
    def arm_status_evidence(self):
        return tuple(MappingProxyType({
            "context_id": self.context_ids[k], "input_hash": self.input_hashes[k],
            "execution_status": None if arm is None else arm.execution_status,
            "output_status": None if arm is None else arm.output_status,
            "failure_type": self.failure_type if arm is None else arm.failure_type,
            "failure_message": self.failure_message if arm is None else arm.failure_message,
            "failure_provenance": self.failure_provenance if arm is None else arm.failure_provenance,
            "estimation_role": "not_independently_executed" if arm is None else "context_result",
        }) for k, arm in enumerate(self.arm_results))

    @property
    def context_intervals(self):
        return tuple(() if arm is None else arm.intervals for arm in self.arm_results)


def execute_contract(request, task, method_id, callback):
    """Boundary adapter for a future implementation; never estimates by itself.

    No retry, resampling, model loading or evaluation. Callback result keys are
    constructor arguments only. A malformed result is explicitly rejected.
    """
    validate_binding(request, task)
    cap = method_capability(method_id)
    if cap.privileged:
        reject("PRIVILEGED_METHOD", "Oracle execution is outside ordinary adapter")
    cls = EstimateBundle if request.scope == "context" else PairEstimateBundle
    if request.status == "NOT_IDENTIFIED":
        statuses = {}
        for name in request.requested_outputs:
            key = request.scope + "." + ("ate_interval" if name == "interval" else name)
            if key in cap.outputs:
                statuses[name] = "FAILED"
        records = tuple(IntervalRecord(request.context_ids[0], e, "FAILED")
                        for e in EFFECT_ORDER) if (
                            request.scope == "context" and "interval" in statuses) else ()
        extra = {"arm_results": (None, None)} if request.scope == "pair" else {}
        return cls.from_outputs(request, task, method_id, output_status=statuses,
                                intervals=records, failure_type=request.failure_type,
                                failure_message=request.failure_message,
                                failure_provenance=request.failure_provenance, **extra)
    if not callable(callback):
        reject("CALLBACK", "Normal execution requires a callable")
    # Metadata is NOT passed to a predictor/estimator.
    inputs = tuple(predictor_inputs(t) for t in task_contexts(task))
    for payload in inputs:
        validate_predictor_inputs(payload)
    if not any(request.scope + "." + ("ate_interval" if name == "interval" else name)
               in cap.outputs for name in request.requested_outputs):
        reject("NO_APPLICABLE_OUTPUT", "Method has no applicable requested output")
    if request.scope == "pair" and cap.method_id not in ATE_ONLY:
        reject("PAIR_ARMS", "Build context-derived pair results from explicit arm bundles")
    try:
        result = callback(inputs[0] if request.scope == "context" else inputs)
    except Exception as exc:
        statuses = {name: "FAILED" for name in request.requested_outputs
                    if request.scope + "." + ("ate_interval" if name == "interval" else name)
                    in cap.outputs}
        records = tuple(IntervalRecord(request.context_ids[0], e, "FAILED")
                        for e in EFFECT_ORDER) if (
                            request.scope == "context" and "interval" in statuses) else ()
        extra = {"arm_results": (None, None)} if request.scope == "pair" else {}
        return cls.from_outputs(
            request, task, method_id, output_status=statuses, intervals=records,
            failure_type=type(exc).__name__, failure_message=str(exc) or type(exc).__name__,
            failure_provenance={"stage": "ordinary_callback", "attempts": 1}, **extra)
    if not isinstance(result, Mapping):
        reject("RESULT", "Callback must return explicit contract fields")
    allowed = {"output_status", "mu", "ite", "ate", "intervals", "sampling_uncertainty",
               "estimability", "diagnostics", "failure_type", "failure_message",
               "failure_provenance", "asset_hashes"}
    if request.scope == "pair":
        allowed.add("arm_results")
    if "output_status" not in result:
        reject("RESULT", "Explicit output_status is required")
    if not set(result) <= allowed:
        reject("RESULT", "Extra or privileged result fields rejected",
               fields=tuple(sorted(set(result) - allowed)))
    return cls.from_outputs(request, task, method_id, **result)
