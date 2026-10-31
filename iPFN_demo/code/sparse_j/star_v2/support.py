"""Support diagnostics from canonical factual tasks and public Star design only.

No privileged simulator input, estimator, prediction, RNG draw or decision run.
Capability declarations are independent of all Support masks.
"""
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np

from .config import CONFIG
from .contracts import validate_task, validate_pair_task

REASONS = ("conditional_structural_zero", "assignment_zero",
           "observation_sampling_zero", "covered")
GROUP_ORDER = ("center", "leaf")


def _freeze(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def majority_exposure(adjacency, treatment):
    """Pure public graph rule; small mathematical fixtures may have ties."""
    a, t = np.asarray(adjacency), np.asarray(treatment)
    if (a.dtype.kind not in "biuf" or t.dtype.kind not in "biuf"
            or a.ndim != 2 or a.shape[0] < 2 or a.shape[0] != a.shape[1]
            or t.shape != (len(a),) or not np.isfinite(a).all()
            or not np.isfinite(t).all() or not np.isin(a, [0, 1]).all()
            or not np.isin(t, [0, 1]).all()
            or not np.array_equal(a, a.T) or np.any(np.diag(a))):
        raise ValueError("Require binary finite T and an undirected loopless graph")
    a, t = a.astype(np.float64), t.astype(np.float64)
    degree = a.sum(1)
    if np.any(degree <= 0):
        raise ValueError("Require positive node degrees")
    e = a @ t / degree
    return _freeze(e), _freeze((e > CONFIG.exposure_threshold).astype(np.int64))


@dataclass(frozen=True, slots=True)
class ContextSupport:
    context_id: str
    source_input_hash: str
    center_t: int
    node_ids: np.ndarray
    population_support: np.ndarray
    design_support: np.ndarray
    realized_state_mask: np.ndarray
    observed_node_mask: np.ndarray
    empirical_support_mask: np.ndarray
    support_reason: np.ndarray
    full_graph_count: np.ndarray  # [group(center,leaf), t, s]
    observed_count: np.ndarray
    center_observed: bool
    empirical_extrapolation: np.ndarray
    context_design_extrapolation: np.ndarray
    arm_order: tuple[str, ...] = CONFIG.arm_order
    group_order: tuple[str, ...] = GROUP_ORDER

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, f.name, _freeze(value))


def support_for_task(task):
    """Validate canonical facts, then compute using A, T and observed indices."""
    validate_task(task)
    _, state = majority_exposure(task.adjacency, task.treatment)
    population = np.ones((200, 2, 2), dtype=bool)
    design = np.zeros_like(population)
    design[0, task.center_t, :] = True
    design[1:, :, task.center_t] = True
    realized = np.zeros_like(population)
    realized[np.arange(200), task.treatment.astype(np.int64), state] = True
    if np.any(realized & ~design):
        raise ValueError("Realized state contradicts the public context design")
    observed = np.zeros(200, dtype=bool)
    observed[task.observed_indices] = True
    empirical = realized & observed[:, None, None]

    reason = np.full(population.shape, REASONS[0], dtype="<U32")
    reason[design] = REASONS[1]
    reason[design & realized] = REASONS[2]
    reason[design & empirical] = REASONS[3]
    full_counts = np.stack((realized[:1].sum(0), realized[1:].sum(0))).astype(np.int64)
    obs_counts = np.stack((empirical[:1].sum(0), empirical[1:].sum(0))).astype(np.int64)
    return ContextSupport(
        task.context_id, task.input_hash, task.center_t, task.node_ids,
        population, design, realized, observed, empirical, reason, full_counts,
        obs_counts, bool(observed[0]), ~empirical, ~design)


@dataclass(frozen=True, slots=True)
class PairSupport:
    pair_id: str
    context_supports: tuple[ContextSupport, ContextSupport]
    node_ids: np.ndarray
    population_support: np.ndarray
    design_support: np.ndarray
    realized_state_mask: np.ndarray
    empirical_support_mask: np.ndarray
    empirical_extrapolation: np.ndarray
    observed_node_ids: np.ndarray
    n_unique_observed_nodes: int = 50
    n_factual_records: int = 100
    arm_order: tuple[str, ...] = CONFIG.arm_order

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, f.name, _freeze(value))


def support_for_pair(pair_task):
    """Keep both context counts; design union is NOT realized-state coverage."""
    validate_pair_task(pair_task)
    left, right = (support_for_task(t) for t in pair_task.context_refs)
    design = left.design_support | right.design_support
    realized = left.realized_state_mask | right.realized_state_mask
    empirical = left.empirical_support_mask | right.empirical_support_mask
    if not design.all() or not np.all(realized.sum((1, 2)) == 2):
        raise ValueError("Pair design/realized state invariant failed")
    return PairSupport(
        pair_task.pair_id, (left, right), left.node_ids,
        left.population_support & right.population_support,
        design, realized, empirical, ~empirical,
        pair_task.context_refs[0].observed_indices)


@dataclass(frozen=True, slots=True)
class MethodCapability:
    method_id: str
    display_name: str
    outputs: frozenset[str]
    privileged: bool = False
    reference_only: bool = False

    @property
    def ordinary_ranking(self):
        return not self.privileged and not self.reference_only


_CONTEXT_POINTS = frozenset(("context.mu", "context.ite", "context.ate", "pair.ate"))
_OUTPUTS = _CONTEXT_POINTS | frozenset(("context.ate_interval", "pair.ate_interval"))
ORDINARY_METHOD_IDS = ("state-mean", "ht", "f-z2", "lin-z2",
                       "kernel-old", "kernel-new", "j")
METHOD_CAPABILITIES = MappingProxyType({
    "state-mean": MethodCapability("state-mean", "State Mean", frozenset(("pair.ate",))),
    "ht": MethodCapability("ht", "HT", frozenset(("pair.ate",))),
    "f-z2": MethodCapability("f-z2", "F-z2", frozenset(("pair.ate",))),
    "lin-z2": MethodCapability("lin-z2", "Lin-z2", frozenset(("pair.ate",))),
    "kernel-old": MethodCapability("kernel-old", "kernel-old", _CONTEXT_POINTS),
    "kernel-new": MethodCapability("kernel-new", "kernel-new", _CONTEXT_POINTS),
    "j": MethodCapability("j", "iPFN-J", _CONTEXT_POINTS | frozenset(("context.ate_interval",))),
    "oracle-context": MethodCapability(
        "oracle-context", "Oracle-context",
        _CONTEXT_POINTS | frozenset(("context.ate_interval",)), True, True),
    "oracle-pair": MethodCapability(
        "oracle-pair", "Oracle-pair",
        frozenset(("pair.ate", "pair.ate_interval")), True, True),
})
_ALIASES = MappingProxyType({v.display_name: k for k, v in METHOD_CAPABILITIES.items()})


def method_capability(method_id):
    if type(method_id) is not str:
        raise TypeError("method_id must be a string")
    key = _ALIASES.get(method_id, method_id)
    if key not in METHOD_CAPABILITIES:
        raise ValueError("Unknown registered method")
    return METHOD_CAPABILITIES[key]


def output_capability(method_id, output, *, requested):
    """Declaration only: SUPPORTED is not an execution/output_status=OK."""
    if type(requested) is not bool:
        raise TypeError("requested must be bool")
    if type(output) is not str or output not in _OUTPUTS:
        raise ValueError("Unknown output capability")
    capability = method_capability(method_id)
    if not requested:
        return "NOT_REQUESTED"
    return "SUPPORTED" if output in capability.outputs else "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class PluginUseCheck:
    allowed: bool
    unsupported_queries: tuple[tuple[int, int, int], ...]
    design_extrapolation_queries: tuple[tuple[int, int, int], ...]
    extrapolation_authorized: bool


class PluginSupportError(ValueError):
    def __init__(self, check):
        super().__init__("Empirically unsupported queries cannot be ordinary plug-in inputs")
        self.check = check


def check_plugin_use(support, queries, *, extrapolation_authorized=False):
    """Pure gate: preserves predictions; authorization must come from outside."""
    if type(support) not in (ContextSupport, PairSupport):
        raise TypeError("Expected a Support diagnostic")
    if type(extrapolation_authorized) is not bool:
        raise TypeError("Explicit boolean extrapolation authorization required")
    if type(queries) is not tuple:
        raise TypeError("queries must be a tuple of (node,t,s) tuples")
    unsupported, design_extra = [], []
    for query in queries:
        if (type(query) is not tuple or len(query) != 3
                or any(type(v) is not int for v in query)):
            raise ValueError("Each query requires three Python integer indices")
        node, t, s = query
        if not 0 <= node < 200 or t not in (0, 1) or s not in (0, 1):
            raise ValueError("Query outside canonical node/state range")
        if not support.empirical_support_mask[node, t, s]:
            unsupported.append(query)
        if not support.design_support[node, t, s]:
            design_extra.append(query)
    return PluginUseCheck(not unsupported or extrapolation_authorized,
                          tuple(unsupported), tuple(design_extra),
                          extrapolation_authorized)


def require_plugin_support(support, queries, *, extrapolation_authorized=False):
    check = check_plugin_use(support, queries,
                             extrapolation_authorized=extrapolation_authorized)
    if not check.allowed:
        raise PluginSupportError(check)
    return check
