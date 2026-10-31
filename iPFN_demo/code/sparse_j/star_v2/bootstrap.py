"""Mechanism-block Bootstrap, frozen 6.3. In-memory evaluator evidence only.

The sole random stream draws indices, never tasks or outcomes. Original task
identities are retained; repeated mechanisms count at distinct draw_positions.
Synthetic entry points require explicit indices and never claim formal evidence.
"""
from dataclasses import dataclass, replace
from itertools import combinations
from collections.abc import Mapping

import numpy as np

from .rng import bootstrap_identity, make_generator
from .config import CONFIG
from .task_registry import TaskRegistry, DataTask, ExecutionRequest, Attempt
from .metrics import (EvaluationPlan, ExpectedTask, Prediction, MechanismTruth,
                      pair_ate_metrics, _support_index)
from .support import ContextSupport, ORDINARY_METHOD_IDS
from .estimation import EFFECT_ORDER, frozen_array
from .coverage import check, freeze, unique_attempts, evaluate_j_coverage
from .ranking import evaluate_rankability, CONTEXT_METHODS

METHOD_ORDER = ORDINARY_METHOD_IDS
COVERAGE_SCOPES = ("overall", "center_t=0", "center_t=1")


@dataclass(frozen=True, slots=True)
class BootstrapDraws:
    indices: np.ndarray
    replicate_ids: np.ndarray
    rng_metadata: Mapping
    numpy_version: str

    def __post_init__(self):
        check(type(self.indices) is np.ndarray and self.indices.dtype == np.dtype("int64")
              and self.indices.shape == (10000, 200), "Frozen Bootstrap shape/dtype required")
        check(np.all((self.indices >= 0) & (self.indices < 200)), "Index out of range")
        check(type(self.replicate_ids) is np.ndarray
              and self.replicate_ids.dtype == np.dtype("int64")
              and np.array_equal(self.replicate_ids, np.arange(10000)), "Replicate IDs must be 0..9999")
        check(tuple(self.rng_metadata["entropy"]) == (20261030, 2026103001, 2, 0, 7, 0)
              and self.rng_metadata["bit_generator"] == "PCG64", "Wrong Bootstrap RNG")
        object.__setattr__(self, "indices", frozen_array(self.indices))
        object.__setattr__(self, "replicate_ids", frozen_array(self.replicate_ids))
        object.__setattr__(self, "rng_metadata", freeze(self.rng_metadata))


def generate_bootstrap_indices() -> BootstrapDraws:
    """One Generator, 10000 consecutive draws of 200 indices, int64."""
    identity = bootstrap_identity()
    generator = make_generator(identity)  # exactly once; no reseeding/spawn
    indices = np.empty((10000, 200), dtype=np.int64)
    for replicate in range(10000):
        indices[replicate] = generator.integers(0, 200, size=200, dtype=np.int64)
    return BootstrapDraws(indices, np.arange(10000, dtype=np.int64),
                          identity.to_metadata(), np.__version__)


@dataclass(frozen=True, slots=True)
class PairBlock:
    pair_repeat_id: int
    pair_task: DataTask
    context_tasks: tuple[DataTask, DataTask]
    requests: tuple[ExecutionRequest, ...]
    predictions: tuple[Prediction, ...]
    support: tuple[ContextSupport, ContextSupport]
    j_interval_records: tuple
    attempt_history: tuple[Attempt, ...]


@dataclass(frozen=True, slots=True)
class MechanismBlock:
    mechanism_id: int
    pairs: tuple[PairBlock, ...]
    target: MechanismTruth
    method_mse: np.ndarray  # [method, effect]; unavailable blocks remain NaN
    coverage_counts: np.ndarray  # [overall/center0/center1, effect], failures uncovered

    def __post_init__(self):
        object.__setattr__(self, "method_mse", frozen_array(self.method_mse))
        object.__setattr__(self, "coverage_counts", frozen_array(self.coverage_counts))


@dataclass(frozen=True, slots=True)
class MechanismBlocks:
    blocks: tuple[MechanismBlock, ...]
    registry_seal: str
    rankability: Mapping
    coverage: Mapping
    point_metrics: Mapping
    synthetic: bool


def _layout(registry, *, formal):
    """Check registered data and request grid, independently of execution success."""
    tasks = {k: v.document for k, v in registry.tasks.items()}
    pairs = [v for v in tasks.values() if v["scope"] == "pair"]
    mechanisms = sorted({v["identity"]["mechanism_id"] for v in pairs})
    check(mechanisms and (not formal or mechanisms == list(range(200))),
          "Formal blocks require mechanisms 0..199")
    parents_by_slot = {}
    children_by_slot = {}
    for key, execution in registry.requests.items():
        request = execution.document["request"]
        if "ate" not in request["requested_outputs"]:
            continue
        if key[1] == "pair":
            parents_by_slot.setdefault((key[0], key[2]), []).append((key, request))
        else:
            children_by_slot.setdefault((key[0], key[2], request["parent_request_id"]), []).append(request)
    expected_ids = set()
    layout = {}
    for m in mechanisms:
        group = [v for v in pairs if v["identity"]["mechanism_id"] == m]
        check(len(group) == 3 and {v["identity"]["pair_repeat_id"] for v in group} == {0, 1, 2},
              "Missing/duplicate Pair in mechanism block")
        layout[m] = tuple(sorted(group, key=lambda v: v["identity"]["pair_repeat_id"]))
        for pair in layout[m]:
            check(not formal or pair["identity"]["cohort_code"] == CONFIG.test_cohort_code,
                  "Formal blocks require Test cohort")
            expected_ids.add(pair["task_id"])
            for center, cid in enumerate(pair["context_ids"]):
                check(cid in tasks and tasks[cid]["scope"] == "context"
                      and tasks[cid]["pair_id"] == pair["pair_id"]
                      and tasks[cid]["identity"]["center_t"] == [center]
                      and tasks[cid]["input_hashes"] == [pair["input_hashes"][center]],
                      "Missing or inconsistent Context in block")
                expected_ids.add(cid)
            for method in METHOD_ORDER:
                parents = parents_by_slot.get((method, pair["pair_id"]), [])
                check(len(parents) == 1, "Missing/duplicate ordinary Pair request")
                key, parent = parents[0]
                if method in CONTEXT_METHODS:
                    for cid in pair["context_ids"]:
                        children = children_by_slot.get((method, cid, key[3]), [])
                        check(len(children) == 1, "Missing/duplicate necessary Context request")
                        check(method != "j" or "interval" in children[0]["requested_outputs"],
                              "Missing pre-requested J interval")
    check(set(tasks) == expected_ids, "Extra data tasks outside mechanism blocks")
    return tasks, layout


def _supports_for_blocks(registry, layout, supports):
    """Reuse the 6.2 Support validator, not exposure/design reconstruction.

    The local ExpectedTask view is solely a Support validation schema. It is
    not an EstimationRequest and does not add a requested model capability.
    """
    expected_contexts = {cid for group in layout.values() for pair in group
                         for cid in pair["context_ids"]}
    supplied = {}
    for cid, support in supports:
        check(cid in expected_contexts and cid not in supplied,
              "Extra/duplicate Support context")
        check(type(support) is ContextSupport, "Existing ContextSupport required")
        supplied[cid] = replace(support)  # detached immutable arrays
    check(set(supplied) == expected_contexts, "Missing Context Support")
    validation_tasks = []
    support_records = []
    for group in layout.values():
        for pair in group:
            for center, cid in enumerate(pair["context_ids"]):
                key = next(key for key, execution in registry.requests.items()
                           if key[:3] == ("j", "context", cid)
                           and "ate" in execution.document["request"]["requested_outputs"])
                validation_tasks.append(ExpectedTask(key, pair["pair_id"],
                    pair["identity"]["mechanism_id"], pair["identity"]["pair_repeat_id"],
                    center, (pair["input_hashes"][center],), tuple(pair["node_ids"]), ("mu",)))
                support_records.append((key, supplied[cid]))
    plan = EvaluationPlan(tuple(validation_tasks), source_seal=registry.seal_sha256)
    _support_index(plan, support_records)  # existing shape/binding/partition/count rules
    for support in supplied.values():
        check(np.array_equal(support.empirical_extrapolation, ~support.empirical_support_mask)
              and np.array_equal(support.context_design_extrapolation, ~support.design_support),
              "Support extrapolation flags conflict")
    return supplied


def build_mechanism_blocks(registry, predictions, truths, supports, *, formal=True) -> MechanismBlocks:
    """Snapshot sealed data/requests/results/Support and EVERY Attempt.

    supports is an iterable of (context_id, ContextSupport), one per data
    Context independent of method. Missing outputs/Attempts remain failures;
    missing pre-requested data, requests or Support is an inventory error.
    """
    check(type(formal) is bool, "Explicit boolean formal flag required")
    check(type(registry) is TaskRegistry and registry.sealed, "Sealed registry required")
    # Includes Oracle history too: there is no implicit authority policy.
    unique_attempts(registry, registry.requests)
    tasks, layout = _layout(registry, formal=formal)
    support_map = _supports_for_blocks(registry, layout, tuple(supports))
    predictions, truths = tuple(predictions), tuple(truths)
    plan = EvaluationPlan.from_registry(registry)
    point = pair_ate_metrics(plan, predictions, truths)  # validates ALL provided bindings
    rankability = evaluate_rankability(registry,
        tuple(p for p in predictions if p.key[1] == "pair"), truths, formal=formal)
    coverage = evaluate_j_coverage(registry, truths, formal=formal)
    truth_map = {t.mechanism_id: t for t in truths}
    # Keep ordinary point metrics consistent with necessary-arm qualification.
    point = freeze({"ordinary": {method: {effect:
        rankability["methods"][method]["effects"][effect]["statistics"]
        for effect in EFFECT_ORDER} for method in METHOD_ORDER},
        "oracle_reference": point["oracle_reference"]})
    found = {p.key: p for p in predictions}
    requests_by_pair = {}
    for execution in registry.requests.values():
        requests_by_pair.setdefault(execution.document["request"]["pair_id"], []).append(execution)
    history_by_pair = {}
    for attempt in registry.attempts:
        pid = tasks[attempt.execution_key[2]]["pair_id"]
        history_by_pair.setdefault(pid, []).append(attempt)
    blocks = []
    for m, group in layout.items():
        pair_blocks = []
        mse = np.full((7, 3), np.nan, dtype=np.float64)
        for pair in group:
            pid = pair["pair_id"]
            reqs = tuple(requests_by_pair.get(pid, ()))
            keys = {r.key for r in reqs}
            pair_blocks.append(PairBlock(pair["identity"]["pair_repeat_id"],
                registry.tasks[pid], tuple(registry.tasks[cid] for cid in pair["context_ids"]),
                reqs, tuple(found[k] for k in sorted(keys) if k in found),
                tuple(support_map[cid] for cid in pair["context_ids"]),
                tuple(row for row in coverage["records"] if row["pair_id"] == pid),
                tuple(history_by_pair.get(pid, ()))))
        for j, method in enumerate(METHOD_ORDER):
            for k, effect in enumerate(EFFECT_ORDER):
                qualified = rankability["methods"][method]["effects"][effect]
                if qualified["rankability_status"] == "RANKABLE":
                    # Use the existing metric's per-mechanism MSE, never regroup
                    # repeated Bootstrap draws by the original mechanism ID.
                    row = next(r for r in qualified["statistics"]["per_mechanism"]
                               if r["mechanism_id"] == m)
                    mse[j, k] = row["mse"]
        counts = np.zeros((3, 3), dtype=np.int64)
        for k, effect in enumerate(EFFECT_ORDER):
            rows = [r for r in coverage["records"] if r["mechanism_id"] == m and r["effect"] == effect]
            check(len(rows) == 6 and all(sum(r["center_t"] == t for r in rows) == 3 for t in (0, 1)),
                  "Incomplete J interval block")
            counts[0, k] = sum(r["covered"] for r in rows)
            counts[1, k] = sum(r["covered"] for r in rows if r["center_t"] == 0)
            counts[2, k] = sum(r["covered"] for r in rows if r["center_t"] == 1)
        blocks.append(MechanismBlock(m, tuple(pair_blocks), truth_map[m], mse, counts))
    return MechanismBlocks(tuple(blocks), registry.seal_sha256, rankability,
                           coverage, point, not formal)


@dataclass(frozen=True, slots=True)
class BootstrapDistribution:
    status: str  # OK or NOT_COMPARABLE
    values: np.ndarray | None
    interval: tuple[float, float] | None
    failure_reasons: Mapping

    def __post_init__(self):
        check(self.status in ("OK", "NOT_COMPARABLE"), "Invalid Bootstrap status")
        if self.status == "OK":
            check(type(self.values) is np.ndarray and self.values.dtype == np.dtype("float64")
                  and self.values.ndim == 1 and len(self.values) > 0
                  and np.isfinite(self.values).all(), "Invalid Bootstrap distribution")
            endpoints = tuple(float(v) for v in np.quantile(
                self.values, [0.025, 0.975], method="linear"))
            check(self.interval == endpoints, "Percentile interval mismatch")
            object.__setattr__(self, "values", frozen_array(self.values))
        else:
            check(self.values is None and self.interval is None and self.failure_reasons,
                  "NOT_COMPARABLE must retain reasons, without success-subset values")
        object.__setattr__(self, "failure_reasons", freeze(self.failure_reasons))


def _distribution(values, reasons=None):
    if values is None:
        return BootstrapDistribution("NOT_COMPARABLE", None, None, reasons)
    values = np.asarray(values, dtype=np.float64)
    bounds = tuple(float(v) for v in np.quantile(values, [0.025, 0.975], method="linear"))
    return BootstrapDistribution("OK", values, bounds, {})


@dataclass(frozen=True, slots=True)
class MethodComparison:
    method_a: str
    method_b: str
    effects: Mapping


@dataclass(frozen=True, slots=True)
class BootstrapEvidence:
    mechanism_blocks: tuple[MechanismBlock, ...]
    replicate_ids: np.ndarray
    indices: np.ndarray
    draw_positions: np.ndarray
    rng_metadata: Mapping
    numpy_version: str
    rankability: Mapping
    method_rmse: np.ndarray  # [replicate, method, effect]; not-rankable columns all NaN
    comparisons: tuple[MethodComparison, ...]
    j_coverage: Mapping
    point_metrics: Mapping
    registry_seal: str
    evidence_scope: str
    method_order: tuple = METHOD_ORDER
    effect_order: tuple = EFFECT_ORDER
    interval_level: float = .95
    interval_interpretation: str = "pointwise percentile; linear quantile; no multiplicity adjustment"

    def __post_init__(self):
        for name in ("replicate_ids", "indices", "draw_positions", "method_rmse"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        for name in ("rng_metadata", "rankability", "j_coverage", "point_metrics"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


def _resample(collection, indices, rng_metadata):
    check(type(collection) is MechanismBlocks and collection.blocks, "MechanismBlocks required")
    m = len(collection.blocks)
    check(type(indices) is np.ndarray and indices.dtype == np.dtype("int64")
          and indices.ndim == 2 and indices.shape[1] == m and indices.shape[0] > 0
          and np.all((indices >= 0) & (indices < m)), "Invalid draw-position matrix")
    mse = np.stack([block.method_mse for block in collection.blocks])
    counts = np.stack([block.coverage_counts for block in collection.blocks])
    # Second axis is draw_position: repeats intentionally remain separate. Sum
    # in chunks to avoid a large [10000,200,7,3] intermediate.
    rmse = np.empty((len(indices), 7, 3), dtype=np.float64)
    coverage = np.empty((len(indices), 3, 3), dtype=np.float64)
    for start in range(0, len(indices), 128):
        drawn = indices[start:start+128]
        rmse[start:start+128] = np.sqrt(mse[drawn].mean(axis=1))
        coverage[start:start+128] = counts[drawn].mean(axis=1) / np.array([6., 3., 3.])[:, None]
    comparisons = []
    for a, b in combinations(range(7), 2):
        effects = {}
        for k, effect in enumerate(EFFECT_ORDER):
            reasons = {method: collection.rankability["methods"][method]["effects"][effect]
                       for method in (METHOD_ORDER[a], METHOD_ORDER[b])
                       if collection.rankability["methods"][method]["effects"][effect][
                           "rankability_status"] != "RANKABLE"}
            effects[effect] = (_distribution(None, reasons) if reasons else
                               _distribution(rmse[:, a, k] - rmse[:, b, k]))
        comparisons.append(MethodComparison(METHOD_ORDER[a], METHOD_ORDER[b], freeze(effects)))
    coverage_out = {scope: {effect: _distribution(coverage[:, s, k])
                           for k, effect in enumerate(EFFECT_ORDER)}
                    for s, scope in enumerate(COVERAGE_SCOPES)}
    return BootstrapEvidence(collection.blocks, np.arange(len(indices), dtype=np.int64),
        indices, np.arange(m, dtype=np.int64), rng_metadata, np.__version__,
        collection.rankability, rmse, tuple(comparisons), coverage_out,
        collection.point_metrics, collection.registry_seal,
        "SYNTHETIC" if collection.synthetic else "FORMAL_TEST_PLAN")


def bootstrap_from_synthetic_indices(collection, indices) -> BootstrapEvidence:
    """Deterministic test-only math entry: explicit draws, no RNG or formal claim."""
    check(type(collection) is MechanismBlocks and collection.synthetic,
          "Explicit synthetic mechanism blocks required")
    return _resample(collection, indices, {"source": "explicit_synthetic_indices", "rng_used": False})


def mechanism_block_bootstrap(registry, predictions, truths, supports) -> BootstrapEvidence:
    """Formal entry: validate complete frozen plan BEFORE consuming Bootstrap RNG."""
    collection = build_mechanism_blocks(registry, predictions, truths, supports)
    draws = generate_bootstrap_indices()
    return _resample(collection, draws.indices, draws.rng_metadata)
