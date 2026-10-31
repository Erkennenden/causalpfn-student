"""Task views and five-array isolation, without estimation or Identification.

Hash schema: sorted compact ASCII JSON; length framing uses unsigned big-endian
64-bit lengths. Array headers name dtype/shape/C layout; data are little-endian,
C-order bytes. No repr(), Python hash(), RNG draws, persistence or predictions.
Task hashes include identity/link metadata and node_ids as well as five arrays.
"""
from dataclasses import dataclass, field
import hashlib
import json
import struct
from types import MappingProxyType

import numpy as np

from .config import CONFIG
from .dgp import star_adjacency
from .pair import PrivateContext, PrivatePair, validate_pair
from .rng import RNGIdentity

FACTUAL_KEYS = ("x", "adjacency", "treatment", "observed_indices", "outcomes_observed")
_ARRAY_ORDER = ("node_ids",) + FACTUAL_KEYS
_LAYOUT = {"node_ids": ((200,), "<i8"), "x": ((200, 20), "<f4"),
           "adjacency": ((200, 200), "<f8"), "treatment": ((200,), "<f8"),
           "observed_indices": ((50,), "<i8"), "outcomes_observed": ((50,), "<f8")}
_REFERENCE = "iid Bernoulli(0.5)"
_META = ("protocol_id", "protocol_version", "protocol_hash", "protocol_code", "root_seed",
         "cohort", "cohort_code", "mechanism_id", "pair_repeat_id", "center_t",
         "pair_id", "context_id", "target_id", "request_id", "reference_assignment",
         "arm_order", "rng_identity")
_CHANNELS = ("population", "x", "coefficient",
             "leaf_assignment", "noise", "observation_index")
_PROOF_TOKEN = object()


class ContractError(ValueError):
    """Rejected input evidence is diagnostic/private, never predictor payload."""
    def __init__(self, code, message, evidence):
        super().__init__(f"{code}: {message}")
        self.code, self.evidence = code, evidence


def _fail(code, message, evidence):
    raise ContractError(code, message, evidence)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _freeze(a):
    # Fresh exact-budget immutable bytes: .base cannot expose source full Y.
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _isolated(a):
    current = a
    while type(current) is np.ndarray:
        if current.nbytes != a.nbytes or current.flags.writeable:
            return False
        current = current.base
    return type(current) is bytes and len(current) == a.nbytes


def _array(name, a, evidence):
    shape, dtype = _LAYOUT[name]
    if type(a) is not np.ndarray or a.shape != shape or a.dtype.str != dtype:
        _fail("ARRAY", f"{name} requires plain ndarray {shape} {dtype}", evidence)
    if not np.isfinite(a).all():
        _fail("NONFINITE", f"{name} contains nonfinite values", evidence)


def _arrays(values, evidence, *, include_nodes=True):
    for name in (_ARRAY_ORDER if include_nodes else FACTUAL_KEYS):
        _array(name, values[name], evidence)
    if include_nodes and not np.array_equal(values["node_ids"], np.arange(200)):
        _fail("NODE_ORDER", "node_ids must be 0..199 in order", evidence)
    if not np.array_equal(values["adjacency"], star_adjacency()):
        _fail("TOPOLOGY", "Require the registered center-0 Star", evidence)
    if not np.isin(values["treatment"], [0, 1]).all():
        _fail("TREATMENT", "Treatment must be binary", evidence)
    ids = values["observed_indices"]
    if np.any(ids < 0) or np.any(ids >= 200) or len(np.unique(ids)) != 50:
        _fail("INDICES", "Require 50 unique ordered indices in 0..199", evidence)


def _digest(domain, metadata, arrays):
    headers = [dict(name=n, dtype=a.dtype.str, shape=list(a.shape), order="C",
                    nbytes=a.nbytes) for n, a in arrays]
    header = _json(dict(schema=domain, metadata=metadata, arrays=headers)).encode("ascii")
    digest = hashlib.sha256()
    for blob in [header] + [a.tobytes(order="C") for _, a in arrays]:
        digest.update(struct.pack(">Q", len(blob)))
        digest.update(blob)
    return digest.hexdigest()


def _fingerprint(name, a):
    return _digest("star-v2-array-v1", {}, [(name, a)])


def _identity(cohort, mid, repeat, evidence):
    if type(mid) is not int or type(repeat) is not int:
        _fail("IDENTITY", "IDs require Python int, excluding bool/float", evidence)
    try:
        mechanisms, repeats = CONFIG.cohort_limits(cohort)
    except (ValueError, TypeError) as error:
        _fail("IDENTITY", str(error), evidence)
    if not 0 <= mid < mechanisms or not 0 <= repeat < repeats:
        _fail("IDENTITY", "ID outside registered cohort", evidence)


def pair_id_for(cohort_code, mechanism_id, pair_repeat_id):
    _identity(cohort_code, mechanism_id, pair_repeat_id, None)
    identity = [CONFIG.protocol_id, cohort_code, mechanism_id, pair_repeat_id]
    return "star_v2_pair_" + hashlib.sha256(_json(identity).encode("ascii")).hexdigest()


def context_id_for(cohort_code, mechanism_id, pair_repeat_id, center_t):
    _identity(cohort_code, mechanism_id, pair_repeat_id, None)
    if type(center_t) is not int or center_t not in (0, 1):
        _fail("CENTER", "center_t must be integer 0/1", None)
    identity = [CONFIG.protocol_id, cohort_code, mechanism_id, pair_repeat_id, center_t]
    return "star_v2_context_" + hashlib.sha256(_json(identity).encode("ascii")).hexdigest()


def _rng(cohort, mid, repeat):
    return _json([RNGIdentity(cohort, mid, name, 0 if i < 3 else repeat).to_metadata()
                  for i, name in enumerate(_CHANNELS)])


def _links(target_id, request_id, evidence):
    if target_id is not None and (type(target_id) is not str or target_id != CONFIG.target_id):
        _fail("TARGET", "Only an explicitly supplied registered target is allowed", evidence)
    if request_id is not None and (type(request_id) is not str or not request_id.strip()
                                   or target_id is None):
        _fail("REQUEST_LINK", "A nonempty request_id requires an explicit target_id", evidence)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSpec:
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    protocol_code: int
    root_seed: int
    cohort: str
    cohort_code: int
    mechanism_id: int
    pair_repeat_id: int
    center_t: int
    pair_id: str
    context_id: str
    node_ids: np.ndarray
    x: np.ndarray
    adjacency: np.ndarray
    treatment: np.ndarray
    observed_indices: np.ndarray
    outcomes_observed: np.ndarray
    rng_identity: str  # Immutable canonical JSON for all six channel metadata records.
    input_hash: str
    target_id: str | None = None  # Unbound until explicitly supplied by upstream.
    request_id: str | None = None
    reference_assignment: str = _REFERENCE
    arm_order: tuple[str, ...] = CONFIG.arm_order

    @property
    def task_id(self):
        return self.context_id

    def __post_init__(self):
        _validate_task(self, storage=False)
        for name in _ARRAY_ORDER:
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        validate_task(self)


def _validate_task(task, *, storage):
    if type(task) is not TaskSpec:
        _fail("TASK_TYPE", "Only a TaskSpec can cross the extraction boundary", task)
    _identity(task.cohort_code, task.mechanism_id, task.pair_repeat_id, task)
    expected = {"protocol_id": CONFIG.protocol_id, "protocol_version": CONFIG.protocol_version,
                "protocol_hash": CONFIG.protocol_sha256, "protocol_code": CONFIG.protocol_code,
                "root_seed": CONFIG.root_seed, "reference_assignment": _REFERENCE,
                "arm_order": CONFIG.arm_order,
                "cohort": {1: "pilot", 2: "test"}[task.cohort_code]}
    for name, value in expected.items():
        actual = getattr(task, name)
        if type(actual) is not type(value) or actual != value:
            _fail("PROTOCOL", f"Invalid {name}", task)
    if type(task.center_t) is not int or task.center_t not in (0, 1):
        _fail("CENTER", "Invalid center identity", task)
    if task.pair_id != pair_id_for(task.cohort_code, task.mechanism_id, task.pair_repeat_id):
        _fail("IDENTITY", "pair_id mismatch", task)
    if task.context_id != context_id_for(task.cohort_code, task.mechanism_id,
                                         task.pair_repeat_id, task.center_t):
        _fail("IDENTITY", "context_id mismatch", task)
    _links(task.target_id, task.request_id, task)
    if type(task.rng_identity) is not str or task.rng_identity != _rng(
            task.cohort_code, task.mechanism_id, task.pair_repeat_id):
        _fail("RNG", "RNG metadata differs from registered six channels", task)
    values = {name: getattr(task, name) for name in _ARRAY_ORDER}
    _arrays(values, task)
    if task.treatment[0] != task.center_t:
        _fail("CENTER", "Center treatment does not match context identity", task)
    expected_hash = _digest("star-v2-task-input-v1",
                            {n: getattr(task, n) for n in _META}, list(values.items()))
    if type(task.input_hash) is not str or task.input_hash != expected_hash:
        _fail("HASH", "Task input_hash mismatch", task)
    if storage and any(not _isolated(a) for a in values.values()):
        _fail("STORAGE", "Task arrays must own detached immutable byte budgets", task)


def validate_task(task):
    _validate_task(task, storage=True)


def task_from_context(context, *, pair_repeat_id, rng_identity,
                      target_id=None, request_id=None):
    """Project observable data only; raw shared noise is NOT verified here."""
    if type(context) is not PrivateContext:
        _fail("CONTEXT_TYPE", "Expected a simulator PrivateContext", context)
    mechanism = context.mechanism
    cohort, mid = mechanism.cohort_code, mechanism.mechanism_id
    _identity(cohort, mid, pair_repeat_id, context)
    expected = tuple(RNGIdentity(cohort, mid, n, pair_repeat_id) for n in _CHANNELS[3:])
    if type(rng_identity) is not tuple or rng_identity != expected:
        _fail("RNG", "Explicit pair RNG identities are required", context)
    expected_mechanism = tuple(RNGIdentity(cohort, mid, n, 0) for n in _CHANNELS[:3])
    if mechanism.rng_identities != expected_mechanism:
        _fail("RNG", "Mechanism RNG identities conflict", context)
    fields = dict(protocol_id=CONFIG.protocol_id, protocol_version=CONFIG.protocol_version,
                  protocol_hash=CONFIG.protocol_sha256, protocol_code=CONFIG.protocol_code,
                  root_seed=CONFIG.root_seed, cohort={1: "pilot", 2: "test"}[cohort],
                  cohort_code=cohort, mechanism_id=mid, pair_repeat_id=pair_repeat_id,
                  center_t=context.center_t, pair_id=pair_id_for(cohort, mid, pair_repeat_id),
                  context_id=context_id_for(cohort, mid, pair_repeat_id, context.center_t),
                  target_id=target_id, request_id=request_id,
                  reference_assignment=_REFERENCE, arm_order=CONFIG.arm_order,
                  rng_identity=_rng(cohort, mid, pair_repeat_id))
    values = {"node_ids": mechanism.node_ids, "x": mechanism.x,
              "adjacency": mechanism.adjacency, "treatment": context.treatment,
              "observed_indices": context.observed_indices,
              "outcomes_observed": context.outcomes_observed}
    _arrays(values, context)
    full = context.full_outcomes
    if (type(full) is not np.ndarray or full.shape != (200,)
            or full.dtype.str != "<f8" or not np.isfinite(full).all()
            or not np.array_equal(context.outcomes_observed,
                                  full[context.observed_indices])):
        _fail("OBSERVATION_ALIGNMENT", "Context observed values do not match ordered full Y", context)
    if not np.array_equal(context.treatment[1:], context.leaf_treatment):
        _fail("TREATMENT", "Context leaf assignment conflicts with treatment", context)
    digest = _digest("star-v2-task-input-v1", {n: fields[n] for n in _META},
                     list(values.items()))
    return TaskSpec(**fields, **values, input_hash=digest)


def validate_predictor_inputs(inputs):
    if type(inputs) not in (dict, MappingProxyType) or set(inputs) != set(FACTUAL_KEYS):
        _fail("PREDICTOR_KEYS", "Exactly five factual keys; no task or metadata", inputs)
    _arrays(inputs, inputs, include_nodes=False)
    if any(not _isolated(inputs[name]) for name in FACTUAL_KEYS):
        _fail("STORAGE", "Predictor arrays must be detached immutable exact budgets", inputs)


def predictor_inputs(task):
    """Fresh five-key mapping, not a Task/PairTask or privileged storage view."""
    validate_task(task)
    result = MappingProxyType({name: _freeze(getattr(task, name)) for name in FACTUAL_KEYS})
    validate_predictor_inputs(result)
    return result


@dataclass(frozen=True, slots=True)
class _NoiseProof:
    pair_id: str
    context_hashes: tuple[str, str]
    noise_hash: str
    token: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self.token is not _PROOF_TOKEN:
            _fail("NOISE_PROVENANCE", "Only private simulator validation issues proof", None)


def _shared(tasks):
    left, right = tasks
    for name in ("x", "adjacency", "observed_indices"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            _fail("PAIR_SHARED", f"Pair differs in {name}", tasks)
    if not np.array_equal(left.treatment[1:], right.treatment[1:]):
        _fail("PAIR_SHARED", "Pair leaf treatment differs", tasks)
    return tuple((name, _fingerprint(name, value)) for name, value in (
        ("x", left.x), ("adjacency", left.adjacency),
        ("leaf_treatment", left.treatment[1:]),
        ("observed_indices", left.observed_indices)))


@dataclass(frozen=True, slots=True, kw_only=True)
class PairTaskSpec:
    context_refs: tuple[TaskSpec, TaskSpec]
    pair_id: str
    shared_input_hashes: tuple[tuple[str, str], ...]
    protocol_hash: str = CONFIG.protocol_sha256
    pair_design_id: str = "star_v2_fixed_center_0_1_shared_pair_v1"
    _noise_proof: _NoiseProof | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        validate_pair_task(self)

    @property
    def context_ids(self):
        return tuple(t.context_id for t in self.context_refs)

    @property
    def node_ids(self):
        return self.context_refs[0].node_ids

    @property
    def target_id(self):
        return self.context_refs[0].target_id

    @property
    def request_id(self):
        return self.context_refs[0].request_id

    @property
    def cohort_code(self):
        return self.context_refs[0].cohort_code

    @property
    def cohort(self):
        return self.context_refs[0].cohort

    @property
    def mechanism_id(self):
        return self.context_refs[0].mechanism_id

    @property
    def pair_repeat_id(self):
        return self.context_refs[0].pair_repeat_id

    @property
    def reference_assignment(self):
        return self.context_refs[0].reference_assignment

    @property
    def arm_order(self):
        return self.context_refs[0].arm_order

    @property
    def rng_identity(self):
        return self.context_refs[0].rng_identity

    @property
    def n_unique_observed_nodes(self):
        return 50

    @property
    def n_factual_records(self):
        return 100

    @property
    def noise_verification(self):
        return ("VERIFIED_SIMULATOR_PRIVATE" if self._noise_proof is not None
                else "NOT_VERIFIED_FROM_FACTUAL_INPUTS")


def validate_pair_task(pair):
    if type(pair) is not PairTaskSpec or type(pair.context_refs) is not tuple or len(pair.context_refs) != 2:
        _fail("PAIR_CONTEXTS", "Exactly two TaskSpec references required", pair)
    left, right = pair.context_refs
    validate_task(left)
    validate_task(right)
    if (left.center_t, right.center_t) != (0, 1):
        _fail("PAIR_CONTEXTS", "Require ordered, unique center_t=0/1", pair)
    for name in ("pair_id", "protocol_hash", "cohort_code", "mechanism_id", "pair_repeat_id",
                 "target_id", "request_id", "reference_assignment", "arm_order", "rng_identity"):
        if getattr(left, name) != getattr(right, name):
            _fail("PAIR_IDENTITY", f"Pair identity conflicts in {name}", pair)
    if pair.pair_id != left.pair_id or pair.protocol_hash != CONFIG.protocol_sha256:
        _fail("PAIR_IDENTITY", "Pair/protocol ID mismatch", pair)
    if pair.pair_design_id != "star_v2_fixed_center_0_1_shared_pair_v1":
        _fail("PAIR_IDENTITY", "Unregistered engineering design label", pair)
    expected = _shared(pair.context_refs)
    proof = pair._noise_proof
    if proof is not None:
        if (type(proof) is not _NoiseProof or proof.token is not _PROOF_TOKEN
                or proof.pair_id != pair.pair_id
                or proof.context_hashes != (left.input_hash, right.input_hash)
                or type(proof.noise_hash) is not str or len(proof.noise_hash) != 64
                or any(c not in "0123456789abcdef" for c in proof.noise_hash)):
            _fail("NOISE_PROVENANCE", "Noise evidence not bound to these task inputs", pair)
        expected += (("raw_noise", proof.noise_hash),)
    if type(pair.shared_input_hashes) is not tuple or pair.shared_input_hashes != expected:
        _fail("PAIR_HASH", "Shared-input evidence mismatch", pair)


def pair_from_tasks(context_refs):
    """Observable consistency only: no raw-noise verification claim."""
    if type(context_refs) is not tuple or len(context_refs) != 2:
        _fail("PAIR_CONTEXTS", "Expected exactly two task references", context_refs)
    for task in context_refs:
        validate_task(task)
    return PairTaskSpec(context_refs=context_refs, pair_id=context_refs[0].pair_id,
                        shared_input_hashes=_shared(context_refs))


def pair_from_private(private_pair, *, target_id=None, request_id=None):
    """Verify shared raw noise in simulator, retain ONLY its hash/provenance."""
    if type(private_pair) is not PrivatePair:
        _fail("PAIR_TYPE", "Expected simulator PrivatePair", private_pair)
    validate_pair(private_pair)
    tasks = tuple(task_from_context(
        c, pair_repeat_id=private_pair.pair_repeat_id,
        rng_identity=private_pair.rng_identities,
        target_id=target_id, request_id=request_id) for c in private_pair.contexts)
    pid = tasks[0].pair_id
    noise_hash = _fingerprint("raw_noise", private_pair.noise)
    proof = _NoiseProof(pid, (tasks[0].input_hash, tasks[1].input_hash),
                        noise_hash, _PROOF_TOKEN)
    return PairTaskSpec(context_refs=tasks, pair_id=pid,
                        shared_input_hashes=_shared(tasks) + (("raw_noise", noise_hash),),
                        _noise_proof=proof)
