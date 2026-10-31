"""Private Star pair simulation; no predictor API, persistence or cohort runner.

Sampling APIs and observation order are fixed here. There are exactly three
pair-level draws, each made once. Validation never resamples or repairs data.
Formal Pilot/Test calls require separate authorization.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .config import CONFIG
from .dgp import SimulatorMechanism, array_sha256, star_adjacency
from .rng import CHANNELS, RNGIdentity, make_generator, make_identity

_PAIR_CHANNELS = ("leaf_assignment", "noise", "observation_index")
_SAMPLING_APIS = {
    "leaf_assignment": "Generator.binomial(1, 0.5, size=199)",
    "noise": "Generator.normal(0.0, noise_sd, size=200)",
    "observation_index": "Generator.choice(200, size=50, replace=False)",
}


class PairValidationError(ValueError):
    """Diagnostics retain the rejected pair or raw evidence, without repairs."""
    def __init__(self, code, message, evidence):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.evidence = evidence


def _fail(code, message, evidence):
    raise PairValidationError(code, message, evidence)


def _array(value, shape, name, evidence, *, dtype=None, integer=False, binary=False):
    if not isinstance(value, np.ndarray):
        _fail("ARRAY_TYPE", f"{name} must be a NumPy array", evidence)
    if value.shape != shape:
        _fail("SHAPE", f"{name}: expected {shape}, got {value.shape}", evidence)
    if dtype is not None and value.dtype != np.dtype(dtype):
        _fail("DTYPE", f"{name} must have dtype {np.dtype(dtype)}", evidence)
    if integer and value.dtype.kind not in "iu":
        _fail("DTYPE", f"{name} must have integer dtype", evidence)
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        _fail("NONFINITE", f"{name} must contain finite real values", evidence)
    if binary and not np.isin(value, [0, 1]).all():
        _fail("TREATMENT", f"{name} must contain only 0 and 1", evidence)
    return value


def _readonly(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _limits(cohort_code, mechanism_id, pair_repeat_id, evidence):
    if type(mechanism_id) is not int or type(pair_repeat_id) is not int:
        _fail("IDENTITY", "mechanism/repeat IDs must be Python ints", evidence)
    try:
        mechanisms, repeats = CONFIG.cohort_limits(cohort_code)
    except (TypeError, ValueError) as error:
        _fail("IDENTITY", str(error), evidence)
    if not 0 <= mechanism_id < mechanisms or not 0 <= pair_repeat_id < repeats:
        _fail("IDENTITY", "mechanism/repeat ID outside registered cohort", evidence)


def _check_identities(identities, channels, cohort, mechanism_id, repeat, evidence):
    if type(identities) is not tuple or len(identities) != len(channels):
        _fail("IDENTITY", "Wrong RNG identity collection", evidence)
    for identity, channel in zip(identities, channels):
        expected = (CONFIG.root_seed, CONFIG.protocol_code, cohort,
                    mechanism_id, CHANNELS[channel].code, repeat)
        if (type(identity) is not RNGIdentity or identity.channel != channel
                or identity.entropy != expected):
            _fail("IDENTITY", f"Wrong {channel} RNG identity", evidence)


def _check_mechanism(mechanism, evidence):
    if type(mechanism) is not SimulatorMechanism:
        _fail("MECHANISM", "Expected the accepted simulator-private mechanism", evidence)
    _limits(mechanism.cohort_code, mechanism.mechanism_id, 0, evidence)
    _check_identities(mechanism.rng_identities,
                      ("population", "x", "coefficient"),
                      mechanism.cohort_code, mechanism.mechanism_id, 0, evidence)
    shapes = {
        "node_ids": ((200,), np.int64), "population_mean": ((20,), np.float64),
        "x": ((200, 20), np.float32), "adjacency": ((200, 200), np.float64),
        "projection": ((20, 2), np.float64), "h": ((200, 3), np.float64),
        "coefficients": ((25,), np.float64), "components": ((200, 4), np.float64),
        "mu": ((200, 2, 2), np.float64), "ite": ((200, 3), np.float64),
        "ate": ((3,), np.float64),
    }
    for name, (shape, dtype) in shapes.items():
        value = _array(getattr(mechanism, name), shape, "mechanism."+name,
                       evidence, dtype=dtype)
        if value.flags.writeable:
            _fail("MUTABLE", f"mechanism.{name} must be read-only", evidence)
    if not np.array_equal(mechanism.node_ids, np.arange(CONFIG.N)):
        _fail("NODE_ORDER", "Mechanism node order is not 0..199", evidence)
    if not np.array_equal(mechanism.adjacency, star_adjacency()):
        _fail("TOPOLOGY", "Mechanism is not the registered Star", evidence)


def _indices(value, evidence):
    ids = _array(value, (CONFIG.M,), "observed_indices", evidence, integer=True)
    if np.any(ids < 0) or np.any(ids >= CONFIG.N) or len(np.unique(ids)) != CONFIG.M:
        _fail("OBSERVATION_INDEX", "Require 50 unique indices in 0..199", evidence)
    return ids


@dataclass(frozen=True, slots=True)
class PrivateContext:
    """Privileged simulator record; never pass this object to a predictor."""
    mechanism: SimulatorMechanism
    center_t: int
    leaf_treatment: np.ndarray
    noise: np.ndarray
    observed_indices: np.ndarray
    treatment: np.ndarray
    structural_mean: np.ndarray
    full_outcomes: np.ndarray
    outcomes_observed: np.ndarray


@dataclass(frozen=True, slots=True)
class PrivatePair:
    """One mechanism, one repeat and exactly two private contexts."""
    cohort_code: int
    mechanism_id: int
    pair_repeat_id: int
    mechanism: SimulatorMechanism
    rng_identities: tuple[RNGIdentity, ...]
    leaf_treatment: np.ndarray
    noise: np.ndarray
    observed_indices: np.ndarray
    contexts: tuple[PrivateContext, PrivateContext]

    def difference_diagnostics(self):
        """Floating-point diagnostic only; not an Oracle/SVD acceptance test."""
        left, right = self.contexts
        residual = ((right.full_outcomes-left.full_outcomes)
                    - (right.structural_mean-left.structural_mean))
        return {
            "full_max_abs_residual": float(np.max(np.abs(residual))),
            "observed_max_abs_residual": float(
                np.max(np.abs(residual[self.observed_indices]))),
        }

    def metadata(self):
        validation = validate_pair(self)
        arrays = {"leaf_treatment": self.leaf_treatment,
                  "noise": self.noise, "observed_indices": self.observed_indices}
        for context in self.contexts:
            for name in ("treatment", "structural_mean", "full_outcomes", "outcomes_observed"):
                arrays[f"center_t{context.center_t}.{name}"] = getattr(context, name)
        return {
            "access": "simulator_evaluator_only",
            "protocol_id": CONFIG.protocol_id,
            "protocol_sha256": CONFIG.protocol_sha256,
            "cohort_code": self.cohort_code,
            "mechanism_id": self.mechanism_id,
            "pair_repeat_id": self.pair_repeat_id,
            "center_order": [0, 1],
            "rng_identities": [i.to_metadata() for i in self.rng_identities],
            "sampling_apis": dict(_SAMPLING_APIS),
            "observation_order": "Generator.choice return order; never sorted",
            "array_shapes": {name: list(a.shape) for name, a in arrays.items()},
            "array_dtypes": {name: a.dtype.str for name, a in arrays.items()},
            "array_sha256": {name: array_sha256(a) for name, a in arrays.items()},
            "shared_mechanism_sha256": {
                name: array_sha256(getattr(self.mechanism, name))
                for name in ("x", "adjacency", "h", "coefficients", "mu")},
            "validation": validation,
            "difference_diagnostics": self.difference_diagnostics(),
        }


def validate_pair(pair):
    """Reject inconsistent records in place; never copy, reorder or resample."""
    evidence = pair
    if type(pair) is not PrivatePair:
        _fail("PAIR_TYPE", "Expected a private pair", evidence)
    _check_mechanism(pair.mechanism, evidence)
    _limits(pair.cohort_code, pair.mechanism_id, pair.pair_repeat_id, evidence)
    if (pair.cohort_code, pair.mechanism_id) != (
            pair.mechanism.cohort_code, pair.mechanism.mechanism_id):
        _fail("IDENTITY", "Pair identity differs from its mechanism", evidence)
    _check_identities(pair.rng_identities, _PAIR_CHANNELS, pair.cohort_code,
                      pair.mechanism_id, pair.pair_repeat_id, evidence)
    _array(pair.leaf_treatment, (199,), "leaf_treatment", evidence,
           dtype=np.float64, binary=True)
    _array(pair.noise, (200,), "noise", evidence, dtype=np.float64)
    _indices(pair.observed_indices, evidence)
    if type(pair.contexts) is not tuple or len(pair.contexts) != 2:
        _fail("CONTEXTS", "Exactly two contexts are required", evidence)
    for center_t, context in enumerate(pair.contexts):
        if type(context) is not PrivateContext:
            _fail("CONTEXT_TYPE", "Expected a private context", evidence)
        if type(context.center_t) is not int or context.center_t != center_t:
            _fail("CENTER_ORDER", "Contexts must be ordered center_t=0,1", evidence)
        for name in ("mechanism", "leaf_treatment", "noise", "observed_indices"):
            if getattr(context, name) is not getattr(pair, name):
                _fail("SHARED_REFERENCE", f"Context does not share {name}", evidence)
        for name, shape in (("treatment", (200,)), ("structural_mean", (200,)),
                            ("full_outcomes", (200,)), ("outcomes_observed", (50,))):
            _array(getattr(context, name), shape, name, evidence, dtype=np.float64,
                   binary=(name == "treatment"))
        if (context.treatment[0] != center_t
                or not np.array_equal(context.treatment[1:], pair.leaf_treatment)):
            _fail("TREATMENT", "Only node 0 may change between arms", evidence)
        expected_mean = pair.mechanism.structural_mean(context.treatment)
        if not np.array_equal(context.structural_mean, expected_mean):
            _fail("STRUCTURAL_MEAN", "Stored mean differs from fixed DGP", evidence)
        if not np.array_equal(context.full_outcomes,
                              context.structural_mean + pair.noise):
            _fail("FACTUAL_OUTCOME", "Outcomes do not use the one shared noise", evidence)
        if not np.array_equal(context.outcomes_observed,
                              context.full_outcomes[pair.observed_indices]):
            _fail("OBSERVATION_ALIGNMENT", "Observed outcomes do not match ordered indices", evidence)
        for name in ("leaf_treatment", "noise", "observed_indices", "treatment",
                     "structural_mean", "full_outcomes", "outcomes_observed"):
            if getattr(context, name).flags.writeable:
                _fail("MUTABLE", f"Context {name} must be read-only", evidence)
    return {"status": "OK", "n_contexts": 2,
            "n_unique_observed_nodes": 50, "n_factual_records": 100}


def generate_pair(mechanism: SimulatorMechanism, *, pair_repeat_id: int):
    """Draw three pair channels once; reuse the supplied immutable mechanism.

    No seed override, mechanism generation, public TaskSpec or persistence.
    """
    raw = {"mechanism": mechanism, "pair_repeat_id": pair_repeat_id}
    evidence = MappingProxyType(raw)
    _check_mechanism(mechanism, evidence)
    _limits(mechanism.cohort_code, mechanism.mechanism_id, pair_repeat_id, evidence)
    identities = tuple(make_identity(
        cohort_code=mechanism.cohort_code, mechanism_id=mechanism.mechanism_id,
        channel=channel, pair_repeat_id=pair_repeat_id) for channel in _PAIR_CHANNELS)
    raw["rng_identities"] = identities

    leaf_raw = make_generator(identities[0]).binomial(
        1, CONFIG.leaf_treatment_probability, size=CONFIG.N-1)
    raw["leaf_treatment"] = leaf_raw
    _array(leaf_raw, (199,), "leaf draw", evidence, integer=True, binary=True)
    leaf = _readonly(leaf_raw.astype(np.float64))

    noise_raw = make_generator(identities[1]).normal(0.0, CONFIG.noise_sd, size=CONFIG.N)
    raw["noise"] = noise_raw
    _array(noise_raw, (200,), "noise draw", evidence, dtype=np.float64)
    noise = _readonly(noise_raw)

    indices_raw = make_generator(identities[2]).choice(CONFIG.N, size=CONFIG.M, replace=False)
    raw["observed_indices"] = indices_raw
    _indices(indices_raw, evidence)
    indices = _readonly(indices_raw)  # Preserve sampler order, including center if sampled.

    contexts = []
    raw["contexts"] = contexts
    for center_t in (0, 1):
        treatment = _readonly(np.concatenate((np.array([center_t], dtype=np.float64), leaf)))
        mean_raw = mechanism.structural_mean(treatment)
        raw[f"mean{center_t}"] = mean_raw
        _array(mean_raw, (200,), "structural mean", evidence, dtype=np.float64)
        mean = _readonly(mean_raw)
        full = _readonly(mean + noise)
        observed = _readonly(full[indices])
        contexts.append(PrivateContext(mechanism, center_t, leaf, noise, indices,
                                       treatment, mean, full, observed))
    pair = PrivatePair(mechanism.cohort_code, mechanism.mechanism_id, pair_repeat_id,
                       mechanism, identities, leaf, noise, indices, tuple(contexts))
    validate_pair(pair)
    return pair
