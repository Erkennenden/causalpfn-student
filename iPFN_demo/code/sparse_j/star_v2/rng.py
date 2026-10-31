"""Six-field SeedSequence/PCG64 identities; no mechanism or factual draws.

Canonical identity text uses ASCII JSON, sorted keys and separators (',', ':').
Its SHA256 is an identifier only and is never used as RNG entropy.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .config import CONFIG

_ENTROPY_FIELDS = (
    "root_seed", "protocol_code", "cohort_code",
    "mechanism_id", "channel_code", "pair_repeat_id",
)
# A strict protocol assertion, not a second selectable numbering scheme.
_FROZEN_CHANNELS = (
    ("population", 1, "mechanism"),
    ("x", 2, "mechanism"),
    ("coefficient", 3, "mechanism"),
    ("leaf_assignment", 4, "pair"),
    ("noise", 5, "pair"),
    ("observation_index", 6, "pair"),
    ("bootstrap", 7, "global_evaluator"),
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    name: str
    code: int
    scope: str


def validate_registry(document):
    """Fail closed on any unregistered mapping, scope, field, type or order."""
    expected = {
        "schema_version": 1,
        "protocol_id": CONFIG.protocol_id,
        "root_seed": CONFIG.root_seed,
        "protocol_code": CONFIG.protocol_code,
        "rng_derivation": "SeedSequence",
        "bit_generator": "PCG64",
        "entropy_fields": list(_ENTROPY_FIELDS),
        "channels": [dict(name=n, code=c, scope=s) for n, c, s in _FROZEN_CHANNELS],
        "bootstrap_identity": [20261030, 2026103001, 2, 0, 7, 0],
    }
    if type(document) is not dict or _canonical(document) != _canonical(expected):
        raise ValueError("RNG registry differs from FROZEN Candidate v0.3.2")
    return MappingProxyType({
        row["name"]: ChannelSpec(**row) for row in document["channels"]
    })


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate registry JSON key: {key}")
        result[key] = value
    return result


def load_registry():
    path = Path(__file__).with_name("protocol") / "rng_registry.json"
    document = json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object)
    return validate_registry(document)


CHANNELS = load_registry()


def _integer(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} requires a Python int, not bool or float")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class RNGIdentity:
    cohort_code: int
    mechanism_id: int
    channel: str
    pair_repeat_id: int

    def __post_init__(self):
        _integer("cohort_code", self.cohort_code)
        _integer("mechanism_id", self.mechanism_id)
        _integer("pair_repeat_id", self.pair_repeat_id)
        mechanisms, repeats = CONFIG.cohort_limits(self.cohort_code)
        if type(self.channel) is not str:
            raise TypeError("channel requires a string")
        if self.channel not in CHANNELS:
            raise ValueError(f"Unknown registered channel: {self.channel}")
        spec = CHANNELS[self.channel]
        if spec.scope == "global_evaluator":
            if (self.cohort_code, self.mechanism_id, self.pair_repeat_id) != (2, 0, 0):
                raise ValueError("Bootstrap requires the unique Test/global identity")
        else:
            if self.mechanism_id >= mechanisms:
                raise ValueError("mechanism_id is outside the registered cohort")
            if spec.scope == "mechanism" and self.pair_repeat_id != 0:
                raise ValueError("Mechanism channels require pair_repeat_id=0")
            if spec.scope == "pair" and self.pair_repeat_id >= repeats:
                raise ValueError("pair_repeat_id is outside the registered cohort")

    @property
    def entropy(self) -> tuple[int, ...]:
        return (CONFIG.root_seed, CONFIG.protocol_code, self.cohort_code,
                self.mechanism_id, CHANNELS[self.channel].code, self.pair_repeat_id)

    def seed_sequence(self):
        return np.random.SeedSequence(list(self.entropy))

    def generator(self):
        """Always return a fresh independent object at this identity's start."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence()))

    def canonical_json(self) -> str:
        return _canonical({
            "schema_version": 1,
            "protocol_id": CONFIG.protocol_id,
            "entropy_fields": list(_ENTROPY_FIELDS),
            "entropy": list(self.entropy),
            "channel_name": self.channel,
            "scope": CHANNELS[self.channel].scope,
            "rng_derivation": "SeedSequence",
            "bit_generator": "PCG64",
        })

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    def to_metadata(self) -> dict:
        seed = self.seed_sequence()
        return {
            "protocol_id": CONFIG.protocol_id,
            "root_seed": CONFIG.root_seed,
            "protocol_code": CONFIG.protocol_code,
            "cohort_code": self.cohort_code,
            "mechanism_id": self.mechanism_id,
            "pair_repeat_id": self.pair_repeat_id,
            "channel_name": self.channel,
            "channel_code": CHANNELS[self.channel].code,
            "scope": CHANNELS[self.channel].scope,
            "entropy_fields": list(_ENTROPY_FIELDS),
            "entropy": list(self.entropy),
            "seed_sequence": {
                "entropy": list(seed.entropy),
                "spawn_key": list(seed.spawn_key),
                "pool_size": seed.pool_size,
                "n_children_spawned": seed.n_children_spawned,
            },
            "bit_generator": "PCG64",
            "numpy_version": np.__version__,
            "canonical_identity": self.canonical_json(),
            "identity_sha256": self.identity_sha256,
        }


def make_identity(*, cohort_code, mechanism_id, channel, pair_repeat_id):
    """Mechanism/pair identity factory; global Bootstrap uses its own entry."""
    if channel == "bootstrap":
        raise ValueError("Use bootstrap_identity() for global evaluator RNG")
    return RNGIdentity(cohort_code, mechanism_id, channel, pair_repeat_id)


def make_generator(identity: RNGIdentity):
    if type(identity) is not RNGIdentity:
        raise TypeError("An explicit validated RNGIdentity is required")
    return identity.generator()


def bootstrap_identity() -> RNGIdentity:
    """No resampling here: only the one registered global evaluator identity."""
    return RNGIdentity(2, 0, "bootstrap", 0)
