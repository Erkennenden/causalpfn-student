"""Simulator/evaluator-only Star mechanisms and noiseless causal truth.

No treatment/noise/observation draws, predictor payloads, model loading or I/O.
X is quantized to float32 exactly once as registered; every structural operation
uses float64 thereafter. Array SHA256 hashes canonical dtype/shape JSON + LF +
C-order bytes (not Git object IDs). Formal calls need separate execution approval.
"""

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from pfn_social.phase3_hierarchical import (
    SETTINGS as BASE_SETTINGS,
    hidden_features,
    coefficient_prior,
    _components,
    structural_outcome,
)
from pfn_social.phase5_large_network import (
    common_projection,
    REPRESENTATION_SEED,
)
from pfn_social.contracts import arm_mean_exposure

from .config import CONFIG
from .rng import RNGIdentity, make_generator, make_identity


def star_adjacency(n: int = CONFIG.N) -> np.ndarray:
    """Ordered, loopless Star; n!=200 is a pure mathematical test helper."""
    if type(n) is not int:
        raise TypeError("n must be a Python int")
    if n < 2:
        raise ValueError("A Star requires at least two nodes")
    a = np.zeros((n, n), dtype=np.float64)
    a[0, 1:] = a[1:, 0] = 1.0
    return a


def _numeric(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _graph(adjacency):
    a = _numeric(adjacency, "adjacency")
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be a square matrix with N>=2")
    if (not np.isin(a, [0.0, 1.0]).all() or not np.array_equal(a, a.T)
            or np.any(np.diag(a) != 0)):
        raise ValueError("Require binary, undirected, loopless adjacency")
    degree = a.sum(axis=1, dtype=np.float64)
    if np.any(degree <= 0):
        raise ValueError("Every node must have positive degree")
    return a, degree


def _treatment(treatment, n):
    t = _numeric(treatment, "treatment")
    if t.shape != (n,) or not np.isin(t, [0.0, 1.0]).all():
        raise ValueError("treatment must be a binary vector in fixed node order")
    return t


def exposure(adjacency, treatment):
    """Return float64 neighbor fractions and integer S=1{fraction>0.5}."""
    a, degree = _graph(adjacency)
    t = _treatment(treatment, len(a))
    fraction = (a @ t) / degree
    return fraction, (fraction > CONFIG.exposure_threshold).astype(np.int64)


def _structure(adjacency, h, coefficients):
    a, _ = _graph(adjacency)
    h = _numeric(h, "h")
    beta = _numeric(coefficients, "coefficients")
    if h.shape != (len(a), 3) or beta.shape != (25,):
        raise ValueError("Require h[N,3] and the fixed 25 coefficients")
    # Conversion occurs BEFORE the fixed functions, never as an output upcast.
    return SimpleNamespace(adjacency=a, h=h, coefficients=beta,
                           components=_components(a, h, beta))


def structural_components(adjacency, h, coefficients):
    """Float64 columns: b, tau, gamma, eta, using the fixed common function."""
    return _structure(adjacency, h, coefficients).components


def treated_neighbor_features(adjacency, h, treatment):
    a, degree = _graph(adjacency)
    h = _numeric(h, "h")
    if h.shape != (len(a), 3):
        raise ValueError("Require h[N,3]")
    t = _treatment(treatment, len(a))
    return a @ (t[:, None] * h) / degree[:, None]


def structural_mean(adjacency, h, coefficients, treatment):
    """Fixed structural_outcome with validated float64 inputs; no epsilon."""
    world = _structure(adjacency, h, coefficients)
    t = _treatment(treatment, len(world.adjacency))
    result = structural_outcome(world, t)
    if result.dtype != np.float64 or not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite or non-float64 structural mean")
    return result


def four_state_truth(adjacency, h, coefficients):
    """Noise-free mu[node, own t, exposure s], canonical 00,01,10,11."""
    world = _structure(adjacency, h, coefficients)
    a, h, beta = world.adjacency, world.h, world.coefficients
    degree = a.sum(axis=1, dtype=np.float64)
    hbar = a @ h / degree[:, None]
    b, tau, gamma, eta = world.components.T
    mu = np.empty((len(a), 2, 2), dtype=np.float64)
    for s in (0, 1):
        by_degree = {int(d): arm_mean_exposure(int(d), 0.5, s)
                     for d in np.unique(degree)}
        mean_e = np.array([by_degree[int(d)] for d in degree], dtype=np.float64)
        for t in (0, 1):
            mu[:, t, s] = b + t * tau + mean_e * (
                gamma + t * eta + hbar @ (beta[19:22] + t * beta[22:25]))
    if not np.isfinite(mu).all():
        raise FloatingPointError("Nonfinite four-state truth")
    return mu


def effects_from_mu(mu):
    mu = _numeric(mu, "mu")
    if mu.ndim != 3 or mu.shape[0] < 2 or mu.shape[1:] != (2, 2):
        raise ValueError("Require mu[N,2,2], never legacy flattened state codes")
    ite = np.column_stack((mu[:, 1, 0] - mu[:, 0, 0],
                           mu[:, 1, 1] - mu[:, 1, 0],
                           mu[:, 1, 1] - mu[:, 0, 0]))
    return ite, ite.mean(axis=0, dtype=np.float64)


def group_effects(ite):
    ite = _numeric(ite, "ite")
    if ite.ndim != 2 or ite.shape[0] < 2 or ite.shape[1] != 3:
        raise ValueError("Require ITE[N,3] in Direct, Spillover, Total order")
    return {"center": ite[0].copy(),
            "leaf": ite[1:].mean(axis=0, dtype=np.float64),
            "all": ite.mean(axis=0, dtype=np.float64)}


def array_sha256(array):
    a = np.ascontiguousarray(array)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + a.tobytes(order="C")).hexdigest()


def _readonly(array):
    a = np.ascontiguousarray(array)
    # Bytes-backed arrays cannot be made writable by merely flipping a flag.
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class SimulatorMechanism:
    """Privileged object, deliberately NOT a mapping or five-array request."""
    cohort_code: int
    mechanism_id: int
    rng_identities: tuple[RNGIdentity, ...]
    node_ids: np.ndarray
    population_mean: np.ndarray
    x: np.ndarray
    adjacency: np.ndarray
    projection: np.ndarray
    h: np.ndarray
    coefficients: np.ndarray
    components: np.ndarray
    mu: np.ndarray
    ite: np.ndarray
    ate: np.ndarray

    def structural_mean(self, treatment):
        return structural_mean(self.adjacency, self.h, self.coefficients, treatment)

    def group_effects(self):
        return group_effects(self.ite)

    def metadata(self):
        arrays = ("node_ids", "population_mean", "x", "adjacency", "projection",
                  "h", "coefficients", "components", "mu", "ite", "ate")
        return {
            "access": "simulator_evaluator_only",
            "protocol_id": CONFIG.protocol_id,
            "protocol_version": CONFIG.protocol_version,
            "protocol_sha256": CONFIG.protocol_sha256,
            "source_commit": CONFIG.source_commit,
            "target_id": CONFIG.target_id,
            "cohort_code": self.cohort_code,
            "mechanism_id": self.mechanism_id,
            "base_setting_id": CONFIG.base_setting_id,
            "representation": CONFIG.representation,
            "representation_seed": CONFIG.representation_seed,
            "noise_sd": CONFIG.noise_sd,
            "arm_order": list(CONFIG.arm_order),
            "effect_order": list(CONFIG.effect_order),
            "rng_identities": [i.to_metadata() for i in self.rng_identities],
            "array_sha256": {name: array_sha256(getattr(self, name)) for name in arrays},
            "array_dtypes": {name: getattr(self, name).dtype.str for name in arrays},
            "array_hash_format": "SHA256(sorted compact ASCII dtype/shape JSON + LF + C-order bytes)",
            "precision": {
                "x_generation": "float64 population + 0.6 * float64 normal, then one float32 cast",
                "structural_arithmetic": "float64",
                "x_quantization": "float32 X quantization is retained in hidden_features; not recovered by float64",
                "coefficient_sampling": "Generator.multivariate_normal(mean,cov), fixed NumPy defaults",
            },
        }


def generate_star_mechanism(*, cohort_code: int, mechanism_id: int):
    """Implement the registered mechanism path, without pair generation.

    This function has no alternate seed/cohort/config injection. Unit tests mock
    make_generator with explicitly separate test-only streams. Formal calls are
    reserved for separately authorized execution stages.
    """
    setting = BASE_SETTINGS[CONFIG.base_setting_id]
    if (setting.representation != CONFIG.representation
            or setting.noise_sd != CONFIG.noise_sd
            or REPRESENTATION_SEED != CONFIG.representation_seed):
        raise RuntimeError("Fixed common source and frozen config disagree")
    identities = tuple(make_identity(
        cohort_code=cohort_code, mechanism_id=mechanism_id,
        channel=channel, pair_repeat_id=0)
        for channel in ("population", "x", "coefficient"))
    population_rng, x_rng, beta_rng = (make_generator(i) for i in identities)
    population = population_rng.normal(0.0, 0.8, CONFIG.x_dim)
    x = (population + 0.6 * x_rng.normal(size=(CONFIG.N, CONFIG.x_dim))).astype(np.float32)
    a = star_adjacency()
    projection = common_projection().copy()
    h = hidden_features(x, projection, setting.representation)
    mean, covariance = coefficient_prior()
    beta = beta_rng.multivariate_normal(mean, covariance)
    components = structural_components(a, h, beta)
    mu = four_state_truth(a, h, beta)
    ite, ate = effects_from_mu(mu)
    arrays = (np.arange(CONFIG.N, dtype=np.int64), population, x, a, projection,
              h, beta, components, mu, ite, ate)
    return SimulatorMechanism(cohort_code, mechanism_id, identities,
                              *(_readonly(value) for value in arrays))
