"""S4.3-A: injected, lazy kernel-old boundary only.

No asset I/O, legacy imports, real construction, refit or calibration here.
asset_hashes are protocol EXPECTATIONS, explicitly NOT runtime byte receipts.
Native effects/masks are redundancy checks; public adapters own all estimates.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json

import numpy as np

from .contracts import ContractError, validate_predictor_inputs
from . import model_adapters as common

ATOL = 1e-12  # Cross-check tolerance fixed before mock acceptance; rtol=0.
_NATIVE_KEYS = frozenset(("arm_means", "ite_effects", "ate_effects",
                         "arm_supported", "ite_supported", "ate_supported", "diagnostics"))
_DIAGNOSTIC_KEYS = frozenset(("observed_rows", "query_units", "joint_weights", "likelihood"))
_LIKELIHOOD = "Normalized M-dimensional Gaussian density; no division of log likelihood by M or N"


def _reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


@dataclass(frozen=True, slots=True)
class KernelOldIdentity:
    components: tuple = (("models/kernel_subspace_prior.npz", 1.0),
                         ("models/kernel_linear_prior.npz", 0.3))
    representation_weights: tuple = (0.9994774966898915, 0.0005225033101084962)
    noise_weights: tuple = (4.776321066414275e-07, 0.8851869523745477,
                            0.11481239587152346, 1.741218223865947e-07)
    noise_variances: tuple = (0.24760918477397018, 0.9904367390958807,
                              3.961746956383523, 15.846987825534091)
    arm_order: tuple = ("00", "01", "10", "11")
    asset_hashes: tuple = (
        ("models/kernel_subspace_prior.npz", "02734745ee96c712da1494ced0960538a5ac4b8eeb31b2dfa70c90ccf5ebe130"),
        ("models/kernel_linear_prior.npz", "90e040ecdf4eba9d92e90f98c457dbffc24821498cc0f189c8f77cc52239806c"),
        ("models/kernel_components.json", "594aca0221f35e364d012e2067546f0ea4c61bb6cf89db8a87cbdd9f6b19fb3d"),
        ("models/kernel_mixture.json", "3e8a76ab443649831824c61e3157be497345c66d9c8548852dcf8d5320da3981"))


FROZEN_KERNEL_OLD = KernelOldIdentity()


def _identity_json(identity):
    if type(identity) is not KernelOldIdentity:
        _reject("KERNEL_IDENTITY", "Injected reader must return KernelOldIdentity")
    for name in ("components", "representation_weights", "noise_weights",
                 "noise_variances", "arm_order", "asset_hashes"):
        if type(getattr(identity, name)) is not tuple:
            _reject("KERNEL_IDENTITY", "Immutable tuple identity fields required", field=name)
    if any(type(row) is not tuple for row in (*identity.components, *identity.asset_hashes)):
        _reject("KERNEL_IDENTITY", "Immutable component/asset tuples required")
    try:
        return json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError):
        _reject("KERNEL_IDENTITY", "Identity must contain finite canonical metadata")


def validate_kernel_old_identity(identity):
    if _identity_json(identity) != _identity_json(FROZEN_KERNEL_OLD):
        _reject("KERNEL_IDENTITY", "Frozen kernel-old configuration conflict")
    return FROZEN_KERNEL_OLD


def query_cache_identity(x, adjacency, identity=FROZEN_KERNEL_OLD):
    """Configuration-qualified public geometry key, not an estimator input.

    Small unit checks may change A here; the prediction entry STILL requires
    the canonical N=200 Star via validate_predictor_inputs.
    """
    validate_kernel_old_identity(identity)
    if (type(x) is not np.ndarray or x.shape != (200, 20) or x.dtype.str != "<f4"
            or type(adjacency) is not np.ndarray or adjacency.shape != (200, 200)
            or adjacency.dtype.str != "<f8" or not np.isfinite(x).all()
            or not np.isfinite(adjacency).all()):
        _reject("QUERY_KEY", "Require finite canonical X/A shapes and dtypes")
    header = _identity_json(identity).encode("ascii")
    digest = hashlib.sha256()
    for part in (header, x.tobytes(order="C"), adjacency.tobytes(order="C")):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _legacy_cache_key(x, adjacency):
    """Mirror phase6_sparse_classical.validate() before its query_key().

    These local float64 views/copies are for the legacy key only. The predictor
    still receives the untouched canonical five-array payload.
    """
    normalized_x = np.asarray(x, dtype=float)
    normalized_adjacency = np.asarray(adjacency, dtype=float)
    return hashlib.sha256(
        normalized_x.tobytes() + normalized_adjacency.tobytes()).digest()


def _array(value, shape, dtype, field):
    if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.dtype(dtype):
        _reject("KERNEL_OUTPUT", "Native field shape/dtype conflict", field=field)
    return value


def _equal(actual, expected, field):
    # Only redundant native evidence is compared; these values are never used
    # to populate a public EstimateBundle.
    if not np.allclose(actual, expected, atol=ATOL, rtol=0, equal_nan=True):
        _reject("KERNEL_CONSISTENCY", "Inconsistent native field", field=field)


def _native_prediction(raw):
    if type(raw) is not dict or set(raw) != _NATIVE_KEYS:
        _reject("KERNEL_OUTPUT", "Only the canonical FrozenMixture dict is accepted")
    mu = _array(raw["arm_means"], (200, 4), "float64", "arm_means")
    ite = _array(raw["ite_effects"], (200, 3), "float64", "ite_effects")
    ate = _array(raw["ate_effects"], (3,), "float64", "ate_effects")
    # Redundancy/axis assertions, NOT a replacement numerical core.
    for k, (positive, negative) in enumerate(((2, 0), (3, 2), (3, 0))):
        with np.errstate(over="ignore", invalid="ignore"):
            _equal(ite[:, k], mu[:, positive]-mu[:, negative], "ite_effects")
    with np.errstate(over="ignore", invalid="ignore"):
        _equal(ate, ite.mean(0), "ate_effects")
    for name, value, shape in (("arm_supported", mu, (4,)),
                              ("ite_supported", ite, (3,)),
                              ("ate_supported", ate, (3,))):
        mask = _array(raw[name], shape, "bool", name)
        expected = np.isfinite(value).all(0) if value.ndim == 2 else np.isfinite(value)
        if not np.array_equal(mask, expected):
            _reject("KERNEL_CONSISTENCY", "Native finite mask conflict", field=name)
    diagnostics = raw["diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != _DIAGNOSTIC_KEYS:
        _reject("KERNEL_DIAGNOSTICS", "Canonical diagnostics required")
    if (type(diagnostics["observed_rows"]) is not int or diagnostics["observed_rows"] != 50
            or type(diagnostics["query_units"]) is not int or diagnostics["query_units"] != 200
            or diagnostics["likelihood"] != _LIKELIHOOD):
        _reject("KERNEL_DIAGNOSTICS", "Observed/query/evidence budget conflict")
    weights = diagnostics["joint_weights"]
    if (type(weights) not in (list, tuple) or len(weights) != 8
            or any(type(v) not in (float, int) for v in weights)):
        _reject("KERNEL_DIAGNOSTICS", "Eight scalar posterior weights required")
    weights = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or abs(float(weights.sum())-1.) > ATOL):
        _reject("KERNEL_DIAGNOSTICS", "Invalid normalized posterior weights")
    # Native masks do not become empirical Support. Never retain raw effects.
    return common.ModelPrediction(mu.copy(), diagnostics={
        "native_cross_checks": "passed", "observed_rows": 50, "query_units": 200,
        "joint_weights": weights.tolist(), "likelihood": _LIKELIHOOD})


def _cache_state(model):
    caches = getattr(model, "caches", None)
    if type(caches) is not list or len(caches) != 2:
        _reject("KERNEL_CACHE", "Two explicit query caches required")
    result = []
    for cache in caches:
        if type(cache) is not dict:
            _reject("KERNEL_CACHE", "Plain query cache dictionaries required")
        if not cache:
            result.append(None)
            continue
        if set(cache) != {"key", "query_mean", "query_covariance"}:
            _reject("KERNEL_CACHE", "Dynamic/residual/evidence cache entries forbidden")
        key = cache["key"]
        if type(key) is not bytes or len(key) != 32:
            _reject("KERNEL_CACHE", "Legacy X/A key must be SHA256 bytes")
        mean = _array(cache["query_mean"], (200, 4), "float64", "query_mean")
        covariance = cache["query_covariance"]
        if (type(covariance) is not np.ndarray or covariance.dtype != np.dtype("float64")
                or covariance.ndim != 3 or covariance.shape[:2] != (200, 4)
                or covariance.shape[2] < 1 or not np.isfinite(mean).all()
                or not np.isfinite(covariance).all()):
            _reject("KERNEL_CACHE", "Invalid finite query geometry cache")
        result.append((key, hashlib.sha256(mean.tobytes()).hexdigest(),
                       hashlib.sha256(covariance.tobytes()).hexdigest()))
    return tuple(result)


class KernelOldAdapter:
    """Lazy injected loader plus read-only identity reader; no default loader.

    Only the frozen model and load-failure state persist here. No context input,
    observed residual/evidence or native prediction is retained by this object.
    Runtime asset bytes and genuine model state verification remain S4.5.
    """
    def __init__(self, loader, *, identity_reader):
        if not callable(loader) or not callable(identity_reader):
            _reject("KERNEL_LOADER", "Loader and identity reader must be injected")
        self._loader, self._identity_reader = loader, identity_reader
        self._model, self._load_error, self._attempted = None, None, False

    def _get_model(self):
        if not self._attempted:
            self._attempted = True
            try:
                model = self._loader()
                validate_kernel_old_identity(self._identity_reader(model))
                if not callable(getattr(model, "predict", None)):
                    _reject("KERNEL_LOADER", "FrozenMixture-like predict is missing")
                _cache_state(model)
                self._model = model
            except Exception as error:
                # Keep strings only: cached exception tracebacks could retain
                # an observed payload from a failed context.
                self._load_error = (type(error).__name__, str(error) or type(error).__name__)
        if self._load_error is not None:
            _reject("KERNEL_LOAD_FAILURE",
                    self._load_error[0] + ": " + self._load_error[1])
            # Never reload/retry or retain an original context traceback.
        return self._model

    def __call__(self, **payload):
        validate_predictor_inputs(payload)  # Before any loader/identity callback.
        model = self._get_model()
        validate_kernel_old_identity(self._identity_reader(model))
        before = _cache_state(model)
        legacy_key = _legacy_cache_key(payload["x"], payload["adjacency"])
        geometry_key = query_cache_identity(payload["x"], payload["adjacency"])
        raw = model.predict(**payload)  # Exactly five arrays; no identity or metadata.
        validate_kernel_old_identity(self._identity_reader(model))
        after = _cache_state(model)
        for old, new in zip(before, after):
            if new is not None and new[0] != legacy_key:
                _reject("KERNEL_CACHE", "Query cache key does not match current X/A")
            if old is not None and old[0] == legacy_key and old != new:
                _reject("KERNEL_CACHE", "Same-geometry cache changed with dynamic inputs")
        prediction = _native_prediction(raw)
        return common.ModelPrediction(prediction.arm_means, diagnostics={
            **prediction.diagnostics, "configuration_identity_checked": True,
            "qualified_query_key": geometry_key,
            "runtime_asset_bytes_verified": False,
            "cache_check_scope": "declared_query_geometry_only_not_real_model_validation"})

    def _stamp(self, bundle, *, arm_results=None):
        diagnostics = dict(bundle.diagnostics)
        diagnostics["kernel_old_identity"] = {
            "configuration": asdict(FROZEN_KERNEL_OLD),
            "asset_identity_source": "protocol_expected_not_runtime_verified",
            "runtime_asset_bytes_verified": False,
            "identification_blocked": bundle.status == "NOT_IDENTIFIED"}
        updates = dict(asset_hashes=dict(FROZEN_KERNEL_OLD.asset_hashes), diagnostics=diagnostics)
        if arm_results is not None:
            updates["arm_results"] = arm_results
        # Existing dataclass constructor revalidates assets/status/arm bindings.
        return replace(bundle, **updates)

    def estimate_context(self, task, request):
        return self._stamp(common.estimate_context(task, request, "kernel-old", self))

    def estimate_pair(self, pair_task, request, *, context_requests):
        result = common.estimate_pair(pair_task, request, "kernel-old",
                                     context_requests=context_requests, predictors=(self, self))
        arms = tuple(self._stamp(arm) for arm in result.arm_results)
        return self._stamp(result, arm_results=arms)
