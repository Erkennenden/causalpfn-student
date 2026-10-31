"""S4.3-B injected kernel-new adapter; no torch/legacy imports or asset I/O.

The supplied manifest pins a mock/checkpoint declaration; no actual checkpoint
bytes are certified here. Legacy frozen_state_sha256 is a SUBSET identity,
separate from the complete state_dict fingerprint and gradient/eval guards.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np

from .contracts import ContractError, validate_predictor_inputs
from . import model_adapters as common

CHECKPOINT = "models/kernel_calibration.pt"
CHECKPOINT_SHA256 = "6273c1abb1255a7d8159fbbd6558c56490915e60ebd312c7c36b33c26b064a2e"
FORMAT = "sparse_classical_scalar_calibration_v1"
_PARAMS = frozenset(("log_covariance_scales", "representation_logits", "noise_logits",
                     "log_grid_scale", "log1p_gamma"))
_BUFFERS = frozenset(("noise_variance_grid", "priors.0.mean", "priors.0.factor",
                      "priors.1.mean", "priors.1.factor"))


def _reject(code, message, **evidence):
    raise ContractError(code, message, evidence)


def _digest_string(value):
    return type(value) is str and len(value) == 64 and all(v in "0123456789abcdef" for v in value)


def _numpy(value):
    if type(value) is np.ndarray:
        return value
    # Duck typed only, enabling mock tensors without importing torch. Future
    # genuine tensors require explicit authorization in S4.5.
    if all(callable(getattr(value, name, None)) for name in ("detach", "cpu", "numpy")):
        array = value.detach().cpu().numpy()
        if type(array) is np.ndarray:
            return array
    _reject("STATE_VALUE", "State requires plain ndarray or tensor snapshot")


def state_fingerprint(value):
    """Framed SHA256: ordered string keys, types, shapes, little-endian C bytes.

    Copies/canonicalizes bytes only; never repr(), Python hash() or file I/O.
    Covers every supplied entry, including scalar calibration parameters.
    """
    digest = hashlib.sha256()
    def emit(tag, blob=b""):
        for part in (tag.encode("ascii"), blob):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    def visit(item):
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                _reject("STATE_VALUE", "State keys must be strings")
            emit("mapping", str(len(item)).encode("ascii"))
            for key in sorted(item):
                emit("key", key.encode("utf-8"))
                visit(item[key])
        elif type(item) in (list, tuple):
            emit("sequence", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif item is None:
            emit("none")
        elif type(item) is str:
            emit("string", item.encode("utf-8"))
        elif type(item) is bool:
            emit("bool", b"1" if item else b"0")
        elif type(item) is int:
            emit("int", str(item).encode("ascii"))
        elif type(item) is float:
            if not math.isfinite(item):
                _reject("STATE_VALUE", "Nonfinite scalar metadata")
            emit("float", item.hex().encode("ascii"))
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            array = _numpy(item)
            if array.dtype.kind not in "biufUS":
                _reject("STATE_VALUE", "Unsafe/object state dtype")
            if array.dtype.kind in "biuf" and not np.isfinite(array).all():
                _reject("STATE_VALUE", "Nonfinite state array")
            dtype = array.dtype.newbyteorder("<")
            canonical = np.asarray(array, dtype=dtype, order="C")
            header = json.dumps({"dtype": dtype.str, "shape": list(array.shape), "order": "C"},
                                sort_keys=True, separators=(",", ":")).encode("ascii")
            emit("array_header", header)
            emit("array_data", canonical.tobytes(order="C"))
    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelNewManifest:
    """Caller-pinned declaration; actual internal checkpoint values unknown here."""
    identity: str
    design_kinds: tuple
    specs_sha256: str
    model_state_sha256: str
    frozen_state_sha256: str
    checkpoint: str = CHECKPOINT
    checkpoint_sha256: str = CHECKPOINT_SHA256
    format: str = FORMAT
    arm_order: tuple = ("00", "01", "10", "11")

    def __post_init__(self):
        if ((self.checkpoint, self.checkpoint_sha256, self.format) !=
                (CHECKPOINT, CHECKPOINT_SHA256, FORMAT)):
            _reject("CHECKPOINT_IDENTITY", "Frozen checkpoint path/hash/format conflict")
        if self.identity not in ("population", "phase5"):
            _reject("CHECKPOINT_IDENTITY", "Unknown source identity")
        if (type(self.design_kinds) is not tuple or len(self.design_kinds) != 2
                or any(kind not in ("phase4", "phase5") for kind in self.design_kinds)):
            _reject("CHECKPOINT_IDENTITY", "Two source-defined design kinds required")
        if self.arm_order != ("00", "01", "10", "11"):
            _reject("STATE_ORDER", "Require canonical four-arm order")
        if not all(_digest_string(value) for value in (
                self.specs_sha256, self.model_state_sha256, self.frozen_state_sha256)):
            _reject("CHECKPOINT_IDENTITY", "Explicit SHA256 declaration required")


def _checkpoint_matches(model, metadata, manifest, frozen_state_reader):
    fields = {"format", "identity", "specs", "design_kinds", "model_state", "frozen_state_sha256"}
    if type(metadata) is not dict or not fields <= set(metadata):
        _reject("CHECKPOINT_FIELDS", "Missing source checkpoint fields")
    if (metadata["format"] != manifest.format or metadata["identity"] != manifest.identity
            or tuple(metadata["design_kinds"]) != manifest.design_kinds
            or state_fingerprint(metadata["specs"]) != manifest.specs_sha256
            or state_fingerprint(metadata["model_state"]) != manifest.model_state_sha256
            or metadata["frozen_state_sha256"] != manifest.frozen_state_sha256):
        _reject("CHECKPOINT_IDENTITY", "Injected checkpoint declaration conflict")
    _structure_matches(model, manifest, frozen_state_reader)


def _structure_matches(model, manifest, frozen_state_reader):
    if (getattr(model, "identity", None) != manifest.identity
            or tuple(getattr(model, "design_kinds", ())) != manifest.design_kinds):
        _reject("MODEL_IDENTITY", "Model identity/design kinds conflict")
    specs = getattr(model, "specs", None)
    if (type(specs) not in (tuple, list) or len(specs) != 2
            or state_fingerprint(specs) != manifest.specs_sha256):
        _reject("MODEL_IDENTITY", "Model specs differ from pinned declaration")
    if frozen_state_reader(model) != manifest.frozen_state_sha256:
        _reject("FROZEN_SUBSET", "Legacy subset identity mismatch")


def _named(model):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = model.state_dict()
    if (set(parameters) != _PARAMS or set(buffers) != _BUFFERS
            or not isinstance(state, Mapping) or set(state) != _PARAMS | _BUFFERS):
        _reject("MODEL_STATE", "Source parameter/buffer/state inventory conflict")
    expected_shapes = {"log_covariance_scales": (2,), "representation_logits": (2,),
                       "noise_logits": (4,), "log_grid_scale": (), "log1p_gamma": (),
                       "noise_variance_grid": (4,)}
    for name, shape in expected_shapes.items():
        array = _numpy(state[name])
        if array.shape != shape or array.dtype != np.dtype("float64"):
            _reject("MODEL_STATE", "Source calibration shape/dtype conflict", field=name)
    for index in (0, 1):
        mean = _numpy(state[f"priors.{index}.mean"])
        lower = _numpy(state[f"priors.{index}.factor"])
        if (mean.ndim != 1 or not len(mean) or lower.shape != (len(mean), len(mean))
                or mean.dtype != np.dtype("float64") or lower.dtype != np.dtype("float64")):
            _reject("MODEL_STATE", "Source prior buffer shape/dtype conflict")
    for name, parameter in parameters.items():
        if not np.array_equal(_numpy(parameter), _numpy(state[name])):
            _reject("MODEL_STATE", "Parameter omitted or changed in state_dict", field=name)
    for name, buffer in buffers.items():
        if not np.array_equal(_numpy(buffer), _numpy(state[name])):
            _reject("MODEL_STATE", "Buffer omitted or changed in state_dict", field=name)
    return parameters, state


def _guard(model, manifest, frozen_state_reader):
    _structure_matches(model, manifest, frozen_state_reader)
    parameters, state = _named(model)
    modules = tuple(model.modules())
    if not modules or modules[0] is not model or any(getattr(v, "training", True) is not False for v in modules):
        _reject("MODEL_EVAL", "All modules must remain in eval mode")
    if any(getattr(value, "requires_grad", True) is not False for value in parameters.values()):
        _reject("MODEL_GRAD", "Every parameter must have requires_grad=False")
    identity = state_fingerprint(state)
    if identity != manifest.model_state_sha256:
        _reject("MODEL_STATE_CHANGED", "Complete parameters/buffers fingerprint changed",
                expected=manifest.model_state_sha256, actual=identity)
    return identity


class KernelNewAdapter:
    """Only public predict_arm_means is invoked; no external context cache."""
    def __init__(self, loader, *, manifest, frozen_state_reader):
        if type(manifest) is not KernelNewManifest:
            _reject("MANIFEST", "Explicit immutable pinned declaration required")
        if not callable(loader) or not callable(frozen_state_reader):
            _reject("LOADER", "Explicit injected loader and subset reader required")
        self.manifest = manifest
        self._loader, self._subset_reader = loader, frozen_state_reader
        self._model, self._load_error, self._attempted = None, None, False

    def _get_model(self):
        if not self._attempted:
            self._attempted = True
            try:
                result = self._loader()
                if type(result) is not tuple or len(result) != 2:
                    _reject("LOADER", "Source loader must return (model, checkpoint_metadata)")
                model, checkpoint = result
                _checkpoint_matches(model, checkpoint, self.manifest, self._subset_reader)
                parameters, state = _named(model)
                if state_fingerprint(state) != self.manifest.model_state_sha256:
                    _reject("MODEL_STATE_CHANGED", "Loaded complete state differs from declaration")
                model.eval()
                for parameter in parameters.values():
                    parameter.requires_grad_(False)
                _guard(model, self.manifest, self._subset_reader)
                if not callable(getattr(model, "predict_arm_means", None)):
                    _reject("PREDICTOR", "Source public prediction method required")
                self._model = model
            except Exception as error:
                # Strings only; never cache a traceback holding context arrays.
                self._load_error = (type(error).__name__, str(error) or type(error).__name__)
        if self._load_error is not None:
            _reject("KERNEL_NEW_LOAD", ": ".join(self._load_error))
        return self._model

    def __call__(self, **payload):
        validate_predictor_inputs(payload)  # Always before any loader.
        model = self._get_model()
        before = _guard(model, self.manifest, self._subset_reader)
        try:
            raw = model.predict_arm_means(**payload)
        finally:
            # Check state even when prediction raises; no repair/reload.
            after = _guard(model, self.manifest, self._subset_reader)
        if type(raw) is not np.ndarray or raw.shape != (200, 4):
            _reject("MODEL_OUTPUT", "Public kernel-new requires ndarray[200,4]")
        return common.ModelPrediction(raw, arm_order=self.manifest.arm_order, diagnostics={
            "full_state_before": before, "full_state_after": after,
            "full_state_scope": "all_parameters_and_buffers",
            "legacy_subset_sha256": self.manifest.frozen_state_sha256,
            "runtime_checkpoint_bytes_verified": False,
            "evaluation_mode": True, "all_parameters_require_grad": False,
            "cache_policy": "public_predict_per_context_no_external_cache"})

    def _stamp(self, bundle, *, arms=None):
        diagnostics = dict(bundle.diagnostics)
        diagnostics["kernel_new_identity"] = {
            "checkpoint": CHECKPOINT, "format": FORMAT, "identity": self.manifest.identity,
            "design_kinds": self.manifest.design_kinds,
            "expected_full_state_sha256": self.manifest.model_state_sha256,
            "legacy_subset_sha256": self.manifest.frozen_state_sha256,
            "legacy_subset_is_complete_state": False,
            "runtime_checkpoint_bytes_verified": False,
            "asset_identity_source": "caller_pinned_declaration_not_runtime_byte_verification"}
        changes = dict(asset_hashes={CHECKPOINT: CHECKPOINT_SHA256}, diagnostics=diagnostics)
        if arms is not None:
            changes["arm_results"] = arms
        return replace(bundle, **changes)

    def estimate_context(self, task, request):
        return self._stamp(common.estimate_context(task, request, "kernel-new", self))

    def estimate_pair(self, pair_task, request, *, context_requests):
        result = common.estimate_pair(pair_task, request, "kernel-new",
                                     context_requests=context_requests, predictors=(self, self))
        return self._stamp(result, arms=tuple(self._stamp(arm) for arm in result.arm_results))
