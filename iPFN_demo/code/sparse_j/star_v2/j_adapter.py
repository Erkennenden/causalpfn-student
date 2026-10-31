"""Lazy frozen-J Context bridge. Model/interval modules import only on use."""
from dataclasses import replace
from collections.abc import Mapping
import hashlib
import json

import numpy as np

from .config import CONFIG
from .contracts import validate_predictor_inputs
from .estimation import SamplingUncertainty
from .model_adapters import ModelPrediction, ModelIntervals
from .model_adapters import estimate_context as _estimate_context
from .model_adapters import estimate_pair as _estimate_pair
from .model_adapters import context_requests_from_parent

INTERVAL_KIND = "frozen_J_marginal_posterior_credible"


def _default_loader():
    # Preserve the frozen loader's default path and verification unchanged.
    from predict import load_selected_model
    return load_selected_model()


def _default_intervals(model, **inputs):
    from pfn_social.intervals import predict_intervals
    return predict_intervals(model, **inputs)


def _j_inputs(inputs):
    validate_predictor_inputs(inputs)
    return dict(
        x=inputs["x"], adjacency=inputs["adjacency"],
        treatment=inputs["treatment"], observed_indices=inputs["observed_indices"],
        outcomes_observed=np.array(inputs["outcomes_observed"],
                                   dtype=np.float32, copy=True))


def model_state_fingerprint(model):
    """SHA256 of complete state_dict; None denotes a stateless test double.

    Frozen J wraps its nn.Module as .model. Tensor bytes are detached copies
    read on CPU; no Torch import, parameter update or model serialization.
    """
    owner = getattr(model, "model", model)
    if not callable(getattr(type(owner), "state_dict", None)):
        return None
    state = owner.state_dict()
    if not isinstance(state, Mapping) or any(type(k) is not str for k in state):
        raise TypeError("Model state_dict must have string keys")
    digest = hashlib.sha256()
    def emit(blob):
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    emit(b"star-v2-j-state-v1")
    for key in sorted(state):
        value = state[key]
        if all(callable(getattr(value, name, None)) for name in ("detach", "cpu", "numpy")):
            value = value.detach().cpu().numpy()
        if type(value) is not np.ndarray or value.dtype.hasobject:
            raise TypeError("Model state must contain numeric arrays or tensors")
        array = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=True))
        emit(key.encode("utf-8"))
        emit(array.dtype.str.encode("ascii"))
        emit(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        emit(array.tobytes(order="C"))
    return digest.hexdigest()


class JAdapter:
    """One lazy model per instance; public contracts own all output statuses."""

    def __init__(self, *, loader=None, interval_predictor=None):
        if loader is not None and not callable(loader):
            raise TypeError("loader must be callable")
        if interval_predictor is not None and not callable(interval_predictor):
            raise TypeError("interval_predictor must be callable")
        self._loader = _default_loader if loader is None else loader
        self._interval_predictor = (_default_intervals if interval_predictor is None
                                    else interval_predictor)
        self._model = None
        self._model_state = None

    def _get_model(self):
        if self._model is None:
            import torch
            # Includes default-loader import/construction; never reseed globally.
            with torch.random.fork_rng(devices=[]):
                model = self._loader()
            if not callable(getattr(model, "predict_arm_means", None)):
                raise TypeError("J model must expose predict_arm_means")
            self._model_state = model_state_fingerprint(model)
            self._model = model
        return self._model

    def _check_state(self):
        if model_state_fingerprint(self._model) != self._model_state:
            raise RuntimeError("J parameters/buffers state changed")

    def _invoke(self, callback):
        model = self._get_model()
        self._check_state()
        try:
            return callback(model)
        finally:
            self._check_state()

    def __call__(self, **inputs):
        """The frozen point method takes exactly five arrays, no diagnostics."""
        payload = _j_inputs(inputs)
        points = self._invoke(lambda model: model.predict_arm_means(**payload))
        if (type(points) is not np.ndarray or points.shape != (200, 4)
                or points.dtype not in (np.dtype("float32"), np.dtype("float64"))):
            raise ValueError("J arm_means must be a plain floating [200,4] array")
        # The public layer checks finiteness per field/effect, preserving valid
        # effects when only an unused state (01) is nonfinite.
        return ModelPrediction(points, arm_order=CONFIG.arm_order)

    def predict_intervals(self, **inputs):
        """Bridge the frozen marginal 95% algorithm, without modifying it."""
        payload = _j_inputs(inputs)
        native = self._invoke(
            lambda model: self._interval_predictor(model, **payload, level=0.95))
        if type(native) is not dict or set(native) != {
                "ate_lower", "ate_upper", "interval_level", "interval_kind"}:
            raise ValueError("Expected the frozen J interval result fields")
        level = np.asarray(native["interval_level"])
        kind = np.asarray(native["interval_kind"])
        if (level.shape != () or level.dtype.kind not in "fi"
                or float(level) != 0.95):
            raise ValueError("J interval level must be 0.95")
        if (kind.shape != () or kind.dtype.kind != "U"
                or str(kind.item()) != INTERVAL_KIND):
            raise ValueError("Unexpected frozen J interval_kind")
        return ModelIntervals(
            lower=native["ate_lower"], upper=native["ate_upper"], level=0.95,
            diagnostics={"interval_kind": str(kind.item())})

    def estimate_context(self, task, request):
        """Delegate the state machine, then normalize unavailable provenance."""
        result = _estimate_context(
            task, request, "j", self, interval_predictor=self.predict_intervals)
        return self._normalize_context(result)

    @staticmethod
    def _normalize_context(result):
        if "interval" not in result.request.requested_outputs:
            return result
        # Metadata-only repair after public validation; never change a status,
        # endpoint, point estimate, Identification identity or failure reason.
        records = tuple(replace(record, source="model_posterior")
                        if record.status == "FAILED" else record
                        for record in result.intervals)
        uncertainty = result.sampling_uncertainty
        if uncertainty.status == "FAILED":
            uncertainty = SamplingUncertainty(
                "FAILED", "model_posterior", "context", None)
        return replace(result, intervals=records, sampling_uncertainty=uncertainty)


    def estimate_pair(self, pair_task, request, *, context_requests=None, request_ids=None):
        """Public estimate_pair -> estimate_context -> compose_pair, unchanged.

        Callers supply child requests or explicit child IDs; no ID is invented.
        Context intervals remain in arm_results, never averaged into pair bounds.
        """
        if context_requests is None:
            context_requests = context_requests_from_parent(
                pair_task, request, request_ids=request_ids)
        elif request_ids is not None:
            raise TypeError("Supply context_requests or request_ids, not both")
        result = _estimate_pair(
            pair_task, request, "j", context_requests=context_requests,
            predictors=(self, self),
            interval_predictors=(self.predict_intervals, self.predict_intervals))
        return replace(result, arm_results=tuple(
            self._normalize_context(arm) for arm in result.arm_results))
