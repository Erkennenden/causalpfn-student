"""Frozen causalPFN inference using factual covariates X, treatment and outcome.

The module path and loader are retained for the unchanged training CLI.
The removed exposure-augmented adapter is not used. Retrieval and inference
are delegated to the existing local S-learner implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from third_party.causalpfn.models import InContextModel

CAUSALPFN_METHOD_NAME = "causalPFN"


def load_causalpfn_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> InContextModel:
    """Restore the official pretrained CausalPFN checkpoint without training."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CausalPFN checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(
            "CausalPFN checkpoint must contain model_state_dict and model_config."
        )
    model = InContextModel.load(
        model_state=checkpoint["model_state_dict"],
        model_config=checkpoint["model_config"],
    ).to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class CausalPFNDirectResult:
    """Direct predictions and their whole-graph average."""

    ite_direct: torch.Tensor
    ate_direct: torch.Tensor


def _model_device(model: object) -> torch.device:
    """Return the device holding a torch model, defaulting to CPU for test doubles."""

    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device
    return torch.device("cpu")


def predict_causalpfn_direct_on_batch(
    *,
    model: object,
    batch: Mapping[str, torch.Tensor],
    treatment_prob: float,
    query_chunk_size: int = 512,
) -> CausalPFNDirectResult:
    """Run the retained S-learner once per graph using X, T and factual Y.

    treatment_prob is retained for call compatibility. This covariate-only
    learner does not query or integrate neighborhood exposure. As in the
    previous X-only comparator, its predictions are scored in the Direct row.
    """
    from .causalpfn_xonly import estimate_x_only

    del treatment_prob
    required = {"x", "observed_treatment", "y_obs"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"batch is missing causalPFN context keys: {sorted(missing)}")
    x = batch["x"].detach().cpu().numpy()
    treatment = batch["observed_treatment"].detach().cpu().numpy()
    outcome = batch["y_obs"].detach().cpu().numpy()
    if x.ndim == 2:
        x = x[..., None]
    if (x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0
            or treatment.shape != x.shape[:2] or outcome.shape != treatment.shape):
        raise ValueError("Expected X=[graphs,units,features], T/Y=[graphs,units].")
    predictions = []
    device = _model_device(model)
    for index in range(x.shape[0]):
        mu0, mu1, _ = estimate_x_only(
            model, x[index], treatment[index], outcome[index],
            device=device, max_query_length=query_chunk_size,
        )
        predictions.append(torch.from_numpy(mu1 - mu0))
    ite = torch.stack(predictions)
    return CausalPFNDirectResult(
        ite_direct=ite,
        # Preserve the previous comparator's float64 full-node average.
        ate_direct=ite.to(torch.float64).mean(dim=1),
    )
