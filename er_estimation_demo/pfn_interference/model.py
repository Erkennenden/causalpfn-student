#!/usr/bin/env python3
"""Network encoder, 10-layer Transformer and four-arm GMM predictions.

The main ER1000 experiment is orchestrated by data.py, training.py and
evaluation.py. Its DGP is in dgp.py; graph/data utilities shared with other
experiments remain here. Run the project with ``python -m pfn_interference``.
"""

from __future__ import annotations

from pfn_interference.baselines.localized_config import add_localized_arguments, localized_benchmark_kwargs
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import timedelta

from pfn_interference.runtime import MetricAccumulator, cached_cpu_batch, read_csv_arrays, validation_indices

import numpy as np
import networkx as nx
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from pfn_interference.estimands import oracle_ate, oracle_ite_from_parameters, sample_majority_exposures


def load_causalpfn_checkpoint(*args, **kwargs):
    """Lazy proxy so CSV generation does not import the CausalPFN stack."""
    from pfn_interference.baselines.causalpfn_direct import load_causalpfn_checkpoint as _impl
    return _impl(*args, **kwargs)


def evaluate_unified_benchmark(*args, **kwargs):
    """Lazy proxy so data preparation does not import paper pfn_interference.baselines."""
    from pfn_interference.metrics.unified import evaluate_unified_benchmark as _impl
    return _impl(*args, **kwargs)


def save_unified_benchmark(*args, **kwargs):
    """Lazy proxy for the unified benchmark writer."""
    from pfn_interference.metrics.unified import save_unified_benchmark as _impl
    return _impl(*args, **kwargs)


from pfn_interference.cepo import (PREDICTION_PROTOCOL, ARM_NAMES, cepo_queries, cepo_targets,
                  effects_from_mu, effect_targets, compute_cepo_losses, compute_cepo_metrics)

TensorBatch = Dict[str, torch.Tensor]
ER_GRAPH = 2
CONFIGURATION_GRAPH = 3
RGG_GRAPH = 4
SBM_GRAPH = 5
GRAPH_FAMILY_ORDER = (ER_GRAPH, CONFIGURATION_GRAPH, RGG_GRAPH, SBM_GRAPH)
MIXED_TARGET_MEAN_DEGREE_MIN = 4.0
MIXED_TARGET_MEAN_DEGREE_MAX = 10.0
GRAPH_TYPE_NAMES = {
    ER_GRAPH: "er",
    CONFIGURATION_GRAPH: "configuration",
    RGG_GRAPH: "rgg",
    SBM_GRAPH: "sbm",
}
ER_MAX_RESAMPLE_ATTEMPTS = 10000
QUERY_DIRECT = 0
QUERY_SPILLOVER = 1
QUERY_TOTAL = 2
QUERY_TYPE_NAMES = ("direct", "spillover", "total")
EFFECT_PREFIXES = (
    "direct_effect",
    "spillover_effect",
    "total_effect",
)

N_UNITS = 1000
TOKEN_DIM = 5
QUERY_DIM = 4
CSV_SCHEMA_VERSION = "majority_arm_v12_noise_free_cepo"
CEPO_LABEL_PROTOCOL = "noise_free_structural_baseline_v2"
CSV_ADJACENCY_FLOAT_CHUNK_SIZE = 256


@dataclass(frozen=True)
class RootedConfiguration:
    """Radius-one rooted graph with neighbor-treatment marks.

    The root node's own treatment is intentionally excluded from
    ``treatment_marks`` so that own-treatment and interference contrasts remain
    separate. The first entry in every local tensor is the root.
    """

    root_index: int
    node_indices: torch.Tensor
    adjacency: torch.Tensor
    x: torch.Tensor
    treatment_marks: torch.Tensor
    root_indicator: torch.Tensor

    @property
    def exposure(self) -> float:
        """Return the treated-neighbor proportion encoded by the marks."""

        neighbor_count = int(self.node_indices.numel()) - 1
        if neighbor_count <= 0:
            raise ValueError("A rooted configuration must contain a neighbor.")
        return float(self.treatment_marks[1:].sum().item() / neighbor_count)


def extract_radius_one_rooted_configuration(
    *,
    adjacency: torch.Tensor,
    x: torch.Tensor,
    neighbor_treatment: torch.Tensor,
    root_index: int,
) -> RootedConfiguration:
    """Extract the radius-one configuration for one root from full tensors."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    n_units = int(adjacency.shape[0])
    if x.shape != (n_units,):
        raise ValueError("x must have shape [units].")
    if neighbor_treatment.shape != (n_units,):
        raise ValueError("neighbor_treatment must have shape [units].")
    if not 0 <= root_index < n_units:
        raise ValueError("root_index is outside the graph.")
    if bool(((neighbor_treatment != 0) & (neighbor_treatment != 1)).any()):
        raise ValueError("neighbor_treatment must be binary.")
    neighbor_mask = adjacency[root_index] > 0
    if not bool(neighbor_mask.any()):
        raise ValueError("The root must have at least one neighbor.")
    if bool((neighbor_treatment[~neighbor_mask] != 0).any()):
        raise ValueError("Only neighbors of the root may carry treatment marks.")

    neighbors = torch.nonzero(neighbor_mask, as_tuple=False).flatten().sort().values
    root = torch.tensor([root_index], dtype=torch.long, device=adjacency.device)
    node_indices = torch.cat([root, neighbors])
    local_adjacency = adjacency.index_select(0, node_indices).index_select(
        1, node_indices
    )
    local_x = x.index_select(0, node_indices)
    treatment_marks = neighbor_treatment.index_select(0, node_indices).clone()
    treatment_marks[0] = 0
    root_indicator = torch.zeros(
        node_indices.numel(), dtype=torch.float32, device=adjacency.device
    )
    root_indicator[0] = 1.0
    return RootedConfiguration(
        root_index=root_index,
        node_indices=node_indices,
        adjacency=local_adjacency,
        x=local_x,
        treatment_marks=treatment_marks,
        root_indicator=root_indicator,
    )


@dataclass(frozen=True)
class DistributedRuntime:
    """Single-node torchrun metadata with a single-process fallback."""

    rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def distributed_runtime_from_env() -> DistributedRuntime:
    """Read the single-node torchrun rank variables without extra device metadata."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError("Invalid torchrun RANK/WORLD_SIZE values.")
    return DistributedRuntime(rank=rank, world_size=world_size)


def build_observed_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Validate and return factual node observations for every unit."""

    if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
        raise ValueError("tokens must have shape [batch, units, 5].")
    return tokens


def query_effect_targets(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return all-node ITE targets with shape ``[batch, units, 3]``."""

    target = batch["query_effect"]
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("query_effect must have shape [batch, units, 3].")
    tokens = batch.get("tokens")
    if tokens is not None:
        n_units = int(tokens.shape[1])
        if target.shape[1] != n_units:
            raise ValueError("query_effect must have shape [batch, units, 3].")
    return target


@dataclass(frozen=True)
class DataConfig:
    """Synthetic prior for the scalar-X ER-only demo.

    The default outcome is Gao--Ding Design 2 on the ER graph generated here.
    Legacy TNet and v7 modes remain loadable so existing checkpoints keep their
    original data-generating process.
    """

    n_units: int = N_UNITS
    er_edge_probability: float = 0.02
    graph_family: str = "er"
    graph_protocol: str = "resampled"
    fixed_graph_seed: int = 12345
    fixed_graph_upper_hex: Optional[str] = None
    sbm_between_probability: float = 0.006
    configuration_degree_sigma: float = 1.0
    configuration_max_degree: int = 25
    rgg_torus: bool = True
    treatment_prob: float = 0.5
    dgp_version: str = "gao_ding_design2_outcome"
    interference_lambda: float = 1.0
    spillover_gamma_min: float = -3.6
    spillover_gamma_max: float = 0.0
    spillover_gamma: Optional[float] = None
    neighbor_beta_min: float = -6.0
    neighbor_beta_max: float = 0.0
    neighbor_beta: Optional[float] = None
    x_sd: float = 1.0
    x_outcome_coefficient: float = 0.0
    own_score_baseline_coefficient: float = 1.0
    neighbor_score_baseline_coefficient: float = 0.5
    baseline_sd: float = 1.0
    unit_sd: float = 0.2
    tau_mean: float = 0.4
    tau_episode_sd: float = 0.0
    tau_unit_sd: float = 0.0
    tau_own_score_slope: float = 0.2
    tau_neighbor_score_slope: float = 0.2
    gamma_mean: float = 0.2
    gamma_sd: float = 0.0
    gamma_own_score_slope: float = 0.1
    gamma_neighbor_score_slope: float = 0.2
    eta_mean: float = 0.0
    eta_sd: float = 0.0
    noise_sd: float = 0.0
    # Random DGP options are inactive for legacy checkpoints.
    random_nointerference_prob: float = 0.5
    random_noise_sd: float = 1.0
    random_prior_version: str = "random_lpe_two_regime_v3_gh256"

    def __post_init__(self) -> None:
        if not math.isfinite(self.interference_lambda) or self.interference_lambda < 0:
            raise ValueError("interference_lambda must be nonnegative and finite.")
        if self.interference_lambda != 1.0 and self.dgp_version != "a_group_fixed_v1":
            raise ValueError("interference_lambda only applies to a_group_fixed_v1.")
        if self.dgp_version in ("a_group_random_gamma_v1", "a_group_random_coeff_v1"):
            if not (math.isfinite(self.spillover_gamma_min) and math.isfinite(self.spillover_gamma_max)
                    and self.spillover_gamma_min < self.spillover_gamma_max):
                raise ValueError("spillover gamma bounds must be finite and increasing.")
            if self.spillover_gamma is not None and not math.isfinite(self.spillover_gamma):
                raise ValueError("spillover_gamma must be finite.")
        elif (self.spillover_gamma is not None or self.spillover_gamma_min != -3.6
              or self.spillover_gamma_max != 0.0):
            raise ValueError("spillover gamma options require an A-group random-coefficient prior.")
        if self.dgp_version == "a_group_random_coeff_v1":
            if not (math.isfinite(self.neighbor_beta_min) and math.isfinite(self.neighbor_beta_max)
                    and self.neighbor_beta_min < self.neighbor_beta_max):
                raise ValueError("neighbor beta bounds must be finite and increasing.")
            if self.neighbor_beta is not None and not math.isfinite(self.neighbor_beta):
                raise ValueError("neighbor_beta must be finite.")
            if (self.neighbor_beta is None) != (self.spillover_gamma is None):
                raise ValueError("Fix both beta and gamma for evaluation, or draw both for training.")
        elif (self.neighbor_beta is not None or self.neighbor_beta_min != -6.
              or self.neighbor_beta_max != 0.):
            raise ValueError("neighbor beta options require a_group_random_coeff_v1.")
        if self.n_units < 2:
            raise ValueError("n_units must be at least 2.")
        if self.dgp_version not in {
            "gao_ding_design2_outcome",
            "tnet_linear_neighbor_covariate",
            "legacy_v7",
            "random_lpe_v1",
            "a_group_fixed_v1",
            "a_group_random_gamma_v1",
            "a_group_random_coeff_v1",
        }:
            raise ValueError("Unsupported dgp_version.")
        if self.dgp_version == "random_lpe_v1":
            if self.n_units < 24:
                raise ValueError("random_lpe_v1 requires n_units >= 24 (1000 for current formal runs).")
            if self.treatment_prob != 0.5:
                raise ValueError("random_lpe_v1 uses independent Bernoulli(0.5) treatment.")
            if not math.isfinite(self.random_nointerference_prob) or not 0 <= self.random_nointerference_prob <= 1:
                raise ValueError("random_nointerference_prob must be finite and in [0,1].")
            if not math.isfinite(self.random_noise_sd) or self.random_noise_sd <= 0:
                raise ValueError("random_noise_sd must be positive and finite.")
            if self.random_prior_version != "random_lpe_two_regime_v3_gh256":
                raise ValueError("Unsupported random_prior_version for the coupled two-regime protocol; use the original package for old checkpoints and prepare new data for this protocol.")
            if self.x_sd != 1.0:
                raise ValueError("random_lpe_v1 restores fixed X~N(0,1); x_sd must be 1.")
            if self.noise_sd != 0.0:
                raise ValueError("Use --random-noise-sd for random_lpe_v1; legacy --noise-sd is extra noise and must stay zero.")
        if self.dgp_version in ("a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
            if self.n_units < 24 or self.treatment_prob != 0.5 or self.x_sd != 1.0:
                raise ValueError("A group requires N>=24, iid Bernoulli(0.5) treatment and X~N(0,1).")
            if self.noise_sd != 0.0:
                raise ValueError("A group fixes epsilon SD=4; legacy extra --noise-sd must stay zero.")
        if self.graph_family not in {"er", "configuration", "rgg", "sbm", "mixed"}:
            raise ValueError("graph_family must be one of er/configuration/rgg/sbm/mixed.")
        if not 0.0 <= self.sbm_between_probability < 1.0:
            raise ValueError("sbm_between_probability must lie in [0, 1).")
        if self.configuration_degree_sigma < 0.0:
            raise ValueError("configuration_degree_sigma must be nonnegative.")
        if self.configuration_max_degree < 1:
            raise ValueError("configuration_max_degree must be positive.")
        if not 0.0 < self.er_edge_probability < 1.0:
            raise ValueError(
                "er_edge_probability must lie strictly between 0 and 1."
            )
        if not 0.0 < self.treatment_prob < 1.0:
            raise ValueError("treatment_prob must lie strictly between 0 and 1.")
        for name in (
            "x_sd",
            "baseline_sd",
            "unit_sd",
            "tau_episode_sd",
            "tau_unit_sd",
            "gamma_sd",
            "eta_sd",
            "noise_sd",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.graph_protocol not in {"resampled", "fixed_er_v1"}:
            raise ValueError("Unsupported graph_protocol.")
        if self.graph_protocol == "fixed_er_v1":
            if self.graph_family != "er" or self.dgp_version not in ("a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
                raise ValueError("Fixed ER protocol requires graph_family=er and the A-group DGP.")
            from pfn_interference.graph import _validate_parameters, make_graph_record, adjacency_from_config
            _validate_parameters(self.n_units, self.er_edge_probability, self.fixed_graph_seed)
            if self.fixed_graph_upper_hex is None:
                record = make_graph_record(n_units=self.n_units,
                    edge_probability=self.er_edge_probability, seed=self.fixed_graph_seed)
                object.__setattr__(self, "fixed_graph_upper_hex", record["upper_hex"])
            adjacency_from_config(self)
        elif self.fixed_graph_upper_hex is not None:
            raise ValueError("Saved fixed graph requires graph_protocol=fixed_er_v1.")


def data_config_from_mapping(values: Mapping[str, object]) -> DataConfig:
    """Load ER-only graph-configuration metadata.

    Checkpoints written before the neighbor-covariate DGP did not store the
    new score coefficients.  Treat those mappings as ``legacy_v7`` so old
    checkpoints reproduce their original synthetic prior instead of silently
    inheriting the new defaults.
    """

    data = dict(values)
    if data.get("graph_protocol") == "fixed_er_v1" and not data.get("fixed_graph_upper_hex"):
        raise ValueError("Saved fixed ER configuration is missing its adjacency; refusing to regenerate the graph.")
    if "graph_family" not in data:
        data["graph_family"] = "er"
    if "dgp_version" not in data:
        data.update(
            dgp_version="legacy_v7",
            own_score_baseline_coefficient=0.0,
            neighbor_score_baseline_coefficient=0.0,
            tau_own_score_slope=0.0,
            tau_neighbor_score_slope=0.0,
            gamma_own_score_slope=0.0,
            gamma_neighbor_score_slope=0.0,
        )
    return DataConfig(**data)


@dataclass(frozen=True)
class ModelConfig:
    """Permutation-equivariant graph encoder plus a shared CEPO GMM head."""

    input_dim: int = TOKEN_DIM
    query_dim: int = 2
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 10
    ffn_dim: int = 512
    dropout: float = 0.0
    gmm_n_components: int = 5
    gmm_min_sigma: float = 1e-3
    gmm_pi_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim != TOKEN_DIM:
            raise ValueError("Each node token has four factual features plus normalized degree.")
        if self.query_dim != 2:
            raise ValueError("CEPO queries use two binary coordinates (d,s).")
        if self.d_model <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("Transformer dimensions must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.gmm_n_components <= 0:
            raise ValueError("gmm_n_components must be positive.")
        if self.gmm_min_sigma <= 0.0:
            raise ValueError("gmm_min_sigma must be positive.")
        if self.gmm_pi_temperature <= 0.0:
            raise ValueError("gmm_pi_temperature must be positive.")



@dataclass(frozen=True)
class TrainConfig:
    """Epoch-based optimization and validation-NLL scheduler settings."""

    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    training_seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimizer settings.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must lie strictly between 0 and 1.")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be nonnegative.")


def make_gmm_head(config: ModelConfig) -> nn.Sequential:
    """Build one task-specific one-dimensional GMM prediction head."""

    return nn.Sequential(
        nn.Linear(config.d_model, config.ffn_dim, bias=False),
        nn.GELU(),
        nn.Linear(
            config.ffn_dim,
            3 * config.gmm_n_components,
            bias=False,
        ),
    )


def split_gmm_predictions(
    raw_predictions: torch.Tensor,
    config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """Convert raw [...,4,3K] CEPO outputs into valid GMM parameters."""

    if raw_predictions.ndim < 4 or raw_predictions.shape[-2] != 4:
        raise ValueError(
            "raw_predictions must have shape [batch, units, ..., 4, 3K]."
        )
    expected = 3 * config.gmm_n_components
    if raw_predictions.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} GMM outputs, got {raw_predictions.shape[-1]}."
        )
    pi_logits, mu, raw_sigma = raw_predictions.chunk(3, dim=-1)
    pi = torch.softmax(pi_logits / config.gmm_pi_temperature, dim=-1)
    sigma = F.softplus(raw_sigma) + config.gmm_min_sigma
    return {"gmm_pi": pi, "gmm_mu": mu, "gmm_sigma": sigma}


def _validate_gmm_inputs(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if pi.shape != mu.shape or pi.shape != sigma.shape:
        raise ValueError("pi, mu, and sigma must have matching shapes.")
    if pi.ndim < 2:
        raise ValueError("GMM tensors must include a final component dimension.")
    if target.shape != pi.shape[:-1]:
        raise ValueError("target must match GMM tensors except for components.")
    if not bool(torch.all(torch.isfinite(pi))):
        raise ValueError("pi contains non-finite values.")
    if not bool(torch.all(torch.isfinite(mu))):
        raise ValueError("mu contains non-finite values.")
    if not bool(torch.all(torch.isfinite(sigma))):
        raise ValueError("sigma contains non-finite values.")
    if not bool(torch.all(sigma > 0.0)):
        raise ValueError("sigma must be strictly positive.")


def gmm_log_prob(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return log p(target) under a one-dimensional Gaussian mixture."""

    _validate_gmm_inputs(pi, mu, sigma, target)
    eps = torch.finfo(pi.dtype).tiny
    normalized_pi = pi.clamp_min(eps)
    normalized_pi = normalized_pi / normalized_pi.sum(dim=-1, keepdim=True)
    safe_sigma = sigma.clamp_min(eps)
    target_expanded = target.unsqueeze(-1)
    z = (target_expanded - mu) / safe_sigma
    component_log_prob = (
        -0.5 * z.square()
        - torch.log(safe_sigma)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return torch.logsumexp(torch.log(normalized_pi) + component_log_prob, dim=-1)


def gmm_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute -log sum_k pi_k Normal(target; mu_k, sigma_k^2)."""

    nll = -gmm_log_prob(pi, mu, sigma, target)
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError("reduction must be one of: none, mean, sum.")


def gmm_posterior_mean(
    pi: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    if pi.shape != mu.shape:
        raise ValueError("pi and mu must have matching shapes.")
    return (pi * mu).sum(dim=-1)


def _gmm_cdf(
    value: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    standardized = (value.unsqueeze(-1) - mu) / (sigma * math.sqrt(2.0))
    normal_cdf = 0.5 * (1.0 + torch.erf(standardized))
    return (pi * normal_cdf).sum(dim=-1)


def gmm_posterior_interval(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    mass: float = 0.9,
    iterations: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic equal-tail interval from the GMM CDF by bisection."""

    if pi.shape != mu.shape or pi.shape != sigma.shape:
        raise ValueError("pi, mu, and sigma must have matching shapes.")
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must lie strictly between 0 and 1.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    lower_bound = (mu - 12.0 * sigma).min(dim=-1).values
    upper_bound = (mu + 12.0 * sigma).max(dim=-1).values

    def quantile(probability: float) -> torch.Tensor:
        lo = lower_bound.clone()
        hi = upper_bound.clone()
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            cdf = _gmm_cdf(mid, pi, mu, sigma)
            lo = torch.where(cdf < probability, mid, lo)
            hi = torch.where(cdf >= probability, mid, hi)
        return 0.5 * (lo + hi)

    alpha = (1.0 - mass) / 2.0
    return quantile(alpha), quantile(1.0 - alpha)


def _try_import_torch_musa() -> bool:
    try:
        import torch_musa  # noqa: F401
    except ModuleNotFoundError:
        return False
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("musa") if _try_import_torch_musa() else torch.device("cpu")
    if requested == "musa" or requested.startswith("musa:"):
        if not _try_import_torch_musa():
            raise RuntimeError(
                "MUSA was requested but torch_musa/torch.musa is unavailable."
            )
        return torch.device(requested)
    raise ValueError("device must be one of: auto, cpu, musa, musa:N.")


def distributed_backend_for_device(device: torch.device) -> str:
    """Return the torch.distributed backend for the selected accelerator."""

    return "mccl" if device.type in {"musa", "privateuseone"} else "gloo"


def process_group_init_kwargs(
    runtime: DistributedRuntime,
    device: torch.device,
) -> Dict[str, object]:
    """Build process-group arguments with an explicit accelerator mapping."""

    kwargs: Dict[str, object] = {
        "backend": distributed_backend_for_device(device),
        "rank": runtime.rank,
        "world_size": runtime.world_size,
    }
    if device.type in {"musa", "privateuseone"}:
        kwargs["device_id"] = device
    return kwargs


def initialize_distributed(
    device_name: str,
) -> tuple[DistributedRuntime, torch.device]:
    """Initialize torchrun DDP when WORLD_SIZE is greater than one."""

    runtime = distributed_runtime_from_env()
    if not runtime.is_distributed:
        return runtime, resolve_device(device_name)

    requested = device_name.strip().lower()
    if requested == "cpu":
        device = torch.device("cpu")
    else:
        if not _try_import_torch_musa():
            raise RuntimeError(
                "Distributed MUSA training was requested but torch_musa is unavailable."
            )
        set_device = getattr(torch.musa, "set_device", None)
        if not callable(set_device):
            raise RuntimeError("torch.musa.set_device is unavailable.")
        # This runner is single-node: global rank 0..7 maps directly to
        # the eight visible MUSA devices.  No extra per-process device rank is used.
        set_device(runtime.rank)
        device = torch.device(f"musa:{runtime.rank}")

    # CPU control plane first. No MCCL collective is launched while CSV data
    # or validation diagnostics are being prepared on individual ranks.
    if not dist.is_gloo_available():
        raise RuntimeError("Distributed data preparation requires the Gloo CPU backend.")
    dist.init_process_group(
        **process_group_init_kwargs(runtime, torch.device("cpu")),
        timeout=timedelta(seconds=int(os.environ.get("PFN_CONTROL_TIMEOUT", "3600"))),
    )
    return runtime, device


def distributed_barrier(
    runtime: DistributedRuntime,
    device: torch.device,
) -> None:
    """Synchronize ranks while explicitly identifying each accelerator."""

    if not runtime.is_distributed:
        return
    # Default group is Gloo: idle ranks wait on CPU, not inside a MUSA kernel.
    dist.barrier()


def cleanup_distributed(runtime: DistributedRuntime) -> None:
    """Destroy an initialized process group."""

    if runtime.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    """Return the underlying module from DDP-like wrappers."""

    return getattr(model, "module", model)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "musa") and getattr(torch.musa, "is_available", lambda: False)():
        manual_seed_all = getattr(torch.musa, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def _make_generator(seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator.manual_seed(seed)
    return generator


def _generate_er_adjacency(
    n_units: int,
    batch_size: int,
    *,
    edge_probability: object,
    generator: torch.Generator,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate loop-free undirected ER graphs with no isolated nodes."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    probabilities = torch.as_tensor(edge_probability, dtype=torch.float32)
    if probabilities.ndim == 0:
        probabilities = probabilities.expand(batch_size).clone()
    if probabilities.shape != (batch_size,):
        raise ValueError("edge_probability must be scalar or have shape [batch_size].")
    if bool(((probabilities <= 0.0) | (probabilities >= 1.0)).any()):
        raise ValueError("edge_probability must lie strictly between 0 and 1.")

    adjacency = torch.zeros(batch_size, n_units, n_units, dtype=torch.float32)
    pending_rows = torch.arange(batch_size)
    edge_start, edge_end = torch.triu_indices(n_units, n_units, offset=1)

    for _ in range(max_attempts):
        if pending_rows.numel() == 0:
            return adjacency
        pending_count = int(pending_rows.numel())
        sampled_edges = (
            torch.rand(
                pending_count,
                edge_start.numel(),
                generator=generator,
            )
            < probabilities[pending_rows].unsqueeze(1)
        ).to(torch.float32)
        candidates = torch.zeros(
            pending_count,
            n_units,
            n_units,
            dtype=torch.float32,
        )
        candidates[:, edge_start, edge_end] = sampled_edges
        candidates[:, edge_end, edge_start] = sampled_edges
        valid = (candidates.sum(dim=-1) >= 1).all(dim=-1)
        if bool(valid.any()):
            adjacency[pending_rows[valid]] = candidates[valid]
        pending_rows = pending_rows[~valid]
        if pending_rows.numel() == 0:
            return adjacency

    raise RuntimeError(
        "ER graph generation still had "
        f"{pending_rows.numel()} graph(s) with isolated nodes after "
        f"{max_attempts} attempts."
    )


def _target_mean_degree(config: DataConfig) -> float:
    return float((config.n_units - 1) * config.er_edge_probability)


def _sample_mixed_target_mean_degrees(
    batch_size: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one continuous target mean degree per mixed-prior episode."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    width = MIXED_TARGET_MEAN_DEGREE_MAX - MIXED_TARGET_MEAN_DEGREE_MIN
    return MIXED_TARGET_MEAN_DEGREE_MIN + width * torch.rand(
        batch_size, generator=generator
    )


def _sample_mixed_graph_types(
    batch_size: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one of the four graph families independently for each episode."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    family_lookup = torch.tensor(GRAPH_FAMILY_ORDER, dtype=torch.long)
    family_index = torch.randint(
        0, len(GRAPH_FAMILY_ORDER), (batch_size,), generator=generator
    )
    return family_lookup[family_index]


def _coerce_target_mean_degrees(
    config: DataConfig,
    batch_size: int,
    target_mean_degree: Optional[torch.Tensor],
) -> torch.Tensor:
    if target_mean_degree is None:
        return torch.full(
            (batch_size,), _target_mean_degree(config), dtype=torch.float32
        )
    target = torch.as_tensor(target_mean_degree, dtype=torch.float32)
    if target.ndim == 0:
        target = target.expand(batch_size).clone()
    if target.shape != (batch_size,):
        raise ValueError("target_mean_degree must be scalar or have shape [batch_size].")
    if bool((target <= 0.0).any()):
        raise ValueError("target_mean_degree must be positive.")
    if bool((target >= config.n_units).any()):
        raise ValueError("target_mean_degree must be smaller than n_units.")
    return target


def _balanced_graph_type_schedule(
    batch_size: int,
    *,
    dataset_id_offset: int = 0,
) -> torch.Tensor:
    """Alternate ER/configuration deterministically for exact 50:50 balance."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    family_count = len(GRAPH_FAMILY_ORDER)
    values = [
        GRAPH_FAMILY_ORDER[(dataset_id_offset + index) % family_count]
        for index in range(batch_size)
    ]
    return torch.tensor(values, dtype=torch.long)


def _single_family_graph_types(graph_family: str, batch_size: int) -> torch.Tensor:
    mapping = {
        "er": ER_GRAPH,
        "configuration": CONFIGURATION_GRAPH,
        "rgg": RGG_GRAPH,
        "sbm": SBM_GRAPH,
    }
    if graph_family not in mapping:
        raise ValueError("graph_family is not a single graph family.")
    return torch.full((batch_size,), mapping[graph_family], dtype=torch.long)


def _generate_sbm_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate equal-size two-block SBM graphs at episode-specific mean degree."""

    n_units = config.n_units
    target_degree = _coerce_target_mean_degrees(
        config, batch_size, target_mean_degree
    )
    block0 = (n_units + 1) // 2
    block1 = n_units - block0
    p_out = float(config.sbm_between_probability)
    within_pair_degree_weight = block0 * (block0 - 1) + block1 * (block1 - 1)
    between_pair_degree_weight = 2 * block0 * block1
    if within_pair_degree_weight <= 0:
        raise ValueError("SBM requires at least two units in one block.")
    p_in = (
        n_units * target_degree - between_pair_degree_weight * p_out
    ) / within_pair_degree_weight
    if bool(((p_in <= 0.0) | (p_in >= 1.0)).any()):
        raise ValueError(
            "SBM probabilities cannot match the target mean degree; "
            "adjust p_out or the target degree range."
        )

    labels = torch.zeros(n_units, dtype=torch.long)
    labels[block0:] = 1
    edge_start, edge_end = torch.triu_indices(n_units, n_units, offset=1)
    same_block = labels[edge_start] == labels[edge_end]
    adjacency = torch.zeros(batch_size, n_units, n_units, dtype=torch.float32)
    pending = torch.arange(batch_size)
    for _ in range(max_attempts):
        if pending.numel() == 0:
            return adjacency
        count = int(pending.numel())
        pending_p_in = p_in[pending].unsqueeze(1)
        edge_prob = torch.where(
            same_block.unsqueeze(0),
            pending_p_in,
            torch.full((count, 1), p_out, dtype=torch.float32),
        )
        sampled = (
            torch.rand(count, edge_start.numel(), generator=generator) < edge_prob
        ).to(torch.float32)
        candidates = torch.zeros(count, n_units, n_units, dtype=torch.float32)
        candidates[:, edge_start, edge_end] = sampled
        candidates[:, edge_end, edge_start] = sampled
        valid = (candidates.sum(dim=-1) >= 1).all(dim=-1)
        if bool(valid.any()):
            adjacency[pending[valid]] = candidates[valid]
        pending = pending[~valid]
    raise RuntimeError(
        f"SBM generation still had {pending.numel()} graph(s) with isolated nodes "
        f"after {max_attempts} attempts."
    )


def _generate_rgg_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate 2-D random geometric graphs at episode-specific mean degree."""

    n_units = config.n_units
    target_degree = _coerce_target_mean_degrees(
        config, batch_size, target_mean_degree
    )
    radius = torch.sqrt(target_degree / ((n_units - 1) * math.pi))
    if config.rgg_torus and bool((radius >= 0.5).any()):
        raise ValueError("Toroidal RGG radius must be below 0.5 for this calibration.")
    adjacency = torch.zeros(batch_size, n_units, n_units, dtype=torch.float32)
    pending = torch.arange(batch_size)
    for _ in range(max_attempts):
        if pending.numel() == 0:
            return adjacency
        count = int(pending.numel())
        positions = torch.rand(count, n_units, 2, generator=generator)
        diff = torch.abs(positions[:, :, None, :] - positions[:, None, :, :])
        if config.rgg_torus:
            diff = torch.minimum(diff, 1.0 - diff)
        distance_sq = (diff * diff).sum(dim=-1)
        pending_radius_sq = radius[pending].square().view(count, 1, 1)
        candidates = (distance_sq <= pending_radius_sq).to(torch.float32)
        diagonal = torch.arange(n_units)
        candidates[:, diagonal, diagonal] = 0.0
        valid = (candidates.sum(dim=-1) >= 1).all(dim=-1)
        if bool(valid.any()):
            adjacency[pending[valid]] = candidates[valid]
        pending = pending[~valid]
    raise RuntimeError(
        f"RGG generation still had {pending.numel()} graph(s) with isolated nodes "
        f"after {max_attempts} attempts."
    )


def _configuration_degree_sequence(
    config: DataConfig,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[float] = None,
) -> torch.Tensor:
    """Draw a heterogeneous graphical target degree sequence at a target mean."""

    n_units = config.n_units
    max_degree = min(int(config.configuration_max_degree), n_units - 1)
    if target_mean_degree is None:
        target_mean_degree = _target_mean_degree(config)
    target_total = int(round(n_units * float(target_mean_degree)))
    target_total = min(target_total, n_units * max_degree)
    target_total = max(target_total, n_units)
    if target_total % 2:
        target_total += 1 if target_total < n_units * max_degree else -1

    weights = torch.exp(
        config.configuration_degree_sigma
        * torch.randn(n_units, generator=generator)
    )
    degrees = torch.round(weights / weights.sum() * target_total).to(torch.long)
    degrees.clamp_(1, max_degree)
    delta = target_total - int(degrees.sum().item())
    while delta != 0:
        if delta > 0:
            candidates = torch.nonzero(degrees < max_degree, as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise RuntimeError("Cannot increase configuration degrees to target total.")
            order = candidates[torch.randperm(candidates.numel(), generator=generator)]
            take = min(delta, int(order.numel()))
            degrees[order[:take]] += 1
            delta -= take
        else:
            candidates = torch.nonzero(degrees > 1, as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise RuntimeError("Cannot decrease configuration degrees to target total.")
            order = candidates[torch.randperm(candidates.numel(), generator=generator)]
            take = min(-delta, int(order.numel()))
            degrees[order[:take]] -= 1
            delta += take
    return degrees


def _generate_configuration_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = 100,
) -> torch.Tensor:
    """Generate simple heterogeneous-degree graphs at episode-specific mean degree."""

    n_units = config.n_units
    target_degree = _coerce_target_mean_degrees(
        config, batch_size, target_mean_degree
    )
    adjacency = torch.zeros(batch_size, n_units, n_units, dtype=torch.float32)
    for graph_index in range(batch_size):
        graph = None
        for _ in range(max_attempts):
            degrees = _configuration_degree_sequence(
                config,
                generator=generator,
                target_mean_degree=float(target_degree[graph_index].item()),
            )
            if not nx.is_graphical(degrees.tolist(), method="eg"):
                continue
            nx_seed = int(
                torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
            )
            try:
                graph = nx.havel_hakimi_graph(degrees.tolist())
                permutation = torch.randperm(n_units, generator=generator).tolist()
                graph = nx.relabel_nodes(
                    graph,
                    {index: permutation[index] for index in range(n_units)},
                    copy=True,
                )
                edge_count = graph.number_of_edges()
                if edge_count > 1:
                    nx.double_edge_swap(
                        graph,
                        nswap=edge_count,
                        max_tries=max(20 * edge_count, 100),
                        seed=nx_seed,
                    )
            except (nx.NetworkXAlgorithmError, nx.NetworkXError, nx.NetworkXUnfeasible):
                graph = None
                continue
            if graph.number_of_nodes() != n_units:
                graph.add_nodes_from(range(n_units))
            if min(dict(graph.degree()).values()) >= 1:
                break
            graph = None
        if graph is None:
            raise RuntimeError(
                "Configuration-model generation failed to produce a simple graph "
                f"after {max_attempts} degree-sequence attempts."
            )
        edges = list(graph.edges())
        if edges:
            edge_tensor = torch.tensor(edges, dtype=torch.long)
            adjacency[graph_index, edge_tensor[:, 0], edge_tensor[:, 1]] = 1.0
            adjacency[graph_index, edge_tensor[:, 1], edge_tensor[:, 0]] = 1.0
    return adjacency


def generate_graphs(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    graph_types: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one graph per episode from the four-family mixed prior.

    For ``graph_family="mixed"`` the sampling hierarchy is intentionally:
    (1) sample one target mean degree from Uniform(4, 10) per episode;
    (2) sample one graph family uniformly from ER/configuration/RGG/SBM;
    (3) generate the graph conditional on that episode's degree and family.
    Single-family modes retain their legacy fixed-density behavior.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # The degree draw comes first by design.  Explicit graph_types may override
    # only step (2), e.g. for controlled evaluation, but not the mixed degree prior.
    if config.graph_family == "mixed":
        target_mean_degree = _sample_mixed_target_mean_degrees(
            batch_size, generator=generator
        )
    else:
        target_mean_degree = torch.full(
            (batch_size,), _target_mean_degree(config), dtype=torch.float32
        )

    if graph_types is None:
        if config.graph_family == "mixed":
            graph_type = _sample_mixed_graph_types(
                batch_size, generator=generator
            )
        else:
            graph_type = _single_family_graph_types(config.graph_family, batch_size)
    else:
        graph_type = torch.as_tensor(graph_types, dtype=torch.long).clone()
        if graph_type.shape != (batch_size,):
            raise ValueError("graph_types must have shape [batch_size].")
        if not set(graph_type.tolist()).issubset(set(GRAPH_FAMILY_ORDER)):
            raise ValueError("graph_types contains an unsupported graph-family code.")

    adjacency = torch.zeros(
        batch_size, config.n_units, config.n_units, dtype=torch.float32
    )
    for graph_value in GRAPH_FAMILY_ORDER:
        indices = torch.nonzero(graph_type == graph_value, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        count = int(indices.numel())
        family_target_degree = target_mean_degree[indices]
        if graph_value == ER_GRAPH:
            family_adjacency = _generate_er_adjacency(
                config.n_units,
                count,
                edge_probability=family_target_degree / float(config.n_units - 1),
                generator=generator,
            )
        elif graph_value == CONFIGURATION_GRAPH:
            family_adjacency = _generate_configuration_adjacency(
                config,
                count,
                generator=generator,
                target_mean_degree=family_target_degree,
            )
        elif graph_value == RGG_GRAPH:
            family_adjacency = _generate_rgg_adjacency(
                config,
                count,
                generator=generator,
                target_mean_degree=family_target_degree,
            )
        elif graph_value == SBM_GRAPH:
            family_adjacency = _generate_sbm_adjacency(
                config,
                count,
                generator=generator,
                target_mean_degree=family_target_degree,
            )
        else:  # pragma: no cover - GRAPH_FAMILY_ORDER is closed above.
            raise RuntimeError("Unsupported graph-family code.")
        adjacency[indices] = family_adjacency

    star_center = torch.full((batch_size,), -1, dtype=torch.long)
    return adjacency, graph_type, star_center


def _sample_treatment_with_overlap(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    treatment = (
        torch.rand(batch_size, config.n_units, generator=generator)
        < config.treatment_prob
    )
    invalid = (treatment.sum(dim=1) == 0) | (
        treatment.sum(dim=1) == config.n_units
    )
    while bool(invalid.any()):
        count = int(invalid.sum().item())
        treatment[invalid] = (
            torch.rand(count, config.n_units, generator=generator)
            < config.treatment_prob
        )
        invalid = (treatment.sum(dim=1) == 0) | (
            treatment.sum(dim=1) == config.n_units
        )
    return treatment.to(torch.float32)


def generate_batch(
    config: DataConfig,
    batch_size: int,
    *,
    seed: Optional[int],
    device: torch.device,
    graph_types: Optional[torch.Tensor] = None,
    task_ids=None,
    stream: Optional[str] = None,
) -> TensorBatch:
    """Generate factual data and support-valid general exposure queries."""

    if config.dgp_version in ("random_lpe_v1", "a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
        from pfn_interference.simulation.csv_tasks import generate_random_batch, TRAIN_STREAM
        ids = range(batch_size) if task_ids is None else list(task_ids)
        if len(ids) != batch_size:
            raise ValueError("task_ids length must equal batch_size.")
        if seed is None:
            raise ValueError("The versioned random DGP requires an explicit seed.")
        batch = generate_random_batch(config, task_ids=ids, seed=seed,
                                     stream=stream or TRAIN_STREAM,
                                     device=device, graph_types=graph_types)
        batch["design_treatment_prob"] = torch.full((batch_size,), config.treatment_prob,
                                                    dtype=torch.float32, device=device)
        return batch
    if task_ids is not None or stream is not None:
        raise ValueError("task_ids/stream apply only to random_lpe_v1 or a_group_fixed_v1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    generator = _make_generator(seed)
    adjacency, graph_type, star_center = generate_graphs(
        config, batch_size, generator=generator, graph_types=graph_types
    )
    treatment = _sample_treatment_with_overlap(
        config, batch_size, generator=generator
    )
    degree = adjacency.sum(dim=-1)
    degree_norm = degree / float(config.n_units - 1)
    exposure = torch.bmm(adjacency, treatment.unsqueeze(-1)).squeeze(-1) / degree

    x = config.x_sd * torch.randn(
        batch_size, config.n_units, generator=generator
    )
    covariate_score = torch.sigmoid(x)
    neighbor_covariate_score = torch.bmm(
        adjacency, covariate_score.unsqueeze(-1)
    ).squeeze(-1) / degree
    centered_score = covariate_score - 0.5
    centered_neighbor_score = neighbor_covariate_score - 0.5

    observation_noise = torch.zeros_like(x)
    if config.dgp_version == "gao_ding_design2_outcome":
        neighbor_x = torch.bmm(adjacency, x.unsqueeze(-1)).squeeze(-1) / degree
        epsilon = 4.0 * torch.randn(
            batch_size, config.n_units, generator=generator
        )
        baseline = 1.0 - x - 3.0 * neighbor_x
        observation_noise = epsilon
        tau = 6.0 + 0.2 * torch.exp(x)
        gamma = torch.full_like(x, -0.9)
        eta = torch.zeros_like(x)
    else:
        baseline = (
            config.baseline_sd * torch.randn(batch_size, 1, generator=generator)
            + config.unit_sd
            * torch.randn(batch_size, config.n_units, generator=generator)
            + config.x_outcome_coefficient * x
            + config.own_score_baseline_coefficient * covariate_score
            + config.neighbor_score_baseline_coefficient
            * neighbor_covariate_score
        )
        tau = (
            config.tau_mean
            + config.tau_episode_sd
            * torch.randn(batch_size, 1, generator=generator)
            + config.tau_unit_sd
            * torch.randn(batch_size, config.n_units, generator=generator)
            + config.tau_own_score_slope * centered_score
            + config.tau_neighbor_score_slope * centered_neighbor_score
        )
        gamma = (
            config.gamma_mean
            + config.gamma_sd
            * torch.randn(batch_size, config.n_units, generator=generator)
            + config.gamma_own_score_slope * centered_score
            + config.gamma_neighbor_score_slope * centered_neighbor_score
        )
        eta = config.eta_mean + config.eta_sd * torch.randn(
            batch_size, config.n_units, generator=generator
        )

    def structural_outcome(t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if t.shape != e.shape:
            raise ValueError("treatment and exposure query tensors must match.")
        if t.ndim == 2:
            return baseline + tau * t + gamma * e + eta * t * e
        if t.ndim == 3:
            return (
                baseline.unsqueeze(-1)
                + tau.unsqueeze(-1) * t
                + gamma.unsqueeze(-1) * e
                + eta.unsqueeze(-1) * t * e
            )
        raise ValueError("structural_outcome expects [B,N] or [B,N,Q] tensors.")

    factual_mean = structural_outcome(treatment, exposure)
    y_obs = factual_mean + observation_noise + config.noise_sd * torch.randn(
        batch_size, config.n_units, generator=generator
    )

    ones = torch.ones_like(treatment)
    zeros = torch.zeros_like(treatment)
    low_exposure, high_exposure = sample_majority_exposures(
        degree,
        treatment_prob=config.treatment_prob,
        generator=generator,
    )
    observed_counts = torch.round(exposure * degree).to(torch.int64)
    majority_arm = (
        observed_counts > torch.floor(degree / 2).to(torch.int64)
    ).to(torch.int64)

    # Pointwise training queries sample exact exposures from the two majority
    # arms. Formal ITE/ATE evaluation integrates over every exact count level.
    t_a = torch.stack([ones, ones, ones], dim=-1)
    e_a = torch.stack([low_exposure, high_exposure, high_exposure], dim=-1)
    t_b = torch.stack([zeros, ones, zeros], dim=-1)
    e_b = torch.stack([low_exposure, low_exposure, low_exposure], dim=-1)
    queries = torch.stack([t_a, e_a, t_b, e_b], dim=-1)

    outcome_a = structural_outcome(t_a, e_a)
    outcome_b = structural_outcome(t_b, e_b)
    query_effect = outcome_a - outcome_b
    oracle_ite = oracle_ite_from_parameters(
        degree=degree,
        tau=tau,
        gamma=gamma,
        eta=eta,
        treatment_prob=config.treatment_prob,
    )
    oracle_graph_ate = oracle_ate(oracle_ite)

    tokens = build_observed_tokens(
        torch.stack([x, treatment, y_obs, exposure, degree_norm], dim=-1)
    )

    return {
        "tokens": tokens.to(device),
        "queries": queries.to(device),
        "adjacency": adjacency.to(device),
        "graph_type": graph_type.to(device),
        "star_center": star_center.to(device),
        "x": x.to(device),
        "covariate_score": covariate_score.to(device),
        "neighbor_covariate_score": neighbor_covariate_score.to(device),
        "observed_treatment": treatment.to(device),
        "observed_exposure": exposure.to(device),
        "degree": degree.to(device),
        "majority_arm": majority_arm.to(device),
        "low_sampled_exposure": low_exposure.to(device),
        "high_sampled_exposure": high_exposure.to(device),
        "y_obs": y_obs.to(device),
        "outcome_a": outcome_a.to(device),
        "outcome_b": outcome_b.to(device),
        "query_effect": query_effect.to(device),
        "oracle_ite": oracle_ite.to(device),
        "oracle_ate": oracle_graph_ate.to(device),
        "structural_baseline": baseline.to(device),
        "design_treatment_prob": torch.full((batch_size,), config.treatment_prob,
                                             dtype=torch.float32, device=device),
        "tau": tau.to(device),
        "gamma": gamma.to(device),
        "eta": eta.to(device),
    }


def _encode_binary_mask_hex(mask: torch.Tensor) -> str:
    """Encode a one-dimensional binary tensor as a compact hexadecimal bitset."""

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError("mask must be binary.")
    value = 0
    for index in torch.nonzero(mask > 0, as_tuple=False).flatten().tolist():
        value |= 1 << int(index)
    return format(value, "x")


def _decode_binary_mask_hex(value: str, length: int) -> torch.Tensor:
    """Decode a hexadecimal bitset into a uint8 binary tensor."""

    if length <= 0:
        raise ValueError("length must be positive.")
    try:
        encoded = int(value, 16)
    except ValueError as error:
        raise ValueError("Invalid hexadecimal binary mask.") from error
    if encoded < 0 or (encoded >> length) != 0:
        raise ValueError("Encoded binary mask exceeds the requested length.")
    return torch.tensor(
        [(encoded >> index) & 1 for index in range(length)],
        dtype=torch.uint8,
    )


CSV_TENSOR_KEYS = (
    "tokens",
    "queries",
    "adjacency",
    "graph_type",
    "star_center",
    "x",
    "observed_treatment",
    "observed_exposure",
    "degree",
    "majority_arm",
    "low_sampled_exposure",
    "high_sampled_exposure",
    "y_obs",
    "outcome_a",
    "outcome_b",
    "query_effect",
    "oracle_ite",
    "structural_baseline",
    "tau",
    "gamma",
    "eta",
)


def _episode_csv_fieldnames(n_units: int = N_UNITS) -> list[str]:
    del n_units
    fields = [
        "schema_version", "n_units", "dataset_id", "unit_id",
        "graph_type", "star_center", "x", "observed_treatment",
        "observed_exposure", "degree", "majority_arm",
        "low_sampled_exposure", "high_sampled_exposure",
        "y_obs", "structural_baseline", "tau", "gamma", "eta",
        "oracle_direct_ite", "oracle_spillover_ite", "oracle_total_ite",
    ]
    fields.extend(f"token_{index}" for index in range(TOKEN_DIM))
    fields.append("adjacency_bits")
    for query_name in QUERY_TYPE_NAMES:
        fields.extend(f"{query_name}_query_{index}" for index in range(QUERY_DIM))
        fields.extend([
            f"{query_name}_outcome_a",
            f"{query_name}_outcome_b",
            f"{query_name}_effect",
        ])
    return fields


def save_episode_batch_csv(
    batch: TensorBatch,
    path: Path,
    *,
    dataset_id_offset: int = 0,
    append: bool = False,
) -> Path:
    """Write one compact CSV row per unit while preserving complete episodes."""

    missing = [key for key in CSV_TENSOR_KEYS if key not in batch]
    if missing:
        raise ValueError(f"batch is missing CSV fields: {missing}")
    tokens = batch["tokens"]
    if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
        raise ValueError("tokens must have shape [B, units, 5].")
    n_units = int(tokens.shape[1])
    if batch["queries"].shape[1:] != (n_units, 3, QUERY_DIM):
        raise ValueError("queries have incompatible episode dimensions.")
    if batch["adjacency"].shape[1:] != (n_units, n_units):
        raise ValueError("adjacency has incompatible episode dimensions.")
    batch_size = int(tokens.shape[0])
    cpu = {key: batch[key].detach().cpu() for key in CSV_TENSOR_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _episode_csv_fieldnames(n_units)
    mode = "a" if append else "w"
    if append and not path.is_file():
        raise FileNotFoundError("Cannot append to a CSV that does not exist.")
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for dataset_index in range(batch_size):
            dataset_id = dataset_id_offset + dataset_index
            for unit_index in range(n_units):
                row: Dict[str, object] = {
                    "schema_version": CSV_SCHEMA_VERSION,
                    "n_units": n_units,
                    "dataset_id": dataset_id,
                    "unit_id": unit_index,
                    "graph_type": int(cpu["graph_type"][dataset_index].item()),
                    "star_center": int(cpu["star_center"][dataset_index].item()),
                    "x": float(cpu["x"][dataset_index, unit_index].item()),
                    "observed_treatment": float(
                        cpu["observed_treatment"][dataset_index, unit_index].item()
                    ),
                    "observed_exposure": float(
                        cpu["observed_exposure"][dataset_index, unit_index].item()
                    ),
                    "degree": int(cpu["degree"][dataset_index, unit_index].item()),
                    "majority_arm": int(cpu["majority_arm"][dataset_index, unit_index].item()),
                    "low_sampled_exposure": float(
                        cpu["low_sampled_exposure"][dataset_index, unit_index].item()
                    ),
                    "high_sampled_exposure": float(
                        cpu["high_sampled_exposure"][dataset_index, unit_index].item()
                    ),
                    "y_obs": float(cpu["y_obs"][dataset_index, unit_index].item()),
                    "structural_baseline": float(cpu["structural_baseline"][dataset_index, unit_index].item()),
                    "tau": float(cpu["tau"][dataset_index, unit_index].item()),
                    "gamma": float(cpu["gamma"][dataset_index, unit_index].item()),
                    "eta": float(cpu["eta"][dataset_index, unit_index].item()),
                    "oracle_direct_ite": float(cpu["oracle_ite"][dataset_index, unit_index, 0].item()),
                    "oracle_spillover_ite": float(cpu["oracle_ite"][dataset_index, unit_index, 1].item()),
                    "oracle_total_ite": float(cpu["oracle_ite"][dataset_index, unit_index, 2].item()),
                    "adjacency_bits": _encode_binary_mask_hex(
                        cpu["adjacency"][dataset_index, unit_index]
                    ),
                }
                for feature_index in range(TOKEN_DIM):
                    row[f"token_{feature_index}"] = float(
                        cpu["tokens"][dataset_index, unit_index, feature_index].item()
                    )
                for query_index, query_name in enumerate(QUERY_TYPE_NAMES):
                    for feature_index in range(QUERY_DIM):
                        row[f"{query_name}_query_{feature_index}"] = float(
                            cpu["queries"][
                                dataset_index,
                                unit_index,
                                query_index,
                                feature_index,
                            ].item()
                        )
                    row[f"{query_name}_outcome_a"] = float(
                        cpu["outcome_a"][dataset_index, unit_index, query_index].item()
                    )
                    row[f"{query_name}_outcome_b"] = float(
                        cpu["outcome_b"][dataset_index, unit_index, query_index].item()
                    )
                    row[f"{query_name}_effect"] = float(
                        cpu["query_effect"][dataset_index, unit_index, query_index].item()
                    )
                writer.writerow(row)
    return path


def _validate_csv_fieldnames(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("CSV dataset has no header.")
    if fieldnames != _episode_csv_fieldnames():
        raise ValueError(
            f"CSV schema does not match {CSV_SCHEMA_VERSION}; regenerate the CSV "
            "with this version."
        )


def csv_schema_matches(path: Path, *, expected_n_units: int) -> bool:
    """Return whether a CSV uses the current schema and requested graph size."""

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_csv_fieldnames(reader.fieldnames)
            first = next(reader)
        return (
            first["schema_version"] == CSV_SCHEMA_VERSION
            and int(first["n_units"]) == expected_n_units
        )
    except (OSError, StopIteration, KeyError, TypeError, ValueError):
        return False


def compute_neighbor_covariate_score_chunked(
    adjacency: torch.Tensor,
    covariate_score: torch.Tensor,
    degree: torch.Tensor,
    *,
    chunk_size: int = CSV_ADJACENCY_FLOAT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute neighbor covariate means without materializing full float adjacency.

    CSV adjacency is stored compactly as uint8. Converting an entire large bank to
    float32 at once can allocate tens of GiB, so conversion is deliberately
    bounded to ``chunk_size`` episodes. The numerical definition is unchanged.
    """

    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape [episodes, units, units].")
    if covariate_score.shape != adjacency.shape[:2]:
        raise ValueError("covariate_score must have shape [episodes, units].")
    if degree.shape != adjacency.shape[:2]:
        raise ValueError("degree must have shape [episodes, units].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if bool((degree <= 0).any()):
        raise ValueError("degree must be positive for every unit.")

    result = torch.empty_like(covariate_score, dtype=torch.float32)
    episode_count = int(adjacency.shape[0])
    for start in range(0, episode_count, chunk_size):
        end = min(start + chunk_size, episode_count)
        local_adjacency = adjacency[start:end].to(torch.float32)
        local_score = covariate_score[start:end].to(torch.float32)
        result[start:end] = torch.bmm(
            local_adjacency, local_score.unsqueeze(-1)
        ).squeeze(-1) / degree[start:end].to(torch.float32)
    return result


def _load_episode_batch_csv_reference(
    path: Path,
    *,
    device: torch.device,
    expected_dataset_count: Optional[int] = None,
) -> tuple[TensorBatch, list[int]]:
    """Load a compact unit-row CSV and reconstruct dynamic episode tensors."""

    if not path.is_file():
        raise FileNotFoundError(f"CSV dataset does not exist: {path}")

    if expected_dataset_count is not None and expected_dataset_count <= 0:
        raise ValueError("expected_dataset_count must be positive when provided.")

    if expected_dataset_count is None:
        dataset_ids_seen: set[int] = set()
        n_units_values: set[int] = set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_csv_fieldnames(reader.fieldnames)
            for row in reader:
                if row["schema_version"] != CSV_SCHEMA_VERSION:
                    raise ValueError(
                        "CSV schema version is stale; regenerate the dataset."
                    )
                n_units_values.add(int(row["n_units"]))
                dataset_ids_seen.add(int(row["dataset_id"]))
        if not dataset_ids_seen:
            raise ValueError("CSV dataset is empty.")
        if len(n_units_values) != 1:
            raise ValueError("CSV rows contain inconsistent n_units values.")
        n_units = n_units_values.pop()
        dataset_ids = sorted(dataset_ids_seen)
        dataset_index_by_id = {value: index for index, value in enumerate(dataset_ids)}
        batch_size = len(dataset_ids)
    else:
        # Rank-local shard metadata already tells us the exact episode count.
        # Read only the first row to discover n_units, then fill tensors in one
        # full CSV pass instead of scanning millions of rows twice.
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_csv_fieldnames(reader.fieldnames)
            try:
                first = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV dataset is empty.") from exc
            if first["schema_version"] != CSV_SCHEMA_VERSION:
                raise ValueError("CSV schema version is stale; regenerate the dataset.")
            n_units = int(first["n_units"])
        batch_size = int(expected_dataset_count)
        dataset_ids = []
        dataset_index_by_id: Dict[int, int] = {}
    tokens = torch.empty(batch_size, n_units, TOKEN_DIM, dtype=torch.float32)
    queries = torch.empty(batch_size, n_units, 3, QUERY_DIM, dtype=torch.float32)
    adjacency = torch.empty(batch_size, n_units, n_units, dtype=torch.uint8)
    graph_type = torch.empty(batch_size, dtype=torch.long)
    star_center = torch.empty(batch_size, dtype=torch.long)
    x = torch.empty(batch_size, n_units, dtype=torch.float32)
    observed_treatment = torch.empty(batch_size, n_units, dtype=torch.float32)
    observed_exposure = torch.empty(batch_size, n_units, dtype=torch.float32)
    degree = torch.empty(batch_size, n_units, dtype=torch.float32)
    majority_arm = torch.empty(batch_size, n_units, dtype=torch.int64)
    low_sampled_exposure = torch.empty(batch_size, n_units, dtype=torch.float32)
    high_sampled_exposure = torch.empty(batch_size, n_units, dtype=torch.float32)
    y_obs = torch.empty(batch_size, n_units, dtype=torch.float32)
    structural_baseline = torch.empty(batch_size, n_units, dtype=torch.float32)
    outcome_a = torch.empty(batch_size, n_units, 3, dtype=torch.float32)
    outcome_b = torch.empty(batch_size, n_units, 3, dtype=torch.float32)
    query_effect = torch.empty(batch_size, n_units, 3, dtype=torch.float32)
    oracle_ite = torch.empty(batch_size, n_units, 3, dtype=torch.float32)
    tau = torch.empty(batch_size, n_units, dtype=torch.float32)
    gamma = torch.empty(batch_size, n_units, dtype=torch.float32)
    eta = torch.empty(batch_size, n_units, dtype=torch.float32)
    seen = torch.zeros(batch_size, n_units, dtype=torch.bool)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_csv_fieldnames(reader.fieldnames)
        for row in reader:
            if row["schema_version"] != CSV_SCHEMA_VERSION:
                raise ValueError("CSV schema version is stale; regenerate the dataset.")
            if int(row["n_units"]) != n_units:
                raise ValueError("CSV rows contain inconsistent n_units values.")
            dataset_id = int(row["dataset_id"])
            if expected_dataset_count is None:
                dataset_index = dataset_index_by_id[dataset_id]
            else:
                dataset_index = dataset_index_by_id.get(dataset_id, -1)
                if dataset_index < 0:
                    if len(dataset_ids) >= batch_size:
                        raise ValueError(
                            "CSV episode count exceeds expected_dataset_count="
                            f"{batch_size}."
                        )
                    dataset_index = len(dataset_ids)
                    dataset_index_by_id[dataset_id] = dataset_index
                    dataset_ids.append(dataset_id)
            unit_index = int(row["unit_id"])
            if not 0 <= unit_index < n_units:
                raise ValueError(
                    f"dataset_id={dataset_id} has unit_id={unit_index} outside "
                    f"0 through {n_units - 1}."
                )
            if bool(seen[dataset_index, unit_index]):
                raise ValueError(
                    f"dataset_id={dataset_id} repeats unit_id={unit_index}."
                )
            seen[dataset_index, unit_index] = True
            current_graph = int(row["graph_type"])
            current_center = int(row["star_center"])
            if unit_index == 0:
                graph_type[dataset_index] = current_graph
                star_center[dataset_index] = current_center
            elif (
                int(graph_type[dataset_index].item()) != current_graph
                or int(star_center[dataset_index].item()) != current_center
            ):
                raise ValueError(
                    f"dataset_id={dataset_id} has inconsistent episode metadata."
                )
            x[dataset_index, unit_index] = float(row["x"])
            observed_treatment[dataset_index, unit_index] = float(
                row["observed_treatment"]
            )
            observed_exposure[dataset_index, unit_index] = float(
                row["observed_exposure"]
            )
            degree[dataset_index, unit_index] = float(row["degree"])
            majority_arm[dataset_index, unit_index] = int(row["majority_arm"])
            low_sampled_exposure[dataset_index, unit_index] = float(
                row["low_sampled_exposure"]
            )
            high_sampled_exposure[dataset_index, unit_index] = float(
                row["high_sampled_exposure"]
            )
            y_obs[dataset_index, unit_index] = float(row["y_obs"])
            structural_baseline[dataset_index, unit_index] = float(row["structural_baseline"])
            tau[dataset_index, unit_index] = float(row["tau"])
            gamma[dataset_index, unit_index] = float(row["gamma"])
            eta[dataset_index, unit_index] = float(row["eta"])
            oracle_ite[dataset_index, unit_index, 0] = float(row["oracle_direct_ite"])
            oracle_ite[dataset_index, unit_index, 1] = float(row["oracle_spillover_ite"])
            oracle_ite[dataset_index, unit_index, 2] = float(row["oracle_total_ite"])
            for feature_index in range(TOKEN_DIM):
                tokens[dataset_index, unit_index, feature_index] = float(
                    row[f"token_{feature_index}"]
                )
            adjacency[dataset_index, unit_index] = _decode_binary_mask_hex(
                row["adjacency_bits"], n_units
            )
            for query_index, query_name in enumerate(QUERY_TYPE_NAMES):
                for feature_index in range(QUERY_DIM):
                    queries[
                        dataset_index, unit_index, query_index, feature_index
                    ] = float(row[f"{query_name}_query_{feature_index}"])
                outcome_a[dataset_index, unit_index, query_index] = float(
                    row[f"{query_name}_outcome_a"]
                )
                outcome_b[dataset_index, unit_index, query_index] = float(
                    row[f"{query_name}_outcome_b"]
                )
                query_effect[dataset_index, unit_index, query_index] = float(
                    row[f"{query_name}_effect"]
                )

    if expected_dataset_count is not None and len(dataset_ids) != batch_size:
        raise ValueError(
            "CSV episode count does not match expected_dataset_count: "
            f"{len(dataset_ids)} != {batch_size}."
        )
    if not bool(seen.all()):
        missing = torch.nonzero(~seen, as_tuple=False)[0].tolist()
        missing_dataset = dataset_ids[missing[0]]
        raise ValueError(
            f"dataset_id={missing_dataset} must contain unit_id 0 through "
            f"{n_units - 1} exactly once."
        )

    covariate_score = torch.sigmoid(x)
    neighbor_covariate_score = compute_neighbor_covariate_score_chunked(
        adjacency, covariate_score, degree
    )

    batch: TensorBatch = {
        "tokens": tokens,
        "queries": queries,
        "adjacency": adjacency,
        "graph_type": graph_type,
        "star_center": star_center,
        "x": x,
        "covariate_score": covariate_score,
        "neighbor_covariate_score": neighbor_covariate_score,
        "observed_treatment": observed_treatment,
        "observed_exposure": observed_exposure,
        "degree": degree,
        "majority_arm": majority_arm,
        "low_sampled_exposure": low_sampled_exposure,
        "high_sampled_exposure": high_sampled_exposure,
        "y_obs": y_obs,
        "structural_baseline": structural_baseline,
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "query_effect": query_effect,
        "oracle_ite": oracle_ite,
        "oracle_ate": oracle_ite.mean(dim=1),
        "tau": tau,
        "gamma": gamma,
        "eta": eta,
    }
    return {key: value.to(device) for key, value in batch.items()}, dataset_ids

def episode_split_paths(source_path: Path) -> tuple[Path, Path]:
    """Return sibling paths for the 80% training and 20% validation CSVs."""

    suffix = source_path.suffix or ".csv"
    stem = source_path.stem
    return (
        source_path.with_name(f"{stem}_train{suffix}"),
        source_path.with_name(f"{stem}_validation{suffix}"),
    )


def load_episode_batch_csv(
    path: Path, *, device: torch.device,
    expected_dataset_count: Optional[int] = None,
) -> tuple[TensorBatch, list[int]]:
    """Exactly the same CSV fields, reconstructed through NumPy CPU buffers."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        _validate_csv_fieldnames(csv.DictReader(handle).fieldnames)
    arrays, ids = read_csv_arrays(
        path, CSV_SCHEMA_VERSION, TOKEN_DIM, QUERY_DIM, QUERY_TYPE_NAMES,
        expected_dataset_count, progress=lambda msg: print(f"{Path(path).name}: {msg}", flush=True),
    )
    batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
    batch["covariate_score"] = torch.sigmoid(batch["x"])
    batch["neighbor_covariate_score"] = compute_neighbor_covariate_score_chunked(
        batch["adjacency"], batch["covariate_score"], batch["degree"],
    )
    batch["oracle_ate"] = batch["oracle_ite"].mean(dim=1)
    return {key: value.to(device) for key, value in batch.items()}, ids


def load_cached_episode_csv(path: Path, expected_dataset_count: int):
    return cached_cpu_batch(
        path,
        lambda source, **kwargs: load_episode_batch_csv(source, device=torch.device("cpu"), **kwargs),
        expected_dataset_count,
    )


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_ids_sha256(dataset_ids) -> str:
    """Stable digest of the ordered episode IDs assigned to one shard."""

    digest = hashlib.sha256()
    for dataset_id in dataset_ids:
        digest.update(f"{int(dataset_id)}\n".encode("ascii"))
    return digest.hexdigest()


def _ddp_shard_integrity_record(path: Path, dataset_ids) -> Dict[str, object]:
    ids = [int(value) for value in dataset_ids]
    stat = Path(path).stat()
    return {
        "name": Path(path).name,
        "size_bytes": int(stat.st_size),
        "sha256": _file_sha256(Path(path)),
        "dataset_count": len(ids),
        "dataset_ids_sha256": _dataset_ids_sha256(ids),
    }


def _verify_ddp_shard_file(path: Path, record: Mapping[str, object], *, role: str) -> None:
    """Reject a rank shard whose bytes no longer match preparation metadata."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DDP shard file is missing ({role}): {path}")
    expected_name = str(record.get("name", ""))
    if expected_name and path.name != expected_name:
        raise ValueError(
            f"DDP shard integrity failure ({role}): expected file {expected_name}, got {path.name}."
        )
    expected_size = int(record.get("size_bytes", -1))
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"DDP shard integrity failure ({role}): size mismatch for {path}."
        )
    expected_sha256 = str(record.get("sha256", ""))
    if not expected_sha256 or _file_sha256(path) != expected_sha256:
        raise ValueError(
            f"DDP shard integrity failure ({role}): SHA256 mismatch for {path}."
        )


def ddp_shard_metadata_path(shard_dir: Path) -> Path:
    """Return the completion metadata path for physical DDP train shards."""

    return Path(shard_dir) / "metadata.json"


def prepare_ddp_train_shards(
    data_csv: Path,
    *,
    shard_dir: Path,
    world_size: int,
) -> Dict[str, object]:
    """Create equal rank-local train shards and one validation file.

    For random-LPE banks, the completion manifest already contains the exact
    train/validation episode IDs. In that case this function streams the master
    CSV exactly once and reconstructs the original split directly into rank-local
    files, so the large derived ``*_train.csv``/``*_validation.csv`` files are not
    required. Legacy banks fall back to their existing prepared split files.
    """

    if world_size <= 1:
        raise ValueError("world_size must be greater than one for DDP train shards.")
    data_csv = Path(data_csv)
    shard_dir = Path(shard_dir)
    if not data_csv.is_file():
        raise FileNotFoundError(f"Master episode CSV does not exist: {data_csv}")

    manifest_path = data_csv.with_suffix(data_csv.suffix + ".manifest.json")
    manifest_data: Optional[Dict[str, object]] = None
    training_ids: Optional[list[int]] = None
    validation_ids: Optional[list[int]] = None
    if manifest_path.is_file():
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_values = [int(value) for value in candidate.get("training_dataset_ids", [])]
            validation_values = [int(value) for value in candidate.get("validation_dataset_ids", [])]
            if train_values and validation_values and not (set(train_values) & set(validation_values)):
                manifest_data = candidate
                training_ids = sorted(train_values)
                validation_ids = sorted(validation_values)
        except (OSError, ValueError, TypeError):
            manifest_data = None

    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = ddp_shard_metadata_path(shard_dir)
    final_train_paths = [shard_dir / f"train_rank{rank}.csv" for rank in range(world_size)]
    final_validation_path = shard_dir / "validation.csv"

    if manifest_data is not None and training_ids is not None and validation_ids is not None:
        if len(training_ids) % world_size != 0:
            raise ValueError(
                "Training episodes must divide evenly across DDP ranks; "
                f"got {len(training_ids)} episodes for world_size={world_size}."
            )
        source_mode = "manifest_master"
        source_path = data_csv
        source_stat = source_path.stat()
        manifest_stat = manifest_path.stat()
        expected_counts = [len(training_ids) // world_size] * world_size
        rank_by_dataset = {
            dataset_id: position % world_size
            for position, dataset_id in enumerate(training_ids)
        }
        validation_set = set(validation_ids)
        expected_train_set = set(training_ids)
        reuse_identity = {
            "source_mode": source_mode,
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "manifest_size_bytes": manifest_stat.st_size,
            "manifest_mtime_ns": manifest_stat.st_mtime_ns,
        }
    else:
        train_path, source_validation_path = episode_split_paths(data_csv)
        if not train_path.is_file() or not source_validation_path.is_file():
            raise FileNotFoundError(
                "Prepared train/validation CSV files are required for a bank without "
                "a split manifest."
            )
        source_mode = "split_files"
        source_path = train_path
        source_stat = source_path.stat()
        validation_stat = source_validation_path.stat()
        rank_by_dataset = {}
        validation_set = set()
        expected_train_set = set()
        expected_counts = []
        reuse_identity = {
            "source_mode": source_mode,
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "validation_source_size_bytes": validation_stat.st_size,
            "validation_source_mtime_ns": validation_stat.st_mtime_ns,
        }

    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            reusable = (
                int(existing.get("world_size", -1)) == world_size
                and all(existing.get(key) == value for key, value in reuse_identity.items())
                and all(path.is_file() for path in final_train_paths)
                and final_validation_path.is_file()
            )
            files_metadata = existing.get("files", {})
            train_records = files_metadata.get("train", []) if isinstance(files_metadata, dict) else []
            validation_record = files_metadata.get("validation", {}) if isinstance(files_metadata, dict) else {}
            reusable = reusable and len(train_records) == world_size and isinstance(validation_record, dict)
            if reusable:
                for rank, path in enumerate(final_train_paths):
                    _verify_ddp_shard_file(
                        path, cast(Mapping[str, object], train_records[rank]), role=f"train rank {rank}"
                    )
                _verify_ddp_shard_file(
                    final_validation_path,
                    cast(Mapping[str, object], validation_record),
                    role="validation",
                )
                print(f"Reusing rank-local DDP train shards: {shard_dir}", flush=True)
                return existing
        except (OSError, ValueError, TypeError, FileNotFoundError):
            # Corrupt or legacy shard sets are rebuilt from the immutable source bank.
            pass

    metadata_path.unlink(missing_ok=True)
    temp_train_paths = [path.with_name(path.name + ".building") for path in final_train_paths]
    temp_validation_path = final_validation_path.with_name(final_validation_path.name + ".building")
    for path in [*temp_train_paths, temp_validation_path]:
        path.unlink(missing_ok=True)

    counts = [0 for _ in range(world_size)]
    rank_dataset_ids: list[list[int]] = [[] for _ in range(world_size)]
    seen_train: set[int] = set()
    seen_validation: set[int] = set()

    if source_mode == "manifest_master":
        handles = []
        try:
            with source_path.open("r", encoding="utf-8", newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                fieldnames = reader.fieldnames
                if fieldnames is None or "dataset_id" not in fieldnames:
                    raise ValueError("master CSV must contain a dataset_id column.")
                train_writers = []
                for path in temp_train_paths:
                    handle = path.open("w", encoding="utf-8", newline="")
                    handles.append(handle)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    train_writers.append(writer)
                validation_handle = temp_validation_path.open("w", encoding="utf-8", newline="")
                handles.append(validation_handle)
                validation_writer = csv.DictWriter(validation_handle, fieldnames=fieldnames)
                validation_writer.writeheader()

                for row in reader:
                    dataset_id = int(row["dataset_id"])
                    rank = rank_by_dataset.get(dataset_id)
                    if rank is not None:
                        if dataset_id not in seen_train:
                            seen_train.add(dataset_id)
                            counts[rank] += 1
                            rank_dataset_ids[rank].append(dataset_id)
                        train_writers[rank].writerow(row)
                    elif dataset_id in validation_set:
                        seen_validation.add(dataset_id)
                        validation_writer.writerow(row)
                    else:
                        raise ValueError(
                            f"dataset_id={dataset_id} is absent from manifest train/validation split."
                        )
        finally:
            for handle in handles:
                handle.close()
        if seen_train != expected_train_set or seen_validation != validation_set:
            for path in [*temp_train_paths, temp_validation_path]:
                path.unlink(missing_ok=True)
            raise ValueError("Master CSV does not match the manifest episode split.")
        validation_datasets = len(validation_set)
    else:
        assignment: Dict[int, int] = {}
        handles = []
        try:
            with source_path.open("r", encoding="utf-8", newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                fieldnames = reader.fieldnames
                if fieldnames is None or "dataset_id" not in fieldnames:
                    raise ValueError("training CSV must contain a dataset_id column.")
                train_writers = []
                for path in temp_train_paths:
                    handle = path.open("w", encoding="utf-8", newline="")
                    handles.append(handle)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    train_writers.append(writer)
                for row in reader:
                    dataset_id = int(row["dataset_id"])
                    rank = assignment.get(dataset_id)
                    if rank is None:
                        rank = len(assignment) % world_size
                        assignment[dataset_id] = rank
                        counts[rank] += 1
                        rank_dataset_ids[rank].append(dataset_id)
                    train_writers[rank].writerow(row)
        finally:
            for handle in handles:
                handle.close()
        train_path, source_validation_path = episode_split_paths(data_csv)
        with (
            source_validation_path.open("r", encoding="utf-8", newline="") as source_handle,
            temp_validation_path.open("w", encoding="utf-8", newline="") as target_handle,
        ):
            reader = csv.DictReader(source_handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("validation CSV must contain a header.")
            writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                seen_validation.add(int(row["dataset_id"]))
                writer.writerow(row)
        validation_datasets = len(seen_validation)
        expected_counts = counts.copy()

    training_datasets = sum(counts)
    if training_datasets <= 0 or validation_datasets <= 0:
        for path in [*temp_train_paths, temp_validation_path]:
            path.unlink(missing_ok=True)
        raise ValueError("DDP shard preparation requires nonempty train and validation splits.")
    if len(set(counts)) != 1:
        for path in [*temp_train_paths, temp_validation_path]:
            path.unlink(missing_ok=True)
        raise ValueError(
            "Training episodes must divide evenly across DDP ranks; "
            f"got per-rank counts {counts}."
        )
    if expected_counts and counts != expected_counts:
        for path in [*temp_train_paths, temp_validation_path]:
            path.unlink(missing_ok=True)
        raise ValueError(f"Unexpected per-rank episode counts: {counts} != {expected_counts}.")

    for temp_path, final_path in zip(temp_train_paths, final_train_paths):
        temp_path.replace(final_path)
    temp_validation_path.replace(final_validation_path)

    train_integrity = [
        _ddp_shard_integrity_record(path, rank_dataset_ids[rank])
        for rank, path in enumerate(final_train_paths)
    ]
    validation_integrity = _ddp_shard_integrity_record(
        final_validation_path, sorted(seen_validation)
    )

    metadata: Dict[str, object] = {
        "total_datasets": training_datasets + validation_datasets,
        "training_datasets": training_datasets,
        "validation_datasets": validation_datasets,
        "world_size": world_size,
        "train_datasets_per_rank": counts[0],
        "train_dataset_counts": counts,
        "validation_file": final_validation_path.name,
        "files": {"train": train_integrity, "validation": validation_integrity},
        **reuse_identity,
    }
    temp_metadata = metadata_path.with_name(metadata_path.name + ".building")
    temp_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_metadata.replace(metadata_path)
    print(
        f"Prepared {world_size} equal DDP train shards: "
        f"{counts[0]} episodes/rank; validation={validation_datasets} episodes; "
        f"dir={shard_dir}",
        flush=True,
    )
    return metadata


def _empty_validation_batch(n_units: int) -> TensorBatch:
    """Minimal placeholder used on non-main ranks, which never evaluate validation."""

    return {
        "tokens": torch.empty(0, n_units, TOKEN_DIM, dtype=torch.float32),
    }


def load_rank_sharded_episode_csv_splits(
    data_csv: Path,
    *,
    shard_dir: Path,
    runtime: DistributedRuntime,
    distributed_validation: bool = False,
) -> Dict[str, object]:
    """Load a local train shard and optionally partition validation without padding."""

    data_csv = Path(data_csv)
    shard_dir = Path(shard_dir)
    metadata_path = ddp_shard_metadata_path(shard_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"DDP shard metadata does not exist: {metadata_path}. "
            "Run --prepare-ddp-shards-only first."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid DDP shard metadata: {metadata_path}") from exc
    if int(metadata.get("world_size", -1)) != runtime.world_size:
        raise ValueError(
            "DDP shard world_size does not match the current torchrun world_size: "
            f"{metadata.get('world_size')} != {runtime.world_size}."
        )

    source_mode = str(metadata.get("source_mode", ""))
    if source_mode == "manifest_master":
        if not data_csv.is_file():
            raise FileNotFoundError(f"Master episode CSV is missing: {data_csv}")
        source_stat = data_csv.stat()
        manifest_path = data_csv.with_suffix(data_csv.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Random CSV manifest is missing: {manifest_path}")
        manifest_stat = manifest_path.stat()
        stale = (
            int(metadata.get("source_size_bytes", -1)) != source_stat.st_size
            or int(metadata.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns
            or int(metadata.get("manifest_size_bytes", -1)) != manifest_stat.st_size
            or int(metadata.get("manifest_mtime_ns", -1)) != manifest_stat.st_mtime_ns
        )
    elif source_mode == "split_files":
        train_source, validation_source = episode_split_paths(data_csv)
        if not train_source.is_file() or not validation_source.is_file():
            raise FileNotFoundError("Prepared source split files are missing.")
        train_stat = train_source.stat()
        validation_stat = validation_source.stat()
        stale = (
            int(metadata.get("source_size_bytes", -1)) != train_stat.st_size
            or int(metadata.get("source_mtime_ns", -1)) != train_stat.st_mtime_ns
            or int(metadata.get("validation_source_size_bytes", -1)) != validation_stat.st_size
            or int(metadata.get("validation_source_mtime_ns", -1)) != validation_stat.st_mtime_ns
        )
    else:
        raise ValueError(f"Unsupported DDP shard source_mode: {source_mode!r}")
    if stale:
        raise ValueError(
            "DDP train shards are stale relative to their source bank; "
            "re-run --prepare-ddp-shards-only."
        )

    train_path = shard_dir / f"train_rank{runtime.rank}.csv"
    validation_path = shard_dir / str(metadata.get("validation_file", "validation.csv"))
    if not train_path.is_file():
        raise FileNotFoundError(f"Rank-local training shard is missing: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Sharded validation CSV is missing: {validation_path}")
    files_metadata = metadata.get("files", {})
    if not isinstance(files_metadata, dict):
        raise ValueError("DDP shard metadata lacks integrity records; re-run --prepare-ddp-shards-only.")
    train_records = files_metadata.get("train", [])
    validation_record = files_metadata.get("validation", {})
    if not isinstance(train_records, list) or len(train_records) != runtime.world_size:
        raise ValueError("DDP shard metadata has invalid train integrity records; re-run shard preparation.")
    if not isinstance(validation_record, dict):
        raise ValueError("DDP shard metadata has invalid validation integrity record; re-run shard preparation.")
    local_train_record = cast(Mapping[str, object], train_records[runtime.rank])
    _verify_ddp_shard_file(train_path, local_train_record, role=f"train rank {runtime.rank}")
    if runtime.is_main:
        _verify_ddp_shard_file(
            validation_path, cast(Mapping[str, object], validation_record), role="validation"
        )
    print(f"[rank {runtime.rank}] loading train shard: {train_path}", flush=True)
    train_batch, train_dataset_ids = load_cached_episode_csv(
        train_path, expected_dataset_count=int(metadata["train_datasets_per_rank"]),
    )
    if _dataset_ids_sha256(train_dataset_ids) != str(local_train_record.get("dataset_ids_sha256", "")):
        raise ValueError(
            f"DDP shard integrity failure (train rank {runtime.rank}): dataset-ID digest mismatch."
        )
    print(
        f"[rank {runtime.rank}] train shard loaded: {len(train_dataset_ids)} datasets",
        flush=True,
    )

    if runtime.is_main or distributed_validation:
        print(f"[rank {runtime.rank}] loading validation cache: {validation_path}", flush=True)
        validation_batch, validation_dataset_ids = load_cached_episode_csv(
            validation_path, expected_dataset_count=int(metadata["validation_datasets"]),
        )
        if _dataset_ids_sha256(sorted(validation_dataset_ids)) != str(
            validation_record.get("dataset_ids_sha256", "")
        ):
            raise ValueError("DDP shard integrity failure (validation): dataset-ID digest mismatch.")
        if distributed_validation:
            local_indices = validation_indices(len(validation_dataset_ids), runtime.rank, runtime.world_size)
            # Only this rank's actual validation episodes are copied. Shared
            # mmap cache pages do not create one full bank per rank in RAM.
            idx = torch.tensor(local_indices, dtype=torch.long)
            validation_batch = {key: value.index_select(0, idx) for key, value in validation_batch.items()}
            validation_dataset_ids = [validation_dataset_ids[i] for i in local_indices]
        print(f"[rank {runtime.rank}] validation loaded: {len(validation_dataset_ids)} datasets", flush=True)
    else:
        validation_batch = _empty_validation_batch(int(train_batch["tokens"].shape[1]))
        validation_dataset_ids = []

    summary = {
        "total_datasets": int(metadata["total_datasets"]),
        "training_datasets": int(metadata["training_datasets"]),
        "validation_datasets": int(metadata["validation_datasets"]),
    }
    return {
        "data_csv": data_csv,
        "train_path": train_path,
        "validation_path": validation_path,
        "summary": summary,
        "train_batch": train_batch,
        "validation_batch": validation_batch,
        "train_dataset_ids": train_dataset_ids,
        "validation_dataset_ids": validation_dataset_ids,
        "rank_sharded_train": True,
        "validation_is_sharded": distributed_validation,
    }



def _allocate_stratified_validation_counts(
    grouped: Mapping[int, list[int]],
    validation_count: int,
) -> Dict[int, int]:
    """Allocate an exact global validation total across graph families."""

    total = sum(len(values) for values in grouped.values())
    graph_values = sorted(grouped)
    if not graph_values:
        raise ValueError("Cannot stratify an empty episode collection.")
    if validation_count < len(graph_values) or validation_count > total - len(graph_values):
        raise ValueError("Exact stratification cannot keep every graph family in both splits.")
    raw = {
        graph_value: validation_count * len(grouped[graph_value]) / total
        for graph_value in graph_values
    }
    allocation = {
        graph_value: min(
            max(int(math.floor(raw[graph_value])), 1),
            len(grouped[graph_value]) - 1,
        )
        for graph_value in graph_values
    }
    while sum(allocation.values()) < validation_count:
        candidates = [
            graph_value
            for graph_value in graph_values
            if allocation[graph_value] < len(grouped[graph_value]) - 1
        ]
        if not candidates:
            raise RuntimeError("Unable to allocate requested stratified validation total.")
        chosen = max(candidates, key=lambda value: (raw[value] - allocation[value], -value))
        allocation[chosen] += 1
    while sum(allocation.values()) > validation_count:
        candidates = [graph_value for graph_value in graph_values if allocation[graph_value] > 1]
        if not candidates:
            raise RuntimeError("Unable to reduce stratified validation allocation.")
        chosen = max(candidates, key=lambda value: (allocation[value] - raw[value], -value))
        allocation[chosen] -= 1
    return allocation


def select_episode_split_ids_from_graph_types(
    dataset_graph_type: Mapping[int, int],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    stratify_by_graph_type: bool = False,
) -> tuple[list[int], list[int], Dict[str, int]]:
    """Select deterministic split IDs from episode-level graph metadata."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    dataset_ids = sorted(int(value) for value in dataset_graph_type)
    total = len(dataset_ids)
    if total < 2:
        raise ValueError("At least two episodes are required for a train/validation split.")
    validation_count = int(round(total * validation_fraction))
    validation_count = min(max(validation_count, 1), total - 1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(split_seed)
    validation_ids: set[int] = set()
    grouped: Dict[int, list[int]] = {}
    if stratify_by_graph_type:
        for dataset_id in dataset_ids:
            grouped.setdefault(int(dataset_graph_type[dataset_id]), []).append(dataset_id)
    can_stratify = (
        stratify_by_graph_type
        and all(len(group_ids) >= 2 for group_ids in grouped.values())
        and len(grouped) <= validation_count <= total - len(grouped)
    )
    if can_stratify:
        allocation = _allocate_stratified_validation_counts(grouped, validation_count)
        for graph_value in sorted(grouped):
            group_ids = grouped[graph_value]
            permutation = torch.randperm(len(group_ids), generator=generator).tolist()
            validation_ids.update(
                group_ids[index] for index in permutation[: allocation[graph_value]]
            )
    else:
        permutation = torch.randperm(total, generator=generator).tolist()
        validation_ids = {dataset_ids[index] for index in permutation[:validation_count]}

    training_ids = sorted(set(dataset_ids) - validation_ids)
    validation_ids_sorted = sorted(validation_ids)
    if len(validation_ids_sorted) != validation_count:
        raise RuntimeError("Validation split does not match the exact requested global count.")
    summary = {
        "total_datasets": total,
        "training_datasets": len(training_ids),
        "validation_datasets": len(validation_ids_sorted),
    }
    return training_ids, validation_ids_sorted, summary


def select_episode_split_ids(
    source_path: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    stratify_by_graph_type: bool = False,
) -> tuple[list[int], list[int], Dict[str, int]]:
    """Read episode graph metadata, then select the deterministic split IDs."""

    dataset_graph_type: Dict[int, int] = {}
    with Path(source_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("source CSV must contain a header.")
        for row in reader:
            dataset_id = int(row["dataset_id"])
            graph_value = int(row["graph_type"])
            previous = dataset_graph_type.setdefault(dataset_id, graph_value)
            if previous != graph_value:
                raise ValueError("One episode contains inconsistent graph_type values.")
    return select_episode_split_ids_from_graph_types(
        dataset_graph_type,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        stratify_by_graph_type=stratify_by_graph_type,
    )

def split_episode_csv(
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    stratify_by_graph_type: bool = False,
) -> Dict[str, int]:
    """Split complete episodes with an exact global validation count."""

    training_id_list, validation_id_list, summary = select_episode_split_ids(
        source_path,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        stratify_by_graph_type=stratify_by_graph_type,
    )
    training_ids = set(training_id_list)
    validation_ids = set(validation_id_list)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        source_path.open("r", encoding="utf-8", newline="") as source_handle,
        train_path.open("w", encoding="utf-8", newline="") as train_handle,
        validation_path.open("w", encoding="utf-8", newline="") as validation_handle,
    ):
        reader = csv.DictReader(source_handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("source CSV must contain a header.")
        train_writer = csv.DictWriter(train_handle, fieldnames=fieldnames)
        validation_writer = csv.DictWriter(validation_handle, fieldnames=fieldnames)
        train_writer.writeheader()
        validation_writer.writeheader()
        for row in reader:
            dataset_id = int(row["dataset_id"])
            writer = validation_writer if dataset_id in validation_ids else train_writer
            writer.writerow(row)

    return summary

def prepare_episode_csv_files(
    data_config: DataConfig,
    *,
    data_csv: Path,
    num_datasets: int,
    data_seed: int,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    overwrite: bool = False,
    generation_batch_size: int = 64,
) -> Dict[str, object]:
    """Create/reuse a master CSV and write its 80/20 split files."""

    if data_config.dgp_version in ("random_lpe_v1", "a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
        from pfn_interference.simulation.csv_tasks import prepare_random_csv
        return prepare_random_csv(
            data_config, data_csv=Path(data_csv), num_datasets=num_datasets,
            data_seed=data_seed, validation_fraction=validation_fraction,
            split_seed=split_seed, overwrite=overwrite,
            generation_batch_size=generation_batch_size,
        )
    from pfn_interference.simulation.csv_tasks import reject_random_csv_for_legacy
    reject_random_csv_for_legacy(data_config, Path(data_csv), allow_overwrite=overwrite)
    if num_datasets < 2:
        raise ValueError("num_datasets must be at least 2.")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive.")
    data_csv = Path(data_csv)
    needs_generation = (
        overwrite
        or not data_csv.is_file()
        or not csv_schema_matches(data_csv, expected_n_units=data_config.n_units)
    )
    if needs_generation:
        data_csv.parent.mkdir(parents=True, exist_ok=True)
        if data_csv.exists():
            data_csv.unlink()
        for start in range(0, num_datasets, generation_batch_size):
            chunk_size = min(generation_batch_size, num_datasets - start)
            chunk = generate_batch(
                data_config,
                chunk_size,
                seed=data_seed + start,
                device=torch.device("cpu"),
                graph_types=None,
            )
            save_episode_batch_csv(
                chunk,
                data_csv,
                dataset_id_offset=start,
                append=start > 0,
            )
            completed = start + chunk_size
            print(
                f"data_generation_progress={completed}/{num_datasets} "
                f"({100.0 * completed / num_datasets:.1f}%)",
                flush=True,
            )

    train_path, validation_path = episode_split_paths(data_csv)
    summary = split_episode_csv(
        data_csv,
        train_path,
        validation_path,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        stratify_by_graph_type=(data_config.graph_family == "mixed"),
    )
    return {
        "data_csv": data_csv,
        "train_path": train_path,
        "validation_path": validation_path,
        "summary": summary,
    }


def prepare_episode_csv_splits(
    data_config: DataConfig,
    *,
    data_csv: Path,
    num_datasets: int,
    data_seed: int,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    overwrite: bool = False,
    generation_batch_size: int = 64,
) -> Dict[str, object]:
    """Create/reuse CSV files, then load their train/validation tensors."""

    prepared = prepare_episode_csv_files(
        data_config,
        data_csv=data_csv,
        num_datasets=num_datasets,
        data_seed=data_seed,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        overwrite=overwrite,
        generation_batch_size=generation_batch_size,
    )
    train_path = cast(Path, prepared["train_path"])
    validation_path = cast(Path, prepared["validation_path"])
    train_batch, train_dataset_ids = load_episode_batch_csv(
        train_path, device=torch.device("cpu")
    )
    validation_batch, validation_dataset_ids = load_episode_batch_csv(
        validation_path, device=torch.device("cpu")
    )
    return {
        **prepared,
        "train_path": train_path,
        "validation_path": validation_path,
        "train_batch": train_batch,
        "validation_batch": validation_batch,
        "train_dataset_ids": train_dataset_ids,
        "validation_dataset_ids": validation_dataset_ids,
        "rank_sharded_train": False,
    }


def load_existing_episode_csv_splits(data_csv: Path) -> Dict[str, object]:
    """Load train/validation files that were already created by rank zero."""

    data_csv = Path(data_csv)
    train_path, validation_path = episode_split_paths(data_csv)
    if not data_csv.is_file():
        raise FileNotFoundError(f"Master episode CSV does not exist: {data_csv}")
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Train/validation CSV splits do not exist; rank zero must create them first."
        )
    train_batch, train_dataset_ids = load_episode_batch_csv(
        train_path, device=torch.device("cpu")
    )
    validation_batch, validation_dataset_ids = load_episode_batch_csv(
        validation_path, device=torch.device("cpu")
    )
    summary = {
        "total_datasets": len(train_dataset_ids) + len(validation_dataset_ids),
        "training_datasets": len(train_dataset_ids),
        "validation_datasets": len(validation_dataset_ids),
    }
    return {
        "data_csv": data_csv,
        "train_path": train_path,
        "validation_path": validation_path,
        "summary": summary,
        "train_batch": train_batch,
        "validation_batch": validation_batch,
        "train_dataset_ids": train_dataset_ids,
        "validation_dataset_ids": validation_dataset_ids,
        "rank_sharded_train": False,
    }


def select_episode_batch(
    batch: TensorBatch,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> TensorBatch:
    """Select complete episodes, allowing repeated indices for mini-batching."""

    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if indices.dtype != torch.long:
        indices = indices.to(torch.long)
    episode_count = int(batch["tokens"].shape[0])
    if bool(((indices < 0) | (indices >= episode_count)).any()):
        raise IndexError("episode index is outside the loaded split.")
    selected: TensorBatch = {}
    for key, value in batch.items():
        local_indices = indices.to(value.device)
        selected[key] = value.index_select(0, local_indices).to(device)
    return selected


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer with standard self-attention and a feedforward block.

    Network structure is encoded in the input node representations before
    this layer. The ``adjacency`` argument is accepted by the encoder interface
    but does not add a graph bias or mask to self-attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.d_model, config.ffn_dim)
        self.linear2 = nn.Linear(config.ffn_dim, config.d_model)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        src: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        # Graph structure enters through the input embeddings, not attention.
        _ = adjacency
        normalized = self.norm1(src)
        attention_output, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        src = src + self.dropout1(attention_output)
        normalized = self.norm2(src)
        feedforward = self.linear2(
            self.ffn_dropout(self.activation(self.linear1(normalized)))
        )
        return src + self.dropout2(feedforward)


class LocalNetworkQueryTransformer(nn.Module):
    """Encode observed nodes and predict four query-conditioned CEPO GMMs."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.self_encoder = nn.Sequential(
            nn.Linear(3, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.embedding_fusion = nn.Sequential(
            nn.Linear(2 * config.d_model + 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(config)
            for _ in range(config.num_layers)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(config.query_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.query_fusion = nn.LayerNorm(config.d_model)
        self.mu_head = make_gmm_head(config)

    def _network_aware_embedding(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Build self and one-hop treated-neighbor representations."""

        observed = build_observed_tokens(tokens)
        x = observed[..., 0:1]
        treatment = observed[..., 1:2]
        y_obs = observed[..., 2:3]
        exposure_degree = observed[..., 3:5]

        self_representation = self.self_encoder(
            torch.cat([x, treatment, y_obs], dim=-1)
        )
        base_neighbor_representation = self.neighbor_encoder(x)
        adjacency_float = adjacency.to(dtype=base_neighbor_representation.dtype)
        treated_representation = base_neighbor_representation * treatment
        interference_representation = torch.zeros_like(
            base_neighbor_representation
        )

        # Build topology-aware edge states only for direct edges i -> j. For
        # each such edge, common_neighbor_mask selects exactly those k with
        # A_ik = A_jk = 1, so k lies inside root i's induced one-hop
        # neighborhood. This excludes outside two-hop nodes while using A_jk.
        for batch_index in range(tokens.shape[0]):
            root_index, neighbor_index = torch.nonzero(
                adjacency_float[batch_index] > 0,
                as_tuple=True,
            )
            if root_index.numel() == 0:
                continue

            common_neighbor_mask = (
                adjacency_float[batch_index, root_index, :]
                * adjacency_float[batch_index, neighbor_index, :]
            )
            topology_sum = (
                common_neighbor_mask @ treated_representation[batch_index]
            )
            topology_degree = common_neighbor_mask.sum(
                dim=-1, keepdim=True
            )
            topology_message = topology_sum / topology_degree.clamp_min(1.0)

            neighbor_state = base_neighbor_representation[
                batch_index, neighbor_index
            ]
            edge_state = self.topology_encoder(
                torch.cat([neighbor_state, topology_message], dim=-1)
            )
            edge_state = edge_state * treatment[
                batch_index, neighbor_index
            ]

            root_sum = torch.zeros_like(
                base_neighbor_representation[batch_index]
            )
            root_sum.index_add_(0, root_index, edge_state)
            root_degree = adjacency_float[batch_index].sum(
                dim=-1, keepdim=True
            )
            interference_representation[batch_index] = (
                root_sum / root_degree.clamp_min(1.0)
            )

        fused_input = torch.cat(
            [
                self_representation,
                interference_representation,
                exposure_degree,
            ],
            dim=-1,
        )
        return self.embedding_fusion(fused_input)

    def encode_context(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the observed graph once so many causal queries can reuse it."""

        if tokens.ndim != 3 or tokens.shape[-1] != self.config.input_dim:
            raise ValueError("tokens must have shape [batch, units, input_dim].")
        n_units = int(tokens.shape[1])
        if adjacency.ndim != 3 or adjacency.shape[1:] != (n_units, n_units):
            raise ValueError(
                "adjacency must have shape [batch, units, units]."
            )
        if tokens.shape[0] != adjacency.shape[0]:
            raise ValueError("tokens and adjacency must have the same batch size.")
        encoded = self._network_aware_embedding(tokens, adjacency)
        for layer in self.encoder_layers:
            encoded = layer(encoded, adjacency)
        return encoded

    def predict_from_context(
        self,
        encoded: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict one or a grid of four-arm CEPO queries from cached context.

        ``queries`` may be [B,N,4,Q] or [B,N,P,4,Q]. Any dimensions between
        the unit and arm axes are treated as independent query-grid axes.
        """

        if encoded.ndim != 3 or encoded.shape[-1] != self.config.d_model:
            raise ValueError("encoded must have shape [batch, units, d_model].")
        if queries.ndim < 4 or queries.shape[-2:] != (4, self.config.query_dim):
            raise ValueError(
                "queries must have shape [batch, units, ..., 4, query_dim]."
            )
        if queries.shape[:2] != encoded.shape[:2]:
            raise ValueError(
                "encoded and queries must have the same batch/unit dimensions."
            )
        if not bool(torch.all((queries == 0) | (queries == 1))):
            raise ValueError("CEPO queries require binary own treatment and majority arm.")
        query_embedding = self.query_encoder(queries)
        context = encoded
        for _ in range(queries.ndim - 4):
            context = context.unsqueeze(2)
        context = context.unsqueeze(-2)
        fused = self.query_fusion(context + query_embedding)
        raw_predictions = self.mu_head(fused)
        return split_gmm_predictions(raw_predictions, self.config)

    def forward(
        self,
        tokens: torch.Tensor,
        queries: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.input_dim:
            raise ValueError("tokens must have shape [batch, units, input_dim].")
        n_units = int(tokens.shape[1])
        if queries.ndim != 4 or queries.shape[1:] != (
            n_units,
            4,
            self.config.query_dim,
        ):
            raise ValueError(
                "CEPO queries must have shape [batch, units, 4, 2]."
            )
        encoded = self.encode_context(tokens, adjacency)
        return self.predict_from_context(encoded, queries)


# Backward-compatible import name for older notebooks; new code should use the
# query-conditioned class name above.
PotentialOutcomeGraphTransformer = LocalNetworkQueryTransformer


def compute_regression_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute point metrics on CPU to avoid accelerator float64/sqrt gaps."""

    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes.")
    prediction64 = prediction.detach().cpu().numpy().astype("float64", copy=False)
    truth64 = truth.detach().cpu().numpy().astype("float64", copy=False)
    error = prediction64 - truth64
    absolute_error = abs(error)
    mse = float((error ** 2).mean())
    mae = float(absolute_error.mean())
    rmse = math.sqrt(mse)
    scale = max(float(abs(truth64).mean()), eps)
    mape = float((absolute_error / np.maximum(abs(truth64), eps)).mean() * 100.0)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "bias": float(error.mean()),
        "nmae_pct": mae / scale * 100.0,
        "nrmse_pct": rmse / scale * 100.0,
        "mape_pct": mape,
    }


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def compute_training_losses(model, predictions, batch):
    """Equal-weight GMM NLL of mu00, mu01, mu10, mu11; no effect loss."""
    return compute_cepo_losses(predictions, batch)


def compute_query_metrics(model, predictions, batch, *, interval_mass=.9, include_intervals=True):
    """CEPO distribution diagnostics and derived effect point metrics."""
    return compute_cepo_metrics(predictions, batch, interval_mass=interval_mass,
                                include_intervals=include_intervals)


def diagnose_batch_effect_support(
    batch: TensorBatch,
    model_config: Optional[ModelConfig] = None,
) -> Dict[str, object]:
    """Report effect-label ranges; a continuous GMM has no finite support."""

    del model_config
    diagnostic: Dict[str, object] = {}
    target = query_effect_targets(batch)
    for query_index, prefix in enumerate(EFFECT_PREFIXES):
        values = target[..., query_index].detach().flatten()
        diagnostic[prefix] = {
            "minimum": float(values.min().item()),
            "maximum": float(values.max().item()),
            "q001": float(torch.quantile(values, 0.001).item()),
            "q999": float(torch.quantile(values, 0.999).item()),
            "overflow_rate": 0.0,
        }
    diagnostic["overflow_rate_macro"] = 0.0
    diagnostic["support"] = "continuous_gmm"
    diagnostic["sample_size_per_effect"] = int(
        target.shape[0] * target.shape[1]
    )
    return diagnostic


def diagnose_batch_cepo_support(batch):
    """Describe the four exact CEPO training labels, not sampled contrasts."""
    target = cepo_targets(batch)
    diagnostic = {"prediction_protocol": PREDICTION_PROTOCOL,
                  "sample_size_per_arm": int(target.shape[0]*target.shape[1]),
                  "support": "continuous_gmm"}
    for index, arm in enumerate(ARM_NAMES):
        values = target[..., index].detach().cpu().flatten()
        diagnostic[arm] = ({"minimum": float(values.min()), "maximum": float(values.max()),
                           "q001": float(torch.quantile(values, .001)),
                           "q999": float(torch.quantile(values, .999))} if values.numel() else None)
    return diagnostic


def diagnose_query_effect_support(
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """Inspect simulator labels; retained for diagnostics and unit tests."""

    batch = generate_batch(
        data_config, batch_size, seed=seed, device=device
    )
    return diagnose_batch_effect_support(batch, model_config)


def create_optimizer(
    parameters,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """Create the requested Adam optimizer."""

    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def create_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float = 0.5,
    patience: int = 5,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Reduce LR only when validation GMM NLL plateaus."""

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
    )


def epoch_episode_indices(
    episode_count: int,
    *,
    training_seed: int,
    epoch: int,
) -> torch.Tensor:
    """Deterministic no-replacement episode order for one epoch."""

    if episode_count <= 0:
        raise ValueError("episode_count must be positive.")
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    generator = _make_generator(training_seed + epoch)
    return torch.randperm(episode_count, generator=generator)


def distributed_epoch_episode_indices(
    episode_count: int,
    *,
    training_seed: int,
    epoch: int,
    rank: int,
    world_size: int,
    batch_size: int,
) -> torch.Tensor:
    """Return one equally sized deterministic DDP shard for an epoch.

    The global permutation is padded from its beginning so every rank executes
    the same number of optimizer steps. Padding is needed only when the episode
    count is not divisible by ``world_size * batch_size``.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must lie in [0, world_size).")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    order = epoch_episode_indices(
        episode_count,
        training_seed=training_seed,
        epoch=epoch,
    )
    global_batch_size = world_size * batch_size
    total_size = int(math.ceil(episode_count / global_batch_size) * global_batch_size)
    padding = total_size - episode_count
    if padding > 0:
        repeats = int(math.ceil(padding / episode_count))
        pad_values = order.repeat(repeats)[:padding]
        order = torch.cat((order, pad_values), dim=0)
    return order[rank:total_size:world_size].clone()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
) -> float:
    """Update model parameters once using CEPO GMM NLL."""

    metrics = train_step_metrics(model, optimizer, batch)
    return metrics["total_loss"]


def train_step_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
) -> Dict[str, float]:
    """Update once and return pre-update batch diagnostics for logging."""

    model.train()
    predictions = model(batch["tokens"], cepo_queries(batch), batch["adjacency"])
    losses = compute_training_losses(model, predictions, batch)
    metrics = compute_query_metrics(
        model,
        {key: value.detach() for key, value in predictions.items()},
        batch,
        include_intervals=os.environ.get("PFN_TRAIN_INTERVALS", "0") == "1",
    )
    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    optimizer.step()
    return {
        **metrics,
        **{
            key: float(value.detach().item())
            for key, value in losses.items()
            if value.ndim == 0
        },
    }


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batch: TensorBatch,
    *,
    batch_size: int,
    training_seed: int,
    epoch: int,
    device: torch.device,
    runtime: Optional[DistributedRuntime] = None,
    return_metrics: bool = False,
    data_is_rank_sharded: bool = False,
) -> Union[float, Dict[str, float]]:
    """Train one epoch, optionally from a pre-sharded rank-local episode bank."""

    runtime = runtime or DistributedRuntime()
    episode_count = int(train_batch["tokens"].shape[0])
    if runtime.is_distributed and not data_is_rank_sharded:
        order = distributed_epoch_episode_indices(
            episode_count,
            training_seed=training_seed,
            epoch=epoch,
            rank=runtime.rank,
            world_size=runtime.world_size,
            batch_size=batch_size,
        )
    else:
        order = epoch_episode_indices(
            episode_count,
            training_seed=training_seed,
            epoch=epoch,
        )
    accumulator = MetricAccumulator()
    for start in range(0, int(order.numel()), batch_size):
        indices = order[start : start + batch_size]
        batch = select_episode_batch(train_batch, indices, device=device)
        batch_metrics = train_step_metrics(model, optimizer, batch)
        step = start // batch_size + 1
        if step == 1 or step % 20 == 0 or start + batch_size >= int(order.numel()):
            print(f"[rank {runtime.rank}] epoch={epoch} train_step={step}/{math.ceil(order.numel()/batch_size)} loss={batch_metrics.get('total_loss', float('nan')):.6f}", flush=True)
        target = effect_targets(batch)
        batch_weight = int(target.shape[0] * target.shape[1])
        # Normalized metrics need a pooled denominator too. Keep float64
        # statistics on CPU: MUSA kernels do not support every double op.
        abs_truth_means = target.detach().cpu().double().abs().mean(dim=(0, 1)).tolist()
        accumulator.add(batch_metrics, batch_weight, abs_truth_means)
    if runtime.is_distributed:
        metric_names = sorted(accumulator.sums)
        aggregate = torch.tensor(
            [accumulator.sums[name] for name in metric_names]
            + accumulator.abs_truth + [float(accumulator.count)],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM, group=getattr(model, "process_group", None))
        values = aggregate.detach().cpu().tolist()
        accumulator.count = int(round(values[-1]))
        accumulator.abs_truth = values[-4:-1]
        accumulator.sums = {
            name: values[index]
            for index, name in enumerate(metric_names)
        }
    # sqrt(sum of squared errors / count), never an average of batch RMSEs.
    # These remain pre-update training diagnostics; the objective is GMM NLL.
    averaged_metrics = accumulator.result()
    if return_metrics:
        return averaged_metrics
    return float(
        averaged_metrics.get(
            "cepo_gmm_nll_macro",
            averaged_metrics.get("total_loss", float("nan")),
        )
    )


def evaluate(
    model: LocalNetworkQueryTransformer,
    validation_batch: TensorBatch,
    *, batch_size: int, device: torch.device,
    runtime: Optional[DistributedRuntime] = None,
    distributed_validation: bool = False,
) -> Dict[str, float]:
    """Bounded CPU metric batches; exact pooled point/NLL/interval statistics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    runtime = runtime or DistributedRuntime()
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator()
    count = int(validation_batch["tokens"].shape[0])
    try:
        with torch.no_grad():
            for start in range(0, count, batch_size):
                indices = torch.arange(start, min(start+batch_size,count), dtype=torch.long)
                batch = select_episode_batch(validation_batch, indices, device=device)
                pred = model(batch["tokens"], cepo_queries(batch), batch["adjacency"])
                # CPU float32 GMM calculations preserve the previous validation
                # path, but never allocate CDF work arrays for the entire bank.
                pred_cpu = {key:value.detach().cpu() for key,value in pred.items()}
                truth = effect_targets(batch).detach().cpu()
                metrics = compute_query_metrics(model, pred_cpu, {"cepo_target":cepo_targets(batch).detach().cpu(), "oracle_ite":truth})
                accumulator.add(metrics, truth.shape[0]*truth.shape[1],
                                [truth[...,q].abs().double().mean().item() for q in range(3)])
                step = start//batch_size+1
                if step == 1 or step%20 == 0 or start+batch_size>=count:
                    print(f"[rank {runtime.rank}] validation_step={step}/{math.ceil(count/batch_size)}", flush=True)
        if distributed_validation and runtime.is_distributed:
            # Small CPU objects only. Ranks with zero validation examples still
            # participate; there is no padding, duplication or MUSA wait kernel.
            parts = [None for _ in range(runtime.world_size)]
            dist.all_gather_object(parts, accumulator.payload())
            accumulator = MetricAccumulator()
            for part in parts:
                accumulator.merge(part)
        return accumulator.result()
    finally:
        model.train(was_training)


def checkpoint_paths(base_path: Path) -> Dict[str, Path]:
    base = Path(base_path)
    return {
        "best_nll": base,
        "best_point": base.with_name(f"{base.stem}_best_point{base.suffix}"),
        "latest": base.with_name(f"{base.stem}_latest{base.suffix}"),
    }


def save_checkpoint(
    *,
    path: Path,
    model: LocalNetworkQueryTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    epoch: int,
    best_validation_gmm_nll: float,
    best_point_rmse: float,
    monitor_name: str,
    monitor_value: float,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prediction_protocol": PREDICTION_PROTOCOL,
            "cepo_label_protocol": CEPO_LABEL_PROTOCOL,
            "epoch": int(epoch),
            "best_validation_gmm_nll": float(best_validation_gmm_nll),
            "best_point_rmse": float(best_point_rmse),
            "monitor_name": str(monitor_name),
            "monitor_value": float(monitor_value),
            "data_config": asdict(data_config),
            "model_config": asdict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        },
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    device: torch.device,
) -> tuple[LocalNetworkQueryTransformer, torch.optim.Adam, Dict[str, object]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("prediction_protocol") != PREDICTION_PROTOCOL:
        raise ValueError("This CEPO model requires a majority_cepo_four_gmm_v2 checkpoint; old three-arm/effect checkpoints must be retrained.")
    data_config = data_config_from_mapping(checkpoint["data_config"])
    if (data_config.dgp_version == "gao_ding_design2_outcome"
            and checkpoint.get("cepo_label_protocol") != CEPO_LABEL_PROTOCOL):
        raise ValueError("This default-DGP checkpoint may contain noisy CEPO labels; retrain with noise-free labels.")
    model_config = ModelConfig(**checkpoint["model_config"])
    model = LocalNetworkQueryTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = create_optimizer(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    metadata: Dict[str, object] = {
        "prediction_protocol": PREDICTION_PROTOCOL,
        "epoch": int(checkpoint["epoch"]),
        "best_validation_gmm_nll": float(
            checkpoint["best_validation_gmm_nll"]
        ),
        "best_point_rmse": float(checkpoint["best_point_rmse"]),
        "monitor_name": str(checkpoint["monitor_name"]),
        "monitor_value": float(checkpoint["monitor_value"]),
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
    }
    return model, optimizer, metadata


def train(
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_batch: TensorBatch,
    validation_batch: TensorBatch,
    device: torch.device,
    checkpoint_path: Path,
    runtime: Optional[DistributedRuntime] = None,
    rank_sharded_train: bool = False,
    distributed_validation: bool = False,
) -> LocalNetworkQueryTransformer:
    """Train by complete CSV epochs with optional torchrun DDP."""

    train_batch = {**train_batch, "cepo_target": cepo_targets(train_batch, data_config.treatment_prob)}
    validation_batch = {**validation_batch, "cepo_target": cepo_targets(validation_batch, data_config.treatment_prob)}
    runtime = runtime or DistributedRuntime()

    if train_batch["tokens"].shape[0] <= 0:
        raise ValueError("training split must contain at least one episode.")
    if runtime.is_main and not distributed_validation and validation_batch["tokens"].shape[0] <= 0:
        raise ValueError("validation split must contain at least one episode on rank zero.")

    if runtime.is_main:
        training_diagnostic_name = (
            "rank0_training_label_diagnostic"
            if runtime.is_distributed and rank_sharded_train
            else "training_label_diagnostic"
        )
        print(
            training_diagnostic_name
            + "="
            + json.dumps(
                diagnose_batch_cepo_support(train_batch),
                ensure_ascii=False,
            )
        )
        print(
            ("rank0_validation_label_diagnostic=" if distributed_validation else "validation_label_diagnostic=")
            + json.dumps(
                diagnose_batch_cepo_support(validation_batch),
                ensure_ascii=False,
            )
        )

    if runtime.is_distributed and dist.is_initialized():
        print(f"[rank {runtime.rank}] 数据与统计准备完成，等待 CPU 同步", flush=True)
        distributed_barrier(runtime, device)
    seed_everything(train_config.training_seed)
    raw_model = LocalNetworkQueryTransformer(model_config).to(device)
    if device.type in {"musa", "privateuseone"}:
        torch.musa.synchronize()
    print(f"[rank {runtime.rank}] 模型上卡完成", flush=True)
    train_group = None
    if runtime.is_distributed and dist.is_initialized() and device.type in {"musa", "privateuseone"}:
        print(f"[rank {runtime.rank}] 开始 MCCL 训练通信初始化", flush=True)
        train_group = dist.new_group(
            backend="mccl", timeout=timedelta(seconds=int(os.environ.get("PFN_TRAIN_TIMEOUT", "600"))),
        )
    if runtime.is_distributed:
        device_index = device.index
        ddp_device_ids = None if device.type == "cpu" else [device_index]
        model: nn.Module = DistributedDataParallel(
            raw_model,
            device_ids=ddp_device_ids,
            output_device=(device_index if ddp_device_ids else None),
            broadcast_buffers=False,
            process_group=train_group,
        )
        print(f"[rank {runtime.rank}] DDP 初始化完成", flush=True)
    else:
        model = raw_model
    optimizer = create_optimizer(
        raw_model.parameters(),
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = create_plateau_scheduler(
        optimizer,
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
    )
    paths = checkpoint_paths(checkpoint_path)
    best_validation_gmm_nll = float("inf")
    best_point_rmse = float("inf")

    for epoch in range(1, train_config.epochs + 1):
        train_result = train_epoch(
            model,
            optimizer,
            train_batch,
            batch_size=train_config.batch_size,
            training_seed=train_config.training_seed,
            epoch=epoch,
            device=device,
            runtime=runtime,
            return_metrics=True,
            data_is_rank_sharded=rank_sharded_train,
        )
        if isinstance(train_result, Mapping):
            train_metrics = {
                str(name): float(value) for name, value in train_result.items()
            }
        else:
            train_metrics = {
                "cepo_gmm_nll_macro": float(train_result),
                "total_loss": float(train_result),
            }
        metric_names = (
            "validation_gmm_nll",
            "effect_rmse_macro",
            "direct_effect_rmse",
            "spillover_effect_rmse",
            "total_effect_rmse",
        )
        if runtime.is_main or distributed_validation:
            kwargs = {"runtime":runtime, "distributed_validation":True} if distributed_validation else {}
            metrics = evaluate(raw_model, validation_batch,
                               batch_size=train_config.batch_size, device=device, **kwargs)
        else:
            metrics = {name: 0.0 for name in metric_names}
        if runtime.is_distributed and not distributed_validation:
            metric_tensor = torch.tensor(
                [metrics[name] for name in metric_names],
                dtype=torch.float32,
                device="cpu",
            )
            dist.broadcast(metric_tensor, src=0)
            metrics.update({
                name: float(metric_tensor[index].item())
                for index, name in enumerate(metric_names)
            })
        validation_gmm_nll = metrics["validation_gmm_nll"]

        # The scheduler receives only the complete validation-set GMM NLL.
        scheduler.step(validation_gmm_nll)

        nll_improved = validation_gmm_nll < best_validation_gmm_nll
        point_improved = metrics["effect_rmse_macro"] < best_point_rmse
        if nll_improved:
            best_validation_gmm_nll = validation_gmm_nll
        if point_improved:
            best_point_rmse = metrics["effect_rmse_macro"]
        if runtime.is_main and nll_improved:
            save_checkpoint(
                path=paths["best_nll"],
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_gmm_nll=best_validation_gmm_nll,
                best_point_rmse=best_point_rmse,
                monitor_name="validation_gmm_nll",
                monitor_value=best_validation_gmm_nll,
                data_config=data_config,
                model_config=model_config,
            )
        if runtime.is_main and point_improved:
            save_checkpoint(
                path=paths["best_point"],
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_gmm_nll=best_validation_gmm_nll,
                best_point_rmse=best_point_rmse,
                monitor_name="effect_rmse_macro",
                monitor_value=best_point_rmse,
                data_config=data_config,
                model_config=model_config,
            )
        if runtime.is_main:
            save_checkpoint(
                path=paths["latest"],
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_gmm_nll=best_validation_gmm_nll,
                best_point_rmse=best_point_rmse,
                monitor_name="latest",
                monitor_value=validation_gmm_nll,
                data_config=data_config,
                model_config=model_config,
            )

        if runtime.is_main:
            entries = [f"epoch={epoch:04d}/{train_config.epochs:04d}"]
            for split, values in (("train", train_metrics), ("validation", metrics)):
                entries.append(f"{split}_cepo_gmm_nll={values.get('cepo_gmm_nll_macro', float('nan')):.8f}")
                for arm in ARM_NAMES:
                    entries.append(f"{split}_{arm}_nll={values.get(arm+'_gmm_nll', float('nan')):.8f}")
                for effect in QUERY_TYPE_NAMES:
                    entries.append(f"{split}_{effect}_rmse={values.get(effect+'_effect_rmse', float('nan')):.8f}")
            entries.extend([f"lr={optimizer.param_groups[0]['lr']:.8g}",
                            f"best_nll={int(nll_improved)}", f"best_point={int(point_improved)}"])
            print(" ".join(entries), flush=True)
        if runtime.is_main:
            history_path = Path(checkpoint_path).parent / "history.jsonl"
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch":epoch,"validation":metrics,"train":train_metrics,
                                         "lr":optimizer.param_groups[0]["lr"]}, ensure_ascii=False)+"\n")
            if os.environ.get("PFN_PLOT_VALIDATION_CURVES") == "1":
                from pfn_interference.plotting import plot_history
                plot_history(history_path, history_path.parent)
        if runtime.is_distributed and dist.is_initialized():
            distributed_barrier(runtime, device)

    distributed_barrier(runtime, device)
    if runtime.is_main:
        print(f"best_nll_checkpoint={paths['best_nll']}")
        print(f"best_point_checkpoint={paths['best_point']}")
        print(f"latest_checkpoint={paths['latest']}")
    return raw_model


def apply_algorithm_2(model, data_config, *, num_datasets=10, seed=2026, device, batch=None):
    """Predict marginal CEPOs and derive coherent majority-effect point estimates."""
    from pfn_interference.cepo import predict_mu_distributions
    if num_datasets <= 0:
        raise ValueError("num_datasets must be positive.")
    if batch is None:
        batch = generate_batch(data_config, num_datasets, seed=seed, device=device)
    if batch['tokens'].shape[0] != num_datasets:
        raise ValueError("shared test batch size must equal num_datasets.")
    batch = {**batch, 'cepo_target': cepo_targets(batch, data_config.treatment_prob)}
    predictions = predict_mu_distributions(model, batch)
    means = gmm_posterior_mean(predictions['gmm_pi'], predictions['gmm_mu'])
    effects = effects_from_mu(means)
    truth = effect_targets(batch)
    overall = compute_query_metrics(model, predictions, batch)
    dataset_rows, unit_rows, mu_rows = [], [], []
    for b in range(num_datasets):
        local = {key:value[b:b+1] for key,value in batch.items()}
        pp = {key:value[b:b+1] for key,value in predictions.items()}
        graph_name = GRAPH_TYPE_NAMES[int(batch['graph_type'][b])]
        dataset_rows.append(dict(dataset_id=b+1,graph_type=graph_name,
            **compute_query_metrics(model,pp,local)))
        for i in range(means.shape[1]):
            for j, effect in enumerate(QUERY_TYPE_NAMES):
                row = dict(dataset_id=b+1,unit_id=i+1,graph_type=graph_name,query_type=effect,
                           effect_true=float(truth[b,i,j]),effect_pred=float(effects[b,i,j]),
                           effect_absolute_error=float((effects[b,i,j]-truth[b,i,j]).abs()))
                for key in ('x','tau','gamma','eta','observed_treatment','observed_exposure'):
                    row[key] = float(batch[key][b,i])
                unit_rows.append(row)
            for j, arm in enumerate(ARM_NAMES):
                row = dict(dataset_id=b+1,unit_id=i+1,arm=arm,
                           mu_true=float(batch['cepo_target'][b,i,j]),mu_pred=float(means[b,i,j]))
                for key in ('gmm_pi','gmm_mu','gmm_sigma'):
                    row[key] = json.dumps(predictions[key][b,i,j].cpu().tolist())
                mu_rows.append(row)
    n = int(means.shape[1])
    return dict(algorithm='CEPO marginal GMMs with shared-mean effect contrasts',
        prediction_protocol=PREDICTION_PROTOCOL, test_seed=seed,num_datasets=num_datasets,
        num_units_per_dataset=n,num_observed_units_per_dataset=n,num_queried_units_per_dataset=n,
        num_evaluated_units_per_dataset=n,num_query_types=3,gmm_n_components=model.config.gmm_n_components,
        effect_uncertainty='not computed: no joint posterior assumed',
        overall=overall,datasets=dataset_rows,units=unit_rows,mu_predictions=mu_rows)


def save_algorithm_2_report(
    report: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "algorithm2_summary.json"
    dataset_path = output_dir / "algorithm2_dataset_metrics.csv"
    unit_path = output_dir / "algorithm2_unit_estimates.csv"
    summary = {
        key: value for key, value in report.items() if key not in {"datasets", "units"}
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(summary),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    dataset_rows = list(report["datasets"])
    unit_rows = list(report["units"])
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dataset_rows)
    with unit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unit_rows)
    return {
        "summary_json": summary_path,
        "dataset_csv": dataset_path,
        "unit_csv": unit_path,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_result_number(value: object, *, digits: int = 6) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _run_paper_baselines(
    *,
    model: nn.Module,
    data_config: "DataConfig",
    args: argparse.Namespace,
    batch: TensorBatch,
) -> None:
    """Run the unified paper-facing ITE/ATE benchmark on one shared test batch."""

    if bool(getattr(args, "skip_paper_baselines", False)):
        print("unified_benchmark_skipped=1")
        return
    if bool(getattr(args, "skip_localized_baselines", False)):
        raise ValueError(
            "--skip-localized-baselines is incompatible with the unified main benchmark; "
            "use --skip-paper-baselines to skip the complete benchmark."
        )

    causalpfn_model = None
    if not bool(getattr(args, "skip_causalpfn", False)):
        model_parameter = next(model.parameters(), None)
        project_device = model_parameter.device if model_parameter is not None else torch.device("cpu")
        # The upstream checkpoint is verified here on standard PyTorch CPU/CUDA.
        # Keep it on CPU when the project PFN runs on MUSA rather than assuming
        # third-party long-context kernels are MUSA-compatible.
        causalpfn_device = (
            torch.device("cpu")
            if project_device.type == "musa"
            else project_device
        )
        try:
            causalpfn_model = load_causalpfn_checkpoint(
                Path(getattr(args, "causalpfn_checkpoint", Path("checkpoints/causalpfn_v0.pt"))),
                causalpfn_device,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc} Provide --causalpfn-checkpoint PATH or use --skip-causalpfn."
            ) from exc

    report = evaluate_unified_benchmark(
        model=model,
        batch=batch,
        treatment_prob=float(data_config.treatment_prob),
        ate_bandwidth=int(getattr(args, "baseline_bandwidth", 3)),
        ate_ridge=float(getattr(args, "baseline_ridge", 0.0)),
        **localized_benchmark_kwargs(args),
        seed=int(getattr(args, "test_seed", 2026)),
        nsi_pcr_rank_max=int(getattr(args, "nsi_pcr_rank_max", 6)),
        nsi_min_effective_donors=float(getattr(args, "nsi_min_effective_donors", 12.0)),
        nsi_max_bandwidth=float(getattr(args, "nsi_max_bandwidth", 0.50)),
        nsi_min_support_mass=float(getattr(args, "nsi_min_support_mass", 0.99)),
        nsi_svd_rcond=float(getattr(args, "nsi_svd_rcond", 1.0e-6)),
        standard_ite_gps_ridge=float(getattr(args, "standard_ite_gps_ridge", 1e-3)),
        standard_ite_epochs=int(getattr(args, "standard_ite_epochs", 500)),
        standard_ite_learning_rate=float(
            getattr(args, "standard_ite_learning_rate", 1e-3)
        ),
        standard_ite_balance_weight=float(
            getattr(args, "standard_ite_balance_weight", 1e-2)
        ),
        standard_ite_hidden_dim=int(getattr(args, "standard_ite_hidden_dim", 32)),
        hypersci_arm_samples=int(getattr(args, "hypersci_arm_samples", 4096)),
        tnet_epochs=int(getattr(args, "tnet_epochs", 160)),
        tnet_hidden_dim=int(getattr(args, "tnet_hidden_dim", 64)),
        tnet_grid_size=int(getattr(args, "tnet_grid_size", 20)),
        tnet_spline_basis=int(getattr(args, "tnet_spline_basis", 12)),
        tnet_learning_rate_1step=float(
            getattr(args, "tnet_learning_rate_1step", 1.0e-4)
        ),
        tnet_learning_rate_2step=float(
            getattr(args, "tnet_learning_rate_2step", 1.0e-2)
        ),
        causalpfn_model=causalpfn_model,
        causalpfn_query_chunk_size=int(
            getattr(args, "causalpfn_query_chunk_size", 512)
        ),
    )
    paths = save_unified_benchmark(report, Path(args.test_output_dir))
    for name, path in paths.items():
        print(f"unified_benchmark_{name}={path}")


def run_smoke_test(device: torch.device, checkpoint_path: Path) -> None:
    data_config = DataConfig(n_units=N_UNITS)
    model_config = ModelConfig(
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=32,
        gmm_n_components=5,
    )
    seed_everything(123)
    master = generate_batch(
        data_config,
        batch_size=10,
        seed=12345,
        device=torch.device("cpu"),
    )
    smoke_csv = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_smoke_episodes.csv"
    )
    train_csv, validation_csv = episode_split_paths(smoke_csv)
    save_episode_batch_csv(master, smoke_csv)
    split_episode_csv(
        smoke_csv,
        train_csv,
        validation_csv,
        validation_fraction=0.2,
        split_seed=0,
    )
    training_batch, _ = load_episode_batch_csv(
        train_csv,
        device=torch.device("cpu"),
    )
    validation_batch, _ = load_episode_batch_csv(
        validation_csv,
        device=torch.device("cpu"),
    )

    training_batch["cepo_target"] = cepo_targets(training_batch, data_config.treatment_prob)
    validation_batch["cepo_target"] = cepo_targets(validation_batch, data_config.treatment_prob)
    model = LocalNetworkQueryTransformer(model_config).to(device)
    optimizer = create_optimizer(model.parameters())
    scheduler = create_plateau_scheduler(optimizer)
    batch = select_episode_batch(
        training_batch,
        torch.arange(8),
        device=device,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, optimizer, batch)
    if not any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    ):
        raise RuntimeError("Smoke test failed: optimizer did not update parameters.")
    metrics = evaluate(
        model,
        validation_batch,
        batch_size=2,
        device=device,
    )
    scheduler.step(metrics["validation_gmm_nll"])
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_validation_gmm_nll=metrics["validation_gmm_nll"],
        best_point_rmse=metrics["effect_rmse_macro"],
        monitor_name="validation_gmm_nll",
        monitor_value=metrics["validation_gmm_nll"],
        data_config=data_config,
        model_config=model_config,
    )
    restored, _, metadata = load_checkpoint(
        path=checkpoint_path,
        device=device,
    )
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(batch["tokens"], cepo_queries(batch), batch["adjacency"])
        actual = restored(batch["tokens"], cepo_queries(batch), batch["adjacency"])
    for key in expected:
        if not torch.equal(expected[key], actual[key]):
            raise RuntimeError(
                f"Smoke test failed: checkpoint changed {key} predictions."
            )
    print(f"device={device}")
    print(f"master_csv={smoke_csv}")
    print(f"training_csv={train_csv}")
    print(f"validation_csv={validation_csv}")
    print(f"training_episodes={training_batch['tokens'].shape[0]}")
    print(f"validation_episodes={validation_batch['tokens'].shape[0]}")
    print(f"tokens_shape={tuple(batch['tokens'].shape)}")
    print(f"cepo_queries_shape={tuple(cepo_queries(batch).shape)}")
    print(f"gmm_pi_shape={tuple(expected['gmm_pi'].shape)}")
    print(f"gmm_mu_shape={tuple(expected['gmm_mu'].shape)}")
    print(f"gmm_sigma_shape={tuple(expected['gmm_sigma'].shape)}")
    print(f"train_gmm_nll={loss:.8f}")
    print(f"validation_gmm_nll={metrics['validation_gmm_nll']:.8f}")
    print(f"effect_rmse_macro={metrics['effect_rmse_macro']:.8f}")
    print(f"checkpoint_epoch={metadata['epoch']}")
    print("smoke_test=passed")


def _print_algorithm_2_summary(
    report: Mapping[str, object],
    paths: Mapping[str, Path],
) -> None:
    overall = cast(Mapping[str, float], report["overall"])
    print(f"algorithm2_test_seed={report['test_seed']}")
    print(f"algorithm2_num_datasets={report['num_datasets']}")
    print(f"gmm_n_components={report['gmm_n_components']}")
    for prefix in EFFECT_PREFIXES:
        print(f"{prefix}_rmse={overall[prefix+'_rmse']:.8f}")
    print(f"cepo_gmm_nll_macro={overall['cepo_gmm_nll_macro']:.8f} "
          f"effect_rmse_macro={overall['effect_rmse_macro']:.8f}")
    for name, path in paths.items():
        print(f"{name}={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-arm CEPO GMM pretraining and derived interference effects."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("local_interference_gmm.pt"),
    )

    parser.add_argument(
        "--n-units",
        type=int,
        default=N_UNITS,
        help=(
            "Units per episode. All units are observed context nodes and all "
            "units receive causal queries; default: 1000."
        ),
    )
    parser.add_argument(
        "--er-edge-probability",
        type=float,
        default=0.02,
        help="ER edge probability; default 0.02 gives expected degree 19.98 at N=1000.",
    )
    parser.add_argument(
        "--graph-family",
        choices=("er", "configuration", "rgg", "sbm", "mixed"),
        default="er",
        help=(
            "Graph family for synthetic episodes. 'mixed' cycles equally over "
            "ER/configuration/RGG/SBM while matching the ER mean degree."
        ),
    )
    parser.add_argument("--dgp-version", choices=("gao_ding_design2_outcome", "tnet_linear_neighbor_covariate", "legacy_v7", "random_lpe_v1", "a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"),
                        default="a_group_fixed_v1",
                        help="Default A-group DGP (epsilon SD=4); legacy DGPs require an explicit selection.")
    parser.add_argument("--interference-lambda", type=float, default=1.0,
                        help="A-group multiplier of neighbor-X and exposure outcome coefficients.")
    parser.add_argument("--neighbor-beta-min", type=float, default=-6.0)
    parser.add_argument("--neighbor-beta-max", type=float, default=0.0)
    parser.add_argument("--neighbor-beta", type=float, default=None,
                        help="Fixed neighbor coefficient for paired evaluation; otherwise uniform per task.")
    parser.add_argument("--spillover-gamma-min", type=float, default=-3.6,
                        help="Task gamma lower bound for random-gamma or random-coefficient A-group priors.")
    parser.add_argument("--spillover-gamma-max", type=float, default=0.0,
                        help="Random-gamma prior upper bound.")
    parser.add_argument("--spillover-gamma", type=float, default=None,
                        help="Fixed task gamma override; omit for uniform training.")
    parser.add_argument("--nointerference-prob", type=float, default=0.5,
                        help="Random-DGP probability of BOTH neighbor coefficients being zero (only random_lpe_v1).")
    parser.add_argument("--random-noise-sd", type=float, default=1.0,
                        help="Only random_lpe_v1: outcome noise SD. A group fixes SD=4 independently of this option.")
    parser.add_argument("--treatment-prob", type=float, default=0.5)
    parser.add_argument("--graph-protocol", choices=("resampled", "fixed_er_v1"), default=None,
                        help="Default: fixed_er_v1 for A group, resampled for explicit legacy DGPs.")
    parser.add_argument("--fixed-graph-seed", type=int, default=12345)
    parser.add_argument("--fixed-graph-file", type=Path,
                        help="Create once, then read the persisted fixed ER graph (JSON).")
    parser.add_argument("--x-sd", type=float, default=1.0)
    parser.add_argument("--x-outcome-coefficient", type=float, default=0.0)
    parser.add_argument("--own-score-baseline-coefficient", type=float, default=1.0)
    parser.add_argument("--neighbor-score-baseline-coefficient", type=float, default=0.5)
    parser.add_argument("--baseline-sd", type=float, default=1.0)
    parser.add_argument("--unit-sd", type=float, default=0.2)
    parser.add_argument("--tau-mean", type=float, default=0.4)
    parser.add_argument("--tau-episode-sd", type=float, default=0.0)
    parser.add_argument("--tau-unit-sd", type=float, default=0.0)
    parser.add_argument("--tau-own-score-slope", type=float, default=0.2)
    parser.add_argument("--tau-neighbor-score-slope", type=float, default=0.2)
    parser.add_argument("--gamma-mean", type=float, default=0.2)
    parser.add_argument("--gamma-sd", type=float, default=0.0)
    parser.add_argument("--gamma-own-score-slope", type=float, default=0.1)
    parser.add_argument("--gamma-neighbor-score-slope", type=float, default=0.2)
    parser.add_argument("--eta-mean", type=float, default=0.0)
    parser.add_argument("--eta-sd", type=float, default=0.0)
    parser.add_argument("--noise-sd", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gmm-n-components", type=int, default=5)
    parser.add_argument("--gmm-min-sigma", type=float, default=1e-3)
    parser.add_argument("--gmm-pi-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument(
        "--training-seed",
        "--seed",
        dest="training_seed",
        type=int,
        default=0,
        help="Seed for model initialization and per-epoch training shuffles.",
    )

    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("training_episodes.csv"),
        help="Master episode CSV used for the 80/20 split.",
    )
    parser.add_argument(
        "--csv-datasets",
        type=int,
        default=2560,
        help="Episodes generated when --data-csv does not yet exist.",
    )
    parser.add_argument(
        "--csv-generation-batch-size",
        type=int,
        default=64,
        help="Episodes generated at once while building the master CSV.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=12345,
        help="Seed used only to create the master training CSV.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of complete episodes reserved for validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for deterministic episode-level train/validation splitting.",
    )
    parser.add_argument(
        "--overwrite-data-csv",
        action="store_true",
        help="Regenerate the master CSV before splitting.",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help=(
            "Generate/split the CSV in a single process and exit without "
            "loading tensors or starting training."
        ),
    )
    parser.add_argument(
        "--ddp-shard-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing physical train_rank{rank}.csv files. In DDP "
            "mode each rank loads only its own shard; rank zero alone loads validation."
        ),
    )
    parser.add_argument(
        "--prepare-ddp-shards-only",
        action="store_true",
        help=(
            "Shard the already-prepared training CSV for DDP in one CPU process "
            "and exit. Requires --ddp-shard-dir."
        ),
    )
    parser.add_argument(
        "--ddp-shard-world-size",
        type=int,
        default=8,
        help="Number of equal physical training shards to prepare; default: 8.",
    )

    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-test-after-training", action="store_true")
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument("--test-datasets", type=int, default=10)
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=Path("algorithm2_gmm_test_results"),
    )
    parser.add_argument(
        "--skip-paper-baselines",
        action="store_true",
        help="Skip repeated PFN-versus-paper average-effect RMSE benchmarks.",
    )
    parser.add_argument(
        "--skip-localized-baselines",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The unified main benchmark includes "
            "Localized DR-Lasso; use --skip-paper-baselines to skip all pfn_interference.baselines."
        ),
    )
    parser.add_argument(
        "--causalpfn-checkpoint",
        type=Path,
        default=Path("checkpoints/causalpfn_v0.pt"),
        help=(
            "Local official CausalPFN checkpoint used for inference-only Direct "
            "baseline."
        ),
    )
    parser.add_argument(
        "--causalpfn-query-chunk-size",
        type=int,
        default=512,
        help="Maximum number of CausalPFN CEPO queries per inference chunk.",
    )
    parser.add_argument(
        "--skip-causalpfn",
        action="store_true",
        help="Skip the pretrained CausalPFN Direct baseline.",
    )
    parser.add_argument(
        "--baseline-repetitions",
        type=int,
        default=20,
        help=(
            "Deprecated and ignored. Query-unit baselines now reuse the exact "
            "shared --test-datasets batch."
        ),
    )
    parser.add_argument(
        "--baseline-datasets",
        type=int,
        default=64,
        help=(
            "Deprecated and ignored. Increase --test-datasets to evaluate both "
            "PFN and baselines on more identical test episodes."
        ),
    )
    parser.add_argument(
        "--baseline-bandwidth",
        type=int,
        default=3,
        help="Strict network-distance cutoff used by the HAC objectives.",
    )
    parser.add_argument(
        "--baseline-ridge",
        type=float,
        default=0.0,
        help="Legacy option, unused by paper-aligned F/L and reg-net (all use zero ridge).",
    )
    add_localized_arguments(parser)
    parser.add_argument(
        "--nsi-pcr-rank-max",
        type=int,
        default=6,
        help="Maximum retained PCR rank for the cross-sectional NSI adaptation.",
    )
    parser.add_argument(
        "--nsi-min-effective-donors",
        type=float,
        default=12.0,
        help="Minimum Gaussian-kernel effective donor count for an NSI exact-exposure query.",
    )
    parser.add_argument(
        "--nsi-max-bandwidth",
        type=float,
        default=0.50,
        help="Maximum exposure-localization radius allowed by NSI before declaring unsupported.",
    )
    parser.add_argument(
        "--nsi-min-support-mass",
        type=float,
        default=0.99,
        help="Minimum conditional exact-count probability mass required for an NSI majority arm.",
    )
    parser.add_argument(
        "--nsi-svd-rcond",
        type=float,
        default=1.0e-6,
        help="Relative singular-value threshold used to define NSI local numerical rank.",
    )
    parser.add_argument(
        "--standard-ite-gps-ridge",
        type=float,
        default=1e-3,
        help="Ridge stabilizer for the factual GPS outcome regression baseline.",
    )
    parser.add_argument(
        "--standard-ite-epochs", "--hypersci-epochs",
        type=int,
        default=500,
        help="Per-graph training epochs for HyperSCI ITE pfn_interference.baselines.",
    )
    parser.add_argument(
        "--standard-ite-learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate for HyperSCI ITE pfn_interference.baselines.",
    )
    parser.add_argument(
        "--standard-ite-balance-weight",
        type=float,
        default=1e-2,
        help="Wasserstein representation-balancing weight for HyperSCI.",
    )
    parser.add_argument(
        "--standard-ite-hidden-dim",
        type=int,
        default=32,
        help="Hidden representation width for HyperSCI.",
    )
    parser.add_argument("--hypersci-arm-samples", type=int, default=4096,
                        help="Conditional neighbor assignments per root and majority arm for HyperSCI.")
    parser.add_argument(
        "--tnet-epochs",
        type=int,
        default=160,
        help="Per-graph nuisance/targeted training epochs for the Chen et al. (ICML 2024) TNet adaptation.",
    )
    parser.add_argument(
        "--tnet-hidden-dim",
        type=int,
        default=64,
        help="GCN/MLP hidden width for the Chen et al. (ICML 2024) TNet adaptation.",
    )
    parser.add_argument(
        "--tnet-grid-size",
        type=int,
        default=20,
        help="Number of intervals for TNet's piecewise-linear neighborhood-exposure density model.",
    )
    parser.add_argument(
        "--tnet-spline-basis",
        type=int,
        default=12,
        help="Number of truncated-power basis functions for TNet's targeted perturbation epsilon(t,z).",
    )
    parser.add_argument(
        "--tnet-learning-rate-1step",
        type=float,
        default=1.0e-4,
        help="Adam learning rate for TNet's first/base optimization step.",
    )
    parser.add_argument(
        "--tnet-learning-rate-2step",
        type=float,
        default=1.0e-2,
        help="Adam learning rate for TNet's second/fluctuation optimization step.",
    )
    parser.add_argument("--distributed-validation", action="store_true",
                        help="Partition validation across ranks and pool metrics on CPU.")
    args = parser.parse_args()
    if args.graph_protocol is None:
        args.graph_protocol = "fixed_er_v1" if args.dgp_version in ("a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1") else "resampled"
    if args.graph_protocol == "fixed_er_v1" and args.fixed_graph_file is None:
        args.fixed_graph_file = Path(__file__).resolve().parents[1] / "data" / (
            f"fixed_er_N{args.n_units}_p{args.er_edge_probability}_seed{args.fixed_graph_seed}.json"
        )
    return args


def load_or_prepare_training_data(
    args: argparse.Namespace,
    data_config: DataConfig,
    runtime: DistributedRuntime,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Load prebuilt splits for DDP; prepare them only in single-process mode."""

    from pfn_interference.simulation.csv_tasks import reject_random_csv_for_legacy
    reject_random_csv_for_legacy(data_config, Path(args.data_csv),
                                 allow_overwrite=bool(args.overwrite_data_csv and not runtime.is_distributed))
    if runtime.is_distributed:
        if args.overwrite_data_csv:
            raise RuntimeError(
                "Distributed training cannot use --overwrite-data-csv. "
                "Run --prepare-data-only before torchrun."
            )
        shard_dir = getattr(args, "ddp_shard_dir", None)
        if data_config.dgp_version in ("random_lpe_v1", "a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
            if shard_dir is not None:
                from pfn_interference.simulation.csv_tasks import validate_prepared_random_master as validate_random_bank
            else:
                from pfn_interference.simulation.csv_tasks import validate_prepared_random_csv as validate_random_bank
            # Hash the immutable source bank once before any rank loads tensors.
            # Rank-sharded training needs only the master+manifest; legacy loading
            # still validates the derived train/validation split files as before.
            collective = dist.is_initialized()
            validation_error = [None]
            if runtime.is_main or not collective:
                try:
                    validate_random_bank(
                        data_config, data_csv=Path(args.data_csv), num_datasets=args.csv_datasets,
                        data_seed=args.data_seed, validation_fraction=args.validation_fraction,
                        split_seed=args.split_seed, verify_hashes=(shard_dir is None),
                    )
                except Exception as exc:
                    validation_error[0] = f"{type(exc).__name__}: {exc}"
            if collective:
                dist.broadcast_object_list(validation_error, src=0)
            if validation_error[0] is not None:
                raise ValueError(validation_error[0])
        if shard_dir is not None:
            return load_rank_sharded_episode_csv_splits(
                args.data_csv, shard_dir=Path(shard_dir), runtime=runtime,
                distributed_validation=bool(getattr(args, "distributed_validation", False)),
            )
        return load_existing_episode_csv_splits(args.data_csv)
    return prepare_episode_csv_splits(
        data_config,
        data_csv=args.data_csv,
        num_datasets=args.csv_datasets,
        data_seed=args.data_seed,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        overwrite=args.overwrite_data_csv,
        generation_batch_size=getattr(args, "csv_generation_batch_size", 64),
    )


def run_main(
    args: argparse.Namespace,
    runtime: DistributedRuntime,
    device: torch.device,
) -> None:
    """Execute smoke testing, evaluation, or training for one process."""

    if args.smoke_test:
        if runtime.is_distributed:
            raise RuntimeError("--smoke-test must be launched as a single process.")
        run_smoke_test(device, args.checkpoint_path)
        return
    if args.test_only:
        if runtime.is_distributed:
            raise RuntimeError("--test-only must be launched as a single process.")
        model, _, metadata = load_checkpoint(
            path=args.checkpoint_path,
            device=device,
        )
        data_config = data_config_from_mapping(
            cast(Mapping[str, object], metadata["data_config"])
        )
        shared_test_batch = generate_batch(
            data_config,
            args.test_datasets,
            seed=args.test_seed,
            device=device,
        )
        report = apply_algorithm_2(
            model,
            data_config,
            num_datasets=args.test_datasets,
            seed=args.test_seed,
            device=device,
            batch=shared_test_batch,
        )
        paths = save_algorithm_2_report(report, args.test_output_dir)
        _print_algorithm_2_summary(report, paths)
        _run_paper_baselines(
            model=model,
            data_config=data_config,
            args=args,
            batch=shared_test_batch,
        )
        return

    graph_protocol = getattr(args, "graph_protocol", "resampled")
    fixed_graph_seed = getattr(args, "fixed_graph_seed", 12345)
    fixed_graph_upper_hex = None
    if getattr(args, "fixed_graph_file", None) is not None:
        if graph_protocol != "fixed_er_v1":
            raise ValueError("--fixed-graph-file requires --graph-protocol fixed_er_v1.")
        from pfn_interference.graph import load_or_create_graph
        graph_record = load_or_create_graph(args.fixed_graph_file, n_units=args.n_units,
            edge_probability=args.er_edge_probability, seed=fixed_graph_seed)
        fixed_graph_upper_hex = graph_record["upper_hex"]
    data_config = DataConfig(
        dgp_version=getattr(args, "dgp_version", "gao_ding_design2_outcome"),
        interference_lambda=getattr(args, "interference_lambda", 1.0),
        spillover_gamma_min=getattr(args, "spillover_gamma_min", -3.6),
        spillover_gamma_max=getattr(args, "spillover_gamma_max", 0.0),
        spillover_gamma=getattr(args, "spillover_gamma", None),
        neighbor_beta_min=getattr(args, "neighbor_beta_min", -6.0),
        neighbor_beta_max=getattr(args, "neighbor_beta_max", 0.0),
        neighbor_beta=getattr(args, "neighbor_beta", None),
        random_nointerference_prob=getattr(args, "nointerference_prob", 0.5),
        random_noise_sd=getattr(args, "random_noise_sd", 1.0),
        n_units=args.n_units,
        er_edge_probability=args.er_edge_probability,
        graph_family=args.graph_family,
        graph_protocol=graph_protocol,
        fixed_graph_seed=fixed_graph_seed,
        fixed_graph_upper_hex=fixed_graph_upper_hex,
        treatment_prob=args.treatment_prob,
        x_sd=args.x_sd,
        x_outcome_coefficient=args.x_outcome_coefficient,
        own_score_baseline_coefficient=args.own_score_baseline_coefficient,
        neighbor_score_baseline_coefficient=args.neighbor_score_baseline_coefficient,
        baseline_sd=args.baseline_sd,
        unit_sd=args.unit_sd,
        tau_mean=args.tau_mean,
        tau_episode_sd=args.tau_episode_sd,
        tau_unit_sd=args.tau_unit_sd,
        tau_own_score_slope=args.tau_own_score_slope,
        tau_neighbor_score_slope=args.tau_neighbor_score_slope,
        gamma_mean=args.gamma_mean,
        gamma_sd=args.gamma_sd,
        gamma_own_score_slope=args.gamma_own_score_slope,
        gamma_neighbor_score_slope=args.gamma_neighbor_score_slope,
        eta_mean=args.eta_mean,
        eta_sd=args.eta_sd,
        noise_sd=args.noise_sd,
    )
    if args.prepare_ddp_shards_only:
        if runtime.is_distributed:
            raise RuntimeError(
                "--prepare-ddp-shards-only must be launched as a single process."
            )
        if args.ddp_shard_dir is None:
            raise ValueError("--prepare-ddp-shards-only requires --ddp-shard-dir.")
        if data_config.dgp_version in ("random_lpe_v1", "a_group_fixed_v1", "a_group_random_gamma_v1", "a_group_random_coeff_v1"):
            from pfn_interference.simulation.csv_tasks import ensure_prepared_random_master_split
            ensure_prepared_random_master_split(
                data_config,
                data_csv=Path(args.data_csv),
                num_datasets=args.csv_datasets,
                data_seed=args.data_seed,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                verify_hashes=True,
            )
        metadata = prepare_ddp_train_shards(
            args.data_csv,
            shard_dir=args.ddp_shard_dir,
            world_size=args.ddp_shard_world_size,
        )
        print(f"prepared_ddp_shard_dir={args.ddp_shard_dir}")
        print(f"prepared_ddp_shards={json.dumps(metadata, ensure_ascii=False)}")
        return
    if args.prepare_data_only:
        if runtime.is_distributed:
            raise RuntimeError(
                "--prepare-data-only must be launched as a single process."
            )
        prepared_files = prepare_episode_csv_files(
            data_config,
            data_csv=args.data_csv,
            num_datasets=args.csv_datasets,
            data_seed=args.data_seed,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            overwrite=args.overwrite_data_csv,
            generation_batch_size=args.csv_generation_batch_size,
        )
        print(f"prepared_master_csv={prepared_files['data_csv']}")
        print(f"prepared_training_csv={prepared_files['train_path']}")
        print(f"prepared_validation_csv={prepared_files['validation_path']}")
        print(f"prepared_csv_split={dict(cast(Mapping[str, int], prepared_files['summary']))}")
        return
    model_config = ModelConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        gmm_n_components=args.gmm_n_components,
        gmm_min_sigma=args.gmm_min_sigma,
        gmm_pi_temperature=args.gmm_pi_temperature,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        training_seed=args.training_seed,
    )

    prepared = load_or_prepare_training_data(args, data_config, runtime, device=device)

    train_batch = cast(TensorBatch, prepared["train_batch"])
    validation_batch = cast(TensorBatch, prepared["validation_batch"])
    split_summary = cast(Mapping[str, int], prepared["summary"])

    if runtime.is_main:
        print(f"device={device}")
        print(
            f"distributed={runtime.is_distributed} "
            f"world_size={runtime.world_size}"
        )
        print(f"data_config={asdict(data_config)}")
        print(f"model_config={asdict(model_config)}")
        print(f"train_config={asdict(train_config)}")
        print(
            f"per_device_batch_size={train_config.batch_size} "
            f"global_batch_size={train_config.batch_size * runtime.world_size}"
        )
        print(f"master_csv={prepared['data_csv']}")
        print(f"training_csv={prepared['train_path']}")
        print(f"validation_csv={prepared['validation_path']}")
        print(f"csv_split={dict(split_summary)}")

    train(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        train_batch=train_batch,
        validation_batch=validation_batch,
        device=device,
        checkpoint_path=args.checkpoint_path,
        runtime=runtime,
        rank_sharded_train=bool(prepared.get("rank_sharded_train", False)),
        distributed_validation=bool(prepared.get("validation_is_sharded", False)),
    )

    if not args.skip_test_after_training:
        distributed_barrier(runtime, device)
        if runtime.is_main:
            best_model, _, _ = load_checkpoint(
                path=args.checkpoint_path,
                device=device,
            )
            shared_test_batch = generate_batch(
                data_config,
                args.test_datasets,
                seed=args.test_seed,
                device=device,
            )
            report = apply_algorithm_2(
                best_model,
                data_config,
                num_datasets=args.test_datasets,
                seed=args.test_seed,
                device=device,
                batch=shared_test_batch,
            )
            paths = save_algorithm_2_report(report, args.test_output_dir)
            _print_algorithm_2_summary(report, paths)
            _run_paper_baselines(
                model=best_model,
                data_config=data_config,
                args=args,
                batch=shared_test_batch,
            )
        distributed_barrier(runtime, device)


def main() -> None:
    args = parse_args()
    runtime = None
    try:
        runtime, device = initialize_distributed(args.device)
        run_main(args, runtime, device)
    except BaseException:
        print(f"[rank {os.environ.get('RANK', '0')}] 原始异常（清理通信组之前）：", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            os._exit(1)
        raise
    else:
        cleanup_distributed(runtime)


if __name__ == "__main__":
    main()
