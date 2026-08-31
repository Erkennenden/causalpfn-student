#!/usr/bin/env python3
"""Three-query local-network interference GMM demo.

Each synthetic episode contains 300 units by default on an undirected Erdos-Renyi
graph with edge probability 0.02.  The default structural outcome follows the
Gao--Ding Design 2 equation, adapted only by using this demo's adjacency matrix:

    E_i = sum_j A_ij D_j / sum_j A_ij
    Xbar_Ni = sum_j A_ij X_j / sum_j A_ij
    Y_i = 1 - 0.9 E_i + 6 D_i - X_i
          + 0.2 D_i exp(X_i) - 3 Xbar_Ni + epsilon_i,
    epsilon_i ~ Normal(0, 16).

For every unit, the demo retains exact treated-neighbor exposure and partitions
its exact support into majority-low and majority-high arms. Training queries draw
exact exposure values from those arms, while formal ITE/ATE evaluation integrates
over the complete arm support under the Bernoulli treatment design. All factual node
observations first receive a topology-aware embedding and then pass through a
standard global self-attention Transformer. A continuous one-dimensional
Gaussian mixture model (GMM) is predicted for every node-level ITE query.

Training data are read from the 80% episode-level CSV split. Every epoch visits
each training episode exactly once in shuffled mini-batches, then evaluates the
complete fixed 20% validation split. Adam updates model parameters and
ReduceLROnPlateau monitors only the average validation GMM negative
log-likelihood. Final testing uses seed 2026 independently of both CSV splits.
For real-data application there is no extra held-out test dataset; validation
and test graphs are simulation-only model-selection/evaluation devices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from estimands import oracle_ate, oracle_ite_from_parameters, sample_majority_exposures



TensorBatch = Dict[str, torch.Tensor]
ER_MAX_RESAMPLE_ATTEMPTS = 1000
QUERY_DIRECT = 0
QUERY_SPILLOVER = 1
QUERY_TOTAL = 2
QUERY_TYPE_NAMES = ("direct", "spillover", "total")
EFFECT_PREFIXES = (
    "direct_effect",
    "spillover_effect",
    "total_effect",
)

N_UNITS = 300
TOKEN_DIM = 5
QUERY_DIM = 4
CSV_SCHEMA_VERSION = "majority_arm_unified_ite_ate_v10_er_only_topology_embedding"
CHECKPOINT_FORMAT_VERSION = 2


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
    Legacy TNet and v7 DGP modes remain available for newly configured runs.
    """

    n_units: int = N_UNITS
    er_edge_probability: float = 0.02
    treatment_prob: float = 0.5
    dgp_version: str = "gao_ding_design2_outcome"
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

    def __post_init__(self) -> None:
        if self.n_units < 2:
            raise ValueError("n_units must be at least 2.")
        if self.dgp_version not in {
            "gao_ding_design2_outcome",
            "tnet_linear_neighbor_covariate",
            "legacy_v7",
        }:
            raise ValueError("Unsupported dgp_version.")
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


def data_config_from_mapping(values: Mapping[str, object]) -> DataConfig:
    """Load ER-only data-generation metadata.

    Metadata without an explicit DGP version uses the legacy-v7 coefficients
    rather than silently inheriting newer defaults.
    """

    data = dict(values)
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
    """Permutation-equivariant graph-aware encoder plus effect-specific GMM heads."""

    input_dim: int = TOKEN_DIM
    query_dim: int = QUERY_DIM
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    gmm_n_components: int = 5
    gmm_min_sigma: float = 1e-3
    gmm_pi_temperature: float = 1.0
    direct_loss_weight: float = 1.0
    spillover_loss_weight: float = 1.0
    total_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim != TOKEN_DIM:
            raise ValueError("Each node token has four factual features plus normalized degree.")
        if self.query_dim != QUERY_DIM:
            raise ValueError("Each causal query has four contrast coordinates.")
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
        weights = (
            self.direct_loss_weight,
            self.spillover_loss_weight,
            self.total_loss_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("effect loss weights must be nonnegative.")
        if sum(weights) <= 0.0:
            raise ValueError("At least one effect loss weight must be positive.")


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


def effect_loss_weights_from_config(config: ModelConfig) -> tuple[float, float, float]:
    """Return direct, spillover, and total loss weights in query-index order."""

    return (
        float(config.direct_loss_weight),
        float(config.spillover_loss_weight),
        float(config.total_loss_weight),
    )


def effect_loss_weights_from_model(model: nn.Module) -> tuple[float, float, float]:
    """Read effect loss weights from a raw or DDP-wrapped model."""

    config = unwrap_model(model).config
    return effect_loss_weights_from_config(config)


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
    """Convert raw [...,3,3K] effect-head outputs into valid GMM parameters."""

    if raw_predictions.ndim < 4 or raw_predictions.shape[-2] != 3:
        raise ValueError(
            "raw_predictions must have shape [batch, units, ..., 3, 3K]."
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

    dist.init_process_group(**process_group_init_kwargs(runtime, device))
    return runtime, device


def distributed_barrier(
    runtime: DistributedRuntime,
    device: torch.device,
) -> None:
    """Synchronize ranks while explicitly identifying each accelerator."""

    if not runtime.is_distributed:
        return
    if device.type in {"musa", "privateuseone"}:
        device_index = device.index
        if device_index is None:
            device_index = runtime.rank
        dist.barrier(device_ids=[device_index])
    else:
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
    edge_probability: float,
    generator: torch.Generator,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate loop-free undirected ER graphs with no isolated nodes."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
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
            < edge_probability
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


def generate_graphs(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate loop-free undirected ER graphs and legacy center metadata."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    adjacency = _generate_er_adjacency(
        config.n_units,
        batch_size,
        edge_probability=config.er_edge_probability,
        generator=generator,
    )
    star_center = torch.full((batch_size,), -1, dtype=torch.long)
    return adjacency, star_center

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
) -> TensorBatch:
    """Generate factual data and support-valid general exposure queries."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    generator = _make_generator(seed)
    adjacency, star_center = generate_graphs(
        config, batch_size, generator=generator
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

    if config.dgp_version == "gao_ding_design2_outcome":
        neighbor_x = torch.bmm(adjacency, x.unsqueeze(-1)).squeeze(-1) / degree
        epsilon = 4.0 * torch.randn(
            batch_size, config.n_units, generator=generator
        )
        baseline = 1.0 - x - 3.0 * neighbor_x + epsilon
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
    y_obs = factual_mean + config.noise_sd * torch.randn(
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
        "star_center", "x", "observed_treatment",
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


def load_episode_batch_csv(
    path: Path,
    *,
    device: torch.device,
) -> tuple[TensorBatch, list[int]]:
    """Load a compact unit-row CSV and reconstruct dynamic episode tensors."""

    if not path.is_file():
        raise FileNotFoundError(f"CSV dataset does not exist: {path}")

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
    tokens = torch.empty(batch_size, n_units, TOKEN_DIM, dtype=torch.float32)
    queries = torch.empty(batch_size, n_units, 3, QUERY_DIM, dtype=torch.float32)
    adjacency = torch.empty(batch_size, n_units, n_units, dtype=torch.uint8)
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
            dataset_id = int(row["dataset_id"])
            dataset_index = dataset_index_by_id[dataset_id]
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
            current_center = int(row["star_center"])
            if unit_index == 0:
                star_center[dataset_index] = current_center
            elif int(star_center[dataset_index].item()) != current_center:
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

    if not bool(seen.all()):
        missing = torch.nonzero(~seen, as_tuple=False)[0].tolist()
        missing_dataset = dataset_ids[missing[0]]
        raise ValueError(
            f"dataset_id={missing_dataset} must contain unit_id 0 through "
            f"{n_units - 1} exactly once."
        )

    covariate_score = torch.sigmoid(x)
    neighbor_covariate_score = torch.bmm(
        adjacency.to(torch.float32), covariate_score.unsqueeze(-1)
    ).squeeze(-1) / degree

    batch: TensorBatch = {
        "tokens": tokens,
        "queries": queries,
        "adjacency": adjacency,
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


def split_episode_csv(
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
) -> Dict[str, int]:
    """Split complete ER episodes into train and validation CSV files."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    dataset_ids_seen: set[int] = set()
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("source CSV must contain a header.")
        for row in reader:
            dataset_ids_seen.add(int(row["dataset_id"]))
    dataset_ids = sorted(dataset_ids_seen)
    total = len(dataset_ids)
    if total < 2:
        raise ValueError("At least two episodes are required for a train/validation split.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(split_seed)
    validation_count = int(round(total * validation_fraction))
    validation_count = min(max(validation_count, 1), total - 1)
    permutation = torch.randperm(total, generator=generator).tolist()
    validation_ids = {dataset_ids[index] for index in permutation[:validation_count]}
    training_ids = set(dataset_ids) - validation_ids

    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        source_path.open("r", encoding="utf-8", newline="") as source_handle,
        train_path.open("w", encoding="utf-8", newline="") as train_handle,
        validation_path.open("w", encoding="utf-8", newline="") as validation_handle,
    ):
        reader = csv.DictReader(source_handle)
        train_writer = csv.DictWriter(train_handle, fieldnames=fieldnames)
        validation_writer = csv.DictWriter(validation_handle, fieldnames=fieldnames)
        train_writer.writeheader()
        validation_writer.writeheader()
        for row in reader:
            dataset_id = int(row["dataset_id"])
            writer = validation_writer if dataset_id in validation_ids else train_writer
            writer.writerow(row)

    return {
        "total_datasets": total,
        "training_datasets": len(training_ids),
        "validation_datasets": len(validation_ids),
    }



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
    """Pre-norm Transformer layer with standard global self-attention."""

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
    ) -> torch.Tensor:
        if src.ndim != 3:
            raise ValueError("src must have shape [batch, nodes, d_model].")

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
    """Encode all observed nodes and predict every node-level ITE query."""

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
        self.effect_heads = nn.ModuleDict(
            {
                "direct": make_gmm_head(config),
                "spillover": make_gmm_head(config),
                "total": make_gmm_head(config),
            }
        )

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
            encoded = layer(encoded)
        return encoded

    def predict_from_context(
        self,
        encoded: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict one or a grid of three-effect queries from cached graph context.

        ``queries`` may be [B,N,3,Q] or [B,N,P,3,Q]. Any dimensions between
        the unit and effect axes are treated as independent query-grid axes.
        """

        if encoded.ndim != 3 or encoded.shape[-1] != self.config.d_model:
            raise ValueError("encoded must have shape [batch, units, d_model].")
        if queries.ndim < 4 or queries.shape[-2:] != (3, self.config.query_dim):
            raise ValueError(
                "queries must have shape [batch, units, ..., 3, query_dim]."
            )
        if queries.shape[:2] != encoded.shape[:2]:
            raise ValueError(
                "encoded and queries must have the same batch/unit dimensions."
            )
        query_embedding = self.query_encoder(queries)
        context = encoded
        for _ in range(queries.ndim - 4):
            context = context.unsqueeze(2)
        context = context.unsqueeze(-2)
        fused = self.query_fusion(context + query_embedding)
        direct_raw = self.effect_heads["direct"](fused[..., QUERY_DIRECT, :])
        spillover_raw = self.effect_heads["spillover"](
            fused[..., QUERY_SPILLOVER, :]
        )
        total_raw = self.effect_heads["total"](fused[..., QUERY_TOTAL, :])
        raw_predictions = torch.stack(
            [direct_raw, spillover_raw, total_raw],
            dim=-2,
        )
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
            3,
            self.config.query_dim,
        ):
            raise ValueError(
                "queries must have shape [batch, units, 3, query_dim]."
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


def compute_training_losses(
    model: LocalNetworkQueryTransformer,
    predictions: Mapping[str, torch.Tensor],
    batch: TensorBatch,
) -> Dict[str, torch.Tensor]:
    """Effect-only GMM NLL, normalized-weighted across three query types."""

    pi = predictions["gmm_pi"]
    mu = predictions["gmm_mu"]
    sigma = predictions["gmm_sigma"]
    target = query_effect_targets(batch)
    losses: Dict[str, torch.Tensor] = {}
    type_losses = []
    for query_index, prefix in enumerate(EFFECT_PREFIXES):
        query_name = QUERY_TYPE_NAMES[query_index]
        value = gmm_nll_loss(
            pi[..., query_index, :],
            mu[..., query_index, :],
            sigma[..., query_index, :],
            target[..., query_index],
        )
        losses[f"{prefix}_gmm_nll"] = value
        losses[f"{query_name}_gmm_nll"] = value
        type_losses.append(value)
    macro = torch.stack(type_losses).mean()
    weights = torch.tensor(
        effect_loss_weights_from_model(model),
        dtype=macro.dtype,
        device=macro.device,
    )
    weight_sum = weights.sum()
    weighted_terms = torch.stack(type_losses) * weights / weight_sum
    weighted = weighted_terms.sum()
    losses["effect_gmm_nll_macro"] = macro
    losses["weighted_effect_gmm_nll"] = weighted
    for query_index, query_name in enumerate(QUERY_TYPE_NAMES):
        losses[f"{query_name}_loss_weight"] = weights[query_index]
        losses[f"weighted_{query_name}_contribution"] = weighted_terms[query_index]
    losses["total_loss"] = weighted
    return losses


def compute_query_metrics(
    model: LocalNetworkQueryTransformer,
    predictions: Mapping[str, torch.Tensor],
    batch: TensorBatch,
    *,
    interval_mass: float = 0.9,
) -> Dict[str, float]:
    """Compute GMM NLL and point/interval diagnostics for all three effects."""

    pi = predictions["gmm_pi"]
    mu = predictions["gmm_mu"]
    sigma = predictions["gmm_sigma"]
    target = query_effect_targets(batch)
    effect_pred = gmm_posterior_mean(pi, mu)
    lower, upper = gmm_posterior_interval(
        pi,
        mu,
        sigma,
        mass=interval_mass,
    )
    log_density = gmm_log_prob(pi, mu, sigma, target)
    metrics: Dict[str, float] = {}
    nll_values = []
    rmse_values = []
    coverage_values = []
    width_values = []
    pct = int(round(interval_mass * 100.0))

    for query_index, prefix in enumerate(EFFECT_PREFIXES):
        query_name = QUERY_TYPE_NAMES[query_index]
        true_effect = target[..., query_index]
        predicted_effect = effect_pred[..., query_index]
        regression = compute_regression_metrics(predicted_effect, true_effect)
        metrics.update(_prefix_metrics(prefix, regression))
        for metric_name in ("mae", "mse", "rmse", "bias", "nmae_pct", "nrmse_pct", "mape_pct"):
            metrics[f"{query_name}_{metric_name}"] = regression[metric_name]
        graph_true = true_effect.mean(dim=1)
        graph_pred = predicted_effect.mean(dim=1)
        graph_error = graph_pred - graph_true
        nll = -log_density[..., query_index].mean()
        coverage = (
            (true_effect >= lower[..., query_index])
            & (true_effect <= upper[..., query_index])
        ).float().mean()
        width = (
            upper[..., query_index] - lower[..., query_index]
        ).mean()

        metrics[f"{prefix}_gmm_nll"] = float(nll.item())
        metrics[f"{prefix}_coverage_{pct}"] = float(coverage.item())
        metrics[f"{prefix}_interval_width_{pct}"] = float(width.item())
        metrics[f"{query_name}_gmm_nll"] = metrics[f"{prefix}_gmm_nll"]
        metrics[f"{query_name}_coverage_{pct}"] = metrics[
            f"{prefix}_coverage_{pct}"
        ]
        metrics[f"{query_name}_interval_width_{pct}"] = metrics[
            f"{prefix}_interval_width_{pct}"
        ]
        metrics[f"{prefix}_log_density_mean"] = float(
            log_density[..., query_index].mean().item()
        )
        metrics[f"average_{prefix}_true"] = float(graph_true.mean().item())
        metrics[f"average_{prefix}_pred"] = float(graph_pred.mean().item())
        metrics[f"average_{prefix}_absolute_error"] = float(
            graph_error.abs().mean().item()
        )
        nll_values.append(metrics[f"{prefix}_gmm_nll"])
        rmse_values.append(metrics[f"{prefix}_rmse"])
        coverage_values.append(metrics[f"{prefix}_coverage_{pct}"])
        width_values.append(metrics[f"{prefix}_interval_width_{pct}"])

    weights = effect_loss_weights_from_model(model)
    weight_sum = sum(weights)
    weighted_contributions = [
        weights[index] * nll_values[index] / weight_sum
        for index in range(len(QUERY_TYPE_NAMES))
    ]
    weighted_nll = sum(weighted_contributions)
    for query_index, query_name in enumerate(QUERY_TYPE_NAMES):
        metrics[f"{query_name}_loss_weight"] = weights[query_index]
        metrics[f"weighted_{query_name}_contribution"] = weighted_contributions[
            query_index
        ]
    metrics["effect_gmm_nll_macro"] = sum(nll_values) / 3.0
    metrics["weighted_effect_gmm_nll"] = weighted_nll
    metrics["validation_gmm_nll"] = weighted_nll
    metrics["effect_rmse_macro"] = sum(rmse_values) / 3.0
    metrics[f"effect_coverage_{pct}_macro"] = sum(coverage_values) / 3.0
    metrics[f"effect_interval_width_{pct}_macro"] = sum(width_values) / 3.0
    return metrics


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
    """Update model parameters once using effect-only GMM NLL."""

    metrics = train_step_metrics(model, optimizer, batch)
    return metrics["total_loss"]


def train_step_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
) -> Dict[str, float]:
    """Update once and return pre-update batch diagnostics for logging."""

    model.train()
    predictions = model(batch["tokens"], batch["queries"], batch["adjacency"])
    losses = compute_training_losses(model, predictions, batch)
    metrics = compute_query_metrics(
        model,
        {key: value.detach() for key, value in predictions.items()},
        batch,
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
) -> Union[float, Dict[str, float]]:
    """Train one epoch, sharding complete episodes equally across DDP ranks."""

    runtime = runtime or DistributedRuntime()
    episode_count = int(train_batch["tokens"].shape[0])
    if runtime.is_distributed:
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
    metric_totals: Dict[str, float] = {}
    total_weight = 0
    for start in range(0, int(order.numel()), batch_size):
        indices = order[start : start + batch_size]
        batch = select_episode_batch(train_batch, indices, device=device)
        batch_metrics = train_step_metrics(model, optimizer, batch)
        target = query_effect_targets(batch)
        batch_weight = int(target.shape[0] * target.shape[1])
        for name, value in batch_metrics.items():
            metric_totals[name] = (
                metric_totals.get(name, 0.0) + value * batch_weight
            )
        total_weight += batch_weight
    if runtime.is_distributed:
        metric_names = sorted(metric_totals)
        aggregate = torch.tensor(
            [metric_totals[name] for name in metric_names] + [float(total_weight)],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        total_weight = int(round(float(aggregate[-1].item())))
        metric_totals = {
            name: float(aggregate[index].item())
            for index, name in enumerate(metric_names)
        }
    averaged_metrics = {
        name: value / max(total_weight, 1)
        for name, value in metric_totals.items()
    }
    if return_metrics:
        return averaged_metrics
    return float(
        averaged_metrics.get(
            "weighted_effect_gmm_nll",
            averaged_metrics.get("total_loss", float("nan")),
        )
    )


def evaluate(
    model: LocalNetworkQueryTransformer,
    validation_batch: TensorBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate every validation episode and average GMM NLL over all targets."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    was_training = model.training
    model.eval()
    prediction_parts: Dict[str, list[torch.Tensor]] = {
        "gmm_pi": [],
        "gmm_mu": [],
        "gmm_sigma": [],
    }
    target_parts = []
    episode_count = int(validation_batch["tokens"].shape[0])
    try:
        with torch.no_grad():
            for start in range(0, episode_count, batch_size):
                indices = torch.arange(
                    start,
                    min(start + batch_size, episode_count),
                    dtype=torch.long,
                )
                batch = select_episode_batch(
                    validation_batch,
                    indices,
                    device=device,
                )
                predictions = model(batch["tokens"], batch["queries"], batch["adjacency"])
                for key in prediction_parts:
                    prediction_parts[key].append(predictions[key].detach().cpu())
                target_parts.append(query_effect_targets(batch).detach().cpu())

        combined_predictions = {
            key: torch.cat(parts, dim=0)
            for key, parts in prediction_parts.items()
        }
        combined_batch: TensorBatch = {
            "query_effect": torch.cat(target_parts, dim=0)
        }
        metrics = compute_query_metrics(
            model,
            combined_predictions,
            combined_batch,
        )
        # This exact key is the weighted effect GMM NLL passed to scheduler.step().
        return metrics
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
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
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
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Checkpoint is incompatible with the ER-only standard-attention "
            "model; retrain this version before using --test-only."
        )
    data_config = data_config_from_mapping(checkpoint["data_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    model = LocalNetworkQueryTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = create_optimizer(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    metadata: Dict[str, object] = {
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
) -> LocalNetworkQueryTransformer:
    """Train by complete CSV epochs with optional torchrun DDP."""

    runtime = runtime or DistributedRuntime()

    if train_batch["tokens"].shape[0] <= 0:
        raise ValueError("training split must contain at least one episode.")
    if validation_batch["tokens"].shape[0] <= 0:
        raise ValueError("validation split must contain at least one episode.")

    if runtime.is_main:
        print(
            "training_label_diagnostic="
            + json.dumps(
                diagnose_batch_effect_support(train_batch),
                ensure_ascii=False,
            )
        )
        print(
            "validation_label_diagnostic="
            + json.dumps(
                diagnose_batch_effect_support(validation_batch),
                ensure_ascii=False,
            )
        )

    seed_everything(train_config.training_seed)
    raw_model = LocalNetworkQueryTransformer(model_config).to(device)
    if runtime.is_distributed:
        device_index = device.index
        ddp_device_ids = None if device.type == "cpu" else [device_index]
        model: nn.Module = DistributedDataParallel(
            raw_model,
            device_ids=ddp_device_ids,
            output_device=(device_index if ddp_device_ids else None),
            broadcast_buffers=False,
        )
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
        )
        if isinstance(train_result, Mapping):
            train_metrics = {
                str(name): float(value) for name, value in train_result.items()
            }
        else:
            train_metrics = {
                "weighted_effect_gmm_nll": float(train_result),
                "total_loss": float(train_result),
            }
        metric_names = (
            "validation_gmm_nll",
            "effect_rmse_macro",
            "direct_effect_rmse",
            "spillover_effect_rmse",
            "total_effect_rmse",
        )
        if runtime.is_main:
            metrics = evaluate(
                raw_model,
                validation_batch,
                batch_size=train_config.batch_size,
                device=device,
            )
        else:
            metrics = {name: 0.0 for name in metric_names}
        if runtime.is_distributed:
            metric_tensor = torch.tensor(
                [metrics[name] for name in metric_names],
                dtype=torch.float32,
                device=device,
            )
            dist.broadcast(metric_tensor, src=0)
            metrics = {
                name: float(metric_tensor[index].item())
                for index, name in enumerate(metric_names)
            }
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
            def metric_value(source: Mapping[str, float], name: str) -> float:
                return float(source.get(name, float("nan")))

            print(
                f"epoch={epoch:04d}/{train_config.epochs:04d} "
                f"train_weighted_effect_gmm_nll="
                f"{metric_value(train_metrics, 'weighted_effect_gmm_nll'):.8f} "
                f"train_direct_gmm_nll="
                f"{metric_value(train_metrics, 'direct_effect_gmm_nll'):.8f} "
                f"train_direct_mae={metric_value(train_metrics, 'direct_effect_mae'):.8f} "
                f"train_direct_rmse={metric_value(train_metrics, 'direct_effect_rmse'):.8f} "
                f"train_direct_bias={metric_value(train_metrics, 'direct_effect_bias'):.8f} "
                f"train_direct_coverage_90="
                f"{metric_value(train_metrics, 'direct_effect_coverage_90'):.4f} "
                f"train_direct_interval_width_90="
                f"{metric_value(train_metrics, 'direct_effect_interval_width_90'):.8f} "
                f"train_spillover_gmm_nll="
                f"{metric_value(train_metrics, 'spillover_effect_gmm_nll'):.8f} "
                f"train_spillover_mae="
                f"{metric_value(train_metrics, 'spillover_effect_mae'):.8f} "
                f"train_spillover_rmse="
                f"{metric_value(train_metrics, 'spillover_effect_rmse'):.8f} "
                f"train_spillover_bias="
                f"{metric_value(train_metrics, 'spillover_effect_bias'):.8f} "
                f"train_spillover_coverage_90="
                f"{metric_value(train_metrics, 'spillover_effect_coverage_90'):.4f} "
                f"train_spillover_interval_width_90="
                f"{metric_value(train_metrics, 'spillover_effect_interval_width_90'):.8f} "
                f"train_total_gmm_nll="
                f"{metric_value(train_metrics, 'total_effect_gmm_nll'):.8f} "
                f"train_total_mae={metric_value(train_metrics, 'total_effect_mae'):.8f} "
                f"train_total_rmse={metric_value(train_metrics, 'total_effect_rmse'):.8f} "
                f"train_total_bias={metric_value(train_metrics, 'total_effect_bias'):.8f} "
                f"train_total_coverage_90="
                f"{metric_value(train_metrics, 'total_effect_coverage_90'):.4f} "
                f"train_total_interval_width_90="
                f"{metric_value(train_metrics, 'total_effect_interval_width_90'):.8f} "
                f"train_weighted_direct_contribution="
                f"{metric_value(train_metrics, 'weighted_direct_contribution'):.8f} "
                f"train_weighted_spillover_contribution="
                f"{metric_value(train_metrics, 'weighted_spillover_contribution'):.8f} "
                f"train_weighted_total_contribution="
                f"{metric_value(train_metrics, 'weighted_total_contribution'):.8f} "
                f"validation_gmm_nll={validation_gmm_nll:.8f} "
                f"validation_direct_gmm_nll="
                f"{metric_value(metrics, 'direct_effect_gmm_nll'):.8f} "
                f"validation_direct_mae={metric_value(metrics, 'direct_effect_mae'):.8f} "
                f"validation_direct_rmse={metric_value(metrics, 'direct_effect_rmse'):.8f} "
                f"validation_direct_bias={metric_value(metrics, 'direct_effect_bias'):.8f} "
                f"validation_direct_coverage_90="
                f"{metric_value(metrics, 'direct_effect_coverage_90'):.4f} "
                f"validation_direct_interval_width_90="
                f"{metric_value(metrics, 'direct_effect_interval_width_90'):.8f} "
                f"validation_spillover_gmm_nll="
                f"{metric_value(metrics, 'spillover_effect_gmm_nll'):.8f} "
                f"validation_spillover_mae="
                f"{metric_value(metrics, 'spillover_effect_mae'):.8f} "
                f"validation_spillover_rmse="
                f"{metric_value(metrics, 'spillover_effect_rmse'):.8f} "
                f"validation_spillover_bias="
                f"{metric_value(metrics, 'spillover_effect_bias'):.8f} "
                f"validation_spillover_coverage_90="
                f"{metric_value(metrics, 'spillover_effect_coverage_90'):.4f} "
                f"validation_spillover_interval_width_90="
                f"{metric_value(metrics, 'spillover_effect_interval_width_90'):.8f} "
                f"validation_total_gmm_nll="
                f"{metric_value(metrics, 'total_effect_gmm_nll'):.8f} "
                f"validation_total_mae={metric_value(metrics, 'total_effect_mae'):.8f} "
                f"validation_total_rmse={metric_value(metrics, 'total_effect_rmse'):.8f} "
                f"validation_total_bias={metric_value(metrics, 'total_effect_bias'):.8f} "
                f"validation_total_coverage_90="
                f"{metric_value(metrics, 'total_effect_coverage_90'):.4f} "
                f"validation_total_interval_width_90="
                f"{metric_value(metrics, 'total_effect_interval_width_90'):.8f} "
                f"direct_loss_weight={metric_value(metrics, 'direct_loss_weight'):.8f} "
                f"spillover_loss_weight="
                f"{metric_value(metrics, 'spillover_loss_weight'):.8f} "
                f"total_loss_weight={metric_value(metrics, 'total_loss_weight'):.8f} "
                f"validation_weighted_direct_contribution="
                f"{metric_value(metrics, 'weighted_direct_contribution'):.8f} "
                f"validation_weighted_spillover_contribution="
                f"{metric_value(metrics, 'weighted_spillover_contribution'):.8f} "
                f"validation_weighted_total_contribution="
                f"{metric_value(metrics, 'weighted_total_contribution'):.8f} "
                f"point_rmse_macro={metrics['effect_rmse_macro']:.8f} "
                f"direct_rmse={metrics['direct_effect_rmse']:.8f} "
                f"spillover_rmse={metrics['spillover_effect_rmse']:.8f} "
                f"total_rmse={metrics['total_effect_rmse']:.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.8g} "
                f"best_nll={int(nll_improved)} "
                f"best_point={int(point_improved)}"
            )

    distributed_barrier(runtime, device)
    if runtime.is_main:
        print(f"best_nll_checkpoint={paths['best_nll']}")
        print(f"best_point_checkpoint={paths['best_point']}")
        print(f"latest_checkpoint={paths['latest']}")
    return raw_model


def apply_algorithm_2(
    model: LocalNetworkQueryTransformer,
    data_config: DataConfig,
    *,
    num_datasets: int = 10,
    seed: int = 2026,
    device: torch.device,
    batch: Optional[TensorBatch] = None,
) -> Dict[str, object]:
    """Apply direct, spillover, and total GMM queries to each unit."""

    if num_datasets <= 0:
        raise ValueError("num_datasets must be positive.")
    if batch is None:
        batch = generate_batch(
            data_config,
            num_datasets,
            seed=seed,
            device=device,
        )
    elif int(batch["tokens"].shape[0]) != num_datasets:
        raise ValueError("shared test batch size must equal num_datasets.")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            predictions = model(batch["tokens"], batch["queries"], batch["adjacency"])
    finally:
        model.train(was_training)

    pi = predictions["gmm_pi"]
    mu = predictions["gmm_mu"]
    sigma = predictions["gmm_sigma"]
    effect_pred = gmm_posterior_mean(pi, mu)
    lower, upper = gmm_posterior_interval(pi, mu, sigma, mass=0.9)
    target = query_effect_targets(batch)
    n_units = int(batch["tokens"].shape[1])
    query_coordinates = batch["queries"]
    log_density = gmm_log_prob(pi, mu, sigma, target)
    overall = compute_query_metrics(model, predictions, {"query_effect": target})

    dataset_rows = []
    unit_rows = []
    for dataset_index in range(num_datasets):
        dataset_predictions = {
            key: value[dataset_index : dataset_index + 1]
            for key, value in predictions.items()
        }
        dataset_batch: TensorBatch = {
            "query_effect": target[dataset_index : dataset_index + 1]
        }
        dataset_metrics = compute_query_metrics(
            model,
            dataset_predictions,
            dataset_batch,
        )
        dataset_rows.append(
            {
                "dataset_id": dataset_index + 1,
                "star_center": int(
                    batch["star_center"][dataset_index].item()
                ),
                "treated_count": int(
                    batch["observed_treatment"][dataset_index].sum().item()
                ),
                **dataset_metrics,
            }
        )

        for unit_index in range(n_units):
            for query_index, query_name in enumerate(QUERY_TYPE_NAMES):
                query = query_coordinates[
                    dataset_index, unit_index, query_index
                ]
                true_effect = float(
                    target[
                        dataset_index,
                        unit_index,
                        query_index,
                    ].item()
                )
                predicted_effect = float(
                    effect_pred[
                        dataset_index,
                        unit_index,
                        query_index,
                    ].item()
                )
                unit_rows.append(
                    {
                        "dataset_id": dataset_index + 1,
                        "unit_id": unit_index + 1,
                        "query_type": query_name,
                        "t_a": float(query[0].item()),
                        "e_a": float(query[1].item()),
                        "t_b": float(query[2].item()),
                        "e_b": float(query[3].item()),
                        "x": float(
                            batch["x"][dataset_index, unit_index].item()
                        ),
                        "tau": float(
                            batch["tau"][dataset_index, unit_index].item()
                        ),
                        "gamma": float(
                            batch["gamma"][dataset_index, unit_index].item()
                        ),
                        "eta": float(
                            batch["eta"][dataset_index, unit_index].item()
                        ),
                        "observed_treatment": int(
                            batch["observed_treatment"][
                                dataset_index,
                                unit_index,
                            ].item()
                        ),
                        "observed_exposure": float(
                            batch["observed_exposure"][
                                dataset_index,
                                unit_index,
                            ].item()
                        ),
                        "effect_true": true_effect,
                        "effect_pred": predicted_effect,
                        "effect_absolute_error": abs(
                            predicted_effect - true_effect
                        ),
                        "effect_lower_90": float(
                            lower[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].item()
                        ),
                        "effect_upper_90": float(
                            upper[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].item()
                        ),
                        "effect_log_density": float(
                            log_density[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].item()
                        ),
                        "gmm_weights": json.dumps(
                            pi[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].detach().cpu().tolist()
                        ),
                        "gmm_means": json.dumps(
                            mu[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].detach().cpu().tolist()
                        ),
                        "gmm_sigmas": json.dumps(
                            sigma[
                                dataset_index,
                                unit_index,
                                query_index,
                            ].detach().cpu().tolist()
                        ),
                    }
                )

    return {
        "algorithm": "Algorithm 2: direct, spillover, and total effect GMM queries",
        "test_seed": seed,
        "num_datasets": num_datasets,
        "num_units_per_dataset": n_units,
        "num_observed_units_per_dataset": n_units,
        "num_queried_units_per_dataset": n_units,
        "num_evaluated_units_per_dataset": n_units,
        "num_query_types": 3,
        "gmm_n_components": model.config.gmm_n_components,
        "overall": overall,
        "datasets": dataset_rows,
        "units": unit_rows,
    }


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
        expected = model(batch["tokens"], batch["queries"], batch["adjacency"])
        actual = restored(batch["tokens"], batch["queries"], batch["adjacency"])
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
    print(f"queries_shape={tuple(batch['queries'].shape)}")
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
        print(
            f"{prefix}_rmse={overall[f'{prefix}_rmse']:.8f} "
            f"{prefix}_gmm_nll={overall[f'{prefix}_gmm_nll']:.8f} "
            f"{prefix}_coverage_90={overall[f'{prefix}_coverage_90']:.4f}"
        )
    print(
        f"effect_gmm_nll_macro={overall['effect_gmm_nll_macro']:.8f} "
        f"weighted_effect_gmm_nll={overall['weighted_effect_gmm_nll']:.8f} "
        f"weighted_direct_contribution={overall['weighted_direct_contribution']:.8f} "
        f"weighted_spillover_contribution={overall['weighted_spillover_contribution']:.8f} "
        f"weighted_total_contribution={overall['weighted_total_contribution']:.8f} "
        f"effect_rmse_macro={overall['effect_rmse_macro']:.8f}"
    )
    for name, path in paths.items():
        print(f"{name}={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-query local-network interference GMM demo."
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
            "units receive causal queries; default: 300."
        ),
    )
    parser.add_argument(
        "--er-edge-probability",
        type=float,
        default=0.02,
        help="ER edge probability; default 0.02 gives expected degree about 6 at N=300.",
    )
    parser.add_argument("--treatment-prob", type=float, default=0.5)
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
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gmm-n-components", type=int, default=5)
    parser.add_argument("--gmm-min-sigma", type=float, default=1e-3)
    parser.add_argument("--gmm-pi-temperature", type=float, default=1.0)
    parser.add_argument(
        "--direct-loss-weight",
        type=float,
        default=1.0,
        help="Nonnegative training weight for the direct-effect GMM NLL.",
    )
    parser.add_argument(
        "--spillover-loss-weight",
        type=float,
        default=1.0,
        help="Nonnegative training weight for the spillover-effect GMM NLL.",
    )
    parser.add_argument(
        "--total-loss-weight",
        type=float,
        default=1.0,
        help="Nonnegative training weight for the total-effect GMM NLL.",
    )

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

    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-test-after-training", action="store_true")
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument("--test-datasets", type=int, default=10)
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=Path("algorithm2_gmm_test_results"),
    )
    return parser.parse_args()


def load_or_prepare_training_data(
    args: argparse.Namespace,
    data_config: DataConfig,
    runtime: DistributedRuntime,
) -> Dict[str, object]:
    """Load prebuilt splits for DDP; prepare them only in single-process mode."""

    if runtime.is_distributed:
        if args.overwrite_data_csv:
            raise RuntimeError(
                "Distributed training cannot use --overwrite-data-csv. "
                "Run --prepare-data-only before torchrun."
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
        return

    data_config = DataConfig(
        n_units=args.n_units,
        er_edge_probability=args.er_edge_probability,
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
        direct_loss_weight=args.direct_loss_weight,
        spillover_loss_weight=args.spillover_loss_weight,
        total_loss_weight=args.total_loss_weight,
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

    prepared = load_or_prepare_training_data(args, data_config, runtime)

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
        distributed_barrier(runtime, device)


def main() -> None:
    args = parse_args()
    runtime, device = initialize_distributed(args.device)
    try:
        run_main(args, runtime, device)
    finally:
        cleanup_distributed(runtime)


if __name__ == "__main__":
    main()
