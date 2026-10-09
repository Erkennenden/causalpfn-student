"""Chen et al. (ICML 2024) TNet adapted to the majority-arm benchmark.

The implementation preserves the paper's four structural components:
(1) one-hop GCN feature aggregation, (2) generalized propensity heads for
own treatment and neighborhood exposure, (3) treatment-specific outcome
heads, and (4) the paper/code-aligned truncated-power perturbation used in the targeted loss.

The paper's double-robustness guarantee is for average dose-response / average
causal-effect targets.  Node-level effects produced here are useful predictions
but are not claimed to inherit an individual-effect DR guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pfn_interference.estimands import conditional_count_weights


TNET_METHOD_NAME = "Tnet"


@dataclass(frozen=True)
class TNetConfig:
    hidden_dim: int = 64
    grid_size: int = 20
    spline_basis: int = 12
    spline_degree: int = 2
    epochs: int = 160
    base_steps: int = 1
    targeted_steps: int = 50
    learning_rate_1step: float = 1e-4
    learning_rate_2step: float = 1e-2
    alpha: float = 0.5
    gamma: float = 1.0
    beta: float | None = None  # Author default: 20 / sqrt(fitted sample size).
    propensity_floor: float = 1e-3
    weight_decay: float = 1e-3
    gcn_dropout: float = 0.05
    normalize_outcomes: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if self.spline_degree < 1:
            raise ValueError("spline_degree must be positive.")
        if self.spline_basis <= self.spline_degree:
            raise ValueError("spline_basis must exceed spline_degree.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.base_steps <= 0 or self.targeted_steps <= 0:
            raise ValueError("base_steps and targeted_steps must be positive.")
        if self.beta is not None and (not math.isfinite(self.beta) or self.beta <= 0):
            raise ValueError("beta must be positive and finite, or None for 20/sqrt(n).")
        if not 0.0 <= self.gcn_dropout < 1.0:
            raise ValueError("gcn_dropout must lie in [0,1).")
        if self.learning_rate_1step <= 0.0:
            raise ValueError("learning_rate_1step must be positive.")
        if self.learning_rate_2step <= 0.0:
            raise ValueError("learning_rate_2step must be positive.")
        if not 0.0 < self.propensity_floor < 0.5:
            raise ValueError("propensity_floor must lie in (0,0.5).")


@dataclass
class FittedTNet:
    model: "TNetModel"
    final_losses: Dict[str, float]
    training_config: dict


@dataclass
class TNetMajorityPrediction:
    arm_means: np.ndarray
    node_effects: np.ndarray


def _truncated_power_basis(
    z: torch.Tensor,
    *,
    n_basis: int,
    degree: int,
) -> torch.Tensor:
    """Evaluate the truncated-power spline family used by the TNet code.

    For degree ``r`` the basis is
    ``1, z, ..., z^r, (z-kappa_1)_+^r, ...``.  The official implementation
    fixes ``r=2`` and varies the interior-knot spacing by dataset.  Here the
    number of basis terms remains configurable for benchmark compatibility,
    with equally spaced interior knots on (0, 1).
    """

    z = torch.as_tensor(z)
    original_shape = z.shape
    flat = z.reshape(-1).clamp(0.0, 1.0)
    if degree < 1:
        raise ValueError("degree must be positive.")
    if n_basis <= degree:
        raise ValueError("n_basis must exceed degree.")

    columns = [torch.ones_like(flat)]
    columns.extend(flat.pow(power) for power in range(1, degree + 1))
    interior_count = n_basis - (degree + 1)
    if interior_count > 0:
        knots = torch.linspace(
            0.0,
            1.0,
            interior_count + 2,
            device=flat.device,
            dtype=flat.dtype,
        )[1:-1]
        columns.extend(torch.relu(flat - knot).pow(degree) for knot in knots)
    basis = torch.stack(columns, dim=1)
    return basis.reshape(*original_shape, n_basis)

def _interpolate_grid(grid_values: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of row-specific values on [0,1]."""

    if grid_values.ndim != 2:
        raise ValueError("grid_values must have shape [rows, grid_size+1].")
    z = torch.as_tensor(z, device=grid_values.device, dtype=grid_values.dtype)
    if z.ndim != 1 or z.shape[0] != grid_values.shape[0]:
        raise ValueError("z must be one-dimensional with one value per row.")
    intervals = grid_values.shape[1] - 1
    position = z.clamp(0.0, 1.0) * intervals
    lower = torch.floor(position).long().clamp(0, intervals)
    upper = torch.ceil(position).long().clamp(0, intervals)
    frac = position - lower.to(position.dtype)
    rows = torch.arange(grid_values.shape[0], device=grid_values.device)
    low_value = grid_values[rows, lower]
    high_value = grid_values[rows, upper]
    return low_value + frac * (high_value - low_value)


class _Predictor(nn.Module):
    """Three-linear-layer predictor matching the official TNet blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.leaky_relu(self.linear1(values), negative_slope=0.2)
        values = self.dropout(values)
        values = F.leaky_relu(self.linear2(values), negative_slope=0.2)
        values = self.dropout(values)
        return self.linear3(values)


class _TreatmentPropensityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.leaky_relu(self.linear1(values), negative_slope=0.2)
        return torch.sigmoid(self.linear2(values)).squeeze(1)


class _OutcomeHead(nn.Module):
    def __init__(self, representation_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _Predictor(
            representation_dim + 1,
            2 * hidden_dim,
            hidden_dim,
            1,
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z[:, None]], dim=1)).squeeze(1)


class TNetModel(nn.Module):
    def __init__(self, *, input_dim: int, config: TNetConfig):
        super().__init__()
        self.config = config
        self.representation_dim = max(config.hidden_dim // 2, 1)
        self.gcn = nn.Linear(input_dim, config.hidden_dim, bias=False)
        self.gcn_bias = nn.Parameter(torch.zeros(config.hidden_dim))
        self.feature_mlp = _Predictor(
            input_dim + config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            self.representation_dim,
        )
        self.g1_head = _TreatmentPropensityHead(
            self.representation_dim, config.hidden_dim
        )
        self.g2_head = nn.Linear(self.representation_dim, config.grid_size + 1)
        self.mu0 = _OutcomeHead(self.representation_dim, config.hidden_dim)
        self.mu1 = _OutcomeHead(self.representation_dim, config.hidden_dim)
        self.eps0 = nn.Parameter(torch.zeros(config.spline_basis))
        self.eps1 = nn.Parameter(torch.zeros(config.spline_basis))
        self.register_buffer("outcome_location", torch.tensor(0.0))
        self.register_buffer("outcome_scale", torch.tensor(1.0))
        # Match the author's explicit Normal(0, .1) weight initialization.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.g2_head.bias, mean=0.0, std=0.1)
        nn.init.uniform_(self.gcn_bias, -1 / math.sqrt(config.hidden_dim),
                         1 / math.sqrt(config.hidden_dim))

    def encode(self, adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square.")
        if x.ndim != 2 or x.shape[0] != adjacency.shape[0]:
            raise ValueError("x must have shape [nodes, features].")
        # The official TNet GCN applies one graph-convolution layer to A + I
        # (without symmetric normalization), then concatenates the result with X.
        with_self = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        neighbor = F.relu(with_self @ self.gcn(x) + self.gcn_bias)
        neighbor = F.dropout(neighbor, p=self.config.gcn_dropout, training=self.training)
        return self.feature_mlp(torch.cat([neighbor, x], dim=1))

    def g1_probability(self, h: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        p1 = self.g1_head(h)
        return torch.where(treatment >= 0.5, p1, 1.0 - p1).clamp_min(
            self.config.propensity_floor
        )

    def g2_grid(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.g2_head(h), dim=1)

    def g2_probability(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return _interpolate_grid(self.g2_grid(h), z).clamp_min(
            self.config.propensity_floor
        )

    def outcome_mean(self, h: torch.Tensor, treatment: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mu0 = self.mu0(h, z)
        mu1 = self.mu1(h, z)
        return torch.where(treatment >= 0.5, mu1, mu0)

    def perturbation(self, treatment: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        basis = _truncated_power_basis(
            z,
            n_basis=self.config.spline_basis,
            degree=self.config.spline_degree,
        )
        eps0 = basis @ self.eps0
        eps1 = basis @ self.eps1
        return torch.where(treatment >= 0.5, eps1, eps0)

    def potential_outcome_from_features(
        self, h: torch.Tensor, treatment: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        mu = self.outcome_mean(h, treatment, z)
        g1 = self.g1_probability(h, treatment)
        g2 = self.g2_probability(h, z)
        eps = self.perturbation(treatment, z)
        denom = (g1 * g2).clamp_min(self.config.propensity_floor)
        return (mu + eps / denom) * self.outcome_scale + self.outcome_location


def _as_graph_tensors(
    *,
    adjacency: np.ndarray,
    covariates: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    covariates = np.asarray(covariates, dtype=np.float32)
    treatment = np.asarray(treatment, dtype=np.float32)
    exposure = np.asarray(exposure, dtype=np.float32)
    outcomes = np.asarray(outcomes, dtype=np.float32)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    n = adjacency.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError("adjacency must be square.")
    if covariates.ndim != 2 or covariates.shape[0] != n:
        raise ValueError("covariates must have shape [nodes,features].")
    for name, value in (("treatment", treatment), ("exposure", exposure), ("outcomes", outcomes)):
        if value.shape != (n,):
            raise ValueError(f"{name} must have one value per node.")
    if np.any(adjacency.sum(axis=1) <= 0):
        raise ValueError("TNet adaptation requires positive node degree.")
    return tuple(torch.tensor(v, dtype=torch.float32) for v in (adjacency, covariates, treatment, exposure, outcomes))


def _build_tnet_optimizers(
    model: TNetModel,
    config: TNetConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build the two optimizers used by the official two-step TNet training loop."""

    epsilon_params = {id(model.eps0), id(model.eps1)}
    base_params = [p for p in model.parameters() if id(p) not in epsilon_params]
    first_step = torch.optim.Adam(
        base_params,
        lr=config.learning_rate_1step,
        weight_decay=config.weight_decay,
    )
    second_step = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate_2step,
        weight_decay=config.weight_decay,
    )
    return first_step, second_step


def fit_tnet_graph(
    *,
    adjacency: np.ndarray,
    covariates: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    outcomes: np.ndarray,
    config: TNetConfig | None = None,
) -> FittedTNet:
    """Fit one TNet model to one observed network snapshot."""

    config = config or TNetConfig()
    torch.manual_seed(int(config.seed))
    a, x, t, z, y = _as_graph_tensors(
        adjacency=adjacency,
        covariates=covariates,
        treatment=treatment,
        exposure=exposure,
        outcomes=outcomes,
    )
    model = TNetModel(input_dim=x.shape[1], config=config)
    if config.normalize_outcomes:
        location = y.mean()
        scale = y.std(unbiased=True) if y.numel() > 1 else y.new_tensor(1.0)
        # Constant observed outcomes do not define a scale; use one without
        # importing counterfactual outcomes or altering effect definitions.
        if not torch.isfinite(scale) or scale <= torch.finfo(y.dtype).eps:
            scale = y.new_tensor(1.0)
        model.outcome_location.copy_(location)
        model.outcome_scale.copy_(scale)
        y = (y - location) / scale
    effective_beta = 20.0 / math.sqrt(y.numel()) if config.beta is None else config.beta
    nuisance_optimizer, targeted_optimizer = _build_tnet_optimizers(model, config)

    nuisance_value = float("nan")
    targeted_value = float("nan")
    for _ in range(config.epochs):
        model.train()

        # Step 1 mirrors the official base-parameter update: fit Q, g_T and g_Z
        # while also including the targeted residual term.  The fluctuation
        # coefficients are not in this optimizer, and propensity scores are
        # detached inside the clever-covariate denominator.
        for _base in range(config.base_steps):
            nuisance_optimizer.zero_grad(set_to_none=True)
            h = model.encode(a, x)
            p1 = model.g1_head(h).clamp(1e-6, 1.0 - 1e-6)
            g2 = model.g2_probability(h, z)
            mu = model.outcome_mean(h, t, z)
            eps = model.perturbation(t, z)
            g1_observed = torch.where(t >= 0.5, p1, 1.0 - p1).clamp_min(
                config.propensity_floor
            )
            loss_g1 = F.binary_cross_entropy(p1, t)
            loss_g2 = -torch.log(g2).mean()
            loss_mu = F.mse_loss(mu, y)
            nuisance_loss = loss_mu + config.alpha * loss_g1 + config.gamma * loss_g2
            corrected = mu + eps / (
                g1_observed.detach() * g2.detach()
            ).clamp_min(config.propensity_floor)
            base_targeted_loss = F.mse_loss(corrected, y)
            base_loss = nuisance_loss + effective_beta * base_targeted_loss
            base_loss.backward()
            nuisance_optimizer.step()
            nuisance_value = float(nuisance_loss.detach().item())

        # Step 2 follows the official fluctuation update: all non-propensity
        # outcome/representation parameters plus epsilon can move, while g_T
        # and g_Z are treated as fixed nuisance estimates in the denominator.
        for _targeted in range(config.targeted_steps):
            targeted_optimizer.zero_grad(set_to_none=True)
            h = model.encode(a, x)
            mu = model.outcome_mean(h, t, z)
            p1 = model.g1_head(h).clamp(1e-6, 1.0 - 1e-6)
            g1_observed = torch.where(t >= 0.5, p1, 1.0 - p1).clamp_min(
                config.propensity_floor
            )
            g2 = model.g2_probability(h, z)
            eps = model.perturbation(t, z)
            corrected = mu + eps / (
                g1_observed.detach() * g2.detach()
            ).clamp_min(config.propensity_floor)
            targeted_loss = effective_beta * F.mse_loss(corrected, y)
            targeted_loss.backward()
            targeted_optimizer.step()
            targeted_value = float(targeted_loss.detach().item())

    model.eval()
    return FittedTNet(
        model=model,
        final_losses={"nuisance": nuisance_value, "targeted": targeted_value},
        training_config={**asdict(config), "effective_beta": effective_beta,
            "n_observations": y.numel(),
            "base_optimizer_updates": config.epochs * config.base_steps,
            "targeted_optimizer_updates": config.epochs * config.targeted_steps,
            "outcome_location": float(model.outcome_location),
            "outcome_scale": float(model.outcome_scale)},
    )


def predict_tnet_majority_effects(
    model: TNetModel,
    *,
    adjacency: np.ndarray,
    covariates: np.ndarray,
    treatment_prob: float,
) -> TNetMajorityPrediction:
    """Integrate TNet pointwise potential outcomes over exact majority supports."""

    adjacency_np = np.asarray(adjacency, dtype=np.float32)
    covariates_np = np.asarray(covariates, dtype=np.float32)
    if covariates_np.ndim == 1:
        covariates_np = covariates_np[:, None]
    degree = adjacency_np.sum(axis=1).astype(np.int64)
    a = torch.tensor(adjacency_np, dtype=torch.float32)
    x = torch.tensor(covariates_np, dtype=torch.float32)
    arm_means = np.empty((degree.size, 4), dtype=np.float64)

    with torch.no_grad():
        h_all = model.encode(a, x)
        for i, d in enumerate(degree.tolist()):
            if d <= 0:
                raise ValueError("majority prediction requires positive node degree.")
            for arm_code, (own_treatment, majority_state) in enumerate(
                ((0, 0), (1, 0), (0, 1), (1, 1))
            ):
                weights = conditional_count_weights(d, treatment_prob, majority_state)
                supported = np.flatnonzero(weights > 0.0)
                z = torch.tensor(supported / float(d), dtype=torch.float32)
                t = torch.full((supported.size,), float(own_treatment), dtype=torch.float32)
                h = h_all[i : i + 1].expand(supported.size, -1)
                values = model.potential_outcome_from_features(h, t, z).cpu().numpy()
                arm_means[i, arm_code] = float(np.dot(weights[supported], values))

    node_effects = np.column_stack(
        [
            arm_means[:, 1] - arm_means[:, 0],
            arm_means[:, 3] - arm_means[:, 1],
            arm_means[:, 3] - arm_means[:, 0],
        ]
    )
    return TNetMajorityPrediction(arm_means=arm_means, node_effects=node_effects)
