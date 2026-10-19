"""Shared estimation and integration utilities for the baseline methods."""
from __future__ import annotations


from dataclasses import dataclass


from pathlib import Path


from types import SimpleNamespace


import math


from torch import nn


import numpy as np


import torch


import torch.nn.functional as F


from pfn_social._vendor.causalpfn import InContextModel


from pfn_social.contracts import arm_probability, conditional_count_weights


from pfn_social.runtime import fork_device_rng, seed_device


import pfn_social._vendor.hypersci as UPSTREAM


import pfn_social._vendor.hypersci as UTILS


def load_causalpfn_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> InContextModel:
    """Restore the official pretrained CausalPFN checkpoint without training."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CausalPFN checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
class TNetConfig:
    hidden_dim: int = 64
    grid_size: int = 20
    spline_basis: int = 12
    spline_degree: int = 2
    epochs: int = 160
    base_steps: int = 1
    targeted_steps: int = 50
    learning_rate_1step: float = 0.0001
    learning_rate_2step: float = 0.01
    alpha: float = 0.5
    gamma: float = 1.0
    beta: float | None = None
    propensity_floor: float = 0.001
    weight_decay: float = 0.001
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
            raise ValueError(
                "beta must be positive and finite, or None for 20/sqrt(n)."
            )
        if not 0.0 <= self.gcn_dropout < 1.0:
            raise ValueError("gcn_dropout must lie in [0,1).")
        if self.learning_rate_1step <= 0.0:
            raise ValueError("learning_rate_1step must be positive.")
        if self.learning_rate_2step <= 0.0:
            raise ValueError("learning_rate_2step must be positive.")
        if not 0.0 < self.propensity_floor < 0.5:
            raise ValueError("propensity_floor must lie in (0,0.5).")


@dataclass
class TNetMajorityPrediction:
    arm_means: np.ndarray
    node_effects: np.ndarray


def _truncated_power_basis(
    z: torch.Tensor, *, n_basis: int, degree: int
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
    columns.extend((flat.pow(power) for power in range(1, degree + 1)))
    interior_count = n_basis - (degree + 1)
    if interior_count > 0:
        knots = torch.linspace(
            0.0, 1.0, interior_count + 2, device=flat.device, dtype=flat.dtype
        )[1:-1]
        columns.extend((torch.relu(flat - knot).pow(degree) for knot in knots))
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
        self.net = _Predictor(representation_dim + 1, 2 * hidden_dim, hidden_dim, 1)

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
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.g2_head.bias, mean=0.0, std=0.1)
        nn.init.uniform_(
            self.gcn_bias,
            -1 / math.sqrt(config.hidden_dim),
            1 / math.sqrt(config.hidden_dim),
        )

    def encode(self, adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square.")
        if x.ndim != 2 or x.shape[0] != adjacency.shape[0]:
            raise ValueError("x must have shape [nodes, features].")
        with_self = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        neighbor = F.relu(with_self @ self.gcn(x) + self.gcn_bias)
        neighbor = F.dropout(
            neighbor, p=self.config.gcn_dropout, training=self.training
        )
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

    def outcome_mean(
        self, h: torch.Tensor, treatment: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        mu0 = self.mu0(h, z)
        mu1 = self.mu1(h, z)
        return torch.where(treatment >= 0.5, mu1, mu0)

    def perturbation(self, treatment: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        basis = _truncated_power_basis(
            z, n_basis=self.config.spline_basis, degree=self.config.spline_degree
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


def _build_tnet_optimizers(
    model: TNetModel, config: TNetConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build the two optimizers used by the official two-step TNet training loop."""
    epsilon_params = {id(model.eps0), id(model.eps1)}
    base_params = [p for p in model.parameters() if id(p) not in epsilon_params]
    first_step = torch.optim.Adam(
        base_params, lr=config.learning_rate_1step, weight_decay=config.weight_decay
    )
    second_step = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate_2step,
        weight_decay=config.weight_decay,
    )
    return (first_step, second_step)


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
    device = next(model.parameters()).device
    a = torch.tensor(adjacency_np, dtype=torch.float32, device=device)
    x = torch.tensor(covariates_np, dtype=torch.float32, device=device)
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
                z = torch.tensor(
                    supported / float(d), dtype=torch.float32, device=device
                )
                t = torch.full(
                    (supported.size,),
                    float(own_treatment),
                    dtype=torch.float32,
                    device=device,
                )
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


def pair_incidence(adjacency):
    a = torch.as_tensor(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or (not torch.equal(a, a.T)):
        raise ValueError("Need an undirected square adjacency.")
    if (
        torch.any(a.diag() != 0)
        or torch.any((a != 0) & (a != 1))
        or torch.any(a.sum(1) <= 0)
    ):
        raise ValueError("Need a binary graph without loops or isolated nodes.")
    pairs = torch.nonzero(torch.triu(a, diagonal=1), as_tuple=False)
    index = torch.stack(
        [pairs.reshape(-1), torch.arange(len(pairs)).repeat_interleave(2)]
    )
    return (index, pairs)


def make_model(input_dim, hidden=32):
    args = SimpleNamespace(
        h_dim=hidden,
        g_dim=hidden,
        n_out=0,
        dropout=0.5,
        graph_model="hypergraph",
        encoder="gat",
        skip="123",
        num_gnn_layer=1,
        phi_layer=2,
        activate=1,
    )
    return UPSTREAM.HyperSCI(args, input_dim)


@torch.no_grad()
def build_pair_cache(model, features, adjacency):
    """Exact decomposition of one PyG HypergraphConv layer on size-two edges.

    Attention is normalized inside each edge. Evaluate its four binary endpoint
    assignments once, then sum incident messages with the original node degree.
    Dropout must be disabled; treatment-dependent attention is recomputed in all
    four cases. This is an evaluation optimization, not an approximate model.
    """
    if (
        model.training
        or model.num_gnn_layer != 1
        or model.encoder != "gat"
        or (model.n_out != 0)
    ):
        raise ValueError(
            "Cache requires the one-layer attention evaluation configuration."
        )
    if model.hgnn.attention_mode != "node":
        raise ValueError("Only within-edge attention supports this decomposition.")
    device = next(model.parameters()).device
    a = torch.as_tensor(adjacency).cpu()
    index, pairs = pair_incidence(a)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    phi = model.phi_x(x)
    edge_count = len(pairs)
    duplicated = phi[pairs.reshape(-1).to(device)]
    disjoint = torch.stack(
        [torch.arange(2 * edge_count), torch.arange(edge_count).repeat_interleave(2)]
    ).to(device)
    messages = torch.empty(edge_count, 2, 2, 2, model.g_dim)
    bias = model.hgnn.bias.detach().cpu()
    for tu in (0, 1):
        for tv in (0, 1):
            treatments = torch.tensor([tu, tv], dtype=x.dtype, device=device).repeat(
                edge_count
            )
            masked = duplicated * treatments[:, None]
            attrs = masked.reshape(edge_count, 2, -1).mean(1)
            pair_outputs = model.hgnn(masked, disjoint, hyperedge_attr=attrs)
            messages[:, :, tu, tv] = (pair_outputs.cpu() - bias).reshape(
                edge_count, 2, -1
            )
    degree = a.sum(1).long()
    neighbors = [torch.where(row > 0)[0] for row in a]
    edge_lookup = {tuple(pair): e for (e, pair) in enumerate(pairs.tolist())}
    base = torch.empty(len(x), 2, model.g_dim)
    deltas = []
    for i, adjacent in enumerate(neighbors):
        delta = torch.empty(2, len(adjacent), model.g_dim)
        for ti in (0, 1):
            zero = []
            for k, j in enumerate(adjacent.tolist()):
                u, v = sorted((i, j))
                e = edge_lookup[u, v]
                if i == u:
                    q0 = messages[e, 0, ti, 0]
                    q1 = messages[e, 0, ti, 1]
                else:
                    q0 = messages[e, 1, 0, ti]
                    q1 = messages[e, 1, 1, ti]
                zero.append(q0)
                delta[ti, k] = (q1 - q0) / degree[i]
            base[i, ti] = torch.stack(zero).sum(0) / degree[i] + bias
        deltas.append(delta)
    return dict(
        features=x,
        phi=phi,
        base=base.to(device),
        deltas=[d.to(device) for d in deltas],
        degree=degree,
        neighbors=neighbors,
    )


@torch.no_grad()
def predict_root_samples(model, cache, root, own_treatment, assignments):
    t = int(own_treatment)
    mask = torch.as_tensor(
        assignments, dtype=cache["features"].dtype, device=cache["features"].device
    )
    if mask.ndim != 2 or mask.shape[1] != int(cache["degree"][root]):
        raise ValueError("Need one assignment per immediate neighbor.")
    interference = cache["base"][root, t] + mask @ cache["deltas"][root][t]
    if model.activate:
        interference = F.relu(interference)
    n = len(mask)
    x = cache["features"][root].expand(n, -1)
    phi = cache["phi"][root].expand(n, -1)
    rep = torch.cat([x, phi if t else torch.zeros_like(phi), interference], dim=-1)
    head = model.out_t11 if t else model.out_t01
    return head(rep).squeeze(-1).cpu().numpy()


def conditional_assignments(degree, arm, samples, *, seed, treatment_prob=0.5):
    rng = np.random.default_rng(seed)
    weights = conditional_count_weights(int(degree), float(treatment_prob), int(arm))
    counts = rng.choice(degree + 1, size=samples, p=weights)
    ranks = rng.random((samples, degree)).argsort(axis=1).argsort(axis=1)
    return (ranks < counts[:, None]).astype(np.float32)


def majority_arm_propensities(
    degree: np.ndarray, *, treatment_prob: float
) -> np.ndarray:
    """Return P((D_i,S_i)=arm) for fixed codes 00,10,01,11."""
    degree = np.asarray(degree)
    if degree.ndim != 1:
        raise ValueError("degree must be one-dimensional.")
    if np.any(degree <= 0) or not np.allclose(degree, np.rint(degree)):
        raise ValueError("degree must contain positive integers.")
    p = float(treatment_prob)
    if not 0.0 < p < 1.0:
        raise ValueError("treatment_prob must lie strictly between zero and one.")
    result = np.empty((degree.size, 4), dtype=np.float64)
    for index, value in enumerate(degree.astype(np.int64).tolist()):
        low = arm_probability(value, p, 0)
        high = arm_probability(value, p, 1)
        result[index] = [(1.0 - p) * low, p * low, (1.0 - p) * high, p * high]
    return result


from pfn_social.simulation import HINET_PROTOCOL, _masks


HINET_ALPHA_GRID = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3)


def new_hinet(input_dim, hidden=16):
    from pfn_social._vendor.hinet import HINet

    return HINet(input_dim, hidden, dropout=0.0, gnn_layer="GIN")


@torch.no_grad()
def build_hinet_prediction_cache(model, x, adjacency):
    model.eval()
    embed = model.encoder(x)
    return dict(
        embed=embed,
        adjacency=adjacency.detach().cpu().numpy(),
        zero=model.outcome_message(
            torch.cat((embed, torch.zeros(len(x), 1, device=x.device)), 1)
        ),
        one=model.outcome_message(
            torch.cat((embed, torch.ones(len(x), 1, device=x.device)), 1)
        ),
    )


@torch.no_grad()
def predict_hinet_root(model, cache, root, own_treatment, assignments):
    """Exact local evaluation of the upstream one-layer GIN outcome branch."""
    ix = np.flatnonzero(cache["adjacency"][root])
    zero, one = cache["zero"], cache["one"]
    z = torch.as_tensor(assignments, dtype=zero.dtype, device=zero.device)
    agg = zero[ix].sum(0) + z @ (one[ix] - zero[ix])
    own = one[root] if own_treatment else zero[root]
    agg = agg + (1 + model.gnn_y.gin.eps) * own
    neighbor = model.gnn_y.mlp(agg)
    emb = cache["embed"][root].expand(len(z), -1)
    t = torch.full((len(z), 1), float(own_treatment), device=zero.device)
    return model.pred_y(torch.cat((neighbor, emb, t), 1)).flatten().cpu().numpy()


@torch.no_grad()
def integrate_hinet(model, x, adjacency, *, samples=4096, seed=0):
    if type(samples) is not int or samples < 2 or samples & (samples - 1):
        raise ValueError("HINet integration samples must be a power of two >= 2")
    cache = build_hinet_prediction_cache(model, x, adjacency)
    d = cache["adjacency"].sum(1).astype(int)
    if (d <= 0).any():
        raise ValueError("No high-majority target for isolated nodes")
    mu = np.empty((len(x), 4))
    half = np.empty_like(mu)
    for degree in np.unique(d):
        exact = degree <= 8
        for arm in (0, 1):
            masks = _masks(
                int(degree),
                arm,
                samples,
                (seed + 1009 * int(degree) + 37 * arm) % 2**32,
                exact,
            ).astype(np.float32)
            for root in np.flatnonzero(d == degree):
                for own in (0, 1):
                    total = first = 0.0
                    for start in range(0, len(masks), 512):
                        values = predict_hinet_root(
                            model, cache, root, own, masks[start : start + 512]
                        ).astype(np.float64)
                        total += values.sum()
                        take = max(0, min(len(values), len(masks) // 2 - start))
                        first += values[:take].sum()
                    mu[root, 2 * own + arm] = total / len(masks)
                    half[root, 2 * own + arm] = (
                        mu[root, 2 * own + arm] if exact else first / (len(masks) // 2)
                    )
    if not np.isfinite(mu).all():
        raise FloatingPointError("HINet conditional predictions nonfinite")
    return mu, dict(
        reference_probability=0.5,
        samples=samples,
        exact_nodes=int((d <= 8).sum()),
        half_full_mu_max_abs=float(np.abs(mu - half).max()),
        half_full_mu_rmse=float(np.sqrt(np.mean((mu - half) ** 2))),
        note="Half/full Sobol discrepancy is diagnostic, not a confidence bound",
        protocol=HINET_PROTOCOL,
    )

