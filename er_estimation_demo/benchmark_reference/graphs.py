"""Frozen C-era TEST graph randomization for historical input comparability.
The training generator continues using the unmodified OLD samplers.
"""
from __future__ import annotations
import math
import torch
import networkx as nx
from train_local_network_interference import (
    DataConfig, GRAPH_FAMILY_ORDER, GRAPH_TYPE_NAMES, TensorBatch, ER_MAX_RESAMPLE_ATTEMPTS,
    _coerce_target_mean_degrees, _configuration_degree_sequence, _generate_er_adjacency,
)
def _generate_configuration_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = 100,
) -> torch.Tensor:
    """Generate simple heterogeneous-degree graphs at episode-specific mean degree."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

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
                is_complete = edge_count == n_units * (n_units - 1) // 2
                is_star = (
                    edge_count == n_units - 1
                    and max(dict(graph.degree()).values()) == n_units - 1
                )
                if n_units >= 4 and edge_count > 1 and not (is_complete or is_star):
                    # Reduce Havel-Hakimi initializer structure; this is an
                    # approximate randomization, not a uniform-sampling proof.
                    swap_count = 10 * edge_count
                    try:
                        nx.double_edge_swap(
                            graph,
                            nswap=swap_count,
                            max_tries=max(20 * swap_count, 100),
                            seed=nx_seed,
                        )
                    except nx.NetworkXAlgorithmError:
                        # Completed swaps leave a valid graph with the same
                        # degrees. Retain it instead of biasing the degree
                        # prior by redrawing sequences that are hard to mix.
                        pass
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

def _generate_rgg_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate 2-D random geometric graphs at episode-specific mean degree."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

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
        if pending.numel() == 0:
            return adjacency
    raise RuntimeError(
        f"RGG generation still had {pending.numel()} graph(s) with isolated nodes "
        f"after {max_attempts} attempts."
    )

def _generate_sbm_adjacency(
    config: DataConfig,
    batch_size: int,
    *,
    generator: torch.Generator,
    target_mean_degree: Optional[torch.Tensor] = None,
    max_attempts: int = ER_MAX_RESAMPLE_ATTEMPTS,
) -> torch.Tensor:
    """Generate equal-size two-block SBM graphs at episode-specific mean degree."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

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
        if pending.numel() == 0:
            return adjacency
    raise RuntimeError(
        f"SBM generation still had {pending.numel()} graph(s) with isolated nodes "
        f"after {max_attempts} attempts."
    )
