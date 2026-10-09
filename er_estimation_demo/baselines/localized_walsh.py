"""Localized DR-Lasso: manuscript-aligned binary-treatment ITEs.

This module adapts the main point-estimation pipeline in
"Individualized Causal Effects under Network Interference with Combinatorial
Treatments" for the current p=1 benchmark:

* radius-one rooted, treatment-marked configurations;
* a root-treatment-masked radius-one adaptation of the configuration distance;
* explicit Epanechnikov-kernel or kNN localization in configuration space;
* cross-fitted nuisance residualization conditional on (g, X);
* localized weighted Walsh Lasso;
* separate fits at the reference and counterfactual configurations;
* the paper's one-step debiasing formula for the direct/own-treatment contrast;
* Kish effective sample size and local-design diagnostics.

The paper permits generic nuisance learners.  Here nuisances are estimated by
cross-fitted kernel smoothing using the same rooted-configuration distance and
the target covariate.  ``nuisance_ridge`` is retained for command-line
compatibility and acts only as a training-sample-mean shrinkage stabilizer; this
is not a Ridge S-learner.

The benchmark itself has p=1 and an exposure-mapped outcome DGP.  PFN receives
e=h(g), while this baseline receives the complete rooted configuration g.

Manuscript notation: treatment=T_i in {0,1}; treatment_r=R_i=2*T_i-1;
outcome=Y_i^obs; nu_y=nu_Y(g,x); nu_r=nu_R(g,x); dr_beta=beta_hat(g);
weighted_gram=Sigma_R(g); direct_debiased=delta_i^Direct(g).
The majority adapter samples k with q_ik(s), then a uniform neighbor subset;
its plain Monte Carlo mean does not multiply by q_ik(s) a second time.
This is an empirical ITE baseline; original inference guarantees are not asserted.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from estimands import conditional_count_weights
from baselines.localized_config import LOCALIZED_METHOD_NAME, LOCALIZED_PROTOCOL


EFFECT_NAMES = ("direct", "spillover", "total")
def _weisfeiler_lehman_hash(
    graph: nx.Graph,
    *,
    node_attr: str | None = None,
) -> str:
    """Compute a within-run structural filter without NetworkX version noise."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The hashes produced for graphs without node or edge attributes changed.*",
            category=UserWarning,
        )
        return nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr=node_attr,
            iterations=4,
        )


ITE_METHOD_NAME = LOCALIZED_METHOD_NAME
LOCALIZED_METHOD_NAMES = (ITE_METHOD_NAME,)
_EPS = 1e-12


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _as_1d(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _validate_binary(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if bool(np.any((array != 0) & (array != 1))):
        raise ValueError(f"{name} must contain only 0/1 values.")
    return array.astype(np.uint8, copy=False)


def _treatment_to_walsh(treatment: np.ndarray | float) -> np.ndarray:
    return 2.0 * np.asarray(treatment, dtype=np.float64) - 1.0


@dataclass(frozen=True)
class _MarkedComponent:
    """A connected component of the neighbor-induced graph."""

    graph: nx.Graph
    treatment_marks: np.ndarray
    signature: tuple[int, tuple[int, ...], str]


def _build_marked_components(
    graph: nx.Graph,
    treatment_marks: np.ndarray,
) -> tuple[_MarkedComponent, ...]:
    if graph.number_of_nodes() == 0:
        return ()
    components: list[_MarkedComponent] = []
    for nodes in nx.connected_components(graph):
        ordered = sorted(int(node) for node in nodes)
        subgraph = nx.convert_node_labels_to_integers(
            graph.subgraph(ordered).copy(),
            ordering="sorted",
        )
        marks = np.asarray([treatment_marks[node] for node in ordered], dtype=np.int8)
        signature = (
            subgraph.number_of_nodes(),
            tuple(sorted(int(value) for _, value in subgraph.degree())),
            _weisfeiler_lehman_hash(subgraph),
        )
        components.append(
            _MarkedComponent(
                graph=subgraph,
                treatment_marks=marks,
                signature=signature,
            )
        )
    return tuple(sorted(components, key=lambda item: item.signature))


def _component_minimum_mark_mismatch(
    first: _MarkedComponent,
    second: _MarkedComponent,
) -> float:
    """Exact mark mismatch over isomorphisms of two connected components."""

    if first.signature != second.signature:
        return math.inf
    n = first.graph.number_of_nodes()
    if n == 1:
        return float(first.treatment_marks[0] != second.treatment_marks[0])
    edge_count = first.graph.number_of_edges()
    if edge_count == n * (n - 1) // 2:
        return float(
            abs(
                int(np.sum(first.treatment_marks == 1))
                - int(np.sum(second.treatment_marks == 1))
            )
        )
    matcher = nx.algorithms.isomorphism.GraphMatcher(first.graph, second.graph)
    best = n + 1
    found = False
    for mapping in matcher.isomorphisms_iter():
        found = True
        mismatch = sum(
            int(first.treatment_marks[node] != second.treatment_marks[mapping[node]])
            for node in range(n)
        )
        best = min(best, mismatch)
        if best == 0:
            break
    return float(best) if found else math.inf


@dataclass
class RootedConfiguration:
    """A radius-one rooted graph with binary treatment marks.

    Local node 0 is always the root. Treatment marks use the paper's {-1,+1}
    coding. Because equations (4)--(6) parameterize own treatment t separately
    from the interference environment g, the root's treatment mark is stored
    but masked when configuration distances are computed. Neighbor marks and
    the full rooted unmarked graph remain part of g.
    """

    adjacency: np.ndarray
    treatment_marks: np.ndarray
    global_nodes: np.ndarray
    root_global_index: int
    graph: nx.Graph = field(init=False, repr=False)
    degree_signature: tuple[int, ...] = field(init=False)
    structural_hash: str = field(init=False)
    neighbor_components: tuple[_MarkedComponent, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.adjacency, dtype=np.uint8)
        marks = np.asarray(self.treatment_marks, dtype=np.int8)
        nodes = np.asarray(self.global_nodes, dtype=np.int64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("configuration adjacency must be square.")
        if adjacency.shape[0] != marks.size or marks.ndim != 1:
            raise ValueError("configuration marks must match adjacency size.")
        if nodes.ndim != 1 or nodes.size != marks.size:
            raise ValueError("global_nodes must match configuration size.")
        if marks.size == 0:
            raise ValueError("configuration must contain its root.")
        if bool(np.any((marks != -1) & (marks != 1))):
            raise ValueError("treatment marks must use {-1,+1} coding.")
        if bool(np.any(adjacency != adjacency.T)):
            raise ValueError("configuration adjacency must be symmetric.")
        if bool(np.any(np.diag(adjacency) != 0)):
            raise ValueError("configuration adjacency must have zero diagonal.")
        self.adjacency = adjacency
        self.treatment_marks = marks
        self.global_nodes = nodes
        graph = nx.from_numpy_array(adjacency)
        nx.set_node_attributes(
            graph,
            {node: int(node == 0) for node in graph.nodes},
            "is_root",
        )
        self.graph = graph
        self.degree_signature = tuple(sorted(int(value) for _, value in graph.degree()))
        self.structural_hash = _weisfeiler_lehman_hash(
            graph,
            node_attr="is_root",
        )
        neighbor_graph = graph.subgraph(range(1, marks.size)).copy()
        self.neighbor_components = _build_marked_components(
            neighbor_graph,
            marks,
        )

    @property
    def neighbor_exposure(self) -> float:
        if self.treatment_marks.size <= 1:
            return 0.0
        return float(np.mean(self.treatment_marks[1:] == 1))


def build_radius_one_configuration(
    adjacency: np.ndarray,
    treatment: np.ndarray | Sequence[int],
    *,
    root: int,
    neighbor_treatment: np.ndarray | Sequence[int] | None = None,
) -> RootedConfiguration:
    """Construct G_i from a factual or query-specific neighbor assignment.

    ``neighbor_treatment`` is a length-N 0/1 vector.  Only entries at the
    root's neighbors are used; the root mark remains its factual treatment.
    """

    adjacency_array = np.asarray(adjacency, dtype=np.uint8)
    if adjacency_array.ndim != 2 or adjacency_array.shape[0] != adjacency_array.shape[1]:
        raise ValueError("adjacency must be square.")
    if bool(np.any(adjacency_array != adjacency_array.T)):
        raise ValueError("adjacency must be symmetric.")
    n = int(adjacency_array.shape[0])
    if not 0 <= int(root) < n:
        raise ValueError("root is outside the graph.")
    factual = _validate_binary(np.asarray(treatment), name="treatment")
    if factual.size != n:
        raise ValueError("treatment length must equal graph size.")
    override: np.ndarray | None = None
    if neighbor_treatment is not None:
        override = _validate_binary(
            np.asarray(neighbor_treatment),
            name="neighbor_treatment",
        )
        if override.size != n:
            raise ValueError("neighbor_treatment length must equal graph size.")

    neighbors = np.flatnonzero(adjacency_array[int(root)] > 0).astype(np.int64)
    local_nodes = np.concatenate(
        [np.asarray([int(root)], dtype=np.int64), np.sort(neighbors)]
    )
    local_adjacency = adjacency_array[np.ix_(local_nodes, local_nodes)].copy()
    local_binary_marks = factual[local_nodes].copy()
    if override is not None and neighbors.size:
        local_binary_marks[1:] = override[local_nodes[1:]]
    marks = (2 * local_binary_marks.astype(np.int8)) - 1
    return RootedConfiguration(
        adjacency=local_adjacency,
        treatment_marks=marks,
        global_nodes=local_nodes,
        root_global_index=int(root),
    )


def build_exact_count_configuration(
    adjacency: np.ndarray,
    treatment: np.ndarray | Sequence[int],
    *,
    root: int,
    treated_neighbor_count: int,
    seed: int,
) -> RootedConfiguration:
    """Construct a radius-one target with exactly ``k`` treated neighbors.

    Conditional on the count, treated neighbors are sampled uniformly without
    replacement.  The root's own treatment remains a separate argument of the
    response surface and is therefore not changed here.
    """

    adjacency_array = np.asarray(adjacency, dtype=np.uint8)
    factual = _validate_binary(np.asarray(treatment), name="treatment")
    neighbors = np.flatnonzero(adjacency_array[int(root)] > 0).astype(np.int64)
    degree = int(neighbors.size)
    count = int(treated_neighbor_count)
    if not 0 <= count <= degree:
        raise ValueError("treated_neighbor_count must lie between zero and degree.")
    override = np.zeros_like(factual, dtype=np.uint8)
    if count:
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(neighbors, size=count, replace=False)
        override[chosen] = 1
    return build_radius_one_configuration(
        adjacency_array,
        factual,
        root=int(root),
        neighbor_treatment=override,
    )


def _is_star_or_rooted_clique(configuration: RootedConfiguration) -> bool:
    neighbors = configuration.adjacency[1:, 1:]
    if neighbors.size == 0:
        return True
    upper = neighbors[np.triu_indices(neighbors.shape[0], k=1)]
    return bool(np.all(upper == 0) or np.all(upper == 1))


def _minimum_mark_mismatch_fraction(
    first: RootedConfiguration,
    second: RootedConfiguration,
) -> float:
    """Compute Delta_1 exactly without enumerating global ego-graph symmetries.

    Radius-one rooted isomorphisms reduce to isomorphisms of the graph induced
    by the neighbors. That graph is a disjoint union of connected components.
    We match isomorphic components with a minimum-cost assignment and optimize
    treatment-mark mismatches inside each component. This is mathematically
    equivalent to the paper's minimum over root-preserving isomorphisms, while
    avoiding factorial permutations of isolated/symmetric neighbors.
    """

    if first.adjacency.shape != second.adjacency.shape:
        return 1.0
    if first.degree_signature != second.degree_signature:
        return 1.0
    if first.structural_hash != second.structural_hash:
        return 1.0

    n = int(first.treatment_marks.size)
    if _is_star_or_rooted_clique(first) and _is_star_or_rooted_clique(second):
        treated_first = int(np.sum(first.treatment_marks[1:] == 1))
        treated_second = int(np.sum(second.treatment_marks[1:] == 1))
        return abs(treated_first - treated_second) / float(n)

    first_groups: dict[tuple[int, tuple[int, ...], str], list[_MarkedComponent]] = {}
    second_groups: dict[tuple[int, tuple[int, ...], str], list[_MarkedComponent]] = {}
    for component in first.neighbor_components:
        first_groups.setdefault(component.signature, []).append(component)
    for component in second.neighbor_components:
        second_groups.setdefault(component.signature, []).append(component)
    if set(first_groups) != set(second_groups):
        return 1.0

    mismatch_total = 0.0
    for signature, left_components in first_groups.items():
        right_components = second_groups[signature]
        if len(left_components) != len(right_components):
            return 1.0
        count = len(left_components)
        if signature[0] == 1:
            treated_left = sum(int(component.treatment_marks[0] == 1) for component in left_components)
            treated_right = sum(int(component.treatment_marks[0] == 1) for component in right_components)
            mismatch_total += abs(treated_left - treated_right)
            continue
        costs = np.empty((count, count), dtype=np.float64)
        for left_index, left_component in enumerate(left_components):
            for right_index, right_component in enumerate(right_components):
                costs[left_index, right_index] = _component_minimum_mark_mismatch(
                    left_component,
                    right_component,
                )
        rows, columns = linear_sum_assignment(costs)
        assigned = costs[rows, columns]
        if bool(np.any(~np.isfinite(assigned))):
            return 1.0
        mismatch_total += float(np.sum(assigned))
    return mismatch_total / float(n)

def rooted_configuration_distance(
    first: RootedConfiguration,
    second: RootedConfiguration,
    *,
    radius: int = 1,
) -> float:
    """Compute equation (2) for R in {0,1}."""

    if radius not in (0, 1):
        raise ValueError("this benchmark currently supports radius 0 or 1.")
    # The root treatment is the separate own-treatment argument t in
    # f(t; g, x), so Delta_0 is zero under the interference-environment
    # convention used for own-treatment contrasts.
    distance = 0.0
    if radius == 1:
        distance += 0.25 * _minimum_mark_mismatch_fraction(first, second)
    return float(distance)


def _pairwise_configuration_distances(
    configurations: Sequence[RootedConfiguration],
    *,
    radius: int,
) -> np.ndarray:
    n = len(configurations)
    distances = np.zeros((n, n), dtype=np.float64)
    for left in range(n):
        for right in range(left + 1, n):
            value = rooted_configuration_distance(
                configurations[left],
                configurations[right],
                radius=radius,
            )
            distances[left, right] = value
            distances[right, left] = value
    return distances


@dataclass(frozen=True)
class _WeightResult:
    weights: np.ndarray
    effective_sample_size: float
    local_radius: float
    used_adaptive_knn: bool
    n_positive: int


def _localized_weights(
    distances: np.ndarray,
    *,
    bandwidth: float,
    min_neighbors: int,
    allowed: np.ndarray | None = None,
    mode: str = "kernel",
) -> _WeightResult:
    """Construct the paper's compact-kernel or kNN configuration weights.

    The modes are deliberately explicit. ``kernel`` never expands its support
    when too few configurations fall inside the compact bandwidth. ``knn``
    uses exactly the nearest ``min_neighbors`` available configurations and an
    adaptive Epanechnikov radius. This avoids silently changing estimators.
    """

    distances = _as_1d(distances, name="distances")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive.")
    if min_neighbors <= 0:
        raise ValueError("min_neighbors must be positive.")
    if mode not in {"kernel", "knn"}:
        raise ValueError("mode must be 'kernel' or 'knn'.")
    n = distances.size
    if allowed is None:
        allowed_mask = np.ones(n, dtype=bool)
    else:
        allowed_mask = np.asarray(allowed, dtype=bool)
        if allowed_mask.shape != (n,):
            raise ValueError("allowed mask must match distances.")
    available = np.flatnonzero(allowed_mask)
    if available.size == 0:
        raise ValueError("no observations are available for localization.")

    raw = np.zeros(n, dtype=np.float64)
    used_adaptive = mode == "knn"
    if mode == "kernel":
        scaled = distances / bandwidth
        inside = allowed_mask & (scaled < 1.0)
        raw[inside] = 1.0 - np.square(scaled[inside])
    else:
        k = min(min_neighbors, available.size)
        ordered = available[np.argsort(distances[available], kind="stable")[:k]]
        max_distance = float(np.max(distances[ordered]))
        radius = max(np.nextafter(max_distance, math.inf), _EPS)
        adaptive = np.maximum(
            1.0 - np.square(distances[ordered] / radius),
            0.0,
        )
        if float(np.sum(adaptive)) <= _EPS:
            adaptive = np.ones(k, dtype=np.float64)
        raw[ordered] = adaptive

    total = float(np.sum(raw))
    if total <= _EPS:
        return _WeightResult(
            weights=np.zeros(n, dtype=np.float64),
            effective_sample_size=0.0,
            local_radius=float("nan"),
            used_adaptive_knn=used_adaptive,
            n_positive=0,
        )
    weights = raw / total
    positive = weights > 0.0
    neff = 1.0 / max(float(np.sum(np.square(weights))), _EPS)
    local_radius = float(np.max(distances[positive]))
    return _WeightResult(
        weights=weights,
        effective_sample_size=neff,
        local_radius=local_radius,
        used_adaptive_knn=used_adaptive,
        n_positive=int(np.count_nonzero(positive)),
    )

def _weighted_mean_with_shrinkage(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    shrinkage: float,
) -> float:
    if shrinkage < 0.0:
        raise ValueError("nuisance_ridge/shrinkage must be nonnegative.")
    local = float(np.sum(weights * values))
    if shrinkage == 0.0:
        return local
    global_mean = float(np.mean(values))
    return (local + shrinkage * global_mean) / (1.0 + shrinkage)


def _combined_nuisance_distance(
    graph_distance: np.ndarray,
    x: np.ndarray,
    target_x: float,
    *,
    graph_bandwidth: float,
    x_scale: float,
) -> np.ndarray:
    graph_part = graph_distance / max(graph_bandwidth, _EPS)
    x_part = np.abs(x - float(target_x)) / max(x_scale, _EPS)
    return np.sqrt(np.square(graph_part) + np.square(x_part))


def _cross_fitted_nuisance_predictions(
    *,
    pairwise_graph_distance: np.ndarray,
    x: np.ndarray,
    outcome: np.ndarray,
    treatment_r: np.ndarray,
    graph_bandwidth: float,
    min_neighbors: int,
    localization_mode: str,
    folds: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(outcome.size)
    folds = min(max(int(folds), 2), n)
    indices = np.arange(n)
    x_scale = max(float(np.std(x)), 1.0e-6)
    nu_y = np.empty(n, dtype=np.float64)
    nu_r = np.empty(n, dtype=np.float64)
    for fold in range(folds):
        holdout = indices[indices % folds == fold]
        train_mask = indices % folds != fold
        for unit in holdout:
            combined = _combined_nuisance_distance(
                pairwise_graph_distance[unit],
                x,
                float(x[unit]),
                graph_bandwidth=graph_bandwidth,
                x_scale=x_scale,
            )
            weight_result = _localized_weights(
                combined,
                bandwidth=1.0,
                min_neighbors=min_neighbors,
                allowed=train_mask,
                mode=localization_mode,
            )
            if weight_result.n_positive == 0:
                # The paper permits generic nuisance learners. When a strict
                # compact kernel has no cross-fold neighbor, use the training-
                # fold mean for the nuisance only; target support is still
                # diagnosed separately and is never fabricated.
                nu_y[unit] = float(np.mean(outcome[train_mask]))
                nu_r[unit] = float(np.mean(treatment_r[train_mask]))
            else:
                nu_y[unit] = _weighted_mean_with_shrinkage(
                    outcome[train_mask],
                    weight_result.weights[train_mask],
                    shrinkage=shrinkage,
                )
                nu_r[unit] = _weighted_mean_with_shrinkage(
                    treatment_r[train_mask],
                    weight_result.weights[train_mask],
                    shrinkage=shrinkage,
                )
    return nu_y, nu_r


def _weighted_walsh_lasso(
    *,
    treatment_r: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    lasso_alpha: float,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> np.ndarray:
    """Fit Y ~= alpha_0 + alpha_1 R with an unpenalized intercept."""

    if lasso_alpha < 0.0:
        raise ValueError("lasso_alpha must be nonnegative.")
    weights = weights / max(float(weights.sum()), _EPS)
    mean_r = float(np.sum(weights * treatment_r))
    var_r = float(np.sum(weights * np.square(treatment_r - mean_r)))
    if var_r <= _EPS:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)
    if lasso_alpha == 0.0:
        design = np.column_stack([np.ones_like(treatment_r), treatment_r])
        gram = design.T @ (design * weights[:, None])
        rhs = design.T @ (weights * outcome)
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ rhs

    beta0 = float(np.sum(weights * outcome))
    beta1 = 0.0
    z1 = float(np.sum(weights * treatment_r * treatment_r))
    for _ in range(max_iter):
        previous = (beta0, beta1)
        beta0 = float(np.sum(weights * (outcome - treatment_r * beta1)))
        rho = float(np.sum(weights * treatment_r * (outcome - beta0)))
        beta1 = _soft_threshold(rho, lasso_alpha / 2.0) / max(z1, _EPS)
        if max(abs(beta0 - previous[0]), abs(beta1 - previous[1])) <= tol:
            break
    return np.asarray([beta0, beta1], dtype=np.float64)


def _weighted_residual_lasso(
    *,
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
    weights: np.ndarray,
    lasso_alpha: float,
) -> float:
    denom = float(np.sum(weights * treatment_residual * treatment_residual))
    if denom <= _EPS:
        return float("nan")
    rho = float(np.sum(weights * treatment_residual * outcome_residual))
    return _soft_threshold(rho, lasso_alpha / 2.0) / denom


@dataclass(frozen=True)
class _TargetFit:
    wlasso_alpha: np.ndarray
    dr_beta: float
    nu_y: float
    nu_r: float
    weights: np.ndarray
    effective_sample_size: float
    local_radius: float
    used_adaptive_knn: bool
    n_positive: int
    weighted_raw_variance: float
    weighted_gram: float
    wlasso_supported: bool
    drlasso_supported: bool
    direct_debiased: float
    direct_standard_error: float

    def wlasso_response(self, treatment: float) -> float:
        if not self.wlasso_supported:
            return float("nan")
        r_value = float(_treatment_to_walsh(treatment))
        return float(self.wlasso_alpha[0] + self.wlasso_alpha[1] * r_value)

    def drlasso_response(self, treatment: float) -> float:
        if not self.drlasso_supported:
            return float("nan")
        r_value = float(_treatment_to_walsh(treatment))
        return float(self.nu_y + (r_value - self.nu_r) * self.dr_beta)

    def diagnostics(self) -> Dict[str, float | int | bool]:
        return {
            "effective_sample_size": self.effective_sample_size,
            "local_radius": self.local_radius,
            "used_adaptive_knn": self.used_adaptive_knn,
            "n_positive": self.n_positive,
            "weighted_raw_variance": self.weighted_raw_variance,
            "weighted_gram": self.weighted_gram,
            "wlasso_supported": self.wlasso_supported,
            "drlasso_supported": self.drlasso_supported,
        }


@dataclass
class PaperLocalizedWalshEstimator:
    adjacency: np.ndarray
    treatment: np.ndarray
    treatment_r: np.ndarray
    x: np.ndarray
    outcome: np.ndarray
    configurations: list[RootedConfiguration]
    pairwise_distance: np.ndarray
    outcome_residual: np.ndarray
    treatment_residual: np.ndarray
    bandwidth: float
    min_neighbors: int
    localization_mode: str
    lasso_alpha: float
    nuisance_shrinkage: float
    radius: int
    x_scale: float

    @classmethod
    def fit(
        cls,
        *,
        adjacency: np.ndarray,
        treatment: np.ndarray | Sequence[float],
        x: np.ndarray | Sequence[float],
        outcome: np.ndarray | Sequence[float],
        bandwidth: float,
        min_neighbors: int,
        localization_mode: str,
        lasso_alpha: float,
        nuisance_ridge: float,
        cross_fit_folds: int,
        radius: int = 1,
    ) -> "PaperLocalizedWalshEstimator":
        adjacency_array = np.asarray(adjacency, dtype=np.uint8)
        treatment_array = _validate_binary(np.asarray(treatment), name="treatment")
        x_array = _as_1d(x, name="x")
        outcome_array = _as_1d(outcome, name="outcome")
        n = int(treatment_array.size)
        if adjacency_array.shape != (n, n):
            raise ValueError("adjacency and treatment dimensions disagree.")
        if x_array.size != n or outcome_array.size != n:
            raise ValueError("x and outcome must match treatment length.")
        if n < 4:
            raise ValueError("at least four factual units are required.")
        configurations = [
            build_radius_one_configuration(
                adjacency_array,
                treatment_array,
                root=unit,
            )
            for unit in range(n)
        ]
        pairwise = _pairwise_configuration_distances(
            configurations,
            radius=radius,
        )
        treatment_r = _treatment_to_walsh(treatment_array)
        nu_y_hat, nu_r_hat = _cross_fitted_nuisance_predictions(
            pairwise_graph_distance=pairwise,
            x=x_array,
            outcome=outcome_array,
            treatment_r=treatment_r,
            graph_bandwidth=bandwidth,
            min_neighbors=min_neighbors,
            localization_mode=localization_mode,
            folds=cross_fit_folds,
            shrinkage=nuisance_ridge,
        )
        return cls(
            adjacency=adjacency_array,
            treatment=treatment_array,
            treatment_r=treatment_r,
            x=x_array,
            outcome=outcome_array,
            configurations=configurations,
            pairwise_distance=pairwise,
            outcome_residual=outcome_array - nu_y_hat,
            treatment_residual=treatment_r - nu_r_hat,
            bandwidth=bandwidth,
            min_neighbors=min(min_neighbors, n),
            localization_mode=localization_mode,
            lasso_alpha=lasso_alpha,
            nuisance_shrinkage=nuisance_ridge,
            radius=radius,
            x_scale=max(float(np.std(x_array)), 1.0e-6),
        )

    def _distances_to_target(self, target: RootedConfiguration) -> np.ndarray:
        # Reuse the exact factual distance row when the object is one of the
        # realized configurations; otherwise evaluate the query configuration.
        for index, configuration in enumerate(self.configurations):
            if target is configuration:
                return self.pairwise_distance[index]
        return np.asarray(
            [
                rooted_configuration_distance(
                    configuration,
                    target,
                    radius=self.radius,
                )
                for configuration in self.configurations
            ],
            dtype=np.float64,
        )

    def fit_at(self, target: RootedConfiguration, *, target_x: float) -> _TargetFit:
        graph_distances = self._distances_to_target(target)
        local = _localized_weights(
            graph_distances,
            bandwidth=self.bandwidth,
            min_neighbors=self.min_neighbors,
            mode=self.localization_mode,
        )
        weights = local.weights
        if local.n_positive == 0:
            return _TargetFit(
                wlasso_alpha=np.asarray([float("nan"), float("nan")]),
                dr_beta=float("nan"),
                nu_y=float("nan"),
                nu_r=float("nan"),
                weights=weights,
                effective_sample_size=0.0,
                local_radius=float("nan"),
                used_adaptive_knn=local.used_adaptive_knn,
                n_positive=0,
                weighted_raw_variance=float("nan"),
                weighted_gram=float("nan"),
                wlasso_supported=False,
                drlasso_supported=False,
                direct_debiased=float("nan"),
                direct_standard_error=float("nan"),
            )
        raw_mean = float(np.sum(weights * self.treatment_r))
        raw_variance = float(
            np.sum(weights * np.square(self.treatment_r - raw_mean))
        )
        wlasso_alpha = _weighted_walsh_lasso(
            treatment_r=self.treatment_r,
            outcome=self.outcome,
            weights=weights,
            lasso_alpha=self.lasso_alpha,
        )

        combined = _combined_nuisance_distance(
            graph_distances,
            self.x,
            target_x,
            graph_bandwidth=self.bandwidth,
            x_scale=self.x_scale,
        )
        nuisance_local = _localized_weights(
            combined,
            bandwidth=1.0,
            min_neighbors=self.min_neighbors,
            mode=self.localization_mode,
        )
        if nuisance_local.n_positive == 0:
            nu_y = float(np.mean(self.outcome))
            nu_r = float(np.mean(self.treatment_r))
        else:
            nu_y = _weighted_mean_with_shrinkage(
                self.outcome,
                nuisance_local.weights,
                shrinkage=self.nuisance_shrinkage,
            )
            nu_r = _weighted_mean_with_shrinkage(
                self.treatment_r,
                nuisance_local.weights,
                shrinkage=self.nuisance_shrinkage,
            )
        dr_beta = _weighted_residual_lasso(
            treatment_residual=self.treatment_residual,
            outcome_residual=self.outcome_residual,
            weights=weights,
            lasso_alpha=self.lasso_alpha,
        )
        gram = float(
            np.sum(weights * self.treatment_residual * self.treatment_residual)
        )
        minimum_neff = 2.0
        wlasso_supported = bool(
            local.effective_sample_size >= minimum_neff
            and raw_variance > 1.0e-10
            and np.all(np.isfinite(wlasso_alpha))
        )
        dr_supported = bool(
            local.effective_sample_size >= minimum_neff
            and gram > 1.0e-10
            and math.isfinite(dr_beta)
        )

        direct_debiased = float("nan")
        direct_se = float("nan")
        if dr_supported:
            direction = 2.0  # Z(1)-Z(0) for the nonconstant p=1 Walsh feature.
            gamma = direction / gram
            regression_error = (
                self.outcome_residual - self.treatment_residual * dr_beta
            )
            influence = gamma * self.treatment_residual * regression_error
            direct_debiased = float(
                direction * dr_beta + np.sum(weights * influence)
            )
            direct_se = float(
                np.sqrt(np.sum(np.square(weights * influence)))
            )

        return _TargetFit(
            wlasso_alpha=wlasso_alpha,
            dr_beta=dr_beta,
            nu_y=nu_y,
            nu_r=nu_r,
            weights=weights,
            effective_sample_size=local.effective_sample_size,
            local_radius=local.local_radius,
            used_adaptive_knn=local.used_adaptive_knn,
            n_positive=local.n_positive,
            weighted_raw_variance=raw_variance,
            weighted_gram=gram,
            wlasso_supported=wlasso_supported,
            drlasso_supported=dr_supported,
            direct_debiased=direct_debiased,
            direct_standard_error=direct_se,
        )

    def predict_query_effects(
        self,
        *,
        low_configuration: RootedConfiguration,
        high_configuration: RootedConfiguration,
        queries: np.ndarray,
        target_x: float,
    ) -> Dict[str, object]:
        query_array = np.asarray(queries, dtype=np.float64)
        if query_array.shape != (3, 4):
            raise ValueError("queries must have shape [3,4].")
        low = self.fit_at(low_configuration, target_x=target_x)
        high = self.fit_at(high_configuration, target_x=target_x)

        predictions = {ITE_METHOD_NAME: np.full(3, np.nan, dtype=np.float64)}
        target_fits = {"low": low, "high": high}
        predictions[ITE_METHOD_NAME][0] = low.direct_debiased
        predictions[ITE_METHOD_NAME][1] = (
            high.drlasso_response(float(query_array[1, 0]))
            - low.drlasso_response(float(query_array[1, 2]))
        )
        predictions[ITE_METHOD_NAME][2] = (
            high.drlasso_response(float(query_array[2, 0]))
            - low.drlasso_response(float(query_array[2, 2]))
        )

        lower = float("nan")
        upper = float("nan")
        if math.isfinite(low.direct_debiased) and math.isfinite(
            low.direct_standard_error
        ):
            lower = low.direct_debiased - 1.96 * low.direct_standard_error
            upper = low.direct_debiased + 1.96 * low.direct_standard_error
        return {
            "predictions": predictions,
            "diagnostics": {
                "low": target_fits["low"].diagnostics(),
                "high": target_fits["high"].diagnostics(),
                "low_high_configuration_distance": rooted_configuration_distance(
                    low_configuration,
                    high_configuration,
                    radius=self.radius,
                ),
                "direct_ci_lower": lower,
                "direct_ci_upper": upper,
                "direct_standard_error": low.direct_standard_error,
            },
        }


def evaluate_rooted_effects(
    *,
    adjacency: np.ndarray,
    treatment: np.ndarray | Sequence[float],
    x: np.ndarray | Sequence[float],
    outcome: np.ndarray | Sequence[float],
    low_configuration: RootedConfiguration,
    high_configuration: RootedConfiguration,
    queries: np.ndarray,
    target_x: float,
    bandwidth: float,
    min_neighbors: int,
    localization_mode: str,
    lasso_alpha: float,
    nuisance_ridge: float,
    cross_fit_folds: int = 5,
    radius: int = 1,
) -> Dict[str, object]:
    estimator = PaperLocalizedWalshEstimator.fit(
        adjacency=adjacency,
        treatment=treatment,
        x=x,
        outcome=outcome,
        bandwidth=bandwidth,
        min_neighbors=min_neighbors,
        localization_mode=localization_mode,
        lasso_alpha=lasso_alpha,
        nuisance_ridge=nuisance_ridge,
        cross_fit_folds=cross_fit_folds,
        radius=radius,
    )
    return estimator.predict_query_effects(
        low_configuration=low_configuration,
        high_configuration=high_configuration,
        queries=queries,
        target_x=target_x,
    )


def summarize_effect_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    methods: Iterable[str],
    effects: Iterable[str] = EFFECT_NAMES,
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    result: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    methods = tuple(methods)
    for effect in effects:
        selected = [row for row in rows if row["effect"] == effect]
        result[effect] = {}
        for method in methods:
            truths: list[float] = []
            estimates: list[float] = []
            for row in selected:
                value = row.get(method)
                if value is None:
                    continue
                estimate = float(value)
                if not math.isfinite(estimate):
                    continue
                estimates.append(estimate)
                truths.append(float(row["truth"]))
            support_rate = len(estimates) / max(len(selected), 1)
            if not estimates:
                result[effect][method] = {
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "bias": float("nan"),
                    "average_true_effect": float("nan"),
                    "average_pred_effect": float("nan"),
                    "support_rate": support_rate,
                    "n_supported": 0,
                }
                continue
            estimate_array = np.asarray(estimates, dtype=np.float64)
            truth_array = np.asarray(truths, dtype=np.float64)
            error = estimate_array - truth_array
            result[effect][method] = {
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "average_true_effect": float(np.mean(truth_array)),
                "average_pred_effect": float(np.mean(estimate_array)),
                "support_rate": support_rate,
                "n_supported": len(estimates),
            }
    return result


def _summarize_direct_inference(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, float | int]:
    selected = [row for row in rows if row["effect"] == "direct"]
    covered: list[float] = []
    widths: list[float] = []
    standard_errors: list[float] = []
    for row in selected:
        lower = float(row.get("direct_ci_lower", float("nan")))
        upper = float(row.get("direct_ci_upper", float("nan")))
        standard_error = float(row.get("direct_standard_error", float("nan")))
        truth = float(row["truth"])
        if not all(math.isfinite(value) for value in (lower, upper, standard_error)):
            continue
        covered.append(float(lower <= truth <= upper))
        widths.append(upper - lower)
        standard_errors.append(standard_error)
    if not covered:
        return {
            "coverage_95": float("nan"),
            "average_interval_width": float("nan"),
            "average_standard_error": float("nan"),
            "n_intervals": 0,
        }
    return {
        "coverage_95": float(np.mean(covered)),
        "average_interval_width": float(np.mean(widths)),
        "average_standard_error": float(np.mean(standard_errors)),
        "n_intervals": len(covered),
    }


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _posterior_mean(model: object, batch: Mapping[str, object]) -> np.ndarray:
    import torch

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            predictions = model(batch["tokens"], batch["queries"], batch["adjacency"])
    finally:
        model.train(was_training)
    pi = predictions["gmm_pi"]
    mu = predictions["gmm_mu"]
    return (pi * mu).sum(dim=-1).detach().cpu().numpy()


def _query_truth(
    query_effect: np.ndarray,
    *,
    dataset: int,
    unit: int,
    query_offset: int,
) -> np.ndarray:
    if query_effect.ndim != 3 or query_effect.shape[-1] != 3:
        raise ValueError("query_effect must have shape [datasets, units_or_queries, 3].")
    if query_effect.shape[1] > unit:
        return query_effect[dataset, unit]
    if query_effect.shape[1] > query_offset:
        return query_effect[dataset, query_offset]
    raise ValueError("query_effect does not contain the requested query unit.")



def _sample_majority_target_fits(
    *,
    estimator: PaperLocalizedWalshEstimator,
    adjacency: np.ndarray,
    treatment: np.ndarray,
    root: int,
    target_x: float,
    treatment_prob: float,
    arm: int,
    arm_samples: int,
    seed: int,
) -> list[_TargetFit]:
    if arm_samples <= 0:
        raise ValueError("arm_samples must be positive.")
    degree = int(np.sum(adjacency[int(root)] > 0))
    # q_ik(s), with T_i in {0,1}; s denotes the majority arm, not Walsh coding.
    probabilities = conditional_count_weights(degree, treatment_prob, arm)
    rng = np.random.default_rng(int(seed))
    counts = rng.choice(
        np.arange(degree + 1, dtype=np.int64),
        size=int(arm_samples),
        replace=True,
        p=probabilities,
    )
    fits: list[_TargetFit] = []
    for sample_index, count in enumerate(counts.tolist()):
        configuration = build_exact_count_configuration(
            adjacency,
            treatment,
            root=int(root),
            treated_neighbor_count=int(count),
            seed=int(seed) * 1009 + sample_index * 9176 + int(count),
        )
        fits.append(estimator.fit_at(configuration, target_x=float(target_x)))
    return fits


def _arm_response(fits: Sequence[_TargetFit], treatment: float) -> float:
    if not fits or not all(fit.drlasso_supported for fit in fits):
        return float("nan")
    values = np.asarray(
        [fit.drlasso_response(float(treatment)) for fit in fits],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return float("nan")
    # Counts were already sampled under q_ik(s): no second probability factor.
    return float(values.mean())


def _direct_arm_effect(fits: Sequence[_TargetFit]) -> float:
    if not fits or not all(fit.drlasso_supported for fit in fits):
        return float("nan")
    values = np.asarray([fit.direct_debiased for fit in fits], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return float("nan")
    # Counts were already sampled under q_ik(s): no second probability factor.
    return float(values.mean())


def _fit_diagnostics(fits: Sequence[_TargetFit]) -> tuple[float, float]:
    if not fits:
        return 0.0, float("nan")
    neff = min(float(fit.effective_sample_size) for fit in fits)
    radii = [float(fit.local_radius) for fit in fits if math.isfinite(fit.local_radius)]
    radius = max(radii) if radii else float("nan")
    return neff, radius


def evaluate_localized_dr_lasso_majority_ite(
    *,
    batch: Mapping[str, object],
    treatment_prob: float,
    bandwidth: float,
    min_neighbors: int,
    localization_mode: str,
    lasso_alpha: float,
    nuisance_ridge: float,
    cross_fit_folds: int,
    arm_samples: int = 16,
    seed: int = 2026,
) -> Dict[str, object]:
    """Estimate majority-arm ITEs with Localized DR-Lasso.

    Each factual unit is a root.  The coarse majority arm is integrated by
    conditional randomization: sample exact treated-neighbor counts from the
    Binomial design conditional on the arm, then sample uniformly among neighbor
    subsets with that count.  Strict support is used: if any sampled target fit
    needed for an effect is unsupported, that effect is returned as NaN.
    """

    required = {
        "adjacency", "x", "observed_treatment", "observed_exposure",
        "y_obs", "oracle_ite",
    }
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"shared test batch is missing keys: {sorted(missing)}")
    arrays = {key: _to_numpy(batch[key]) for key in required}
    adjacency = arrays["adjacency"].astype(np.uint8, copy=False)
    treatment = arrays["observed_treatment"].astype(np.uint8, copy=False)
    oracle = arrays["oracle_ite"].astype(np.float64, copy=False)
    num_datasets, n_units = adjacency.shape[:2]
    if oracle.shape != (num_datasets, n_units, 3):
        raise ValueError("oracle_ite must have shape [datasets, units, 3].")

    rows: list[Dict[str, object]] = []
    localization_geometry: list[Dict[str, object]] = []
    localization_warnings: list[str] = []
    for dataset in range(num_datasets):
        estimator = PaperLocalizedWalshEstimator.fit(
            adjacency=adjacency[dataset],
            treatment=treatment[dataset],
            x=arrays["x"][dataset],
            outcome=arrays["y_obs"][dataset],
            bandwidth=bandwidth,
            min_neighbors=min_neighbors,
            localization_mode=localization_mode,
            lasso_alpha=lasso_alpha,
            nuisance_ridge=nuisance_ridge,
            cross_fit_folds=cross_fit_folds,
            radius=1,
        )
        distances = estimator.pairwise_distance
        off_diagonal = distances[~np.eye(n_units, dtype=bool)]
        geometry = {
            "dataset_id": dataset + 1,
            "scope": "factual_to_factual_distance_matrix",
            "maximum_graph_distance": float(distances.max()),
            "fraction_nonself_at_distance_ceiling": float(np.mean(off_diagonal == .25)),
            "minimum_to_maximum_raw_kernel_weight": None,
            "near_uniform_kernel": False,
        }
        if localization_mode == "kernel":
            raw = np.maximum(1. - np.square(distances / bandwidth), 0.)
            ratio = float(raw.min() / raw.max())
            geometry["minimum_to_maximum_raw_kernel_weight"] = ratio
            # Descriptive warning only: .95 is not a support or inference test.
            geometry["near_uniform_kernel"] = ratio >= .95
            if geometry["near_uniform_kernel"]:
                localization_warnings.append(
                    f"Dataset {dataset + 1}: Localized DR-Lasso kernel is nearly uniform "
                    f"(raw min/max={ratio:.6f}, bandwidth={bandwidth:g}). "
                    "Factual-to-factual geometry only; inspect per-query support separately. "
                    "Large effective sample size does not imply exposure discrimination."
                )
        localization_geometry.append(geometry)
        degrees = adjacency[dataset].sum(axis=1)
        for unit in range(n_units):
            unit_seed = int(seed) + dataset * 1_000_003 + unit * 10_007
            low_fits = _sample_majority_target_fits(
                estimator=estimator,
                adjacency=adjacency[dataset],
                treatment=treatment[dataset],
                root=unit,
                target_x=float(arrays["x"][dataset, unit]),
                treatment_prob=treatment_prob,
                arm=0,
                arm_samples=arm_samples,
                seed=unit_seed,
            )
            high_fits = _sample_majority_target_fits(
                estimator=estimator,
                adjacency=adjacency[dataset],
                treatment=treatment[dataset],
                root=unit,
                target_x=float(arrays["x"][dataset, unit]),
                treatment_prob=treatment_prob,
                arm=1,
                arm_samples=arm_samples,
                seed=unit_seed + 7919,
            )
            direct = _direct_arm_effect(low_fits)
            low_treated = _arm_response(low_fits, 1.0)
            high_treated = _arm_response(high_fits, 1.0)
            low_untreated = _arm_response(low_fits, 0.0)
            spillover = (
                high_treated - low_treated
                if math.isfinite(high_treated) and math.isfinite(low_treated)
                else float("nan")
            )
            total = (
                high_treated - low_untreated
                if math.isfinite(high_treated) and math.isfinite(low_untreated)
                else float("nan")
            )
            estimates = (direct, spillover, total)
            low_neff, low_radius = _fit_diagnostics(low_fits)
            high_neff, high_radius = _fit_diagnostics(high_fits)
            for effect_index, effect_name in enumerate(EFFECT_NAMES):
                estimate = float(estimates[effect_index])
                needs_high = effect_index in (1, 2)
                relevant_neff = min(low_neff, high_neff) if needs_high else low_neff
                relevant_radius = (
                    max(value for value in (low_radius, high_radius) if math.isfinite(value))
                    if needs_high and any(math.isfinite(v) for v in (low_radius, high_radius))
                    else low_radius
                )
                rows.append(
                    {
                        "dataset_id": dataset + 1,
                        "query_unit_id": unit + 1,
                        "effect": effect_name,
                        "truth": float(oracle[dataset, unit, effect_index]),
                        ITE_METHOD_NAME: estimate,
                        "supported": bool(math.isfinite(estimate)),
                        "effective_sample_size": float(relevant_neff),
                        "local_radius": float(relevant_radius),
                        "low_neff": float(low_neff),
                        "high_neff": float(high_neff),
                        "low_local_radius": float(low_radius),
                        "high_local_radius": float(high_radius),
                        "degree": int(degrees[unit]),
                        "observed_treatment": int(treatment[dataset, unit]),
                        "observed_exposure": float(arrays["observed_exposure"][dataset, unit]),
                        "x": float(arrays["x"][dataset, unit]),
                    }
                )

    return {
        "evaluation": "majority-arm individualized effects with strict local support",
        "method": ITE_METHOD_NAME,
        "implementation_protocol": LOCALIZED_PROTOCOL,
        "inference_scope": "empirical ITE point estimation; no original-theorem guarantees asserted",
        "num_datasets": num_datasets,
        "num_observed_units_per_dataset": n_units,
        "num_queried_units_per_dataset": n_units,
        "num_evaluated_units_per_dataset": n_units,
        "n_targets_per_effect": num_datasets * n_units,
        "configuration_radius": 1,
        "localization_geometry": localization_geometry,
        "localization_warnings": localization_warnings,
        "localization_mode": localization_mode,
        "bandwidth": bandwidth,
        "min_neighbors": min_neighbors,
        "lasso_alpha": lasso_alpha,
        "nuisance_ridge": nuisance_ridge,  # legacy key
        "nuisance_shrinkage": nuisance_ridge,
        "nuisance_mean_scope": "training fold only during cross-fitting",
        "cross_fit_folds": cross_fit_folds,
        "arm_samples": int(arm_samples),
        "methods": LOCALIZED_METHOD_NAMES,
        "effects": summarize_effect_rows(rows, methods=LOCALIZED_METHOD_NAMES),
        "query_units": rows,
        "estimand_bridge": {
            "coarse_arm": "S_i=1{treated-neighbor count > floor(degree/2)}",
            "integration": "sample k from q_ik(s), sample uniform size-k neighbor subset, plain MC mean",
            "treatment_coding": "T_i in {0,1}; R_i=2*T_i-1",
            "direct_estimator": "mean debiased delta_i^Direct over low-arm configurations",
            "response_estimator": "nu_Y(g,X_i)+(2*t-1-nu_R(g,X_i))*beta(g)",
            "direct_estimand": "mu_i(1,0)-mu_i(0,0)",
            "spillover_estimand": "mu_i(1,1)-mu_i(1,0)",
            "total_estimand": "mu_i(1,1)-mu_i(0,0)",
        },
    }
