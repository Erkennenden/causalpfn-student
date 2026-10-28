"""E1.4: radius graphs in [0,1]^2, with the existing response/assignment DGP.

Graphs are conditioned on having no isolates because the frozen estimator's
majority-exposure contract requires positive degree. No edges are repaired;
disconnected graphs are allowed. Positions are simulator metadata, not inputs.
"""
from dataclasses import asdict, dataclass
import json

import networkx as nx
import numpy as np

from pfn_social.contracts import arm_mean_exposure
from pfn_social.phase3_hierarchical import (
    Setting, _components, _hash_arrays, coefficient_prior, hidden_features,
)
from pfn_social.phase5_large_network import (
    LargeNetworkMechanism, common_projection, rng_for,
)
from pfn_social.phase6_limited_outcomes import LimitedOutcomeMechanism


@dataclass(frozen=True)
class RGGConfig:
    distribution: str
    radius: float
    setting_id: int
    n: int = 200
    split_seed: int = 2026102704
    base_setting_id: int = 0
    split: str = 'dev'
    cluster_sd: float = .12
    max_attempts: int = 10000

    def __post_init__(self):
        if self.distribution not in ('uniform', 'clustered'):
            raise ValueError('Unknown spatial distribution')
        if not 0 < self.radius <= np.sqrt(2) or self.n != 200:
            raise ValueError('Require N=200 and a positive radius <= sqrt(2)')
        if self.setting_id < 7 or self.split_seed < 0 or self.cluster_sd <= 0 or self.max_attempts < 1:
            raise ValueError('Invalid RGG configuration')

    @property
    def setting(self):
        return Setting(f'rgg_{self.distribution}_r{self.radius:g}', self.n,
                       'rgg', 0, 1., 'shared_nonlinear')

    def canonical_json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))


SCENARIOS = tuple(RGGConfig(distribution, radius, 7 + 3*d + k)
                  for d, distribution in enumerate(('uniform', 'clustered'))
                  for k, radius in enumerate((.12, .18, .24)))


def radius_graph(positions, radius):
    positions = np.asarray(positions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or not np.isfinite(positions).all() or np.any((positions < 0) | (positions > 1))
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Require finite positions[N,2] in the unit square and radius > 0')
    distance2 = np.sum((positions[:, None] - positions[None, :])**2, axis=-1)
    adjacency = (distance2 <= radius**2).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def draw_positions(config, rng):
    if config.distribution == 'uniform':
        return rng.uniform(size=(config.n, 2))
    # Equal-probability cluster membership; Gaussian points truncated to the
    # square by resampling within the same cluster (no clipping at boundaries).
    centers = np.array([[.3, .3], [.7, .7]])[rng.integers(0, 2, config.n)]
    positions = centers + config.cluster_sd*rng.normal(size=(config.n, 2))
    outside = np.any((positions < 0) | (positions > 1), axis=1)
    while outside.any():
        positions[outside] = centers[outside] + config.cluster_sd*rng.normal(size=(outside.sum(), 2))
        outside = np.any((positions < 0) | (positions > 1), axis=1)
    return positions


def generate_rgg(config, mechanism_id):
    """Return a LimitedOutcomeMechanism plus auditable spatial metadata."""
    rng = rng_for(config, mechanism_id, 'graph')
    for attempt in range(1, config.max_attempts + 1):
        positions = draw_positions(config, rng)
        a = radius_graph(positions, config.radius)
        if np.all(a.sum(1) > 0):
            break
    else:
        raise RuntimeError('No isolate-free graph found; do not silently repair edges')
    population = rng_for(config, mechanism_id, 'population').normal(0, .8, 20)
    x = (population + .6*rng_for(config, mechanism_id, 'x').normal(size=(config.n, 20))).astype(np.float32)
    projection = common_projection().copy()
    h = hidden_features(x, projection, config.setting.representation)
    mean, covariance = coefficient_prior()
    beta = rng_for(config, mechanism_id, 'coefficient').multivariate_normal(mean, covariance)
    components = _components(a, h, beta)
    b, tau, gamma, eta = components.T
    degree = a.sum(1)
    hbar = a @ h / degree[:, None]
    truth = np.empty((config.n, 4))
    for own, arm in ((0, 0), (0, 1), (1, 0), (1, 1)):
        exposure = np.array([arm_mean_exposure(int(d), .5, arm) for d in degree])
        truth[:, 2*own + arm] = (b + own*tau + exposure*(gamma + own*eta
                                 + hbar @ (beta[19:22] + own*beta[22:25])))
    identity = np.frombuffer(config.canonical_json().encode(), dtype=np.uint8)
    parent = LargeNetworkMechanism(config, mechanism_id, a, x, projection, h, beta,
        components, truth, population, _hash_arrays(a, x, beta, positions, identity), _hash_arrays(truth))
    # Independent of graph acceptance, treatment and noise; nested fixed masks
    # across repeats follow the upstream design. RNG channel remains separate.
    mask_rng = np.random.default_rng(np.random.SeedSequence(
        [config.split_seed, 2026102708, config.setting_id, mechanism_id]))
    mechanism = LimitedOutcomeMechanism(parent, mask_rng.permutation(config.n))
    graph = nx.from_numpy_array(a)
    metadata = dict(family=config.setting_id, mechanism=mechanism_id,
        distribution=config.distribution, radius=config.radius, n=config.n,
        attempts=attempt, edges=graph.number_of_edges(), density=nx.density(graph),
        mean_degree=float(degree.mean()), min_degree=int(degree.min()), max_degree=int(degree.max()),
        components=nx.number_connected_components(graph), clustering=nx.average_clustering(graph),
        mechanism_sha256=parent.mechanism_sha256, truth_sha256=parent.truth_sha256)
    return mechanism, positions, metadata
