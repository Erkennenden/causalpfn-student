"""CausalFM outcome primitives adapted to f(x, neighbor_x, t, e), without U.

Vendored classes sample the actual networks and parameters. Evaluation below
only vectorizes their deterministic, zero-internal-noise forward propagation.
The front-door *dataset* wrappers are intentionally not called.
"""
from dataclasses import dataclass
from numbers import Integral
from threading import RLock

import numpy as np

from ._upstream_outcome import OutcomeGenerator
from ._upstream_base import BaseMLPGenerator
from ._upstream_frontdoor import MediatorGenerator, OutcomeGeneratorWithMediator

INPUT_NAMES = ("x", "neighbor_x", "t", "e")
OUTCOME_FAMILIES = ("standard", "dense", "composition")
_SOURCE_RNG_LOCK = RLock()


def _inputs(z, names):
    value = np.asarray(z, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != len(names) or not np.isfinite(value).all():
        raise ValueError(f"f requires finite inputs with shape [..., {len(names)}]: {names}.")
    return value


@dataclass(frozen=True)
class RandomOutcomeFunction:
    """One sampled, deterministic MLP; read-only matrices use original parameters."""
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    seed: int
    num_layers: int
    hidden_size: int
    input_names: tuple[str, ...] = INPUT_NAMES
    family: str = "standard"
    source_class: str = "OutcomeGenerator"
    edge_drop_prob: float = .4

    def __call__(self, z):
        value = _inputs(z, self.input_names)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight + bias
            if i < len(self.weights) - 1:
                value = np.tanh(value)
        return value[..., 0]

    def to_dict(self):
        return dict(seed=self.seed, family=self.family, source_class=self.source_class,
                    num_layers=self.num_layers, hidden_size=self.hidden_size,
                    input_names=list(self.input_names), activation="tanh", output_activation="identity",
                    edge_drop_prob=self.edge_drop_prob, internal_noise=False,
                    weights=[w.tolist() for w in self.weights], biases=[b.tolist() for b in self.biases])


@dataclass(frozen=True)
class CompositeOutcomeFunction:
    """Two fixed networks; intermediate values are calculations, never latent data."""
    first: RandomOutcomeFunction
    second: RandomOutcomeFunction
    seed: int
    family: str

    def __call__(self, z):
        value = _inputs(z, INPUT_NAMES)
        first = self.first(value)
        if self.family == "composition":
            return self.second(np.concatenate((value[..., :2], first[..., None]), axis=-1))
        raise ValueError(f"Unknown composite family: {self.family}")

    def to_dict(self):
        formula = "g(x, neighbor_x, h(x, neighbor_x, t, e))"
        return dict(seed=self.seed, family=self.family, input_names=list(INPUT_NAMES), formula=formula,
                    internal_noise=False, first=self.first.to_dict(), second=self.second.to_dict())


def _vectorize(source, graph, *, seed, layers, width, names, family, edge_drop_prob):
    nodes_by_layer = [[n for n in graph if graph.nodes[n]["layer"] == i] for i in range(layers)]
    weights, biases = [], []
    for parents, children in zip(nodes_by_layer, nodes_by_layer[1:]):
        weight = np.array([[source.weights.get(child, {}).get(parent, 0.) for child in children]
                           for parent in parents], dtype=np.float64)
        bias = np.array([source.biases[child] for child in children], dtype=np.float64)
        weight.setflags(write=False)
        bias.setflags(write=False)
        weights.append(weight)
        biases.append(bias)
    return RandomOutcomeFunction(tuple(weights), tuple(biases), seed, layers, width,
                                 tuple(names), family, type(source).__name__, edge_drop_prob)


def _sample_sparse(source, seed, names, family):
    layers, width = max(3, int(source.prior_layers())), int(source.prior_hidden_size())
    source.outcome_network = source.construct_outcome_network(layers, width, len(names))
    source.sample_outcome_network_parameters()
    return _vectorize(source, source.outcome_network, seed=seed, layers=layers, width=width,
                      names=names, family=family, edge_drop_prob=source.edge_drop_prob)


def _sample_dense(source, seed, names, family):
    layers, width = max(3, int(source.prior_layers())), int(source.prior_hidden_size())
    source.network = source._construct_network(layers, width, len(names), 1)
    source._sample_network_parameters()
    return _vectorize(source, source.network, seed=seed, layers=layers, width=width,
                      names=names, family=family, edge_drop_prob=0.)


def sample_outcome_function(seed: int, *, family: str = "mixture"):
    """Draw one fixed f. Mixture selects each of three families with probability 1/3.

    Standard/dense networks use 3..5 total layers, widths 10..24.
    The composition's inner network uses the source's factual-mediator prior:
    3..4 layers, widths 10..24; its outer network uses the standard sparse prior.
    All weights/biases are N(0,1), hidden activations tanh, output linear.

    Original samplers use NumPy's global RNG; save/restore it under a lock.
    Other threads that manipulate np.random directly must not run concurrently.
    Selecting a family uses a separate RNG, preserving old standard(seed) draws.
    """
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("outcome seed must be an integer in [0, 2**32).")
    if family not in (*OUTCOME_FAMILIES, "mixture"):
        raise ValueError(f"Unknown outcome family: {family!r}; choose {OUTCOME_FAMILIES} or mixture.")
    seed = int(seed)
    if family == "mixture":
        family = OUTCOME_FAMILIES[np.random.RandomState(seed ^ 0xCA05A1).randint(len(OUTCOME_FAMILIES))]
    with _SOURCE_RNG_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            priors = dict(prior_layers=lambda: np.random.randint(3, 6),
                          prior_hidden_size=lambda: np.random.randint(10, 25))
            if family == "standard":
                return _sample_sparse(OutcomeGenerator(**priors, edge_drop_prob=.4), seed, INPUT_NAMES, family)
            if family == "dense":
                return _sample_dense(BaseMLPGenerator(**priors, use_layer_noise=False), seed, INPUT_NAMES, family)
            source = MediatorGenerator(prior_layers=lambda: np.random.randint(3, 5),
                                       prior_hidden_size=lambda: np.random.randint(10, 25))
            layers, width = int(source.prior_layers()), int(source.prior_hidden_size())
            source.mediator_network = source.construct_mediator_network(layers, width, len(INPUT_NAMES))
            source.sample_mediator_network_parameters()
            first = _vectorize(source, source.mediator_network, seed=seed, layers=layers, width=width,
                               names=INPUT_NAMES, family="dense", edge_drop_prob=0.)
            second = _sample_sparse(OutcomeGeneratorWithMediator(**priors, edge_drop_prob=.4), seed,
                                    ("x", "neighbor_x", "h"), "standard")
            return CompositeOutcomeFunction(first, second, seed, family)
        finally:
            np.random.set_state(state)
