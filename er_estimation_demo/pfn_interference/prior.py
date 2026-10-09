"""Random SCM in observed standard-normal X with keyed task/component streams.

Only the legacy graph samplers and majority estimands are reused. Every task
shares three independently drawn functions; function normalization uses the
fixed Gaussian reference, never the realized X or any evaluation labels.
All randomness is local to CPU generators, including graph rejection sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Iterable

import numpy as np
import torch

from pfn_interference.estimands import oracle_ate, oracle_ite_from_parameters, sample_majority_exposures
from pfn_interference.model import (
    DataConfig, GRAPH_FAMILY_ORDER, GRAPH_TYPE_NAMES, TensorBatch,
    _generate_configuration_adjacency, _generate_er_adjacency,
    _generate_rgg_adjacency, _generate_sbm_adjacency,
)


PRIOR_VERSION = "random_lpe_two_regime_v3_gh256"
# Keep unrelated graph, parameter, treatment and noise draws stable across the X change.
RNG_NAMESPACE = "c_prior_v1_gh256"
FUNCTION_FAMILIES = ("linear", "polynomial", "fourier", "tanh", "exponential")
FUNCTION_CODES = {name: index for index, name in enumerate((*FUNCTION_FAMILIES, "spline"))}
REFERENCE_POINTS = 256
MIN_REFERENCE_VARIANCE = 1e-10
MAX_FUNCTION_ATTEMPTS = 100

# NumPy produces the physicists' Hermite rule, converted to standard Normal.
_roots, _weights = np.polynomial.hermite.hermgauss(REFERENCE_POINTS)
_REFERENCE_NODES = torch.from_numpy(_roots * math.sqrt(2.0))
_REFERENCE_WEIGHTS = torch.from_numpy(_weights / math.sqrt(math.pi))


@dataclass(frozen=True)
class PriorConfig:
    n_units: int = 1000
    graph_family: str = "mixed"
    treatment_prob: float = 0.5
    nointerference_prob: float = 0.5
    target_mean_degree_min: float = 4.0
    target_mean_degree_max: float = 10.0
    function_families: tuple[str, ...] = FUNCTION_FAMILIES
    noise_min: float = 0.5
    noise_max: float = 6.0
    config_version: str = PRIOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.n_units, Integral) or isinstance(self.n_units, bool) or self.n_units < 12:
            raise ValueError("n_units must be an integer at least 12; use 300 for the C design.")
        if self.config_version != PRIOR_VERSION:
            raise ValueError(f"config_version must be {PRIOR_VERSION!r}.")
        if self.graph_family not in ("mixed", *GRAPH_TYPE_NAMES.values()):
            raise ValueError("Unsupported graph_family.")
        if self.treatment_prob != 0.5:
            raise ValueError("treatment_prob is fixed to iid Bernoulli(0.5) in C v1.")
        for field in ("nointerference_prob",):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be finite and in [0,1].")
        for field in ("noise_min", "noise_max", "target_mean_degree_min", "target_mean_degree_max"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite.")
        if self.noise_min > self.noise_max:
            raise ValueError("noise_min must not exceed noise_max.")
        if self.target_mean_degree_min > self.target_mean_degree_max:
            raise ValueError("target_mean_degree_min must not exceed target_mean_degree_max.")
        object.__setattr__(self, "function_families", tuple(self.function_families))
        if (not self.function_families or len(set(self.function_families)) != len(self.function_families)
                or not set(self.function_families).issubset(FUNCTION_FAMILIES)):
            raise ValueError("function_families must be distinct training families; spline is held out.")
        low, high = self.effective_degree_bounds()
        if low > high:
            raise ValueError("target_mean_degree_min exceeds the explicit small-graph smoke cap.")
        # Validate all family calibrations, including controlled family overrides.
        n = self.n_units
        b0, b1 = (n+1)//2, n//2
        within = b0*(b0-1)+b1*(b1-1)
        outside = 2*b0*b1*self.graph_config().sbm_between_probability
        if low*n <= outside or high*n >= within+outside:
            raise ValueError("target mean degree bounds cannot calibrate the legacy SBM.")
        if high >= (n-1)*math.pi/4:
            raise ValueError("target mean degree bounds cannot calibrate the toroidal RGG.")
        if high > self.graph_config().configuration_max_degree:
            raise ValueError("target_mean_degree_max exceeds the configuration degree cap.")

    def effective_degree_bounds(self) -> tuple[float, float]:
        """N<24 is a smoke regime with an explicit, reported graph density cap."""
        high = self.target_mean_degree_max
        if self.n_units < 24:
            high = min(high, 0.4*(self.n_units-1))
        return self.target_mean_degree_min, high

    def graph_config(self) -> DataConfig:
        """Legacy graph settings for integration; C outcome fields stay separate."""
        return DataConfig(n_units=int(self.n_units), graph_family=self.graph_family,
                          treatment_prob=self.treatment_prob)


def standard_normal_reference() -> tuple[torch.Tensor, torch.Tensor]:
    """Return defensive copies of the fixed float64 256-point Gaussian rule."""
    return _REFERENCE_NODES.clone(), _REFERENCE_WEIGHTS.clone()


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low+(high-low)*float(torch.rand((), generator=generator, dtype=torch.float64))


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (), generator=generator))


def _signed_uniform(generator: torch.Generator, low: float, high: float) -> float:
    sign = 1 if _randint(generator, 0, 2) else -1
    return sign*_uniform(generator, low, high)


def _task_generator(seed: int, stream: str, task_id: int, channel: str) -> torch.Generator:
    identity = json.dumps([RNG_NAMESPACE, int(seed), stream, int(task_id), channel],
                          ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(identity, digest_size=8, person=b"c-pfn-prior-v1").digest()
    # Torch stores initial_seed as uint64, but signed63 also fits audit tensors.
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") % (2**63-1))


@dataclass(frozen=True)
class NormalizedFunction:
    family: str
    parameters: tuple[torch.Tensor, ...]
    reference_mean: float
    reference_scale: float

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        p = tuple(value.to(device=z.device, dtype=z.dtype) for value in self.parameters)
        if self.family == "linear":
            return p[0]*z
        if self.family == "polynomial":
            powers = torch.arange(1, p[0].numel()+1, device=z.device, dtype=z.dtype)
            return (z[..., None].pow(powers)*p[0]).sum(-1)
        if self.family == "fourier":
            coefficients, frequencies, phases = p
            return (torch.sin(2*math.pi*z[..., None]*frequencies+phases)*coefficients).sum(-1)
        if self.family == "tanh":
            coefficients, centers, widths = p
            return (torch.tanh((z[..., None]-centers)/widths)*coefficients).sum(-1)
        if self.family == "exponential":
            return p[0]*torch.exp(p[1]*z)
        if self.family == "spline":
            knots, heights, slopes = p
            indices = torch.searchsorted(knots, z.contiguous(), right=True).sub(1).clamp(0, knots.numel()-2)
            left, right = knots[indices], knots[indices+1]
            width = right-left
            u = (z-left)/width
            interior = ((2*u**3-3*u**2+1)*heights[indices]
                        +(u**3-2*u**2+u)*width*slopes[indices]
                        +(-2*u**3+3*u**2)*heights[indices+1]
                        +(u**3-u**2)*width*slopes[indices+1])
            return torch.where(z < knots[0], heights[0]+slopes[0]*(z-knots[0]),
                   torch.where(z > knots[-1], heights[-1]+slopes[-1]*(z-knots[-1]), interior))
        raise ValueError(f"Unknown function family: {self.family}")

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if not z.dtype.is_floating_point:
            raise ValueError("Function inputs must be floating-point tensors.")
        # Normalize in float64 even when returning float32 SCM values.
        value = (self.raw(z.to(torch.float64))-self.reference_mean)/self.reference_scale
        return value.to(z.dtype)


def sample_function(family: str, generator: torch.Generator) -> NormalizedFunction:
    """Sample a shared random function, normalized solely under fixed N(0,1).

    Spline is a held-out C1 cubic Hermite family with 4..8 equally spaced
    knots on [-2.5,2.5], independent N(0,1) heights and linear extrapolation.
    Degenerate/nonfinite reference draws are retried locally, never clipped.
    """
    if family not in FUNCTION_CODES:
        raise ValueError(f"Unknown function family: {family}")
    for _ in range(MAX_FUNCTION_ATTEMPTS):
        if family == "linear":
            params = (torch.tensor(1. if _randint(generator, 0, 2) else -1., dtype=torch.float64),)
        elif family == "polynomial":
            params = (torch.randn(_randint(generator, 2, 4), generator=generator, dtype=torch.float64),)
        elif family == "fourier":
            count = _randint(generator, 1, 5)
            params = (torch.randn(count, generator=generator, dtype=torch.float64),
                      .25+1.25*torch.rand(count, generator=generator, dtype=torch.float64),
                      2*math.pi*torch.rand(count, generator=generator, dtype=torch.float64))
        elif family == "tanh":
            count = _randint(generator, 2, 7)
            params = (torch.randn(count, generator=generator, dtype=torch.float64),
                      -2+4*torch.rand(count, generator=generator, dtype=torch.float64),
                      torch.exp(math.log(.25)+math.log(8)*torch.rand(count, generator=generator, dtype=torch.float64)))
        elif family == "exponential":
            params = (torch.tensor(1. if _randint(generator, 0, 2) else -1., dtype=torch.float64),
                      torch.tensor(_uniform(generator, .25, 1), dtype=torch.float64))
        else:
            count = _randint(generator, 4, 9)
            knots = torch.linspace(-2.5, 2.5, count, dtype=torch.float64)
            heights = torch.randn(count, generator=generator, dtype=torch.float64)
            slopes = torch.empty_like(heights)
            slopes[0] = (heights[1]-heights[0])/(knots[1]-knots[0])
            slopes[-1] = (heights[-1]-heights[-2])/(knots[-1]-knots[-2])
            slopes[1:-1] = (heights[2:]-heights[:-2])/(knots[2:]-knots[:-2])
            params = (knots, heights, slopes)
        raw = NormalizedFunction(family, params, 0., 1.).raw(_REFERENCE_NODES)
        mean = float(_REFERENCE_WEIGHTS @ raw)
        variance = float(_REFERENCE_WEIGHTS @ (raw-mean).square())
        if torch.isfinite(raw).all() and math.isfinite(mean) and math.isfinite(variance) and variance > MIN_REFERENCE_VARIANCE:
            return NormalizedFunction(family, params, mean, math.sqrt(variance))
    raise RuntimeError(f"Could not draw nondegenerate {family} function after {MAX_FUNCTION_ATTEMPTS} attempts.")


def _draw_graph(config: PriorConfig, task_id: int, generator: torch.Generator,
                *, balanced: bool) -> tuple[torch.Tensor, int, float]:
    low, high = config.effective_degree_bounds()
    target_degree = _uniform(generator, low, high)
    # Consume the family draw regardless of overrides to keep target/graph streams stable.
    family_index = _randint(generator, 0, len(GRAPH_FAMILY_ORDER))
    if balanced:
        family_index = (task_id//4) % len(GRAPH_FAMILY_ORDER)
    elif config.graph_family != "mixed":
        family_index = tuple(GRAPH_TYPE_NAMES.values()).index(config.graph_family)
    graph_type = GRAPH_FAMILY_ORDER[family_index]
    graph_config = config.graph_config()
    target = torch.tensor([target_degree], dtype=torch.float32)
    if graph_type == GRAPH_FAMILY_ORDER[0]:
        adjacency = _generate_er_adjacency(config.n_units, 1,
            edge_probability=target/(config.n_units-1), generator=generator)
    else:
        sampler = {_graph: _sampler for _graph, _sampler in zip(GRAPH_FAMILY_ORDER[1:],
            (_generate_configuration_adjacency, _generate_rgg_adjacency, _generate_sbm_adjacency))}
        adjacency = sampler[graph_type](graph_config, 1, generator=generator, target_mean_degree=target)
    return adjacency[0], graph_type, float(target[0])


def _assemble_task(*, x: torch.Tensor, adjacency: torch.Tensor, treatment: torch.Tensor,
                   structural_baseline: torch.Tensor, tau: torch.Tensor, gamma: torch.Tensor,
                   noise: torch.Tensor, query_generator: torch.Generator) -> TensorBatch:
    degree = adjacency.sum(-1)
    exposure = torch.mv(adjacency, treatment)/degree
    y = structural_baseline+tau*treatment+gamma*exposure+noise
    low, high = sample_majority_exposures(degree, treatment_prob=.5, generator=query_generator)
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    ta = torch.stack((one, one, one), -1)
    ea = torch.stack((low, high, high), -1)
    tb = torch.stack((zero, one, zero), -1)
    eb = torch.stack((low, low, low), -1)
    query_effect = tau[:, None]*(ta-tb)+gamma[:, None]*(ea-eb)
    outcome_a = structural_baseline[:, None]+tau[:, None]*ta+gamma[:, None]*ea
    outcome_b = structural_baseline[:, None]+tau[:, None]*tb+gamma[:, None]*eb
    eta = torch.zeros_like(x)
    truth = oracle_ite_from_parameters(degree=degree, tau=tau, gamma=gamma, eta=eta, treatment_prob=.5)
    covariate_score = torch.sigmoid(x)
    return {
        "tokens": torch.stack((x, treatment, y, exposure, degree/(x.numel()-1)), -1),
        "queries": torch.stack((ta, ea, tb, eb), -1), "query_effect": query_effect,
        "adjacency": adjacency, "degree": degree, "x": x, "y_obs": y,
        "observed_treatment": treatment, "observed_exposure": exposure,
        "majority_arm": (torch.round(exposure*degree) > torch.floor(degree/2)).long(),
        "low_sampled_exposure": low, "high_sampled_exposure": high,
        "outcome_a": outcome_a, "outcome_b": outcome_b,
        "oracle_ite": truth, "oracle_ate": oracle_ate(truth),
        "structural_baseline": structural_baseline, "noise": noise,
        "tau": tau, "gamma": gamma, "eta": eta,
        "covariate_score": covariate_score,
        "neighbor_covariate_score": torch.mv(adjacency, covariate_score)/degree,
        "neighbor_x": torch.mv(adjacency, x)/degree,
    }


def generate_tasks(config: PriorConfig, task_ids: Iterable[int], *, stream: str = "train",
                   seed: int = 12345, device: torch.device | str = "cpu",
                   function_shift: bool = False, balanced: bool = False) -> TensorBatch:
    """Generate task ids deterministically, independent of batches/ranks/order.

    Both neighbor mechanisms always switch together (regime 0 or 3).
    At p0=.5 consecutive task-ID pairs have one of each, in a seeded order.
    balanced=True fixes regime=3*(id%2), graph=GRAPH_FAMILY_ORDER[(id//4)%4].
    function_shift replaces only f_tau with spline. Oracle/audit tensors are
    separate from the five observed token features and must never be inputs.
    """
    ids = list(task_ids)
    if not ids or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= 2**63 for i in ids):
        raise ValueError("task_ids must be nonempty nonnegative signed-64-bit integers.")
    if len(set(ids)) != len(ids):
        raise ValueError("task_ids must not contain duplicates.")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    tasks: list[TensorBatch] = []
    for task_id in ids:
        rng = lambda channel: _task_generator(seed, stream, task_id, channel)
        graph, graph_type, target_degree = _draw_graph(config, task_id, rng("graph"), balanced=balanced)
        # Restore fourgraph_degree's default X ~ N(0,1), directly observed.
        # No task-specific shift/scale, mixture, or latent covariate construction.
        x = torch.randn(config.n_units, generator=rng("data/normal"))
        neighbor_x = torch.mv(graph, x)/graph.sum(-1)
        parameters = rng("parameters")
        b0 = _uniform(parameters, -2, 2)
        a_b = _signed_uniform(parameters, .5, 3)
        b_n_active = _signed_uniform(parameters, .5, 4)
        a0 = _uniform(parameters, -8, 8)
        a_tau = _uniform(parameters, 0, 4)
        # (1-u) makes the active gamma strictly negative for every finite RNG draw.
        gamma_active = -2*(1-_uniform(parameters, 0, 1))
        sigma = _uniform(rng("noise/scale"), config.noise_min, config.noise_max)
        if balanced:
            active = task_id % 2 == 1
        elif config.nointerference_prob == .5:
            # One null and one active task per pair; all other task streams remain independent.
            pair_rng = _task_generator(seed, stream, task_id // 2, "gate/both_or_none")
            first_active = _uniform(pair_rng, 0, 1) >= .5
            active = first_active if task_id % 2 == 0 else not first_active
        else:
            active = _uniform(rng("gate/both_or_none"), 0, 1) >= config.nointerference_prob
        interference = neighbor = active
        gamma_value = gamma_active if interference else 0.
        b_n = b_n_active if neighbor else 0.
        functions = []
        for role in ("baseline", "neighbor", "direct"):
            family_rng = rng(f"functions/{role}/family")
            family = config.function_families[_randint(family_rng, 0, len(config.function_families))]
            if function_shift and role == "direct":
                family = "spline"
            functions.append(sample_function(family, rng(f"functions/{role}/parameters")))
        baseline_values = functions[0](x)
        neighbor_values = functions[1](neighbor_x)
        direct_values = functions[2](x)
        structural_baseline = b0+a_b*baseline_values+b_n*neighbor_values
        tau = a0+a_tau*direct_values
        noise = sigma*torch.randn(config.n_units, generator=rng("noise/realization"))
        task = _assemble_task(x=x, adjacency=graph, treatment=(torch.rand(config.n_units,
                generator=rng("treatment")) < config.treatment_prob).float(),
            structural_baseline=structural_baseline, tau=tau, gamma=torch.full_like(x, gamma_value),
            noise=noise, query_generator=rng("query"))
        task.update(prior_baseline_function=baseline_values,
                    prior_neighbor_function=neighbor_values, prior_direct_function=direct_values,
                    prior_function_family=torch.tensor([FUNCTION_CODES[f.family] for f in functions]),
                    prior_function_center=torch.tensor([f.reference_mean for f in functions], dtype=torch.float64),
                    prior_function_scale=torch.tensor([f.reference_scale for f in functions], dtype=torch.float64))
        scalars = dict(prior_baseline_intercept=b0,
            prior_baseline_amplitude=a_b, prior_neighbor_coefficient=b_n, prior_tau_level=a0,
            prior_tau_amplitude=a_tau, prior_gamma=gamma_value, prior_noise_sd=sigma,
            prior_target_mean_degree=target_degree,
            prior_effective_degree_min=config.effective_degree_bounds()[0],
            prior_effective_degree_max=config.effective_degree_bounds()[1])
        task.update({key: torch.tensor(value, dtype=torch.float32) for key, value in scalars.items()})
        integers = dict(task_id=task_id, regime=2*int(interference)+int(neighbor), graph_type=graph_type,
            star_center=-1,
            prior_interference_active=int(interference), prior_neighbor_active=int(neighbor),
            task_seed=rng("identity").initial_seed())
        task.update({key: torch.tensor(value, dtype=torch.long) for key, value in integers.items()})
        if any(not bool(torch.isfinite(value).all()) for value in task.values()):
            raise FloatingPointError(f"Nonfinite C prior tensor for stream={stream!r}, task_id={task_id}.")
        tasks.append(task)
    return {key: torch.stack([task[key] for task in tasks]).to(device) for key in tasks[0]}


def original_formula_reference(*, x: torch.Tensor, adjacency: torch.Tensor,
                               treatment: torch.Tensor, noise: torch.Tensor | None = None,
                               seed: int = 12345) -> TensorBatch:
    """Fixed original exponential SCM reference, excluded from random training.

    m=1-X-3*mean_neighbor(X), tau=6+.2*exp(X), gamma=-.9, eta=0.
    ``noise`` is the realized outcome noise (normally 4*epsilon). Omitting it
    gives the conditional mean reference, with no hidden added noise.
    Inputs are [B,N], adjacency is [B,N,N]. Supports small hand-check fixtures.
    """
    if x.ndim != 2 or x.shape != treatment.shape or adjacency.shape != (*x.shape, x.shape[-1]):
        raise ValueError("Reference requires x/treatment [B,N] and adjacency [B,N,N].")
    if x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("Reference requires a nonempty batch with at least two nodes.")
    if noise is None:
        noise = torch.zeros_like(x)
    if noise.shape != x.shape:
        raise ValueError("noise must match x.")
    if x.device.type != "cpu" or adjacency.device.type != "cpu" or treatment.device.type != "cpu" or noise.device.type != "cpu":
        raise ValueError("Original reference fixture expects CPU tensors.")
    if (not torch.isfinite(x).all() or not torch.isfinite(noise).all()
        or not ((treatment == 0) | (treatment == 1)).all()
        or not ((adjacency == 0) | (adjacency == 1)).all()
        or adjacency.diagonal(dim1=-2, dim2=-1).count_nonzero()
        or not torch.equal(adjacency, adjacency.transpose(-1, -2))
        or not (adjacency.sum(-1) > 0).all()):
        raise ValueError("Reference requires finite data and simple symmetric graphs without isolates.")
    tasks = []
    for index in range(x.shape[0]):
        graph, value = adjacency[index].to(x.dtype), x[index]
        baseline = 1-value-3*torch.mv(graph, value)/graph.sum(-1)
        tasks.append(_assemble_task(x=value, adjacency=graph, treatment=treatment[index].to(x.dtype),
            structural_baseline=baseline, tau=6+.2*torch.exp(value), gamma=torch.full_like(value, -.9),
            noise=noise[index], query_generator=_task_generator(seed, "original_reference", index, "query")))
    return {key: torch.stack([task[key] for task in tasks]) for key in tasks[0]}
