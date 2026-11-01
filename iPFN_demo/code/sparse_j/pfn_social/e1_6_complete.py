"""E1.6 complete-graph tasks built from the supplied synthetic DGP.

The existing feature, coefficient, treatment, noise, truth, and observation-mask
rules remain unchanged.  Only the graph is replaced by a loopless complete
graph.  This module deliberately lives beside, rather than inside, the frozen
model code so the shipped checkpoint and its hash remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pfn_social.contracts import arm_mean_exposure, arm_probability
from pfn_social.phase3_hierarchical import Setting, _components, _hash_arrays
from pfn_social.phase5_large_network import (
    LargeNetworkConfig,
    generate_mechanism as generate_parent_mechanism,
)
from pfn_social.phase6_limited_outcomes import (
    LimitedOutcomeMechanism,
    make_observed_draw as make_parent_observed_draw,
)


TOPOLOGY_NAMESPACE = 1606
ARM_LABELS = ("00", "01", "10", "11")
EFFECT_LABELS = ("direct", "spillover", "total")
EFFECT_ARM_PAIRS = ((2, 0), (3, 2), (3, 0))


def complete_adjacency(n: int) -> np.ndarray:
    """Return a binary, symmetric, loopless K_n adjacency matrix."""
    n = int(n)
    if n < 2:
        raise ValueError("A complete graph requires at least two nodes.")
    return np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)


@dataclass(frozen=True)
class CompleteGraphMechanism:
    """A Phase-5-compatible mechanism whose topology is exactly K_n."""

    config: LargeNetworkConfig
    mechanism_id: int
    adjacency: np.ndarray
    x: np.ndarray
    projection: np.ndarray
    h: np.ndarray
    coefficients: np.ndarray
    components: np.ndarray
    truth: np.ndarray
    population_mean: np.ndarray
    mechanism_sha256: str
    truth_sha256: str

    @property
    def setting(self) -> Setting:
        base = self.config.setting
        return replace(
            base,
            name=f"e1_6_complete_n{base.n}",
            graph="complete",
            degree=base.n - 1,
        )

    @property
    def setting_id(self) -> int:
        return self.config.setting_id

    @property
    def split(self) -> str:
        return self.config.split


def _complete_truth(adjacency, h, coefficients, components, treatment_prob):
    n = len(adjacency)
    degree = adjacency.sum(1)
    if not np.all(degree == n - 1):
        raise ValueError("E1.6 truth requires a complete graph.")
    hbar = adjacency @ h / degree[:, None]
    baseline, direct, spillover, interaction = components.T
    truth = np.empty((n, 4), dtype=np.float64)
    for own, peer_state in ((0, 0), (0, 1), (1, 0), (1, 1)):
        mean_exposure = arm_mean_exposure(n - 1, treatment_prob, peer_state)
        truth[:, 2 * own + peer_state] = (
            baseline
            + own * direct
            + mean_exposure
            * (
                spillover
                + own * interaction
                + hbar @ (coefficients[19:22] + own * coefficients[22:25])
            )
        )
    return truth


def generate_complete_mechanism(
    mechanism_id: int,
    *,
    n: int = 200,
    base_setting_id: int = 0,
    setting_id: int = 16,
    split: str = "dev",
    split_seed: int = 331629,
    treatment_probability: float = 0.5,
) -> CompleteGraphMechanism:
    """Generate one mechanism while replacing only the parent adjacency."""
    if n != 200:
        raise ValueError("The supplied limited-outcome pipeline fixes E1.6 at N=200.")
    config = LargeNetworkConfig(
        base_setting_id=base_setting_id,
        n=n,
        setting_id=setting_id,
        split=split,
        split_seed=split_seed,
        role="main",
    )
    # NumPy's macOS Accelerate backend can leave floating-point status flags set
    # after otherwise finite matrix products.  The explicit finite check below
    # is the correctness condition; the local errstate only removes those
    # spurious RuntimeWarnings.
    with np.errstate(all="ignore"):
        parent = generate_parent_mechanism(config, int(mechanism_id))
        adjacency = complete_adjacency(n)
        components = _components(adjacency, parent.h, parent.coefficients)
        truth = _complete_truth(
            adjacency,
            parent.h,
            parent.coefficients,
            components,
            treatment_probability,
        )
    if not all(
        np.isfinite(value).all()
        for value in (parent.x, parent.h, parent.coefficients, components, truth)
    ):
        raise FloatingPointError("E1.6 mechanism contains a nonfinite value.")
    identity = np.frombuffer(
        (config.canonical_json() + "|E1.6|complete").encode(), dtype=np.uint8
    )
    return CompleteGraphMechanism(
        config=config,
        mechanism_id=int(mechanism_id),
        adjacency=adjacency,
        x=parent.x.copy(),
        projection=parent.projection.copy(),
        h=parent.h.copy(),
        coefficients=parent.coefficients.copy(),
        components=components,
        truth=truth,
        population_mean=parent.population_mean.copy(),
        mechanism_sha256=_hash_arrays(
            adjacency,
            parent.x,
            parent.projection,
            parent.coefficients,
            parent.population_mean,
            identity,
        ),
        truth_sha256=_hash_arrays(truth),
    )


def observation_permutation(n: int, mechanism_id: int, mask_seed: int) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(mask_seed), TOPOLOGY_NAMESPACE, int(mechanism_id)]
        )
    )
    return rng.permutation(int(n))


def generate_complete_task(mechanism_id: int, **kwargs) -> LimitedOutcomeMechanism:
    """Return the wrapper expected by ``make_observed_draw``."""
    mask_seed = int(kwargs.pop("mask_seed", 16062026))
    mechanism = generate_complete_mechanism(mechanism_id, **kwargs)
    permutation = observation_permutation(
        mechanism.setting.n, mechanism_id, mask_seed
    )
    return LimitedOutcomeMechanism(mechanism, permutation)


def make_complete_observed_draw(mechanism, repeat_id):
    """Use the shared draw/mask code and require five finite factual arrays."""
    with np.errstate(all="ignore"):
        rows, shared = make_parent_observed_draw(mechanism, repeat_id)
    for row in rows:
        if not all(np.isfinite(value.numpy()).all() for value in row["batch"].values()):
            raise FloatingPointError("E1.6 factual batch contains a nonfinite value.")
    return rows, shared


def effects_from_arm_means(arm_means: np.ndarray) -> np.ndarray:
    """Convert [00, 01, 10, 11] means to direct, spillover, total effects."""
    means = np.asarray(arm_means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 4:
        raise ValueError("arm_means must have shape [N, 4].")
    return np.column_stack(
        (means[:, 2] - means[:, 0], means[:, 3] - means[:, 2], means[:, 3] - means[:, 0])
    )


def empirical_support(batch: dict, treatment_probability: float = 0.5) -> dict:
    """Separate realized-cell coverage from population positivity.

    On K_n, every node has degree n-1 and its treated-neighbor count equals the
    population treated count minus its own treatment.  Therefore a single
    assignment can realize at most two of the four (t, s) cells.  The target
    arms nevertheless all have positive probability under Bernoulli treatment.
    """
    adjacency = np.asarray(batch["adjacency"])
    treatment = np.asarray(batch["treatment"], dtype=np.int64)
    observed = np.asarray(batch["observed_indices"], dtype=np.int64)
    n = len(treatment)
    degree = adjacency.sum(1).astype(np.int64)
    if adjacency.shape != (n, n) or not np.all(degree == n - 1):
        raise ValueError("Support diagnostics require a complete graph.")
    with np.errstate(all="ignore"):
        treated_neighbor_count = adjacency @ treatment
    if not np.isfinite(treated_neighbor_count).all():
        raise FloatingPointError("Nonfinite treated-neighbor count.")
    peer_state = (treated_neighbor_count > degree // 2).astype(np.int64)
    arm_index = 2 * treatment + peer_state
    population_counts = np.bincount(arm_index, minlength=4)
    observed_counts = np.bincount(arm_index[observed], minlength=4)
    empirical_arm_mask = observed_counts > 0
    empirical_effect_mask = np.asarray(
        [empirical_arm_mask[left] and empirical_arm_mask[right]
         for left, right in EFFECT_ARM_PAIRS],
        dtype=bool,
    )
    low = arm_probability(n - 1, treatment_probability, 0)
    high = arm_probability(n - 1, treatment_probability, 1)
    population_probabilities = np.asarray(
        [(1 - treatment_probability) * low,
         (1 - treatment_probability) * high,
         treatment_probability * low,
         treatment_probability * high],
        dtype=np.float64,
    )
    if np.count_nonzero(population_counts) > 2:
        raise AssertionError("A complete graph assignment realized more than two arms.")
    return {
        "arm_labels": ARM_LABELS,
        "population_arm_counts": population_counts,
        "observed_arm_counts": observed_counts,
        "empirical_arm_mask": empirical_arm_mask,
        "empirical_effect_mask": empirical_effect_mask,
        "population_arm_probabilities": population_probabilities,
        "population_support_mask": population_probabilities > 0,
        "treated_total": int(treatment.sum()),
        "degree": n - 1,
    }
