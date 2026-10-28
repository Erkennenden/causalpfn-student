"""Shared graph, hidden-feature and response functions for synthetic data."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Setting:
    name: str
    n: int
    graph: str
    degree: int
    noise_sd: float
    representation: str = "shared_nonlinear"


@dataclass(frozen=True)
class ConfigurationSetting(Setting):
    # A subclass preserves the original Setting fields and positional interface.
    degree_spread: int = 0


SETTINGS = (
    Setting("shared_ws20_k4_noise1", 20, "ws", 4, 1.),
    Setting("shared_ws20_k4_noise2", 20, "ws", 4, 2.),
    Setting("shared_ws40_k4_noise1", 40, "ws", 4, 1.),
    Setting("shared_ws40_k8_noise1", 40, "ws", 8, 1.),
    Setting("shared_er40_degree4_noise1", 40, "er", 4, 1.),
    Setting("shared_sbm40_degree4_noise1", 40, "sbm", 4, 1.),
    Setting("unshared_ws40_k4_noise1", 40, "ws", 4, 1., "unshared_nonlinear"),
    Setting("linear_ws80_k8_noise1", 80, "ws", 8, 1., "shared_linear"),
)

# Append-only registry: original base_setting_id 0..7 must never move.
CONFIGURATION_BASE_IDS = tuple(range(8, 14))
SETTINGS += tuple(
    ConfigurationSetting(f"shared_configuration200_k{k}_spread{s}_noise1",
                         200, "configuration", k, 1., degree_spread=s)
    for k in (4, 8) for s in (0, k//4, k//2)
)


def _hash_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def hidden_features(x, projection, representation):
    z = np.asarray(x, dtype=np.float64) @ projection
    if representation == "shared_linear":
        return np.column_stack([z[:, 0], z[:, 1], (z[:, 0]+z[:, 1])/np.sqrt(2)])
    return np.column_stack([np.tanh(1.5*z[:, 0]), np.sin(1.5*z[:, 1]),
                            np.tanh(z[:, 0]*z[:, 1])])


def connected_graph(setting, rng):
    """Legacy ring-backbone WS/ER/SBM, or the E1.2 fixed-degree variant.

    WS retains all nearest-neighbor ring edges, adds ring distances 2..k/2,
    then rewires those added edges with probability .15.  ER/SBM add independent
    non-ring edges with expected total degree k; SBM has within:between ratio4.
    These are explicitly ring-connected graph variants, not textbook draws
    conditioned post hoc on connectivity.
    Configuration graphs delegate to the separately versioned simple connected
    generator and have no ring backbone.
    """
    if setting.graph == "configuration":
        from pfn_social.configuration_graph import configuration_graph
        return configuration_graph(setting.n, setting.degree,
                                   getattr(setting, 'degree_spread', 0), rng)
    n, k = setting.n, setting.degree
    a = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        a[i, (i+1) % n] = a[(i+1) % n, i] = 1
    if setting.graph == "ws":
        added = []
        for distance in range(2, k//2+1):
            for i in range(n):
                j = (i+distance) % n
                a[i, j] = a[j, i] = 1
                added.append((i, j))
        for i, j in added:
            if rng.random() < .15:
                candidates = np.flatnonzero((a[i] == 0) & (np.arange(n) != i))
                if len(candidates):
                    target = int(rng.choice(candidates))
                    a[i, j] = a[j, i] = 0
                    a[i, target] = a[target, i] = 1
    else:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not a[i, j]]
        weights = np.array([4. if setting.graph == "sbm" and (i < n//2) == (j < n//2)
                            else 1. for i, j in pairs])
        probabilities = (n*(k-2)/2) * weights/weights.sum()
        if probabilities.max() > 1:
            raise ValueError("Registered graph probabilities must be <=1")
        for (i, j), included in zip(pairs, rng.random(len(pairs)) < probabilities):
            if included:
                a[i, j] = a[j, i] = 1
    return a


def coefficient_prior():
    """25 coefficients: baseline7, direct4, exposure4, interaction4, J3, T*J3."""
    mean = np.zeros(25); mean[7] = .6; mean[11] = .6
    scale = np.r_[.5, np.repeat(.8/np.sqrt(3), 3), np.repeat(.35/np.sqrt(3), 3),
                  .5, np.repeat(.7/np.sqrt(3), 3),
                  .5, np.repeat(.7/np.sqrt(3), 3),
                  .25, np.repeat(.35/np.sqrt(3), 3),
                  np.repeat(.4/np.sqrt(3), 3), np.repeat(.2/np.sqrt(3), 3)]
    covariance = np.diag(scale**2)
    for j in range(4):
        covariance[7+j, 11+j] = covariance[11+j, 7+j] = .35*scale[7+j]*scale[11+j]
    return mean, covariance


def _components(a, h, beta):
    degree = a.sum(1)
    hbar = a @ h/degree[:, None]
    hetero = np.column_stack([np.ones(len(a)), h])
    baseline = np.column_stack([np.ones(len(a)), h, hbar]) @ beta[:7]
    return np.column_stack([baseline, hetero@beta[7:11], hetero@beta[11:15], hetero@beta[15:19]])


def structural_outcome(mechanism, treatment, *, own_override=None):
    a, beta = mechanism.adjacency, mechanism.coefficients
    t = np.asarray(treatment, dtype=np.float64)
    degree = a.sum(1); e = a@t/degree
    j = a@(t[:, None]*mechanism.h)/degree[:, None]
    own = t if own_override is None else np.full(len(a), own_override, dtype=np.float64)
    b, tau, gamma, eta = mechanism.components.T
    return b + own*tau + (gamma+own*eta)*e + j@beta[19:22] + own*(j@beta[22:25])
