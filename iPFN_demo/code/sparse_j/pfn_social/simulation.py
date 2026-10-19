"""Conditional neighbor-assignment masks for HINet integration."""
from __future__ import annotations


import math


from scipy.stats import binom, qmc


import numpy as np


def _masks(d, s, draws, seed, exact):
    if exact:
        masks = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(float)
        return masks[(masks.sum(1) > d // 2) == bool(s)]
    # K | strict-majority group, then a uniform subset of size K. Sobol random
    # rankings implement uniform subsets, not deterministic first-K neighbors.
    u = qmc.Sobol(d + 1, scramble=True, seed=seed).random_base2(int(math.log2(draws)))
    counts = np.arange(d + 1)
    counts = counts[(counts > d // 2) == bool(s)]
    probs = binom.pmf(counts, d, 0.5)
    cdf = np.cumsum(probs / probs.sum())
    cdf[-1] = 1.0
    k = counts[np.searchsorted(cdf, u[:, 0])]
    ranks = np.argsort(np.argsort(u[:, 1:], axis=1), axis=1)
    return (ranks < k[:, None]).astype(float)


HINET_PROTOCOL = "hinet_bc_majority_policy_v1"

