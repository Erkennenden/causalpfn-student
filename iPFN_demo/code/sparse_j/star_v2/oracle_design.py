"""Privileged Oracle design only: no posterior, simulator, beta or truth input.

Columns are fixed to the common 25-coefficient DGP. Query axes are [node,t,s,
coefficient], flattening to 00,01,10,11. Arbitrary valid graph sizes support
small deterministic references; production TaskSpec constraints are unchanged.
"""
import numpy as np

from pfn_social.phase3_hierarchical import coefficient_prior as _common_prior
from pfn_social.contracts import arm_mean_exposure


def _freeze(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite Oracle design/prior")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _numeric(value, name):
    if type(value) is not np.ndarray or value.dtype.kind not in "iuf":
        raise TypeError(name + " must be a plain real numeric ndarray")
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _graph(adjacency):
    a = _numeric(adjacency, "adjacency")
    if (a.ndim != 2 or a.shape[0] < 2 or a.shape[0] != a.shape[1]
            or not np.isin(a, (0., 1.)).all()
            or not np.array_equal(a, a.T) or np.any(np.diag(a) != 0)):
        raise ValueError("Require a binary symmetric graph without self loops")
    degree = a.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("Isolated nodes have undefined exposure")
    return a, degree


def _structure(true_h, adjacency):
    a, degree = _graph(adjacency)
    h = _numeric(true_h, "true_h")
    if h.shape != (len(a), 3):
        raise ValueError("Require true_h[N,3] in original node order")
    return a, degree, h, a @ h / degree[:, None]


def _treatment(treatment, n):
    t = _numeric(treatment, "treatment")
    if t.shape != (n,) or not np.isin(t, (0., 1.)).all():
        raise ValueError("Require binary treatment[N] in original node order")
    return t


def coefficient_prior():
    """Exact fixed common prior, returned as detached immutable float64 arrays."""
    mean, covariance = _common_prior()
    return _freeze(mean), _freeze(covariance)


def factual_exposure(*, adjacency, treatment):
    """Actual continuous exposure and S=1{exposure>0.5}; ties belong to S=0."""
    a, degree = _graph(adjacency)
    t = _treatment(treatment, len(a))
    e = a @ t / degree
    s = (e > .5).astype(np.int64)
    return _freeze(e), np.frombuffer(s.tobytes(), dtype=np.int64)


def _rows(h, hbar, t, e, j):
    hetero = np.column_stack((np.ones(len(h)), h))
    rows = np.column_stack((
        np.ones(len(h)), h, hbar, t[:, None] * hetero,
        e[:, None] * hetero, (t * e)[:, None] * hetero,
        j, t[:, None] * j))
    return _freeze(rows)


def factual_design(*, true_h, adjacency, treatment):
    """Actual e=A@T/d and j=A@(T*h)/d, never query-state substitutes."""
    a, degree, h, hbar = _structure(true_h, adjacency)
    t = _treatment(treatment, len(a))
    e = a @ t / degree
    j = a @ (t[:, None] * h) / degree[:, None]
    return _rows(h, hbar, t, e, j)


def query_design(*, true_h, adjacency):
    """Conditional iid Bernoulli(.5) design, float64 [N,2,2,25]."""
    a, degree, h, hbar = _structure(true_h, adjacency)
    result = np.empty((len(a), 2, 2, 25), dtype=np.float64)
    for s in (0, 1):
        by_degree = {int(d): arm_mean_exposure(int(d), .5, s)
                     for d in np.unique(degree)}
        e = np.array([by_degree[int(d)] for d in degree], dtype=np.float64)
        j = e[:, None] * hbar
        for t in (0, 1):
            result[:, t, s] = _rows(h, hbar, np.full(len(a), float(t)), e, j)
    return _freeze(result)
