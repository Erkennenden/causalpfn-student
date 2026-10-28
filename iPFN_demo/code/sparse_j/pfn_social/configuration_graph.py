"""Connected simple fixed-degree configuration variant (version 1).

Havel--Hakimi initialization, degree-preserving component joins and a finite
connected switch chain. This is NOT an exact uniform configuration-model draw.
Never simplify a multigraph: doing so would change the requested degrees.
"""
from numbers import Integral

import networkx as nx
import numpy as np

GENERATOR_VERSION = 'simple_connected_configuration_v1'
SWITCH_ATTEMPTS_PER_EDGE = 20


def degree_sequence(n, mean_degree, spread):
    """Half k-s, half k+s: exact mean k, population SD s, and CV s/k."""
    if any(isinstance(v, bool) or not isinstance(v, Integral)
           for v in (n, mean_degree, spread)):
        raise ValueError('n, mean_degree and spread must be integers')
    if n < 4 or n % 2 or spread < 0 or mean_degree-spread < 2 or mean_degree+spread >= n:
        raise ValueError('Require even n>=4 and 2<=k-spread<=k+spread<n')
    degrees = np.repeat([mean_degree-spread, mean_degree+spread], n//2)
    if not nx.is_graphical(degrees.tolist(), method='eg'):
        raise ValueError('Non-graphical configuration degree sequence')
    return degrees


def _join_components(graph):
    """Join cycles in different components without changing any degree.

    Minimum degree >=2 guarantees a cycle in each component. Removing a
    non-bridge edge in each leaves both connected; two cross edges merge them.
    Sorted traversal makes this independent of hash/set iteration order.
    """
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
    if len(components) < 2:
        return
    merged = components[0]
    for component in components[1:]:
        selected = []
        for nodes in (merged, component):
            subgraph = graph.subgraph(nodes)
            bridges = {tuple(sorted(e)) for e in nx.bridges(subgraph)}
            selected.append(next(e for e in sorted(tuple(sorted(e)) for e in subgraph.edges)
                                 if e not in bridges))
        (a, b), (c, d) = selected
        graph.remove_edges_from(selected)
        graph.add_edges_from(((a, c), (b, d)))
        merged = sorted(merged + component)


def configuration_graph(n, mean_degree, spread, rng):
    degrees = rng.permutation(degree_sequence(n, mean_degree, spread))
    graph = nx.havel_hakimi_graph(degrees.tolist())
    _join_components(graph)
    edges = sorted(tuple(sorted(e)) for e in graph.edges)
    # Uniform edge-pair proposals, random orientation, rejected proposals remain
    # self transitions. A fixed attempt budget avoids an unbounded rejection loop.
    for _ in range(SWITCH_ATTEMPTS_PER_EDGE * len(edges)):
        i, j = rng.choice(len(edges), 2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if rng.integers(2):
            c, d = d, c
        if len({a, b, c, d}) != 4 or graph.has_edge(a, c) or graph.has_edge(b, d):
            continue
        graph.remove_edges_from(((a, b), (c, d)))
        graph.add_edges_from(((a, c), (b, d)))
        if nx.is_connected(graph):
            edges[i], edges[j] = tuple(sorted((a, c))), tuple(sorted((b, d)))
        else:
            graph.remove_edges_from(((a, c), (b, d)))
            graph.add_edges_from(((a, b), (c, d)))
    adjacency = nx.to_numpy_array(graph, nodelist=range(n), dtype=np.uint8)
    if not np.array_equal(adjacency.sum(1), degrees) or not nx.is_connected(graph):
        raise RuntimeError('Configuration generator invariant violated')
    return adjacency


def degree_statistics(adjacency):
    """Actual statistics; histogram and degree vector permit independent audits."""
    a = np.asarray(adjacency)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.array_equal(a, a.T) or np.any(np.diag(a))
            or not np.isin(a, [0, 1]).all()):
        raise ValueError('Require a nonempty undirected simple adjacency matrix')
    degrees = a.sum(1).astype(int)
    values, counts = np.unique(degrees, return_counts=True)
    mean, sd = float(degrees.mean()), float(degrees.std())
    return dict(n=len(a), edges=int(degrees.sum()//2), degree_mean=mean,
                degree_sd=sd, degree_variance=float(degrees.var()),
                degree_cv=sd/mean if mean else None,
                degree_min=int(degrees.min()), degree_max=int(degrees.max()),
                degree_histogram={str(d): int(c) for d, c in zip(values, counts)},
                degrees=degrees.tolist(), connected=nx.is_connected(nx.from_numpy_array(a)))
