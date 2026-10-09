"""One persisted ER network, independent of episode/data/test random streams."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

PROTOCOL = 'fixed_er_v1'


@dataclass(frozen=True)
class FixedERPriorConfig:
    n_units: int
    er_edge_probability: float
    treatment_prob: float = .5
    graph_family: str = 'er'
    graph_protocol: str = PROTOCOL

    def effective_degree_bounds(self):
        degree = (self.n_units - 1) * self.er_edge_probability
        return degree, degree


def _validate_parameters(n_units, edge_probability, seed):
    if not isinstance(n_units, Integral) or isinstance(n_units, bool) or n_units < 2:
        raise ValueError('Fixed graph n_units must be an integer >= 2.')
    if not math.isfinite(edge_probability) or not 0 < edge_probability < 1:
        raise ValueError('Fixed ER edge probability must be finite and in (0,1).')
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError('Fixed graph seed must be a nonnegative signed-64-bit integer.')


def adjacency_sha256(adjacency):
    matrix = adjacency.detach().cpu().to(torch.uint8).contiguous().numpy()
    return hashlib.sha256(matrix.tobytes()).hexdigest()


@lru_cache(maxsize=16)
def _decode_graph(n_units, upper_hex):
    count = n_units * (n_units - 1) // 2
    try:
        packed = bytes.fromhex(upper_hex)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid fixed graph hex encoding.') from exc
    if len(packed) != (count + 7) // 8:
        raise ValueError('Fixed graph hex length does not match n_units.')
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    if bits[count:].any():
        raise ValueError('Fixed graph padding bits must be zero.')
    graph = torch.zeros((n_units, n_units), dtype=torch.float32)
    start, end = torch.triu_indices(n_units, n_units, offset=1)
    graph[start, end] = torch.from_numpy(bits[:count].copy()).float()
    graph[end, start] = graph[start, end]
    if not bool((graph.sum(-1) > 0).all()):
        raise ValueError('Fixed graph must have no isolated nodes, as required by the unchanged DGP.')
    return graph


def adjacency_from_config(config):
    if config.graph_protocol != PROTOCOL or config.graph_family != 'er':
        raise ValueError('Expected fixed ER graph configuration.')
    return _decode_graph(config.n_units, config.fixed_graph_upper_hex).clone()


def make_graph_record(*, n_units, edge_probability, seed):
    _validate_parameters(n_units, edge_probability, seed)
    from pfn_interference.model import _generate_er_adjacency
    graph = _generate_er_adjacency(n_units, 1, edge_probability=edge_probability,
                                  generator=torch.Generator().manual_seed(seed))[0]
    start, end = torch.triu_indices(n_units, n_units, offset=1)
    upper_hex = np.packbits(graph[start, end].numpy().astype(np.uint8)).tobytes().hex()
    return dict(protocol=PROTOCOL, n_units=n_units, edge_probability=edge_probability,
                seed=seed, upper_hex=upper_hex, adjacency_sha256=adjacency_sha256(graph),
                unique_graph_count=1, realized_mean_degree=float(graph.sum(-1).mean()),
                edge_count=int(graph.sum().item() / 2), no_isolates=True)


def graph_metadata(config):
    graph = adjacency_from_config(config)
    return dict(protocol=PROTOCOL, n_units=config.n_units,
                edge_probability=config.er_edge_probability, seed=config.fixed_graph_seed,
                adjacency_sha256=adjacency_sha256(graph), unique_graph_count=1,
                expected_degree=(config.n_units - 1)*config.er_edge_probability,
                realized_mean_degree=float(graph.sum(-1).mean()),
                edge_count=int(graph.sum().item() / 2), no_isolates=True)


def load_or_create_graph(path, *, n_units, edge_probability, seed):
    """Publish one complete graph file atomically; never replace an existing graph."""
    _validate_parameters(n_units, edge_probability, seed)
    path = Path(path)
    if not path.exists():
        record = make_graph_record(n_units=n_units, edge_probability=edge_probability, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                             encoding='utf-8') as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, indent=2, allow_nan=False)
                handle.write('\n')
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    record = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(protocol=PROTOCOL, n_units=n_units, edge_probability=edge_probability, seed=seed)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Fixed graph file parameters do not match this run; use the matching file.')
    graph = _decode_graph(n_units, record.get('upper_hex'))
    if record.get('adjacency_sha256') != adjacency_sha256(graph):
        raise ValueError('Fixed graph file adjacency SHA256 mismatch.')
    return record
