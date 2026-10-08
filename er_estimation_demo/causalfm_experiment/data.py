"""Task cache with one shared graph, independent splits and nested task IDs."""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from linear_b_dgp import generate_task, VERSION as DGP_VERSION, FORMULA, A_LOW, A_HIGH
from cepo import PREDICTION_PROTOCOL, ARM_NAMES
from fixed_er_graph import _decode_graph, adjacency_sha256
from random_lpe_prior import _task_generator

ROOT = Path(__file__).resolve().parents[1]
TASK_KEYS = ("tokens", "queries", "cepo_target", "design_treatment_prob", "degree", "x", "neighbor_x", "observed_treatment", "observed_exposure", "y_obs", "oracle_arm_means", "oracle_ite", "oracle_ate", "task_id")
SPLITS = ("train", "validation", "test")


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TaskBank:
    """Generate/cache only accessed task IDs; never cache adjacency per task."""

    def __init__(self, graph_file, cache_dir, *, data_seed=12345, noise_sd=4.):
        self.graph_file = Path(graph_file)
        self.record = json.loads(self.graph_file.read_text(encoding="utf-8"))
        self.graph = _decode_graph(self.record["n_units"], self.record["upper_hex"]).clone()
        if adjacency_sha256(self.graph) != self.record["adjacency_sha256"]:
            raise ValueError("Saved graph failed its SHA256 check.")
        self.data_seed, self.noise_sd = data_seed, noise_sd
        sources = ("linear_b_dgp.py", "causalfm_dgp_v1/dgp.py",
                   "estimands.py", "random_lpe_prior.py", "cepo.py", "causalfm_experiment/data.py")
        self.identity = dict(schema=5, outcome_prior="linear_b_random_a", formula=FORMULA,
            a_distribution=dict(name="uniform", low=A_LOW, high=A_HIGH, scope="one per dataset"),
            prediction_protocol=PREDICTION_PROTOCOL, cepo_order=list(ARM_NAMES),
            dgp_version=DGP_VERSION, data_seed=data_seed, noise_sd=noise_sd,
            treatment_prob=.5, graph_sha256=self.record["adjacency_sha256"],
            sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
        self.fingerprint = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.directory = Path(cache_dir) / self.fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        # Same contents for all processes using this identity; task writes are atomic.
        graph_cache = self.directory / "graph.pt"
        if not graph_cache.exists():
            atomic_save(self.graph, graph_cache)
        self._memory = OrderedDict()
        write_json(self.directory/"identity.json", self.identity)

    def task(self, split, task_id):
        if split not in SPLITS or not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError("Expected a known split and a nonnegative integer task ID.")
        key = (split, task_id)
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self.directory / split / f"task_{task_id:08d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != self.fingerprint or payload["split"] != split or payload["task_id"] != task_id:
                raise ValueError(f"Task cache identity mismatch: {path}")
            tensors = payload["tensors"]
            if set(tensors) != set(TASK_KEYS):
                raise ValueError("Task cache fields differ from CEPO schema")
        else:
            task = generate_task(self.graph, task_id=task_id, seed=self.data_seed,
                                 stream=f"scaling_v1/{split}", noise_sd=self.noise_sd)
            tensors = {name: task.batch[name][0].clone() for name in TASK_KEYS}
            atomic_save(dict(identity=self.fingerprint, split=split, task_id=task_id, tensors=tensors, metadata=task.metadata), path)
        self._memory[key] = tensors
        if len(self._memory) > 32:
            self._memory.popitem(last=False)
        return tensors

    def batch(self, split, task_ids):
        ids = list(task_ids)
        if not ids:
            raise ValueError("Cannot create an empty task batch.")
        tasks = [self.task(split, i) for i in ids]
        result = {name: torch.stack([task[name] for task in tasks]) for name in TASK_KEYS}
        result["adjacency"] = self.graph.unsqueeze(0).expand(len(ids), -1, -1)
        return result


@lru_cache(maxsize=8)
def _epoch_order(count, seed, epoch):
    rng = _task_generator(seed, "scaling_v1/order", epoch, "permutation")
    return torch.randperm(count, generator=rng).tolist()


def batch_indices(step, count, batch_size, seed):
    """Zero-based update index; fixed batch size even across epoch boundaries."""
    if step < 0 or count < 1 or batch_size < 1:
        raise ValueError("Invalid training sampler settings.")
    indices = []
    position = step * batch_size
    while len(indices) < batch_size:
        epoch, offset = divmod(position, count)
        take = min(batch_size-len(indices), count-offset)
        indices.extend(_epoch_order(count, seed, epoch)[offset:offset+take])
        position += take
    return indices
