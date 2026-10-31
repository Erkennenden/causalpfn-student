"""Create-only array/JSON output partitions linked to the existing Attempt ledger.

execution_records.jsonl is only a file-to-Attempt index; it has no execution
status authority. No pickle, retries, result selection or scientific algorithms.
"""
from pathlib import Path
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import io
import json
import os
import threading
import weakref

import numpy as np

from .contracts import (FACTUAL_KEYS, TaskSpec, predictor_inputs,
                        validate_predictor_inputs, _freeze)
from .estimation import EstimateBundle, PairEstimateBundle
from .task_registry import Attempt, strict_json, request_document
from . import artifacts
from .provenance import relative_path, _no_links

PARTITIONS = ("factual/inputs", "factual/simulator_private", "truth", "predictions",
              "pair_estimates", "support", "intervals", "runtime")
_LOCK = threading.RLock()
# One serial writer per output root. The registry is weakly keyed so completed
# runs do not leave process-global cache entries behind.
_APPEND_INDEX = weakref.WeakKeyDictionary()


class DataArtifactError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DataArtifactError(message)


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    require(value is None or type(value) in (str, int, float, bool),
            "Metadata must be JSON-only, no arrays/model/private objects")
    return value


def json_bytes(value):
    return (json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _target(root, relative):
    relative_path(relative)
    root = Path(root).absolute()
    _no_links(root)
    require(root.is_dir(), "Existing output root required")
    path = root.joinpath(*relative.split("/"))
    _no_links(path)
    require(path.resolve().is_relative_to(root.resolve()), "Outside output root")
    return path


def _write(root, relative, blob):
    path = _target(root, relative)
    require(not path.exists(), "Create-only output exists: "+relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_links(path.parent)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": relative, "bytes": len(blob), "sha256": sha(blob)}


def verify_file(root, descriptor):
    require(set(descriptor) == {"path", "bytes", "sha256"}, "File receipt schema mismatch")
    path = _target(root, descriptor["path"])
    raw = path.read_bytes()
    require(len(raw) == descriptor["bytes"] and sha(raw) == descriptor["sha256"],
            "Output bytes/hash mismatch")
    return raw


def save_arrays(root, relative_stem, arrays, metadata):
    """Raw dtypes/shapes and logical C-order bytes preserved, including NaN."""
    relative_path(relative_stem)
    require(any(relative_stem.startswith(part+"/") for part in PARTITIONS),
            "Unknown output partition")
    require(isinstance(arrays, Mapping) and arrays, "Explicit nonempty arrays required")
    if relative_stem.startswith("factual/inputs/"):
        require(set(arrays)==set(FACTUAL_KEYS), "Exactly five factual keys required")
        validate_predictor_inputs(MappingProxyType({k:_freeze(v) for k,v in arrays.items()}))
    copied = {}
    records = {}
    for key, value in arrays.items():
        require(type(key) is str and key and "/" not in key and "\\" not in key,
                "Invalid NPZ key")
        require(type(value) is np.ndarray and not value.dtype.hasobject,
                "Plain non-object ndarray required")
        copied[key] = value.copy(order="K")
        records[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                        "sha256_c_bytes": sha(value.tobytes(order="C"))}
    # Prevalidate every destination/metadata before the first write.
    require(not _target(root, relative_stem+".npz").exists()
            and not _target(root, relative_stem+".json").exists(), "Create-only output exists")
    out = io.BytesIO()
    np.savez(out, **copied)
    header = json_bytes({"schema_version": 1, "arrays": records, "metadata": metadata})
    return (_write(root, relative_stem+".npz", out.getvalue()),
            _write(root, relative_stem+".json", header))


def load_arrays(root, descriptors):
    require(len(descriptors) == 2, "NPZ and JSON receipts required")
    raw = verify_file(root, descriptors[0])
    document = strict_json(verify_file(root, descriptors[1]).decode("ascii"))
    require(document["schema_version"] == 1, "Unknown array record version")
    arrays = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        require(set(archive.files) == set(document["arrays"])
                and len(archive.files) == len(set(archive.files)), "NPZ key mismatch")
        for key, record in document["arrays"].items():
            value = archive[key]
            require(not value.dtype.hasobject and value.dtype.str == record["dtype"]
                    and list(value.shape) == record["shape"]
                    and sha(value.tobytes(order="C")) == record["sha256_c_bytes"],
                    "Array identity mismatch")
            arrays[key] = _freeze(value)
    return MappingProxyType(arrays), document["metadata"]


def save_factual_inputs(root, task):
    require(type(task) is TaskSpec, "Only public TaskSpec factual inputs accepted")
    inputs = predictor_inputs(task)
    validate_predictor_inputs(inputs)
    return save_arrays(root, "factual/inputs/"+task.context_id, inputs,
        {"context_id": task.context_id, "pair_id": task.pair_id, "input_hash": task.input_hash})


def load_factual_inputs(root, descriptors):
    require(all(d["path"].startswith("factual/inputs/") for d in descriptors),
            "Predictors cannot read truth/private/prediction partitions")
    arrays, metadata = load_arrays(root, descriptors)
    require(set(arrays) == set(FACTUAL_KEYS), "Exactly five factual keys required")
    validate_predictor_inputs(arrays)
    return arrays


def save_evaluator_arrays(root, partition, identity, arrays, metadata):
    """Explicit simulator/evaluator-only entry, never supplied to a predictor."""
    require(partition in ("truth", "factual/simulator_private"), "Privileged partition required")
    relative_path(identity)
    return save_arrays(root, partition+"/"+identity, arrays,
                       {**plain(metadata), "access": "simulator_evaluator_only"})


def _split_support(support):
    arrays = {}
    def visit(value, prefix):
        if type(value) is np.ndarray:
            arrays[prefix] = value
            return {"array_key": prefix}
        if is_dataclass(value):
            return {f.name: visit(getattr(value, f.name), prefix+"."+f.name)
                    for f in fields(value)}
        if isinstance(value, (list, tuple)):
            return [visit(v, prefix+"."+str(i)) for i, v in enumerate(value)]
        return plain(value)
    return arrays, visit(support, "support")


def _attempt(registry, bundle, attempt_no):
    key = (bundle.method_id, bundle.request.scope, bundle.task_id, bundle.request_id)
    actual = registry.get_attempt(key, attempt_no)
    require(actual is not None, "Output needs its already recorded authoritative Attempt")
    expected = Attempt.from_bundle(bundle, attempt_no=attempt_no,
                                   authorization=actual.document["authorization"])
    require(actual == expected, "Bundle/Attempt content mismatch")
    return actual


def _append_execution_record(path, encoded):
    """Append one complete JSONL record and durably flush it."""
    with path.open("ab") as stream:
        written = stream.write(encoded)
        require(written == len(encoded), "Short execution-record append")
        stream.flush()
        os.fsync(stream.fileno())


def _execution_index_signature(path):
    """Cheaply detect out-of-band changes to the append-only index."""
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _append_index_state(root, registry, current_attempt):
    """Return a run-local Attempt-key index without rescanning stable history.

    The Pilot runner is a single serial writer. If the JSONL file's filesystem
    identity changes outside this writer, rebuild the index only after the
    existing records and their referenced files pass full validation. The
    final read_execution_records() remains the authoritative full audit.
    """
    count = registry.attempt_count
    require(bool(count) and registry.latest_attempt is current_attempt,
            "Attempt outputs must be persisted in single-writer Registry order")
    root_key = Path(root).resolve()
    index_path = _target(root, "execution_records.jsonl")
    signature = _execution_index_signature(index_path)
    by_root = _APPEND_INDEX.setdefault(registry, {})
    state = by_root.get(root_key)
    if state is None or state["signature"] != signature:
        if signature is None:
            records = ()
        else:
            records = read_execution_records(root, registry)
        keys = {(tuple(record["execution_key"]), record["attempt_no"])
                for record in records}
        # Cold/rebuild boundary keeps the authoritative full-history check.
        prior_attempts = {attempt.key for attempt in registry.attempts
                          if attempt is not current_attempt}
        require(prior_attempts.issubset(keys),
                "Existing Registry Attempts are missing execution output records")
        state = {"keys": keys, "signature": signature}
        by_root[root_key] = state
    require(len(state["keys"]) == count-1,
            "Prior Registry Attempts are missing indexed output records")
    return state


def read_execution_records(root, registry, *, require_complete=False):
    path = _target(root, "execution_records.jsonl")
    if not path.exists():
        require(not require_complete or not registry.attempts, "Missing execution records")
        return ()
    lines = path.read_bytes().splitlines()
    records = []
    seen = set()
    attempts = {a.key: a for a in registry.attempts}
    for line in lines:
        require(bool(line), "Empty execution record")
        record = strict_json(line.decode("ascii"))
        require(set(record) == {"record_version", "execution_key", "attempt_no",
            "attempt_identity", "outputs", "runtime"}, "Execution record schema mismatch")
        key = (tuple(record["execution_key"]), record["attempt_no"])
        require(record["record_version"] == 1 and key in attempts and key not in seen,
                "Unknown/duplicate execution record")
        require(record["attempt_identity"] == {
            "task_id": attempts[key].document["task_id"],
            "attempt_sha256": sha(attempts[key].content.encode("ascii"))},
            "Attempt identity mismatch")
        tag=sha(json_bytes(list(key[0])))+"/attempt_"+str(key[1])
        request=attempts[key].document["execution_request"]["request"]
        sources=["factual/inputs/"+cid for cid in request["context_ids"]]
        sources+=["truth/mechanism_"+str(request["identity"]["mechanism_id"]),
                  "factual/simulator_private/"+request["pair_id"]]
        for descriptor in record["outputs"]:
            require("/"+tag in descriptor["path"] or any(
                descriptor["path"] in (stem+".npz",stem+".json") for stem in sources),
                "Output path/Attempt association mismatch")
            verify_file(root, descriptor)
        verify_file(root, record["runtime"])
        require(record["runtime"] in record["outputs"], "Runtime receipt missing from output list")
        seen.add(key)
        records.append(record)
    require(not require_complete or seen == set(attempts), "Incomplete execution output linkage")
    return tuple(records)


def save_attempt_outputs(root, registry, bundle, *, attempt_no, runtime, source_receipts=()):
    """After Registry.record_attempt, persist ALL partial/failed fields and link."""
    require(type(bundle) in (EstimateBundle, PairEstimateBundle), "Public Bundle required")
    require(registry.sealed, "Sealed Registry required")
    attempt = _attempt(registry, bundle, attempt_no)
    key = attempt.execution_key
    tag = sha(json_bytes(list(key)))+"/attempt_"+str(attempt_no)
    with _LOCK:
        # The Pilot runner writes one Attempt at a time. Check the current key
        # against the run-local index; do not re-read/re-hash all prior outputs.
        # A changed index-file signature forces a full validation/rebuild.
        state = _append_index_state(root, registry, attempt)
        require(attempt.key not in state["keys"], "Attempt outputs already recorded")
        arrays = {name: getattr(bundle, name) for name in ("mu", "ite", "ate")
                  if getattr(bundle, name) is not None}
        metadata = {"request": request_document(bundle.request), **attempt.document}
        outputs = []
        stems=["factual/inputs/"+cid for cid in bundle.context_ids]
        stems+=["truth/mechanism_"+str(bundle.request.identity["mechanism_id"]),
                "factual/simulator_private/"+bundle.pair_id]
        for descriptor in source_receipts:
            require(any(descriptor["path"] in (stem+".npz",stem+".json") for stem in stems),
                    "Source receipt/Attempt identity mismatch")
            verify_file(root,descriptor)
            outputs.append(dict(descriptor))
        partition = "pair_estimates" if type(bundle) is PairEstimateBundle else "predictions"
        if arrays:
            outputs.extend(save_arrays(root, partition+"/"+tag, arrays, metadata))
        else:
            outputs.append(_write(root, partition+"/"+tag+".json", json_bytes(metadata)))
        support_arrays, support_meta = _split_support(bundle.support)
        outputs.extend(save_arrays(root, "support/"+tag, support_arrays, support_meta))
        intervals = {"execution_key": list(key), "attempt_no": attempt_no,
            "intervals": attempt.document["intervals"],
            "sampling_uncertainty": attempt.document["sampling_uncertainty"],
            "diagnostics": attempt.document["diagnostics"],
            "failure_provenance": attempt.document["failure_provenance"],
            "failure_type": bundle.failure_type, "failure_message": bundle.failure_message}
        outputs.append(_write(root, "intervals/"+tag+".json", json_bytes(intervals)))
        runtime_receipt = _write(root, "runtime/"+tag+".json", json_bytes(runtime))
        outputs.append(runtime_receipt)
        record = {"record_version": 1, "execution_key": list(key), "attempt_no": attempt_no,
            "attempt_identity": {"task_id": bundle.task_id,
                "attempt_sha256": sha(attempt.content.encode("ascii"))},
            "outputs": outputs, "runtime": runtime_receipt}
        # Only the index is appended; no Attempt fields or history are replaced.
        index_path = _target(root, "execution_records.jsonl")
        encoded = json_bytes(record)
        _append_execution_record(index_path, encoded)
        # Update only after write, flush and fsync all succeed. If persistence
        # fails, the next call observes a changed signature and revalidates.
        state["keys"].add(attempt.key)
        state["signature"] = _execution_index_signature(index_path)
    return record


def export_ledger(registry, destination):
    return artifacts.write_snapshot(registry, destination)
