"""Auditable JSON/JSONL ledger snapshots; no pickle or prediction packaging.

Each destination is create-only. Reads require a trusted manifest SHA256 receipt:
internal hashes alone cannot detect coordinated deletion/rewrite of history.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import shutil
import tempfile

from .task_registry import (TaskRegistry, DataTask, ExecutionRequest, Attempt,
                            LedgerError, canonical, strict_json)

FILES = ("tasks.jsonl","requests.jsonl","attempts.jsonl")
MANIFEST = "ledger_manifest.json"


class ArtifactError(LedgerError):
    pass


@dataclass(frozen=True,slots=True)
class SnapshotReceipt:
    manifest_sha256: str
    seal_sha256: str


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _write(path,blob):
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def write_snapshot(registry,directory):
    """Persist EVERY registered task/request and ALL attempts; never overwrite."""
    if type(registry) is not TaskRegistry or not registry.sealed:
        raise ArtifactError("A sealed TaskRegistry is required")
    target = Path(directory).absolute()
    if target.exists() or target.is_symlink():
        raise ArtifactError("Snapshot destination already exists")
    if not target.parent.is_dir():
        raise ArtifactError("Snapshot parent directory must already exist")
    # Copy a complete immutable view before any I/O.
    tasks = [v.document for v in registry.tasks.values()]
    requests = [v.document for v in registry.requests.values()]
    attempts = [v.document for v in registry.attempts]
    content = dict(zip(FILES,(tasks,requests,attempts)))
    blobs = {name:("".join(canonical(row)+"\n" for row in rows)).encode("ascii")
             for name,rows in content.items()}
    manifest = {"schema_version":1,"seal_sha256":registry.seal_sha256,
        "expected_task_ids":sorted(registry.tasks),
        "expected_request_keys":[list(k) for k in sorted(registry.requests)],
        "files":{name:{"count":len(content[name]),"sha256":_sha(blobs[name])} for name in FILES}}
    manifest_blob = (canonical(manifest)+"\n").encode("ascii")
    staging = Path(tempfile.mkdtemp(prefix=".ledger-stage-",dir=target.parent)).resolve()
    parent = target.parent.resolve()
    try:
        for name in FILES:
            _write(staging/name,blobs[name])
        _write(staging/MANIFEST,manifest_blob)
        if target.exists() or target.is_symlink():
            raise ArtifactError("Destination appeared during snapshot write")
        # Windows rename refuses an existing destination; no replacement API.
        os.rename(staging,target)
    except Exception:
        # Only our fresh staging directory, never a caller's existing target.
        if staging.exists():
            if staging.parent != parent or not staging.name.startswith(".ledger-stage-"):
                raise ArtifactError("Unsafe staging cleanup path")
            shutil.rmtree(staging)
        raise
    return SnapshotReceipt(_sha(manifest_blob),registry.seal_sha256)


def read_snapshot(directory,*,expected_manifest_sha256):
    """Strict anchored read/replay; any corruption/missing/extra row is an error."""
    root = Path(directory)
    try:
        if root.is_symlink() or not root.is_dir():
            raise ArtifactError("Expected snapshot directory, not a symlink")
        if {p.name for p in root.iterdir()} != set(FILES)|{MANIFEST}:
            raise ArtifactError("Missing or extra snapshot files")
        if any((root/name).is_symlink() or not (root/name).is_file() for name in (*FILES,MANIFEST)):
            raise ArtifactError("Snapshot entries must be regular files")
        blob = (root/MANIFEST).read_bytes()
        if type(expected_manifest_sha256) is not str or _sha(blob) != expected_manifest_sha256:
            raise ArtifactError("Trusted manifest SHA256 mismatch")
        manifest = strict_json(blob.decode("ascii"))
        if set(manifest) != {"schema_version","seal_sha256","expected_task_ids",
                             "expected_request_keys","files"} or manifest["schema_version"] != 1:
            raise ArtifactError("Manifest schema mismatch")
        if set(manifest["files"]) != set(FILES):
            raise ArtifactError("Manifest file set mismatch")
        docs = {}
        for name in FILES:
            info = manifest["files"][name]
            if set(info) != {"count","sha256"} or type(info["count"]) is not int or info["count"] < 0:
                raise ArtifactError("Invalid file count/hash declaration")
            raw = (root/name).read_bytes()
            if _sha(raw) != info["sha256"]:
                raise ArtifactError("File SHA256 mismatch: "+name)
            text = raw.decode("ascii")
            if text and not text.endswith("\n"):
                raise ArtifactError("Truncated JSONL record: "+name)
            lines = text.splitlines()
            if any(not line for line in lines):
                raise ArtifactError("Blank JSONL record: "+name)
            rows = [strict_json(line) for line in lines]
            if len(rows) != info["count"]:
                raise ArtifactError("Record count mismatch: "+name)
            docs[name] = rows
        ledger = TaskRegistry()
        for doc in docs["tasks.jsonl"]:
            ledger.register_task(DataTask.from_document(doc))
        for doc in docs["requests.jsonl"]:
            ledger.register_execution(ExecutionRequest.from_document(doc))
        ledger.seal(expected_task_ids=manifest["expected_task_ids"],
                    expected_request_keys=manifest["expected_request_keys"])
        if ledger.seal_sha256 != manifest["seal_sha256"]:
            raise ArtifactError("Frozen task/request content seal mismatch")
        for doc in docs["attempts.jsonl"]:
            ledger.record_attempt(Attempt.from_document(doc))
        return ledger
    except (OSError,UnicodeError,ValueError,TypeError,KeyError,AttributeError) as exc:
        if type(exc) is ArtifactError:
            raise
        raise ArtifactError("Invalid snapshot: "+str(exc)) from exc
