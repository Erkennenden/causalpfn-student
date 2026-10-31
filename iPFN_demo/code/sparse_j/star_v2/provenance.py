"""Deterministic source provenance and minimal run/ledger bindings (6.4-A).

runtime_root='.' means the root of an independently extracted source snapshot.
Model weights are NOT source files. No project runner/model is imported here.
Trusted manifest hashes must be retained outside the snapshot. Run manifests
are explicit caller-supplied records, never auto-created by source packaging.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
import hashlib
import io
import json
import os
import re
import stat
import zipfile

from .config import CONFIG
from . import artifacts
from .task_registry import strict_json
from .coverage import unique_attempts

ROLES = frozenset(("implementation", "test", "protocol", "documentation",
                   "dependency", "entrypoint", "configuration"))
ARCHIVE_NAME = "source_snapshot.zip"
MANIFEST_NAME = "source_manifest.json"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FORBIDDEN = frozenset((".git", "__pycache__", ".pytest_cache", ".mypy_cache",
                       ".venv", "venv", "env", "results", "outputs"))
RESERVED = frozenset(("CON", "PRN", "AUX", "NUL") +
                     tuple("COM"+str(i) for i in range(1, 10)) +
                     tuple("LPT"+str(i) for i in range(1, 10)))


class ProvenanceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProvenanceError(message)


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _json_bytes(value):
    def plain(v):
        if isinstance(v, Mapping):
            require(all(type(k) is str for k in v), "JSON keys must be strings")
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        require(v is None or type(v) in (str, int, float, bool), "JSON-only provenance")
        return v
    try:
        return (json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("Invalid JSON provenance") from exc


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _digest(value, size=64):
    return type(value) is str and re.fullmatch("[0-9a-f]{"+str(size)+"}", value) is not None


def relative_path(value):
    require(type(value) is str and bool(value)
            and re.fullmatch(r"[A-Za-z0-9_./-]+", value) is not None,
            "Illegal portable relative path")
    parts = value.split("/")
    require(all(p not in ("", ".", "..") and not p.endswith((".", " "))
                and p.split(".")[0].upper() not in RESERVED
                and p.casefold() not in FORBIDDEN for p in parts),
            "Unsafe path component")
    require(str(PurePosixPath(value)) == value and not value.startswith("/"),
            "Noncanonical relative path")
    return value


def _no_links(path):
    for part in (path, *path.parents):
        try:
            s = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(s.st_mode)
                and not getattr(s, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                "Symlink/reparse point is forbidden: "+str(part))


def _root(path):
    p = Path(path).absolute()
    _no_links(p)
    require(p.is_dir(), "Existing directory required")
    return p.resolve(strict=True)


def _file(root, name):
    relative_path(name)
    p = root.joinpath(*name.split("/"))
    _no_links(p)
    require(p.is_file() and p.resolve(strict=True).is_relative_to(root), "Missing/outside source file: "+name)
    return p


def _read_once(root, name):
    p = _file(root, name)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(p, flags), "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular source file required")
        blob = stream.read()  # exactly one content read for BOTH hash and ZIP
    _file(root, name)  # recheck links after reading, without a second content read
    return blob


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    role: str

    def __post_init__(self):
        relative_path(self.path)
        require(self.role in ROLES, "Unknown source role")
        require(Path(self.path).suffix.lower() not in
                (".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".pyc", ".zip"),
                "Model/binary artifacts must be managed separately")


# Explicit closed filename whitelist, not recursive discovery. Additions require
# a reviewed whitelist update; no result/cache/checkpoint is silently included.
_STAR_MODULES = (
    "__init__", "artifacts", "baselines", "baselines_regression", "baselines_state_ht",
    "bootstrap", "config", "contracts", "coverage", "dgp", "estimation", "j_adapter",
    "kernel_adapters", "kernel_new_adapter", "metrics", "model_adapters",
    "oracle_adapters", "oracle_context", "oracle_design", "oracle_numerics",
    "oracle_pair", "pair", "provenance", "ranking", "rng", "support", "task_registry",
    "data_artifacts", "runtime_monitor", "run_pilot")
_STAR_TESTS = (
    "artifacts", "bootstrap", "config", "contracts", "coverage", "cross_method_contracts",
    "dgp_truth", "estimation_contract", "f_z2_lin_z2", "j_adapter", "kernel_adapters",
    "kernel_new_adapter", "metrics", "model_adapters", "oracle_adapters", "oracle_context",
    "oracle_design", "oracle_pair", "pair_data", "pair_ht", "provenance", "ranking",
    "rng_identity", "sampling_uncertainty_provenance", "stage2_acceptance",
    "stage3_acceptance", "state_mean_hajek", "support", "task_registry",
    "data_artifacts", "runtime_monitor", "run_pilot")
_PFN_MODULES = (
    "__init__", "contracts", "estimation", "intervals", "phase3_hierarchical",
    "phase5_large_network", "phase5_meta_bayes", "phase6_limited_outcomes",
    "phase6_masked_hypersci", "phase6_sparse_classical", "phase6_sparse_deep",
    "runtime", "simulation", "sparse_classical_calibration", "sparse_frozen_gaussian",
    "sparse_posterior_calibration", "_vendor/__init__", "_vendor/causalpfn",
    "_vendor/hypersci", "_vendor/hinet/__init__", "_vendor/hinet/components",
    "_vendor/hinet/hinet", "_vendor/hinet/layers")
_SCRIPTS = (
    "__init__", "analyze_node_support", "diagnose_e1_5_intervals", "e1_5_star",
    "e1_5_star_analysis", "e1_5_star_ate", "e1_5_star_resources",
    "e1_5_star_resource_worker", "e1_5_star_timing", "phase3_kernel_controls",
    "phase4_strong_controls", "phase5_frozen_controls", "phase5_strong_controls",
    "test_e1_5_star", "test_e1_5_star_ate", "test_e1_5_star_resources", "test_node_support")
_ENTRYPOINTS = ("benchmark", "evaluate", "make_figures", "make_table", "predict",
                "predict_current", "regional_inference")
SOURCE_ALLOWLIST = tuple(
    [SourceFile("star_v2/"+n+".py", "implementation") for n in _STAR_MODULES] +
    [SourceFile("star_v2/tests/test_"+n+".py", "test") for n in _STAR_TESTS] +
    [SourceFile("star_v2/"+n, "documentation") for n in (
        "README.md", "STAGE2_ACCEPTANCE.md", "STAGE3_ACCEPTANCE.md", "STAGE4_SOURCE_AUDIT.md")] +
    [SourceFile("star_v2/protocol/"+n, "protocol") for n in (
        "E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md", "freeze_record.json", "rng_registry.json")] +
    [SourceFile("pfn_social/"+n+".py", "dependency") for n in _PFN_MODULES] +
    [SourceFile("scripts/"+n+".py", "dependency") for n in _SCRIPTS] +
    [SourceFile(n+".py", "entrypoint") for n in _ENTRYPOINTS] +
    [SourceFile(n, "configuration") for n in (
        "experiment.json", "regional_selected.json", "models/kernel_components.json",
        "models/kernel_mixture.json", "requirements.txt")])


def _specs(files):
    require(type(files) is tuple and files and all(type(f) is SourceFile for f in files),
            "Explicit nonempty SourceFile tuple required")
    names = [f.path.casefold() for f in files]
    require(len(set(names)) == len(names), "Duplicate/case-colliding source path")
    return tuple(sorted(files, key=lambda f: f.path))


def _snapshot_id(records):
    return sha256(_json_bytes(records))


def validate_source_manifest(document):
    require(isinstance(document, Mapping) and set(document) == {
        "schema_version", "snapshot_id", "runtime_root", "git_head", "files", "archive"},
        "Source manifest schema mismatch")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1
            and document["runtime_root"] == "." and _digest(document["git_head"], 40),
            "Source manifest identity mismatch")
    records = document["files"]
    require(isinstance(records, (list, tuple)) and records, "Empty source manifest")
    specs = []
    for record in records:
        require(isinstance(record, Mapping) and set(record) == {"path", "role", "bytes", "sha256"},
                "Source file schema mismatch")
        specs.append(SourceFile(record["path"], record["role"]))
        require(type(record["bytes"]) is int and record["bytes"] >= 0
                and _digest(record["sha256"]), "Invalid source bytes/hash")
    ordered = _specs(tuple(specs))
    require([f.path for f in ordered] == [r["path"] for r in records], "Source manifest must be path-sorted")
    require(document["snapshot_id"] == _snapshot_id(records), "Snapshot ID mismatch")
    archive = document["archive"]
    require(isinstance(archive, Mapping) and set(archive) == {"path", "bytes", "sha256"}
            and archive["path"] == ARCHIVE_NAME and type(archive["bytes"]) is int
            and archive["bytes"] > 0 and _digest(archive["sha256"]), "Archive identity mismatch")
    return _freeze(document)


@dataclass(frozen=True, slots=True)
class SourceSnapshotReceipt:
    directory: str
    snapshot_id: str
    manifest_sha256: str
    archive_sha256: str


def _fresh_directory(destination):
    target = Path(destination).absolute()
    _no_links(target)
    require(not target.exists(), "Output already exists; never overwrite")
    _root(target.parent)
    return target


def _zip_bytes(payloads):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, blob in sorted(payloads.items()):
            info = zipfile.ZipInfo(path, ZIP_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, blob)
    return out.getvalue()


def build_source_snapshot(source_root, destination, *, git_head, files=SOURCE_ALLOWLIST):
    """Create-only delivery. A source byte buffer supplies BOTH ZIP and hashes.

    No Git, imports of snapshot project modules or model loads are performed.
    Failure never overwrites/deletes old outputs; a failed partial new delivery
    is not a valid receipt and must not be used.
    """
    root = _root(source_root)
    target = _fresh_directory(destination)
    require(not target.resolve().is_relative_to(root), "Delivery must be outside source root")
    require(_digest(git_head, 40), "Explicit Git HEAD required")
    specs = _specs(files)
    payloads = {f.path: _read_once(root, f.path) for f in specs}
    records = [{"path": f.path, "role": f.role, "bytes": len(payloads[f.path]),
                "sha256": sha256(payloads[f.path])} for f in specs]
    archive = _zip_bytes(payloads)
    manifest = {"schema_version": 1, "snapshot_id": _snapshot_id(records),
        "runtime_root": ".", "git_head": git_head, "files": records,
        "archive": {"path": ARCHIVE_NAME, "bytes": len(archive), "sha256": sha256(archive)}}
    validate_source_manifest(manifest)
    header = _json_bytes(manifest)
    target.mkdir()  # exclusive; races fail rather than overwrite
    for name, blob in ((ARCHIVE_NAME, archive), (MANIFEST_NAME, header)):
        with (target/name).open("xb") as stream:
            stream.write(blob)
    return SourceSnapshotReceipt(str(target), manifest["snapshot_id"],
                                 sha256(header), sha256(archive))


def _archive_payloads(archive_bytes, manifest):
    require(len(archive_bytes) == manifest["archive"]["bytes"]
            and sha256(archive_bytes) == manifest["archive"]["sha256"], "Archive bytes/hash mismatch")
    expected = {r["path"]: r for r in manifest["files"]}
    payloads = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = [info.filename for info in archive.infolist()]
            for name in names:
                relative_path(name)
            require(len(set(n.casefold() for n in names)) == len(names), "Duplicate ZIP path")
            require(names == sorted(expected), "ZIP/manifest file set or order mismatch")
            for info in archive.infolist():
                require(not info.is_dir() and stat.S_ISREG(info.external_attr >> 16)
                        and not info.flag_bits & 1, "ZIP nonregular/encrypted member")
                record = expected[info.filename]
                require(info.file_size == record["bytes"], "ZIP member size mismatch")
                blob = archive.read(info)
                require(sha256(blob) == record["sha256"], "ZIP source hash mismatch")
                payloads[info.filename] = blob
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ProvenanceError("Damaged source ZIP") from exc
    return payloads


def _read_delivery(directory, expected_manifest_sha256):
    root = _root(directory)
    require({p.name for p in root.iterdir()} == {ARCHIVE_NAME, MANIFEST_NAME},
            "Missing/extra delivery files")
    header = _read_once(root, MANIFEST_NAME)
    require(_digest(expected_manifest_sha256) and sha256(header) == expected_manifest_sha256,
            "Trusted source manifest SHA256 mismatch")
    try:
        manifest = validate_source_manifest(strict_json(header.decode("ascii")))
    except (UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise ProvenanceError("Invalid source manifest: "+str(exc)) from exc
    payloads = _archive_payloads(_read_once(root, ARCHIVE_NAME), manifest)
    return manifest, payloads


def verify_source_snapshot(directory, *, expected_manifest_sha256):
    """Verify anchored manifest, ZIP metadata paths, membership and every byte."""
    manifest, _ = _read_delivery(directory, expected_manifest_sha256)
    return manifest


def extract_source_snapshot(directory, destination, *, expected_manifest_sha256):
    manifest, payloads = _read_delivery(directory, expected_manifest_sha256)
    target = _fresh_directory(destination)
    target.mkdir()
    for name, blob in payloads.items():
        path = target.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        _no_links(path.parent)
        with path.open("xb") as stream:
            stream.write(blob)
    verify_runtime_tree(target, manifest)
    return manifest


def verify_runtime_tree(snapshot_root, manifest):
    manifest = validate_source_manifest(manifest)
    root = _root(snapshot_root)
    expected = {r["path"]: r for r in manifest["files"]}
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            _no_links(Path(directory)/name)
        for name in files:
            actual.add((Path(directory)/name).relative_to(root).as_posix())
    require(actual == set(expected), "Runtime tree missing/extra files")
    for name, record in expected.items():
        blob = _read_once(root, name)
        require(len(blob) == record["bytes"] and sha256(blob) == record["sha256"],
                "Runtime tree source hash mismatch")
    return True


def verify_runtime_modules(modules, snapshot_root, manifest, *, required_modules=()):
    """Check actual module __file__ locations and canonical source hashes.

    Pass the loaded project modules explicitly (e.g. a filtered sys.modules
    mapping). This is a path/content identity gate, not proof of execution or a
    general sandbox. No module is imported or executed by this function.
    """
    manifest = validate_source_manifest(manifest)
    root = _root(snapshot_root)
    require(isinstance(modules, Mapping) and modules, "Loaded project modules required")
    require(set(required_modules) <= set(modules), "Required project module missing")
    records = {r["path"]: r for r in manifest["files"]}
    verified = []
    for name, module in sorted(modules.items()):
        require(type(module) is ModuleType and module.__name__ == name, "Actual module object/name required")
        filename = getattr(module, "__file__", None)
        require(type(filename) is str, "Module __file__ missing")
        path = Path(filename).absolute()
        _no_links(path)
        real = path.resolve(strict=True)
        require(real.is_relative_to(root), "Project module loaded outside Snapshot root")
        rel = real.relative_to(root).as_posix()
        canonical = name.replace(".", "/")
        require(rel in (canonical+".py", canonical+"/__init__.py") and rel in records,
                "Module name/path not in Source Manifest")
        spec = getattr(module, "__spec__", None)
        if spec is not None and getattr(spec, "origin", None) is not None:
            require(Path(spec.origin).resolve() == real, "Module origin/__file__ conflict")
        blob = _read_once(root, rel)
        require(len(blob) == records[rel]["bytes"] and sha256(blob) == records[rel]["sha256"],
                "Loaded module source hash mismatch")
        verified.append({"module": name, "path": rel, "sha256": records[rel]["sha256"]})
    return _freeze(verified)


def collect_environment(dependencies=("numpy", "torch", "scipy", "networkx",
                                      "torch-geometric", "threadpoolctl", "matplotlib")):
    """Observe stdlib/package metadata only; no GPU/model import/probe."""
    import platform
    import sys
    from importlib.metadata import version, PackageNotFoundError
    observed = lambda v: {"status": "OBSERVED", "value": v}
    versions = {}
    for name in dependencies:
        try:
            versions[name] = observed(version(name))
        except PackageNotFoundError:
            versions[name] = {"status": "UNAVAILABLE", "reason": "Distribution metadata not found"}
    cpu = platform.processor()
    return _freeze({"python": observed({"version": platform.python_version(), "executable": sys.executable}),
        "os": observed(platform.platform()),
        "cpu": observed(cpu) if cpu else {"status": "UNAVAILABLE", "reason": "platform.processor was empty"},
        "gpu": {"status": "NOT_MEASURED", "reason": "No GPU probe performed"},
        "dependencies": versions})


def _observation(record):
    require(isinstance(record, Mapping) and record.get("status") in
            ("OBSERVED", "UNAVAILABLE", "NOT_MEASURED"), "Explicit observation status required")
    if record["status"] == "OBSERVED":
        require(set(record) == {"status", "value"} and record["value"] is not None,
                "Observed environment needs value")
    else:
        require(set(record) == {"status", "reason"} and type(record["reason"]) is str
                and bool(record["reason"].strip()), "Unavailable observation needs reason")


def validate_run_manifest(document, *, source_manifest=None):
    require(isinstance(document, Mapping) and set(document) == {
        "schema_version", "run_id", "run_kind", "protocol", "git", "source_snapshot",
        "entrypoint", "environment", "model_assets", "task_ledger_manifest_sha256",
        "registry_seal_sha256"}, "Run manifest schema mismatch")
    _json_bytes(document)  # no arrays, opaque objects or nonfinite values
    require(type(document["schema_version"]) is int and document["schema_version"] == 1
            and type(document["run_id"]) is str and bool(document["run_id"].strip())
            and document["run_kind"] in ("synthetic", "pilot", "test"), "Run identity required")
    require(dict(document["protocol"]) == {
        "protocol_id": CONFIG.protocol_id, "protocol_version": CONFIG.protocol_version,
        "protocol_sha256": CONFIG.protocol_sha256}, "Frozen protocol mismatch")
    git = document["git"]
    require(isinstance(git, Mapping) and set(git) == {"head", "branch", "status_short"}
            and _digest(git["head"], 40) and type(git["branch"]) is str and git["branch"].strip()
            and isinstance(git["status_short"], (list, tuple))
            and all(type(s) is str for s in git["status_short"]), "Git provenance required")
    snapshot = document["source_snapshot"]
    require(isinstance(snapshot, Mapping) and set(snapshot) == {
        "snapshot_id", "manifest_sha256", "archive_sha256"}
        and all(_digest(v) for v in snapshot.values()), "Source Snapshot identity required")
    relative_path(document["entrypoint"])
    environment = document["environment"]
    require(isinstance(environment, Mapping) and set(environment) == {
        "python", "os", "cpu", "gpu", "dependencies"}, "Complete environment schema required")
    for field in ("python", "os", "cpu", "gpu"):
        _observation(environment[field])
    require(isinstance(environment["dependencies"], Mapping), "Dependency observations required")
    for record in environment["dependencies"].values():
        _observation(record)
    assets = document["model_assets"]
    require(isinstance(assets, (tuple, list)), "Explicit model asset declarations required")
    seen = set()
    for asset in assets:
        require(isinstance(asset, Mapping) and set(asset) == {"path", "status", "sha256", "bytes"},
                "Asset schema mismatch")
        relative_path(asset["path"])
        require(asset["path"].casefold() not in seen, "Duplicate asset identity")
        seen.add(asset["path"].casefold())
        require(asset["status"] in ("DECLARED", "VERIFIED", "UNAVAILABLE"), "Asset provenance status required")
        require((_digest(asset["sha256"]) if asset["status"] != "UNAVAILABLE" else
                 asset["sha256"] is None), "Asset hash/status conflict")
        require(asset["bytes"] is None or type(asset["bytes"]) is int and asset["bytes"] >= 0,
                "Invalid asset size")
        require(asset["status"] != "VERIFIED" or asset["bytes"] is not None,
                "Verified asset needs a measured size")
    require(_digest(document["task_ledger_manifest_sha256"]) and _digest(document["registry_seal_sha256"]),
            "Ledger/Registry identities required")
    if source_manifest is not None:
        source = validate_source_manifest(source_manifest)
        require(document["git"]["head"] == source["git_head"]
                and document["source_snapshot"] == {
                    "snapshot_id": source["snapshot_id"],
                    "manifest_sha256": sha256(_json_bytes(source)),
                    "archive_sha256": source["archive"]["sha256"]}
                and document["entrypoint"] in {r["path"] for r in source["files"]},
                "Run/Snapshot manifest binding mismatch")
    return _freeze(document)


def build_run_manifest(*, run_id, run_kind, source_manifest, source_manifest_sha256,
                       git_identity, entrypoint, environment, model_assets,
                       task_ledger_manifest_sha256, registry_seal_sha256):
    """Schema construction only. Caller supplies observed run data; no runner."""
    source = validate_source_manifest(source_manifest)
    require(sha256(_json_bytes(source)) == source_manifest_sha256, "Source manifest receipt mismatch")
    require(git_identity["head"] == source["git_head"], "Run/Snapshot Git HEAD conflict")
    require(entrypoint in {r["path"] for r in source["files"]} and entrypoint.endswith(".py"),
            "Entrypoint must be an archived source file")
    document = {"schema_version": 1, "run_id": run_id, "run_kind": run_kind,
        "protocol": {"protocol_id": CONFIG.protocol_id, "protocol_version": CONFIG.protocol_version,
                     "protocol_sha256": CONFIG.protocol_sha256},
        "git": git_identity, "source_snapshot": {"snapshot_id": source["snapshot_id"],
            "manifest_sha256": source_manifest_sha256, "archive_sha256": source["archive"]["sha256"]},
        "entrypoint": entrypoint, "environment": environment, "model_assets": model_assets,
        "task_ledger_manifest_sha256": task_ledger_manifest_sha256,
        "registry_seal_sha256": registry_seal_sha256}
    return validate_run_manifest(document, source_manifest=source)


def export_task_manifest(registry, directory):
    """Existing full-ledger format; ALL Attempt history is retained."""
    return artifacts.write_snapshot(registry, directory)


def load_task_manifest(directory, *, expected_manifest_sha256, for_evaluation=True):
    require(type(for_evaluation) is bool, "Explicit evaluation boolean required")
    ledger = artifacts.read_snapshot(directory, expected_manifest_sha256=expected_manifest_sha256)
    if for_evaluation:
        unique_attempts(ledger, ledger.requests)
    return ledger
