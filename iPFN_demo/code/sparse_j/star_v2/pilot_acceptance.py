"""Ledger-only Pilot acceptance, derived from Stage 7 ledger checks.

No estimators, task generation, retries, selection, model loading or statistics.
The initial Run Manifest anchor is retained; completion anchors ledger_final.
"""
from pathlib import Path
import hashlib
from . import artifacts, data_artifacts as data, provenance
from .task_registry import strict_json

def validate_pilot_ledgers(root, *, run_manifest_sha256, expected_tasks=30,
                           expected_requests=170, require_complete=True):
    root = Path(root)
    def load(path):
        return strict_json(path.read_text(encoding="ascii"))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance.require(digest(root/"run_manifest.json")==run_manifest_sha256,
                       "Run Manifest receipt mismatch")
    run = load(root/"run_manifest.json")
    provenance.validate_run_manifest(run)
    provenance.require(run["run_kind"]=="pilot", "Pilot Run Manifest required")
    initial_hash = run["task_ledger_manifest_sha256"]
    initial = artifacts.read_snapshot(root/"ledger_initial",
                                      expected_manifest_sha256=initial_hash)
    completion = load(root/"ledger_completion.json")
    provenance.require(set(completion)=={
        "status","registry_seal_sha256","initial_manifest_sha256","final_manifest_sha256",
        "expected_tasks","expected_requests","recorded_attempts","unexecuted_request_keys","failure"},
        "Completion schema mismatch")
    provenance.require(completion["status"] in ("COMPLETE","INCOMPLETE"),
                       "Explicit completion status required")
    final = artifacts.read_snapshot(root/"ledger_final",
        expected_manifest_sha256=completion["final_manifest_sha256"])
    provenance.require(initial.sealed and final.sealed and not initial.attempts
        and initial.seal_sha256==final.seal_sha256==run["registry_seal_sha256"]
        ==completion["registry_seal_sha256"]
        and completion["initial_manifest_sha256"]==initial_hash,
        "Initial/final/Run identity mismatch")
    provenance.require(dict(initial.tasks)==dict(final.tasks)
        and dict(initial.requests)==dict(final.requests),
        "Initial/final task or request universe changed")
    ledger_dirs={p.name for p in root.iterdir() if p.is_dir()
                 and (p.name.startswith("ledger_after_") or p.name in ("ledger_initial","ledger_final"))}
    provenance.require(ledger_dirs=={"ledger_initial","ledger_final"},
                       "Only initial/final Ledger snapshots allowed")
    provenance.require(len(final.tasks)==expected_tasks==completion["expected_tasks"]
        and len(final.requests)==expected_requests==completion["expected_requests"]
        and len(final.attempts)==completion["recorded_attempts"],
        "Task/request/Attempt counts mismatch")
    missing=[list(k) for k in final.unexecuted_requests()]
    provenance.require(missing==completion["unexecuted_request_keys"],
                       "Unexecuted request evidence mismatch")
    provenance.require(len({a.execution_key for a in final.attempts})==len(final.attempts)
        and all(a.document["attempt_no"]==1 for a in final.attempts),
        "Pilot requires one Attempt per executed request; no automatic selection")
    is_complete=completion["status"]=="COMPLETE"
    provenance.require(not is_complete or (
        not missing and len(final.attempts)==expected_requests and completion["failure"] is None),
        "Incomplete execution cannot be marked COMPLETE")
    provenance.require(is_complete or (
        type(completion["failure"]) is dict and set(completion["failure"])=={"type","message"}),
        "Incomplete run needs the original failure")
    provenance.require(not require_complete or is_complete,"Pilot execution is INCOMPLETE")
    records=data.read_execution_records(root,final,require_complete=True)
    provenance.require(len(records)==len(final.attempts),"Attempt output linkage mismatch")
    # Status/failure provenance and parent linkage are validated by ledger replay.
    return {"status":completion["status"],"tasks":len(final.tasks),
            "requests":len(final.requests),"attempts":len(final.attempts),
            "unexecuted":len(missing),"seal_sha256":final.seal_sha256,
            "execution_records":len(records),"failed_attempts":sum(
                a.document["execution_status"]!="OK" for a in final.attempts)}
