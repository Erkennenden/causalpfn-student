"""Test-scoped Identification receipts. No inference, model calls or RNG."""
from pathlib import Path
import hashlib
from .config import CONFIG
from .task_registry import strict_json
from . import data_artifacts as data

PARENT_SNAPSHOT_ID = "2ea9a449f061e02c3e2a851229ac7eda9a0290a7cc19af3d4d4f7466ff595d42"
PARENT_MANIFEST_SHA256 = "65a53c06fd52cd79da82c8300f5dcbcc6a956df699ff69a9fecefc9738fa9dbf"
PARENT_ARCHIVE_SHA256 = "c2b153b6fa1e95a1405194e1acd0737e42a9697fdbec0c0527355e652b59fc18"
EFFECTS = ("Direct", "Spillover", "Total")
# Fixed protocol interpretation; a caller must supply an independently approved
# Test decision. This module never manufactures approval from a Gate PASS.
BASIS = {
    "protocol_sections": ["3.4", "5.1", "5.2"],
    "scope": "pair", "reference_assignment": "iid Bernoulli(0.5)",
    "exposure": "neighbor treatment proportion > 0.5",
    "pair_inclusion_probability": 0.5, "observation_probability": 0.25,
    "joint_inclusion_probability": 0.125, "ht_weight": 8.0,
    "estimand": "N^-1 sum_i E[Y_i | T_i=t,S_i=s,mechanism]",
    "identification": "pair HT expectation over assignment, observation and noise equals finite-population state mean",
    "contrasts": ["10-00", "11-10", "11-00"],
    "context_node_outputs": "diagnostics; no separate nonparametric Identification assertion"
}

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def digest(value):
    return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)

def validate_approval(value, *, run_id):
    require(type(value) is dict and set(value)=={
        "schema_version","kind","approved","authority","decision_reference","run_id",
        "cohort_code","protocol_sha256","parent_snapshot_id","status","target_id","basis"},
        "Explicit Test POINT approval schema required")
    require(value["schema_version"]==1 and value["kind"]=="TEST_IDENTIFICATION_APPROVAL"
        and value["approved"] is True and type(value["authority"]) is str and value["authority"].strip()
        and type(value["decision_reference"]) is str and value["decision_reference"].strip(),
        "Explicit approval authority/reference required")
    require(value["run_id"]==run_id and type(run_id) is str and bool(run_id.strip())
        and type(value["cohort_code"]) is int and value["cohort_code"]==2
        and value["protocol_sha256"]==CONFIG.protocol_sha256
        and value["parent_snapshot_id"]==PARENT_SNAPSHOT_ID
        and value["status"]=="POINT" and value["target_id"]==CONFIG.target_id
        and value["basis"]==BASIS, "Test approval identity/basis mismatch")
    return value

def make_decision(approval, *, approval_sha256, run_id, snapshot_id):
    validate_approval(approval,run_id=run_id)
    require(digest(snapshot_id) and snapshot_id!=PARENT_SNAPSHOT_ID and digest(approval_sha256),
            "Derived snapshot and approval SHA256 required")
    return {
        "schema_version":1,"kind":"TEST_IDENTIFICATION_DECISION","run_id":run_id,
        "cohort_code":2,"status":"POINT","target_id":CONFIG.target_id,
        "protocol_sha256":CONFIG.protocol_sha256,"parent_snapshot_id":PARENT_SNAPSHOT_ID,
        "snapshot_id":snapshot_id,"root_seed":CONFIG.root_seed,"protocol_code":CONFIG.protocol_code,
        "scope":"pair","effects":list(EFFECTS),"basis":BASIS,"approval_sha256":approval_sha256
    }

def validate_decision(value, approval, *, approval_sha256, run_id, snapshot_id):
    require(value==make_decision(approval,approval_sha256=approval_sha256,run_id=run_id,
                                snapshot_id=snapshot_id), "Test Identification decision mismatch")
    return value

def read_decision(path, *, expected_sha256, approval_path, approval_sha256, run_id, snapshot_id):
    raw=Path(path).read_bytes(); approved=Path(approval_path).read_bytes()
    require(sha(raw)==expected_sha256 and sha(approved)==approval_sha256,
            "Identification/approval SHA256 mismatch")
    return validate_decision(strict_json(raw.decode("ascii")),strict_json(approved.decode("ascii")),
        approval_sha256=approval_sha256,run_id=run_id,snapshot_id=snapshot_id)

def write_decision(path, *, approval_path, approval_sha256, run_id, snapshot_id):
    """Create-only record from an explicit, separately approved Test decision."""
    approved=Path(approval_path).read_bytes()
    require(sha(approved)==approval_sha256,"Approval receipt mismatch")
    record=make_decision(strict_json(approved.decode("ascii")),approval_sha256=approval_sha256,
                         run_id=run_id,snapshot_id=snapshot_id)
    path=Path(path)
    return data._write(path.parent,path.name,data.json_bytes(record))
