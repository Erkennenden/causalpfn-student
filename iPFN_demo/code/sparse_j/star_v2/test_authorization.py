"""Explicit user authorization binding for Test. No approval is inferred."""
from pathlib import Path
from .config import CONFIG
from . import test_identification as identity, data_artifacts as data
from .task_registry import strict_json

def pending_template(*,snapshot_id=None):
    return {"schema_version":1,"kind":"TEST_COHORT_USER_AUTHORIZATION",
        "approval_status":"PENDING","authority":None,"decision_reference":None,
        "run_id":None,"cohort_code":2,"protocol_sha256":CONFIG.protocol_sha256,
        "snapshot_id":snapshot_id,"gate_sha256":None,
        "identification_sha256":None,"approval_sha256":None,
        "target_id":CONFIG.target_id,"proposed_identification":"POINT",
        "basis_sha256":identity.sha(data.json_bytes(identity.BASIS)),
        "effects":list(CONFIG.effect_order)}

def validate_authorization(document, *,run_id,snapshot_id,gate_sha256,
                           identification_sha256,approval_sha256):
    expected=pending_template(snapshot_id=snapshot_id)
    identity.require(type(document) is dict and set(document)==set(expected),
                     "Test authorization schema mismatch")
    identity.require(document["approval_status"]=="APPROVED",
                     "Formal Test user authorization is PENDING/missing; no inference from Pilot")
    identity.require(all(type(document[k]) is str and document[k].strip()
        for k in ("authority","decision_reference")),"Explicit user approval reference required")
    expected.update(approval_status="APPROVED",authority=document["authority"],
        decision_reference=document["decision_reference"],run_id=run_id,
        gate_sha256=gate_sha256,identification_sha256=identification_sha256,
        approval_sha256=approval_sha256)
    identity.require(type(run_id) is str and run_id.strip()
        and all(identity.digest(v) for v in (snapshot_id,gate_sha256,
            identification_sha256,approval_sha256)),"Complete authorization anchors required")
    identity.require(document==expected and type(document["cohort_code"]) is int,
                     "Test authorization Run/Snapshot/Gate/Identification binding mismatch")
    return document

def read_authorization(path, *,expected_sha256,**bindings):
    identity.require(path is not None and identity.digest(expected_sha256),
                     "Explicit Test user authorization file/SHA256 required")
    raw=Path(path).read_bytes()
    identity.require(identity.sha(raw)==expected_sha256,"Test authorization SHA256 mismatch")
    return validate_authorization(strict_json(raw.decode("ascii")),**bindings)
