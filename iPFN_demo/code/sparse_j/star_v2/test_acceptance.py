"""Independent engineering acceptance. No estimators or scientific scoring.

Caller must retain the externally produced run/completion SHA256 receipts.
Self-consistent JSON alone is not unforgeable evidence of execution.
"""
from pathlib import Path
import argparse
import json
from . import artifacts, provenance, data_artifacts as data
from . import run_test as runner, test_identification as identification
from .task_registry import strict_json, request_from_document
from .config import CONFIG

require=runner.require
sha=identification.sha

def _read(root, name, expected=None):
    raw=(root/name).read_bytes()
    require(expected is None or sha(raw)==expected,"Receipt mismatch: "+name)
    return strict_json(raw.decode("ascii"))

def validate_universe(registry, *, run_id):
    """Independently enumerate expected keys; do not infer them from successes."""
    plan=runner.test_plan()
    expected_tasks=set();expected_requests={}
    for pair in plan["pairs"]:
        pid=pair["pair_id"];cids=tuple(pair["context_ids"])
        expected_tasks.update((pid,)+cids)
        for method in plan["methods"]:
            rid=run_id+":"+method+":"+pid
            expected_requests[(method,"pair",pid,rid)]=(pair,None)
            if method in plan["context_methods"]:
                for t,cid in enumerate(cids):
                    expected_requests[(method,"context",cid,rid+":"+str(t))]=(pair,rid)
    require(set(registry.tasks)==expected_tasks and set(registry.requests)==set(expected_requests),
            "Missing/extra Test task or request")
    require(len(registry.tasks)==plan["counts"]["data_tasks"]
        and len(registry.requests)==plan["counts"]["requests"],"Frozen Test counts mismatch")
    for key,(pair,parent) in expected_requests.items():
        request=request_from_document(registry.requests[key].document["request"])
        require(request.identity["cohort_code"]==2 and request.identity["cohort"]=="test"
            and request.identity["mechanism_id"]==pair["mechanism_id"]
            and request.identity["pair_repeat_id"]==pair["pair_repeat_id"]
            and request.parent_request_id==parent and request.status=="POINT"
            and request.target_id==CONFIG.target_id
            and tuple(request.requested_outputs)==runner.shared._outputs(key[0]),
            "Test request binding/provenance mismatch")
        require(request.pair_id==pair["pair_id"]
            and tuple(request.context_ids)==(tuple(pair["context_ids"]) if key[1]=="pair"
                else (key[2],)), "Test context/Pair link mismatch")
    return plan

def validate_test_ledgers(root, *, run_manifest_sha256, run_id):
    root=Path(root)
    require(not (root/"RUN_INCOMPLETE.json").exists(),"Run is explicitly INCOMPLETE")
    run=_read(root,"run_manifest.json",run_manifest_sha256)
    provenance.validate_run_manifest(run)
    require(run["run_kind"]=="test" and run["run_id"]==run_id
        and run["entrypoint"]=="star_v2/run_test.py","Formal Test Run Manifest required")
    initial=artifacts.read_snapshot(root/"ledger_initial",
        expected_manifest_sha256=run["task_ledger_manifest_sha256"])
    done=_read(root,"ledger_completion.json")
    require(set(done)=={"status","registry_seal_sha256","initial_manifest_sha256",
        "final_manifest_sha256","expected_tasks","expected_requests","recorded_attempts",
        "unexecuted_request_keys","failure"} and done["status"]=="COMPLETE"
        and done["failure"] is None and done["unexecuted_request_keys"]==[],
        "Incomplete/interrupted Ledger rejected")
    final=artifacts.read_snapshot(root/"ledger_final",
        expected_manifest_sha256=done["final_manifest_sha256"])
    require(initial.sealed and final.sealed and not initial.attempts
        and initial.seal_sha256==final.seal_sha256==run["registry_seal_sha256"]
        ==done["registry_seal_sha256"]
        and done["initial_manifest_sha256"]==run["task_ledger_manifest_sha256"]
        and dict(initial.tasks)==dict(final.tasks) and dict(initial.requests)==dict(final.requests),
        "Initial/final Ledger universe or seal changed")
    plan=validate_universe(final,run_id=run_id)
    attempts=final.attempts
    require(len(attempts)==done["recorded_attempts"]==plan["counts"]["initial_attempts"]
        and len(final.tasks)==done["expected_tasks"] and len(final.requests)==done["expected_requests"]
        and not final.unexecuted_requests()
        and len({a.execution_key for a in attempts})==len(attempts)
        and all(a.document["attempt_no"]==1 for a in attempts),
        "Missing/extra/retried initial Test Attempts")
    require({p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("ledger_")}
        =={"ledger_initial","ledger_final"},"Exactly initial/final Ledgers required")
    records=data.read_execution_records(root,final,require_complete=True)
    require(len(records)==len(attempts),"Attempt output completeness mismatch")
    # Existing byte and logical array validators are reused, never relaxed.
    descriptors={d["path"]:d for record in records for d in record["outputs"]}
    for name,descriptor in descriptors.items():
        if name.endswith(".npz"):
            require(name[:-4]+".json" in descriptors,"Array metadata missing")
            data.load_arrays(root,(descriptor,descriptors[name[:-4]+".json"]))
    for task in final.tasks.values():
        doc=task.document
        if doc["scope"]!="context":
            continue
        stem="factual/inputs/"+doc["context_ids"][0]
        require(stem+".npz" in descriptors and stem+".json" in descriptors,"Factual inputs missing")
        arrays,meta=data.load_arrays(root,(descriptors[stem+".npz"],descriptors[stem+".json"]))
        data.load_factual_inputs(root,(descriptors[stem+".npz"],descriptors[stem+".json"]))
        require(meta=={"context_id":doc["context_ids"][0],"pair_id":doc["pair_id"],
            "input_hash":doc["input_hashes"][0]} and arrays["outcomes_observed"].dtype.name=="float64",
            "Factual identity/dtype mismatch")
    return run,initial,final,records

def validate_test_run(root, *, run_manifest_sha256, completion_sha256, source_delivery,
                      source_manifest_sha256,execution_receipt_sha256=None,execution_receipt_path=None):
    """Read-only independent validation, anchored in external receipts."""
    root=runner.validate_output_isolation(root,runtime_root=runner.shared.ROOT,
                                          source_delivery=source_delivery)
    require(root.is_dir() and not root.is_symlink(),"Run directory required")
    completion=_read(root,"test_completion.json",completion_sha256)
    require(set(completion)=={"schema_version","status","run_id","cohort_code","run_manifest_sha256",
        "inventory_sha256","link_sha256","ledger_completion_sha256"}
        and completion["schema_version"]==1 and completion["status"]=="COMPLETE"
        and type(completion["cohort_code"]) is int and completion["cohort_code"]==2
        and completion["run_manifest_sha256"]==run_manifest_sha256,"Complete Test receipt required")
    receipt_path=Path(execution_receipt_path) if execution_receipt_path is not None else root.parent/(root.name+".execution_receipt.json")
    require(identification.digest(execution_receipt_sha256),"External execution receipt SHA256 required")
    execution_receipt=_read(receipt_path.parent,receipt_path.name,execution_receipt_sha256)
    require(execution_receipt["kind"]=="TEST_EXECUTION_RECEIPT"
        and execution_receipt["run_id"]==completion["run_id"]
        and execution_receipt["run_manifest_sha256"]==run_manifest_sha256
        and execution_receipt["completion_sha256"]==completion_sha256
        and execution_receipt["total_wall_clock_seconds"]>=0
        and execution_receipt["final_receipt_persistence_seconds"]>=0,
        "Execution timing receipt binding mismatch")
    inventory=_read(root,"test_file_inventory.json",completion["inventory_sha256"])
    require(inventory==runner.file_inventory(root),"Missing, extra or changed run file")
    link=_read(root,"identification_run_link.json",completion["link_sha256"])
    _read(root,"ledger_completion.json",completion["ledger_completion_sha256"])
    source=provenance.verify_source_snapshot(source_delivery,
        expected_manifest_sha256=source_manifest_sha256)
    runner.verify_lineage(source_delivery,source)
    run,initial,final,records=validate_test_ledgers(root,
        run_manifest_sha256=run_manifest_sha256,run_id=completion["run_id"])
    provenance.validate_run_manifest(run,source_manifest=source)
    require(set(link)=={"schema_version","run_id","cohort_code","status","identification_sha256",
        "approval_sha256","test_authorization_sha256","gate_sha256","source_manifest_sha256","snapshot_id","parent_snapshot_id",
        "run_manifest_sha256","registry_seal_sha256","initial_ledger_sha256","final_ledger_sha256","requests"}
        and link["schema_version"]==1 and link["run_id"]==run["run_id"]
        and link["cohort_code"]==2 and link["status"]=="POINT"
        and link["snapshot_id"]==source["snapshot_id"]
        and link["source_manifest_sha256"]==source_manifest_sha256
        and link["parent_snapshot_id"]==identification.PARENT_SNAPSHOT_ID
        and link["run_manifest_sha256"]==run_manifest_sha256
        and link["registry_seal_sha256"]==final.seal_sha256
        and link["initial_ledger_sha256"]==run["task_ledger_manifest_sha256"]
        and link["final_ledger_sha256"]==sha((root/"ledger_final/ledger_manifest.json").read_bytes())
        and link["requests"]==len(final.requests),"Identification/Run linkage mismatch")
    identification.read_decision(root/"identification_decision.json",
        expected_sha256=link["identification_sha256"],approval_path=root/"identification_approval.json",
        approval_sha256=link["approval_sha256"],run_id=run["run_id"],snapshot_id=source["snapshot_id"])
    runner.test_authorization.read_authorization(root/"test_authorization.json",
        expected_sha256=link["test_authorization_sha256"],run_id=run["run_id"],
        snapshot_id=source["snapshot_id"],gate_sha256=link["gate_sha256"],
        identification_sha256=link["identification_sha256"],approval_sha256=link["approval_sha256"])
    gate=_read(root,"real_model_gate.json",link["gate_sha256"])
    runtime=Path(gate["runtime_root"]).resolve()
    runner.validate_output_isolation(root,runtime_root=runtime,source_delivery=source_delivery)
    require(not any((p/".git").exists() for p in (runtime,)+tuple(runtime.parents)),
            "Gate Runtime must be outside worktrees")
    runner.shared.validate_gate(gate,source_snapshot_id=source["snapshot_id"],asset_root=runtime)
    require(run["git"]==gate["git"],"Run/Gate Git identity mismatch")
    require({a["path"]:a["sha256"] for a in run["model_assets"]}==runner.shared.ASSETS
        and all(a["status"]=="VERIFIED" and (runtime/a["path"]).stat().st_size==a["bytes"]
                for a in run["model_assets"]),"Run asset receipts mismatch")
    for record in source["files"]:
        blob=(runtime/record["path"]).read_bytes()
        require(len(blob)==record["bytes"] and sha(blob)==record["sha256"],
                "Gate Runtime source changed: "+record["path"])
    return {"status":"PASS","scope":"engineering completeness only; no scientific performance assertion",
        "run_id":run["run_id"],"cohort_code":2,"tasks":len(final.tasks),"requests":len(final.requests),
        "attempts":len(final.attempts),"execution_records":len(records),
        "failed_attempts":sum(a.document["execution_status"]!="OK" for a in final.attempts),
        "registry_seal_sha256":final.seal_sha256,"source_snapshot_id":source["snapshot_id"]}

def write_acceptance_receipt(root, destination, **anchors):
    result=validate_test_run(root,**anchors)
    root=Path(root)
    target=runner.validate_output_isolation(destination,runtime_root=runner.shared.ROOT,
                                           source_delivery=anchors["source_delivery"])
    require(not target.resolve().is_relative_to(root.resolve()),
            "Acceptance receipt must be outside immutable Run")
    value={"schema_version":1,"kind":"INDEPENDENT_TEST_ACCEPTANCE","status":"PASS",
        "run_manifest_sha256":anchors["run_manifest_sha256"],
        "completion_sha256":anchors["completion_sha256"],
        "source_manifest_sha256":anchors["source_manifest_sha256"],
        "execution_receipt_sha256":anchors["execution_receipt_sha256"],
        "validator_sha256":sha(Path(__file__).read_bytes()),
        "gate_sha256":sha((root/"real_model_gate.json").read_bytes()),"result":result}
    return data._write(target.parent,target.name,data.json_bytes(value))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run-directory","run-manifest-sha256","completion-sha256",
                 "source-delivery","source-manifest-sha256","execution-receipt","execution-receipt-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--acceptance-output")
    args=parser.parse_args(argv)
    anchors=dict(run_manifest_sha256=args.run_manifest_sha256,
        completion_sha256=args.completion_sha256,source_delivery=args.source_delivery,
        source_manifest_sha256=args.source_manifest_sha256,
        execution_receipt_path=args.execution_receipt,execution_receipt_sha256=args.execution_receipt_sha256)
    result=(write_acceptance_receipt(args.run_directory,args.acceptance_output,**anchors)
            if args.acceptance_output else validate_test_run(args.run_directory,**anchors))
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    main()
