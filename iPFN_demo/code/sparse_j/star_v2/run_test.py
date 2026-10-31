"""Frozen-scale Test entry point. PLAN_ONLY by default; no recovery/retry loop."""
from pathlib import Path
from dataclasses import dataclass
import argparse
import hashlib
import json
import zipfile
import time
from . import test_authorization
from .config import CONFIG
from .contracts import pair_id_for, context_id_for, PairTaskSpec, pair_from_private
from .estimation import EstimationRequest
from .task_registry import TaskRegistry, strict_json
from .model_adapters import context_requests_from_parent
from .support import method_capability
from . import run_pilot as shared, provenance, data_artifacts as data
from . import test_identification as identification

require=shared.require
sha=identification.sha

def test_plan(*, cohort_code=2):
    require(type(cohort_code) is int and cohort_code==CONFIG.test_cohort_code==2,
            "Only frozen Test cohort=2 allowed")
    mechanisms,repeats=CONFIG.cohort_limits(cohort_code)
    require((mechanisms,repeats,CONFIG.root_seed,CONFIG.protocol_code,CONFIG.N,CONFIG.M)==
            (200,3,20261030,2026103001,200,50), "Frozen Test configuration mismatch")
    methods=shared.METHODS
    contexts=tuple(m for m in methods if {"context.mu","context.ite"}<=method_capability(m).outputs)
    require(contexts==shared.CONTEXT_METHODS,"Method capability/count mapping changed")
    pairs=tuple({"mechanism_id":m,"pair_repeat_id":r,"pair_id":pair_id_for(2,m,r),
                 "context_ids":tuple(context_id_for(2,m,r,t) for t in (0,1))}
                for m in range(mechanisms) for r in range(repeats))
    n=len(pairs)
    return {"mode":"PLAN_ONLY","cohort_code":2,"mechanisms":mechanisms,
        "pairs_per_mechanism":repeats,"pairs":pairs,"methods":methods,"context_methods":contexts,
        "counts":{"pair_data_tasks":n,"context_data_tasks":2*n,"data_tasks":3*n,
                  "pair_requests":n*len(methods),"context_requests":2*n*len(contexts),
                  "requests":n*(len(methods)+2*len(contexts)),
                  "initial_attempts":n*(len(methods)+2*len(contexts))},
        "root_seed":CONFIG.root_seed,"protocol_code":CONFIG.protocol_code,
        "protocol_sha256":CONFIG.protocol_sha256,"models_called":False,"dgp_generated":False}

@dataclass(frozen=True,slots=True)
class TestRequest:
    method_id: str
    task: PairTaskSpec
    parent: EstimationRequest
    children: tuple

def build_test_registry(pair_tasks, *, run_id, decision):
    plan=test_plan();tasks=tuple(pair_tasks)
    require(decision["run_id"]==run_id and decision["cohort_code"]==2
        and decision["status"]=="POINT" and decision["target_id"]==CONFIG.target_id
        and decision["protocol_sha256"]==CONFIG.protocol_sha256,
        "Explicit bound Test POINT decision required")
    expected={(p["mechanism_id"],p["pair_repeat_id"]) for p in plan["pairs"]}
    require(len(tasks)==len(expected) and all(type(t) is PairTaskSpec and t.cohort_code==2 for t in tasks)
        and {(t.mechanism_id,t.pair_repeat_id) for t in tasks}==expected,
        "Incomplete, duplicate or non-Test Pair inventory")
    registry=TaskRegistry();records=[]
    for task in sorted(tasks,key=lambda t:(t.mechanism_id,t.pair_repeat_id)):
        for public in (task,)+task.context_refs:
            registry.register_task(public)
        for method in plan["methods"]:
            rid=run_id+":"+method+":"+task.pair_id
            parent=EstimationRequest.from_task(task,status=decision["status"],target_id=decision["target_id"],
                request_id=rid,requested_outputs=shared._outputs(method))
            registry.register_request(method,parent);children=()
            if method in plan["context_methods"]:
                children=context_requests_from_parent(task,parent,request_ids=(rid+":0",rid+":1"))
                for child in children:
                    registry.register_request(method,child)
            records.append(TestRequest(method,task,parent,children))
    registry.seal(expected_task_ids=tuple(registry.tasks),expected_request_keys=tuple(registry.requests))
    require(len(registry.tasks)==plan["counts"]["data_tasks"]
        and len(registry.requests)==plan["counts"]["requests"],"Frozen inventory count mismatch")
    return registry,tuple(records)

def verify_lineage(delivery, source):
    require(source["snapshot_id"]!=identification.PARENT_SNAPSHOT_ID,"Derived Test snapshot required")
    with zipfile.ZipFile(Path(delivery)/source["archive"]["path"]) as archive:
        lineage=strict_json(archive.read("SNAPSHOT_PARENT.json").decode("ascii"))
    require(lineage["parent_snapshot_id"]==identification.PARENT_SNAPSHOT_ID
        and lineage["parent_manifest_sha256"]==identification.PARENT_MANIFEST_SHA256
        and lineage["parent_archive_sha256"]==identification.PARENT_ARCHIVE_SHA256,
        "Wrong Stage 8.3-C parent lineage")
    protocol=next((record for record in source["files"] if record["path"]==
        "star_v2/protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md"),None)
    require(protocol is not None and protocol["sha256"]==CONFIG.protocol_sha256,
            "Candidate source identity mismatch")
    return lineage

def verify_runtime_files(runtime, source):
    runtime=Path(runtime).resolve()
    for record in source["files"]:
        path=runtime/record["path"]
        require(not path.is_symlink() and path.resolve().is_relative_to(runtime),
                "Runtime source path/link mismatch")
        blob=path.read_bytes()
        require(len(blob)==record["bytes"] and sha(blob)==record["sha256"],
                "Runtime source hash mismatch: "+record["path"])

def validate_output_isolation(output_root, *, runtime_root, source_delivery):
    """Read-only guard shared by execution and independent acceptance.

    Resolve paths for containment, but retain the existing no-link/reparse
    validation on the original spelling. A .git pointer FILE is as protective
    as a .git directory. No Git command, pointer parsing or output write occurs.
    """
    target=Path(output_root).absolute()
    provenance._no_links(target)
    resolved=target.resolve()
    for label,protected in (("Runtime",runtime_root),("Source Delivery",source_delivery)):
        require(not resolved.is_relative_to(Path(protected).resolve()),
                "Output must be outside "+label)
    for ancestor in (resolved,)+tuple(resolved.parents):
        try:
            (ancestor/".git").lstat()
        except FileNotFoundError:
            continue
        raise ValueError("Output must be outside Git worktrees: "+str(ancestor))
    return target

def preflight(*, source_delivery, source_manifest_sha256, gate_path, gate_sha256,
              identification_path, identification_sha256, approval_path, approval_sha256,
              output_root, run_id,test_authorization_path=None,test_authorization_sha256=None):
    target=validate_output_isolation(output_root,runtime_root=shared.ROOT,
                                     source_delivery=source_delivery)
    source=provenance.verify_source_snapshot(source_delivery,expected_manifest_sha256=source_manifest_sha256)
    verify_lineage(source_delivery,source)
    gate=shared.read_gate(gate_path,expected_sha256=gate_sha256,
        source_snapshot_id=source["snapshot_id"],asset_root=shared.ROOT)
    # Old Stage 8.4 PASS cannot satisfy a newly derived source identity.
    shared.verify_execution_source(gate,source)
    verify_runtime_files(shared.ROOT,source)
    entry=next(r for r in source["files"] if r["path"]=="star_v2/run_test.py")
    require(Path(__file__).resolve()==shared.ROOT/"star_v2"/"run_test.py"
        and sha(Path(__file__).read_bytes())==entry["sha256"],"Test entry loaded outside verified runtime")
    decision=identification.read_decision(identification_path,expected_sha256=identification_sha256,
        approval_path=approval_path,approval_sha256=approval_sha256,run_id=run_id,
        snapshot_id=source["snapshot_id"])
    test_authorization.read_authorization(test_authorization_path,
        expected_sha256=test_authorization_sha256,run_id=run_id,snapshot_id=source["snapshot_id"],
        gate_sha256=gate_sha256,identification_sha256=identification_sha256,
        approval_sha256=approval_sha256)
    require(not (target.parent/(target.name+".execution_receipt.json")).exists(),
            "Execution receipt already exists; no overwrite/recovery")
    target=provenance._fresh_directory(target)
    return source,gate,decision,target

def _generate_test_data():
    """Only called after explicit execution, all receipts and fresh path checks."""
    from .dgp import generate_star_mechanism
    from .pair import generate_pair
    mechanisms={};private=[];pairs=[]
    plan=test_plan()
    for m in range(plan["mechanisms"]):
        mechanisms[m]=generate_star_mechanism(cohort_code=2,mechanism_id=m)
        for repeat in range(plan["pairs_per_mechanism"]):
            item=generate_pair(mechanisms[m],pair_repeat_id=repeat)
            private.append(item);pairs.append(pair_from_private(item))
    return mechanisms,tuple(private),tuple(pairs)

def _persist_inputs(target, mechanisms, private, pairs):
    truths={m:data.save_evaluator_arrays(target,"truth","mechanism_"+str(m),
        {"mu":v.mu,"ite":v.ite,"ate":v.ate},{"mechanism_id":m}) for m,v in mechanisms.items()}
    sources={}
    require(len(private)==len(pairs),"Private/public Pair association mismatch")
    for item,task in zip(private,pairs):
        receipts=list(truths[task.mechanism_id])
        for context in task.context_refs:
            receipts.extend(data.save_factual_inputs(target,context))
        receipts.extend(data.save_evaluator_arrays(target,"factual/simulator_private",task.pair_id,
            {"h":item.mechanism.h,"coefficients":item.mechanism.coefficients,"noise":item.noise,
             "leaf_treatment":item.leaf_treatment,"observed_indices":item.observed_indices,
             "full_y0":item.contexts[0].full_outcomes,"full_y1":item.contexts[1].full_outcomes},
            {"pair_id":task.pair_id}))
        sources[task.pair_id]=tuple(receipts)
    return sources

def file_inventory(root):
    root=Path(root);records=[]
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(),"Symlink in output")
        if not p.is_file():
            continue
        relative=p.relative_to(root).as_posix()
        if relative in ("test_file_inventory.json","test_completion.json"):
            continue
        blob=p.read_bytes()
        records.append({"path":relative,"bytes":len(blob),"sha256":sha(blob)})
    return records

def preload_models(gate):
    """Original verified loader boundaries, called once each before any DGP."""
    adapters,construction=shared.measure_phase(lambda:shared.real_adapters(gate),"adapter_construction")
    require(set(adapters)=={"j","kernel-old","kernel-new"},"Exactly three frozen adapters required")
    loads={}
    for method in ("j","kernel-old","kernel-new"):
        model,window=shared.measure_phase(adapters[method]._get_model,"model_preload:"+method)
        require(model is not None,"Loader returned no model: "+method)
        loads[method]=window
    return adapters,{"adapter_construction":construction,"model_loads":loads,
        "model_setup_seconds":construction["wall_seconds"]+sum(
            window["wall_seconds"] for window in loads.values()),
        "definition":"adapter construction + three explicit original loader calls; no prediction"}

def execute_test(*, execute_test, source_delivery, source_manifest_sha256, gate_path, gate_sha256,
                 identification_path, identification_sha256, approval_path, approval_sha256,
                 output_root, run_id,test_authorization_path=None,test_authorization_sha256=None):
    started=time.perf_counter()
    require(execute_test is True,"Explicit --execute-test required")
    test_plan()
    source,gate,decision,target=preflight(source_delivery=source_delivery,
        source_manifest_sha256=source_manifest_sha256,gate_path=gate_path,gate_sha256=gate_sha256,
        identification_path=identification_path,identification_sha256=identification_sha256,
        approval_path=approval_path,approval_sha256=approval_sha256,output_root=output_root,run_id=run_id,
        test_authorization_path=test_authorization_path,test_authorization_sha256=test_authorization_sha256)
    validate_output_isolation(target,runtime_root=shared.ROOT,source_delivery=source_delivery)
    target.mkdir()
    registry=None
    phase="authorization_evidence"
    timing={"model_setup_seconds":0.0,"data_setup_seconds":0.0,"prediction_seconds":0.0,
            "j_interval_seconds":0.0,"execution_prediction_seconds":0.0,
            "persistence_audit_seconds":time.perf_counter()-started}
    try:
        section=time.perf_counter()
        for relative,path,expected in (
            ("identification_decision.json",identification_path,identification_sha256),
            ("identification_approval.json",approval_path,approval_sha256),
            ("test_authorization.json",test_authorization_path,test_authorization_sha256),
            ("real_model_gate.json",gate_path,gate_sha256)):
            blob=Path(path).read_bytes()
            require(sha(blob)==expected,"Evidence changed after preflight")
            data._write(target,relative,blob)
        timing["persistence_audit_seconds"]+=time.perf_counter()-section
        # No factual DGP or task materialization can occur before all three
        # original loader boundaries return successfully.
        phase="model_preload"
        adapters,setup=preload_models(gate)
        timing["model_setup_seconds"]=setup["model_setup_seconds"]
        phase="data_generation"
        generated,data_time=shared.measure_phase(_generate_test_data,"data_generation")
        mechanisms,private,pairs=generated
        phase="task_materialization"
        (registry,records),task_time=shared.measure_phase(
            lambda:build_test_registry(pairs,run_id=run_id,decision=decision),"task_materialization")
        timing["data_setup_seconds"]=data_time["wall_seconds"]+task_time["wall_seconds"]
        phase="registered_execution"
        section=time.perf_counter()
        initial=data.export_ledger(registry,target/"ledger_initial")
        manifest=provenance.build_run_manifest(run_id=run_id,run_kind="test",source_manifest=source,
            source_manifest_sha256=source_manifest_sha256,git_identity=gate["git"],
            entrypoint="star_v2/run_test.py",environment=provenance.collect_environment(),
            model_assets=tuple({"path":name,"status":"VERIFIED","sha256":digest,
                "bytes":(shared.ROOT/name).stat().st_size} for name,digest in shared.ASSETS.items()),
            task_ledger_manifest_sha256=initial.manifest_sha256,registry_seal_sha256=registry.seal_sha256)
        run_receipt=data._write(target,"run_manifest.json",data.json_bytes(manifest))
        sources=_persist_inputs(target,mechanisms,private,pairs)
        timing["persistence_audit_seconds"]+=time.perf_counter()-section
        for record in records:
            section=time.perf_counter()
            _,index=shared.execute_registered_request(registry,record,adapters,target,
                true_h=mechanisms[record.task.mechanism_id].h,
                source_receipts=sources[record.task.pair_id])
            elapsed=time.perf_counter()-section
            audit_started=time.perf_counter()
            runtime=strict_json(data.verify_file(target,index["runtime"]).decode("ascii"))
            point=runtime["runtime_seconds"]
            intervals=sum(row["seconds"] for row in runtime["interval_runtime_seconds"])
            require(point is not None and point>=0 and intervals>=0,
                    "Unavailable pair timing cannot be fabricated")
            timing["prediction_seconds"]+=point
            timing["j_interval_seconds"]+=intervals
            timing["execution_prediction_seconds"]+=point+intervals
            timing["persistence_audit_seconds"]+=elapsed-point-intervals+time.perf_counter()-audit_started
        section=time.perf_counter()
        _,integrity=shared.measure_phase(
            lambda:data.read_execution_records(target,registry,require_complete=True),"engineering_integrity")
        final=shared._save_final_ledger(registry,target,complete=True)
        shared.verify_execution_source(gate,source)
        verify_runtime_files(shared.ROOT,source)
        link=data._write(target,"identification_run_link.json",data.json_bytes({
            "schema_version":1,"run_id":run_id,"cohort_code":2,"status":"POINT",
            "identification_sha256":identification_sha256,"approval_sha256":approval_sha256,
            "test_authorization_sha256":test_authorization_sha256,
            "gate_sha256":gate_sha256,"source_manifest_sha256":source_manifest_sha256,
            "snapshot_id":source["snapshot_id"],"parent_snapshot_id":identification.PARENT_SNAPSHOT_ID,
            "run_manifest_sha256":run_receipt["sha256"],"registry_seal_sha256":registry.seal_sha256,
            "initial_ledger_sha256":initial.manifest_sha256,"final_ledger_sha256":final.manifest_sha256,
            "requests":len(registry.requests)}))
        timing["persistence_audit_seconds"]+=time.perf_counter()-section
        timing["total_wall_clock_seconds"]=time.perf_counter()-started
        data._write(target,"runtime/run_resources.json",data.json_bytes({
            "setup":setup,"data_generation":data_time,"task_materialization":task_time,"integrity":integrity,
            "timing":timing,"scope":"engineering integrity; no statistical evaluation",
            "timing_boundary":"execute_test API entry through pre-receipt resources; module import excluded",
            "persistence_boundary":"evidence, ledger, request checks/bookkeeping, failure mapping, I/O and audits; not pure disk time",
            "disk":shared.disk_occupancy(target)}))
        inventory=data._write(target,"test_file_inventory.json",data.json_bytes(file_inventory(target)))
        completion=data._write(target,"test_completion.json",data.json_bytes({
            "schema_version":1,"status":"COMPLETE","run_id":run_id,"cohort_code":2,
            "run_manifest_sha256":run_receipt["sha256"],"inventory_sha256":inventory["sha256"],
            "link_sha256":link["sha256"],"ledger_completion_sha256":sha(
                (target/"ledger_completion.json").read_bytes())}))
        full_elapsed=time.perf_counter()-started
        receipt_path=target.parent/(target.name+".execution_receipt.json")
        receipt=data._write(receipt_path.parent,receipt_path.name,data.json_bytes({
            "schema_version":1,"kind":"TEST_EXECUTION_RECEIPT","run_id":run_id,
            "run_manifest_sha256":run_receipt["sha256"],
            "completion_sha256":completion["sha256"],
            "total_wall_clock_seconds":full_elapsed,
            "final_receipt_persistence_seconds":full_elapsed-timing["total_wall_clock_seconds"],
            "boundary":"execute_test API entry through completed Run writes; excludes this external receipt write",
            "timing":timing}))
        return {"status":"COMPLETE","run_manifest_sha256":run_receipt["sha256"],
                "completion_sha256":completion["sha256"],"total_wall_clock_seconds":full_elapsed,
                "execution_receipt_sha256":receipt["sha256"],"execution_receipt_path":str(receipt_path)}
    except BaseException as error:
        if registry is not None and (target/"ledger_initial").is_dir() and not (target/"ledger_final").exists():
            try:
                shared._save_final_ledger(registry,target,complete=False,failure=error)
            except Exception as save_error:
                error.add_note("Incomplete Ledger persistence failed: "+str(save_error))
        try:
            data._write(target,"RUN_INCOMPLETE.json",data.json_bytes({
                "status":"INCOMPLETE","type":type(error).__name__,"message":str(error),
                "phase":phase,
                "formal_tasks_created":registry is not None,
                "elapsed_seconds":time.perf_counter()-started}))
        except Exception as marker_error:
            error.add_note("Incomplete marker could not be saved: "+str(marker_error))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=("test",),required=True)
    parser.add_argument("--execute-test",action="store_true")
    for name in ("source-delivery","source-manifest-sha256","gate-report","gate-sha256",
                 "identification","identification-sha256","approval","approval-sha256","output","run-id",
                 "test-authorization","test-authorization-sha256"):
        parser.add_argument("--"+name)
    args=parser.parse_args(argv)
    if not args.execute_test:
        print(json.dumps(test_plan(),sort_keys=True))
        return 0
    values=vars(args)
    require(all(v for k,v in values.items() if k not in ("cohort","execute_test")),"All Test receipts required")
    print(json.dumps(execute_test(execute_test=True,source_delivery=args.source_delivery,
        source_manifest_sha256=args.source_manifest_sha256,gate_path=args.gate_report,
        gate_sha256=args.gate_sha256,identification_path=args.identification,
        identification_sha256=args.identification_sha256,approval_path=args.approval,
        approval_sha256=args.approval_sha256,output_root=args.output,run_id=args.run_id,
        test_authorization_path=args.test_authorization,
        test_authorization_sha256=args.test_authorization_sha256),sort_keys=True))
    return 0

if __name__=="__main__":
    main()
