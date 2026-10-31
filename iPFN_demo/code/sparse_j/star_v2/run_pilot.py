"""Lean Pilot planning/execution plumbing. Default CLI is PLAN ONLY.

No model or DGP is run at import or planning. Execution requires an explicit
switch, an externally anchored REAL_MODEL_GATE PASS report, matching seven
asset hashes, a source snapshot receipt and explicit upstream Identification.
There is no Test/Formal entry, retry loop, ranking, calibration or bootstrap.
"""
import time as _time
_ENTRY_IMPORT_START=_time.perf_counter()
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import argparse
import hashlib
import json

from .config import CONFIG
from .contracts import pair_id_for, context_id_for, pair_from_private, PairTaskSpec
from .estimation import EstimationRequest
from .task_registry import TaskRegistry, Attempt, strict_json
from .support import ORDINARY_METHOD_IDS
from .model_adapters import context_requests_from_parent
from . import baselines, model_adapters, oracle_adapters, provenance
from . import data_artifacts as data
from .runtime_monitor import (RuntimeWindow, measure_pair, measure_phase,
                              MeasuredFailure, disk_occupancy)

METHODS=ORDINARY_METHOD_IDS+("oracle-context","oracle-pair")
CONTEXT_METHODS=("kernel-old","kernel-new","j","oracle-context")
ASSETS={
 "models/iPFN_J.pt":"a0dde8e928d7d48c0ef8581b81cf38fecd9525408bff7179b4df7ee761fed7a6",
 "models/feature_transform.npz":"b261372fea1cb241e4acbd9734eab5b9506f6478e90d90c0f5f3e633b7a3a737",
 "models/kernel_subspace_prior.npz":"02734745ee96c712da1494ced0960538a5ac4b8eeb31b2dfa70c90ccf5ebe130",
 "models/kernel_linear_prior.npz":"90e040ecdf4eba9d92e90f98c457dbffc24821498cc0f189c8f77cc52239806c",
 "models/kernel_components.json":"594aca0221f35e364d012e2067546f0ea4c61bb6cf89db8a87cbdd9f6b19fb3d",
 "models/kernel_mixture.json":"3e8a76ab443649831824c61e3157be497345c66d9c8548852dcf8d5320da3981",
 "models/kernel_calibration.pt":"6273c1abb1255a7d8159fbbd6558c56490915e60ebd312c7c36b33c26b064a2e"}
ROOT=Path(__file__).resolve().parents[1]
_ENTRY_IMPORT_SECONDS=_time.perf_counter()-_ENTRY_IMPORT_START


def require(value,message):
    if not value:raise ValueError(message)


def pilot_plan(*,cohort_code=CONFIG.pilot_cohort_code):
    require(type(cohort_code) is int and cohort_code==CONFIG.pilot_cohort_code,
            "Only Pilot allowed; Formal/Test cohort is forbidden")
    mechanisms,repeats=CONFIG.cohort_limits(cohort_code)
    pairs=tuple({"mechanism_id":m,"pair_repeat_id":r,
        "pair_id":pair_id_for(cohort_code,m,r),
        "context_ids":tuple(context_id_for(cohort_code,m,r,t) for t in (0,1))}
        for m in range(mechanisms) for r in range(repeats))
    return {"mode":"PLAN_ONLY","cohort_code":cohort_code,"mechanisms":mechanisms,
        "pairs_per_mechanism":repeats,"pairs":pairs,"methods":METHODS,
        "protocol_sha256":CONFIG.protocol_sha256,
        "identification":"requires_explicit_upstream_status_at_registry_binding",
        "models_called":False,"dgp_generated":False}


def _outputs(method):
    if method in baselines.METHOD_IDS:return ("ate",)
    if method=="oracle-pair":return ("ate","interval")
    return ("mu","ite","ate","interval") if method in ("j","oracle-context") else ("mu","ite","ate")


@dataclass(frozen=True,slots=True)
class PilotRequest:
    method_id:str
    task:PairTaskSpec
    parent:EstimationRequest
    children:tuple


def build_pilot_registry(pair_tasks,*,run_id,upstream_status,upstream_failure=None):
    """Bind already-created factual tasks, never invent Identification or task IDs."""
    plan=pilot_plan();tasks=tuple(pair_tasks)
    require(type(run_id) is str and run_id.strip(),"Explicit run correlation ID required")
    require(upstream_status in ("POINT","NOT_IDENTIFIED"),"Explicit upstream Identification required")
    expected={(p["mechanism_id"],p["pair_repeat_id"]) for p in plan["pairs"]}
    require(len(tasks)==len(expected) and {(t.mechanism_id,t.pair_repeat_id) for t in tasks}==expected
            and all(type(t) is PairTaskSpec and t.cohort_code==CONFIG.pilot_cohort_code for t in tasks),
            "Incomplete/duplicate/non-Pilot 5-mechanism x 2-pair inventory")
    registry=TaskRegistry();records=[]
    for task in sorted(tasks,key=lambda t:(t.mechanism_id,t.pair_repeat_id)):
        for public in (task,)+task.context_refs:registry.register_task(public)
        for method in METHODS:
            # Request correlation IDs are engineering IDs; frozen task IDs never change.
            rid=run_id+":"+method+":"+task.pair_id
            if upstream_status=="NOT_IDENTIFIED":
                require(type(upstream_failure) is dict and set(upstream_failure)=={
                    "failure_type","failure_message","failure_provenance"},
                    "Preserve explicit upstream Identification failure details")
            extra=upstream_failure if upstream_status=="NOT_IDENTIFIED" else {}
            parent=EstimationRequest.from_task(task,status=upstream_status,target_id=CONFIG.target_id,
                request_id=rid,requested_outputs=_outputs(method),**extra)
            registry.register_request(method,parent)
            children=()
            if method in CONTEXT_METHODS:
                children=context_requests_from_parent(task,parent,request_ids=(rid+":0",rid+":1"))
                for child in children:registry.register_request(method,child)
            records.append(PilotRequest(method,task,parent,children))
    registry.seal(expected_task_ids=tuple(registry.tasks),expected_request_keys=tuple(registry.requests))
    return registry,tuple(records)


def registry_counts(registry):
    return dict(Counter(k[0]+":"+k[1] for k in registry.requests))


def _kernel_manifest(report):
    from .kernel_new_adapter import KernelNewManifest
    values=dict(report["kernel_new_manifest"])
    require(isinstance(values.get("design_kinds"),(tuple,list)),
            "Gate design kinds must be a JSON sequence")
    values["design_kinds"]=tuple(values["design_kinds"])
    if "arm_order" in values:
        require(isinstance(values["arm_order"],(tuple,list)),"Gate arm order must be a JSON sequence")
        values["arm_order"]=tuple(values["arm_order"])
    return KernelNewManifest(**values)


_GATE_MODEL_EVIDENCE = {
    "j": ("predict.load_selected_model", "full_model_state"),
    "kernel-old": ("benchmark.load_methods", "spec_prior_selection_excluding_query_cache"),
    "kernel-new": ("pfn_social.sparse_classical_calibration.load_model",
                   "full_parameters_and_buffers"),
}


def _gate_centers(records, name):
    require(type(records) is list and len(records)==2,
            name+" requires exactly two context evidence records")
    require(all(type(record) is dict and type(record.get("center_t")) is int
                and record["center_t"] in (0,1) for record in records),
            name+" has invalid context identity")
    require({record["center_t"] for record in records}=={0,1},
            name+" missing/duplicate center_t evidence")
    return records


def _validate_model_evidence(report, kernel_manifest):
    """Consistency/schema gate, NOT proof that arbitrary JSON was executed."""
    models=report.get("models")
    require(type(models) is dict and set(models)==set(_GATE_MODEL_EVIDENCE),
            "Gate needs j, kernel-old and kernel-new model evidence")
    for name,(loader,scope) in _GATE_MODEL_EVIDENCE.items():
        model=models[name]
        require(type(model) is dict and model.get("loader")==loader
                and model.get("load_status")=="PASS" and model.get("status","PASS")=="PASS",
                name+" missing/failed original Loader evidence")
        identity=model.get("identity")
        require((type(identity) is str and bool(identity.strip()))
                or (type(identity) is dict and bool(identity)),
                name+" missing model identity")
        provenance._json_bytes(identity)  # finite JSON-only declared identity
        require(model.get("state_fingerprint_scope")==scope,
                name+" has wrong/ambiguous state fingerprint scope")
        before=model.get("state_fingerprint_before")
        after=model.get("state_fingerprint_after")
        require(provenance._digest(before) and provenance._digest(after) and before==after,
                name+" missing/invalid/changed state fingerprint")
        for context in _gate_centers(model.get("contexts"),name):
            shape=context.get("shape")
            require(context.get("status")=="PASS" and type(shape) is list
                    and shape==[200,4] and all(type(v) is int for v in shape)
                    and context.get("dtype")=="float64"
                    and type(context.get("finite_count")) is int and context["finite_count"]==800,
                    name+" failed/incomplete canonical prediction evidence")
    new=models["kernel-new"]
    require(new["identity"]==kernel_manifest.identity
            and new["state_fingerprint_before"]==kernel_manifest.model_state_sha256,
            "Kernel-new model evidence/Manifest identity or complete-state mismatch")
    # If a report also records these extracted metadata fields, cross-bind them.
    for field in ("specs_sha256","frozen_state_sha256","design_kinds"):
        if field in new:
            expected=getattr(kernel_manifest,field)
            actual=tuple(new[field]) if field=="design_kinds" else new[field]
            require(actual==expected,"Kernel-new extracted metadata/Manifest mismatch: "+field)
    from .j_adapter import INTERVAL_KIND
    for interval in _gate_centers(models["j"].get("intervals"),"J intervals"):
        require(interval.get("status")=="PASS" and type(interval.get("level")) is float
                and interval["level"]==.95 and interval.get("lower_shape")==[3]
                and interval.get("upper_shape")==[3]
                and all(type(v) is int for v in interval["lower_shape"]+interval["upper_shape"])
                and interval.get("interval_kind")==INTERVAL_KIND,
                "J missing/failed frozen 95% interval evidence")


def validate_gate(report,*,source_snapshot_id,asset_root=ROOT,asset_reader=None):
    """Validate evidence structure/identities; JSON alone cannot prove execution."""
    require(type(report) is dict and report.get("schema_version")==1
            and report.get("gate")=="REAL_MODEL_GATE" and report.get("status")=="PASS",
            "Real Model Gate PASS report required")
    require(report.get("protocol_sha256")==CONFIG.protocol_sha256
            and report.get("source_snapshot_id")==source_snapshot_id,"Gate protocol/source mismatch")
    require(report.get("assets")==ASSETS,"Gate asset declarations mismatch")
    git=report.get("git")
    require(type(git) is dict and set(git)=={"head","branch","status_short"}
            and git["head"]==CONFIG.source_commit and type(git["branch"]) is str
            and type(git["status_short"]) is list,"Gate must preserve observed Git identity")
    # Metadata-only constructor, no torch/checkpoint import.
    manifest=report.get("kernel_new_manifest")
    require(type(manifest) is dict,"Gate must pin actual kernel-new metadata")
    _validate_model_evidence(report,_kernel_manifest(report))
    reader=(lambda p:hashlib.sha256(p.read_bytes()).hexdigest()) if asset_reader is None else asset_reader
    for relative,expected in ASSETS.items():
        require(reader(Path(asset_root)/relative)==expected,"Actual asset hash mismatch: "+relative)
    return report


def read_gate(path,*,expected_sha256,source_snapshot_id,asset_root=ROOT,asset_reader=None):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,"Gate report receipt mismatch")
    return validate_gate(strict_json(raw.decode("ascii")),source_snapshot_id=source_snapshot_id,
                         asset_root=asset_root,asset_reader=asset_reader)


def verify_execution_source(gate,source):
    """Actual runs must originate from the Gate-pinned extracted Snapshot."""
    import sys
    pinned=gate.get("runtime_root")
    require(type(pinned) is str and pinned and Path(pinned).resolve()==ROOT,
            "Gate must pin the actual extracted Snapshot runtime root")
    require(not any((parent/".git").exists() for parent in (ROOT,*ROOT.parents)),
            "Actual execution from an original Git worktree is forbidden")
    entry=next(row for row in source["files"] if row["path"]=="star_v2/run_pilot.py")
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==entry["sha256"],
            "Entrypoint source differs from Snapshot")
    modules={name:module for name,module in sys.modules.items()
             if name in ("star_v2","pfn_social","scripts","predict","benchmark")
             or name.startswith(("star_v2.","pfn_social.","scripts."))}
    return provenance.verify_runtime_modules(modules,ROOT,source)


def _once(loader):
    attempted=False;model=None;failure=None
    def load():
        nonlocal attempted,model,failure
        if not attempted:
            attempted=True
            try:model=loader()
            except Exception as exc:failure=(type(exc).__name__,str(exc))
        if failure is not None:raise RuntimeError(": ".join(failure))
        return model
    return load


def real_adapters(gate):
    """All imports/construction below occur only AFTER the execution/Gate checks.

    Reuse original loaders, never torch.load or reconstructed checkpoint logic.
    """
    from .j_adapter import JAdapter,_default_loader
    from .kernel_adapters import KernelOldAdapter,KernelOldIdentity,FROZEN_KERNEL_OLD
    from .kernel_new_adapter import KernelNewAdapter,KernelNewManifest
    j=JAdapter(loader=_once(_default_loader))
    def old_loader():
        from benchmark import configuration,load_methods
        return load_methods(["kernel-old"],configuration())["kernel-old"]
    def old_identity(model):
        require(len(model.components)==2,"Frozen old-kernel component count mismatch")
        return KernelOldIdentity(components=tuple((FROZEN_KERNEL_OLD.components[i][0],component[2])
            for i,component in enumerate(model.components)),
            representation_weights=tuple(model.selection["representation_weights"]),
            noise_weights=tuple(model.selection["noise_weights"]),
            noise_variances=tuple(model.selection["noise_variances"]))
    old=KernelOldAdapter(_once(old_loader),identity_reader=old_identity)
    def new_loader():
        from pfn_social.sparse_classical_calibration import load_model
        path=ROOT/"models/kernel_calibration.pt"
        require(hashlib.sha256(path.read_bytes()).hexdigest()==ASSETS["models/kernel_calibration.pt"],
                "Kernel-new checkpoint changed before original loader")
        return load_model(path)
    def subset(model):
        from pfn_social.sparse_classical_calibration import frozen_state_sha
        return frozen_state_sha(model)
    new=KernelNewAdapter(_once(new_loader),manifest=_kernel_manifest(gate),
                         frozen_state_reader=subset)
    return {"j":j,"kernel-old":old,"kernel-new":new}


def invoke(record,adapters,*,true_h):
    task,parent,method=record.task,record.parent,record.method_id
    if method in baselines.METHOD_IDS:
        return baselines.estimate_pair(task,parent,method)
    if method in ("j","kernel-old","kernel-new"):
        return adapters[method].estimate_pair(task,parent,context_requests=record.children)
    if method=="oracle-context":
        return oracle_adapters.estimate_oracle_context_pair(task,parent,true_h=true_h,
                    noise_sd=CONFIG.noise_sd,context_requests=record.children)
    return oracle_adapters.estimate_oracle_pair(task,parent,true_h=true_h,noise_sd=CONFIG.noise_sd)


def _failed(record,error,*,true_h):
    """Existing failure paths only; no second model/numerical attempt."""
    def fail(*args,**kwargs):raise RuntimeError(type(error).__name__+": "+str(error))
    method=record.method_id
    if method in baselines.METHOD_IDS:
        from .estimation import execute_contract
        return execute_contract(record.parent,record.task,method,fail)
    if method in ("j","kernel-old","kernel-new"):
        result=model_adapters.estimate_pair(record.task,record.parent,method,
            context_requests=record.children,predictors=(fail,fail),interval_predictors=(fail,fail))
        if method=="j":
            from dataclasses import replace
            from .j_adapter import JAdapter
            result=replace(result,arm_results=tuple(JAdapter._normalize_context(a) for a in result.arm_results))
        return result
    if method=="oracle-context":
        return oracle_adapters.estimate_oracle_context_pair(record.task,record.parent,
            true_h=true_h,noise_sd=CONFIG.noise_sd,context_requests=record.children,core=fail)
    return oracle_adapters.estimate_oracle_pair(record.task,record.parent,true_h=true_h,
                                               noise_sd=CONFIG.noise_sd,core=fail)


def execute_registered_request(registry,record,adapters,output_root,*,true_h,
                               attempt_no=1,authorization=None,window_factory=RuntimeWindow,source_receipts=()):
    """Explicit single Attempt, never a retry/selection loop."""
    parent_key=(record.method_id,"pair",record.task.pair_id,record.parent.request_id)
    require(attempt_no==registry.next_attempt_no(parent_key) and (attempt_no==1 or authorization),
            "Explicit next Attempt number and retry authorization required")
    # Validate all parent/child attempt slots BEFORE model work.
    for request in (record.parent,)+record.children:
        key=(record.method_id,request.scope,request.task_id,request.request_id)
        require(key in registry.requests and registry.next_attempt_no(key)==attempt_no,
                "Request/Attempt identity conflict")
    try:
        bundle,runtime=measure_pair(lambda:invoke(record,adapters,true_h=true_h),
                                   record.method_id,window_factory=window_factory)
    except MeasuredFailure as exc:
        runtime=exc.record
        bundle=_failed(record,exc.original,true_h=true_h)
    for arm in bundle.arm_results:
        if arm is None:continue
        registry.append_attempt(arm,attempt_no=attempt_no,authorization=authorization)
        interval=next((r for r in runtime["context_intervals"]
                       if r["context_id"]==arm.context_ids[0]),None)
        data.save_attempt_outputs(output_root,registry,arm,attempt_no=attempt_no,runtime={
            "scope":"context","parent_execution_key":parent_key,
            "point_runtime_seconds":None,"point_runtime_reason":"Measured at Pair scope, not independent contexts",
            "interval":interval,"interval_reason":None if interval else "Not called/requested or dependency failed"},
            source_receipts=tuple(d for d in source_receipts if d["path"].startswith(
                "factual/inputs/"+arm.context_ids[0]+".")))
    registry.append_attempt(bundle,attempt_no=attempt_no,authorization=authorization)
    index=data.save_attempt_outputs(output_root,registry,bundle,attempt_no=attempt_no,runtime=runtime,
                                    source_receipts=source_receipts)
    return bundle,index


def _save_final_ledger(registry, target, *, complete, failure=None):
    """One create-only final ledger; completion is explicit, never inferred."""
    missing = registry.unexecuted_requests()
    require(not complete or not missing, "Cannot mark unexecuted requests complete")
    receipt = data.export_ledger(registry, target/"ledger_final")
    data._write(target, "ledger_completion.json", data.json_bytes({
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "registry_seal_sha256": registry.seal_sha256,
        "initial_manifest_sha256": hashlib.sha256(
            (target/"ledger_initial"/"ledger_manifest.json").read_bytes()).hexdigest(),
        "final_manifest_sha256": receipt.manifest_sha256,
        "expected_tasks": len(registry.tasks), "expected_requests": len(registry.requests),
        "recorded_attempts": len(registry.attempts),
        "unexecuted_request_keys": [list(key) for key in missing],
        "failure": None if failure is None else {
            "type": type(failure).__name__, "message": str(failure)}}))
    return receipt


def execute_pilot(*,execute_pilot,gate_path,gate_sha256,source_delivery,source_manifest_sha256,
                  output_root,run_id,upstream_status,asset_root=ROOT,
                  mechanism_factory=None,pair_factory=None,adapter_factory=None,
                  asset_reader=None,window_factory=RuntimeWindow,upstream_failure=None):
    require(execute_pilot is True,"Explicit --execute-pilot is required")
    # No DGP, true loader or output directory exists before all safety checks.
    source=provenance.verify_source_snapshot(source_delivery,expected_manifest_sha256=source_manifest_sha256)
    gate=read_gate(gate_path,expected_sha256=gate_sha256,source_snapshot_id=source["snapshot_id"],
                   asset_root=asset_root,asset_reader=asset_reader)
    require(upstream_status in ("POINT","NOT_IDENTIFIED"),"Explicit upstream Identification required")
    if adapter_factory is None:
        require(Path(asset_root).resolve()==ROOT,"Real loaders must use their frozen runtime-root paths")
        verify_execution_source(gate,source)
    target=provenance._fresh_directory(output_root)
    from .dgp import generate_star_mechanism
    from .pair import generate_pair
    make_mechanism=generate_star_mechanism if mechanism_factory is None else mechanism_factory
    make_pair=generate_pair if pair_factory is None else pair_factory
    def generate():
        mechanisms={};private=[];public=[]
        for m in range(CONFIG.pilot_mechanisms):
            mechanisms[m]=make_mechanism(cohort_code=CONFIG.pilot_cohort_code,mechanism_id=m)
            for repeat in range(CONFIG.pilot_pairs_per_mechanism):
                item=make_pair(mechanisms[m],pair_repeat_id=repeat)
                private.append(item);public.append(pair_from_private(item))
        return mechanisms,tuple(private),tuple(public)
    generated,data_time=measure_phase(generate,"data_generation",window_factory=window_factory)
    mechanisms,private,pairs=generated
    registry,requests=build_pilot_registry(pairs,run_id=run_id,upstream_status=upstream_status,
                                            upstream_failure=upstream_failure)
    target.mkdir()
    # Only an actually authorized execution can reach this real run record.
    manifest=provenance.build_run_manifest(run_id=run_id,run_kind="pilot",source_manifest=source,
        source_manifest_sha256=source_manifest_sha256,git_identity=gate["git"],
        entrypoint="star_v2/run_pilot.py",environment=provenance.collect_environment(),
        model_assets=tuple({"path":name,"status":"VERIFIED","sha256":digest,
                            "bytes":(Path(asset_root)/name).stat().st_size} for name,digest in ASSETS.items()),
        task_ledger_manifest_sha256=provenance.export_task_manifest(registry,target/"ledger_initial").manifest_sha256,
        registry_seal_sha256=registry.seal_sha256)
    data._write(target,"run_manifest.json",data.json_bytes(manifest))
    # Startup/import/loading costs are distinct; warmup predictions are NOT run.
    adapters,setup=measure_phase(lambda:(real_adapters(gate) if adapter_factory is None
                                       else adapter_factory(gate)),"setup",window_factory=window_factory)
    if upstream_status=="POINT":
        for method,adapter in adapters.items():
            try:
                _,measured=measure_phase(adapter._get_model,"setup:"+method,window_factory=window_factory)
                setup.setdefault("model_loads",{})[method]=measured
            except MeasuredFailure as exc:
                setup.setdefault("model_loads",{})[method]=exc.record
    if adapter_factory is None:verify_execution_source(gate,source)
    truth_receipts={}
    load_windows=tuple(setup.get("model_loads",{}).values())
    setup["entry_module_import_seconds"]=_ENTRY_IMPORT_SECONDS
    setup["setup_seconds"]=(_ENTRY_IMPORT_SECONDS+setup["wall_seconds"]+
        sum(w["wall_seconds"] for w in load_windows)
        if setup["wall_seconds"] is not None and all(w["wall_seconds"] is not None for w in load_windows)
        else None)
    setup["definition"]="entry import + adapter factory + separately observed initial model loads; no warmup"
    for m,mechanism in mechanisms.items():
        truth_receipts[m]=data.save_evaluator_arrays(target,"truth","mechanism_"+str(m),
            {"mu":mechanism.mu,"ite":mechanism.ite,"ate":mechanism.ate},{"mechanism_id":m})
    sources={}
    for item,task in zip(private,pairs):
        receipts=list(truth_receipts[task.mechanism_id])
        for context in task.context_refs:receipts.extend(data.save_factual_inputs(target,context))
        receipts.extend(data.save_evaluator_arrays(target,"factual/simulator_private",task.pair_id,
            {"h":item.mechanism.h,"coefficients":item.mechanism.coefficients,
             "noise":item.noise,"leaf_treatment":item.leaf_treatment,"observed_indices":item.observed_indices,
             "full_y0":item.contexts[0].full_outcomes,"full_y1":item.contexts[1].full_outcomes},
            {"pair_id":task.pair_id}))
        sources[task.pair_id]=tuple(receipts)
    try:
        for record in requests:
            execute_registered_request(registry,record,adapters,target,
                true_h=mechanisms[record.task.mechanism_id].h,window_factory=window_factory,
                source_receipts=sources[record.task.pair_id])
        _,evaluation=measure_phase(lambda:data.read_execution_records(target,registry,require_complete=True),
                                   "evaluation_integrity",window_factory=window_factory)
    except Exception as error:
        # Preserve the original exception; partial evidence is never a success.
        try:
            _save_final_ledger(registry,target,complete=False,failure=error)
        except Exception as save_error:
            error.add_note("Partial final Ledger could not be saved: "+
                           type(save_error).__name__+": "+str(save_error))
        raise
    _save_final_ledger(registry,target,complete=True)
    data._write(target,"runtime/run_resources.json",data.json_bytes({
        "setup":setup,"data_generation":data_time,"evaluation":evaluation,
        "data_generation_seconds":data_time["wall_seconds"],
        "evaluation_seconds":evaluation["wall_seconds"],
        "evaluation_scope":"output/Ledger integrity only; no scientific evaluation or selection",
        "disk":disk_occupancy(target)}))
    return registry


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=("pilot","test","formal"),default="pilot")
    parser.add_argument("--execute-pilot",action="store_true")
    for name in ("gate-report","gate-sha256","source-delivery","source-manifest-sha256",
                 "output","run-id","identification-status"):
        parser.add_argument("--"+name)
    parser.add_argument("--identification-failure",help="JSON file containing original upstream failure")
    args=parser.parse_args(argv)
    require(args.cohort=="pilot","Formal/Test execution is not supported")
    if not args.execute_pilot:
        print(json.dumps(pilot_plan(),sort_keys=True));return 0
    require(all((args.gate_report,args.gate_sha256,args.source_delivery,args.source_manifest_sha256,
                 args.output,args.run_id,args.identification_status)),
            "Gate/source receipts, fresh output, run ID and upstream Identification are required")
    execute_pilot(execute_pilot=True,gate_path=args.gate_report,gate_sha256=args.gate_sha256,
        source_delivery=args.source_delivery,source_manifest_sha256=args.source_manifest_sha256,
        output_root=args.output,run_id=args.run_id,upstream_status=args.identification_status,
        upstream_failure=(strict_json(Path(args.identification_failure).read_text(encoding="ascii"))
                          if args.identification_failure else None))
    return 0


if __name__=="__main__":
    main()
