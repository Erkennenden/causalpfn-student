"""Strict v2 persisted Test recovery. No estimators, DGP, v1 IDs or model loads."""
from dataclasses import dataclass,fields,replace
import sys
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import artifacts,data_artifacts as data,test_acceptance,contracts as c,provenance
from .task_registry import strict_json,request_from_document,Attempt
from .estimation import EstimateBundle,PairEstimateBundle,IntervalRecord,SamplingUncertainty,metadata
from .metrics import EvaluationPlan,Prediction,MechanismTruth,_support_index
from .support import ContextSupport,PairSupport,ORDINARY_METHOD_IDS
from .coverage import unique_attempts
from .config import CONFIG

def require(value,message):
    if not value:raise ValueError(message)

@dataclass(frozen=True,slots=True)
class RestoredRun:
    registry: object
    plan: EvaluationPlan
    bundles: object
    predictions: tuple
    truths: tuple
    context_supports: object
    method_supports: tuple
    pair_supports: object
    runtimes: object
    resources: object
    provenance: object
    input_records: tuple

def _equal(left,right):
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return type(left) is type(right) and left.dtype==right.dtype and np.array_equal(left,right)
    if hasattr(left,"__dataclass_fields__"):
        return type(left) is type(right) and all(_equal(getattr(left,f.name),getattr(right,f.name)) for f in fields(left))
    if isinstance(left,(tuple,list)):
        return len(left)==len(right) and all(_equal(a,b) for a,b in zip(left,right))
    return left==right

def _decode_support(arrays,document):
    used=set()
    def decode(value):
        if type(value) is dict and set(value)=={"array_key"}:
            key=value["array_key"];require(key in arrays,"Missing Support array")
            used.add(key);return arrays[key]
        if type(value) is dict:return {k:decode(v) for k,v in value.items()}
        if type(value) is list:return tuple(decode(v) for v in value)
        return value
    doc=decode(document)
    require(used==set(arrays),"Extra Support arrays")
    if "context_supports" in doc:
        require(set(doc)=={f.name for f in fields(PairSupport)},"Pair Support schema mismatch")
        doc["context_supports"]=tuple(ContextSupport(**v) for v in doc["context_supports"])
        obj=PairSupport(**doc)
    else:
        require(set(doc)=={f.name for f in fields(ContextSupport)},"Context Support schema mismatch")
        obj=ContextSupport(**doc)
    return obj

def _public_context(task,arrays):
    d=task.document;ident=d["identity"];center=ident["center_t"][0];link=d["bound_links"][0]
    meta={k:d[k] for k in ("protocol_id","protocol_version","protocol_hash","root_seed","protocol_code")}
    meta.update({k:ident[k] for k in ("cohort","cohort_code","mechanism_id","pair_repeat_id","rng_identity")})
    meta.update(center_t=center,pair_id=d["pair_id"],context_id=d["context_ids"][0],
        target_id=link["target_id"],request_id=link["request_id"],
        reference_assignment="iid Bernoulli(0.5)",arm_order=CONFIG.arm_order)
    return c.TaskSpec(**meta,**dict(arrays),node_ids=np.array(d["node_ids"],dtype=np.int64),
                      input_hash=d["input_hashes"][0])

def restore_records(root,registry,records,*,input_provenance):
    """Internal codec after complete acceptance; never substitutes missing data."""
    root=Path(root)
    require(registry.sealed and not registry.unexecuted_requests(),"Complete sealed ledger required")
    selected=unique_attempts(registry,registry.requests)
    require(all(v is not None for v in selected.values()),"Unexecuted request cannot be restored")
    by_key={};descriptors={}
    for record in records:
        key=tuple(record["execution_key"])
        require(key not in by_key and record["attempt_no"]==1,"Duplicate/multi Attempt recovery")
        by_key[key]=record
        for descriptor in record["outputs"]:
            path=descriptor["path"]
            require(path not in descriptors or descriptors[path]==descriptor,"Conflicting file descriptor")
            descriptors[path]=descriptor
    require(set(by_key)==set(registry.requests),"Execution output universe mismatch")
    def pair(stem):
        require(stem+".npz" in descriptors and stem+".json" in descriptors,"Missing NPZ/JSON pair: "+stem)
        return data.load_arrays(root,(descriptors[stem+".npz"],descriptors[stem+".json"]))
    contexts={}
    for task in registry.tasks.values():
        doc=task.document
        if doc["scope"]!="context":continue
        stem="factual/inputs/"+doc["context_ids"][0]
        arrays,meta=pair(stem)
        data.load_factual_inputs(root,(descriptors[stem+".npz"],descriptors[stem+".json"]))
        require(meta=={"context_id":doc["context_ids"][0],"pair_id":doc["pair_id"],
            "input_hash":doc["input_hashes"][0]},"Factual input metadata mismatch")
        contexts[doc["task_id"]]=_public_context(task,arrays)
    pairs={doc["task_id"]:c.pair_from_tasks(tuple(contexts[cid] for cid in doc["context_ids"]))
        for doc in (t.document for t in registry.tasks.values()) if doc["scope"]=="pair"}
    mechanism_ids=sorted({t.document["identity"]["mechanism_id"] for t in registry.tasks.values()})
    truths=[]
    for m in mechanism_ids:
        arrays,meta=pair("truth/mechanism_"+str(m))
        require(set(arrays)=={"mu","ite","ate"} and meta=={
            "mechanism_id":m,"access":"simulator_evaluator_only"},"Truth schema/access/identity mismatch")
        require(arrays["mu"].shape==(200,2,2) and arrays["ite"].shape==(200,3)
            and arrays["ate"].shape==(3,),"Truth node/state/effect axis mismatch")
        truths.append(MechanismTruth(m,arrays["ate"],arrays["mu"],arrays["ite"]))
    supports={};raw={}
    for key,record in by_key.items():
        tag=data.sha(data.json_bytes(list(key)))+"/attempt_1"
        s_arrays,s_meta=pair("support/"+tag)
        supports[key]=_decode_support(s_arrays,s_meta)
        partition="pair_estimates" if key[1]=="pair" else "predictions"
        arrays,meta=pair(partition+"/"+tag)
        require(set(arrays)=={"mu","ite","ate"},"Complete public output arrays required")
        attempt=selected[key]
        expected={"request":attempt["execution_request"]["request"],**attempt}
        require(meta==expected,"Prediction metadata/Attempt mismatch")
        raw[key]=arrays
    bundles={}
    # Contexts first; Pair records retain actual children, not independent fits.
    for scope in ("context","pair"):
        for key in sorted(registry.requests):
            if key[1]!=scope:continue
            doc=selected[key];request=request_from_document(doc["execution_request"]["request"])
            kwargs=dict(request=request,method_id=key[0],support=supports[key],
                output_status={name:tuple(value) if type(value) is list else value for name,value in doc["output_status"].items()},
                mu=raw[key]["mu"],ite=raw[key]["ite"],ate=raw[key]["ate"],
                intervals=tuple(IntervalRecord(**v) for v in doc["intervals"]),
                sampling_uncertainty=SamplingUncertainty(**doc["sampling_uncertainty"]),
                **{name:doc[name] for name in ("estimability","diagnostics","failure_type",
                    "failure_message","failure_provenance","asset_hashes")})
            if scope=="pair":
                arms=[]
                for cid in request.context_ids:
                    candidates=[bundle for ck,bundle in bundles.items() if ck[:3]==(key[0],"context",cid)
                                and bundle.request.parent_request_id==key[3]]
                    require(len(candidates)<=1,"Ambiguous child context")
                    arms.append(candidates[0] if candidates else None)
                bundle=PairEstimateBundle(**kwargs,arm_results=tuple(arms))
            else:bundle=EstimateBundle(**kwargs)
            require(Attempt.from_bundle(bundle,attempt_no=1).document==doc,"Bundle replay changed Attempt")
            bundles[key]=bundle
    plan=EvaluationPlan.from_registry(registry)
    predictions=tuple(Prediction.from_bundle(bundles[key]) for key in sorted(bundles))
    method_supports=tuple((t.key,supports[t.key]) for t in plan.tasks if t.key[1]=="context"
                         and t.key[0] in ("j","kernel-old","kernel-new","oracle-context"))
    _support_index(plan,method_supports)
    context_supports={};pair_supports={}
    for key,s in supports.items():
        if type(s) is PairSupport:
            require(_equal(s.node_ids,pairs[s.pair_id].node_ids),"Pair Support node order mismatch")
            left,right=s.context_supports
            for name,expected in (
                ("population_support",left.population_support&right.population_support),
                ("design_support",left.design_support|right.design_support),
                ("realized_state_mask",left.realized_state_mask|right.realized_state_mask),
                ("empirical_support_mask",left.empirical_support_mask|right.empirical_support_mask)):
                require(_equal(getattr(s,name),expected),"Pair Support union mismatch: "+name)
            require(_equal(s.empirical_extrapolation,~s.empirical_support_mask),"Pair extrapolation mismatch")
            require(s.n_unique_observed_nodes==50 and s.n_factual_records==100
                and np.array_equal(s.observed_node_ids,contexts[s.context_supports[0].context_id].observed_indices),
                "Pair observation identity/count mismatch")
            pair_supports[key]=s
            pieces=s.context_supports
        else:pieces=(s,)
        for context in pieces:
            require(context.arm_order==CONFIG.arm_order
                and _equal(context.empirical_extrapolation,~context.empirical_support_mask)
                and _equal(context.context_design_extrapolation,~context.design_support),
                "Support/extrapolation/state order mismatch")
            cid=context.context_id
            require(cid not in context_supports or _equal(context_supports[cid],context),
                    "Conflicting method/Pair Context Support")
            context_supports[cid]=context
    require(set(context_supports)==set(contexts),"Missing Context Support")
    runtimes={}
    for key,record in by_key.items():
        runtimes[key]=strict_json(data.verify_file(root,record["runtime"]).decode("ascii"))
    return RestoredRun(registry,plan,MappingProxyType(bundles),predictions,tuple(truths),
        MappingProxyType(context_supports),method_supports,MappingProxyType(pair_supports),
        MappingProxyType({k:metadata(v) for k,v in runtimes.items()}),
        metadata(strict_json((root/"runtime/run_resources.json").read_text(encoding="ascii"))),
        MappingProxyType(dict(input_provenance)),tuple(metadata(r) for r in records))

def _frozen_analysis_implementation(source):
    base=Path(__file__).resolve().parents[1]
    require(not any((p/".git").exists() for p in (base,)+tuple(base.parents)),
            "Frozen analysis must execute outside Git worktrees")
    modules={name:module for name,module in sys.modules.items() if module is not None and
        (name=="star_v2" or name.startswith("star_v2.") or name=="pfn_social" or name.startswith("pfn_social.")
         or name=="scripts" or name.startswith("scripts.") or name in ("predict","benchmark"))}
    provenance.verify_runtime_modules(modules,base,source)
    lookup={r["path"]:r for r in source["files"]}
    for name in ("analysis_loader.py","run_analysis.py","analysis_acceptance.py",
                 "analysis_schema.py","analysis_tables.py","analysis_figures.py","analysis_runtime.py"):
        rel="star_v2/"+name
        require(rel in lookup and data.sha((base/rel).read_bytes())==lookup[rel]["sha256"],
                "Analysis implementation is not frozen in input Source Snapshot: "+rel)

def load_accepted_test(root,*,run_manifest_sha256,completion_sha256,source_delivery,
                       source_manifest_sha256,execution_receipt_path,execution_receipt_sha256,
                       acceptance_path,acceptance_sha256):
    """Production recovery requires externally anchored independent acceptance."""
    receipt_path=Path(acceptance_path);raw=receipt_path.read_bytes()
    require(data.sha(raw)==acceptance_sha256,"Independent acceptance SHA256 mismatch")
    receipt=strict_json(raw.decode("ascii"))
    require(set(receipt)=={"schema_version","kind","status","run_manifest_sha256","completion_sha256",
        "source_manifest_sha256","execution_receipt_sha256","validator_sha256","gate_sha256","result"}
        and receipt["kind"]=="INDEPENDENT_TEST_ACCEPTANCE" and receipt["status"]=="PASS"
        and receipt["schema_version"]==1,"Independent acceptance receipt required")
    require(all(receipt[key]==value for key,value in (
        ("run_manifest_sha256",run_manifest_sha256),("completion_sha256",completion_sha256),
        ("source_manifest_sha256",source_manifest_sha256),
        ("execution_receipt_sha256",execution_receipt_sha256))),"Acceptance anchor mismatch")
    source=provenance.verify_source_snapshot(source_delivery,expected_manifest_sha256=source_manifest_sha256)
    _frozen_analysis_implementation(source)
    require(receipt["validator_sha256"]==data.sha(Path(test_acceptance.__file__).read_bytes()),
            "Independent validator source changed")
    result=test_acceptance.validate_test_run(root,run_manifest_sha256=run_manifest_sha256,
        completion_sha256=completion_sha256,source_delivery=source_delivery,
        source_manifest_sha256=source_manifest_sha256,execution_receipt_path=execution_receipt_path,
        execution_receipt_sha256=execution_receipt_sha256)
    require(result==receipt["result"],"Acceptance result no longer matches recovered Run")
    root=Path(root)
    require(data.sha((root/"real_model_gate.json").read_bytes())==receipt["gate_sha256"],
            "Acceptance Gate identity mismatch")
    run=strict_json((root/"run_manifest.json").read_text(encoding="ascii"))
    done=strict_json((root/"ledger_completion.json").read_text(encoding="ascii"))
    registry=artifacts.read_snapshot(root/"ledger_final",expected_manifest_sha256=done["final_manifest_sha256"])
    records=data.read_execution_records(root,registry,require_complete=True)
    loaded=restore_records(root,registry,records,input_provenance={
        "run_id":run["run_id"],"cohort_code":2,"protocol_sha256":CONFIG.protocol_sha256,
        "source_snapshot_id":source["snapshot_id"],"source_manifest_sha256":source_manifest_sha256,
        "run_manifest_sha256":run_manifest_sha256,"completion_sha256":completion_sha256,
        "execution_receipt_sha256":execution_receipt_sha256,"acceptance_sha256":acceptance_sha256,
        "gate_sha256":receipt["gate_sha256"],"registry_seal":registry.seal_sha256})
    external=strict_json(Path(execution_receipt_path).read_text(encoding="ascii"))
    return replace(loaded,resources=metadata({"run":loaded.resources,"execution_receipt":external,
                                             "environment":run["environment"]}))
