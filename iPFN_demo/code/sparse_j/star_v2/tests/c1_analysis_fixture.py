"""Complete hand-specified Synthetic persisted Run; NEVER models or DGP."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import contextlib,json,shutil,tempfile
from unittest.mock import patch
import numpy as np
from star_v2 import run_test as r,provenance,data_artifacts as data,test_acceptance as accept
from star_v2 import analysis_loader as loader,test_authorization as auth
from star_v2.estimation import EstimateBundle,PairEstimateBundle,IntervalRecord,SamplingUncertainty,EFFECT_ORDER
from star_v2.support import ORDINARY_METHOD_IDS
from test_coverage import make_pair
from test_run_test import approval
from test_run_pilot import gate as mock_gate

MANUAL_INDICES=np.array([[1,0],[0,1],[1,1],[0,0]],dtype=np.int64)

def make_synthetic(case):
    base=Path(case.enterContext(tempfile.TemporaryDirectory(prefix="c1_synthetic_")))
    code=r.shared.ROOT
    runtime=base/"runtime";runtime.mkdir()
    files=[]
    for p in sorted(code.rglob("*")):
        if not p.is_file():continue
        rel=p.relative_to(code).as_posix()
        if "__pycache__" in rel or p.suffix in (".pyc",".pyo"):continue
        dest=runtime/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        role="test" if "/tests/" in rel else "protocol" if "/protocol/" in rel else "implementation" if rel.endswith(".py") else "documentation" if rel.endswith(".md") else "configuration"
        files.append(provenance.SourceFile(rel,role))
    # Five fake binary placeholders, never deserialized. Two JSON assets retain
    # copied frozen bytes so the source snapshot remains self-consistent.
    for rel in r.shared.ASSETS:
        if not (runtime/rel).exists():(runtime/rel).write_bytes(b"MOCK NEVER DESERIALIZE")
    delivery=base/"source_delivery"
    source_receipt=provenance.build_source_snapshot(runtime,delivery,git_head=r.CONFIG.source_commit,files=tuple(files))
    source=provenance.verify_source_snapshot(delivery,expected_manifest_sha256=source_receipt.manifest_sha256)
    g=mock_gate();g["runtime_root"]=str(runtime);g["source_snapshot_id"]=source["snapshot_id"]
    run_id="MOCK_C1_ONLY"
    a=approval(run_id);a_raw=data.json_bytes(a)
    d=r.identification.make_decision(a,approval_sha256=data.sha(a_raw),run_id=run_id,snapshot_id=source["snapshot_id"])
    d_raw=data.json_bytes(d);g_raw=data.json_bytes(g)
    authorization=auth.pending_template(snapshot_id=source["snapshot_id"])
    authorization.update(approval_status="APPROVED",authority="MOCK ONLY",
        decision_reference="SYNTHETIC TEST; NOT REAL USER AUTHORIZATION",run_id=run_id,
        gate_sha256=data.sha(g_raw),identification_sha256=data.sha(d_raw),approval_sha256=data.sha(a_raw))
    records={}
    for name,raw in (("approval",a_raw),("decision",d_raw),("gate",g_raw),("authorization",data.json_bytes(authorization))):
        path=base/(name+".json");path.write_bytes(raw);records[name]=(path,data.sha(raw))
    pairs=tuple(make_pair(m,repeat) for m in range(2) for repeat in range(3))
    plan=r.test_plan();plan["pairs"]=tuple(p for p in plan["pairs"] if p["mechanism_id"]<2)
    plan["mechanisms"]=2;plan["pairs_per_mechanism"]=3
    plan["counts"]={"pair_data_tasks":6,"context_data_tasks":12,"data_tasks":18,
        "pair_requests":54,"context_requests":48,"requests":102,"initial_attempts":102}
    mechanisms={}
    for mid in range(2):
        base_y=np.arange(200,dtype=np.float64)/32+mid
        mu=np.stack((base_y,base_y+.25,base_y+1.,base_y+3.),axis=1).reshape(200,2,2)
        mechanisms[mid]=SimpleNamespace(h=np.zeros((200,3)),coefficients=np.zeros(25),mu=mu,
            ite=np.tile(np.array([1.,2.,3.]),(200,1)),ate=np.array([1.,2.,3.]))
    private=tuple(SimpleNamespace(mechanism=mechanisms[p.mechanism_id],noise=np.zeros(200),
        leaf_treatment=p.context_refs[0].treatment[1:],observed_indices=p.context_refs[0].observed_indices,
        contexts=(SimpleNamespace(full_outcomes=np.zeros(200)),SimpleNamespace(full_outcomes=np.zeros(200))))
        for p in pairs)
    methods=r.shared.METHODS
    def calculate(record,adapters,*,true_h):
        method=record.method_id;mid=record.task.mechanism_id;repeat=record.task.pair_repeat_id
        index=methods.index(method);truth=mechanisms[mid];arms=[]
        for center,(task,request) in enumerate(zip(record.task.context_refs,record.children)):
            error=np.full(200,index+1+mid/4+repeat/8+center/16,dtype=np.float64);error[0]+=.5
            mu=truth.mu.copy();mu[:,1,0]+=error;mu[:,1,1]+=3*error;mu[:,0,1]+=.5*index
            ite=truth.ite+error[:,None]*np.array([1.,2.,3.])
            values=ite.mean(0)
            kwargs={};status={"mu":"OK","ite":("OK",)*3,"ate":("OK",)*3}
            if method in ("j","oracle-context"):
                intervals=[];codes=[]
                for k,effect in enumerate(EFFECT_ORDER):
                    failed=method=="j" and (mid,repeat,center,k)==(1,2,1,1)
                    codes.append("FAILED" if failed else "OK")
                    intervals.append(IntervalRecord(task.context_id,effect,"FAILED" if failed else "OK",
                        None if failed else float(truth.ate[k]-1),None if failed else float(truth.ate[k]+1),
                        source="model_posterior"))
                status["interval"]=tuple(codes)
                kwargs.update(intervals=tuple(intervals),sampling_uncertainty=SamplingUncertainty("OK","model_posterior","context",None))
                if "FAILED" in codes:kwargs.update(failure_type="SYNTHETIC_INTERVAL",
                    failure_message="retained synthetic interval failure",failure_provenance={"fixture":True})
            arms.append(EstimateBundle.from_outputs(request,task,method,mu=mu,ite=ite,ate=values,output_status=status,**kwargs))
        values=(arms[0].ate/2+arms[1].ate/2) if arms else truth.ate+np.array([1.,2.,3.])*(index+1+mid/4+repeat/8)
        status={"ate":("OK",)*3};kwargs={}
        if method=="state-mean" and (mid,repeat)==(1,2):
            values=values.copy();values[[0,2]]=np.nan;status["ate"]=("FAILED","OK","FAILED")
            kwargs.update(failure_type="SYNTHETIC_EMPTY_00",failure_message="synthetic empty state",
                          failure_provenance={"fixture":True})
        if method=="oracle-pair":
            kwargs.update(intervals=tuple(IntervalRecord(None,e,"OK",float(truth.ate[k]-1),float(truth.ate[k]+1),
                source="joint_model_posterior") for k,e in enumerate(EFFECT_ORDER)),
                sampling_uncertainty=SamplingUncertainty("OK","joint_model_posterior","pair",None))
            status["interval"]=("OK",)*3
        return PairEstimateBundle.from_outputs(record.parent,record.task,method,ate=values,
            output_status=status,arm_results=tuple(arms) if arms else (None,None),**kwargs)
    adapters={name:SimpleNamespace(_get_model=lambda:object()) for name in ("j","kernel-old","kernel-new")}
    out=base/"run"
    with patch.object(r,"test_plan",return_value=plan),patch.object(r,"preflight",return_value=(source,g,d,out)),patch.object(
        r,"_generate_test_data",return_value=(mechanisms,private,pairs)),patch.object(r.shared,"ROOT",runtime),patch.object(
        r.shared,"real_adapters",return_value=adapters),patch.object(r.shared,"invoke",side_effect=calculate),patch.object(
        r.shared,"verify_execution_source"):
        result=r.execute_test(execute_test=True,source_delivery=delivery,source_manifest_sha256=source_receipt.manifest_sha256,
            gate_path=records["gate"][0],gate_sha256=records["gate"][1],
            identification_path=records["decision"][0],identification_sha256=records["decision"][1],
            approval_path=records["approval"][0],approval_sha256=records["approval"][1],output_root=out,run_id=run_id,
            test_authorization_path=records["authorization"][0],test_authorization_sha256=records["authorization"][1])
    f=SimpleNamespace(base=base,out=out,plan=plan,source=source,receipt=source_receipt,result=result,
        gate=g,records=records,run_id=run_id,code=code)
    anchors=dict(run_manifest_sha256=result["run_manifest_sha256"],completion_sha256=result["completion_sha256"],
        source_delivery=delivery,source_manifest_sha256=source_receipt.manifest_sha256,
        execution_receipt_path=result["execution_receipt_path"],execution_receipt_sha256=result["execution_receipt_sha256"])
    f.anchors=anchors
    with validation_patches(f):
        receipt=accept.write_acceptance_receipt(out,base/"run_acceptance.json",**anchors)
    f.anchors={**anchors,"acceptance_path":base/"run_acceptance.json","acceptance_sha256":receipt["sha256"]}
    return f

@contextlib.contextmanager
def validation_patches(f):
    original=r.shared.validate_gate
    with patch.object(r,"test_plan",return_value=f.plan),patch.object(r.shared,"validate_gate",
        side_effect=lambda report,**kwargs:original(report,source_snapshot_id=kwargs["source_snapshot_id"],
            asset_root=kwargs["asset_root"],asset_reader=lambda p:r.shared.ASSETS["models/"+p.name])):
        yield

def load(f):
    with validation_patches(f):return loader.load_accepted_test(f.out,**f.anchors)
