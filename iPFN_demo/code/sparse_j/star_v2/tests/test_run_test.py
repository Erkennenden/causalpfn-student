"""Stage 8.5-A deterministic Mock-only plumbing; never execute a cohort."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from star_v2 import run_test as r, test_identification as ident, test_acceptance as acceptance
from star_v2 import artifacts, provenance, data_artifacts as data, task_registry as ledger
from star_v2.config import CONFIG
from test_kernel_new_adapter import tasks_fixture
from test_task_registry import bundle_fixture
from test_run_pilot import gate as synthetic_gate

def approval(run_id="mock-test"):
    return {"schema_version":1,"kind":"TEST_IDENTIFICATION_APPROVAL","approved":True,
        "authority":"MOCK ONLY","decision_reference":"synthetic unit-test approval; not an actual Test approval",
        "run_id":run_id,"cohort_code":2,"protocol_sha256":CONFIG.protocol_sha256,
        "parent_snapshot_id":ident.PARENT_SNAPSHOT_ID,"status":"POINT","target_id":CONFIG.target_id,
        "basis":copy.deepcopy(ident.BASIS)}

def tiny_plan():
    # Private test patch only. Production API has no size/seed/method override.
    p=r.test_plan()
    p["pairs"]=p["pairs"][:1];p["mechanisms"]=1;p["pairs_per_mechanism"]=1
    p["counts"]={"pair_data_tasks":1,"context_data_tasks":2,"data_tasks":3,
        "pair_requests":9,"context_requests":8,"requests":17,"initial_attempts":17}
    return p

def decision(snapshot="a"*64):
    a=approval()
    return ident.make_decision(a,approval_sha256=ident.sha(data.json_bytes(a)),
                              run_id="mock-test",snapshot_id=snapshot)

def make_mock_run(case, *, interrupt=False,preload_failure=None,events=None):
    """Real storage/ledger functions, ONE hand-built Pair, zero DGP/model calls.

    Snapshot/Gate are explicitly synthetic declarations. Preflight is patched
    only for this storage integration; preflight rejection has separate tests.
    """
    base=Path(case.enterContext(tempfile.TemporaryDirectory()))
    source_root=base/"runtime";source_root.mkdir()
    (source_root/"star_v2").mkdir();(source_root/"models").mkdir()
    (source_root/"star_v2/run_test.py").write_bytes(b"# synthetic entry; never executed\n")
    lineage={"parent_snapshot_id":ident.PARENT_SNAPSHOT_ID,
             "parent_manifest_sha256":ident.PARENT_MANIFEST_SHA256,
             "parent_archive_sha256":ident.PARENT_ARCHIVE_SHA256}
    (source_root/"SNAPSHOT_PARENT.json").write_bytes(data.json_bytes(lineage))
    protocol_path="star_v2/protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md"
    (source_root/"star_v2/protocol").mkdir()
    (source_root/protocol_path).write_bytes((r.shared.ROOT/protocol_path).read_bytes())
    for name in r.shared.ASSETS:
        (source_root/name).write_bytes(b"MOCK ASSET NOT A CHECKPOINT")
    delivery=base/"delivery"
    receipt=provenance.build_source_snapshot(source_root,delivery,git_head=CONFIG.source_commit,
        files=(provenance.SourceFile("star_v2/run_test.py","entrypoint"),
               provenance.SourceFile("SNAPSHOT_PARENT.json","configuration"),
               provenance.SourceFile(protocol_path,"protocol")))
    source=provenance.verify_source_snapshot(delivery,expected_manifest_sha256=receipt.manifest_sha256)
    report=synthetic_gate()
    report["source_snapshot_id"]=source["snapshot_id"];report["runtime_root"]=str(source_root)
    a=approval();d=ident.make_decision(a,approval_sha256=ident.sha(data.json_bytes(a)),
                                     run_id="mock-test",snapshot_id=source["snapshot_id"])
    inputs={}
    for name,value in (("approval",a),("decision",d),("gate",report)):
        path=base/(name+".json");blob=data.json_bytes(value);path.write_bytes(blob)
        inputs[name]=(path,ident.sha(blob))
    auth=r.test_authorization.pending_template(snapshot_id=source["snapshot_id"])
    auth.update(approval_status="APPROVED",authority="MOCK ONLY",decision_reference="synthetic unit authorization",
        run_id="mock-test",gate_sha256=inputs["gate"][1],identification_sha256=inputs["decision"][1],
        approval_sha256=inputs["approval"][1])
    path=base/"authorization.json";blob=data.json_bytes(auth);path.write_bytes(blob)
    inputs["authorization"]=(path,ident.sha(blob))
    if events is None:events=[]
    class MockAdapter:
        def __init__(self,name):self.name=name;self.loads=0;self.model=SimpleNamespace(state=b"frozen mock parameters/buffers")
        def _get_model(self):
            events.append("preload:"+self.name);self.loads+=1
            if preload_failure==self.name:raise RuntimeError("MOCK preload failure: "+self.name)
            return self.model
    mock_adapters={name:MockAdapter(name) for name in ("j","kernel-old","kernel-new")}
    task=tasks_fixture()[0]
    mechanism=SimpleNamespace(h=np.zeros((200,3)),coefficients=np.zeros(25),
        mu=np.zeros((200,2,2)),ite=np.zeros((200,3)),ate=np.zeros(3))
    private=SimpleNamespace(mechanism=mechanism,noise=np.zeros(200),
        leaf_treatment=task.context_refs[0].treatment[1:],observed_indices=task.context_refs[0].observed_indices,
        contexts=tuple(SimpleNamespace(full_outcomes=np.zeros(200)) for _ in (0,1)))
    plan=tiny_plan();out=base/"run";calls=[]
    execute=r.shared.execute_registered_request
    def invoke(record,adapters,*,true_h):
        if record.method_id=="state-mean":
            return bundle_fixture(record.task,record.parent)
        return r.shared._failed(record,RuntimeError("preserved MOCK failure"),true_h=true_h)
    def execute_one(*args,**kwargs):
        calls.append(args[1].method_id)
        if interrupt and len(calls)==2:
            raise OSError("MOCK interruption; no retry")
        return execute(*args,**kwargs)
    arguments=dict(execute_test=True,source_delivery=delivery,source_manifest_sha256=receipt.manifest_sha256,
        gate_path=inputs["gate"][0],gate_sha256=inputs["gate"][1],
        identification_path=inputs["decision"][0],identification_sha256=inputs["decision"][1],
        approval_path=inputs["approval"][0],approval_sha256=inputs["approval"][1],
        output_root=out,run_id="mock-test",test_authorization_path=inputs["authorization"][0],
        test_authorization_sha256=inputs["authorization"][1])
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(r,"test_plan",return_value=plan))
        stack.enter_context(patch.object(r,"preflight",return_value=(source,report,d,out)))
        def generate():
            events.append("DGP")
            return {0:mechanism},(private,),(task,)
        original_builder=r.build_test_registry
        def build(*args,**kwargs):
            events.append("tasks")
            return original_builder(*args,**kwargs)
        stack.enter_context(patch.object(r,"_generate_test_data",side_effect=generate))
        stack.enter_context(patch.object(r,"build_test_registry",side_effect=build))
        stack.enter_context(patch.object(r.shared,"ROOT",source_root))
        stack.enter_context(patch.object(r.shared,"real_adapters",return_value=mock_adapters))
        stack.enter_context(patch.object(r.shared,"verify_execution_source"))
        stack.enter_context(patch.object(r.shared,"invoke",side_effect=invoke))
        stack.enter_context(patch.object(r.shared,"execute_registered_request",side_effect=execute_one))
        snapshots=stack.enter_context(patch.object(data,"export_ledger",wraps=data.export_ledger))
        if preload_failure is not None:
            with case.assertRaisesRegex(RuntimeError,"MOCK preload failure"):
                r.execute_test(**arguments)
            result=None
        elif interrupt:
            with case.assertRaisesRegex(OSError,"MOCK interruption"):
                r.execute_test(**arguments)
            result=None
        else:
            result=r.execute_test(**arguments)
        case.assertEqual(snapshots.call_count,0 if preload_failure is not None else 2)
        case.assertEqual([Path(c.args[1]).name for c in snapshots.call_args_list],
                         [] if preload_failure is not None else ["ledger_initial","ledger_final"])
    return SimpleNamespace(base=base,out=out,source=source,delivery=delivery,receipt=receipt,
        plan=plan,report=report,result=result,calls=calls,arguments=arguments,inputs=inputs,
        events=events,mock_adapters=mock_adapters)

def validate_mock(fixture):
    # Asset byte digests are deliberately mocked, not true checkpoint evidence.
    original=r.shared.validate_gate
    with patch.object(r,"test_plan",return_value=fixture.plan),patch.object(r.shared,"validate_gate",
        side_effect=lambda report,**kwargs:original(report,source_snapshot_id=kwargs["source_snapshot_id"],
            asset_root=kwargs["asset_root"],asset_reader=lambda p:r.shared.ASSETS["models/"+p.name])):
        return acceptance.validate_test_run(fixture.out,
            run_manifest_sha256=fixture.result["run_manifest_sha256"],
            completion_sha256=fixture.result["completion_sha256"],source_delivery=fixture.delivery,
            source_manifest_sha256=fixture.receipt.manifest_sha256,
            execution_receipt_sha256=fixture.result["execution_receipt_sha256"],
            execution_receipt_path=fixture.result["execution_receipt_path"])

class FormalPlanningTests(unittest.TestCase):
    def test_frozen_full_scale_counts_methods_without_dgp(self):
        with patch.object(r,"_generate_test_data",side_effect=AssertionError("DGP forbidden")):
            p=r.test_plan()
        self.assertEqual((p["mechanisms"],p["pairs_per_mechanism"],len(p["pairs"])),(200,3,600))
        self.assertEqual(p["counts"],{"pair_data_tasks":600,"context_data_tasks":1200,"data_tasks":1800,
            "pair_requests":5400,"context_requests":4800,"requests":10200,"initial_attempts":10200})
        self.assertEqual(len({v["pair_id"] for v in p["pairs"]}),600)
        self.assertEqual(len({c for v in p["pairs"] for c in v["context_ids"]}),1200)
        self.assertEqual(tuple(p["methods"]),("state-mean","ht","f-z2","lin-z2","kernel-old",
            "kernel-new","j","oracle-context","oracle-pair"))
        self.assertFalse(p["dgp_generated"]);self.assertFalse(p["models_called"])

    def test_test_cohort_seed_and_rng_metadata(self):
        p=r.test_plan()
        self.assertEqual((p["root_seed"],p["protocol_code"],p["cohort_code"]),(20261030,2026103001,2))
        task=tasks_fixture()[0]
        metadata=json.loads(task.context_refs[0].rng_identity)
        self.assertEqual(len(metadata),6)
        self.assertTrue(all(v["cohort_code"]==2 for v in metadata))
        for invalid in (1,True,2.0,0,3):
            with self.subTest(cohort=invalid),self.assertRaises(ValueError):
                r.test_plan(cohort_code=invalid)

    def test_cli_plan_only_and_pilot_rejected_before_generation(self):
        with patch.object(r,"_generate_test_data",side_effect=AssertionError("forbidden")) as gen:
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(r.main(["--cohort","test"]),0)
            self.assertEqual(json.loads(output.getvalue())["counts"]["requests"],10200)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                r.main(["--cohort","pilot"])
            with self.assertRaises(ValueError):
                r.main(["--cohort","test","--execute-test"])
            gen.assert_not_called()

    def test_bound_registry_unique_parents_and_method_mapping(self):
        with patch.object(r,"test_plan",return_value=tiny_plan()):
            reg,records=r.build_test_registry((tasks_fixture()[0],),run_id="mock-test",decision=decision())
            acceptance.validate_universe(reg,run_id="mock-test")
        self.assertEqual((len(reg.tasks),len(reg.requests),len(records)),(3,17,9))
        self.assertEqual(sum(len(v.children) for v in records),8)
        self.assertEqual(len(set(reg.requests)),17)
        for record in records:
            self.assertEqual(record.parent.status,"POINT")
            for child in record.children:
                self.assertEqual(child.parent_request_id,record.parent.request_id)
                self.assertEqual(child.pair_id,record.parent.pair_id)
                self.assertIn(child.input_hashes[0],record.parent.input_hashes)
                self.assertEqual(child.node_ids.tolist(),list(range(200)))

    def test_duplicate_missing_and_non_test_pair_rejected(self):
        task=tasks_fixture()[0]
        with patch.object(r,"test_plan",return_value=tiny_plan()):
            for tasks in ((),(task,task)):
                with self.subTest(tasks=len(tasks)),self.assertRaises(ValueError):
                    r.build_test_registry(tasks,run_id="mock-test",decision=decision())
            bad=copy.copy(task);arm=copy.copy(task.context_refs[0])
            object.__setattr__(arm,"cohort_code",1)
            object.__setattr__(bad,"context_refs",(arm,task.context_refs[1]))
            with self.assertRaises(ValueError):
                r.build_test_registry((bad,),run_id="mock-test",decision=decision())
            with self.assertRaises(ValueError):
                r.build_test_registry((task,),run_id="wrong",decision=decision())

    def test_test_approval_and_decision_identity_pilot_rejection(self):
        a=approval();d=decision()
        ident.validate_decision(d,a,approval_sha256=ident.sha(data.json_bytes(a)),
                                run_id="mock-test",snapshot_id="a"*64)
        for field,value in (("cohort_code",1),("run_id","pilot"),("status","OK"),
                            ("parent_snapshot_id","b"*64),("basis",{})):
            bad=copy.deepcopy(a);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                ident.make_decision(bad,approval_sha256="b"*64,run_id="mock-test",snapshot_id="a"*64)

    def test_identification_create_only_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=approval();raw=data.json_bytes(a);(root/"approval.json").write_bytes(raw)
            params=dict(approval_path=root/"approval.json",approval_sha256=ident.sha(raw),
                        run_id="mock-test",snapshot_id="a"*64)
            receipt=ident.write_decision(root/"decision.json",**params)
            self.assertEqual(ident.read_decision(root/"decision.json",expected_sha256=receipt["sha256"],
                                                **params)["cohort_code"],2)
            with self.assertRaises((ValueError,FileExistsError)):
                ident.write_decision(root/"decision.json",**params)
            with self.assertRaises(ValueError):
                ident.read_decision(root/"decision.json",expected_sha256="b"*64,**params)

    def test_old_gate_rejected_before_data_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);delivery=root/"delivery";delivery.mkdir();report=synthetic_gate()
            report["source_snapshot_id"]=ident.PARENT_SNAPSHOT_ID
            (root/"gate.json").write_bytes(data.json_bytes(report))
            source={"snapshot_id":"c"*64}
            with patch.object(r.provenance,"verify_source_snapshot",return_value=source),patch.object(
                r,"verify_lineage"),patch.object(r,"_generate_test_data",
                    side_effect=AssertionError("DGP forbidden")) as gen:
                with self.assertRaisesRegex(ValueError,"protocol/source"):
                    r.preflight(source_delivery=delivery,source_manifest_sha256="a"*64,
                        gate_path=root/"gate.json",gate_sha256=ident.sha(data.json_bytes(report)),
                        identification_path=root/"none",identification_sha256="a"*64,
                        approval_path=root/"none",approval_sha256="a"*64,output_root=root/"out",run_id="mock-test")
                gen.assert_not_called()
            self.assertFalse((root/"out").exists())

    def test_complete_mock_run_two_ledgers_preserves_success_and_failures(self):
        f=make_mock_run(self)
        report=validate_mock(f)
        self.assertEqual((report["tasks"],report["requests"],report["attempts"]),(3,17,17))
        self.assertEqual(report["failed_attempts"],16)
        self.assertEqual(f.calls,list(r.shared.METHODS))
        ledger_done=ledger.strict_json((f.out/"ledger_completion.json").read_text())
        reg=artifacts.read_snapshot(f.out/"ledger_final",
                                   expected_manifest_sha256=ledger_done["final_manifest_sha256"])
        self.assertEqual([a.document["attempt_no"] for a in reg.attempts],[1]*17)
        for attempt in reg.attempts:
            doc=attempt.document
            if doc["execution_status"]=="OK":
                continue
            self.assertTrue(doc["failure_type"] and doc["failure_message"])
            original_evidence=doc["failure_message"]+" "+ledger.canonical(doc["failure_provenance"])
            self.assertIn("preserved MOCK failure",original_evidence)
        self.assertEqual({p.name for p in f.out.iterdir() if p.name.startswith("ledger_") and p.is_dir()},
                         {"ledger_initial","ledger_final"})

    def test_interruption_preserves_partial_history_and_never_retries(self):
        f=make_mock_run(self,interrupt=True)
        done=ledger.strict_json((f.out/"ledger_completion.json").read_text())
        self.assertEqual(done["status"],"INCOMPLETE")
        self.assertEqual(done["recorded_attempts"],1)
        self.assertEqual(f.calls,["state-mean","ht"])
        self.assertTrue((f.out/"RUN_INCOMPLETE.json").is_file())
        self.assertFalse((f.out/"test_completion.json").exists())
        with self.assertRaises(ValueError):
            acceptance.validate_test_ledgers(f.out,
                run_manifest_sha256=ident.sha((f.out/"run_manifest.json").read_bytes()),run_id="mock-test")

    def test_attempt_retry_authorization_order_and_single_writer(self):
        with patch.object(r,"test_plan",return_value=tiny_plan()):
            reg,records=r.build_test_registry((tasks_fixture()[0],),run_id="mock-test",decision=decision())
        first=records[0];bundle=bundle_fixture(first.task,first.parent,states=("FAILED","OK","FAILED"))
        reg.append_attempt(bundle,attempt_no=1)
        self.assertEqual(reg.next_attempt_no((first.method_id,"pair",first.task.pair_id,first.parent.request_id)),2)
        for number in (1,2,3):
            with self.subTest(number=number),self.assertRaises(ValueError):
                reg.append_attempt(bundle,attempt_no=number)
        reg.append_attempt(bundle,attempt_no=2,authorization="explicit MOCK retry authorization")
        self.assertEqual([a.document["attempt_no"] for a in reg.attempts],[1,2])
        self.assertEqual(reg.attempts[0].document["failure_message"],"exact synthetic reason")
        for attempt in reg.attempts:
            self.assertEqual(attempt.document["status"],"POINT")
            self.assertEqual(attempt.document["output_status"]["ate"],["FAILED","OK","FAILED"])
            self.assertEqual(attempt.document["execution_request"]["request"]["identity"],
                             reg.attempts[0].document["execution_request"]["request"]["identity"])
        failures=[]
        def other_writer():
            try:reg.append_attempt(bundle,attempt_no=3,authorization="explicit")
            except ValueError as error:failures.append(str(error))
        thread=threading.Thread(target=other_writer);thread.start();thread.join()
        self.assertEqual(len(failures),1);self.assertIn("owning",failures[0])
        self.assertEqual(reg.attempt_count,2)

    def test_preflight_bad_manifest_stops_before_generation_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);delivery=root/"delivery";delivery.mkdir()
            (delivery/"source_manifest.json").write_text("{}")
            with patch.object(r,"_generate_test_data",side_effect=AssertionError("DGP forbidden")) as gen:
                with self.assertRaises(ValueError):
                    r.preflight(source_delivery=delivery,source_manifest_sha256="a"*64,
                        gate_path=root/"none",gate_sha256="a"*64,identification_path=root/"none",
                        identification_sha256="a"*64,approval_path=root/"none",approval_sha256="a"*64,
                        output_root=root/"out",run_id="mock-test")
                gen.assert_not_called()
            self.assertFalse((root/"out").exists())

    def test_runtime_all_source_files_missing_or_tampered_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"source.py";path.write_bytes(b"frozen")
            source={"files":[{"path":"source.py","bytes":6,"sha256":ident.sha(b"frozen")}]}
            r.verify_runtime_files(root,source)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                r.verify_runtime_files(root,source)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                r.verify_runtime_files(root,source)

    def test_existing_output_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/"history";out.mkdir()
            marker=out/"preserved";marker.write_bytes(b"historical evidence")
            with self.assertRaises(ValueError):
                provenance._fresh_directory(out)
            self.assertEqual(marker.read_bytes(),b"historical evidence")


class OutputIsolationTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.desktop=self.root/"Desktop";self.desktop.mkdir()
        self.runtime=self.desktop/"runtime";self.runtime.mkdir()
        self.delivery=self.desktop/"delivery";self.delivery.mkdir()

    def tree_bytes(self):
        return {p.relative_to(self.root).as_posix():p.read_bytes() if p.is_file() else None
                for p in self.root.rglob("*")}

    def reject_execution(self,target,message):
        before=self.tree_bytes()
        with patch.object(r.shared,"ROOT",self.runtime),patch.object(
                r.provenance,"verify_source_snapshot",side_effect=AssertionError("No source read needed")) as source,patch.object(
                r,"_generate_test_data",side_effect=AssertionError("DGP forbidden")) as gen,patch.object(
                r.shared,"real_adapters",side_effect=AssertionError("Real loader forbidden")) as models:
            with self.assertRaisesRegex(ValueError,message):
                r.execute_test(execute_test=True,source_delivery=self.delivery,
                    source_manifest_sha256="a"*64,gate_path=self.root/"unused-gate",
                    gate_sha256="b"*64,identification_path=self.root/"unused-decision",
                    identification_sha256="c"*64,approval_path=self.root/"unused-approval",
                    approval_sha256="d"*64,output_root=target,run_id="isolation-mock")
            source.assert_not_called();gen.assert_not_called();models.assert_not_called()
        self.assertEqual(self.tree_bytes(),before)
        self.assertFalse(target.exists())

    def test_runtime_output_rejected_before_any_output_or_generation(self):
        self.reject_execution(self.runtime/"missing_parent"/"run","outside Runtime")

    def test_delivery_output_rejected_before_any_output_or_generation(self):
        self.reject_execution(self.delivery/"missing_parent"/"run","outside Source Delivery")

    def test_git_directory_ancestor_rejected_without_git_or_writes(self):
        git=self.desktop/"git-directory";git.mkdir();(git/".git").mkdir()
        self.reject_execution(git/"missing_parent"/"run","outside Git worktrees")

    def test_git_pointer_file_ancestor_rejected_without_git_or_writes(self):
        git=self.desktop/"git-pointer";git.mkdir()
        (git/".git").write_bytes(b"gitdir: nonexistent-external-metadata\n")
        self.reject_execution(git/"missing_parent"/"run","outside Git worktrees")

    def test_valid_independent_desktop_preflight_accepts_without_creating_output(self):
        # Only receipt readers are Mocked. Actual output guards, entry hash
        # comparison and fresh-directory validation remain active.
        entry=self.runtime/"star_v2/run_test.py";entry.parent.mkdir();entry.write_bytes(b"# mock entry")
        source={"snapshot_id":"a"*64,"files":[{"path":"star_v2/run_test.py",
                "sha256":ident.sha(entry.read_bytes())}]}
        target=self.desktop/"independent_run"
        before=self.tree_bytes()
        with patch.object(r.shared,"ROOT",self.runtime),patch.object(r,"__file__",str(entry)),patch.object(
                r.provenance,"verify_source_snapshot",return_value=source),patch.object(
                r,"verify_lineage"),patch.object(r.shared,"read_gate",return_value={}),patch.object(
                r.shared,"verify_execution_source"),patch.object(r,"verify_runtime_files"),patch.object(
                r.identification,"read_decision",return_value=decision()),patch.object(
                r.test_authorization,"read_authorization"),patch.object(
                r,"_generate_test_data",side_effect=AssertionError("DGP forbidden")) as gen:
            result=r.preflight(source_delivery=self.delivery,source_manifest_sha256="a"*64,
                gate_path=self.root/"unused",gate_sha256="b"*64,
                identification_path=self.root/"unused",identification_sha256="c"*64,
                approval_path=self.root/"unused",approval_sha256="d"*64,
                output_root=target,run_id="mock-test")
            self.assertEqual(result[3],target.absolute());gen.assert_not_called()
        self.assertEqual(before,self.tree_bytes())
        self.assertFalse(target.exists())

    def test_containment_uses_resolved_components_and_preserves_fresh_guard(self):
        nested=self.runtime/"missing"/".."/"run"
        self.reject_execution(nested,"outside Runtime")
        for root in (self.runtime,self.delivery):
            with self.assertRaises(ValueError):
                r.validate_output_isolation(root,runtime_root=self.runtime,source_delivery=self.delivery)
        sibling=self.desktop/"runtime-sibling";sibling.mkdir()
        target=sibling/"independent"
        self.assertEqual(r.validate_output_isolation(target,runtime_root=self.runtime,
            source_delivery=self.delivery),target.absolute())
        self.assertEqual(provenance._fresh_directory(target),target.absolute())
        with self.assertRaises(ValueError):
            provenance._fresh_directory(sibling)
        self.assertFalse(target.exists())

if __name__=="__main__":
    unittest.main()
