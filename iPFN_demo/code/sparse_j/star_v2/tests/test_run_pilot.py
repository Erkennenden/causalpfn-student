"""Pure synthetic Pilot registry/Mock request execution. NEVER execute a cohort."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from functools import lru_cache
import contextlib,hashlib,io,json,tempfile,unittest
from unittest.mock import Mock,patch
import numpy as np
from star_v2 import run_pilot as r,contracts as c,data_artifacts as a,model_adapters as common
from star_v2.config import CONFIG
from star_v2 import artifacts as ledger_io,estimation as estimation
from star_v2.pilot_acceptance import validate_pilot_ledgers
from star_v2.j_adapter import JAdapter,INTERVAL_KIND
from star_v2.runtime_monitor import RuntimeWindow
from star_v2.coverage import unique_attempts,AttemptSelectionAmbiguous
from test_kernel_new_adapter import tasks_fixture

@lru_cache(maxsize=1)
def synthetic_pilot_tasks():
    base=tasks_fixture()[0]
    pairs=[]
    for m in range(CONFIG.pilot_mechanisms):
        for repeat in range(CONFIG.pilot_pairs_per_mechanism):
            children=[]
            for task in base.context_refs:
                changes={"cohort":"pilot","cohort_code":1,"mechanism_id":m,"pair_repeat_id":repeat,
                    "pair_id":c.pair_id_for(1,m,repeat),"context_id":c.context_id_for(1,m,repeat,task.center_t),
                    "rng_identity":c._rng(1,m,repeat)}
                meta={n:changes.get(n,getattr(task,n)) for n in c._META}
                arrays=[(n,getattr(task,n)) for n in c._ARRAY_ORDER]
                children.append(replace(task,**changes,input_hash=c._digest("star-v2-task-input-v1",meta,arrays)))
            pairs.append(c.pair_from_tasks(tuple(children)))
    return tuple(pairs)

def factory(label):
    return RuntimeWindow(label,rss_reader=lambda:1024,sample_interval=None)

def gate():
    # Synthetic declarations only, not evidence that a real Loader was executed.
    models={}
    declarations=(
        ("j","predict.load_selected_model","J","full_model_state","4"*64),
        ("kernel-old","benchmark.load_methods","FrozenMixture",
         "spec_prior_selection_excluding_query_cache","5"*64),
        ("kernel-new","pfn_social.sparse_classical_calibration.load_model","population",
         "full_parameters_and_buffers","2"*64))
    for name,loader,identity,scope,fingerprint in declarations:
        models[name]={"loader":loader,"load_status":"PASS","identity":identity,
            "state_fingerprint_scope":scope,"state_fingerprint_before":fingerprint,
            "state_fingerprint_after":fingerprint,"contexts":[
                {"center_t":center,"status":"PASS","shape":[200,4],"dtype":"float64",
                 "finite_count":800} for center in (0,1)]}
    models["j"]["intervals"]=[
        {"center_t":center,"status":"PASS","level":.95,"lower_shape":[3],"upper_shape":[3],
         "interval_kind":"frozen_J_marginal_posterior_credible"} for center in (0,1)]
    return {"schema_version":1,"gate":"REAL_MODEL_GATE","status":"PASS",
        "source_snapshot_id":"a"*64,"protocol_sha256":CONFIG.protocol_sha256,
        "assets":dict(r.ASSETS),"git":{"head":CONFIG.source_commit,
            "branch":"synthetic-git","status_short":[]},"models":models,
        "kernel_new_manifest":{"identity":"population","design_kinds":["phase4","phase4"],
            "specs_sha256":"1"*64,"model_state_sha256":"2"*64,"frozen_state_sha256":"3"*64}}

class MockJ:
    def __init__(self,failed=False):self.failed=failed;self.payloads=[]
    def predict_arm_means(self,**payload):
        self.payloads.append(payload)
        if self.failed:raise ValueError("synthetic native point failure")
        return np.tile(np.array([1.,7.,3.,6.]),(200,1))
def interval(model,**payload):
    return {"ate_lower":np.array([0.,0.,0.]),"ate_upper":np.array([10.,10.,10.]),
        "interval_level":np.array(.95),"interval_kind":np.array(INTERVAL_KIND)}

EXAMPLE={}

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pairs=synthetic_pilot_tasks()
    def setUp(self):self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
    def registry(self,**kw):
        return r.build_pilot_registry(self.pairs,run_id="synthetic-only",upstream_status=kw.pop("status","POINT"),**kw)

    def validate_synthetic_gate(self,report):
        return r.validate_gate(report,source_snapshot_id="a"*64,
            asset_reader=lambda path:r.ASSETS["models/"+path.name])

    def test_complete_synthetic_gate_evidence_schema_passes_without_model_loading(self):
        report=gate()
        with patch("torch.load",side_effect=AssertionError("checkpoint forbidden")),patch.object(
                r,"real_adapters",side_effect=AssertionError("real model construction forbidden")):
            self.assertIs(self.validate_synthetic_gate(report),report)
        self.assertEqual(set(report["models"]),{"j","kernel-old","kernel-new"})

    def test_gate_missing_models_model_or_context_rejected(self):
        report=gate();del report["models"]
        with self.assertRaises(ValueError):self.validate_synthetic_gate(report)
        for method in ("j","kernel-old","kernel-new"):
            bad=gate();del bad["models"][method]
            with self.subTest(method=method),self.assertRaises(ValueError):
                self.validate_synthetic_gate(bad)
            for center in (0,1):
                bad=gate();bad["models"][method]["contexts"].pop(center)
                with self.subTest(method=method,center=center),self.assertRaises(ValueError):
                    self.validate_synthetic_gate(bad)
            bad=gate();bad["models"][method]["contexts"][1]["center_t"]=0
            with self.assertRaises(ValueError):self.validate_synthetic_gate(bad)

    def test_gate_loader_load_identity_scope_and_fingerprint_rejections(self):
        fields=("loader","load_status","identity","state_fingerprint_scope",
                "state_fingerprint_before","state_fingerprint_after")
        mutations=(("loader","alternate.loader"),("load_status","FAIL"),("status","FAIL"),("identity",""),
            ("state_fingerprint_scope","historical_subset"),
            ("state_fingerprint_before","G"*64),("state_fingerprint_after","a"*63),
            ("state_fingerprint_after","a"*64))
        for method in ("j","kernel-old","kernel-new"):
            for field in fields:
                bad=gate();del bad["models"][method][field]
                with self.subTest(method=method,missing=field),self.assertRaises(ValueError):
                    self.validate_synthetic_gate(bad)
            for field,value in mutations:
                bad=gate();bad["models"][method][field]=value
                with self.subTest(method=method,field=field,value=value),self.assertRaises(ValueError):
                    self.validate_synthetic_gate(bad)

    def test_gate_failed_or_noncanonical_predictions_rejected_per_context(self):
        changes=(("status","FAIL"),("shape",[50,4]),("shape",[200,2,2]),
                 ("shape",[200.,4.]),("dtype","float32"),("dtype","<f8"),
                 ("finite_count",799),("finite_count",800.0))
        for method in ("j","kernel-old","kernel-new"):
            for center in (0,1):
                for field,value in changes:
                    bad=gate();bad["models"][method]["contexts"][center][field]=value
                    with self.subTest(method=method,center=center,field=field,value=value),self.assertRaises(ValueError):
                        self.validate_synthetic_gate(bad)
                for field in ("status","shape","dtype","finite_count"):
                    bad=gate();del bad["models"][method]["contexts"][center][field]
                    with self.assertRaises(ValueError):self.validate_synthetic_gate(bad)

    def test_gate_j_interval_missing_failed_or_changed_semantics_rejected(self):
        report=gate();del report["models"]["j"]["intervals"]
        with self.assertRaises(ValueError):self.validate_synthetic_gate(report)
        changes=(("status","FAIL"),("level",.90),("lower_shape",[2]),
                 ("upper_shape",[3,1]),("interval_kind","other_kind"))
        for center in (0,1):
            bad=gate();bad["models"]["j"]["intervals"].pop(center)
            with self.assertRaises(ValueError):self.validate_synthetic_gate(bad)
            for field,value in changes:
                bad=gate();bad["models"]["j"]["intervals"][center][field]=value
                with self.subTest(center=center,field=field),self.assertRaises(ValueError):
                    self.validate_synthetic_gate(bad)
            for field in ("status","level","lower_shape","upper_shape","interval_kind"):
                bad=gate();del bad["models"]["j"]["intervals"][center][field]
                with self.assertRaises(ValueError):self.validate_synthetic_gate(bad)

    def test_gate_kernel_new_manifest_identity_and_complete_state_bound_to_evidence(self):
        for field,value in (("identity","phase5"),("model_state_sha256","f"*64),
                ("checkpoint_sha256","f"*64),("format","other_format"),
                ("specs_sha256","invalid"),("frozen_state_sha256","invalid"),
                ("design_kinds",["other","phase4"]),("arm_order",["00","10","01","11"])):
            bad=gate();bad["kernel_new_manifest"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.validate_synthetic_gate(bad)
        # Extra extracted metadata, when supplied, must agree too.
        for field,value in (("specs_sha256","f"*64),("frozen_state_sha256","f"*64),
                            ("design_kinds",["phase5","phase5"])):
            bad=gate();bad["models"]["kernel-new"][field]=value
            with self.assertRaises(ValueError):self.validate_synthetic_gate(bad)

    def test_gate_each_asset_mismatch_rejected_and_incomplete_gate_stops_before_loaders(self):
        for asset in r.ASSETS:
            bad=gate();bad["assets"][asset]="f"*64
            with self.subTest(asset=asset),self.assertRaises(ValueError):
                self.validate_synthetic_gate(bad)
        incomplete=gate();del incomplete["models"]
        path=self.root/"incomplete_gate.json";path.write_bytes(a.json_bytes(incomplete))
        receipt=hashlib.sha256(path.read_bytes()).hexdigest()
        reader=Mock(side_effect=AssertionError("invalid evidence must fail before asset use"))
        with patch.object(r.provenance,"verify_source_snapshot",
                          return_value={"snapshot_id":"a"*64}),patch.object(
             r,"real_adapters",side_effect=AssertionError("no true loader")) as loaders,patch(
             "torch.load",side_effect=AssertionError("no checkpoint")) as checkpoint,patch(
             "star_v2.dgp.generate_star_mechanism",side_effect=AssertionError("no DGP")) as dgp:
            with self.assertRaisesRegex(ValueError,"model evidence"):
                r.execute_pilot(execute_pilot=True,gate_path=path,gate_sha256=receipt,
                    source_delivery=self.root,source_manifest_sha256="b"*64,
                    output_root=self.root/"must_not_exist",run_id="synthetic",
                    upstream_status="POINT",asset_reader=reader)
        loaders.assert_not_called();checkpoint.assert_not_called();dgp.assert_not_called()
        reader.assert_not_called()
        self.assertFalse((self.root/"must_not_exist").exists())

    def test_default_plan_no_dgp_no_model_or_output(self):
        with patch("star_v2.dgp.generate_star_mechanism",side_effect=AssertionError("DGP forbidden")),patch.object(
                r,"real_adapters",side_effect=AssertionError("loader forbidden")):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(r.main([]),0)
        plan=json.loads(out.getvalue())
        self.assertEqual(plan["mode"],"PLAN_ONLY")
        self.assertEqual((plan["mechanisms"],plan["pairs_per_mechanism"],len(plan["pairs"])),(5,2,10))
        self.assertEqual(set(plan["methods"]),set(r.METHODS))
        self.assertFalse(plan["models_called"]);self.assertFalse(plan["dgp_generated"])
        self.assertEqual(list(self.root.iterdir()),[])
    def test_complete_registry_nine_method_identity_context_counts(self):
        reg,records=self.registry()
        self.assertTrue(reg.sealed)
        self.assertEqual(len(reg.tasks),30)
        self.assertEqual(len(records),90)
        self.assertEqual({record.method_id for record in records},set(r.METHODS))
        counts=r.registry_counts(reg)
        for method in r.METHODS:self.assertEqual(counts[method+":pair"],10)
        for method in r.CONTEXT_METHODS:self.assertEqual(counts[method+":context"],20)
        self.assertEqual(len(reg.requests),170)
        self.assertEqual(len(reg.unexecuted_requests()),170)
        j=[v for v in records if v.method_id=="j"]
        self.assertTrue(all("interval" in child.requested_outputs for v in j for child in v.children))
        for v in records:
            self.assertEqual(v.parent.task_id,v.task.pair_id)
            for i,child in enumerate(v.children):
                self.assertEqual(child.parent_request_id,v.parent.request_id)
                self.assertEqual(child.input_hash,v.task.context_refs[i].input_hash) if hasattr(child,"input_hash") else self.assertEqual(child.input_hashes,(v.task.context_refs[i].input_hash,))
    def test_test_formal_incomplete_and_duplicate_pilot_inventory_rejected(self):
        for cohort in (2,True,"test"):
            with self.assertRaises(ValueError):r.pilot_plan(cohort_code=cohort)
        with self.assertRaises(ValueError):r.main(["--cohort","formal"])
        for tasks in (self.pairs[:-1],self.pairs[:-1]+(self.pairs[0],)):
            with self.assertRaises(ValueError):
                r.build_pilot_registry(tasks,run_id="x",upstream_status="POINT")
    def test_no_gate_or_missing_arguments_blocks_execution_before_dgp(self):
        with patch.object(r,"execute_pilot",side_effect=AssertionError("must not execute")):
            with self.assertRaises(ValueError):r.main(["--execute-pilot"])
        with self.assertRaises(ValueError):
            r.validate_gate({},source_snapshot_id="a"*64,asset_reader=lambda p:"x")
    def test_gate_checkpoint_source_and_actual_asset_mismatch_rejected(self):
        valid=gate()
        # Dataclass design_kinds is a tuple in the trusted Gate schema.
        valid["kernel_new_manifest"]["design_kinds"]=("phase4","phase4")
        calls=[]
        def reader(path):
            relative="models/"+path.name;calls.append(relative);return r.ASSETS[relative]
        self.assertIs(r.validate_gate(valid,source_snapshot_id="a"*64,asset_reader=reader),valid)
        self.assertEqual(set(calls),set(r.ASSETS))
        for change in ({"status":"FAILED"},{"source_snapshot_id":"b"*64},{"assets":{}}):
            bad={**valid,**change}
            with self.assertRaises(ValueError):r.validate_gate(bad,source_snapshot_id="a"*64,asset_reader=reader)
        with self.assertRaises(ValueError):
            r.validate_gate(valid,source_snapshot_id="a"*64,asset_reader=lambda path:"f"*64)
    def test_gate_report_anchor_and_loader_called_zero_on_bad_report(self):
        report=gate();report["kernel_new_manifest"]["design_kinds"]=("phase4","phase4")
        report["kernel_new_manifest"]["arm_order"]=["00","01","10","11"]
        path=self.root/"gate.json";path.write_bytes(a.json_bytes(report))
        with self.assertRaises(ValueError):
            r.read_gate(path,expected_sha256="f"*64,source_snapshot_id="a"*64)
        loaded=r.read_gate(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_snapshot_id="a"*64,asset_reader=lambda path:r.ASSETS["models/"+path.name])
        self.assertEqual(loaded["kernel_new_manifest"]["design_kinds"],["phase4","phase4"])
    def test_existing_baseline_model_and_oracle_adapter_dispatch(self):
        reg,records=self.registry();sentinel=object();adapters={}
        for method in ("kernel-old","kernel-new","j"):
            adapters[method]=Mock();adapters[method].estimate_pair.return_value=sentinel
        for method in r.METHODS:
            record=next(v for v in records if v.method_id==method)
            with patch.object(r.baselines,"estimate_pair",return_value=sentinel) as base,patch.object(
                 r.oracle_adapters,"estimate_oracle_context_pair",return_value=sentinel) as ctx,patch.object(
                 r.oracle_adapters,"estimate_oracle_pair",return_value=sentinel) as pair:
                self.assertIs(r.invoke(record,adapters,true_h=np.zeros((200,3))),sentinel)
                if method in r.baselines.METHOD_IDS:base.assert_called_once_with(record.task,record.parent,method)
                elif method=="oracle-context":ctx.assert_called_once()
                elif method=="oracle-pair":pair.assert_called_once()
                else:
                    args,kwargs=adapters[method].estimate_pair.call_args
                    self.assertEqual(args,(record.task,record.parent))
                    self.assertEqual(set(kwargs),{"context_requests"})
    def test_actual_execution_requires_extracted_snapshot_not_original_worktree(self):
        with self.assertRaisesRegex(ValueError,"pin"):
            r.verify_execution_source({},{"files":[]})
        (self.root/".git").mkdir()
        with patch.object(r,"ROOT",self.root),self.assertRaisesRegex(ValueError,"original Git worktree"):
            r.verify_execution_source({"runtime_root":str(self.root)},{"files":[]})

    def test_real_loader_construction_is_lazy_and_reuses_original_entrypoints(self):
        valid=gate();valid["kernel_new_manifest"]["design_kinds"]=("phase4","phase4")
        with patch("star_v2.j_adapter._default_loader",side_effect=AssertionError("no real loader")),patch(
             "torch.load",side_effect=AssertionError("no checkpoint")):
            adapters=r.real_adapters(valid)
        self.assertEqual(set(adapters),{"j","kernel-old","kernel-new"})
        self.assertTrue(all(getattr(v,"_model",None) is None for v in adapters.values()))
    def test_mock_single_request_outputs_runtime_attempt_and_shared_identity(self):
        reg,records=self.registry();record=next(v for v in records if v.method_id=="j")
        model=MockJ();loader=Mock(return_value=model);adapter=JAdapter(loader=loader,interval_predictor=interval)
        # A test-only no-work setup, outside point timing; actual driver does the same.
        adapter._get_model()
        before=record.task.context_refs[0].input_hash
        inputs=tuple(d for context in record.task.context_refs
                     for d in a.save_factual_inputs(self.root,context))
        bundle,index=r.execute_registered_request(reg,record,{"j":adapter},self.root,
            true_h=np.zeros((200,3)),window_factory=factory,source_receipts=inputs)
        self.assertEqual(bundle.status,"POINT")
        self.assertEqual(bundle.output_status["ate"],("OK",)*3)
        np.testing.assert_array_equal(bundle.ate,np.array([2.,3.,5.]))
        self.assertEqual(len(reg.attempts),3)
        outputs=a.read_execution_records(self.root,reg,require_complete=True)
        self.assertEqual(len(outputs),3)
        self.assertEqual(record.task.context_refs[0].input_hash,before)
        self.assertEqual(loader.call_count,1)
        self.assertEqual(len(model.payloads),2)
        self.assertTrue(all(set(p)==set(c.FACTUAL_KEYS) for p in model.payloads))
        self.assertTrue(all(p["outcomes_observed"].dtype==np.float32 for p in model.payloads))
        runtime=json.loads(a.verify_file(self.root,index["runtime"]))
        self.assertEqual(len(runtime["context_intervals"]),2)
        self.assertIsNone(runtime["pair_point"]["gpu"]["peak_allocated_bytes"])
        self.assertFalse(any(self.root.glob("ledger_after_*")))
        self.assertFalse((self.root/"ledger_final").exists())
        self.assertTrue(all(d in index["outputs"] for d in inputs))
        context_record=next(row for row in outputs if row["execution_key"][1]=="context")
        interval_file=next(d for d in context_record["outputs"] if d["path"].startswith("intervals/"))
        interval_doc=json.loads(a.verify_file(self.root,interval_file))
        self.assertEqual(interval_doc["diagnostics"]["interval"]["interval_kind"],INTERVAL_KIND)
        EXAMPLE.update({"counts":r.registry_counts(reg),"execution_key":index["execution_key"],
            "attempt_no":index["attempt_no"],"outputs":len(index["outputs"]),
            "runtime_path":index["runtime"]["path"],"context_intervals":2})
    def test_failure_history_retry_requires_authorization_and_no_selection(self):
        reg,records=self.registry();record=next(v for v in records if v.method_id=="j")
        model=MockJ(failed=True);adapter=JAdapter(loader=lambda:model,interval_predictor=interval)
        failed,_=r.execute_registered_request(reg,record,{"j":adapter},self.root,
            true_h=np.zeros((200,3)),window_factory=factory)
        self.assertEqual(failed.output_status["ate"],("FAILED",)*3)
        self.assertEqual(len(reg.attempts),3)
        with self.assertRaises(ValueError):
            r.execute_registered_request(reg,record,{"j":adapter},self.root,
                true_h=np.zeros((200,3)),attempt_no=2,window_factory=factory)
        model.failed=False
        success,_=r.execute_registered_request(reg,record,{"j":adapter},self.root,
            true_h=np.zeros((200,3)),attempt_no=2,authorization="explicit synthetic retry",
            window_factory=factory)
        self.assertEqual(success.output_status["ate"],("OK",)*3)
        self.assertEqual(len(reg.attempts),6)
        self.assertEqual(len(a.read_execution_records(self.root,reg,require_complete=True)),6)
        with self.assertRaises(AttemptSelectionAmbiguous):unique_attempts(reg,reg.requests)
    def test_not_identified_preserves_upstream_and_zero_model_calls(self):
        reason={"failure_type":"SYNTHETIC_UPSTREAM","failure_message":"exact independent reason",
                "failure_provenance":{"stage":"upstream_fixture"}}
        reg,records=self.registry(status="NOT_IDENTIFIED",upstream_failure=reason)
        record=next(v for v in records if v.method_id=="j")
        loader=Mock(side_effect=AssertionError("NOT_IDENTIFIED must not load"))
        adapter=JAdapter(loader=loader,interval_predictor=Mock(side_effect=AssertionError("no interval")))
        result,_=r.execute_registered_request(reg,record,{"j":adapter},self.root,
            true_h=np.zeros((200,3)),window_factory=factory)
        loader.assert_not_called()
        self.assertEqual(result.status,"NOT_IDENTIFIED")
        self.assertEqual(result.failure_message,reason["failure_message"])
        self.assertEqual(dict(result.failure_provenance),reason["failure_provenance"])

    def mock_complete_driver(self, target, *, interrupt=False):
        # Test-only stand-ins: no DGP, real model, estimator or official RNG.
        source_root=self.root/"synthetic_source"
        (source_root/"star_v2").mkdir(parents=True)
        (source_root/"star_v2/run_pilot.py").write_text("# synthetic source identity only\n")
        receipt=r.provenance.build_source_snapshot(source_root,self.root/"synthetic_delivery",
            git_head=CONFIG.source_commit,files=(
                r.provenance.SourceFile("star_v2/run_pilot.py","implementation"),))
        source=r.provenance.verify_source_snapshot(receipt.directory,
            expected_manifest_sha256=receipt.manifest_sha256)
        report=gate();report["source_snapshot_id"]=receipt.snapshot_id
        gatepath=self.root/"synthetic_gate.json";gatepath.write_bytes(a.json_bytes(report))
        assets=self.root/"synthetic_assets";(assets/"models").mkdir(parents=True)
        for name in r.ASSETS:(assets/name).write_bytes(b"MOCK")
        mechanisms={m:SimpleNamespace(h=np.zeros((200,3)),mu=np.zeros((200,2,2)),
            ite=np.zeros((200,3)),ate=np.zeros(3),coefficients=np.zeros(25))
            for m in range(5)}
        standins={}
        for task in self.pairs:
            standins[(task.mechanism_id,task.pair_repeat_id)]=SimpleNamespace(
                mechanism=mechanisms[task.mechanism_id],noise=np.zeros(200),
                leaf_treatment=task.context_refs[0].treatment[1:],
                observed_indices=task.context_refs[0].observed_indices,
                contexts=tuple(SimpleNamespace(full_outcomes=np.zeros(200)) for _ in (0,1)),
                public_task=task)
        def invoke(record,adapters,*,true_h):
            if record.method_id=="state-mean":
                return estimation.PairEstimateBundle.from_outputs(record.parent,record.task,"state-mean",
                    ate=np.array([1.,2.,3.]),output_status={"ate":("OK",)*3},arm_results=(None,None))
            return r._failed(record,RuntimeError("preserved synthetic failure"),true_h=true_h)
        real_execute=r.execute_registered_request
        calls=[]
        def execute(*args,**kwargs):
            calls.append(args[1].parent.request_id)
            if interrupt and len(calls)==2:raise OSError("synthetic output interruption")
            return real_execute(*args,**kwargs)
        with patch.object(r,"read_gate",return_value=report),patch.object(r.provenance,
             "verify_source_snapshot",return_value=source),patch.object(r,"pair_from_private",
             side_effect=lambda item:item.public_task),patch.object(r,"invoke",side_effect=invoke),patch.object(
             r,"execute_registered_request",side_effect=execute),patch.object(ledger_io,"write_snapshot",
             wraps=ledger_io.write_snapshot) as snapshots,patch("torch.load",side_effect=AssertionError(
             "real checkpoint forbidden")),patch.object(r,"real_adapters",side_effect=AssertionError(
             "real adapter forbidden")):
            args=dict(execute_pilot=True,gate_path=gatepath,gate_sha256=hashlib.sha256(gatepath.read_bytes()).hexdigest(),
                source_delivery=receipt.directory,source_manifest_sha256=receipt.manifest_sha256,
                output_root=target,run_id="storage-mock",upstream_status="POINT",asset_root=assets,
                mechanism_factory=lambda *,cohort_code,mechanism_id:mechanisms[mechanism_id],
                pair_factory=lambda mech,*,pair_repeat_id:standins[(next(m for m in mechanisms if mechanisms[m] is mech),pair_repeat_id)],
                adapter_factory=lambda gate:{},window_factory=factory)
            if interrupt:
                with self.assertRaisesRegex(OSError,"synthetic output interruption"):
                    r.execute_pilot(**args)
                registry=None
            else:registry=r.execute_pilot(**args)
            self.assertEqual(snapshots.call_count,2)
            self.assertEqual([Path(call.args[1]).name for call in snapshots.call_args_list],
                             ["ledger_initial","ledger_final"])
        return registry,calls

    def test_complete_mock_driver_two_ledgers_all_success_failure_and_receipts(self):
        target=self.root/"mock_run"
        registry,calls=self.mock_complete_driver(target)
        self.assertEqual(len(calls),90)
        self.assertEqual((len(registry.tasks),len(registry.requests),len(registry.attempts)),(30,170,170))
        self.assertEqual(sum(at.document["execution_status"]=="OK" for at in registry.attempts),10)
        self.assertEqual(sum(at.document["execution_status"]=="FAILED" for at in registry.attempts),160)
        self.assertTrue(all(at.document["failure_message"] for at in registry.attempts
                            if at.document["execution_status"]=="FAILED"))
        runsha=hashlib.sha256((target/"run_manifest.json").read_bytes()).hexdigest()
        checked=validate_pilot_ledgers(target,run_manifest_sha256=runsha)
        self.assertEqual((checked["status"],checked["attempts"],checked["failed_attempts"]),("COMPLETE",170,160))
        self.assertEqual(checked["seal_sha256"],registry.seal_sha256)
        self.assertEqual(len(a.read_execution_records(target,registry,require_complete=True)),170)
        self.assertEqual({p.name for p in target.glob("ledger_*") if p.is_dir()},
                         {"ledger_initial","ledger_final"})
        completion=json.loads((target/"ledger_completion.json").read_bytes())
        restored=ledger_io.read_snapshot(target/"ledger_final",
            expected_manifest_sha256=completion["final_manifest_sha256"])
        self.assertEqual(tuple(at.content for at in restored.attempts),
                         tuple(at.content for at in registry.attempts))
        self.assertEqual(dict(restored.requests),dict(registry.requests))
        EXAMPLE["storage_optimization"]={
            "initial_ledger_bytes":sum(p.stat().st_size for p in (target/"ledger_initial").iterdir()),
            "final_ledger_bytes":sum(p.stat().st_size for p in (target/"ledger_final").iterdir()),
            "completion_marker_bytes":(target/"ledger_completion.json").stat().st_size,
            "attempts":170,"successful_attempts":10,"failed_attempts":160}
        original=(target/"ledger_completion.json").read_bytes()
        for field,value in (("status","INCOMPLETE"),("recorded_attempts",169),
                            ("final_manifest_sha256","f"*64),("registry_seal_sha256","e"*64)):
            changed=json.loads(original);changed[field]=value
            (target/"ledger_completion.json").write_bytes(a.json_bytes(changed))
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_pilot_ledgers(target,run_manifest_sha256=runsha)
        (target/"ledger_completion.json").write_bytes(original)
        (target/"ledger_after_1").mkdir()
        with self.assertRaisesRegex(ValueError,"Only initial/final"):
            validate_pilot_ledgers(target,run_manifest_sha256=runsha)

    def test_caught_output_error_saves_explicit_incomplete_without_retry(self):
        target=self.root/"interrupted_mock_run"
        _,calls=self.mock_complete_driver(target,interrupt=True)
        self.assertEqual(len(calls),2)
        doc=json.loads((target/"ledger_completion.json").read_bytes())
        self.assertEqual(doc["status"],"INCOMPLETE")
        self.assertEqual(doc["recorded_attempts"],1)
        self.assertEqual(doc["failure"],{"type":"OSError","message":"synthetic output interruption"})
        runsha=hashlib.sha256((target/"run_manifest.json").read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,"INCOMPLETE"):
            validate_pilot_ledgers(target,run_manifest_sha256=runsha)
        partial=validate_pilot_ledgers(target,run_manifest_sha256=runsha,require_complete=False)
        self.assertEqual((partial["attempts"],partial["unexecuted"]),(1,169))

    def test_per_attempt_artifacts_byte_identical_to_frozen_parent_runner(self):
        namespace=dict(vars(r))
        exec(_LEGACY_EXECUTE_SOURCE,namespace)
        legacy=namespace["execute_registered_request"]
        destinations=[self.root/"parent_outputs",self.root/"optimized_outputs"]
        for runner,destination in zip((legacy,r.execute_registered_request),destinations):
            destination.mkdir()
            reg,records=self.registry();record=next(v for v in records if v.method_id=="j")
            adapter=JAdapter(loader=lambda:MockJ(),interval_predictor=interval)
            adapter._get_model()
            def fixed_measure(callback,method_id,**kwargs):
                return callback(),{"context_intervals":[],"test_only_fixed_runtime":True}
            namespace["measure_pair"]=fixed_measure
            with patch.object(r,"measure_pair",side_effect=fixed_measure):
                runner(reg,record,{"j":adapter},destination,
                       true_h=np.zeros((200,3)),window_factory=factory)
        def hashes(root):
            return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*") if p.is_file()
                    and not p.relative_to(root).parts[0].startswith("ledger_after_")}
        self.assertTrue(hashes(destinations[0]))
        self.assertEqual(hashes(destinations[0]),hashes(destinations[1]))
        self.assertEqual(len(list(destinations[0].glob("ledger_after_*"))),3)
        self.assertFalse(list(destinations[1].glob("ledger_*")))


    def test_runner_preflight_and_warm_retry_use_indexes_before_any_model_work(self):
        reg,records=self.registry();record=next(v for v in records if v.method_id=="j")
        model=MockJ();adapter=JAdapter(loader=lambda:model,interval_predictor=interval)
        adapter._get_model()
        r.execute_registered_request(reg,record,{"j":adapter},self.root,
            true_h=np.zeros((200,3)),window_factory=factory)
        with patch.object(type(reg),"attempts",property(
                lambda self:(_ for _ in ()).throw(AssertionError("Runner scanned history")))):
            r.execute_registered_request(reg,record,{"j":adapter},self.root,true_h=np.zeros((200,3)),
                attempt_no=2,authorization="explicit indexed retry",window_factory=factory)
        self.assertEqual([at.document["attempt_no"] for at in reg.attempts],[1,1,1,2,2,2])
        self.assertEqual(len(a.read_execution_records(self.root,reg,require_complete=True)),6)
        parent_key=("j","pair",record.task.pair_id,record.parent.request_id)
        reg._last_attempt_no[parent_key]=0
        before={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file()}
        with patch.object(r,"invoke",side_effect=AssertionError("index corruption must block before model")) as invoke:
            with self.assertRaises(ValueError):
                r.execute_registered_request(reg,record,{"j":adapter},self.root,true_h=np.zeros((200,3)),
                    attempt_no=3,authorization="explicit fixture",window_factory=factory)
            invoke.assert_not_called()
        self.assertEqual(before,{p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in self.root.rglob("*") if p.is_file()})



_LEGACY_EXECUTE_SOURCE = "def execute_registered_request(registry,record,adapters,output_root,*,true_h,\n                               attempt_no=1,authorization=None,window_factory=RuntimeWindow,source_receipts=()):\n    \"\"\"Explicit single Attempt, never a retry/selection loop.\"\"\"\n    parent_key=(record.method_id,\"pair\",record.task.pair_id,record.parent.request_id)\n    previous=[a for a in registry.attempts if a.execution_key==parent_key]\n    require(attempt_no==len(previous)+1 and (attempt_no==1 or authorization),\n            \"Explicit next Attempt number and retry authorization required\")\n    # Validate all parent/child attempt slots BEFORE model work.\n    for request in (record.parent,)+record.children:\n        key=(record.method_id,request.scope,request.task_id,request.request_id)\n        require(key in registry.requests and len([a for a in registry.attempts if a.execution_key==key])\n                ==attempt_no-1,\"Request/Attempt identity conflict\")\n    try:\n        bundle,runtime=measure_pair(lambda:invoke(record,adapters,true_h=true_h),\n                                   record.method_id,window_factory=window_factory)\n    except MeasuredFailure as exc:\n        runtime=exc.record\n        bundle=_failed(record,exc.original,true_h=true_h)\n    for arm in bundle.arm_results:\n        if arm is None:continue\n        registry.append_attempt(arm,attempt_no=attempt_no,authorization=authorization)\n        interval=next((r for r in runtime[\"context_intervals\"]\n                       if r[\"context_id\"]==arm.context_ids[0]),None)\n        data.save_attempt_outputs(output_root,registry,arm,attempt_no=attempt_no,runtime={\n            \"scope\":\"context\",\"parent_execution_key\":parent_key,\n            \"point_runtime_seconds\":None,\"point_runtime_reason\":\"Measured at Pair scope, not independent contexts\",\n            \"interval\":interval,\"interval_reason\":None if interval else \"Not called/requested or dependency failed\"},\n            source_receipts=tuple(d for d in source_receipts if d[\"path\"].startswith(\n                \"factual/inputs/\"+arm.context_ids[0]+\".\")))\n        data.export_ledger(registry,Path(output_root)/(\"ledger_after_\"+str(len(registry.attempts))))\n    registry.append_attempt(bundle,attempt_no=attempt_no,authorization=authorization)\n    index=data.save_attempt_outputs(output_root,registry,bundle,attempt_no=attempt_no,runtime=runtime,\n                                    source_receipts=source_receipts)\n    # Immutable full-ledger snapshots: preserve prior failures and every update.\n    data.export_ledger(registry,Path(output_root)/(\"ledger_after_\"+str(len(registry.attempts))))\n    return bundle,index\n"

if __name__=="__main__":unittest.main()
