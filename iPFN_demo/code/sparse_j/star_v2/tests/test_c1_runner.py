"""C.1 convergence tests: Mock-only preload, timings and authorization."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,unittest
from unittest.mock import patch
from star_v2 import run_test as r,test_authorization as auth,data_artifacts as data
import test_run_test as fixture

class C1RunnerTests(unittest.TestCase):
    def test_preload_precedes_dgp_and_tasks_exactly_once(self):
        f=fixture.make_mock_run(self)
        self.assertEqual(f.events,["preload:j","preload:kernel-old","preload:kernel-new","DGP","tasks"])
        self.assertEqual({k:v.loads for k,v in f.mock_adapters.items()},
                         {"j":1,"kernel-old":1,"kernel-new":1})
        self.assertTrue(all(v.model.state==b"frozen mock parameters/buffers" for v in f.mock_adapters.values()))
        fixture.validate_mock(f)
        self.assertEqual(f.calls,list(r.shared.METHODS))

    def test_each_loader_failure_produces_zero_dgp_and_tasks(self):
        for failed in ("j","kernel-old","kernel-new"):
            with self.subTest(model=failed):
                f=fixture.make_mock_run(self,preload_failure=failed)
                self.assertNotIn("DGP",f.events);self.assertNotIn("tasks",f.events)
                self.assertEqual(f.calls,[])
                self.assertFalse((f.out/"ledger_initial").exists())
                self.assertFalse((f.out/"ledger_final").exists())
                self.assertFalse((f.out/"truth").exists())
                marker=json.loads((f.out/"RUN_INCOMPLETE.json").read_text())
                self.assertEqual(marker["phase"],"model_preload")
                self.assertFalse(marker["formal_tasks_created"])
                self.assertIn("MOCK preload failure",marker["message"])

    def test_preload_timing_independent_scalar_sum(self):
        adapters={name:SimpleNamespace(_get_model=lambda:object()) for name in ("j","kernel-old","kernel-new")}
        durations=iter((1.,2.,3.,5.));labels=[]
        def measure(callback,label):
            labels.append(label);return callback(),{"wall_seconds":next(durations)}
        with patch.object(r.shared,"real_adapters",return_value=adapters),patch.object(r.shared,"measure_phase",side_effect=measure):
            result,setup=r.preload_models({})
        self.assertIs(result,adapters)
        self.assertEqual(labels,["adapter_construction","model_preload:j","model_preload:kernel-old","model_preload:kernel-new"])
        self.assertEqual(setup["model_setup_seconds"],11.)
        self.assertEqual([setup["model_loads"][k]["wall_seconds"] for k in adapters],[2.,3.,5.])

    def test_recorded_timing_phases_and_full_receipt(self):
        f=fixture.make_mock_run(self)
        doc=json.loads((f.out/"runtime/run_resources.json").read_text())
        t=doc["timing"];setup=doc["setup"]
        self.assertEqual(t["model_setup_seconds"],setup["adapter_construction"]["wall_seconds"]+
            sum(v["wall_seconds"] for v in setup["model_loads"].values()))
        self.assertEqual(t["data_setup_seconds"],doc["data_generation"]["wall_seconds"]+
                         doc["task_materialization"]["wall_seconds"])
        self.assertEqual(t["execution_prediction_seconds"],t["prediction_seconds"]+t["j_interval_seconds"])
        self.assertGreaterEqual(t["persistence_audit_seconds"],0.)
        self.assertGreaterEqual(f.result["total_wall_clock_seconds"],t["total_wall_clock_seconds"])
        receipt=json.loads(Path(f.result["execution_receipt_path"]).read_text())
        self.assertEqual(receipt["completion_sha256"],f.result["completion_sha256"])
        self.assertEqual(receipt["total_wall_clock_seconds"],f.result["total_wall_clock_seconds"])
        self.assertEqual(data.sha(Path(f.result["execution_receipt_path"]).read_bytes()),
                         f.result["execution_receipt_sha256"])

    def test_unbound_or_wrong_authorization_rejected(self):
        f=fixture.make_mock_run(self)
        doc=json.loads(f.inputs["authorization"][0].read_text())
        bindings=dict(run_id="mock-test",snapshot_id=f.source["snapshot_id"],
            gate_sha256=f.inputs["gate"][1],identification_sha256=f.inputs["decision"][1],
            approval_sha256=f.inputs["approval"][1])
        auth.validate_authorization(doc,**bindings)
        for field,value in (("approval_status","PENDING"),("gate_sha256","0"*64),
                            ("cohort_code",1),("run_id",None),("snapshot_id",None)):
            bad=copy.deepcopy(doc);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                auth.validate_authorization(bad,**bindings)
        with self.assertRaises(ValueError):
            auth.read_authorization(None,expected_sha256=None,**bindings)

if __name__=="__main__":unittest.main()
