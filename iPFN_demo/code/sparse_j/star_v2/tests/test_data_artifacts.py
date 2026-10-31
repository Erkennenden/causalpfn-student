"""Synthetic in-memory public data only; all outputs in temporary directories."""
from pathlib import Path
from dataclasses import fields
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from star_v2 import data_artifacts as a,artifacts as ledger_io,task_registry as ledger
from star_v2.coverage import unique_attempts,AttemptSelectionAmbiguous
from star_v2.contracts import predictor_inputs,FACTUAL_KEYS
from test_task_registry import registry_fixture,bundle_fixture

EXAMPLE={}

class DataArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reg,self.task,self.parent,self.children=registry_fixture()
    def test_five_arrays_float64_bytes_hash_and_storage_roundtrip(self):
        task=self.task.context_refs[0]
        before={k:(id(v),v.dtype.str,v.tobytes()) for k,v in predictor_inputs(task).items()}
        task_bytes={k:getattr(task,k).tobytes() for k in FACTUAL_KEYS}
        digest=task.input_hash
        descriptors=a.save_factual_inputs(self.root,task)
        loaded=a.load_factual_inputs(self.root,descriptors)
        self.assertEqual(set(loaded),set(FACTUAL_KEYS))
        self.assertEqual(loaded["outcomes_observed"].dtype,np.dtype("float64"))
        for key in FACTUAL_KEYS:
            self.assertEqual(loaded[key].dtype.str,getattr(task,key).dtype.str)
            self.assertEqual(loaded[key].tobytes(),task_bytes[key])
            self.assertFalse(np.shares_memory(loaded[key],getattr(task,key)))
            with self.assertRaises(ValueError):loaded[key].setflags(write=True)
        self.assertEqual(task.input_hash,digest)
        self.assertEqual(task_bytes,{k:getattr(task,k).tobytes() for k in FACTUAL_KEYS})
        for descriptor in descriptors:
            blob=(self.root/descriptor["path"]).read_bytes()
            self.assertEqual(descriptor["sha256"],hashlib.sha256(blob).hexdigest())
            self.assertEqual(descriptor["bytes"],len(blob))
    def test_privileged_extra_input_and_wrong_factual_dtype_rejected(self):
        inputs=dict(predictor_inputs(self.task.context_refs[0]))
        for bad in ({**inputs,"truth":np.zeros(1)},
                    {**inputs,"outcomes_observed":inputs["outcomes_observed"].astype(np.float32)}):
            with self.assertRaises(ValueError):
                a.save_arrays(self.root,"factual/inputs/bad",bad,{})
        with self.assertRaises(ValueError):a.save_factual_inputs(self.root,inputs)
    def test_private_truth_are_separate_and_not_predictor_inputs(self):
        for partition in ("truth","factual/simulator_private"):
            descriptors=a.save_evaluator_arrays(self.root,partition,"synthetic",
                {"full_y":np.arange(200,dtype=np.float64)},{"synthetic":True})
            with patch.object(a.np,"load",side_effect=AssertionError("must not open privileged data")):
                with self.assertRaises(ValueError):a.load_factual_inputs(self.root,descriptors)
            _,meta=a.load_arrays(self.root,descriptors)
            self.assertEqual(meta["access"],"simulator_evaluator_only")
    def test_npz_uses_allow_pickle_false_and_keeps_endian_shape_nan_and_bytes(self):
        array=np.array([[1.,np.nan],[-0.,2.]],dtype=">f8")
        descriptors=a.save_arrays(self.root,"predictions/example",{"values":array},{})
        original=a.np.load
        with patch.object(a.np,"load",wraps=original) as load:
            arrays,_=a.load_arrays(self.root,descriptors)
        self.assertIs(load.call_args.kwargs["allow_pickle"],False)
        self.assertEqual(arrays["values"].dtype.str,">f8")
        self.assertEqual(arrays["values"].shape,(2,2))
        self.assertEqual(arrays["values"].tobytes(),array.tobytes())
        self.assertTrue(np.isnan(arrays["values"][0,1]))
    def test_object_arrays_and_nonfinite_metadata_rejected_before_write(self):
        with self.assertRaises(ValueError):
            a.save_arrays(self.root,"predictions/object",{"x":np.array([{}],dtype=object)},{})
        with self.assertRaises(ValueError):
            a.save_arrays(self.root,"predictions/meta",{"x":np.zeros(1)},{"bad":float("nan")})
        self.assertEqual(list(self.root.iterdir()),[])
    def test_complete_support_all_arrays_and_flags_preserved(self):
        bundle=bundle_fixture(self.task,self.parent)
        self.reg.append_attempt(bundle,attempt_no=1)
        record=a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={"synthetic":True})
        desc=tuple(d for d in record["outputs"] if d["path"].startswith("support/"))
        arrays,meta=a.load_arrays(self.root,desc)
        def compare(s,prefix):
            for f in fields(s):
                value=getattr(s,f.name)
                key=prefix+"."+f.name
                if type(value) is np.ndarray:
                    self.assertEqual(arrays[key].dtype,value.dtype)
                    self.assertEqual(arrays[key].shape,value.shape)
                    self.assertEqual(arrays[key].tobytes(),value.tobytes())
                elif f.name=="context_supports":
                    for i,child in enumerate(value):compare(child,key+"."+str(i))
        compare(bundle.support,"support")
        self.assertIn("support.empirical_extrapolation",arrays)
        self.assertIn("support.context_supports.0.context_design_extrapolation",arrays)
    def test_failed_partial_outputs_nan_and_status_retained(self):
        bundle=bundle_fixture(self.task,self.parent,states=("FAILED","OK","NOT_ESTIMABLE"))
        self.reg.append_attempt(bundle,attempt_no=1)
        record=a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        desc=tuple(d for d in record["outputs"] if d["path"].startswith("pair_estimates/"))
        arrays,meta=a.load_arrays(self.root,desc)
        self.assertTrue(np.isnan(arrays["ate"][0]))
        self.assertEqual(arrays["ate"][1],2.)
        self.assertTrue(np.isnan(arrays["ate"][2]))
        self.assertEqual(meta["output_status"]["ate"],["FAILED","OK","NOT_ESTIMABLE"])
        self.assertEqual(meta["failure_message"],"exact synthetic reason")
    def test_interval_failure_provenance_is_complete(self):
        from test_coverage import coverage_bundle
        reg,task,parent,children=registry_fixture(outputs=("ate","interval"))
        bundle=coverage_bundle(task.context_refs[0],children[0],0,mode="all_failed")
        reg.append_attempt(bundle,attempt_no=1)
        record=a.save_attempt_outputs(self.root,reg,bundle,attempt_no=1,runtime={})
        descriptor=next(d for d in record["outputs"] if d["path"].startswith("intervals/"))
        doc=json.loads(a.verify_file(self.root,descriptor))
        self.assertEqual([i["status"] for i in doc["intervals"]],["FAILED"]*3)
        self.assertEqual([i["source"] for i in doc["intervals"]],["model_posterior"]*3)
        self.assertEqual(doc["sampling_uncertainty"],
            {"status":"FAILED","source":"model_posterior","statistical_unit":"context","effective_sample_count":None})
        self.assertIn("stage",doc["failure_provenance"])
        self.assertEqual(doc["failure_type"],"SCRIPTED_DEPENDENCY")
    def test_multiple_attempts_are_append_only_and_ambiguity_remains(self):
        for number,states in ((1,("FAILED",)*3),(2,("OK",)*3)):
            bundle=bundle_fixture(self.task,self.parent,states=states)
            self.reg.append_attempt(bundle,attempt_no=number,authorization="synthetic retry" if number==2 else None)
            record=a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=number,runtime={})
        records=a.read_execution_records(self.root,self.reg,require_complete=True)
        self.assertEqual([r["attempt_no"] for r in records],[1,2])
        self.assertNotEqual(records[0]["outputs"][0]["path"],records[1]["outputs"][0]["path"])
        with self.assertRaises(AttemptSelectionAmbiguous):unique_attempts(self.reg,self.reg.requests)
        EXAMPLE.update({"execution_key":records[0]["execution_key"],"attempt_no":1,
                        "attempt_identity":records[0]["attempt_identity"],
                        "output_count":len(records[0]["outputs"]),
                        "runtime_path":records[0]["runtime"]["path"]})
    def test_append_uses_run_local_index_and_final_validation_is_full(self):
        original_read = a.read_execution_records
        original_verify = a.verify_file
        with patch.object(a, "read_execution_records", wraps=original_read) as read, \
             patch.object(a, "verify_file", wraps=original_verify) as verify:
            for number, states in ((1, ("FAILED",)*3), (2, ("OK",)*3)):
                bundle = bundle_fixture(self.task, self.parent, states=states)
                self.reg.append_attempt(bundle, attempt_no=number,
                    authorization="synthetic retry" if number == 2 else None)
                a.save_attempt_outputs(self.root, self.reg, bundle,
                                       attempt_no=number, runtime={})
            # No historical scan/hash occurs during either append.
            self.assertEqual(read.call_count, 0)
            self.assertEqual(verify.call_count, 0)
            records = a.read_execution_records(self.root, self.reg, require_complete=True)
            self.assertEqual(len(records), 2)
            # The explicit final audit still reads and hashes every referenced file.
            self.assertEqual(read.call_count, 1)
            self.assertGreater(verify.call_count, 0)

    def test_external_index_change_forces_full_revalidation(self):
        first = bundle_fixture(self.task, self.parent)
        self.reg.append_attempt(first, attempt_no=1)
        first_record = a.save_attempt_outputs(self.root, self.reg, first,
                                              attempt_no=1, runtime={})
        index = self.root / "execution_records.jsonl"
        with index.open("ab") as stream:
            stream.write(a.json_bytes(first_record))
            stream.flush()
        second = bundle_fixture(self.task, self.parent, states=("OK",)*3)
        self.reg.append_attempt(second, attempt_no=2, authorization="synthetic retry")
        with patch.object(a, "read_execution_records", wraps=a.read_execution_records) as read:
            with self.assertRaises(ValueError):
                a.save_attempt_outputs(self.root, self.reg, second,
                                       attempt_no=2, runtime={})
            self.assertEqual(read.call_count, 1)

    def test_attempt_output_must_follow_registry_single_writer_order(self):
        first = bundle_fixture(self.task, self.parent)
        second = bundle_fixture(self.task, self.parent, states=("OK",)*3)
        self.reg.append_attempt(first, attempt_no=1)
        self.reg.append_attempt(second, attempt_no=2, authorization="synthetic retry")
        with self.assertRaisesRegex(ValueError, "single-writer Registry order"):
            a.save_attempt_outputs(self.root, self.reg, first,
                                   attempt_no=1, runtime={})
        self.assertFalse((self.root / "execution_records.jsonl").exists())

    def test_missing_prior_attempt_index_fails_closed(self):
        first = bundle_fixture(self.task, self.parent)
        self.reg.append_attempt(first, attempt_no=1)
        with patch.object(a, "_append_execution_record", side_effect=OSError("synthetic pre-append failure")):
            with self.assertRaisesRegex(OSError, "pre-append failure"):
                a.save_attempt_outputs(self.root, self.reg, first,
                                       attempt_no=1, runtime={})
        second = bundle_fixture(self.task, self.parent, states=("OK",)*3)
        self.reg.append_attempt(second, attempt_no=2, authorization="synthetic retry")
        # The first Attempt has no execution record. A later Attempt must not
        # proceed merely because the append-index cache still exists.
        with self.assertRaisesRegex(ValueError, "Prior Registry Attempts"):
            a.save_attempt_outputs(self.root, self.reg, second,
                                   attempt_no=2, runtime={})

    def test_partial_execution_index_write_fails_closed_without_success(self):
        bundle = bundle_fixture(self.task, self.parent)
        self.reg.append_attempt(bundle, attempt_no=1)
        def partial_then_fail(path, encoded):
            with path.open("ab") as stream:
                stream.write(encoded[:max(1, len(encoded)//2)])
                stream.flush()
            raise OSError("synthetic partial index write")

        with patch.object(a, "_append_execution_record", side_effect=partial_then_fail):
            with self.assertRaisesRegex(OSError, "partial index write"):
                a.save_attempt_outputs(self.root, self.reg, bundle,
                                       attempt_no=1, runtime={})
        self.assertTrue((self.root / "execution_records.jsonl").exists())
        # The failed append did not update the in-memory index. A retry detects
        # the changed file and rejects the malformed tail instead of succeeding.
        with self.assertRaises(ValueError):
            a.save_attempt_outputs(self.root, self.reg, bundle,
                                   attempt_no=1, runtime={})

    def test_final_full_validation_detects_tampering_after_fast_appends(self):
        for number, states in ((1, ("FAILED",)*3), (2, ("OK",)*3)):
            bundle = bundle_fixture(self.task, self.parent, states=states)
            self.reg.append_attempt(bundle, attempt_no=number,
                authorization="synthetic retry" if number == 2 else None)
            a.save_attempt_outputs(self.root, self.reg, bundle,
                                   attempt_no=number, runtime={})
        first = json.loads((self.root / "execution_records.jsonl").read_bytes().splitlines()[0])
        descriptor = next(d for d in first["outputs"] if d["path"].startswith("pair_estimates/"))
        target = self.root / descriptor["path"]
        target.write_bytes(target.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "Output bytes/hash mismatch"):
            a.read_execution_records(self.root, self.reg, require_complete=True)

    def test_duplicate_outputs_never_overwrite_or_append_duplicate_record(self):
        bundle=bundle_fixture(self.task,self.parent);self.reg.append_attempt(bundle,attempt_no=1)
        a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        before={f.relative_to(self.root).as_posix():f.read_bytes() for f in self.root.rglob("*") if f.is_file()}
        with self.assertRaises(ValueError):a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        self.assertEqual(before,{f.relative_to(self.root).as_posix():f.read_bytes() for f in self.root.rglob("*") if f.is_file()})
    def test_missing_attempt_and_wrong_execution_index_identity_rejected(self):
        bundle=bundle_fixture(self.task,self.parent)
        with self.assertRaises(ValueError):a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        self.reg.append_attempt(bundle,attempt_no=1)
        a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        path=self.root/"execution_records.jsonl"
        record=json.loads(path.read_bytes())
        record["attempt_identity"]["attempt_sha256"]="f"*64
        path.write_bytes(a.json_bytes(record))
        with self.assertRaises(ValueError):a.read_execution_records(self.root,self.reg)
    def test_corrupt_npz_receipt_and_path_traversal_rejected(self):
        desc=a.save_arrays(self.root,"predictions/corrupt",{"x":np.array([1.])},{})
        (self.root/desc[0]["path"]).write_bytes(b"bad")
        with self.assertRaises(ValueError):a.load_arrays(self.root,desc)
        with self.assertRaises(ValueError):a.save_arrays(self.root,"predictions/../../bad",{"x":np.zeros(1)},{})


    def test_warm_artifact_writer_uses_registry_index_without_history_copy(self):
        first=bundle_fixture(self.task,self.parent,states=("FAILED",)*3)
        self.reg.append_attempt(first,attempt_no=1)
        a.save_attempt_outputs(self.root,self.reg,first,attempt_no=1,runtime={})
        second=bundle_fixture(self.task,self.parent)
        self.reg.append_attempt(second,attempt_no=2,authorization="explicit fixture retry")
        with patch.object(ledger.TaskRegistry,"attempts",property(
                lambda self:(_ for _ in ()).throw(AssertionError("warm writer copied history")))):
            record=a.save_attempt_outputs(self.root,self.reg,second,attempt_no=2,runtime={})
        self.assertEqual(record["attempt_no"],2)
        self.assertEqual(len(a.read_execution_records(self.root,self.reg,require_complete=True)),2)

    def test_restored_registry_cold_output_index_rebuild_preserves_history(self):
        first=bundle_fixture(self.task,self.parent,states=("FAILED",)*3)
        self.reg.append_attempt(first,attempt_no=1)
        a.save_attempt_outputs(self.root,self.reg,first,attempt_no=1,runtime={})
        receipt=ledger_io.write_snapshot(self.reg,self.root/"saved_registry")
        restored=ledger_io.read_snapshot(self.root/"saved_registry",expected_manifest_sha256=receipt.manifest_sha256)
        second=bundle_fixture(self.task,self.parent)
        restored.append_attempt(second,attempt_no=2,authorization="explicit restored retry")
        with patch.object(a,"read_execution_records",wraps=a.read_execution_records) as audit:
            a.save_attempt_outputs(self.root,restored,second,attempt_no=2,runtime={})
            self.assertEqual(audit.call_count,1)  # Cold boundary is still a full audit.
        rows=a.read_execution_records(self.root,restored,require_complete=True)
        self.assertEqual(len(rows),2)
        self.assertEqual([v.document["execution_status"] for v in restored.attempts],["FAILED","OK"])

    def test_final_full_hash_check_detects_missing_corrupt_and_truncated_history_files(self):
        for mutation in ("missing","same_size_corruption","truncated_index"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);reg,task,parent,_=registry_fixture()
                bundle=bundle_fixture(task,parent)
                reg.append_attempt(bundle,attempt_no=1)
                row=a.save_attempt_outputs(root,reg,bundle,attempt_no=1,runtime={})
                descriptor=next(d for d in row["outputs"] if d["path"].startswith("pair_estimates/") and d["path"].endswith(".npz"))
                path=root/descriptor["path"]
                if mutation=="missing":path.unlink()
                elif mutation=="same_size_corruption":
                    raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
                else:
                    index=root/"execution_records.jsonl";index.write_bytes(index.read_bytes()[:-12])
                with self.assertRaises((ValueError,OSError)):
                    a.read_execution_records(root,reg,require_complete=True)

    def test_bad_registry_index_is_rejected_before_artifact_write(self):
        bundle=bundle_fixture(self.task,self.parent);self.reg.append_attempt(bundle,attempt_no=1)
        self.reg._last_attempt_no.clear()
        with self.assertRaises(ValueError):
            a.save_attempt_outputs(self.root,self.reg,bundle,attempt_no=1,runtime={})
        self.assertFalse(any(self.root.iterdir()))


if __name__=="__main__":unittest.main()
