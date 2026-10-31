"""Small ledger fixtures only: no generator, prediction, RNG stream or assets."""
import unittest
import tempfile,hashlib,json,threading
from pathlib import Path
from unittest.mock import patch
from star_v2 import artifacts
from dataclasses import replace

import numpy as np

from star_v2 import task_registry as ledger
from star_v2.estimation import (EstimateBundle, PairEstimateBundle, IntervalRecord,
                               SamplingUncertainty, EFFECT_ORDER)
from star_v2.model_adapters import context_requests_from_parent
from test_kernel_new_adapter import tasks_fixture


def registry_fixture(*,status="POINT",outputs=("ate",),seal=True):
    task,parent,_ = tasks_fixture(status=status)
    parent = replace(parent,request_id="shared",requested_outputs=outputs)
    children = context_requests_from_parent(task,parent,request_ids=("shared","shared"))
    registry = ledger.TaskRegistry()
    for item in (task,)+task.context_refs: registry.register_task(item)
    records = [registry.register_request("state-mean",parent),
               registry.register_request("j",parent)]
    records += [registry.register_request("j",r) for r in children]
    if seal:
        registry.seal(expected_task_ids=(task.pair_id,)+task.context_ids,
                      expected_request_keys=tuple(r.key for r in records))
    return registry,task,parent,children


def bundle_fixture(task,request,method="state-mean",states=("OK","OK","OK")):
    if method == "j" and request.scope == "pair":
        children = context_requests_from_parent(task,request,request_ids=("shared","shared"))
        arms = tuple(bundle_fixture(t,r,"j",states) for t,r in zip(task.context_refs,children))
    else:
        arms = None
    values = np.array([1.,2.,3.])
    for k,s in enumerate(states):
        if s != "OK": values[k] = np.nan
    unavailable = request.status == "NOT_IDENTIFIED" or states != ("OK",)*3
    kwargs = dict(ate=values,output_status={"ate":states},
        failure_type=request.failure_type if request.status == "NOT_IDENTIFIED" else "SCRIPTED" if unavailable else None,
        failure_message=request.failure_message if request.status == "NOT_IDENTIFIED" else "exact synthetic reason" if unavailable else None,
        failure_provenance=request.failure_provenance if request.status == "NOT_IDENTIFIED"
            else {"stage":"synthetic_fixture","attempts":1} if unavailable else {},
        diagnostics={"synthetic":True,"rows":50},estimability={"Direct":{"status":states[0]}})
    if method == "j" and request.scope == "context" and "interval" in request.requested_outputs:
        kwargs["intervals"] = tuple(IntervalRecord(request.context_ids[0],e,"OK",0.,4.,
                                   source="model_posterior") for e in EFFECT_ORDER)
        kwargs["output_status"]["interval"] = ("OK",)*3
        kwargs["sampling_uncertainty"] = SamplingUncertainty("OK","model_posterior","context",None)
    cls = PairEstimateBundle if request.scope == "pair" else EstimateBundle
    if cls is PairEstimateBundle: kwargs["arm_results"] = arms if arms is not None else (None,None)
    return cls.from_outputs(request,task,method,**kwargs)


class TaskRegistryTests(unittest.TestCase):
    def test_approved_three_layer_keys_preserve_existing_raw_ids(self):
        reg,task,parent,children = registry_fixture()
        self.assertEqual(len(reg.tasks),3)
        self.assertEqual(len(reg.requests),4)
        expected = {
            ("state-mean","pair",task.pair_id,"shared"),
            ("j","pair",task.pair_id,"shared"),
            ("j","context",task.context_ids[0],"shared"),
            ("j","context",task.context_ids[1],"shared")}
        self.assertEqual(set(reg.requests),expected)
        self.assertEqual(parent.task_id,task.pair_id)
        self.assertEqual(tuple(r.task_id for r in children),task.context_ids)
        self.assertEqual({r.document["request"]["request_id"] for r in reg.requests.values()},{"shared"})

    def test_duplicate_data_task_and_conflicting_content_rejected(self):
        reg,task,_,_ = registry_fixture(seal=False)
        with self.assertRaisesRegex(ledger.LedgerError,"DUPLICATE_TASK"):
            reg.register_task(task)
        doc = reg.tasks[task.pair_id].document
        doc["input_hashes"][0] = "a"*64
        with self.assertRaisesRegex(ledger.LedgerError,"TASK_CONFLICT"):
            reg.register_task(ledger.DataTask.from_document(doc))
        self.assertEqual(reg.tasks[task.pair_id].document["input_hashes"][0],
                         task.context_refs[0].input_hash)

    def test_duplicate_composite_request_and_content_conflict_distinguished(self):
        reg,_,parent,_ = registry_fixture(seal=False)
        with self.assertRaisesRegex(ledger.LedgerError,"DUPLICATE_REQUEST"):
            reg.register_request("state-mean",parent)
        changed = replace(parent,requested_outputs=("ate","interval"))
        with self.assertRaisesRegex(ledger.LedgerError,"REQUEST_CONFLICT"):
            reg.register_request("state-mean",changed)

    def test_request_wrong_pair_context_or_input_binding_rejected(self):
        reg,_,parent,children = registry_fixture(seal=False)
        for changed in (
            replace(parent,request_id="other",input_hashes=("0"*64,"1"*64)),
            replace(children[0],request_id="other",context_ids=(children[1].task_id,))):
            with self.assertRaises(ledger.LedgerError):
                reg.register_request("j",changed)

    def test_seal_checks_pair_context_hash_associations(self):
        _,task,_,_ = registry_fixture()
        reg = ledger.TaskRegistry()
        doc = ledger.DataTask.from_task(task).document
        doc["input_hashes"][0] = "a"*64
        reg.register_task(ledger.DataTask.from_document(doc))
        for context in task.context_refs: reg.register_task(context)
        with self.assertRaisesRegex(ledger.LedgerError,"Pair/context link"):
            reg.seal(expected_task_ids=tuple(reg.tasks),expected_request_keys=())

    def test_parent_request_link_is_checked_at_seal(self):
        _,task,parent,children = registry_fixture()
        reg = ledger.TaskRegistry()
        for t in (task,)+task.context_refs: reg.register_task(t)
        reg.register_request("j",parent)
        reg.register_request("j",replace(children[0],parent_request_id="not-registered"))
        with self.assertRaisesRegex(ledger.LedgerError,"Parent execution"):
            reg.seal(expected_task_ids=tuple(reg.tasks),expected_request_keys=tuple(reg.requests))

    def test_inventory_reports_missing_and_extra_independently(self):
        reg,task,_,_ = registry_fixture(seal=False)
        one = next(iter(reg.requests))
        expected_keys = tuple(k for k in reg.requests if k != one)+(("ht","pair",task.pair_id,"missing"),)
        check = reg.inventory(expected_task_ids=(task.pair_id,"missing-data"),
                              expected_request_keys=expected_keys)
        self.assertEqual(check["missing_tasks"],("missing-data",))
        self.assertEqual(set(check["extra_tasks"]),set(task.context_ids))
        self.assertEqual(check["extra_requests"],(one,))
        self.assertEqual(check["missing_requests"],(("ht","pair",task.pair_id,"missing"),))
        with self.assertRaisesRegex(ledger.LedgerError,"Incomplete inventory"):
            reg.seal(expected_task_ids=(task.pair_id,),expected_request_keys=expected_keys)
        self.assertFalse(reg.sealed)

    def test_sealed_inventory_cannot_add_delete_or_replace(self):
        reg,task,parent,_ = registry_fixture()
        with self.assertRaises(ledger.LedgerError): reg.register_task(task)
        with self.assertRaises(ledger.LedgerError): reg.register_request("ht",parent)
        with self.assertRaises(TypeError): del reg.tasks[task.pair_id]
        with self.assertRaises(TypeError): reg.requests[next(iter(reg.requests))] = None
        snapshot = reg.tasks[task.pair_id].document
        snapshot["root_seed"] = 0
        self.assertEqual(reg.tasks[task.pair_id].document["root_seed"],20261030)

    def test_unexecuted_is_not_success_and_recording_does_not_retry(self):
        reg,task,parent,_ = registry_fixture()
        before = tuple(reg.requests)
        self.assertEqual(set(reg.unexecuted_requests()),set(before))
        bundle = bundle_fixture(task,parent,states=("FAILED","OK","FAILED"))
        attempt = reg.append_attempt(bundle,attempt_no=1)
        self.assertEqual(len(reg.attempts),1)
        self.assertEqual(len(reg.unexecuted_requests()),3)
        self.assertEqual(attempt.key,(("state-mean","pair",task.pair_id,"shared"),1))
        self.assertEqual(attempt.document["failure_message"],"exact synthetic reason")
        self.assertEqual(tuple(reg.requests),before)

    def test_failure_then_explicit_success_append_without_overwrite(self):
        reg,task,parent,_ = registry_fixture()
        failed = reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        success = bundle_fixture(task,parent)
        with self.assertRaisesRegex(ledger.LedgerError,"authorization"):
            reg.append_attempt(success,attempt_no=2)
        self.assertEqual(reg.attempts,(failed,))
        second = reg.append_attempt(success,attempt_no=2,authorization="user-approved:synthetic-retry")
        self.assertEqual(reg.attempts[0],failed)
        self.assertEqual([a.document["execution_status"] for a in reg.attempts],["FAILED","OK"])
        self.assertEqual(second.document["authorization"],"user-approved:synthetic-retry")
        self.assertEqual(first_request(failed),first_request(second))

    def test_duplicate_gapped_and_boolean_attempt_numbers_rejected(self):
        reg,task,parent,_ = registry_fixture()
        b = bundle_fixture(task,parent)
        with self.assertRaises(ledger.LedgerError): reg.append_attempt(b,attempt_no=True)
        with self.assertRaises(ledger.LedgerError): reg.append_attempt(b,attempt_no=2,authorization="approved")
        reg.append_attempt(b,attempt_no=1)
        with self.assertRaises(ledger.LedgerError): reg.append_attempt(b,attempt_no=1)
        with self.assertRaises(ledger.LedgerError): reg.append_attempt(b,attempt_no=3,authorization="approved")
        self.assertEqual(len(reg.attempts),1)

    def test_attempt_seed_rng_and_request_semantics_cannot_be_replaced(self):
        reg,task,parent,_ = registry_fixture()
        attempt = ledger.Attempt.from_bundle(bundle_fixture(task,parent),attempt_no=1)
        for mutate in ("rng","outputs","input_hashes"):
            doc = attempt.document
            request = doc["execution_request"]["request"]
            if mutate == "rng": request["identity"]["rng_identity"] += " "
            elif mutate == "outputs": request["requested_outputs"].append("interval")
            else: request["input_hashes"][0] = "a"*64
            # Each record is locally syntactic; registry rejects the frozen identity conflict.
            if mutate == "outputs":
                # interval becomes unsupported for this method, update complete evidence accordingly.
                doc["output_status"]["interval"] = ["UNSUPPORTED"]*3
                doc["sampling_uncertainty"]["status"] = "UNSUPPORTED"
            with self.subTest(mutate=mutate),self.assertRaises(ledger.LedgerError):
                reg.record_attempt(ledger.Attempt.from_document(doc))
        self.assertEqual(reg.attempts,())

    def test_execution_states_and_evaluation_qualification_are_distinct(self):
        for states,expected in ((("OK",)*3,"OK"),(("NOT_ESTIMABLE","OK","NOT_ESTIMABLE"),"NOT_ESTIMABLE"),
                                (("FAILED","OK","OK"),"FAILED")):
            reg,task,parent,_ = registry_fixture()
            attempt = reg.append_attempt(bundle_fixture(task,parent,states=states),attempt_no=1)
            self.assertEqual(attempt.document["execution_status"],expected)
            doc = attempt.document;doc["execution_status"] = "NOT_RANKABLE"
            with self.assertRaises(ledger.LedgerError): ledger.Attempt.from_document(doc)
        reg,task,parent,_ = registry_fixture(status="NOT_IDENTIFIED")
        attempt = reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        self.assertEqual(attempt.document["status"],"NOT_IDENTIFIED")
        self.assertEqual(attempt.document["execution_status"],"NOT_IDENTIFIED")
        self.assertEqual(attempt.document["failure_message"],"exact upstream reason")

    def test_unsupported_not_requested_intervals_and_source_metadata_preserved(self):
        reg,task,parent,children = registry_fixture(outputs=("ate","interval"))
        pair_bundle = bundle_fixture(task,parent,"j")
        recorded = reg.append_attempt(pair_bundle,attempt_no=1)
        self.assertEqual(recorded.document["output_status"]["interval"],["UNSUPPORTED"]*3)
        self.assertEqual(recorded.document["output_status"]["mu"],"NOT_REQUESTED")
        child_bundle = bundle_fixture(task.context_refs[0],children[0],"j")
        child = reg.append_attempt(child_bundle,attempt_no=1)
        self.assertEqual(child.document["sampling_uncertainty"],{
            "status":"OK","source":"model_posterior","statistical_unit":"context",
            "effective_sample_count":None})
        self.assertEqual(len(child.document["intervals"]),3)
        reg2,task2,parent2,_ = registry_fixture()
        sm = reg2.append_attempt(bundle_fixture(task2,parent2),attempt_no=1)
        self.assertEqual(sm.document["output_status"]["interval"],["NOT_REQUESTED"]*3)
        self.assertEqual(sm.document["output_status"]["mu"],"UNSUPPORTED")

    def test_data_identity_changes_and_invalid_effect_order_rejected(self):
        reg,task,parent,_ = registry_fixture()
        for field,value in (("root_seed",1),("protocol_code",1),("task_id","replacement"),
                            ("pair_id","wrong")):
            doc = reg.tasks[task.pair_id].document;doc[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                ledger.DataTask.from_document(doc)
        doc = ledger.ExecutionRequest.from_request("state-mean",parent).document
        doc["effects"] = ["Total","Direct","Spillover"]
        with self.assertRaises(ledger.LedgerError): ledger.ExecutionRequest.from_document(doc)



    def test_index_empty_rebuild_and_snapshot_restore_preserve_serialized_bytes(self):
        empty=ledger.TaskRegistry()
        self.assertEqual(empty.attempt_count,0)
        self.assertIsNone(empty.latest_attempt)
        self.assertEqual(empty.rebuild_attempt_indexes(),0)
        reg,task,parent,_=registry_fixture()
        key=("state-mean","pair",task.pair_id,parent.request_id)
        self.assertEqual(reg.next_attempt_no(key),1)
        failure=reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        success=reg.append_attempt(bundle_fixture(task,parent),attempt_no=2,authorization="explicit index fixture retry")
        contents=[v.content for v in reg.attempts];seal=reg.seal_sha256
        reg._last_attempt_no.clear();reg._attempt_by_key.clear()
        with self.assertRaises(ledger.LedgerError):reg.next_attempt_no(key)
        self.assertEqual(reg.rebuild_attempt_indexes(),2)
        self.assertIs(reg.get_attempt(key,1),failure)
        self.assertIs(reg.get_attempt(key,2),success)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)
            before=artifacts.write_snapshot(reg,target/"before")
            restored=artifacts.read_snapshot(target/"before",expected_manifest_sha256=before.manifest_sha256)
            self.assertEqual(restored.last_attempt_no(key),2)
            self.assertEqual(restored.next_attempt_no(key),3)
            self.assertEqual(restored.validate_attempt_indexes(),2)
            self.assertEqual([v.content for v in restored.attempts],contents)
            after=artifacts.write_snapshot(restored,target/"after")
            self.assertEqual(before.manifest_sha256,after.manifest_sha256)
            for name in artifacts.FILES+(artifacts.MANIFEST,):
                self.assertEqual((target/"before"/name).read_bytes(),(target/"after"/name).read_bytes())
            self.assertEqual(restored.seal_sha256,seal)

    def test_index_invalid_attempt_never_updates_committed_state(self):
        reg,task,parent,_=registry_fixture();key=("state-mean","pair",task.pair_id,parent.request_id)
        first=reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        before=(dict(reg._last_attempt_no),dict(reg._attempt_by_key),reg._history_digest,reg.attempt_count)
        for number,auth in ((1,None),(3,"explicit"),(2,None)):
            with self.subTest(number=number),self.assertRaises(ValueError):
                reg.append_attempt(bundle_fixture(task,parent),attempt_no=number,authorization=auth)
            self.assertEqual((dict(reg._last_attempt_no),dict(reg._attempt_by_key),reg._history_digest,reg.attempt_count),before)
        second=reg.append_attempt(bundle_fixture(task,parent),attempt_no=2,authorization="explicit retry")
        self.assertEqual(reg.last_attempt_no(key),2)
        self.assertIs(reg.get_attempt(key,1),first);self.assertIs(reg.get_attempt(key,2),second)
        self.assertIsNone(reg.get_attempt(key,3))

    def test_index_missing_stale_duplicate_or_wrong_entry_fails_closed(self):
        for kind in ("missing_last","zero_last","stale_last","missing_entry","wrong_position","wrong_key","history_append"):
            reg,task,parent,_=registry_fixture();key=("state-mean","pair",task.pair_id,parent.request_id)
            reg.append_attempt(bundle_fixture(task,parent),attempt_no=1)
            reg.append_attempt(bundle_fixture(task,parent),attempt_no=2,authorization="explicit retry")
            if kind=="missing_last":reg._last_attempt_no.pop(key)
            elif kind=="zero_last":reg._last_attempt_no[key]=0
            elif kind=="stale_last":reg._last_attempt_no[key]=1
            elif kind=="missing_entry":reg._attempt_by_key.pop((key,1))
            elif kind=="wrong_position":
                entry=reg._attempt_by_key[(key,2)];reg._attempt_by_key[(key,2)]=(entry[0],0,*entry[2:])
            elif kind=="wrong_key":
                entry=reg._attempt_by_key[(key,2)];reg._attempt_by_key[(key,2)]=(*entry[:3],("wrong",)*4,2)
            else:reg._attempts.append(reg._attempts[-1])
            with self.subTest(kind=kind),self.assertRaises(ledger.LedgerError):reg.next_attempt_no(key)
            with self.assertRaises(ledger.LedgerError):reg.validate_attempt_indexes()

    def test_index_rebuild_cannot_rewrite_authoritative_failure_history(self):
        reg,task,parent,_=registry_fixture()
        reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        # A locally valid replacement is still an unauthorized history rewrite.
        replacement=ledger.Attempt.from_bundle(bundle_fixture(task,parent),attempt_no=1)
        reg._attempts[0]=replacement
        with self.assertRaises(ledger.LedgerError):reg.validate_attempt_indexes()
        with self.assertRaisesRegex(ledger.LedgerError,"history changed"):reg.rebuild_attempt_indexes()

    def test_single_writer_process_thread_and_reentrancy_are_enforced(self):
        reg,task,parent,_=registry_fixture();key=("state-mean","pair",task.pair_id,parent.request_id)
        errors=[]
        def foreign_writer():
            try:reg.append_attempt(bundle_fixture(task,parent),attempt_no=1)
            except ledger.LedgerError as error:errors.append(str(error))
        thread=threading.Thread(target=foreign_writer);thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertIn("single owning",errors[0])
        with patch.object(ledger.os,"getpid",return_value=-1),self.assertRaises(ledger.LedgerError):
            reg.next_attempt_no(key)
        reg._recording=True
        with self.assertRaisesRegex(ledger.LedgerError,"already in progress"):
            reg.append_attempt(bundle_fixture(task,parent),attempt_no=1)
        reg._recording=False
        self.assertEqual(reg.attempt_count,0)

    def test_hot_operations_do_not_iterate_or_copy_history_at_any_fixture_scale(self):
        class NoScan(list):
            def __iter__(self):raise AssertionError("hot path scanned history")
        for count in (1,25,100):
            reg,task,parent,_=registry_fixture();key=("state-mean","pair",task.pair_id,parent.request_id)
            bundle=bundle_fixture(task,parent)
            for number in range(1,count+1):
                reg.append_attempt(bundle,attempt_no=number,authorization="explicit fixture" if number>1 else None)
            incoming=ledger.Attempt.from_bundle(bundle,attempt_no=count+1,authorization="explicit fixture")
            reg._attempts=NoScan(reg._attempts)
            original=ledger.Attempt.execution_key.fget;reads=[]
            def counted(attempt):
                reads.append(attempt)
                return original(attempt)
            with patch.object(ledger.Attempt,"execution_key",property(counted)),patch.object(
                    ledger.TaskRegistry,"attempts",property(lambda self:(_ for _ in ()).throw(AssertionError("history copied")))):
                self.assertEqual(reg.last_attempt_no(key),count)
                self.assertEqual(reg.next_attempt_no(key),count+1)
                self.assertIsNotNone(reg.get_attempt(key,1))
                reg.record_attempt(incoming)
                self.assertEqual(reg.attempt_count,count+1)
            self.assertEqual(reads,[incoming])  # Exactly one new key; zero historical keys.


def first_request(attempt):
    return attempt.document["execution_request"]


if __name__ == "__main__":
    unittest.main()
