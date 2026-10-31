"""Independent Formal acceptance rejection tests, synthetic artifacts only."""
from pathlib import Path
import copy
import json
import unittest
from unittest.mock import patch
from star_v2 import test_acceptance as a, run_test as r, test_identification as ident
from star_v2 import data_artifacts as data, artifacts
import test_run_test as fixture

def reanchor(f):
    # Explicitly simulate a caller holding new receipts after a controlled
    # mutation. Structural checks must still reject semantically invalid runs.
    inventory=data.json_bytes(r.file_inventory(f.out))
    (f.out/"test_file_inventory.json").write_bytes(inventory)
    c=json.loads((f.out/"test_completion.json").read_text())
    c["inventory_sha256"]=ident.sha(inventory)
    c["ledger_completion_sha256"]=ident.sha((f.out/"ledger_completion.json").read_bytes())
    c["link_sha256"]=ident.sha((f.out/"identification_run_link.json").read_bytes())
    blob=data.json_bytes(c);(f.out/"test_completion.json").write_bytes(blob)
    f.result["completion_sha256"]=ident.sha(blob)
    receipt_path=Path(f.result["execution_receipt_path"])
    receipt=json.loads(receipt_path.read_text())
    receipt["completion_sha256"]=f.result["completion_sha256"]
    receipt_path.write_bytes(data.json_bytes(receipt))
    f.result["execution_receipt_sha256"]=ident.sha(receipt_path.read_bytes())

class FormalAcceptanceTests(unittest.TestCase):
    def test_complete_mock_independent_validator_accepts_with_failed_attempts(self):
        f=fixture.make_mock_run(self);result=fixture.validate_mock(f)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["cohort_code"],2)
        self.assertEqual(result["failed_attempts"],16)
        self.assertIn("engineering",result["scope"])
        self.assertEqual((result["tasks"],result["requests"],result["attempts"]),(3,17,17))

    def test_missing_corrupt_and_tampered_output_rejected(self):
        for change in ("missing","corrupt","tamper"):
            with self.subTest(change=change):
                f=fixture.make_mock_run(self)
                path=next((f.out/"pair_estimates").rglob("*.npz"))
                if change=="missing":
                    path.unlink()
                else:
                    path.write_bytes(b"broken" if change=="corrupt" else path.read_bytes()+b"tampered")
                with self.assertRaises((ValueError,FileNotFoundError)):
                    fixture.validate_mock(f)
                # Even reanchoring inventory cannot bypass per-Attempt hashes.
                reanchor(f)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    fixture.validate_mock(f)

    def test_extra_file_and_extra_ledger_rejected(self):
        f=fixture.make_mock_run(self)
        (f.out/"unexpected.txt").write_text("unexpected")
        with self.assertRaises(ValueError):
            fixture.validate_mock(f)
        f2=fixture.make_mock_run(self);(f2.out/"ledger_after_1").mkdir()
        reanchor(f2)
        with self.assertRaisesRegex(ValueError,"initial/final"):
            fixture.validate_mock(f2)

    def test_complete_receipt_or_ledger_marked_incomplete_rejected(self):
        for target in ("test_completion.json","ledger_completion.json"):
            with self.subTest(target=target):
                f=fixture.make_mock_run(self);p=f.out/target
                doc=json.loads(p.read_text());doc["status"]="INCOMPLETE";p.write_bytes(data.json_bytes(doc))
                if target=="test_completion.json":
                    f.result["completion_sha256"]=ident.sha(p.read_bytes())
                else:
                    reanchor(f)
                with self.assertRaises(ValueError):
                    fixture.validate_mock(f)

    def test_run_incomplete_marker_rejected_even_with_complete_receipt(self):
        f=fixture.make_mock_run(self)
        (f.out/"RUN_INCOMPLETE.json").write_bytes(data.json_bytes({"status":"INCOMPLETE"}))
        reanchor(f)
        with self.assertRaisesRegex(ValueError,"INCOMPLETE"):
            fixture.validate_mock(f)

    def test_hard_interrupt_missing_final_receipt_rejected(self):
        f=fixture.make_mock_run(self)
        (f.out/"test_completion.json").unlink()
        with self.assertRaises(FileNotFoundError):
            fixture.validate_mock(f)

    def test_pilot_run_manifest_cannot_masquerade_as_test(self):
        f=fixture.make_mock_run(self)
        p=f.out/"run_manifest.json";run=json.loads(p.read_text());run["run_kind"]="pilot"
        p.write_bytes(data.json_bytes(run))
        with self.assertRaisesRegex(ValueError,"Formal Test"):
            with patch.object(r,"test_plan",return_value=f.plan):
                a.validate_test_ledgers(f.out,run_manifest_sha256=ident.sha(p.read_bytes()),run_id="mock-test")

    def test_missing_extra_requests_and_duplicate_registration_rejected(self):
        from test_kernel_new_adapter import tasks_fixture
        with patch.object(r,"test_plan",return_value=fixture.tiny_plan()):
            reg,records=r.build_test_registry((tasks_fixture()[0],),run_id="mock-test",decision=fixture.decision())
            key=next(iter(reg.requests));record=reg._requests.pop(key)
            with self.assertRaisesRegex(ValueError,"Missing/extra"):
                a.validate_universe(reg,run_id="mock-test")
            reg._requests[key]=record
            reg._requests[("state-mean","pair","extra","extra")]=record
            with self.assertRaisesRegex(ValueError,"Missing/extra"):
                a.validate_universe(reg,run_id="mock-test")
            del reg._requests[("state-mean","pair","extra","extra")]
            reg._sealed=False
            with self.assertRaisesRegex(ValueError,"DUPLICATE_REQUEST"):
                reg.register_request(records[0].method_id,records[0].parent)

    def test_ledger_seal_and_attempt_count_mismatch_rejected(self):
        for field,value in (("registry_seal_sha256","a"*64),("recorded_attempts",16),
                            ("expected_requests",18),("unexecuted_request_keys",[["missing"]])):
            with self.subTest(field=field):
                f=fixture.make_mock_run(self)
                p=f.out/"ledger_completion.json";done=json.loads(p.read_text());done[field]=value
                p.write_bytes(data.json_bytes(done));reanchor(f)
                with self.assertRaises(ValueError):
                    fixture.validate_mock(f)

    def test_child_parent_linkage_tamper_rejected(self):
        from test_kernel_new_adapter import tasks_fixture
        from star_v2.task_registry import ExecutionRequest
        from dataclasses import replace
        with patch.object(r,"test_plan",return_value=fixture.tiny_plan()):
            reg,records=r.build_test_registry((tasks_fixture()[0],),run_id="mock-test",decision=fixture.decision())
            record=next(v for v in records if v.children)
            child=record.children[0]
            wrong=replace(child,parent_request_id="wrong-parent")
            key=(record.method_id,"context",child.task_id,child.request_id)
            reg._requests[key]=ExecutionRequest.from_request(record.method_id,wrong)
            with self.assertRaisesRegex(ValueError,"binding/provenance"):
                a.validate_universe(reg,run_id="mock-test")

    def test_identification_approval_tamper_and_linkage_rejected(self):
        for name in ("identification_decision.json","identification_approval.json","identification_run_link.json"):
            with self.subTest(name=name):
                f=fixture.make_mock_run(self)
                p=f.out/name;value=json.loads(p.read_text());value["cohort_code"]=1
                p.write_bytes(data.json_bytes(value));reanchor(f)
                with self.assertRaises(ValueError):
                    fixture.validate_mock(f)

    def test_snapshot_and_gate_binding_mismatch_rejected(self):
        f=fixture.make_mock_run(self)
        p=f.out/"real_model_gate.json";gate=json.loads(p.read_text())
        gate["source_snapshot_id"]=ident.PARENT_SNAPSHOT_ID;p.write_bytes(data.json_bytes(gate))
        link_path=f.out/"identification_run_link.json";link=json.loads(link_path.read_text())
        link["gate_sha256"]=ident.sha(p.read_bytes());link_path.write_bytes(data.json_bytes(link));reanchor(f)
        with self.assertRaisesRegex(ValueError,"protocol/source"):
            fixture.validate_mock(f)
        f2=fixture.make_mock_run(self)
        (f2.delivery/"source_snapshot.zip").write_bytes(b"corrupt archive")
        with self.assertRaises(ValueError):
            fixture.validate_mock(f2)

    def test_final_ledger_hash_and_all_attempt_outputs_protected(self):
        f=fixture.make_mock_run(self)
        initial=(f.out/"ledger_initial/ledger_manifest.json").read_bytes()
        final=(f.out/"ledger_final/ledger_manifest.json").read_bytes()
        self.assertNotEqual(ident.sha(initial),ident.sha(final))
        self.assertEqual(len((f.out/"execution_records.jsonl").read_text().splitlines()),17)
        p=f.out/"ledger_final/attempts.jsonl"
        p.write_bytes(p.read_bytes()+b"{}\n");reanchor(f)
        with self.assertRaises(ValueError):
            fixture.validate_mock(f)

    def test_candidate_identity_cannot_be_reinterpreted_by_new_snapshot(self):
        f=fixture.make_mock_run(self)
        source=json.loads(data.json_bytes(f.source))
        for record in source["files"]:
            if record["path"].endswith("FROZEN.md"):
                record["sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"Candidate"):
            r.verify_lineage(f.delivery,source)


class AcceptanceOutputIsolationTests(unittest.TestCase):
    def tree_bytes(self,root):
        return {p.relative_to(root).as_posix():p.read_bytes() if p.is_file() else None
                for p in root.rglob("*")}

    def placed_run(self,kind):
        import shutil
        f=fixture.make_mock_run(self)
        if kind=="runtime":
            destination=Path(f.report["runtime_root"])/"placed_run"
        elif kind=="delivery":
            destination=f.delivery/"placed_run"
        else:
            parent=f.base/("git-directory" if kind=="git-dir" else "git-pointer")
            parent.mkdir()
            if kind=="git-dir":
                (parent/".git").mkdir()
            else:
                (parent/".git").write_bytes(b"gitdir: nonexistent-outside-metadata\n")
            destination=parent/"placed_run"
        shutil.copytree(f.out,destination)
        f.out=destination
        return f

    def reject_placed(self,kind,message):
        f=self.placed_run(kind)
        before=self.tree_bytes(f.base)
        with self.assertRaisesRegex(ValueError,message):
            fixture.validate_mock(f)
        self.assertEqual(before,self.tree_bytes(f.base))

    def test_acceptance_rejects_run_inside_gate_verified_runtime(self):
        # Gate Runtime differs from validator module ROOT. This verifies the
        # second guard uses the actual Gate-pinned Runtime, not merely ROOT.
        self.reject_placed("runtime","outside Runtime")

    def test_acceptance_rejects_run_inside_source_delivery(self):
        self.reject_placed("delivery","outside Source Delivery")

    def test_acceptance_rejects_git_directory_ancestor(self):
        self.reject_placed("git-dir","outside Git worktrees")

    def test_acceptance_rejects_git_pointer_file_ancestor(self):
        self.reject_placed("git-file","outside Git worktrees")

    def test_acceptance_accepts_valid_independent_desktop_run_read_only(self):
        import shutil
        f=fixture.make_mock_run(self)
        desktop=f.base/"Desktop";desktop.mkdir()
        destination=desktop/"independent_run"
        shutil.copytree(f.out,destination);f.out=destination
        before=self.tree_bytes(f.base)
        result=fixture.validate_mock(f)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual((result["tasks"],result["requests"],result["attempts"]),(3,17,17))
        self.assertEqual(result["failed_attempts"],16)
        self.assertEqual(before,self.tree_bytes(f.base))

if __name__=="__main__":
    unittest.main()
