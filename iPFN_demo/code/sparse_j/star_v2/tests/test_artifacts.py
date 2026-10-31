"""Snapshot acceptance; all output lives in TemporaryDirectory only."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from star_v2 import artifacts
from test_task_registry import registry_fixture, bundle_fixture


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def independent_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory(prefix="star-v2-ledger-test-"))
        self.root = Path(self.temp)
        self.reg,self.task,self.parent,self.children = registry_fixture()
        self.reg.append_attempt(bundle_fixture(self.task,self.parent,states=("FAILED",)*3),attempt_no=1)
        self.reg.append_attempt(bundle_fixture(self.task,self.parent),attempt_no=2,
                                authorization="explicit-user-approval:fixture")

    def save(self):
        target = self.root/"snapshot"
        receipt = artifacts.write_snapshot(self.reg,target)
        return target,receipt

    def read(self,target,receipt):
        return artifacts.read_snapshot(target,expected_manifest_sha256=receipt.manifest_sha256)

    def reanchor(self,target,name,rows):
        # Test-only independent writer for deliberately malformed trusted input.
        raw = ("".join(independent_json(row)+"\n" for row in rows)).encode("ascii")
        (target/name).write_bytes(raw)
        manifest = json.loads((target/"ledger_manifest.json").read_text(encoding="ascii"))
        manifest["files"][name] = {"count":len(rows),"sha256":sha(raw)}
        manifest_blob = (independent_json(manifest)+"\n").encode("ascii")
        (target/"ledger_manifest.json").write_bytes(manifest_blob)
        return sha(manifest_blob)

    def test_write_records_match_independent_literal_fields_counts_and_hashes(self):
        target,receipt = self.save()
        manifest_raw = (target/"ledger_manifest.json").read_bytes()
        self.assertEqual(receipt.manifest_sha256,sha(manifest_raw))
        manifest = json.loads(manifest_raw)
        self.assertEqual(manifest["schema_version"],1)
        self.assertEqual(manifest["expected_task_ids"],sorted(
            (self.task.pair_id,)+self.task.context_ids))
        self.assertEqual(manifest["files"]["tasks.jsonl"]["count"],3)
        self.assertEqual(manifest["files"]["requests.jsonl"]["count"],4)
        self.assertEqual(manifest["files"]["attempts.jsonl"]["count"],2)
        attempts = [json.loads(line) for line in (target/"attempts.jsonl").read_text().splitlines()]
        self.assertEqual([row["attempt_no"] for row in attempts],[1,2])
        self.assertEqual([row["execution_status"] for row in attempts],["FAILED","OK"])
        self.assertEqual(attempts[0]["failure_message"],"exact synthetic reason")
        self.assertEqual(attempts[1]["authorization"],"explicit-user-approval:fixture")
        self.assertEqual(attempts[0]["task_id"],self.task.pair_id)
        request = attempts[0]["execution_request"]
        self.assertEqual(request["method_id"],"state-mean")
        self.assertEqual(request["effects"],["Direct","Spillover","Total"])
        self.assertEqual(request["request"]["request_id"],"shared")
        self.assertEqual(request["request"]["input_hashes"],
                         [t.input_hash for t in self.task.context_refs])
        for name in ("tasks.jsonl","requests.jsonl","attempts.jsonl"):
            self.assertEqual(manifest["files"][name]["sha256"],sha((target/name).read_bytes()))
        self.assertNotIn("mu",attempts[0])
        self.assertNotIn("ate",attempts[0])
        self.assertNotIn("outcomes_observed",attempts[0])

    def test_roundtrip_preserves_all_tasks_requests_attempts_and_unexecuted(self):
        target,receipt = self.save()
        loaded = self.read(target,receipt)
        self.assertTrue(loaded.sealed)
        self.assertEqual(loaded.seal_sha256,self.reg.seal_sha256)
        self.assertEqual(dict(loaded.tasks),dict(self.reg.tasks))
        self.assertEqual(dict(loaded.requests),dict(self.reg.requests))
        self.assertEqual(loaded.attempts,self.reg.attempts)
        self.assertEqual(loaded.unexecuted_requests(),self.reg.unexecuted_requests())
        self.assertEqual(len(loaded.unexecuted_requests()),3)

    def test_independent_manual_jsonl_snapshot_can_be_read(self):
        target = self.root/"manual";target.mkdir()
        docs = {
            "tasks.jsonl":[r.document for r in self.reg.tasks.values()],
            "requests.jsonl":[r.document for r in self.reg.requests.values()],
            "attempts.jsonl":[r.document for r in self.reg.attempts]}
        blobs = {name:("".join(independent_json(row)+"\n" for row in rows)).encode("ascii")
                 for name,rows in docs.items()}
        seal_document = {
            "tasks":[self.reg.tasks[k].document for k in sorted(self.reg.tasks)],
            "requests":[self.reg.requests[k].document for k in sorted(self.reg.requests)]}
        independent_seal = sha(independent_json(seal_document).encode("ascii"))
        self.assertEqual(self.reg.seal_sha256,independent_seal)
        manifest = {"schema_version":1,"seal_sha256":independent_seal,
            "expected_task_ids":sorted(self.reg.tasks),
            "expected_request_keys":[list(k) for k in sorted(self.reg.requests)],
            "files":{name:{"count":len(docs[name]),"sha256":sha(blob)} for name,blob in blobs.items()}}
        for name,blob in blobs.items(): (target/name).write_bytes(blob)
        header = (independent_json(manifest)+"\n").encode("ascii")
        (target/"ledger_manifest.json").write_bytes(header)
        loaded = artifacts.read_snapshot(target,expected_manifest_sha256=sha(header))
        self.assertEqual([a.document["execution_status"] for a in loaded.attempts],["FAILED","OK"])
        self.assertEqual(loaded.seal_sha256,independent_seal)

    def test_unexecuted_snapshot_is_not_a_success_subset(self):
        reg,_,_,_ = registry_fixture()
        target = self.root/"unexecuted"
        receipt = artifacts.write_snapshot(reg,target)
        loaded = self.read(target,receipt)
        self.assertEqual(loaded.attempts,())
        self.assertEqual(len(loaded.unexecuted_requests()),4)
        self.assertEqual(len(loaded.requests),4)

    def test_attempt_global_order_and_parent_context_linkage_preserved(self):
        reg,task,parent,children = registry_fixture(outputs=("ate","interval"))
        reg.append_attempt(bundle_fixture(task,parent,states=("FAILED",)*3),attempt_no=1)
        reg.append_attempt(bundle_fixture(task.context_refs[0],children[0],"j"),attempt_no=1)
        reg.append_attempt(bundle_fixture(task,parent),attempt_no=2,authorization="approved")
        target = self.root/"mixed"
        receipt = artifacts.write_snapshot(reg,target)
        loaded = self.read(target,receipt)
        self.assertEqual([a.key for a in loaded.attempts],[a.key for a in reg.attempts])
        child = loaded.attempts[1].document
        self.assertEqual(child["execution_request"]["request"]["parent_request_id"],"shared")
        self.assertEqual(child["intervals"][0]["context_id"],task.context_ids[0])
        self.assertEqual(child["sampling_uncertainty"]["statistical_unit"],"context")

    def test_existing_snapshot_never_overwritten(self):
        target,receipt = self.save()
        before = {p.name:p.read_bytes() for p in target.iterdir()}
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_snapshot(self.reg,target)
        self.assertEqual(before,{p.name:p.read_bytes() for p in target.iterdir()})
        self.assertEqual(self.read(target,receipt).attempts,self.reg.attempts)

    def test_failed_write_is_not_published_and_staging_is_cleaned(self):
        marker = self.root/"unrelated.txt";marker.write_text("keep",encoding="ascii")
        original = artifacts._write
        def fail(path,blob):
            if path.name == "attempts.jsonl": raise OSError("scripted disk failure")
            return original(path,blob)
        with patch.object(artifacts,"_write",side_effect=fail):
            with self.assertRaisesRegex(OSError,"disk failure"):
                artifacts.write_snapshot(self.reg,self.root/"not-published")
        self.assertEqual({p.name for p in self.root.iterdir()},{"unrelated.txt"})
        self.assertEqual(marker.read_text(),"keep")

    def test_duplicate_task_and_request_rows_rejected_even_with_valid_hashes(self):
        target,_ = self.save()
        for name in ("tasks.jsonl","requests.jsonl"):
            original = (target/name).read_bytes()
            rows = [json.loads(line) for line in original.splitlines()]
            digest = self.reanchor(target,name,rows+[rows[0]])
            with self.subTest(file=name),self.assertRaises(artifacts.ArtifactError):
                artifacts.read_snapshot(target,expected_manifest_sha256=digest)
            # Restore the prior valid row set with a fresh independent manifest.
            self.reanchor(target,name,rows)

    def test_missing_task_request_and_extra_request_are_not_silently_skipped(self):
        for mode in ("missing_task","missing_request","extra_request"):
            target = self.root/mode
            artifacts.write_snapshot(self.reg,target)
            name = "tasks.jsonl" if mode == "missing_task" else "requests.jsonl"
            rows = [json.loads(line) for line in (target/name).read_bytes().splitlines()]
            if mode.startswith("missing"): rows = rows[1:]
            else:
                extra = json.loads(json.dumps(rows[0]));extra["method_id"] = "ht";rows.append(extra)
            digest = self.reanchor(target,name,rows)
            with self.subTest(mode=mode),self.assertRaises(artifacts.ArtifactError):
                artifacts.read_snapshot(target,expected_manifest_sha256=digest)

    def test_duplicate_or_missing_initial_attempt_rejected(self):
        for mode in ("duplicate","success_subset"):
            target = self.root/mode
            artifacts.write_snapshot(self.reg,target)
            rows = [json.loads(line) for line in (target/"attempts.jsonl").read_bytes().splitlines()]
            rows = rows+[rows[0]] if mode == "duplicate" else rows[1:]
            digest = self.reanchor(target,"attempts.jsonl",rows)
            with self.subTest(mode=mode),self.assertRaises(artifacts.ArtifactError):
                artifacts.read_snapshot(target,expected_manifest_sha256=digest)

    def test_deleted_failure_or_rewritten_manifest_detected_by_trusted_anchor(self):
        target,receipt = self.save()
        rows = [json.loads(line) for line in (target/"attempts.jsonl").read_bytes().splitlines()]
        self.reanchor(target,"attempts.jsonl",rows[1:])
        with self.assertRaisesRegex(artifacts.ArtifactError,"Trusted manifest"):
            self.read(target,receipt)

    def test_frozen_request_content_seal_detects_semantic_tampering(self):
        target,_ = self.save()
        rows = [json.loads(line) for line in (target/"requests.jsonl").read_bytes().splitlines()]
        rows[0]["request"]["failure_type"] = "changed-request-annotation"
        digest = self.reanchor(target,"requests.jsonl",rows)
        with self.assertRaisesRegex(artifacts.ArtifactError,"seal mismatch"):
            artifacts.read_snapshot(target,expected_manifest_sha256=digest)

    def test_damaged_json_duplicate_fields_and_nonfinite_records_rejected(self):
        for mode in ("syntax","duplicate_key","nonfinite"):
            target = self.root/mode
            receipt = artifacts.write_snapshot(self.reg,target)
            raw = (target/"attempts.jsonl").read_bytes()
            if mode == "syntax": raw = b"{invalid-json}\n"+raw.split(b"\n",1)[1]
            elif mode == "duplicate_key":
                raw = raw.replace(b'"attempt_no":1',b'"attempt_no":1,"attempt_no":1',1)
            else:
                raw = raw.replace(b'"attempt_no":1',b'"attempt_no":NaN',1)
            (target/"attempts.jsonl").write_bytes(raw)
            header = json.loads((target/"ledger_manifest.json").read_text())
            header["files"]["attempts.jsonl"]["sha256"] = sha(raw)
            header_raw = (independent_json(header)+"\n").encode("ascii")
            (target/"ledger_manifest.json").write_bytes(header_raw)
            with self.subTest(mode=mode),self.assertRaises(artifacts.ArtifactError):
                artifacts.read_snapshot(target,expected_manifest_sha256=sha(header_raw))

    def test_missing_extra_files_truncation_and_wrong_hash_rejected(self):
        for mode in ("missing","extra","truncated","hash"):
            target = self.root/mode
            receipt = artifacts.write_snapshot(self.reg,target)
            if mode == "missing": (target/"requests.jsonl").unlink()
            elif mode == "extra": (target/"extra.json").write_text("{}",encoding="ascii")
            elif mode == "truncated":
                raw = (target/"attempts.jsonl").read_bytes()[:-1]
                (target/"attempts.jsonl").write_bytes(raw)
            else:
                (target/"tasks.jsonl").write_bytes(b"corrupted\n")
            with self.subTest(mode=mode),self.assertRaises(artifacts.ArtifactError):
                self.read(target,receipt)

    def test_snapshot_must_be_sealed_and_requires_external_hash_receipt(self):
        reg,_,_,_ = registry_fixture(seal=False)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_snapshot(reg,self.root/"unsealed")
        target,_ = self.save()
        with self.assertRaises(TypeError): artifacts.read_snapshot(target)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.read_snapshot(target,expected_manifest_sha256="0"*64)


if __name__ == "__main__":
    unittest.main()
