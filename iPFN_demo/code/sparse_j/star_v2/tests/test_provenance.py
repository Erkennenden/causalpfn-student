"""6.4-A provenance tests: temporary synthetic sources and ledgers only."""
from pathlib import Path
from collections import Counter
import ast
import copy
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

from star_v2 import provenance as p
from star_v2.coverage import AttemptSelectionAmbiguous
from test_task_registry import registry_fixture, bundle_fixture

HEAD = "fbc0d9e09789248139ff9a201b8f89bf71ae57cf"
LINK_EVIDENCE = []


def independent_hash(blob):
    return hashlib.sha256(blob).hexdigest()


def independent_json(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")


def as_plain(value):
    return json.loads(p._json_bytes(value))


def load_synthetic_module(path, name="pkg.module"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # only a temporary VALUE=... fixture
    return module


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory(prefix="star-v2-provenance-test-"))
        self.root = Path(self.temp)
        self.source = self.root/"original"
        (self.source/"pkg").mkdir(parents=True)
        self.payloads = {"pkg/__init__.py": b"",
                         "pkg/module.py": b"VALUE = 17\n",
                         "config.json": b'{"fixture":true}\r\n'}
        for name, blob in self.payloads.items():
            (self.source/name).write_bytes(blob)
        self.specs = (p.SourceFile("pkg/module.py", "implementation"),
                      p.SourceFile("config.json", "configuration"),
                      p.SourceFile("pkg/__init__.py", "implementation"))

    def build(self, name="delivery", **kwargs):
        return p.build_source_snapshot(self.source, self.root/name, git_head=HEAD,
                                        files=kwargs.get("files", self.specs))

    def read(self, receipt):
        return p.verify_source_snapshot(receipt.directory,
            expected_manifest_sha256=receipt.manifest_sha256)

    def extract(self, receipt, name="extracted"):
        return p.extract_source_snapshot(receipt.directory, self.root/name,
            expected_manifest_sha256=receipt.manifest_sha256)

    def rewrite_archive(self, receipt, mode):
        """Independent attacker writer, including freshly anchored manifest."""
        destination = Path(receipt.directory)
        path = destination/p.ARCHIVE_NAME
        old = json.loads((destination/p.MANIFEST_NAME).read_bytes())
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, blob in sorted(self.payloads.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                if mode == "symlink" and name == "pkg/module.py":
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, blob)
            if mode in ("traversal", "duplicate"):
                name = "../escaped.py" if mode == "traversal" else "pkg/module.py"
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(info, b"attack")
        blob = path.read_bytes()
        old["archive"]["sha256"] = independent_hash(blob)
        old["archive"]["bytes"] = len(blob)
        header = independent_json(old)
        (destination/p.MANIFEST_NAME).write_bytes(header)
        return independent_hash(header)

    def run_manifest(self, receipt):
        source = self.read(receipt)
        missing = {"status": "NOT_MEASURED", "reason": "synthetic fixture: not probed"}
        environment = {name: dict(missing) for name in ("python", "os", "cpu", "gpu")}
        environment["dependencies"] = {"numpy": dict(missing)}
        return p.build_run_manifest(run_id="synthetic-contract-only", run_kind="synthetic",
            source_manifest=source, source_manifest_sha256=receipt.manifest_sha256,
            git_identity={"head": HEAD, "branch": "synthetic-fixture", "status_short": []},
            entrypoint="pkg/module.py", environment=environment, model_assets=(),
            task_ledger_manifest_sha256="a"*64, registry_seal_sha256="b"*64)

    def test_same_bytes_reordered_whitelist_and_mtime_produce_identical_zip_and_id(self):
        first = self.build("first")
        for name in self.payloads:
            os.utime(self.source/name, (1000000000, 1500000000))
        second = self.build("second", files=tuple(reversed(self.specs)))
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(first.archive_sha256, second.archive_sha256)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual((Path(first.directory)/p.ARCHIVE_NAME).read_bytes(),
                         (Path(second.directory)/p.ARCHIVE_NAME).read_bytes())

    def test_source_change_changes_snapshot_identity(self):
        first = self.build("first")
        (self.source/"pkg/module.py").write_bytes(b"VALUE = 18\n")
        second = self.build("second")
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertNotEqual(first.archive_sha256, second.archive_sha256)

    def test_independent_manifest_hashes_sizes_order_and_nonrecursive_snapshot_id(self):
        receipt = self.build()
        header = (Path(receipt.directory)/p.MANIFEST_NAME).read_bytes()
        manifest = json.loads(header)
        records = [{"path": name,
            "role": "configuration" if name == "config.json" else "implementation",
            "bytes": len(blob), "sha256": independent_hash(blob)}
            for name, blob in sorted(self.payloads.items())]
        self.assertEqual(manifest["files"], records)
        expected_id = independent_hash(independent_json(records))
        self.assertEqual(receipt.snapshot_id, expected_id)
        self.assertEqual(manifest["snapshot_id"], expected_id)
        self.assertEqual(receipt.manifest_sha256, independent_hash(header))
        self.assertEqual(manifest["runtime_root"], ".")
        self.assertEqual(manifest["git_head"], HEAD)
        self.assertNotIn("snapshot_id", records[0])
        self.assertNotIn("archive", records[0])

    def test_each_source_read_once_and_same_buffer_survives_later_source_mutation(self):
        original_read = p._read_once
        read_names = []
        def read(root, name):
            read_names.append(name)
            return original_read(root, name)
        original_zip = p._zip_bytes
        def mutate_after_read(payloads):
            (self.source/"pkg/module.py").write_bytes(b"CHANGED AFTER READ\n")
            return original_zip(payloads)
        with patch.object(p, "_read_once", side_effect=read), patch.object(
                p, "_zip_bytes", side_effect=mutate_after_read):
            receipt = self.build()
        self.assertEqual(Counter(read_names), Counter({name: 1 for name in self.payloads}))
        with zipfile.ZipFile(Path(receipt.directory)/p.ARCHIVE_NAME) as archive:
            self.assertEqual(archive.read("pkg/module.py"), b"VALUE = 17\n")
        manifest = self.read(receipt)
        record = next(r for r in manifest["files"] if r["path"] == "pkg/module.py")
        self.assertEqual(record["sha256"], independent_hash(b"VALUE = 17\n"))

    def test_zip_uses_fixed_timestamp_permissions_and_stored_compression(self):
        receipt = self.build()
        with zipfile.ZipFile(Path(receipt.directory)/p.ARCHIVE_NAME) as archive:
            self.assertEqual(archive.namelist(), sorted(self.payloads))
            for member in archive.infolist():
                self.assertEqual(member.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(member.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(member.create_system, 3)
                self.assertEqual(member.external_attr >> 16, stat.S_IFREG | 0o444)
                self.assertEqual(member.extra, b"")
                self.assertEqual(member.comment, b"")
                self.assertEqual(archive.read(member), self.payloads[member.filename])

    def test_missing_source_rejected_without_creating_output(self):
        (self.source/"pkg/module.py").unlink()
        with self.assertRaisesRegex(p.ProvenanceError, "Missing"):
            self.build()
        self.assertFalse((self.root/"delivery").exists())

    def test_invalid_paths_binary_assets_and_case_duplicate_whitelist_rejected(self):
        bad_paths = ("../escape.py", "/absolute.py", "C:/drive.py", "a\\b.py",
                     "a//b.py", "./a.py", "a/../b.py", "a:stream", "CON.py",
                     "a./b.py", "a/.git/config", "a/__pycache__/x.pyc")
        for bad in bad_paths:
            with self.subTest(path=bad), self.assertRaises(p.ProvenanceError):
                p.SourceFile(bad, "implementation")
        for bad in ("weights.pt", "prior.npz", "opaque.pkl"):
            with self.assertRaises(p.ProvenanceError):
                p.SourceFile(bad, "implementation")
        for duplicate in (self.specs+(self.specs[0],),
                          self.specs+(p.SourceFile("PKG/module.py", "implementation"),)):
            with self.assertRaisesRegex(p.ProvenanceError, "Duplicate"):
                self.build(files=duplicate)

    def test_symlink_file_and_parent_paths_are_rejected(self):
        target = self.root/"outside.py"
        target.write_bytes(b"VALUE = 99\n")
        link = self.source/"link.py"
        try:
            os.symlink(target, link)
        except OSError:
            # No skip: directly exercise the OS lstat link/reparse rejection.
            LINK_EVIDENCE.append("mock_lstat: symlink privilege unavailable")
            fake = type("LinkStat", (), {"st_mode": stat.S_IFLNK | 0o777,
                                        "st_file_attributes": 0x400})()
            original_lstat = Path.lstat
            def lstat(path, *args, **kwargs):
                return fake if path == link else original_lstat(path, *args, **kwargs)
            with patch.object(Path, "lstat", lstat), self.assertRaisesRegex(
                    p.ProvenanceError, "Symlink"):
                p.build_source_snapshot(self.source, self.root/"linkout", git_head=HEAD,
                    files=(p.SourceFile("link.py", "implementation"),))
        else:
            LINK_EVIDENCE.append("real_filesystem_symlink")
            with self.assertRaisesRegex(p.ProvenanceError, "Symlink"):
                p.build_source_snapshot(self.source, self.root/"linkout", git_head=HEAD,
                    files=(p.SourceFile("link.py", "implementation"),))
            directory = self.source/"linked_dir"
            os.symlink(target.parent, directory, target_is_directory=True)
            with self.assertRaisesRegex(p.ProvenanceError, "Symlink"):
                p.build_source_snapshot(self.source, self.root/"dirout", git_head=HEAD,
                    files=(p.SourceFile("linked_dir/outside.py", "implementation"),))


    def test_real_windows_directory_junction_is_rejected(self):
        if os.name != "nt":
            directory = self.source/"linked_directory"
            os.symlink(self.root, directory, target_is_directory=True)
        else:
            import subprocess
            directory = self.source/"linked_directory"
            result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J",
                                     str(directory), str(self.root)],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(directory.is_junction())
        LINK_EVIDENCE.append("real_directory_reparse_point")
        with self.assertRaisesRegex(p.ProvenanceError, "Symlink/reparse"):
            p.build_source_snapshot(self.source, self.root/"junctionout", git_head=HEAD,
                files=(p.SourceFile("linked_directory/anything.py", "implementation"),))
        self.assertFalse((self.root/"junctionout").exists())

    def test_zip_traversal_duplicate_and_symlink_members_rejected_after_reanchoring(self):
        for mode in ("traversal", "duplicate", "symlink"):
            receipt = self.build(mode)
            digest = self.rewrite_archive(receipt, mode)
            with self.subTest(mode=mode), self.assertRaises(p.ProvenanceError):
                p.extract_source_snapshot(receipt.directory, self.root/(mode+"_out"),
                                          expected_manifest_sha256=digest)
            self.assertFalse((self.root/(mode+"_out")).exists())
        self.assertFalse((self.root/"escaped.py").exists())

    def test_untrusted_manifest_hash_and_corrupt_archive_rejected(self):
        receipt = self.build()
        with self.assertRaisesRegex(p.ProvenanceError, "Trusted"):
            p.verify_source_snapshot(receipt.directory, expected_manifest_sha256="f"*64)
        (Path(receipt.directory)/p.ARCHIVE_NAME).write_bytes(b"corrupted")
        with self.assertRaisesRegex(p.ProvenanceError, "Archive bytes/hash"):
            self.read(receipt)

    def test_manifest_file_hash_and_snapshot_id_conflicts_rejected(self):
        receipt = self.build()
        manifest_path = Path(receipt.directory)/p.MANIFEST_NAME
        original = json.loads(manifest_path.read_bytes())
        tampered = copy.deepcopy(original)
        tampered["snapshot_id"] = "0"*64
        manifest_path.write_bytes(independent_json(tampered))
        with self.assertRaises(p.ProvenanceError):
            p.verify_source_snapshot(receipt.directory,
                expected_manifest_sha256=independent_hash(manifest_path.read_bytes()))
        tampered = copy.deepcopy(original)
        tampered["files"][0]["sha256"] = "f"*64
        tampered["snapshot_id"] = independent_hash(independent_json(tampered["files"]))
        manifest_path.write_bytes(independent_json(tampered))
        with self.assertRaisesRegex(p.ProvenanceError, "ZIP source hash"):
            p.verify_source_snapshot(receipt.directory,
                expected_manifest_sha256=independent_hash(manifest_path.read_bytes()))

    def test_extracted_tree_has_exact_original_paths_and_bytes(self):
        receipt = self.build()
        manifest = self.extract(receipt)
        output = self.root/"extracted"
        files = {file.relative_to(output).as_posix(): file.read_bytes()
                 for file in output.rglob("*") if file.is_file()}
        self.assertEqual(files, self.payloads)
        self.assertTrue(p.verify_runtime_tree(output, manifest))
        for record in manifest["files"]:
            self.assertEqual(record["sha256"], independent_hash(files[record["path"]]))

    def test_runtime_tree_missing_extra_and_modified_sources_rejected(self):
        receipt = self.build()
        for mode in ("missing", "extra", "modified"):
            manifest = self.extract(receipt, mode)
            root = self.root/mode
            if mode == "missing":
                (root/"config.json").unlink()
            elif mode == "extra":
                (root/"unexpected.py").write_bytes(b"")
            else:
                (root/"pkg/module.py").write_bytes(b"VALUE=999\n")
            with self.subTest(mode=mode), self.assertRaises(p.ProvenanceError):
                p.verify_runtime_tree(root, manifest)

    def test_original_tree_actual_loaded_module_is_rejected(self):
        receipt = self.build()
        manifest = self.extract(receipt)
        module = load_synthetic_module(self.source/"pkg/module.py")
        self.assertEqual(module.VALUE, 17)
        with self.assertRaisesRegex(p.ProvenanceError, "outside Snapshot"):
            p.verify_runtime_modules({"pkg.module": module}, self.root/"extracted", manifest)

    def test_snapshot_actual_loaded_module_path_hash_and_required_set_pass(self):
        receipt = self.build()
        manifest = self.extract(receipt)
        module = load_synthetic_module(self.root/"extracted/pkg/module.py")
        self.assertEqual(module.VALUE, 17)
        result = p.verify_runtime_modules({"pkg.module": module}, self.root/"extracted", manifest,
                                          required_modules=("pkg.module",))
        self.assertEqual(result[0]["module"], "pkg.module")
        self.assertEqual(result[0]["path"], "pkg/module.py")
        self.assertEqual(result[0]["sha256"], independent_hash(b"VALUE = 17\n"))
        self.assertFalse(any(path.name == "__pycache__" for path in self.root.rglob("*")))

    def test_runtime_module_hash_name_and_missing_required_conflicts_rejected(self):
        receipt = self.build()
        manifest = self.extract(receipt)
        module = load_synthetic_module(self.root/"extracted/pkg/module.py")
        with self.assertRaisesRegex(p.ProvenanceError, "Required"):
            p.verify_runtime_modules({"pkg.module": module}, self.root/"extracted", manifest,
                                     required_modules=("pkg.other",))
        with self.assertRaisesRegex(p.ProvenanceError, "object/name"):
            p.verify_runtime_modules({"pkg.other": module}, self.root/"extracted", manifest)
        (self.root/"extracted/pkg/module.py").write_bytes(b"VALUE=18\n")
        with self.assertRaisesRegex(p.ProvenanceError, "hash mismatch"):
            p.verify_runtime_modules({"pkg.module": module}, self.root/"extracted", manifest)

    def test_existing_output_and_recursive_delivery_never_overwritten(self):
        receipt = self.build()
        before = {p.name: p.read_bytes() for p in Path(receipt.directory).iterdir()}
        with self.assertRaisesRegex(p.ProvenanceError, "already exists"):
            self.build()
        self.assertEqual(before, {p.name: p.read_bytes() for p in Path(receipt.directory).iterdir()})
        self.extract(receipt)
        with self.assertRaisesRegex(p.ProvenanceError, "already exists"):
            self.extract(receipt)
        with self.assertRaisesRegex(p.ProvenanceError, "outside source root"):
            p.build_source_snapshot(self.source, self.source/"recursive", git_head=HEAD, files=self.specs)

    def test_run_manifest_minimal_fields_provenance_and_unknown_observations(self):
        receipt = self.build()
        run = self.run_manifest(receipt)
        self.assertEqual(run["run_kind"], "synthetic")
        self.assertEqual(run["source_snapshot"]["snapshot_id"], receipt.snapshot_id)
        self.assertEqual(run["source_snapshot"]["manifest_sha256"], receipt.manifest_sha256)
        self.assertEqual(run["task_ledger_manifest_sha256"], "a"*64)
        self.assertEqual(run["registry_seal_sha256"], "b"*64)
        self.assertEqual(run["environment"]["gpu"]["status"], "NOT_MEASURED")
        self.assertEqual(run["model_assets"], ())
        self.assertEqual(run["entrypoint"], "pkg/module.py")
        p.validate_run_manifest(run, source_manifest=self.read(receipt))
        self.assertEqual({f.name for f in Path(receipt.directory).iterdir()},
                         {"source_snapshot.zip", "source_manifest.json"})

    def test_run_manifest_missing_fields_invalid_protocol_ledger_and_snapshot_rejected(self):
        receipt = self.build()
        valid = as_plain(self.run_manifest(receipt))
        for name in valid:
            bad = copy.deepcopy(valid)
            del bad[name]
            with self.subTest(field=name), self.assertRaises(p.ProvenanceError):
                p.validate_run_manifest(bad)
        for field in ("task_ledger_manifest_sha256", "registry_seal_sha256"):
            bad = copy.deepcopy(valid)
            bad[field] = "not-a-hash"
            with self.assertRaises(p.ProvenanceError):
                p.validate_run_manifest(bad)
        bad = copy.deepcopy(valid)
        bad["protocol"]["protocol_sha256"] = "f"*64
        with self.assertRaises(p.ProvenanceError):
            p.validate_run_manifest(bad)
        bad = copy.deepcopy(valid)
        bad["source_snapshot"]["snapshot_id"] = "f"*64
        with self.assertRaisesRegex(p.ProvenanceError, "binding"):
            p.validate_run_manifest(bad, source_manifest=self.read(receipt))

    def test_run_environment_and_assets_cannot_fabricate_unknown_values(self):
        receipt = self.build()
        run = as_plain(self.run_manifest(receipt))
        for field in ("python", "os", "cpu", "gpu"):
            bad = copy.deepcopy(run)
            bad["environment"][field] = {"status": "OBSERVED", "value": None}
            with self.assertRaises(p.ProvenanceError):
                p.validate_run_manifest(bad)
        bad = copy.deepcopy(run)
        bad["model_assets"] = [{"path": "models/j.pt", "status": "VERIFIED",
                                 "sha256": "a"*64, "bytes": None}]
        with self.assertRaises(p.ProvenanceError):
            p.validate_run_manifest(bad)
        declared = copy.deepcopy(run)
        declared["model_assets"] = [{"path": "models/j.pt", "status": "DECLARED",
                                     "sha256": "a"*64, "bytes": None}]
        self.assertEqual(p.validate_run_manifest(declared)["model_assets"][0]["status"], "DECLARED")

    def test_environment_collector_uses_distribution_metadata_not_torch_import(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "predict", "pfn_social"):
                raise AssertionError("must not import models to observe metadata")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            environment = p.collect_environment(dependencies=("definitely-not-installed-star-v2-fixture",))
        self.assertEqual(environment["python"]["status"], "OBSERVED")
        self.assertEqual(environment["gpu"]["status"], "NOT_MEASURED")
        self.assertEqual(environment["dependencies"]["definitely-not-installed-star-v2-fixture"]["status"],
                         "UNAVAILABLE")

    def test_ledger_export_exact_existing_format_preserves_failed_attempts_and_ambiguity_gate(self):
        registry, task, parent, _ = registry_fixture()
        registry.append_attempt(bundle_fixture(task, parent, states=("FAILED",)*3), attempt_no=1)
        registry.append_attempt(bundle_fixture(task, parent), attempt_no=2, authorization="synthetic-approved")
        destination = self.root/"ledger"
        with patch.object(p.artifacts, "write_snapshot", wraps=p.artifacts.write_snapshot) as writer:
            receipt = p.export_task_manifest(registry, destination)
        writer.assert_called_once_with(registry, destination)
        self.assertEqual({f.name for f in destination.iterdir()},
                         {"tasks.jsonl", "requests.jsonl", "attempts.jsonl", "ledger_manifest.json"})
        loaded = p.load_task_manifest(destination, expected_manifest_sha256=receipt.manifest_sha256,
                                     for_evaluation=False)
        self.assertEqual(loaded.attempts, registry.attempts)
        self.assertEqual([a.document["execution_status"] for a in loaded.attempts], ["FAILED", "OK"])
        with self.assertRaises(AttemptSelectionAmbiguous):
            p.load_task_manifest(destination, expected_manifest_sha256=receipt.manifest_sha256)
        self.assertEqual(len(registry.attempts), 2)
        with self.assertRaises(p.artifacts.ArtifactError):
            p.export_task_manifest(registry, destination)

    def test_no_pickle_or_model_execution_dependencies_and_closed_explicit_allowlist(self):
        source = Path(p.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.split(".")[0] in ("pickle", "torch", "predict", "pfn_social")
                             for name in imports))
        paths = [f.path for f in p.SOURCE_ALLOWLIST]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("star_v2/provenance.py", paths)
        self.assertIn("star_v2/tests/test_provenance.py", paths)
        for module in ("data_artifacts", "runtime_monitor", "run_pilot"):
            self.assertIn("star_v2/"+module+".py", paths)
            self.assertIn("star_v2/tests/test_"+module+".py", paths)
        self.assertIn("predict.py", paths)
        self.assertIn("pfn_social/_vendor/hinet/layers.py", paths)
        self.assertIn("star_v2/protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md", paths)
        self.assertFalse(any(name.endswith((".pt", ".npz", ".pkl")) for name in paths))
        # A future unreviewed file is not silently included.
        (self.source/"unreviewed.py").write_bytes(b"")
        receipt = self.build()
        self.assertNotIn("unreviewed.py", [r["path"] for r in self.read(receipt)["files"]])


if __name__ == "__main__":
    unittest.main()
