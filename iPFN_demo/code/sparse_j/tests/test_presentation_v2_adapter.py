"""Portable presentation unit tests: synthetic fixtures only, never Formal Test evidence.

Collected by pytest or unittest. TemporaryDirectory keeps the same fixture usable
by the adapter's existing unittest entry point without a runtime pytest dependency.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_test_adapter", CODE / "presentation_v2_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def synthetic_metrics():
    identity = dict(source_run_id=adapter.RUN, source_snapshot_id=adapter.SNAPSHOT)
    values = ("1.250000", "-0.2500", "0.2500", "0.5000", "2.1250", "3.3750")
    rows, ranks = [], []
    for method in adapter.METHODS:
        for effect in adapter.EFFECTS:
            ranks.append(dict(identity, row_id=f"fixture-rank-{method}-{effect}",
                              method_id=method, effect=effect, status="RANKABLE"))
            for metric, value in zip(adapter.METRICS.values(), values):
                rows.append(dict(identity, row_id=f"fixture-{method}-{effect}-{metric}",
                    method_id=method, effect=effect, metric=metric, value=value,
                    n_requested="600", n_valid="600", n_failed_or_missing="0",
                    status="COMPLETE", ci_status="NOT_REQUESTED", ci_lower="", ci_upper=""))
    return rows, ranks


def synthetic_intervals():
    rows = []
    for center in ("", "0", "1"):
        for effect in adapter.EFFECTS:
            n = 1200 if center == "" else 600
            details = dict(all_requested=n, valid_intervals=n, failed_or_missing=0,
                interval_level=.95, coverage_including_failures=.95,
                mean_interval_width=2.5, threshold_overall_only=.9 if center == "" else None,
                scope="overall" if center == "" else "center_" + center,
                covered_valid=int(n * .95), interval_success_rate=1.,
                meets_90pct_target=True if center == "" else None)
            rows.append(dict(row_id=f"fixture-j-{effect}-{center}",
                source_run_id=adapter.RUN, source_snapshot_id=adapter.SNAPSHOT,
                method_id="j", effect=effect, center_t=center,
                metric="coverage_including_failures", status="COMPLETE", ci_status="OK",
                ci_level="0.95", ci_lower="0.91", ci_upper="0.98", value="0.95",
                n_requested=str(n), n_valid=str(n), n_failed_or_missing="0",
                details=json.dumps(details)))
    return rows


def synthetic_coordinates():
    """Fixed arithmetic records, not DGP draws, predictions or measured results."""
    ordinary = {}
    for method in adapter.NODE_METHODS:
        contexts, effects = [], {e: dict(raw_rows=[]) for e in adapter.EFFECTS}
        for mechanism in range(200):
            for repeat in range(3):
                for center in (0, 1):
                    context_id = f"fixture-{mechanism}-{repeat}-{center}"
                    key = [method, "context", context_id, adapter.RUN + ":" + method + ":" + context_id]
                    parent = dict(key=key, mechanism_id=mechanism, effects={})
                    for index, effect in enumerate(adapter.EFFECTS):
                        values = dict(true_center=4. + index, true_leaf=1.,
                            predicted_center=4.5 + index, predicted_leaf=1.25,
                            true_difference=3. + index, predicted_difference=3.25 + index,
                            error=.25)
                        parent["effects"][effect] = dict(values)
                        effects[effect]["raw_rows"].append(dict(values, key=list(key),
                            mechanism_id=mechanism, pair_repeat_id=repeat, center_t=center))
                    contexts.append(parent)
        ordinary[method] = dict(context_records=contexts, effects=effects)
    return dict(fixture_kind="SYNTHETIC_ONLY_NOT_FORMAL_EVIDENCE", ordinary=ordinary)


class PresentationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_source, cls.ranks = synthetic_metrics()
        cls.coverage = synthetic_intervals()
        cls.coordinate_source = synthetic_coordinates()

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="presentation-unit-")))

    def template_runtime(self, representation="original"):
        runtime = self.root / representation
        runtime.mkdir()
        for name in adapter.TEMPLATES:
            source = (CODE / name).read_bytes()
            normalized = source.replace(b"\r\n", b"\n")
            payload = normalized.replace(b"\n", b"\r\n") if representation == "original" else normalized
            (runtime / name).write_bytes(payload)
        return runtime

    def test_main_all_126_values_and_21_rows_are_exact_source_strings(self):
        actual = adapter.extract_main(self.main_source, self.ranks)
        self.assertEqual(len(actual), 21)
        lookup = {(r["method_id"], r["effect"], r["metric"]): r for r in self.main_source}
        for row in actual:
            for target, source in adapter.METRICS.items():
                original = lookup[row["method_id"], row["effect"], source]
                self.assertEqual(row[target], original["value"])
                self.assertEqual(row["source_row_ids"][target], original["row_id"])
            self.assertEqual((row["n_requested"], row["n_valid"], row["n_failed_or_missing"]),
                             ("600", "600", "0"))

    def test_absolute_bias_is_not_mislabeled_acb(self):
        main = adapter.extract_main(self.main_source, self.ranks)
        intervals = adapter.extract_intervals(self.coverage)
        self.assertNotEqual(main[0]["absolute_bias"], main[0]["acb"])
        template = SimpleNamespace(METHOD_NAMES={m: m for m in adapter.METHODS}, latex_escape=lambda s: s)
        tex = adapter.render_main_tex(main, intervals, template)
        self.assertEqual(tex.count(r"\begin{table}"), 4)
        for label in ("Signed Bias", "Absolute Bias", "ACB"):
            self.assertIn(label, tex)
        self.assertNotIn("cellcolor", tex)
        self.assertNotIn(r"\textbf{", tex)

    def test_missing_duplicate_wrong_identity_main_rows_fail(self):
        with self.assertRaises(ValueError):
            adapter.extract_main(self.main_source[:-1], self.ranks)
        rows = copy.deepcopy(self.main_source)
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            adapter.extract_main(rows, self.ranks)
        rows = copy.deepcopy(self.main_source)
        rows[0]["source_run_id"] = "old_v1"
        with self.assertRaises(ValueError):
            adapter.extract_main(rows, self.ranks)

    def test_failure_or_nonrankable_is_not_repaired_to_zero(self):
        rows = copy.deepcopy(self.main_source)
        rows[0].update(status="INCOMPLETE", value="")
        with self.assertRaises(ValueError):
            adapter.extract_main(rows, self.ranks)
        ranks = copy.deepcopy(self.ranks)
        ranks[0]["status"] = "NOT_RANKABLE"
        with self.assertRaises(ValueError):
            adapter.extract_main(self.main_source, ranks)

    def test_nine_interval_rows_copy_width_coverage_existing_ci(self):
        actual = adapter.extract_intervals(self.coverage)
        self.assertEqual(len(actual), 9)
        lookup = {(r["effect"], r["center_t"]): r for r in self.coverage}
        for row in actual:
            source = lookup[row["effect"], row["center_t"]]
            self.assertEqual(row["mean_interval_width"], json.loads(source["details"])["mean_interval_width"])
            self.assertEqual(row["coverage"], source["value"])
            for field in ("ci_lower", "ci_upper", "ci_level", "ci_status",
                          "n_requested", "n_valid", "n_failed_or_missing"):
                self.assertEqual(row[field], source[field])
            self.assertEqual(row["width_ci_status"], "NOT_REQUESTED")
            self.assertEqual(row["meets_90pct_target"], True if row["center_t"] == "" else None)

    def test_interval_inventory_or_width_tampering_fails(self):
        with self.assertRaises(ValueError):
            adapter.extract_intervals(self.coverage[:-1])
        rows = copy.deepcopy(self.coverage)
        details = json.loads(rows[0]["details"])
        details["mean_interval_width"] = None
        rows[0]["details"] = json.dumps(details)
        with self.assertRaises(ValueError):
            adapter.extract_intervals(rows)

    def test_10800_coordinates_equal_raw_record_by_pointer(self):
        rows = adapter.extract_coordinates(self.coordinate_source)
        self.assertEqual(len(rows), 10800)
        keys = set()
        for row in rows:
            source = self.coordinate_source
            for part in row["source_json_pointer"].split("/")[1:]:
                source = source[int(part)] if isinstance(source, list) else source[part]
            for field in adapter.COORDINATES:
                self.assertEqual(row[field], source[field])
            self.assertEqual(row["context_id"], source["key"][2])
            self.assertEqual(row["request_id"], source["key"][3])
            for field in ("mechanism_id", "pair_repeat_id", "center_t"):
                self.assertEqual(row[field], source[field])
            keys.add((row["method_id"], row["effect"], row["context_id"], row["request_id"]))
        self.assertEqual(len(keys), 10800)

    def test_missing_coordinate_record_fails_instead_of_fabrication(self):
        document = copy.deepcopy(self.coordinate_source)
        document["ordinary"]["j"]["effects"]["Direct"]["raw_rows"].pop()
        with self.assertRaises(ValueError):
            adapter.extract_coordinates(document)

    def test_conflicting_saved_coordinates_fail(self):
        document = copy.deepcopy(self.coordinate_source)
        document["ordinary"]["j"]["effects"]["Direct"]["raw_rows"][0]["true_center"] += 1
        with self.assertRaises(ValueError):
            adapter.extract_coordinates(document)

    def test_cross_method_truth_conflict_fails(self):
        document = copy.deepcopy(self.coordinate_source)
        item = document["ordinary"]["kernel-new"]
        row = item["effects"]["Direct"]["raw_rows"][0]
        row["true_center"] += 1.
        row["true_difference"] += 1.
        row["error"] -= 1.
        item["context_records"][0]["effects"]["Direct"] = {f: row[f] for f in adapter.COORDINATES}
        with self.assertRaisesRegex(ValueError, "Cross-method truth"):
            adapter.extract_coordinates(document)

    def test_csv_roundtrip_preserves_all_source_strings_and_pointers(self):
        cases = (("main.csv", adapter.MAIN_COLUMNS, adapter.extract_main(self.main_source, self.ranks)),
                 ("j.csv", adapter.J_COLUMNS, adapter.extract_intervals(self.coverage)),
                 ("coordinates.csv", adapter.COORD_COLUMNS, adapter.extract_coordinates(self.coordinate_source)))
        for name, columns, rows in cases:
            path = self.root / name
            adapter.write_csv(path, columns, rows)
            self.assertEqual(adapter.csv_rows(path),
                [{k: adapter.wire_cell(row[k]) for k in columns} for row in rows])

    def test_path_traversal_and_noncanonical_names_rejected(self):
        for name in ("../x", "/absolute", "C:/x", "a/../b", "a//b", "a/./b", "a\\b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.relative_path(name)
        self.assertEqual(adapter.relative_path("tables/main_results.csv"), "tables/main_results.csv")

    def test_safe_path_still_rejects_git_runtime(self):
        root = self.root / "checkout"
        root.mkdir()
        (root / ".git").write_text("fixture", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter.safe_path(root, True)

    def test_original_and_git_lf_template_provenance(self):
        for representation in ("original", "normalized"):
            runtime = self.template_runtime(representation)
            evidence = adapter.template_provenance(runtime)
            for name, item in evidence.items():
                self.assertEqual(item["original_sha256"], adapter.TEMPLATES[name])
                self.assertEqual(item["git_normalized_sha256"], adapter.NORMALIZED_TEMPLATES[name])
                self.assertEqual(item["actual_sha256"], adapter.sha(runtime / name))
                self.assertEqual(item["path"], str((runtime / name).resolve()))
                self.assertEqual(item["representation"],
                    "original_bytes" if representation == "original" else "git_lf_normalized")
                self.assertEqual(item["normalization_rule"],
                    "CRLF_TO_LF_ONLY; no encoding, whitespace or other changes")

    def test_template_source_change_is_rejected(self):
        runtime = self.template_runtime("normalized")
        path = runtime / "make_table.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        with self.assertRaises(ValueError):
            adapter.template_provenance(runtime)

    def test_mixed_newlines_are_not_an_accepted_representation(self):
        runtime = self.template_runtime("original")
        path = runtime / "make_figures.py"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n", 1))
        with self.assertRaisesRegex(ValueError, "representation"):
            adapter.template_provenance(runtime)

    def test_encoding_change_is_rejected(self):
        runtime = self.template_runtime("normalized")
        path = runtime / "make_table.py"
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        with self.assertRaises(ValueError):
            adapter.template_provenance(runtime)

    def test_template_is_rechecked_before_execution(self):
        runtime = self.template_runtime("original")
        adapter.template_provenance(runtime)
        path = runtime / "make_table.py"
        path.write_bytes(path.read_bytes() + b"# tampered\n")
        with self.assertRaises(ValueError):
            adapter.load_templates(runtime)

    def test_template_representation_change_cannot_stale_provenance(self):
        runtime = self.template_runtime("original")
        evidence = adapter.template_provenance(runtime)
        for name in adapter.TEMPLATES:
            path = runtime / name
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            adapter.load_templates(runtime, expected_identity=evidence)

    def test_unknown_template_and_git_runtime_are_rejected(self):
        with self.assertRaises(ValueError):
            adapter._template_payload(self.root / "x.py", "x.py")

    def test_cli_requires_explicit_input_paths(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            adapter.main([])
        self.assertEqual(error.exception.code, 2)

    def test_formal_acceptance_receipt_cannot_be_replaced_by_fixture(self):
        analysis = self.root / "analysis"
        runtime = self.root / "runtime"
        analysis.mkdir()
        runtime.mkdir()
        receipt = self.root / "synthetic_receipt.json"
        receipt.write_text('{"fixture_kind":"SYNTHETIC_ONLY_NOT_FORMAL_EVIDENCE"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt SHA mismatch"):
            adapter.load_inputs(analysis, receipt, runtime)

    def test_runtime_template_loading_uses_verified_original_or_lf_bytes(self):
        for representation in ("original", "normalized"):
            runtime = self.template_runtime(representation)
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / "mpl-cache")}):
                try:
                    table, figures = adapter.load_templates(runtime)
                    self.assertEqual(Path(table.__file__), (runtime / "make_table.py").resolve())
                    self.assertEqual(Path(figures.__file__), (runtime / "make_figures.py").resolve())
                    self.assertTrue(callable(table.latex_escape))
                    self.assertEqual(table.latex_escape("a_b"), r"a\_b")
                finally:
                    sys.modules.pop("make_table", None)
                    sys.modules.pop("_presentation_make_figures", None)

    def test_adapter_has_no_scientific_import_or_personal_path(self):
        source = (CODE / "presentation_v2_adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        self.assertFalse(any(n.startswith(("star_v2", "pfn_social", "predict", "benchmark")) for n in modules))
        for forbidden in ("C:/Users/", "C:\\Users\\", "/home/", "Desktop", "PRESENTATION_V2_OUTPUT"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
