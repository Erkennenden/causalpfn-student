"""Configuration and frozen-source checks only; no DGP imports."""
from dataclasses import FrozenInstanceError, asdict, fields
import hashlib
import json
from pathlib import Path
import unittest

from star_v2.config import CONFIG, StarV2Config
from star_v2.rng import CHANNELS, load_registry, validate_registry


class ConfigTests(unittest.TestCase):
    def test_T1_frozen_constants(self):
        expected = {
            "protocol_id": "e1_5_star_v2", "protocol_version": "Candidate v0.3.2",
            "protocol_sha256": "fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "source_commit": "fbc0d9e09789248139ff9a201b8f89bf71ae57cf",
            "root_seed": 20261030, "protocol_code": 2026103001,
            "pilot_cohort_code": 1, "test_cohort_code": 2,
            "N": 200, "M": 50, "center_node": 0, "x_dim": 20,
            "leaf_treatment_probability": 0.5, "exposure_threshold": 0.5,
            "exposure_operator": "strictly greater than", "base_setting_id": 0,
            "representation": "shared_nonlinear", "noise_sd": 1.0,
            "representation_seed": 2026101604,
            "arm_order": ("00", "01", "10", "11"),
            "effect_order": ("Direct", "Spillover", "Total"),
            "target_id": "star_v2_mu_iid_bernoulli_0p5_majority_gt0p5_v1",
            "pilot_mechanisms": 5, "pilot_pairs_per_mechanism": 2,
            "test_mechanisms": 200, "test_pairs_per_mechanism": 3,
        }
        self.assertEqual(asdict(CONFIG), expected)
        self.assertEqual(CONFIG.cohort_limits(1), (5, 2))
        self.assertEqual(CONFIG.cohort_limits(2), (200, 3))

    def test_T1_configuration_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            CONFIG.N = 4
        with self.assertRaises(TypeError):
            CONFIG.arm_order[0] = "11"

    def test_T1_reject_changes_to_every_registered_field(self):
        for field in fields(StarV2Config):
            original = getattr(CONFIG, field.name)
            if type(original) is int:
                invalid = original + 1
            elif type(original) is float:
                invalid = original + 0.1
            elif type(original) is tuple:
                invalid = original[::-1]
            else:
                invalid = original + "_changed"
            with self.subTest(field=field.name):
                with self.assertRaises(ValueError):
                    StarV2Config(**{field.name: invalid})

    def test_T1_reject_wrong_types_and_mutable_orders(self):
        for kwargs in ({"N": 200.0}, {"N": True}, {"M": -1},
                       {"center_node": False}, {"x_dim": 0},
                       {"leaf_treatment_probability": 1},
                       {"exposure_threshold": float("nan")},
                       {"arm_order": ["00", "01", "10", "11"]}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    StarV2Config(**kwargs)
        for cohort in (0, 3, -1, True, 1.0, "1", None):
            with self.subTest(cohort=cohort):
                with self.assertRaises((TypeError, ValueError)):
                    CONFIG.cohort_limits(cohort)

    def test_T1_registry_codes_scopes_and_immutability(self):
        expected = {
            "population": (1, "mechanism"), "x": (2, "mechanism"),
            "coefficient": (3, "mechanism"), "leaf_assignment": (4, "pair"),
            "noise": (5, "pair"), "observation_index": (6, "pair"),
            "bootstrap": (7, "global_evaluator"),
        }
        self.assertEqual({k: (v.code, v.scope) for k, v in load_registry().items()},
                         expected)
        with self.assertRaises(TypeError):
            CHANNELS["other"] = CHANNELS["x"]
        with self.assertRaises(FrozenInstanceError):
            CHANNELS["x"].code = 9

    def test_T1_registry_rejects_tampering(self):
        path = Path(__file__).resolve().parents[1] / "protocol/rng_registry.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        mutations = [
            lambda d: d["channels"][1].update(code=1),
            lambda d: d["channels"][0].update(name="unknown"),
            lambda d: d["channels"][0].update(scope="pair"),
            lambda d: d["channels"].pop(),
            lambda d: d.update(root_seed=20261030.0),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(extra="unregistered"),
            lambda d: d.update(bootstrap_identity=[20261030, 2026103001, 1, 0, 7, 0]),
            lambda d: d["entropy_fields"].reverse(),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                document = json.loads(json.dumps(original))
                mutate(document)
                with self.assertRaises(ValueError):
                    validate_registry(document)

    def test_T1_registry_rejects_duplicate_json_keys(self):
        from unittest.mock import patch
        with patch("star_v2.rng.Path.read_text",
                   return_value='{"schema_version":1,"schema_version":1}'):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_registry()

    def test_T9_protocol_record_and_readme_unchanged(self):
        directory = Path(__file__).resolve().parents[1]
        expected = {
            "protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md":
                "fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "protocol/freeze_record.json":
                "9690c343fb6adca9aecb5c702dc4965513eeb66fc5dd48d372b89e5bebd71dfe",
            "README.md":
                "0aedd480583ffc16cb0402b6f99f05e3daff29754f1b157c17002f8de8038b6a",
        }
        for rel, digest in expected.items():
            with self.subTest(file=rel):
                self.assertEqual(hashlib.sha256((directory / rel).read_bytes()).hexdigest(),
                                 digest)
        record = json.loads((directory / "protocol/freeze_record.json").read_text())
        self.assertEqual(record["protocol_sha256"], CONFIG.protocol_sha256)
        self.assertEqual(record["source_commit"], CONFIG.source_commit)

    def test_T9_old_and_new_assets_match_frozen_hashes(self):
        root = Path(__file__).resolve().parents[5]
        for checkout in (root, root.parent / "causalpfn-student"):
            code = checkout / "iPFN_demo/code/sparse_j"
            config = json.loads((code / "experiment.json").read_text())
            expected = dict(config["frozen_assets_sha256"])
            regional = json.loads((code / "regional_selected.json").read_text())
            expected["regional_inference.py"] = regional["inference_sha256"]
            for rel, digest in expected.items():
                with self.subTest(checkout=checkout.name, asset=rel):
                    self.assertEqual(hashlib.sha256((code / rel).read_bytes()).hexdigest(),
                                     digest)


if __name__ == "__main__":
    unittest.main()
