"""In-memory identity test vectors, never formal cohort data."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from star_v2.rng import RNGIdentity, bootstrap_identity, make_generator, make_identity


def identity(cohort=1, mechanism=0, channel="population", repeat=0):
    return make_identity(cohort_code=cohort, mechanism_id=mechanism,
                         channel=channel, pair_repeat_id=repeat)


class RNGTests(unittest.TestCase):
    def test_T2_same_identity_fresh_reproducible_generators(self):
        ident = identity()
        a, b = make_generator(ident), make_generator(ident)
        self.assertIsNot(a, b)
        self.assertIsNot(a.bit_generator, b.bit_generator)
        self.assertEqual(a.bit_generator.seed_seq.entropy,
                         [20261030, 2026103001, 1, 0, 1, 0])
        self.assertIsInstance(a.bit_generator, np.random.PCG64)
        np.testing.assert_array_equal(a.bit_generator.random_raw(8),
                                      b.bit_generator.random_raw(8))
        # Consuming a does not advance a newly constructed generator.
        a.bit_generator.random_raw(17)
        c = ident.generator()
        reference = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([20261030, 2026103001, 1, 0, 1, 0])))
        np.testing.assert_array_equal(c.bit_generator.random_raw(8),
                                      reference.bit_generator.random_raw(8))

    def test_T3_each_identity_coordinate_changes_derived_state(self):
        identities = [identity(channel="noise"), identity(cohort=2, channel="noise"),
                      identity(mechanism=1, channel="noise"),
                      identity(channel="noise", repeat=1),
                      identity(channel="leaf_assignment")]
        self.assertEqual(len({i.entropy for i in identities}), len(identities))
        states = [tuple(i.seed_sequence().generate_state(4).tolist()) for i in identities]
        self.assertEqual(len(set(states)), len(states))
        self.assertEqual(len({i.canonical_json() for i in identities}), len(identities))
        # This finite no-collision check is not a proof of stream independence.

    def test_T4_mechanism_scope_and_registered_bounds(self):
        for name in ("population", "x", "coefficient"):
            with self.subTest(channel=name):
                self.assertEqual(identity(channel=name).pair_repeat_id, 0)
                with self.assertRaises(ValueError):
                    identity(channel=name, repeat=1)
        for args in ({"mechanism": 5}, {"cohort": 2, "mechanism": 200},
                     {"channel": "noise", "repeat": 2},
                     {"cohort": 2, "channel": "noise", "repeat": 3}):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    identity(**args)

    def test_T4_pair_scope_without_center_identity(self):
        for name in ("leaf_assignment", "noise", "observation_index"):
            ident = identity(cohort=2, mechanism=199, channel=name, repeat=2)
            self.assertEqual(ident.entropy[-1], 2)
            np.testing.assert_array_equal(ident.generator().bit_generator.random_raw(4),
                                          ident.generator().bit_generator.random_raw(4))
        with self.assertRaises(TypeError):
            make_identity(cohort_code=1, mechanism_id=0, channel="noise",
                          pair_repeat_id=0, center_t=1)

    def test_T4_bootstrap_unique_global_identity(self):
        ident = bootstrap_identity()
        self.assertEqual(ident.entropy, (20261030, 2026103001, 2, 0, 7, 0))
        self.assertEqual(ident.to_metadata()["scope"], "global_evaluator")
        for cohort, mechanism, repeat in ((1, 0, 0), (2, 1, 0), (2, 0, 1)):
            with self.subTest(values=(cohort, mechanism, repeat)):
                with self.assertRaises(ValueError):
                    RNGIdentity(cohort, mechanism, "bootstrap", repeat)
        with self.assertRaises(ValueError):
            identity(cohort=2, channel="bootstrap")
        reference = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([20261030, 2026103001, 2, 0, 7, 0])))
        np.testing.assert_array_equal(ident.generator().bit_generator.random_raw(4),
                                      reference.bit_generator.random_raw(4))

    def test_T5_invalid_input_types_and_values(self):
        valid = dict(cohort_code=1, mechanism_id=0,
                     channel="population", pair_repeat_id=0)
        invalid = {
            "cohort_code": (True, 1.0, "1", None, -1, 0, 3),
            "mechanism_id": (False, 0.0, "0", None, -1, np.int64(0)),
            "pair_repeat_id": (False, 0.0, "0", None, -1),
            "channel": ("unknown", "", None, True, 1),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    args = {**valid, field: value}
                    with self.assertRaises((TypeError, ValueError)):
                        make_identity(**args)
                    with self.assertRaises((TypeError, ValueError)):
                        RNGIdentity(**args)
        with self.assertRaises(TypeError):
            make_generator([20261030, 2026103001, 1, 0, 1, 0])

    def test_T5_identity_is_immutable(self):
        ident = identity()
        with self.assertRaises(FrozenInstanceError):
            ident.mechanism_id = 1

    def test_T6_direct_numpy_reference_mechanism_pair_and_cohort(self):
        cases = [
            (dict(cohort=1, mechanism=0, channel="population", repeat=0),
             [20261030, 2026103001, 1, 0, 1, 0]),
            (dict(cohort=1, mechanism=4, channel="coefficient", repeat=0),
             [20261030, 2026103001, 1, 4, 3, 0]),
            (dict(cohort=2, mechanism=0, channel="population", repeat=0),
             [20261030, 2026103001, 2, 0, 1, 0]),
            (dict(cohort=2, mechanism=3, channel="noise", repeat=2),
             [20261030, 2026103001, 2, 3, 5, 2]),
        ]
        for args, entropy in cases:
            with self.subTest(entropy=entropy):
                reference = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence(entropy)))
                actual = identity(**args)
                self.assertEqual(actual.entropy, tuple(entropy))
                np.testing.assert_array_equal(actual.generator().bit_generator.random_raw(12),
                                              reference.bit_generator.random_raw(12))

    def test_T7_global_numpy_state_unchanged(self):
        before = np.random.get_state()
        identity().generator().bit_generator.random_raw(4)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_T7_no_global_python_time_or_os_entropy_sources(self):
        forbidden = AssertionError("unregistered randomness or clock was consulted")
        with patch.object(np.random, "seed", side_effect=forbidden), \
             patch.object(np.random, "default_rng", side_effect=forbidden), \
             patch.object(random, "seed", side_effect=forbidden), \
             patch.object(random, "random", side_effect=forbidden), \
             patch.object(random, "getrandbits", side_effect=forbidden), \
             patch.object(time, "time", side_effect=forbidden), \
             patch.object(time, "time_ns", side_effect=forbidden), \
             patch.object(os, "urandom", side_effect=forbidden):
            ident = identity(channel="x")
            ident.generator().bit_generator.random_raw(4)
            ident.to_metadata()

    def test_T8_metadata_canonical_serialization_and_hash(self):
        ident = identity(cohort=2, mechanism=3, channel="noise", repeat=2)
        metadata = ident.to_metadata()
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertEqual(metadata["entropy"], [20261030, 2026103001, 2, 3, 5, 2])
        self.assertEqual(metadata["seed_sequence"], {
            "entropy": [20261030, 2026103001, 2, 3, 5, 2],
            "spawn_key": [], "pool_size": 4, "n_children_spawned": 0,
        })
        self.assertEqual(metadata["numpy_version"], np.__version__)
        parsed = json.loads(metadata["canonical_identity"])
        self.assertEqual(metadata["canonical_identity"],
                         json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False))
        self.assertEqual(metadata["identity_sha256"],
                         hashlib.sha256(metadata["canonical_identity"].encode("ascii")).hexdigest())
        self.assertEqual(metadata, ident.to_metadata())
        self.assertEqual(set(metadata), {
            "protocol_id", "root_seed", "protocol_code", "cohort_code",
            "mechanism_id", "pair_repeat_id", "channel_name", "channel_code",
            "scope", "entropy_fields", "entropy", "seed_sequence",
            "bit_generator", "numpy_version", "canonical_identity", "identity_sha256",
        })
        metadata["entropy"][0] = 0
        self.assertEqual(ident.entropy[0], 20261030)

    def test_T8_canonical_identity_across_processes_and_hash_seeds(self):
        code_root = str(Path(__file__).resolve().parents[2])
        command = (
            "import sys; sys.path.insert(0, " + repr(code_root) + "); "
            "from star_v2.rng import make_identity; "
            "i=make_identity(cohort_code=2,mechanism_id=3,channel='noise',pair_repeat_id=2); "
            "print(i.canonical_json()); print(i.identity_sha256)"
        )
        expected = identity(cohort=2, mechanism=3, channel="noise", repeat=2)
        outputs = []
        for hashseed in ("1", "987654"):
            outputs.append(subprocess.check_output(
                [sys.executable, "-B", "-c", command],
                env={**os.environ, "PYTHONHASHSEED": hashseed,
                     "PYTHONDONTWRITEBYTECODE": "1"}, text=True).splitlines())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], [expected.canonical_json(), expected.identity_sha256])

    def test_T8_representation_seed_not_in_six_field_entropy(self):
        ident = identity()
        self.assertEqual(len(ident.entropy), 6)
        self.assertNotIn(2026101604, ident.entropy)


if __name__ == "__main__":
    unittest.main()
