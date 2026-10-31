"""Pair-only synthetic fixtures and mock RNG; no formal cohort draws or files."""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np

from star_v2 import dgp, pair
from star_v2.rng import make_identity

ATOL = 1e-12  # Fixture absolute tolerance; not the Oracle constraint threshold.
ERRORS = {}
TEST_NAMESPACE = 99887766


def readonly(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def mechanism_fixture(cohort=2, mid=3):
    """Artificial arrays, never a call to production mechanism generation."""
    a = dgp.star_adjacency()
    x = np.linspace(-0.75, 0.65, 4000, dtype=np.float32).reshape(200, 20)
    h = np.linspace(-0.5, 0.7, 600, dtype=np.float64).reshape(200, 3)
    beta = np.linspace(-0.23, 0.57, 25, dtype=np.float64)
    beta[0], beta[11], beta[15] = 12.123456789, 0.75, 0.35
    components = dgp.structural_components(a, h, beta)
    mu = dgp.four_state_truth(a, h, beta)
    ite, ate = dgp.effects_from_mu(mu)
    ids = tuple(make_identity(cohort_code=cohort, mechanism_id=mid,
                              channel=name, pair_repeat_id=0)
                for name in ("population", "x", "coefficient"))
    arrays = (np.arange(200, dtype=np.int64), np.zeros(20), x, a,
              np.eye(20, 2), h, beta, components, mu, ite, ate)
    return dgp.SimulatorMechanism(cohort, mid, ids, *(readonly(v) for v in arrays))


def observation_fixture(include_center=True):
    head = [137, 0, 198] if include_center else [137, 199, 198]
    return np.array(head + list(range(1, 48)), dtype=np.int64)


@contextmanager
def scripted_rng(include_center=True, *, leaf=None, noise=None, indices=None):
    """Exactly specified arrays make sampler call arguments independently testable."""
    if leaf is None:
        leaf = (np.arange(199) % 3 == 0).astype(np.int64)
    if noise is None:
        noise = np.linspace(-0.75, 0.95, 200, dtype=np.float64) + np.arange(200)*1e-10
    if indices is None:
        indices = observation_fixture(include_center)
    mocks = {
        "leaf_assignment": Mock(spec=["binomial"]),
        "noise": Mock(spec=["normal"]),
        "observation_index": Mock(spec=["choice"]),
    }
    mocks["leaf_assignment"].binomial.return_value = leaf
    mocks["noise"].normal.return_value = noise
    mocks["observation_index"].choice.return_value = indices
    seen = []
    def factory(identity):
        if identity.channel not in mocks:
            raise AssertionError("Mechanism/global stream requested by pair generation")
        seen.append(identity)
        return mocks[identity.channel]
    with patch.object(pair, "make_generator", side_effect=factory):
        yield mocks, seen


@contextmanager
def separate_test_rng():
    """The only random pair fixtures use a namespace outside all formal entropy."""
    seen = []
    def factory(identity):
        if identity.channel not in ("leaf_assignment", "noise", "observation_index"):
            raise AssertionError("Unexpected channel")
        seen.append(identity)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [TEST_NAMESPACE, identity.entropy[4], identity.pair_repeat_id])))
    with patch.object(pair, "make_generator", side_effect=factory):
        yield seen


def fixture_pair(include_center=True):
    m = mechanism_fixture()
    with scripted_rng(include_center):
        return pair.generate_pair(m, pair_repeat_id=2)


def independent_mean(mechanism, treatment):
    """Scalar neighbor sums and the raw 25-column structural design."""
    a, h, beta = mechanism.adjacency, mechanism.h, mechanism.coefficients
    result = np.empty(200, dtype=np.float64)
    for i in range(200):
        neighbors = [k for k in range(200) if a[i, k] == 1]
        hb = [sum(float(h[k, q]) for k in neighbors)/len(neighbors) for q in range(3)]
        e = sum(float(treatment[k]) for k in neighbors)/len(neighbors)
        j = [sum(float(treatment[k])*float(h[k, q]) for k in neighbors)
             /len(neighbors) for q in range(3)]
        own = float(treatment[i])
        hetero = [1.0, *map(float, h[i])]
        row = [*hetero, *hb, *[own*v for v in hetero], *[e*v for v in hetero],
               *[own*e*v for v in hetero], *j, *[own*v for v in j]]
        assert len(row) == 25
        result[i] = sum(v*float(b) for v, b in zip(row, beta))
    return result


def context_changed(p, index=0, **changes):
    contexts = list(p.contexts)
    contexts[index] = replace(contexts[index], **changes)
    return replace(p, contexts=tuple(contexts))


def common_changed(p, name, value):
    return replace(p, **{name: value},
                   contexts=tuple(replace(c, **{name: value}) for c in p.contexts))


class PairTests(unittest.TestCase):
    def check_error(self, label, actual, expected):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        ERRORS[label] = max(ERRORS.get(label, 0.0), error)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)

    def test_T1_two_contexts_treatments_and_shared_mechanism(self):
        p = fixture_pair()
        self.assertEqual(len(p.contexts), 2)
        left, right = p.contexts
        self.assertEqual((left.center_t, right.center_t), (0, 1))
        for c in p.contexts:
            self.assertEqual(c.treatment.shape, (200,))
            self.assertEqual(c.treatment.dtype, np.float64)
            self.assertIs(c.mechanism, p.mechanism)
            for name in ("x", "adjacency", "h", "coefficients", "mu"):
                self.assertIs(getattr(c.mechanism, name), getattr(p.mechanism, name))
            np.testing.assert_array_equal(c.treatment[1:], p.leaf_treatment)
        np.testing.assert_array_equal(np.flatnonzero(left.treatment != right.treatment), [0])
        np.testing.assert_array_equal(p.leaf_treatment, (np.arange(199) % 3 == 0))

    def test_T1_node_state_not_repeated_across_pair_arms(self):
        p = fixture_pair()
        codes = []
        for c in p.contexts:
            _, s = dgp.exposure(p.mechanism.adjacency, c.treatment)
            codes.append(2*c.treatment.astype(int)+s)
        self.assertTrue(np.all(codes[0] != codes[1]))

    def test_T2_exact_three_pair_identities_and_single_draws(self):
        m = mechanism_fixture()
        with scripted_rng() as (mocks, seen), \
             patch.object(dgp, "generate_star_mechanism", side_effect=AssertionError("mechanism")), \
             patch.object(dgp, "make_generator", side_effect=AssertionError("mechanism RNG")), \
             patch.object(dgp, "common_projection", side_effect=AssertionError("projection")):
            p = pair.generate_pair(m, pair_repeat_id=2)
        self.assertEqual([i.entropy for i in seen], [
            (20261030, 2026103001, 2, 3, 4, 2),
            (20261030, 2026103001, 2, 3, 5, 2),
            (20261030, 2026103001, 2, 3, 6, 2)])
        self.assertEqual(p.rng_identities, tuple(seen))
        mocks["leaf_assignment"].binomial.assert_called_once_with(1, 0.5, size=199)
        mocks["noise"].normal.assert_called_once_with(0.0, 1.0, size=200)
        mocks["observation_index"].choice.assert_called_once_with(200, size=50, replace=False)

    def test_T3_unbalanced_leaf_assignments_are_not_resampled(self):
        for value in (0, 1):
            with self.subTest(value=value), scripted_rng(
                    leaf=np.full(199, value, dtype=np.int64)) as (mocks, seen):
                p = pair.generate_pair(mechanism_fixture(), pair_repeat_id=0)
                np.testing.assert_array_equal(p.leaf_treatment, value)
                self.assertEqual(len(seen), 3)
                mocks["leaf_assignment"].binomial.assert_called_once_with(1, 0.5, size=199)

    def test_T4_one_shared_float64_noise_and_residuals(self):
        noise = np.linspace(-0.75, 0.95, 200) + np.arange(200)*1e-10
        with scripted_rng(noise=noise) as (mocks, _):
            p = pair.generate_pair(mechanism_fixture(), pair_repeat_id=1)
        for c in p.contexts:
            self.assertIs(c.noise, p.noise)
            self.assertEqual(c.noise.dtype, np.float64)
            np.testing.assert_array_equal(c.noise, noise)
            self.check_error("factual_minus_mean_vs_noise",
                             c.full_outcomes-c.structural_mean, noise)
        self.assertEqual(p.noise.shape, (200,))
        mocks["noise"].normal.assert_called_once()

    def test_T5_order_and_center_inclusion_exclusion(self):
        for include in (False, True):
            with self.subTest(include=include):
                p = fixture_pair(include)
                self.assertEqual(0 in p.observed_indices, include)
                np.testing.assert_array_equal(p.observed_indices, observation_fixture(include))
                self.assertFalse(np.array_equal(p.observed_indices, np.sort(p.observed_indices)))
                self.assertEqual(len(np.unique(p.observed_indices)), 50)
                self.assertGreaterEqual(p.observed_indices.min(), 0)
                self.assertLess(p.observed_indices.max(), 200)
                for c in p.contexts:
                    self.assertIs(c.observed_indices, p.observed_indices)
                    np.testing.assert_array_equal(c.outcomes_observed,
                                                  c.full_outcomes[p.observed_indices])
                self.assertEqual(pair.validate_pair(p), {
                    "status": "OK", "n_contexts": 2,
                    "n_unique_observed_nodes": 50, "n_factual_records": 100})

    def test_T6_float64_pipeline_detects_float32_quantization(self):
        m = mechanism_fixture()
        original_hashes = m.metadata()["array_sha256"]
        p = fixture_pair()
        for c in p.contexts:
            reference = independent_mean(p.mechanism, c.treatment)
            for name in ("structural_mean", "full_outcomes", "outcomes_observed"):
                self.assertEqual(getattr(c, name).dtype, np.float64)
            self.check_error("independent_structural_mean", c.structural_mean, reference)
            self.check_error("independent_factual_outcome", c.full_outcomes, reference+p.noise)
            self.assertGreater(float(np.max(np.abs(
                c.full_outcomes.astype(np.float32).astype(np.float64)-c.full_outcomes))), ATOL)
            self.assertFalse(np.shares_memory(c.outcomes_observed, c.full_outcomes))
        self.assertEqual(original_hashes, m.metadata()["array_sha256"])

    def test_T6_noise_addition_does_not_overwrite_inputs(self):
        m = mechanism_fixture()
        before = m.metadata()["array_sha256"]
        leaf = (np.arange(199) % 3 == 0).astype(np.int64)
        noise = np.linspace(-0.7, 0.9, 200)
        ids = observation_fixture()
        original = [a.copy() for a in (leaf, noise, ids)]
        with scripted_rng(leaf=leaf, noise=noise, indices=ids):
            p = pair.generate_pair(m, pair_repeat_id=0)
        self.assertIs(p.mechanism, m)
        self.assertEqual(before, m.metadata()["array_sha256"])
        for a,b in zip((leaf, noise, ids), original):
            np.testing.assert_array_equal(a,b)

    def test_T7_shared_noise_full_and_observed_difference(self):
        for include in (False, True):
            p = fixture_pair(include)
            a,b = p.contexts
            reference_delta = independent_mean(p.mechanism,b.treatment)-independent_mean(
                p.mechanism,a.treatment)
            observed_delta = b.outcomes_observed-a.outcomes_observed
            self.check_error("full_difference_vs_independent", b.full_outcomes-a.full_outcomes,
                             reference_delta)
            self.check_error("observed_difference_vs_independent", observed_delta,
                             reference_delta[p.observed_indices])
            residual = (b.full_outcomes-a.full_outcomes)-(b.structural_mean-a.structural_mean)
            ERRORS["full_shared_noise_cancellation"] = max(
                ERRORS.get("full_shared_noise_cancellation",0.0),float(np.max(np.abs(residual))))
            ERRORS["observed_shared_noise_cancellation"] = max(
                ERRORS.get("observed_shared_noise_cancellation",0.0),
                float(np.max(np.abs(residual[p.observed_indices]))))
            self.assertLessEqual(np.max(np.abs(residual)), ATOL)
            diagnostics = p.difference_diagnostics()
            self.assertEqual(diagnostics["full_max_abs_residual"],
                             float(np.max(np.abs(residual))))
            self.assertEqual(diagnostics["observed_max_abs_residual"],
                             float(np.max(np.abs(residual[p.observed_indices]))))

    def test_T8_center_change_updates_leaf_exposure_and_outcomes(self):
        p = fixture_pair()
        a,b = p.contexts
        e0,s0 = dgp.exposure(p.mechanism.adjacency,a.treatment)
        e1,s1 = dgp.exposure(p.mechanism.adjacency,b.treatment)
        np.testing.assert_array_equal(a.treatment[1:],b.treatment[1:])
        np.testing.assert_array_equal(e0[1:],0)
        np.testing.assert_array_equal(e1[1:],1)
        np.testing.assert_array_equal(s0[1:],0)
        np.testing.assert_array_equal(s1[1:],1)
        self.assertTrue(np.any(a.structural_mean[1:] != b.structural_mean[1:]))
        self.assertTrue(np.any(a.full_outcomes[1:] != b.full_outcomes[1:]))

    def test_T9_separate_test_rng_reproducibility_and_repeat_ids(self):
        m = mechanism_fixture()
        with separate_test_rng():
            a = pair.generate_pair(m,pair_repeat_id=0)
            b = pair.generate_pair(m,pair_repeat_id=0)
            c = pair.generate_pair(m,pair_repeat_id=1)
        self.assertEqual(a.metadata(),b.metadata())
        for x,y in zip(a.contexts,b.contexts):
            np.testing.assert_array_equal(x.full_outcomes,y.full_outcomes)
        self.assertNotEqual(a.rng_identities[0].entropy,c.rng_identities[0].entropy)
        self.assertFalse(np.array_equal(a.noise,c.noise))
        # Finite differing outputs do not prove mathematical independence.

    def test_T9_global_states_and_configuration_unchanged(self):
        m = mechanism_fixture()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        before = m.metadata()
        with separate_test_rng(), \
             patch.object(np.random,"seed",side_effect=AssertionError("global RNG")), \
             patch.object(np.random,"default_rng",side_effect=AssertionError("global RNG")):
            pair.generate_pair(m,pair_repeat_id=0)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0],numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1],numpy_after[1])
        self.assertEqual(numpy_before[2:],numpy_after[2:])
        self.assertEqual(python_before,random.getstate())
        self.assertEqual(before,m.metadata())
        self.assertEqual(pair.CONFIG.N,200)
        self.assertEqual(pair.CONFIG.M,50)

    def test_T10_reject_invalid_request_before_any_draw(self):
        m = mechanism_fixture()
        cases = [
            (None,0), (replace(m,cohort_code=0),0),
            (replace(m,mechanism_id=-1),0), (replace(m,mechanism_id=200),0),
            (replace(m,mechanism_id=True),0),
            (replace(m,x=readonly(m.x[:199])),0),
            (replace(m,h=readonly(m.h.astype(np.float32))),0),
            (replace(m,node_ids=readonly(m.node_ids[::-1])),0),
            (m,-1),(m,3),(m,0.0),(m,False),
        ]
        for mechanism,repeat in cases:
            with self.subTest(repeat=repeat), \
                 patch.object(pair,"make_generator",side_effect=AssertionError("draw")) as factory:
                with self.assertRaises(pair.PairValidationError):
                    pair.generate_pair(mechanism,pair_repeat_id=repeat)
                factory.assert_not_called()
        pilot = mechanism_fixture(cohort=1,mid=0)
        with patch.object(pair,"make_generator",side_effect=AssertionError("draw")):
            with self.assertRaises(pair.PairValidationError):
                pair.generate_pair(pilot,pair_repeat_id=2)

    def test_T10_invalid_sampler_output_preserves_evidence_without_retry(self):
        leaf = np.full(199,2,dtype=np.int64)
        with scripted_rng(leaf=leaf) as (mocks,seen):
            with self.assertRaises(pair.PairValidationError) as caught:
                pair.generate_pair(mechanism_fixture(),pair_repeat_id=0)
            self.assertIs(caught.exception.evidence["leaf_treatment"],leaf)
            self.assertEqual(len(seen),1)
            mocks["leaf_assignment"].binomial.assert_called_once()
            mocks["noise"].normal.assert_not_called()
        noise = np.ones(200,dtype=np.float32)
        with scripted_rng(noise=noise) as (mocks,seen):
            with self.assertRaises(pair.PairValidationError) as caught:
                pair.generate_pair(mechanism_fixture(),pair_repeat_id=0)
            self.assertIs(caught.exception.evidence["noise"],noise)
            self.assertEqual(caught.exception.code,"DTYPE")
            self.assertEqual(len(seen),2)
            mocks["observation_index"].choice.assert_not_called()
        ids = np.zeros(50,dtype=np.int64)
        with scripted_rng(indices=ids) as (mocks,seen):
            with self.assertRaises(pair.PairValidationError) as caught:
                pair.generate_pair(mechanism_fixture(),pair_repeat_id=0)
            self.assertIs(caught.exception.evidence["observed_indices"],ids)
            self.assertEqual(len(seen),3)
            mocks["observation_index"].choice.assert_called_once()

    def test_T10_reject_non_float64_structural_output(self):
        with scripted_rng(), patch.object(dgp.SimulatorMechanism,"structural_mean",
                                          return_value=np.ones(200,dtype=np.float32)):
            with self.assertRaises(pair.PairValidationError) as caught:
                pair.generate_pair(mechanism_fixture(),pair_repeat_id=0)
            self.assertEqual(caught.exception.code,"DTYPE")
            self.assertEqual(caught.exception.evidence["mean0"].dtype,np.float32)

    def test_T10_reject_tampered_sharing_and_context_identity(self):
        p = fixture_pair()
        invalid = [
            replace(p,contexts=p.contexts[:1]),
            replace(p,contexts=p.contexts[::-1]),
            replace(p,mechanism_id=4),
            replace(p,cohort_code=1),
            context_changed(p,center_t=1),
            context_changed(p,noise=readonly(p.noise.copy())),
            context_changed(p,leaf_treatment=readonly(p.leaf_treatment.copy())),
            context_changed(p,observed_indices=readonly(p.observed_indices.copy())),
            context_changed(p,mechanism=replace(p.mechanism)),
        ]
        for candidate in invalid:
            with self.subTest(candidate_id=id(candidate)), \
                 patch.object(pair,"make_generator",side_effect=AssertionError("repair")):
                with self.assertRaises(pair.PairValidationError) as caught:
                    pair.validate_pair(candidate)
                self.assertIs(caught.exception.evidence,candidate)

    def test_T10_reject_invalid_arrays_no_sort_or_overwrite(self):
        p = fixture_pair()
        t = p.contexts[0].treatment.copy(); t[5]=2
        ids_high = p.observed_indices.copy(); ids_high[3]=200
        ids_dup = p.observed_indices.copy(); ids_dup[3]=ids_dup[4]
        nan_y = p.contexts[0].full_outcomes.copy(); nan_y[1]=np.nan
        reversed_ids = readonly(p.observed_indices[::-1])
        invalid = [
            context_changed(p,treatment=readonly(t)),
            context_changed(p,treatment=readonly(p.contexts[0].treatment[:199])),
            context_changed(p,full_outcomes=readonly(nan_y)),
            context_changed(p,full_outcomes=readonly(p.contexts[0].full_outcomes.astype(np.float32))),
            context_changed(p,outcomes_observed=readonly(
                p.contexts[0].outcomes_observed.astype(np.float32))),
            context_changed(p,outcomes_observed=readonly(p.contexts[0].outcomes_observed[::-1])),
            context_changed(p,full_outcomes=readonly(p.contexts[0].full_outcomes+0.01)),
            context_changed(p,structural_mean=readonly(p.contexts[0].structural_mean+0.01)),
            context_changed(p,full_outcomes=p.contexts[0].full_outcomes.copy()),
            common_changed(p,"observed_indices",readonly(ids_high)),
            common_changed(p,"observed_indices",readonly(ids_dup)),
            common_changed(p,"observed_indices",readonly(p.observed_indices.astype(float))),
            common_changed(p,"observed_indices",reversed_ids),
            common_changed(p,"noise",readonly(p.noise.astype(np.float32))),
            common_changed(p,"noise",readonly(np.full(200,np.inf))),
        ]
        for candidate in invalid:
            before = candidate.observed_indices.copy()
            with self.subTest(candidate_id=id(candidate)), \
                 patch.object(pair,"make_generator",side_effect=AssertionError("repair")):
                with self.assertRaises(pair.PairValidationError) as caught:
                    pair.validate_pair(candidate)
                self.assertIs(caught.exception.evidence,candidate)
                np.testing.assert_array_equal(candidate.observed_indices,before)
        np.testing.assert_array_equal(p.observed_indices,observation_fixture())

    def test_T10_private_records_and_arrays_immutable(self):
        p = fixture_pair()
        with self.assertRaises(FrozenInstanceError):
            p.pair_repeat_id=1
        with self.assertRaises(TypeError):
            dict(p)
        self.assertFalse(hasattr(p,"predictor_input"))
        for value in (p.noise,p.leaf_treatment,p.observed_indices,
                      p.contexts[0].full_outcomes,p.contexts[1].outcomes_observed):
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_T12_private_metadata_shapes_hashes_and_sampler_rules(self):
        p = fixture_pair()
        metadata = p.metadata()
        self.assertEqual(json.loads(json.dumps(metadata)),metadata)
        self.assertEqual(metadata["access"],"simulator_evaluator_only")
        self.assertEqual(metadata["sampling_apis"],{
            "leaf_assignment":"Generator.binomial(1, 0.5, size=199)",
            "noise":"Generator.normal(0.0, noise_sd, size=200)",
            "observation_index":"Generator.choice(200, size=50, replace=False)"})
        self.assertEqual(metadata["observation_order"],"Generator.choice return order; never sorted")
        self.assertEqual(metadata["array_dtypes"]["noise"],"<f8")
        self.assertEqual(metadata["array_shapes"]["center_t0.outcomes_observed"],[50])
        self.assertEqual(metadata["array_sha256"]["noise"],dgp.array_sha256(p.noise))

    def test_T12_existing_files_and_no_generated_artifacts(self):
        directory = Path(__file__).resolve().parents[1]
        expected = {
            "protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md":"fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "protocol/freeze_record.json":"9690c343fb6adca9aecb5c702dc4965513eeb66fc5dd48d372b89e5bebd71dfe",
            "protocol/rng_registry.json":"d0ea7726b03824036d0a6a74519c6480c37a29a23734f587177a59c94faecdd3",
            "config.py":"e0c16d12415f4ac1f22b12f02ae7fac680c9aa173cf3b8fe91564d146da25baf",
            "rng.py":"cf2e81947f08ca4be5bcd9d5583b1ec0580a1e7b55f6b7090d93b3974fe356e9",
            "dgp.py":"3509ba90b8278eb9fad142646001e450998088786a0326a869d7b2ad33be20c9"}
        for rel,digest in expected.items():
            self.assertEqual(hashlib.sha256((directory/rel).read_bytes()).hexdigest(),digest)
        self.assertFalse(list(directory.rglob("__pycache__")))
        self.assertFalse((directory.parents[2]/"results/e1_5_star_v2").exists())


def tearDownModule():
    print("S2.4_REFERENCE_MAX_ABS_ERRORS",json.dumps(ERRORS,sort_keys=True))


if __name__=="__main__":
    unittest.main()
