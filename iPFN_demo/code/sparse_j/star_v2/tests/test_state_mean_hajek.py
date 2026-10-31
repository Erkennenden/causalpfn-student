"""State Mean: only deterministic artificial observations, no formal RNG."""
from dataclasses import replace
import hashlib
import math
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import baselines_state_ht as b
from star_v2 import contracts as c
from star_v2 import pair as pair_module
from star_v2.config import CONFIG
from star_v2.estimation import EstimationRequest
from test_pair_data import mechanism_fixture, scripted_rng, fixture_pair


def req(task, **kw):
    return EstimationRequest.from_task(
        task, status=kw.pop("status", "POINT"), target_id=CONFIG.target_id,
        request_id="artificial-s32-request", requested_outputs=kw.pop(
            "requested_outputs", ("mu", "ite", "ate")), **kw)


def missing_fixture(missing=None):
    if missing is None:
        return c.pair_from_private(fixture_pair())
    # Sample center plus 49 leaves. Majority determined by all 199 leaves.
    leaves = np.zeros(199, dtype=np.int64)
    if missing in ("01", "00"):
        leaves[:49] = 1
    if missing in ("00", "10"):
        leaves[49:] = 1
    indices = np.arange(50, dtype=np.int64)
    with scripted_rng(leaf=leaves, indices=indices):
        p = pair_module.generate_pair(mechanism_fixture(), pair_repeat_id=2)
    return c.pair_from_private(p)


def hand_records(task):
    """Scalar state labels and fsum; no baseline state encoding/grouping code."""
    groups = {state: [] for state in ("00", "01", "10", "11")}
    for context in task.context_refs:
        treatment = context.treatment.tolist()
        for node, outcome in zip(context.observed_indices, context.outcomes_observed):
            exposure = (sum(treatment[1:])/199 > 0.5) if node == 0 else treatment[0] == 1
            label = str(int(treatment[node])) + str(int(exposure))
            groups[label].append(float(outcome))
    counts = [len(groups[s]) for s in groups]
    means = [math.fsum(groups[s])/len(groups[s]) if groups[s] else math.nan for s in groups]
    return groups, np.asarray(counts), np.asarray(means)


class StateMeanTests(unittest.TestCase):
    def test_hand_reference_all_states_and_hajek(self):
        task = missing_fixture()
        result = b.estimate_state_mean(req(task), task)
        _, counts, means = hand_records(task)
        self.assertEqual(result.execution_status, "OK")
        for k, state in enumerate(b.ARM_ORDER):
            info = result.diagnostics["states"][state]
            self.assertEqual(info["count"], counts[k])
            self.assertAlmostEqual(info["value"], means[k], delta=1e-12)
            self.assertAlmostEqual(info["hajek_diagnostic"], means[k], delta=1e-12)
        expected = [means[2]-means[0], means[3]-means[2], means[3]-means[0]]
        np.testing.assert_allclose(result.ate, expected, rtol=0, atol=1e-12)

    def test_missing_01_has_three_valid_effects(self):
        task = missing_fixture("01")
        result = b.estimate_state_mean(req(task), task)
        self.assertEqual(result.execution_status, "OK")
        self.assertEqual(result.output_status["ate"], ("OK",)*3)
        info = result.diagnostics["states"]["01"]
        self.assertEqual(info["count"], 0)
        self.assertIsNone(info["value"])
        self.assertIsNone(info["hajek_diagnostic"])
        self.assertEqual(info["status"], "NOT_ESTIMABLE")
        self.assertFalse(info["empirical_supported"])
        self.assertIsNone(result.failure_type)

    def test_missing_00_only_dependent_effects_fail(self):
        self.check_missing("00", ("NOT_ESTIMABLE", "OK", "NOT_ESTIMABLE"))

    def test_missing_10_only_dependent_effects_fail(self):
        self.check_missing("10", ("NOT_ESTIMABLE", "NOT_ESTIMABLE", "OK"))

    def test_missing_11_only_dependent_effects_fail(self):
        self.check_missing("11", ("OK", "NOT_ESTIMABLE", "NOT_ESTIMABLE"))

    def check_missing(self, state, statuses):
        task = missing_fixture(state)
        result = b.estimate_state_mean(req(task), task)
        _, counts, means = hand_records(task)
        self.assertEqual(counts[b.ARM_ORDER.index(state)], 0)
        self.assertEqual(result.execution_status, "NOT_ESTIMABLE")
        self.assertEqual(result.output_status["ate"], statuses)
        self.assertEqual(result.failure_type, "EMPTY_REQUIRED_STATE")
        for k, (positive, negative) in enumerate(((2, 0), (3, 2), (3, 0))):
            if statuses[k] == "OK":
                self.assertAlmostEqual(result.ate[k], means[positive]-means[negative], delta=1e-12)
            else:
                self.assertTrue(np.isnan(result.ate[k]))
                reason = result.estimability[b.EFFECT_ORDER[k]]["reason"]
                self.assertIn(state, reason)
                self.assertEqual(reason, result.failure_provenance["effects"][b.EFFECT_ORDER[k]])

    def test_support_counts_reasons_and_no_independent_arms(self):
        task = missing_fixture("01")
        result = b.estimate_state_mean(req(task), task)
        d = result.diagnostics
        self.assertEqual(sum(s["count"] for s in d["states"].values()), 100)
        self.assertEqual((d["n_factual_records"], d["n_unique_observed_nodes"]), (100, 50))
        self.assertFalse(d["arms_independent"])
        self.assertEqual(d["statistical_unit"], "pair")
        self.assertEqual(result.arm_results, (None, None))
        for k, support in enumerate(result.support.context_supports):
            np.testing.assert_array_equal(np.asarray(d["full_graph_count"][k]).reshape(2,2,2),
                                          support.full_graph_count)
            np.testing.assert_array_equal(np.asarray(d["observed_count"][k]).reshape(2,2,2),
                                          support.observed_count)
            reasons = [int(np.count_nonzero(support.support_reason == reason))
                       for reason in d["support_reason_order"]]
            self.assertEqual(tuple(reasons), d["support_reason_counts"][k])
            self.assertEqual(sum(reasons), 800)

    def test_identity_and_ate_only_contract(self):
        task = missing_fixture()
        request = req(task)
        result = b.estimate_state_mean(request, task)
        self.assertEqual((result.status, result.target_id, result.request_id),
                         (request.status, request.target_id, request.request_id))
        self.assertEqual(result.pair_id, task.pair_id)
        self.assertEqual(result.context_ids, task.context_ids)
        self.assertEqual(result.input_hashes, tuple(t.input_hash for t in task.context_refs))
        self.assertEqual(result.output_status["mu"], "UNSUPPORTED")
        self.assertEqual(result.output_status["ite"], ("UNSUPPORTED",)*3)
        self.assertTrue(np.isnan(result.mu).all())
        self.assertTrue(np.isnan(result.ite).all())
        self.assertEqual(result.effect_order, ("Direct", "Spillover", "Total"))

    def test_not_identified_blocks_computation_and_retains_cause(self):
        task = missing_fixture()
        request = req(task, status="NOT_IDENTIFIED", failure_type="UPSTREAM_RULE",
                      failure_message="keep exact reason", failure_provenance={"code": "R17"})
        with patch.object(b, "_compute", side_effect=AssertionError("must not compute")) as core:
            result = b.estimate_state_mean(request, task)
        core.assert_not_called()
        self.assertEqual((result.status, result.execution_status), ("NOT_IDENTIFIED",)*2)
        self.assertEqual(result.failure_message, request.failure_message)
        self.assertEqual(dict(result.failure_provenance), dict(request.failure_provenance))
        self.assertTrue(np.isnan(result.ate).all())

    def test_readonly_original_data_hashes_and_float64_unchanged(self):
        task = missing_fixture()
        arrays = [getattr(t, name) for t in task.context_refs for name in c.FACTUAL_KEYS]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        hashes = tuple(t.input_hash for t in task.context_refs)
        result = b.estimate_state_mean(req(task), task)
        self.assertEqual(before, [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])
        self.assertEqual(hashes, result.input_hashes)
        for t in task.context_refs:
            self.assertEqual(t.outcomes_observed.dtype, np.float64)
            c.validate_task(t)
        self.assertFalse(result.ate.flags.writeable)
        with self.assertRaises(TypeError):
            result.diagnostics["states"]["00"]["count"] = 999

    def test_five_array_boundary_and_privileged_injection_rejected(self):
        task = missing_fixture()
        inputs = tuple(c.predictor_inputs(t) for t in task.context_refs)
        with patch.object(b, "_public_records", wraps=b._public_records) as parser:
            b.estimate_state_mean(req(task), task)
        seen = parser.call_args.args[0]
        for payload in seen:
            self.assertEqual(tuple(payload), c.FACTUAL_KEYS)
        for forbidden in ("truth", "beta", "h", "full_y", "noise", "rng_identity",
                          "input_hash", "support"):
            changed = dict(inputs[0])
            changed[forbidden] = np.zeros(200)
            with self.assertRaises(c.ContractError):
                b._public_records((changed, inputs[1]), b.StarPairDesign(200,50))

    def test_request_and_task_conflicts_rejected_before_computation(self):
        task = missing_fixture()
        request = req(task)
        with patch.object(b, "_compute", side_effect=AssertionError("not called")) as core:
            with self.assertRaises(c.ContractError):
                b.estimate_state_mean(replace(request, input_hashes=("0"*64,)*2), task)
            with self.assertRaises(c.ContractError):
                b.estimate_state_mean(request, task.context_refs[0])
        core.assert_not_called()

    def test_no_ate_request_is_not_an_implicit_request(self):
        task = missing_fixture()
        with self.assertRaises(c.ContractError):
            b.estimate_state_mean(req(task, requested_outputs=("mu",)), task)

    def test_core_small_fixture_empty_state_and_hajek(self):
        codes = np.array([[0, 2], [2, 3]], dtype=np.int64)
        y = np.array([[1., 4.], [5., 9.]], dtype=np.float64)
        r = b.state_mean_core(codes, y, np.array([0,1]), b.StarPairDesign(4,2))
        np.testing.assert_array_equal(r.counts, [1,0,2,1])
        np.testing.assert_allclose(r.state_values[[0,2,3]], [1.,4.5,9.], atol=0, rtol=0)
        np.testing.assert_allclose(r.hajek_diagnostic[[0,2,3]], [1.,4.5,9.], atol=0, rtol=0)
        self.assertTrue(np.isnan(r.state_values[1]))
        self.assertEqual(r.ate_status, ("OK",)*3)
        self.assertFalse(r.state_values.flags.writeable)

    def test_invalid_raw_shapes_dtypes_sharing_and_topology(self):
        task = missing_fixture()
        original = tuple(dict(c.predictor_inputs(t)) for t in task.context_refs)
        mutations = [
            ("outcomes_observed", original[0]["outcomes_observed"].astype(np.float32)),
            ("x", np.zeros((200,19), dtype=np.float32)),
            ("observed_indices", np.zeros(50, dtype=np.int64)),
            ("adjacency", np.zeros((200,200))),
            ("treatment", original[1]["treatment"].copy())]
        for name, value in mutations:
            payload = dict(original[0]); payload[name] = value
            with self.subTest(name=name), self.assertRaises(c.ContractError):
                b._public_records((payload, original[1]), b.StarPairDesign(200,50))
        changed = dict(original[1])
        ids = original[1]["observed_indices"][::-1].copy()
        changed["observed_indices"] = ids
        with self.assertRaises(c.ContractError):
            b._public_records((original[0], changed), b.StarPairDesign(200,50))


if __name__ == "__main__":
    unittest.main()
