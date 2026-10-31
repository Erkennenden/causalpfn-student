"""Stage 6.3: independent scalar references and small in-memory ledgers only.

ATOL=1e-12, RTOL=0 fixed before the first run. The 10000x200 RNG test
draws indices only; no formal task universe, model, DGP or disk outputs.
"""
from dataclasses import replace
import hashlib
import math
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import bootstrap as b
from star_v2.task_registry import TaskRegistry, Attempt
from star_v2.metrics import Prediction, MechanismTruth
from star_v2.estimation import (EstimateBundle, PairEstimateBundle, IntervalRecord,
                                SamplingUncertainty, EFFECT_ORDER)
from star_v2.support import ORDINARY_METHOD_IDS
from star_v2.model_adapters import context_requests_from_parent
from star_v2.coverage import AttemptSelectionAmbiguous
from test_coverage import make_pair, request

ATOL = 1e-12
MAX_ERRORS = {}
MANUAL_DRAW = np.array([[3, 0, 3, 2]], dtype=np.int64)


def error_reference(method_index, mechanism, repeat, effect):
    direct = (method_index + 1) * (mechanism + 1) + repeat
    spill = (method_index + 1) * (mechanism - 1) - repeat
    return float((direct, spill, direct + spill)[effect])


def interval_reference(m, r, center, effect):
    # One independently specified interval-only failure, point remains valid.
    failed = (m, r, center, effect) == (3, 1, 1, 1)
    return not failed and (m + r + center + effect) % 3 != 0


def fixture():
    registry = TaskRegistry()
    entries = []
    supports = []
    truths = tuple(MechanismTruth(m, np.array([m + .5, 2*m + 1., 3*m + 1.5]))
                   for m in range(4))
    for m in range(4):
        for r in range(3):
            pair = make_pair(m, r)
            for task in (pair,) + pair.context_refs:
                registry.register_task(task)
            for j, method in enumerate(ORDINARY_METHOD_IDS):
                parent = request(pair, f"synthetic-{m}-{r}", outputs=(
                    ("ate", "interval") if method == "j" else ("ate",)))
                registry.register_request(method, parent)
                children = ()
                if method in ("kernel-old", "kernel-new", "j"):
                    children = context_requests_from_parent(pair, parent,
                        request_ids=("shared-child", "shared-child"))
                    for child in children:
                        registry.register_request(method, child)
                entries.append((pair, method, j, parent, children))
            if m == r == 0:
                oracle = request(pair, "synthetic-oracle", outputs=("ate", "interval"))
                registry.register_request("oracle-pair", oracle)
                entries.append((pair, "oracle-pair", 0, oracle, ()))
    registry.seal(expected_task_ids=tuple(registry.tasks),
                  expected_request_keys=tuple(registry.requests))
    predictions = []
    bundles = {}
    for pair, method, j, parent, children in entries:
        m, r = pair.mechanism_id, pair.pair_repeat_id
        target = truths[m].ate
        values = target + np.array([error_reference(j, m, r, k) for k in range(3)])
        arms = []
        for center, (task, child) in enumerate(zip(pair.context_refs, children)):
            extra = {}
            statuses = {"ate": ("OK",) * 3}
            if method == "j":
                records = []
                interval_states = []
                for k, effect in enumerate(EFFECT_ORDER):
                    failed = (m, r, center, k) == (3, 1, 1, 1)
                    state = "FAILED" if failed else "OK"
                    interval_states.append(state)
                    if failed:
                        records.append(IntervalRecord(task.context_id, effect, state,
                                                      source="model_posterior"))
                    else:
                        covered = interval_reference(m, r, center, k)
                        lower = float(target[k] - 1 if covered else target[k] + 1)
                        upper = float(target[k] + 1 if covered else target[k] + 2)
                        records.append(IntervalRecord(task.context_id, effect, state,
                                                      lower, upper, source="model_posterior"))
                statuses["interval"] = tuple(interval_states)
                extra = dict(intervals=tuple(records),
                    sampling_uncertainty=SamplingUncertainty("OK", "model_posterior", "context", None))
                if "FAILED" in interval_states:
                    extra.update(failure_type="SYNTHETIC_INTERVAL",
                        failure_message="retained interval-only failure",
                        failure_provenance={"stage": "synthetic_interval"})
            arm = EstimateBundle.from_outputs(child, task, method,
                ate=values + (-.25 if center == 0 else .25),
                output_status=statuses, **extra)
            registry.append_attempt(arm, attempt_no=1)
            arms.append(arm)
            if method == "j":
                supports.append((task.context_id, arm.support))
        extra = {}
        statuses = {"ate": ("OK",) * 3}
        if method == "oracle-pair":
            extra = dict(intervals=tuple(IntervalRecord(None, e, "OK",
                float(target[k]-1), float(target[k]+1), source="joint_model_posterior")
                for k, e in enumerate(EFFECT_ORDER)),
                sampling_uncertainty=SamplingUncertainty("OK", "joint_model_posterior", "pair", None))
            statuses["interval"] = ("OK",) * 3
        bundle = PairEstimateBundle.from_outputs(parent, pair, method, ate=values,
            output_status=statuses, arm_results=tuple(arms) if children else (None, None), **extra)
        registry.append_attempt(bundle, attempt_no=1)
        prediction = Prediction.from_bundle(bundle)
        predictions.append(prediction)
        bundles[prediction.key] = bundle
    return registry, tuple(predictions), truths, tuple(supports), bundles


def replay(registry, *, remove_pair=None, remove_context_request=None,
           drop_attempt=None, fail_attempt=None):
    """Fresh ledger replay: never mutate an original test fixture or its history."""
    result = TaskRegistry()
    task_ids = {k for k, v in registry.tasks.items()
                if v.document["pair_id"] != remove_pair}
    for k, task in registry.tasks.items():
        if k in task_ids:
            result.register_task(task)
    keys = {k for k in registry.requests if k[2] in task_ids and k != remove_context_request}
    for k, record in registry.requests.items():
        if k in keys:
            result.register_execution(record)
    result.seal(expected_task_ids=task_ids, expected_request_keys=keys)
    for attempt in registry.attempts:
        if attempt.execution_key not in keys or attempt.execution_key == drop_attempt:
            continue
        if fail_attempt and attempt.execution_key == fail_attempt[0]:
            doc = attempt.document
            doc["output_status"]["ate"][fail_attempt[1]] = "FAILED"
            doc["execution_status"] = "FAILED"
            doc["failure_type"] = "SYNTHETIC_POINT"
            doc["failure_message"] = "one retained effect failed"
            doc["failure_provenance"] = {"stage": "synthetic_failure"}
            attempt = Attempt.from_document(doc)
        result.record_attempt(attempt)
    return result


def run_synthetic(data, indices=MANUAL_DRAW, *, registry=None, predictions=None, supports=None):
    reg, pred, truths, supp, _ = data
    blocks = b.build_mechanism_blocks(registry if registry is not None else reg,
        predictions if predictions is not None else pred, truths,
        supports if supports is not None else supp, formal=False)
    return b.bootstrap_from_synthetic_indices(blocks, indices)


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.collection = b.build_mechanism_blocks(*cls.data[:4], formal=False)
        cls.evidence = b.bootstrap_from_synthetic_indices(cls.collection, MANUAL_DRAW)

    def close(self, name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
        MAX_ERRORS[name] = max(MAX_ERRORS.get(name, 0.),
                              float(np.max(np.abs(actual-expected))))

    def test_fixed_rng_identity_single_generator_and_full_matrix_matches_independent_pcg64(self):
        # No production RNG wrapper supplies any expected value.
        seed = np.random.SeedSequence([20261030, 2026103001, 2, 0, 7, 0])
        independent = np.random.Generator(np.random.PCG64(seed))
        expected = np.empty((10000, 200), dtype=np.int64)
        for r in range(10000):
            expected[r] = independent.integers(0, 200, 200, dtype=np.int64)
        with patch.object(b, "make_generator", wraps=b.make_generator) as factory:
            result = b.generate_bootstrap_indices()
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.args[0].entropy,
                         (20261030, 2026103001, 2, 0, 7, 0))
        np.testing.assert_array_equal(result.indices, expected)
        np.testing.assert_array_equal(result.replicate_ids, np.arange(10000, dtype=np.int64))
        self.assertEqual(result.indices.shape, (10000, 200))
        self.assertEqual(result.indices.dtype, np.dtype("int64"))
        self.assertTrue(np.all((result.indices >= 0) & (result.indices < 200)))
        self.assertTrue(all(len(set(row)) < 200 for row in result.indices))
        self.assertEqual(tuple(result.rng_metadata["entropy"]), (20261030, 2026103001, 2, 0, 7, 0))
        self.assertEqual(result.rng_metadata["bit_generator"], "PCG64")
        self.assertEqual(result.numpy_version, np.__version__)
        self.assertFalse(result.indices.flags.writeable)
        with self.assertRaises(ValueError):
            result.indices.setflags(write=True)

    def test_scalar_four_mechanisms_three_pairs_repeated_draw_position_rmse(self):
        evidence = self.evidence
        self.assertEqual([p.pair_repeat_id for p in evidence.mechanism_blocks[3].pairs], [0, 1, 2])
        self.assertEqual(evidence.indices[0].tolist(), [3, 0, 3, 2])
        self.assertEqual(evidence.draw_positions.tolist(), [0, 1, 2, 3])
        for j, method in enumerate(ORDINARY_METHOD_IDS):
            for k, effect in enumerate(EFFECT_ORDER):
                # Scalar independent computation across all 12 drawn Pair records.
                squared = [error_reference(j, mid, r, k)**2
                           for mid in (3, 0, 3, 2) for r in (0, 1, 2)]
                expected = math.sqrt(sum(squared) / 12)
                self.close(method + effect, evidence.method_rmse[0, j, k], expected)
        # Deduplicating repeated mechanism 3 would change the result.
        deduplicated = math.sqrt(sum(error_reference(0, mid, r, 0)**2
            for mid in (3, 0, 2) for r in (0, 1, 2)) / 9)
        self.assertGreater(abs(evidence.method_rmse[0, 0, 0]-deduplicated), .01)

    def test_all_methods_and_effects_share_the_same_draw_positions(self):
        draws = np.array([[3, 0, 3, 2], [1, 1, 1, 1], [0, 1, 2, 3]], dtype=np.int64)
        evidence = b.bootstrap_from_synthetic_indices(self.collection, draws)
        for rep, row in enumerate(draws):
            for j in range(7):
                for k in range(3):
                    expected = math.sqrt(sum(error_reference(j, int(mid), r, k)**2
                                              for mid in row for r in range(3)) / 12)
                    self.close("shared_indices", evidence.method_rmse[rep, j, k], expected)
        self.assertEqual(evidence.replicate_ids.tolist(), [0, 1, 2])

    def test_twenty_one_comparisons_three_effects_and_signed_difference_reference(self):
        evidence = self.evidence
        expected_pairs = {(a, c) for j, a in enumerate(ORDINARY_METHOD_IDS)
                          for c in ORDINARY_METHOD_IDS[j+1:]}
        self.assertEqual(len(evidence.comparisons), 21)
        self.assertEqual({(c.method_a, c.method_b) for c in evidence.comparisons}, expected_pairs)
        for comparison in evidence.comparisons:
            self.assertEqual(tuple(comparison.effects), EFFECT_ORDER)
            a = ORDINARY_METHOD_IDS.index(comparison.method_a)
            c = ORDINARY_METHOD_IDS.index(comparison.method_b)
            for k, effect in enumerate(EFFECT_ORDER):
                first = math.sqrt(sum(error_reference(a, mid, r, k)**2
                    for mid in (3, 0, 3, 2) for r in range(3))/12)
                second = math.sqrt(sum(error_reference(c, mid, r, k)**2
                    for mid in (3, 0, 3, 2) for r in range(3))/12)
                dist = comparison.effects[effect]
                self.assertEqual(dist.status, "OK")
                self.close("paired_difference", dist.values, [first-second])
                self.close("single_replicate_ci", dist.interval, [first-second, first-second])
        self.assertEqual(evidence.evidence_scope, "SYNTHETIC")
        self.assertEqual(evidence.interval_level, .95)
        self.assertIn("pointwise", evidence.interval_interpretation)

    def test_nine_j_coverage_distributions_use_whole_blocks_and_full_denominators(self):
        evidence = self.evidence
        self.assertEqual(tuple(evidence.j_coverage), ("overall", "center_t=0", "center_t=1"))
        for scope, centers, denominator in (("overall", (0, 1), 24),
                                            ("center_t=0", (0,), 12),
                                            ("center_t=1", (1,), 12)):
            for k, effect in enumerate(EFFECT_ORDER):
                expected = sum(interval_reference(mid, r, center, k)
                    for mid in (3, 0, 3, 2) for r in range(3) for center in centers) / denominator
                self.close("coverage_" + scope + effect, evidence.j_coverage[scope][effect].values, [expected])
        self.assertEqual(sum(len(row) for row in evidence.j_coverage.values()), 9)
        self.assertNotIn("coverage_valid", evidence.j_coverage)

    def test_interval_only_failure_retained_and_uncovered_without_invalidating_pair_ate(self):
        block = self.evidence.mechanism_blocks[3]
        failed = [r for pair in block.pairs for r in pair.j_interval_records
                  if r["effective_status"] != "OK"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["effect"], "Spillover")
        self.assertFalse(failed[0]["covered"])
        self.assertEqual(failed[0]["failure_code"], "RECORDED_INTERVAL_FAILURE")
        self.assertIn("interval-only", failed[0]["failure_reason"]["execution_failure"])
        self.assertTrue(self.evidence.rankability["methods"]["j"]["all_effects_rankable"])
        self.assertTrue(np.isfinite(self.evidence.method_rmse[:, 6, :]).all())
        # Total 24 requests per effect; a failed interval is never removed.
        self.assertEqual(self.collection.coverage["effects"]["Spillover"]["all_requested"], 24)
        self.assertEqual(self.collection.coverage["effects"]["Spillover"]["valid_intervals"], 23)

    def test_percentile_uses_linear_interpolation_independent_sorted_reference(self):
        # Four replicates give nonintegral .075 and 2.925 order-statistic positions.
        draws = np.array([[0]*4, [1]*4, [2]*4, [3]*4], dtype=np.int64)
        result = b.bootstrap_from_synthetic_indices(self.collection, draws)
        def scalar_percentiles(values):
            ordered = sorted(float(v) for v in values)
            return (ordered[0] + .075*(ordered[1]-ordered[0]),
                    ordered[2] + .925*(ordered[3]-ordered[2]))
        for comparison in result.comparisons:
            a = ORDINARY_METHOD_IDS.index(comparison.method_a)
            c = ORDINARY_METHOD_IDS.index(comparison.method_b)
            for k, effect in enumerate(EFFECT_ORDER):
                expected_values = [
                    math.sqrt(sum(error_reference(a, mid, r, k)**2 for r in range(3))/3)
                    - math.sqrt(sum(error_reference(c, mid, r, k)**2 for r in range(3))/3)
                    for mid in range(4)]
                dist = comparison.effects[effect]
                self.close("linear_comparison_values", dist.values, expected_values)
                self.close("linear_comparison_ci", dist.interval, scalar_percentiles(expected_values))
        for name, centers, denominator in (("overall", (0, 1), 6),
                                           ("center_t=0", (0,), 3),
                                           ("center_t=1", (1,), 3)):
            for k, effect in enumerate(EFFECT_ORDER):
                expected_values = [sum(interval_reference(mid, r, center, k)
                    for r in range(3) for center in centers)/denominator for mid in range(4)]
                dist = result.j_coverage[name][effect]
                self.close("linear_coverage_values", dist.values, expected_values)
                self.close("linear_coverage_ci", dist.interval, scalar_percentiles(expected_values))

    def test_not_rankable_never_recovers_even_if_draws_omit_failed_mechanism(self):
        reg, predictions, _, _, _ = self.data
        key = next(p.key for p in predictions if p.key[0] == "ht")
        failed = replay(reg, fail_attempt=(key, 0))
        # Parent success artifact deliberately stale: ledger failure must dominate.
        result = run_synthetic(self.data, np.array([[3, 3, 2, 2]], dtype=np.int64), registry=failed)
        self.assertEqual(result.rankability["methods"]["ht"]["effects"]["Direct"][
            "rankability_status"], "NOT_RANKABLE")
        self.assertTrue(np.isnan(result.method_rmse[:, 1, 0]).all())
        for comparison in result.comparisons:
            affected = "ht" in (comparison.method_a, comparison.method_b)
            self.assertEqual(comparison.effects["Direct"].status,
                             "NOT_COMPARABLE" if affected else "OK")
            self.assertEqual(comparison.effects["Spillover"].status, "OK")
            if affected:
                self.assertIsNone(comparison.effects["Direct"].values)
                self.assertIsNone(comparison.effects["Direct"].interval)
                self.assertIn("ht", comparison.effects["Direct"].failure_reasons)
        records = result.mechanism_blocks[0].pairs[0].attempt_history
        self.assertTrue(any(a.execution_key == key and a.document["failure_message"]
                            == "one retained effect failed" for a in records))
        self.assertEqual(len(result.mechanism_blocks[0].pairs), 3)
        self.assertIsNone(result.point_metrics["ordinary"]["ht"]["Direct"]["pooled_rmse"])
        self.assertIsNotNone(result.point_metrics["ordinary"]["ht"]["Direct"]["success_subset_pooled_rmse"])

    def test_missing_attempt_retained_and_required_child_failure_blocks_only_its_effect(self):
        reg, predictions, _, _, _ = self.data
        child = next(k for k in reg.requests if k[:2] == ("j", "context"))
        changed = replay(reg, fail_attempt=(child, 1))
        result = run_synthetic(self.data, registry=changed)
        self.assertEqual(result.rankability["methods"]["j"]["effects"]["Spillover"][
            "rankability_status"], "NOT_RANKABLE")
        self.assertTrue(np.isfinite(result.method_rmse[:, 6, 0]).all())
        self.assertTrue(np.isnan(result.method_rmse[:, 6, 1]).all())
        self.assertEqual(result.mechanism_blocks[0].pairs[0].j_interval_records[1][
            "failure_code"], "DEPENDENT_POINT_FAILURE")
        # Missing Attempt remains in the block's pre-requested inventory.
        missing = replay(reg, drop_attempt=child)
        result = run_synthetic(self.data, registry=missing)
        pair = result.mechanism_blocks[0].pairs[0]
        self.assertTrue(any(r.key == child for r in pair.requests))
        self.assertFalse(any(a.execution_key == child for a in pair.attempt_history))
        self.assertEqual(sum(row["failure_code"] == "UNEXECUTED"
                             for row in pair.j_interval_records), 3)

    def test_multiple_attempts_block_and_preserve_both_histories(self):
        reg, predictions, truths, supports, bundles = self.data
        key = next(k for k in bundles if k[0] == "ht")
        changed = replay(reg, fail_attempt=(key, 0))
        changed.append_attempt(bundles[key], attempt_no=2, authorization="explicit synthetic authorization")
        before = tuple(a.content for a in changed.attempts)
        with self.assertRaises(AttemptSelectionAmbiguous) as caught:
            b.build_mechanism_blocks(changed, predictions, truths, supports, formal=False)
        self.assertEqual(caught.exception.code, "ATTEMPT_SELECTION_AMBIGUOUS")
        self.assertEqual(tuple(a.content for a in changed.attempts), before)
        self.assertEqual([a.document["attempt_no"] for a in changed.attempts if a.execution_key == key], [1, 2])

    def test_oracle_is_kept_separately_not_in_twenty_one_comparisons(self):
        evidence = self.evidence
        self.assertIn("oracle-pair", evidence.point_metrics["oracle_reference"])
        self.assertTrue(any(p.key[0] == "oracle-pair"
            for p in evidence.mechanism_blocks[0].pairs[0].predictions))
        self.assertTrue(any(a.execution_key[0] == "oracle-pair"
            for a in evidence.mechanism_blocks[0].pairs[0].attempt_history))
        self.assertTrue(all(c.method_a in ORDINARY_METHOD_IDS and c.method_b in ORDINARY_METHOD_IDS
                            for c in evidence.comparisons))
        self.assertEqual(len(evidence.method_order), 7)
        self.assertEqual(len(self.collection.coverage["oracle_reference_records"]), 1)

    def test_missing_pair_rejected_before_any_rng(self):
        reg, predictions, truths, supports, _ = self.data
        pid = self.collection.blocks[0].pairs[0].pair_task.task_id
        changed = replay(reg, remove_pair=pid)
        with patch.object(b, "make_generator", side_effect=AssertionError("must not draw")):
            with self.assertRaisesRegex(ValueError, "Pair"):
                b.build_mechanism_blocks(changed, predictions, truths, supports, formal=False)

    def test_missing_context_request_rejected(self):
        reg, predictions, truths, supports, _ = self.data
        child = next(k for k in reg.requests if k[:2] == ("j", "context"))
        changed = replay(reg, remove_context_request=child)
        with self.assertRaisesRegex(ValueError, "Context"):
            b.build_mechanism_blocks(changed, predictions, truths, supports, formal=False)

    def test_missing_duplicate_extra_or_conflicting_support_rejected(self):
        reg, pred, truths, supports, _ = self.data
        with self.assertRaisesRegex(ValueError, "Missing Context Support"):
            b.build_mechanism_blocks(reg, pred, truths, supports[1:], formal=False)
        with self.assertRaisesRegex(ValueError, "duplicate Support"):
            b.build_mechanism_blocks(reg, pred, truths, supports+(supports[0],), formal=False)
        cid, support = supports[0]
        for bad in (replace(support, source_input_hash="f"*64),
                    replace(support, empirical_extrapolation=np.zeros((200, 2, 2), dtype=bool)),
                    replace(support, support_reason=np.full((200, 2, 2), "illegal"))):
            with self.assertRaises(ValueError):
                b.build_mechanism_blocks(reg, pred, truths, ((cid, bad),)+supports[1:], formal=False)

    def test_default_formal_entry_rejects_small_plan_before_rng(self):
        with patch.object(b, "make_generator", side_effect=AssertionError("must not draw")):
            with self.assertRaisesRegex(ValueError, "0..199"):
                b.mechanism_block_bootstrap(*self.data[:4])

    def test_formal_blocks_cannot_use_synthetic_index_entry(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            b.bootstrap_from_synthetic_indices(replace(self.collection, synthetic=False), MANUAL_DRAW)

    def test_bad_draw_indices_shapes_and_dtypes_rejected(self):
        for bad in (np.array([[0, 1, 2]], dtype=np.int64),
                    np.array([[0, 1, 2, 4]], dtype=np.int64),
                    np.array([[0, 1, 2, -1]], dtype=np.int64),
                    MANUAL_DRAW.astype(np.int32), np.empty((0, 4), dtype=np.int64)):
            with self.assertRaisesRegex(ValueError, "draw-position"):
                b.bootstrap_from_synthetic_indices(self.collection, bad)

    def test_request_identity_support_predictions_and_inputs_remain_immutable(self):
        reg, pred, truths, supports, _ = self.data
        before_attempts = tuple(a.content for a in reg.attempts)
        before_requests = tuple(r.content for r in reg.requests.values())
        before_arrays = tuple(hashlib.sha256(p.ate.tobytes()).hexdigest() for p in pred)
        result = b.bootstrap_from_synthetic_indices(self.collection, MANUAL_DRAW)
        self.assertEqual(tuple(a.content for a in reg.attempts), before_attempts)
        self.assertEqual(tuple(r.content for r in reg.requests.values()), before_requests)
        self.assertEqual(tuple(hashlib.sha256(p.ate.tobytes()).hexdigest() for p in pred), before_arrays)
        self.assertEqual(result.registry_seal, reg.seal_sha256)
        for block in result.mechanism_blocks:
            self.assertEqual(block.target.mechanism_id, block.mechanism_id)
            for pair in block.pairs:
                self.assertEqual(pair.pair_task.document["identity"]["mechanism_id"], block.mechanism_id)
                for center, context in enumerate(pair.context_tasks):
                    self.assertEqual(context.document["identity"]["center_t"], [center])
                    self.assertEqual(pair.support[center].source_input_hash, context.document["input_hashes"][0])
        for arr in (result.indices, result.replicate_ids, result.method_rmse,
                    result.mechanism_blocks[0].method_mse,
                    result.mechanism_blocks[0].pairs[0].support[0].population_support):
            self.assertFalse(arr.flags.writeable)
            with self.assertRaises(ValueError):
                arr.setflags(write=True)
        with self.assertRaises(TypeError):
            result.j_coverage["overall"]["Direct"] = None

    def test_repeat_evaluation_determinism_and_point_estimates_reused(self):
        again = b.bootstrap_from_synthetic_indices(self.collection, MANUAL_DRAW.copy())
        np.testing.assert_array_equal(again.method_rmse, self.evidence.method_rmse)
        for j, method in enumerate(ORDINARY_METHOD_IDS):
            for k, effect in enumerate(EFFECT_ORDER):
                expected = math.sqrt(sum(error_reference(j, mid, r, k)**2
                                          for mid in range(4) for r in range(3))/12)
                self.close("frozen_point_metric", again.point_metrics["ordinary"][method][effect][
                    "pooled_rmse"], expected)


if __name__ == "__main__":
    unittest.main()
