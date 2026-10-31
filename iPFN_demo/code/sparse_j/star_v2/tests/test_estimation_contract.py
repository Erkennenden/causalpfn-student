"""S3.1 only: artificial in-memory data and stub callbacks, no estimators."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import unittest
from unittest.mock import Mock

import numpy as np

from star_v2 import contracts as c
from star_v2 import estimation as e
from star_v2.config import CONFIG
from star_v2.support import require_plugin_support, PluginSupportError
from test_pair_data import fixture_pair
from test_contracts import tasks, base_bytes


def request(task, *, scope_id="upstream-req", status="POINT", outputs=("ate",), **kw):
    return e.EstimationRequest.from_task(
        task, status=status, target_id=CONFIG.target_id, request_id=scope_id,
        requested_outputs=outputs, **kw)


def values(statuses=("OK", "OK", "OK"), numbers=(1., 2., 3.)):
    return np.array([v if s == "OK" else np.nan
                     for s, v in zip(statuses, numbers)], dtype=np.float64)


class EstimationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private = fixture_pair()
        cls.tasks = tasks(cls.private)
        cls.pair = c.pair_from_tasks(cls.tasks)

    def context(self, *, task=None, statuses=("OK", "OK", "OK"), method="kernel-old",
                req=None, **kw):
        t = self.tasks[0] if task is None else task
        req = request(t) if req is None else req
        if any(s in ("FAILED", "NOT_ESTIMABLE") for s in statuses):
            kw.setdefault("failure_type", "TEST_REASON")
            kw.setdefault("failure_message", "effect-specific fixture failure")
        return e.EstimateBundle.from_outputs(
            req, t, method, output_status={"ate": statuses}, ate=values(statuses), **kw)

    def joint(self, *, method="state-mean", statuses=("OK", "OK", "OK"), req=None, **kw):
        req = request(self.pair) if req is None else req
        if any(s in ("FAILED", "NOT_ESTIMABLE") for s in statuses):
            kw.setdefault("failure_type", "TEST_REASON")
            kw.setdefault("failure_message", "effect-specific fixture failure")
        return e.PairEstimateBundle.from_outputs(
            req, self.pair, method, output_status={"ate": statuses},
            ate=values(statuses), arm_results=(None, None), **kw)

    def arms(self, parent="pair-request", statuses=("OK", "OK", "OK"), method="kernel-old"):
        return tuple(self.context(
            task=t, req=request(t, scope_id="arm-request-" + str(k), parent_request_id=parent),
            statuses=statuses if k else ("OK", "OK", "OK"), method=method,
            failure_provenance={"arm": k, "effects": {"Spillover": "fixture"}})
            for k, t in enumerate(self.tasks))

    def derived(self, arms, statuses=("OK", "OK", "OK"), method="kernel-old", **kw):
        if any(s != "OK" for s in statuses):
            kw.setdefault("failure_type", "ARM_DEPENDENCY")
            kw.setdefault("failure_message", "preserve independent effect failures")
        return e.PairEstimateBundle.from_outputs(
            request(self.pair, scope_id="pair-request"), self.pair, method,
            output_status={"ate": statuses}, ate=values(statuses), arm_results=arms, **kw)

    def test_01_identity_inheritance_and_unbound_tasks(self):
        t = self.tasks[0]
        r = request(t)
        b = self.context(req=r)
        self.assertEqual((r.status, b.status, b.execution_status), ("POINT", "POINT", "OK"))
        for name in ("request_id", "target_id", "pair_id", "context_ids", "input_hashes",
                     "protocol_hash", "protocol_version", "task_id"):
            self.assertEqual(getattr(b, name), getattr(r, name))
        self.assertEqual(b.input_hash, t.input_hash)
        self.assertEqual(r.identity["rng_identity"], t.rng_identity)
        self.assertEqual(r.identity["center_t"], (0,))
        self.assertEqual(r.identity["mechanism_id"], t.mechanism_id)
        np.testing.assert_array_equal(b.node_ids, t.node_ids)
        self.assertIsNone(t.request_id)
        self.assertIsNone(t.target_id)
        self.assertFalse(hasattr(t, "status"))

    def test_02_upstream_status_mandatory_not_ok(self):
        for status in ("OK", "", None, "FAILED"):
            with self.subTest(status=status), self.assertRaises(c.ContractError):
                request(self.tasks[0], status=status)
        with self.assertRaises(TypeError):
            e.EstimationRequest.from_task(self.tasks[0], target_id=CONFIG.target_id, request_id="r")

    def test_03_explicit_existing_links_preserved(self):
        t = tasks(self.private, target_id=CONFIG.target_id, request_id="bound")[0]
        b = self.context(task=t, req=request(t, scope_id="bound"))
        self.assertEqual(b.request_id, "bound")
        with self.assertRaises(c.ContractError):
            request(t, scope_id="different")

    def test_04_target_and_protocol_semantics_reject_changes(self):
        r = request(self.tasks[0])
        for kw in ({"target_id": "other"}, {"protocol_hash": "0"*64},
                   {"reference_assignment": "conditional"}, {"exposure_definition": ">=0.5"},
                   {"arm_order": ("00", "10", "01", "11")},
                   {"effect_order": ("Total", "Direct", "Spillover")},
                   {"target_functional": "changed"}, {"request_id": ""}):
            with self.subTest(kw=kw), self.assertRaises(c.ContractError):
                replace(r, **kw)

    def test_05_node_order_dtype_shape_readonly(self):
        r = request(self.tasks[0])
        for nodes in (np.arange(199, -1, -1), np.arange(200, dtype=np.int32),
                      np.arange(200).reshape(200, 1)):
            with self.subTest(shape=nodes.shape), self.assertRaises(c.ContractError):
                replace(r, node_ids=nodes)
        for array in (r.node_ids, r.population_support, self.context().ate):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertTrue(r.population_support.all())

    def test_06_binding_rejects_identity_and_hash_conflicts(self):
        r = request(self.tasks[0])
        for bad in (replace(r, input_hashes=("0"*64,)),
                    replace(r, context_ids=("other-context",)),
                    replace(r, pair_id="other-pair"),
                    replace(r, identity={**r.identity, "mechanism_id": 99})):
            with self.subTest(bad=bad.context_ids), self.assertRaises(c.ContractError):
                self.context(req=bad)
        for hashes in (("not-sha",), (), ("0"*64, "1"*64)):
            with self.assertRaises(c.ContractError):
                replace(r, input_hashes=hashes)

    def test_07_not_identified_never_calls_context_callback(self):
        t = self.tasks[0]
        r = request(t, status="NOT_IDENTIFIED", outputs=("mu", "ite", "ate", "interval"),
                    failure_type="UPSTREAM_RULE", failure_message="original reason",
                    failure_provenance={"identification": {"code": "RULE_17"}})
        callback = Mock(side_effect=AssertionError("must not run"))
        b = e.execute_contract(r, t, "j", callback)
        callback.assert_not_called()
        self.assertEqual((b.status, b.execution_status), ("NOT_IDENTIFIED", "NOT_IDENTIFIED"))
        self.assertEqual(b.failure_message, r.failure_message)
        self.assertEqual(dict(b.failure_provenance), dict(r.failure_provenance))
        for a in (b.mu, b.ite, b.ate):
            self.assertTrue(np.isnan(a).all())
        self.assertEqual(len(b.intervals), 3)
        self.assertEqual(b.sampling_uncertainty.status, "FAILED")

    def test_08_not_identified_pair_retains_two_arm_identity_evidence(self):
        for method in ("state-mean", "kernel-old"):
            r = request(self.pair, status="NOT_IDENTIFIED", failure_type="ID",
                        failure_message="unaltered upstream reason")
            callback = Mock()
            b = e.execute_contract(r, self.pair, method, callback)
            callback.assert_not_called()
            self.assertEqual(b.execution_status, "NOT_IDENTIFIED")
            self.assertEqual(b.context_ids, self.pair.context_ids)
            self.assertEqual(len(b.arm_status_evidence), 2)
            for arm in b.arm_status_evidence:
                self.assertIsNone(arm["execution_status"])
                self.assertEqual(arm["failure_message"], "unaltered upstream reason")

    def test_09_not_identified_requires_reason_and_rejects_valid_output(self):
        t = self.tasks[0]
        with self.assertRaises(c.ContractError):
            request(t, status="NOT_IDENTIFIED")
        r = request(t, status="NOT_IDENTIFIED", failure_type="ID", failure_message="upstream")
        with self.assertRaises(c.ContractError):
            self.context(req=r)
        for kw in ({"failure_type": "overwritten"}, {"failure_message": "overwritten"},
                   {"failure_provenance": {"changed": True}}):
            outputs = dict(output_status={"ate": ("FAILED",)*3},
                           failure_type=r.failure_type, failure_message=r.failure_message)
            outputs.update(kw)
            with self.assertRaises(c.ContractError):
                e.EstimateBundle.from_outputs(r, t, "kernel-old", **outputs)

    def test_10_partial_effect_failure_preserves_other_values(self):
        statuses = ("OK", "NOT_ESTIMABLE", "OK")
        b = self.joint(statuses=statuses, failure_provenance={"Spillover": "empty_11"})
        self.assertEqual(b.output_status["ate"], statuses)
        self.assertEqual(b.execution_status, "NOT_ESTIMABLE")
        np.testing.assert_array_equal(b.ate[[0, 2]], [1., 3.])
        self.assertTrue(np.isnan(b.ate[1]))
        self.assertEqual(b.failure_provenance["Spillover"], "empty_11")

    def test_11_engineering_failure_aggregation_and_effect_order(self):
        b = self.joint(statuses=("NOT_ESTIMABLE", "FAILED", "OK"))
        self.assertEqual(b.execution_status, "FAILED")
        self.assertEqual(b.status, "POINT")
        self.assertEqual(b.effect_order, ("Direct", "Spillover", "Total"))
        self.assertEqual(b.ate[2], 3.)

    def test_12_invalid_ate_values_rejected_independently(self):
        for state, number in (("OK", np.nan), ("OK", np.inf),
                              ("FAILED", 0.), ("NOT_ESTIMABLE", 1.),
                              ("FAILED", np.inf)):
            s = (state, "OK", "OK")
            b = self.joint(statuses=s)
            a = b.ate.copy()
            a[0] = number
            with self.subTest(state=state, number=number), self.assertRaises(c.ContractError):
                replace(b, ate=a)

    def test_13_output_shape_dtype_no_float32_conversion(self):
        b = self.context()
        for a in (b.ate.astype(np.float32), np.zeros(4), np.zeros((1, 3)),
                  [1., 2., 3.]):
            with self.subTest(type=type(a)), self.assertRaises(c.ContractError):
                replace(b, ate=a)
        self.assertEqual(b.ate.dtype, np.float64)
        self.assertEqual(self.tasks[0].outcomes_observed.dtype, np.float64)

    def test_14_all_four_ate_only_methods_never_fake_node_outputs(self):
        for method in ("state-mean", "ht", "f-z2", "lin-z2"):
            with self.subTest(method=method):
                b = self.joint(method=method)
                self.assertEqual(b.execution_status, "OK")
                self.assertEqual(b.output_status["mu"], "UNSUPPORTED")
                self.assertEqual(b.output_status["ite"], ("UNSUPPORTED",)*3)
                self.assertTrue(np.isnan(b.mu).all())
                self.assertTrue(np.isnan(b.ite).all())
                with self.assertRaises(c.ContractError):
                    replace(b, mu=np.zeros((200, 2, 2)))
                with self.assertRaises(c.ContractError):
                    replace(b, output_status={**b.output_status, "mu": "OK"})

    def test_15_not_requested_is_not_unsupported(self):
        b = self.context()
        self.assertEqual(b.output_status["mu"], "NOT_REQUESTED")
        self.assertEqual(b.output_status["ite"], ("NOT_REQUESTED",)*3)
        self.assertEqual(b.output_status["interval"], ("NOT_REQUESTED",)*3)
        self.assertEqual(b.sampling_uncertainty.status, "NOT_REQUESTED")
        b = self.joint(req=request(self.pair, outputs=("ate", "interval")))
        self.assertEqual(b.output_status["interval"], ("UNSUPPORTED",)*3)
        self.assertEqual(b.sampling_uncertainty.status, "UNSUPPORTED")

    def test_16_mu_axes_and_ite_effect_statuses(self):
        t = self.tasks[0]
        r = request(t, outputs=("mu", "ite", "ate"))
        mu = np.arange(800, dtype=np.float64).reshape(200, 2, 2)
        ite = np.tile([1., np.nan, 3.], (200, 1))
        b = e.EstimateBundle.from_outputs(
            r, t, "kernel-new", output_status={"mu": "OK",
                "ite": ("OK", "FAILED", "OK"), "ate": ("OK",)*3},
            mu=mu, ite=ite, ate=values(), failure_type="ITE_ERROR",
            failure_message="preserve unaffected columns")
        np.testing.assert_array_equal(b.mu[0].ravel(), [0., 1., 2., 3.])
        self.assertEqual(b.arm_order, ("00", "01", "10", "11"))
        self.assertEqual(b.execution_status, "FAILED")
        self.assertTrue(np.isfinite(b.ate).all())
        self.assertTrue(np.isnan(b.ite[:, 1]).all())

    def test_17_support_capability_estimability_independent(self):
        b = self.joint(estimability={"Direct": {"status": "OK", "reason": "fixture"}},
                       diagnostics={"state_01_empty": True})
        self.assertTrue(b.design_support.all())
        self.assertTrue(b.population_support.all())
        self.assertFalse(b.empirical_support_mask.all())
        self.assertNotIn("context.mu", b.method_capability.outputs)
        self.assertEqual(b.execution_status, "OK")
        self.assertEqual(b.estimability["Direct"]["status"], "OK")
        # No automatic rejection of a valid prediction due to empirical support.
        self.assertTrue(np.isfinite(b.ate).all())

    def test_18_extrapolation_gate_keeps_prediction_for_error_research(self):
        b = self.context()
        query = tuple(map(int, np.argwhere(~b.empirical_support_mask)[0]))
        with self.assertRaises(PluginSupportError):
            require_plugin_support(b.support, (query,))
        self.assertTrue(require_plugin_support(
            b.support, (query,), extrapolation_authorized=True).allowed)
        self.assertTrue(b.extrapolation_flags["empirical"][query])
        self.assertEqual(b.ate[0], 1.)

    def test_19_support_source_mismatch_rejected(self):
        b = self.context()
        other = e.support_for_task(self.tasks[1])
        with self.assertRaises(c.ContractError):
            replace(b, support=other)

    def test_20_pair_joint_does_not_invent_independent_context_estimates(self):
        b = self.joint()
        self.assertEqual(b.context_ids, self.pair.context_ids)
        self.assertEqual(b.input_hashes, tuple(t.input_hash for t in self.tasks))
        self.assertEqual(b.arm_results, (None, None))
        for arm in b.arm_status_evidence:
            self.assertIsNone(arm["execution_status"])
            self.assertIsNone(arm["output_status"])
            self.assertEqual(arm["estimation_role"], "not_independently_executed")
        with self.assertRaises(c.ContractError):
            replace(b, arm_results=self.arms())

    def test_21_pair_context_arm_failure_is_per_effect(self):
        statuses = ("OK", "FAILED", "OK")
        arms = self.arms(statuses=statuses)
        b = self.derived(arms, statuses=statuses)
        self.assertEqual(b.execution_status, "FAILED")
        self.assertEqual(b.ate[0], 1.)
        self.assertEqual(b.ate[2], 3.)
        self.assertEqual(b.arm_status_evidence[1]["failure_message"], arms[1].failure_message)
        self.assertEqual(b.arm_status_evidence[1]["failure_provenance"]["arm"], 1)
        with self.assertRaises(c.ContractError):
            self.derived(arms)  # Cannot claim OK for the failed dependent effect.

    def test_22_pair_arm_identity_order_and_parent_link_checked(self):
        arms = self.arms()
        for bad in ((arms[1], arms[0]), (arms[0], None), (arms[0],),
                    self.arms(parent="wrong-parent")):
            with self.subTest(arms=len(bad)), self.assertRaises(c.ContractError):
                self.derived(bad)

    def test_23_pair_cannot_erase_failure_evidence(self):
        b = self.derived(self.arms(statuses=("FAILED", "OK", "OK")),
                         statuses=("FAILED", "OK", "OK"))
        self.assertEqual(b.context_ids[1], b.arm_results[1].context_ids[0])
        self.assertEqual(b.arm_status_evidence[1]["execution_status"], "FAILED")
        self.assertEqual(b.failure_type, "ARM_DEPENDENCY")
        self.assertEqual(b.arm_results[1].failure_type, "TEST_REASON")

    def test_24_context_intervals_and_uncertainty_provenance(self):
        t = self.tasks[0]
        r = request(t, outputs=("ate", "interval"))
        records = tuple(e.IntervalRecord(t.context_id, effect, "OK", 0., 4.,
                                         source="model_posterior") for effect in e.EFFECT_ORDER)
        b = e.EstimateBundle.from_outputs(
            r, t, "j", output_status={"ate": ("OK",)*3, "interval": ("OK",)*3},
            ate=values(), intervals=records,
            sampling_uncertainty=e.SamplingUncertainty("OK", "model_posterior", "context", None))
        self.assertEqual(b.execution_status, "OK")
        self.assertEqual(b.sampling_uncertainty.statistical_unit, "context")
        self.assertIsNone(b.sampling_uncertainty.effective_sample_count)
        self.assertEqual(len(b.intervals), 3)
        with self.assertRaises(c.ContractError):
            replace(b, sampling_uncertainty=e.SamplingUncertainty(
                "OK", "mechanism_bootstrap", "mechanism", 100))

    def test_25_interval_failures_keep_denominators(self):
        t = self.tasks[0]
        r = request(t, outputs=("ate", "interval"))
        states = ("FAILED", "NOT_ESTIMABLE", "FAILED")
        records = tuple(e.IntervalRecord(t.context_id, effect, state)
                        for effect, state in zip(e.EFFECT_ORDER, states))
        b = e.EstimateBundle.from_outputs(
            r, t, "j", output_status={"ate": ("OK",)*3, "interval": states}, ate=values(),
            intervals=records, failure_type="INTERVAL", failure_message="no interval output")
        self.assertEqual(b.execution_status, "FAILED")
        self.assertEqual(len(b.intervals), 3)
        self.assertTrue(np.isfinite(b.ate).all())
        with self.assertRaises(c.ContractError):
            replace(b, intervals=())

    def test_26_invalid_intervals_and_uncertainty_rejected(self):
        for kwargs in ({"lower": 2., "upper": 1.}, {"lower": np.nan, "upper": 1.},
                       {"lower": 0., "upper": 1., "level": 0.9}):
            with self.assertRaises(c.ContractError):
                e.IntervalRecord("ctx", "Direct", "OK", source="model_posterior", **kwargs)
        with self.assertRaises(c.ContractError):
            e.IntervalRecord("ctx", "Direct", "FAILED", 0., 0.)
        for args in (("OK", None, "context", 100), ("OK", "posterior", "context", 0),
                     ("UNSUPPORTED", "posterior", None, None),
                     ("OK", "mechanism_bootstrap", "context", 10)):
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty(*args)

    def test_27_pair_no_averaging_interval_endpoints(self):
        b = self.derived(self.arms(method="j"), method="j")
        self.assertEqual(b.context_intervals, ((), ()))
        with self.assertRaises(c.ContractError):
            replace(b, intervals=(e.IntervalRecord(None, "Direct", "OK", 0., 1.,
                                                  source="model_posterior"),))

    def test_28_callback_strict_five_arrays_and_detached_storage(self):
        t = self.tasks[0]
        def stub(payload):
            self.assertEqual(tuple(payload), c.FACTUAL_KEYS)
            for name, a in payload.items():
                self.assertFalse(a.flags.writeable)
                self.assertIs(type(base_bytes(a)), bytes)
                self.assertEqual(len(base_bytes(a)), a.nbytes)
                for source in (t.x, t.outcomes_observed,
                               self.private.contexts[0].full_outcomes,
                               self.private.noise, self.private.mechanism.h,
                               self.private.mechanism.coefficients, self.private.mechanism.mu):
                    self.assertFalse(np.shares_memory(a, source))
            self.assertEqual(payload["outcomes_observed"].dtype, np.float64)
            return {"output_status": {"ate": ("OK",)*3}, "ate": values()}
        b = e.execute_contract(request(t), t, "kernel-old", stub)
        self.assertEqual(b.execution_status, "OK")

    def test_29_joint_callback_gets_two_five_array_mappings(self):
        def stub(payloads):
            self.assertEqual(len(payloads), 2)
            for payload in payloads:
                self.assertEqual(tuple(payload), c.FACTUAL_KEYS)
            return {"output_status": {"ate": ("OK",)*3},
                    "ate": values(), "arm_results": (None, None)}
        b = e.execute_contract(request(self.pair), self.pair, "ht", stub)
        self.assertEqual(b.execution_status, "OK")
        self.assertEqual(b.support.n_unique_observed_nodes, 50)
        self.assertEqual(b.support.n_factual_records, 100)

    def test_30_privileged_metadata_and_extra_result_rejected(self):
        for key in ("truth", "beta", "h", "full_y", "noise", "evaluation_results"):
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                self.context(diagnostics={key: [1., 2.]})
            with self.assertRaises(c.ContractError):
                e.execute_contract(request(self.tasks[0]), self.tasks[0], "kernel-old",
                    lambda payload: {"output_status": {"ate": ("OK",)*3},
                                     "ate": values(), key: np.zeros(200)})
        with self.assertRaises(c.ContractError):
            self.context(diagnostics={"innocent": np.ones(200)})
        payload = dict(c.predictor_inputs(self.tasks[0]))
        payload["input_hash"] = self.tasks[0].input_hash
        with self.assertRaises(c.ContractError):
            c.validate_predictor_inputs(payload)

    def test_31_callback_failure_retains_reason_no_retry(self):
        callback = Mock(side_effect=ArithmeticError("deterministic error evidence"))
        b = e.execute_contract(request(self.tasks[0]), self.tasks[0], "kernel-new", callback)
        callback.assert_called_once()
        self.assertEqual(b.status, "POINT")
        self.assertEqual(b.execution_status, "FAILED")
        self.assertEqual(b.failure_type, "ArithmeticError")
        self.assertEqual(b.failure_message, "deterministic error evidence")
        self.assertEqual(b.failure_provenance["attempts"], 1)
        self.assertTrue(np.isnan(b.ate).all())

    def test_32_original_arrays_and_hashes_unchanged(self):
        sources = [self.private.noise, self.private.mechanism.h,
                   self.private.mechanism.coefficients, self.private.mechanism.mu]
        sources += [getattr(t, name) for t in self.tasks for name in c.FACTUAL_KEYS]
        sources += [ctx.full_outcomes for ctx in self.private.contexts]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in sources]
        hashes = tuple(t.input_hash for t in self.tasks)
        for _ in range(2):
            self.context()
            self.joint()
            for t in self.tasks:
                e.execute_contract(request(t), t, "kernel-old",
                    lambda payload: {"output_status": {"ate": ("OK",)*3}, "ate": values()})
        self.assertEqual(before, [hashlib.sha256(a.tobytes()).hexdigest() for a in sources])
        self.assertEqual(hashes, tuple(t.input_hash for t in self.tasks))
        for t in self.tasks:
            c.validate_task(t)

    def test_33_mutated_input_detected_before_callback(self):
        t = tasks(self.private)[0]
        r = request(t)
        changed = t.outcomes_observed.copy()
        changed[0] += 1.
        object.__setattr__(t, "outcomes_observed", changed)
        callback = Mock()
        with self.assertRaises(c.ContractError):
            e.execute_contract(r, t, "kernel-old", callback)
        callback.assert_not_called()

    def test_34_immutable_results_diagnostics_and_failure_metadata(self):
        source = {"nested": {"items": [1, 2]}}
        b = self.context(diagnostics=source, asset_hashes={"fixture_asset": "a"*64})
        source["nested"]["items"][0] = 99
        self.assertEqual(b.diagnostics["nested"]["items"], (1, 2))
        with self.assertRaises(TypeError):
            b.output_status["ate"] = ("FAILED",)*3
        with self.assertRaises(TypeError):
            b.diagnostics["nested"]["new"] = "bad"
        with self.assertRaises(FrozenInstanceError):
            b.method_id = "other"
        with self.assertRaises(c.ContractError):
            replace(b, asset_hashes={"asset": "not-sha"})

    def test_35_no_oracle_execution_or_unsupported_normal_call(self):
        callback = Mock()
        with self.assertRaises(c.ContractError):
            e.execute_contract(request(self.tasks[0]), self.tasks[0], "oracle-context", callback)
        with self.assertRaises(c.ContractError):
            e.execute_contract(request(self.tasks[0]), self.tasks[0], "state-mean", callback)
        with self.assertRaises(c.ContractError):
            e.execute_contract(request(self.pair), self.pair, "kernel-old", callback)
        callback.assert_not_called()

    def test_36_scope_mismatch_and_missing_output_evidence_rejected(self):
        with self.assertRaises(c.ContractError):
            e.EstimateBundle.from_outputs(request(self.pair), self.pair, "ht",
                                          output_status={"ate": ("OK",)*3}, ate=values())
        with self.assertRaises(c.ContractError):
            self.context(req=request(self.tasks[0], outputs=("mu", "ate")))
        with self.assertRaises(c.ContractError):
            replace(self.context(), output_status={"ate": ("OK", "OK")})
        with self.assertRaises(c.ContractError):
            replace(self.context(), output_status={"ate": ("OK",)*3, "unknown": "OK"})


    def test_37_uncertainty_cannot_mislabel_posterior_unit_or_source(self):
        task = self.tasks[0]
        req = request(task, outputs=("ate", "interval"))
        records = tuple(e.IntervalRecord(task.context_id, effect, "OK", 0., 4.,
                        source="model_posterior") for effect in e.EFFECT_ORDER)
        for uncertainty in (
                e.SamplingUncertainty("OK", "model_posterior", "mechanism", 10),
                e.SamplingUncertainty("OK", "mechanism_bootstrap", "mechanism", 10),
                {"status": "OK"}, None):
            with self.subTest(uncertainty=uncertainty), self.assertRaises(c.ContractError):
                e.EstimateBundle.from_outputs(
                    req, task, "j", ate=values(), intervals=records,
                    output_status={"ate": ("OK",)*3, "interval": ("OK",)*3},
                    sampling_uncertainty=uncertainty)

    def test_38_failure_reason_and_invalid_normal_results_explicitly_rejected(self):
        b = self.joint(statuses=("FAILED", "OK", "OK"))
        for changes in ({"failure_type": None}, {"failure_message": ""}):
            with self.assertRaises(c.ContractError):
                replace(b, **changes)
        for result in (None, {"status": "OK"}, {"ate": values()}):
            with self.assertRaises(c.ContractError):
                e.execute_contract(request(self.tasks[0]), self.tasks[0], "kernel-old",
                                   lambda payload: result)

    def test_39_pair_failed_context_intervals_remain_separate_records(self):
        arms = []
        for k, task in enumerate(self.tasks):
            req = request(task, scope_id="arm-" + str(k), parent_request_id="pair-request",
                          outputs=("ate", "interval"))
            records = tuple(e.IntervalRecord(task.context_id, effect, "FAILED")
                            for effect in e.EFFECT_ORDER)
            arms.append(e.EstimateBundle.from_outputs(
                req, task, "j", ate=values(), intervals=records,
                output_status={"ate": ("OK",)*3, "interval": ("FAILED",)*3},
                failure_type="INTERVAL_FAILURE", failure_message="original arm interval reason"))
        b = self.derived(tuple(arms), method="j")
        # Pair point ATE succeeds despite independently failed context intervals.
        self.assertEqual(b.execution_status, "OK")
        self.assertEqual(tuple(len(x) for x in b.context_intervals), (3, 3))
        self.assertEqual(tuple(x["execution_status"] for x in b.arm_status_evidence),
                         ("FAILED", "FAILED"))
        self.assertEqual(b.arm_status_evidence[0]["failure_message"],
                         "original arm interval reason")


if __name__ == "__main__":
    unittest.main()
