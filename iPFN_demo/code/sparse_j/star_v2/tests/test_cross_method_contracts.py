"""6.1.2 cross-method contracts only; no real checkpoint/NPZ/model access.

ATOL=1e-12, RTOL=0 fixed before execution. Regression preprocessing uses an
explicit synthetic transform identity; this is NOT frozen-asset verification.
Nine estimators are never required to have equal values or capabilities.
"""
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import torch

from star_v2 import contracts as c, baselines, baselines_state_ht as state
from star_v2 import baselines_regression as regression, oracle_adapters as oracle
from star_v2.model_adapters import context_requests_from_parent
from star_v2.j_adapter import JAdapter
from star_v2.estimation import EstimationRequest, IntervalRecord, SamplingUncertainty
from star_v2.support import ORDINARY_METHOD_IDS, method_capability, support_for_pair, support_for_task

from test_kernel_adapters import injected as old_injection
from test_kernel_new_adapter import tasks_fixture, injection as new_injection, expected_ate
from test_oracle_adapters import hidden, context_result, joint_result
from test_j_adapter import StatefulPointMock, interval_result, signature
from test_state_mean_hajek import hand_records

METHODS = ORDINARY_METHOD_IDS+("oracle-context","oracle-pair")
CONTEXT_METHODS = ("kernel-old","kernel-new","j","oracle-context")
INTERVAL_CONTEXT_METHODS = ("j","oracle-context")
ATOL = 1e-12
ERRORS = {}
METHOD_EVIDENCE = {m:set() for m in METHODS}
MOCK_TRANSFORM_ID = "a"*64


class Harness:
    def __init__(self, *, status="POINT", outputs=("mu","ite","ate","interval"), task=None):
        original,parent,_ = tasks_fixture(status=status)
        self.task = original if task is None else task
        self.parent = EstimationRequest.from_task(
            self.task,status=status,target_id=parent.target_id,request_id="cross-parent",
            requested_outputs=outputs,failure_type=parent.failure_type,
            failure_message=parent.failure_message,failure_provenance=parent.failure_provenance)
        self.children = context_requests_from_parent(
            self.task,self.parent,request_ids=("cross-child-0","cross-child-1"))
        self.old,self.old_model,self.old_loader,self.old_reader = old_injection()
        self.new,self.new_model,_,_,self.new_loader,self.new_reader = new_injection()
        self.j_model = StatefulPointMock()
        self.j_loader = Mock(return_value=self.j_model)
        self.interval = Mock(return_value=interval_result())
        self.j = JAdapter(loader=self.j_loader,interval_predictor=self.interval)
        self.context_core = Mock(side_effect=lambda **kw:context_result(int(kw["treatment"][0])))
        self.joint_core = Mock(return_value=joint_result())
        self.h = hidden()

    def run(self,method):
        if method in baselines.METHOD_IDS:
            return baselines.estimate_pair(self.task,self.parent,method)
        if method in ("kernel-old","kernel-new","j"):
            adapter = {"kernel-old":self.old,"kernel-new":self.new,"j":self.j}[method]
            return adapter.estimate_pair(self.task,self.parent,context_requests=self.children)
        if method == "oracle-context":
            return oracle.estimate_oracle_context_pair(self.task,self.parent,
                true_h=self.h,noise_sd=1.,context_requests=self.children,core=self.context_core)
        return oracle.estimate_oracle_pair(self.task,self.parent,
            true_h=self.h,noise_sd=1.,core=self.joint_core)


def missing00():
    task,_,_ = tasks_fixture()
    arms = []
    for t in task.context_refs:
        treatment = np.ones(200);treatment[0] = t.center_t
        fields = {name:getattr(t,name) for name in c._META}
        arrays = {name:(treatment if name == "treatment" else getattr(t,name))
                  for name in c._ARRAY_ORDER}
        digest = c._digest("star-v2-task-input-v1",fields,list(arrays.items()))
        arms.append(replace(t,treatment=treatment,input_hash=digest))
    return c.pair_from_tasks(tuple(arms))


class CrossMethodContractTests(unittest.TestCase):
    def setUp(self):
        transform = regression.FrozenTransform(
            np.zeros(20),np.ones(20),np.eye(20,2),MOCK_TRANSFORM_ID)
        self.transform = self.enterContext(patch.object(
            regression,"load_frozen_transform",return_value=transform))
        self.asset_read = self.enterContext(patch.object(
            np,"load",side_effect=AssertionError("real asset loading forbidden")))
        self.torch_load = self.enterContext(patch.object(
            torch,"load",side_effect=AssertionError("checkpoint loading forbidden")))

    def tearDown(self):
        self.asset_read.assert_not_called()
        self.torch_load.assert_not_called()

    def close(self,key,actual,expected):
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=0)
        ERRORS[key] = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))

    def test_protocol_capability_matrix_and_reference_exclusion(self):
        ordinary = ("state-mean","ht","f-z2","lin-z2","kernel-old","kernel-new","j")
        self.assertEqual(ORDINARY_METHOD_IDS,ordinary)
        point = {"context.mu","context.ite","context.ate","pair.ate"}
        for method in METHODS:
            cap = method_capability(method)
            expected = ({"pair.ate"} if method in baselines.METHOD_IDS else
                        {"pair.ate","pair.ate_interval"} if method == "oracle-pair" else
                        point|{"context.ate_interval"} if method in INTERVAL_CONTEXT_METHODS else point)
            self.assertEqual(cap.outputs,frozenset(expected))
            self.assertEqual(cap.ordinary_ranking,method in ordinary)
            self.assertEqual(cap.privileged,method.startswith("oracle-"))
            self.assertEqual(cap.reference_only,method.startswith("oracle-"))
            METHOD_EVIDENCE[method].add("capability")

    def test_nine_method_binding_node_order_and_arm_roles(self):
        h = Harness()
        for method in METHODS:
            with self.subTest(method=method):
                result = h.run(method)
                for name in ("task_id","request_id","pair_id","target_id","input_hashes",
                             "protocol_hash","context_ids","arm_order"):
                    self.assertEqual(getattr(result,name),getattr(h.parent,name))
                self.assertEqual(result.method_id,method)
                self.assertEqual(result.status,"POINT")
                self.assertEqual(result.output_status["ate"],("OK",)*3)
                self.assertEqual(result.execution_status,"OK")
                np.testing.assert_array_equal(result.node_ids,np.arange(200))
                if method in CONTEXT_METHODS:
                    for k,arm in enumerate(result.arm_results):
                        self.assertEqual(arm.request,h.children[k])
                        self.assertEqual(arm.request.parent_request_id,h.parent.request_id)
                        self.assertEqual(arm.context_ids,(h.task.context_refs[k].context_id,))
                        self.assertEqual(arm.input_hashes,(h.task.context_refs[k].input_hash,))
                        self.assertEqual(arm.output_status["mu"],"OK")
                        self.assertEqual(arm.output_status["ite"],("OK",)*3)
                else:
                    self.assertEqual(result.arm_results,(None,None))
                METHOD_EVIDENCE[method].add("identity")

    def test_all_not_identified_zero_numerics_and_verbatim_reason(self):
        h = Harness(status="NOT_IDENTIFIED")
        with patch.object(state,"_compute",side_effect=AssertionError("forbidden")) as sm, \
             patch.object(regression,"_compute",side_effect=AssertionError("forbidden")) as reg:
            for method in METHODS:
                with self.subTest(method=method):
                    result = h.run(method)
                    self.assertEqual(result.status,"NOT_IDENTIFIED")
                    self.assertEqual(result.execution_status,"NOT_IDENTIFIED")
                    self.assertEqual(result.failure_type,h.parent.failure_type)
                    self.assertEqual(result.failure_message,h.parent.failure_message)
                    self.assertEqual(dict(result.failure_provenance),dict(h.parent.failure_provenance))
                    self.assertEqual(result.output_status["ate"],("FAILED",)*3)
                    self.assertTrue(np.isnan(result.ate).all())
                    METHOD_EVIDENCE[method].add("identification")
        sm.assert_not_called();reg.assert_not_called()
        for callback in (h.old_loader,h.old_reader,h.new_loader,h.new_reader,
                         h.j_loader,h.interval,h.context_core,h.joint_core):
            callback.assert_not_called()
        self.assertEqual(h.old_model.calls,0)
        self.assertEqual(h.new_model.calls,0)
        self.assertEqual(h.j_model.calls,[])

    def test_identity_conflicts_block_every_entry_before_core(self):
        for method in METHODS:
            h = Harness()
            h.parent = replace(h.parent,input_hashes=("0"*64,"1"*64))
            with self.subTest(method=method),self.assertRaises(ValueError):
                h.run(method)
            for loader in (h.old_loader,h.new_loader,h.j_loader,h.context_core,h.joint_core):
                loader.assert_not_called()

    def test_support_flags_and_all_original_input_hashes_remain_unchanged(self):
        h = Harness()
        before = tuple(signature(t) for t in h.task.context_refs)
        h_before = h.h.tobytes()
        for method in METHODS:
            result = h.run(method)
            bundles = (result,)+tuple(a for a in result.arm_results if a is not None)
            for k,bundle in enumerate(bundles):
                reference = support_for_pair(h.task) if k == 0 else support_for_task(h.task.context_refs[k-1])
                for name in ("population_support","design_support","empirical_support_mask"):
                    actual,expected = getattr(bundle,name),getattr(reference,name)
                    self.assertEqual(actual.shape,expected.shape)
                    self.assertEqual(actual.dtype,expected.dtype)
                    np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(bundle.extrapolation_flags["empirical"],reference.empirical_extrapolation)
                design = (tuple(s.context_design_extrapolation for s in reference.context_supports)
                          if k == 0 else reference.context_design_extrapolation)
                np.testing.assert_array_equal(bundle.extrapolation_flags["context_design"],design)
                self.assertFalse(bundle.empirical_support_mask.all())
                self.assertTrue(bundle.extrapolation_flags["empirical"].any())
            self.assertEqual(tuple(signature(t) for t in h.task.context_refs),before)
            self.assertEqual(h.h.tobytes(),h_before)
            METHOD_EVIDENCE[method].add("support_and_immutability")

    def test_independent_references_respect_different_estimators(self):
        h = Harness()
        groups,counts,means = hand_records(h.task)
        for method in ("state-mean","ht"):
            result = h.run(method)
            response = means if method == "state-mean" else np.array([
                sum(groups[s])*8/200 for s in ("00","01","10","11")])
            self.close(method,result.ate,[response[2]-response[0],response[3]-response[2],
                                          response[3]-response[0]])
        for method in ("kernel-old","kernel-new"):
            self.close(method,h.run(method).ate,(expected_ate(0)+expected_ate(1))/2)
        # Independent scalar sums of the distinct scripted J / Oracle outputs.
        for method in ("j","oracle-context"):
            if method == "j":
                values = [[sum(i+3+4*a for i in range(200))/200,
                           sum(2*i+5+8*a for i in range(200))/200,
                           sum(3*i+8+12*a for i in range(200))/200] for a in (0,1)]
            else:
                values = [[sum(i+3+6*a for i in range(200))/200,
                           sum(2*i+5+10*a for i in range(200))/200,
                           sum(3*i+8+16*a for i in range(200))/200] for a in (0,1)]
            self.close(method,h.run(method).ate,[(values[0][k]+values[1][k])/2 for k in range(3)])
        self.close("oracle-pair",h.run("oracle-pair").ate,[7.25,-2.,5.25])
        # Regression numerical reference is intentionally NOT repeated here.
        self.assertEqual(h.run("f-z2").asset_hashes["feature_transform.npz"],MOCK_TRANSFORM_ID)
        self.assertEqual(h.run("lin-z2").asset_hashes["feature_transform.npz"],MOCK_TRANSFORM_ID)

    def test_interval_scope_and_sampling_uncertainty_across_all_capabilities(self):
        h = Harness()
        for method in METHODS:
            result = h.run(method)
            self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
            self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)
            self.assertTrue(np.isnan(result.mu).all() and np.isnan(result.ite).all())
            if method in INTERVAL_CONTEXT_METHODS:
                self.assertEqual(result.intervals,())
                self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
                self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
                for k,arm in enumerate(result.arm_results):
                    self.assertEqual(arm.method_id,method)
                    self.assertEqual(len(arm.intervals),3)
                    self.assertEqual(arm.sampling_uncertainty.source,"model_posterior")
                    self.assertEqual(arm.sampling_uncertainty.statistical_unit,"context")
                    self.assertIsNone(arm.sampling_uncertainty.effective_sample_count)
                    for record in arm.intervals:
                        self.assertEqual(record.context_id,h.task.context_refs[k].context_id)
                        self.assertEqual(record.source,"model_posterior")
                if method == "oracle-context":
                    a,b = result.arm_results
                    self.assertNotEqual(a.intervals[0].lower,b.intervals[0].lower)
                    self.assertEqual(a.intervals[0].lower,101.5)
                    self.assertEqual(b.intervals[0].lower,107.5)
            elif method == "oracle-pair":
                self.assertEqual(result.output_status["interval"],("OK",)*3)
                self.assertEqual(result.sampling_uncertainty.source,"joint_model_posterior")
                self.assertEqual(result.sampling_uncertainty.statistical_unit,"pair")
                self.assertTrue(all(r.context_id is None and r.source == "joint_model_posterior"
                                    for r in result.intervals))
                self.assertNotIn("context.mu",result.method_capability.outputs)
            else:
                self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
                self.assertEqual(result.intervals,())
                self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
            METHOD_EVIDENCE[method].add("interval_scope")

    def test_unrequested_intervals_never_invalidate_point_requests(self):
        h = Harness(outputs=("ate",))
        h.interval.side_effect = FloatingPointError("unused J interval failure")
        def fail_context(**kw):
            raw = context_result(int(kw["treatment"][0]))
            return replace(raw,intervals=tuple(replace(r,status="FAILED",lower=None,upper=None,
                failure_reason="unused interval failure") for r in raw.intervals))
        h.context_core.side_effect = fail_context
        raw = joint_result()
        h.joint_core.return_value = replace(raw,status="FAILED",intervals=tuple(
            replace(r,status="FAILED",lower=None,upper=None,failure_reason="unused interval failure")
            for r in raw.intervals))
        for method in METHODS:
            result = h.run(method)
            self.assertEqual(result.status,"POINT")
            self.assertEqual(result.execution_status,"OK")
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            self.assertEqual(result.output_status["interval"],("NOT_REQUESTED",)*3)
            self.assertEqual(result.intervals,())
        h.interval.assert_not_called()

    def test_interval_failure_provenance_j_versus_non_j_and_valid_ate(self):
        h = Harness()
        h.interval.side_effect = FloatingPointError("J exact interval reason")
        def fail_context(**kw):
            raw = context_result(int(kw["treatment"][0]))
            return replace(raw,intervals=tuple(replace(r,status="FAILED",lower=None,upper=None,
                failure_reason="Oracle-context exact interval reason") for r in raw.intervals))
        h.context_core.side_effect = fail_context
        raw = joint_result()
        h.joint_core.return_value = replace(raw,status="FAILED",intervals=tuple(
            replace(r,status="FAILED",lower=None,upper=None,
                    failure_reason="Oracle-pair exact interval reason") for r in raw.intervals))
        for method in ("j","oracle-context","oracle-pair"):
            result = h.run(method)
            self.assertEqual(result.status,"POINT")
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            bundles = result.arm_results if method != "oracle-pair" else (result,)
            self.assertEqual(result.execution_status,"FAILED" if method == "oracle-pair" else "OK")
            for arm in bundles:
                self.assertEqual(arm.output_status["interval"],("FAILED",)*3)
                self.assertEqual(arm.sampling_uncertainty.status,"FAILED")
                self.assertIsNone(arm.sampling_uncertainty.effective_sample_count)
                if method == "j":
                    self.assertEqual(arm.sampling_uncertainty.source,"model_posterior")
                    self.assertEqual(arm.sampling_uncertainty.statistical_unit,"context")
                    self.assertEqual(arm.failure_provenance["intervals"]["Direct"]["message"],
                                     "J exact interval reason")
                else:
                    self.assertIsNone(arm.sampling_uncertainty.source)
                    self.assertIsNone(arm.sampling_uncertainty.statistical_unit)
                    origin = arm.failure_provenance["uncertainty_origin"]
                    self.assertEqual(origin["statistical_unit"],"pair" if method == "oracle-pair" else "context")
                    self.assertIn("exact interval reason",arm.failure_provenance["intervals"]["Direct"])
                self.assertTrue(all(r.source is not None for r in arm.intervals))

    def test_not_estimable_differs_from_failed_and_ht_empty_sum_remains_valid(self):
        h = Harness(task=missing00())
        for method in ("state-mean","f-z2","lin-z2"):
            result = h.run(method)
            self.assertEqual(result.status,"POINT")
            self.assertEqual(result.output_status["ate"],("NOT_ESTIMABLE","OK","NOT_ESTIMABLE"))
            self.assertEqual(result.execution_status,"NOT_ESTIMABLE")
            self.assertTrue(np.isnan(result.ate[[0,2]]).all())
            self.assertTrue(np.isfinite(result.ate[1]))
        ht = h.run("ht")
        self.assertEqual(ht.execution_status,"OK")
        self.assertEqual(ht.output_status["ate"],("OK",)*3)
        self.assertFalse(ht.empirical_support_mask[:,0,0].any())

    def test_partial_failures_preserve_other_effects_in_context_and_joint_methods(self):
        h = Harness()
        def corrupt(means):
            means = means.copy();means[3,0] = np.nan
            return means
        h.old_model.mean_change = lambda means: means.__setitem__((3,0),np.nan)
        h.new_model.output_change = corrupt
        original = h.j_model.predict_arm_means
        h.j_model.predict_arm_means = lambda **inputs:corrupt(original(**inputs))
        def corrupt_context(**kw):
            raw = context_result(int(kw["treatment"][0]));ate = raw.ate.copy();ate[[0,2]] = np.nan
            return replace(raw,ate=ate,failure_reasons={"point":"scripted point field failure"})
        h.context_core.side_effect = corrupt_context
        raw = joint_result();ate = raw.ate.copy();ate[[0,2]] = np.nan
        h.joint_core.return_value = replace(raw,status="FAILED",ate=ate)
        for method in CONTEXT_METHODS+("oracle-pair",):
            result = h.run(method)
            self.assertEqual(result.status,"POINT")
            self.assertEqual(result.output_status["ate"],("FAILED","OK","FAILED"))
            self.assertTrue(np.isnan(result.ate[[0,2]]).all())
            self.assertTrue(np.isfinite(result.ate[1]))
            self.assertEqual(result.execution_status,"FAILED")

    def test_core_failure_isolation_does_not_delete_other_methods_or_requests(self):
        h = Harness()
        original = regression._compute
        def fail(inputs,method):
            if method == "f-z2": raise ArithmeticError("one method only")
            return original(inputs,method)
        with patch.object(regression,"_compute",side_effect=fail):
            results = {method:h.run(method) for method in METHODS}
        self.assertEqual(tuple(results),METHODS)
        self.assertEqual(results["f-z2"].failure_message,"one method only")
        self.assertEqual(results["f-z2"].failure_provenance["attempts"],1)
        for method,result in results.items():
            self.assertEqual(result.request_id,h.parent.request_id)
            self.assertEqual(result.method_id,method)
            self.assertEqual(result.status,"POINT")
            if method != "f-z2": self.assertEqual(result.output_status["ate"],("OK",)*3)

    def test_observed_input_budgets_and_privileged_oracle_boundary(self):
        h = Harness()
        captured = []
        sm,reg = state._compute,regression._compute
        def spy(fn):
            def call(inputs,method):
                captured.extend(inputs)
                return fn(inputs,method)
            return call
        with patch.object(state,"_compute",side_effect=spy(sm)), \
             patch.object(regression,"_compute",side_effect=spy(reg)):
            for method in baselines.METHOD_IDS: h.run(method)
        for model,name,method in ((h.old_model,"predict","kernel-old"),
                                  (h.new_model,"predict_arm_means","kernel-new"),
                                  (h.j_model,"predict_arm_means","j")):
            original = getattr(model,name)
            def call(_original=original,**inputs):
                captured.append(inputs)
                return _original(**inputs)
            with patch.object(model,name,side_effect=call):
                h.run(method)
        for payload in captured:
            self.assertEqual(set(payload),set(c.FACTUAL_KEYS))
            for value in payload.values():
                self.assertIs(type(value),np.ndarray)
            self.assertIn(payload["outcomes_observed"].dtype,(np.dtype("float64"),np.dtype("float32")))
        for payload in captured[:8]:
            self.assertEqual(payload["outcomes_observed"].dtype,np.dtype("float64"))
        h.run("oracle-context");h.run("oracle-pair")
        allowed = set(c.FACTUAL_KEYS)|{"true_h","noise_sd","context_identity"}
        self.assertTrue(all(set(call.kwargs) == allowed for call in h.context_core.call_args_list))
        self.assertEqual(set(h.joint_core.call_args.kwargs),{
            "true_h","noise_sd","adjacency","treatment0","treatment1","observed_indices0",
            "observed_indices1","outcomes_observed0","outcomes_observed1","pair_identity"})
        for call in h.context_core.call_args_list:
            self.assertFalse(np.shares_memory(call.kwargs["true_h"],h.h))
            self.assertFalse(call.kwargs["true_h"].flags.writeable)
            self.assertNotIn("beta",call.kwargs)

    def test_capability_and_source_forgery_are_rejected_by_public_bundle(self):
        h = Harness()
        for method in METHODS:
            result = h.run(method)
            states = dict(result.output_status);states["mu"] = "OK"
            with self.subTest(method=method),self.assertRaises(ValueError):
                replace(result,output_status=states,mu=np.zeros((200,2,2)))
        joint = h.run("oracle-pair")
        wrong = tuple(replace(r,source="model_posterior") for r in joint.intervals)
        with self.assertRaises(ValueError): replace(joint,intervals=wrong)
        with self.assertRaises(ValueError):
            replace(joint,sampling_uncertainty=SamplingUncertainty("OK","model_posterior","context",None))
        for method in ("j","oracle-context"):
            arm = h.run(method).arm_results[0]
            with self.assertRaises(ValueError):
                replace(arm,sampling_uncertainty=SamplingUncertainty("OK","model_posterior","pair",None))
        for method in ("state-mean","ht","f-z2","lin-z2","kernel-old","kernel-new"):
            result = h.run(method)
            with self.assertRaises(ValueError):
                replace(result,sampling_uncertainty=SamplingUncertainty(
                    "OK","model_posterior","context",None))


if __name__ == "__main__":
    unittest.main()
