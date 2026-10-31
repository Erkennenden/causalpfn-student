"""S4.2 only: pure memory fixtures and injected mocks, no real model/assets.

Independent references use explicit formulas/scalar loops. Absolute tolerance
is fixed at 1e-12 BEFORE first execution; no production target calculations
are called to construct expected values.
"""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from star_v2 import contracts as c, model_adapters as m
from star_v2.config import CONFIG
from star_v2.estimation import EstimationRequest, EstimateBundle, PairEstimateBundle
from star_v2.support import support_for_pair, require_plugin_support, PluginSupportError
from test_pair_data import fixture_pair
from test_contracts import base_bytes

ATOL = 1e-12
ERRORS = {}


def point(arm=0):
    i = np.arange(200,dtype=np.float64)
    return np.column_stack([1+i/8+2*arm,3+i/2-7*arm,
                            8+3*i/8+5*arm,17+7*i/8+11*arm])


def expected_ite(arm=0):
    return np.array([[7+node/4+3*arm,9+node/2+6*arm,16+3*node/4+9*arm]
                     for node in range(200)],dtype=np.float64)


def expected_ate(arm=0):
    return np.array([sum(7+node/4+3*arm for node in range(200))/200,
                     sum(9+node/2+6*arm for node in range(200))/200,
                     sum(16+3*node/4+9*arm for node in range(200))/200])


def pair_fixture(bound=False):
    return c.pair_from_private(fixture_pair(),
        **({"target_id":CONFIG.target_id,"request_id":"bound"} if bound else {}))


def parent(task, *, outputs=("mu","ite","ate"), status="POINT", **kw):
    return EstimationRequest.from_task(
        task,status=status,target_id=CONFIG.target_id,
        request_id="bound" if task.request_id else "upstream-parent",
        requested_outputs=outputs,**kw)


def setup_pair(task=None, **kw):
    task = pair_fixture() if task is None else task
    request = parent(task,**kw)
    ids = ("bound","bound") if task.request_id else ("upstream-arm-0","upstream-arm-1")
    children = m.context_requests_from_parent(task,request,request_ids=ids)
    return task,request,children


def run_pair(task=None, *, method="kernel-old", predictions=None, **kw):
    task,request,children = setup_pair(task,**kw)
    predictors = tuple(Mock(return_value=point(k)) for k in (0,1)) if predictions is None else predictions
    return m.estimate_pair(task,request,method,context_requests=children,predictors=predictors)


class ModelAdapterTests(unittest.TestCase):
    def close(self,name,actual,expected):
        actual,expected = np.asarray(actual),np.asarray(expected)
        finite = np.isfinite(actual)&np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        ERRORS[name] = max(ERRORS.get(name,0.),error)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=ATOL,equal_nan=True)

    def test_canonical_non_symmetric_axis_mapping(self):
        task,request,children = setup_pair()
        result = m.estimate_context(task.context_refs[0],children[0],"kernel-old",
                                    Mock(return_value=point()))
        self.assertIs(type(result),EstimateBundle)
        self.assertEqual(result.arm_order,("00","01","10","11"))
        self.assertEqual(result.effect_order,("Direct","Spillover","Total"))
        self.close("mu_00",result.mu[:,0,0],[1+node/8 for node in range(200)])
        self.close("mu_01",result.mu[:,0,1],[3+node/2 for node in range(200)])
        self.close("mu_10",result.mu[:,1,0],[8+3*node/8 for node in range(200)])
        self.close("mu_11",result.mu[:,1,1],[17+7*node/8 for node in range(200)])
        self.close("ite",result.ite,expected_ite())
        self.close("ate",result.ate,expected_ate())
        self.close("total_identity",result.ite[:,2],result.ite[:,0]+result.ite[:,1])

    def test_all_200_nodes_ate_not_observed_50(self):
        task,request,children = setup_pair()
        result = m.estimate_context(task.context_refs[0],children[0],"j",Mock(return_value=point()))
        self.close("all_graph_ate",result.ate,expected_ate())
        observed = task.context_refs[0].observed_indices
        wrong = np.array([sum(7+int(node)/4 for node in observed)/50,
                          sum(9+int(node)/2 for node in observed)/50,
                          sum(16+3*int(node)/4 for node in observed)/50])
        self.assertGreater(np.linalg.norm(result.ate-wrong),1.)
        self.assertEqual(result.diagnostics["query_nodes"],200)

    def test_pair_different_arms_independent_hand_mean(self):
        result = run_pair()
        self.assertIs(type(result),PairEstimateBundle)
        self.close("pair_ate",result.ate,[33.375,61.75,95.125])
        self.close("arm0",result.arm_results[0].ate,expected_ate(0))
        self.close("arm1",result.arm_results[1].ate,expected_ate(1))
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertFalse(result.diagnostics["arms_independent"])
        self.assertEqual(result.diagnostics["n_unique_observed_nodes"],50)
        self.assertEqual(result.diagnostics["n_factual_records"],100)

    def test_strict_five_keyword_arrays_no_privileged_storage(self):
        private = fixture_pair()
        task = c.pair_from_private(private)
        task,request,children = setup_pair(task)
        def predictor(**payload):
            self.assertEqual(tuple(payload),c.FACTUAL_KEYS)
            self.assertEqual(payload["outcomes_observed"].dtype,np.float64)
            for name,value in payload.items():
                self.assertFalse(value.flags.writeable)
                self.assertIs(type(base_bytes(value)),bytes)
                self.assertEqual(len(base_bytes(value)),value.nbytes)
                for context in task.context_refs:
                    for source_name in c.FACTUAL_KEYS:
                        self.assertFalse(np.shares_memory(value,getattr(context,source_name)))
                for source in (private.contexts[0].full_outcomes,private.noise,
                               private.mechanism.h,private.mechanism.coefficients,
                               private.mechanism.mu):
                    self.assertFalse(np.shares_memory(value,source))
                with self.assertRaises(ValueError):
                    value.setflags(write=True)
            return point()
        result = m.estimate_pair(task,request,"kernel-new",context_requests=children,
                                predictors=(predictor,predictor))
        self.assertEqual(result.execution_status,"OK")

    def test_private_objects_and_extra_keywords_rejected(self):
        private = fixture_pair()
        with self.assertRaises(c.ContractError):
            m.estimate_context(private.contexts[0],None,"j",Mock())
        with self.assertRaises(c.ContractError):
            m.estimate_pair(private,None,"j",context_requests=(),predictors=())
        task,request,children = setup_pair()
        for name in ("truth","beta","h","full_y","noise","support","rng_identity"):
            with self.assertRaises(TypeError):
                m.estimate_context(task.context_refs[0],children[0],"j",Mock(),
                                   **{name:np.zeros(200)})

    def test_unbound_requests_explicit_child_ids_and_parent_links(self):
        task,request,children = setup_pair()
        result = run_pair(task)
        for k,arm in enumerate(result.arm_results):
            self.assertEqual(arm.request_id,children[k].request_id)
            self.assertEqual(arm.request.parent_request_id,request.request_id)
            self.assertEqual(arm.input_hash,task.context_refs[k].input_hash)
            self.assertEqual(arm.context_ids,(task.context_ids[k],))
            self.assertEqual(arm.target_id,request.target_id)
            self.assertEqual(arm.status,request.status)
            self.assertIsNone(task.context_refs[k].request_id)
        self.assertEqual(result.input_hashes,request.input_hashes)
        self.assertEqual(result.request_id,request.request_id)

    def test_bound_request_ids_preserved_without_rewriting(self):
        task,request,children = setup_pair(pair_fixture(bound=True))
        predictors = (Mock(return_value=point()),Mock(return_value=point(1)))
        result = m.estimate_pair(task,request,"j",context_requests=children,predictors=predictors)
        self.assertEqual(result.request_id,"bound")
        self.assertEqual(tuple(arm.request_id for arm in result.arm_results),("bound","bound"))
        self.assertEqual(len({(arm.request_id,arm.task_id) for arm in result.arm_results}),2)
        for context in task.context_refs:
            self.assertEqual(context.request_id,"bound")
            c.validate_task(context)
        with self.assertRaises(c.ContractError):
            m.context_requests_from_parent(task,request,request_ids=("changed","changed"))

    def test_no_generated_or_missing_request_ids(self):
        task,request,children = setup_pair()
        with self.assertRaises(TypeError):
            m.context_requests_from_parent(task,request)
        for ids in ((),("only-one",),("",None)):
            with self.assertRaises(c.ContractError):
                m.context_requests_from_parent(task,request,request_ids=ids)

    def test_parent_child_conflicts_rejected_before_predictors(self):
        task,request,children = setup_pair()
        predictors = (Mock(),Mock())
        mutations = (
            (children[1],children[0]),
            (replace(children[0],parent_request_id="wrong"),children[1]),
            (replace(children[0],requested_outputs=("ate",)),children[1]),
            (replace(children[0],status="NOT_IDENTIFIED",failure_type="ID",failure_message="bad"),children[1]))
        for invalid in mutations:
            with self.assertRaises(c.ContractError):
                m.estimate_pair(task,request,"j",context_requests=invalid,predictors=predictors)
        for predictor in predictors:
            predictor.assert_not_called()

    def test_not_identified_zero_point_and_interval_calls(self):
        task,request,children = setup_pair(
            outputs=("mu","ite","ate","interval"),status="NOT_IDENTIFIED",
            failure_type="UPSTREAM_ID",failure_message="original exact reason",
            failure_provenance={"identification":{"code":"UP17"}})
        predictors,intervals = (Mock(),Mock()),(Mock(),Mock())
        for method in m.MODEL_METHODS:
            result = m.estimate_pair(task,request,method,context_requests=children,
                                    predictors=predictors,interval_predictors=intervals)
            self.assertEqual((result.status,result.execution_status),("NOT_IDENTIFIED",)*2)
            self.assertEqual(result.failure_message,request.failure_message)
            self.assertEqual(dict(result.failure_provenance),dict(request.failure_provenance))
            self.assertTrue(np.isnan(result.ate).all())
            for arm in result.arm_results:
                self.assertEqual(arm.execution_status,"NOT_IDENTIFIED")
                self.assertTrue(np.isnan(arm.mu).all())
        for callback in (*predictors,*intervals):
            callback.assert_not_called()

    def test_whole_predictor_failure_no_fake_partial_success(self):
        task,request,children = setup_pair(outputs=("mu","ite","ate","interval"))
        first = Mock(side_effect=ArithmeticError("whole model failed"))
        second = Mock(return_value=point(1))
        interval0 = Mock()
        interval1 = Mock(return_value=m.ModelIntervals(np.array([0.,1.,2.]),np.array([3.,4.,5.])))
        result = m.estimate_pair(task,request,"j",context_requests=children,predictors=(first,second),
                                interval_predictors=(interval0,interval1))
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        failed = result.arm_results[0]
        self.assertTrue(np.isnan(failed.mu).all())
        self.assertTrue(np.isnan(failed.ite).all())
        self.assertTrue(np.isnan(failed.ate).all())
        self.assertEqual(failed.failure_message,"whole model failed")
        self.assertEqual(result.arm_results[1].output_status["ate"],("OK",)*3)
        interval0.assert_not_called()
        self.assertEqual(first.call_count,1)
        self.assertEqual(second.call_count,1)

    def test_partial_nonfinite_10_fails_direct_spillover_preserves_total(self):
        damaged = point()
        damaged[150,2] = np.nan
        result = run_pair(predictions=(Mock(return_value=damaged),Mock(return_value=point(1))))
        self.assertEqual(result.output_status["ate"],("FAILED","FAILED","OK"))
        self.assertTrue(np.isnan(result.ate[:2]).all())
        self.assertEqual(result.ate[2],95.125)
        self.assertEqual(result.arm_results[0].output_status["ite"],("FAILED","FAILED","OK"))
        self.assertEqual(result.arm_results[0].output_status["mu"],"FAILED")
        self.assertTrue(np.isnan(result.arm_results[0].mu).all())
        self.assertIn("Direct",result.arm_results[0].failure_provenance["effects"])

    def test_nonfinite_unused_01_does_not_erase_valid_effects(self):
        damaged = point();damaged[60,1] = np.inf
        result = run_pair(predictions=(Mock(return_value=damaged),Mock(return_value=point(1))))
        self.assertEqual(result.arm_results[0].execution_status,"FAILED")
        self.assertEqual(result.arm_results[0].output_status["mu"],"FAILED")
        self.assertEqual(result.arm_results[0].output_status["ate"],("OK",)*3)
        self.assertEqual(result.execution_status,"OK")
        self.close("unused01_pair",result.ate,[33.375,61.75,95.125])

    def test_wrong_shape_dtype_or_axis_is_complete_failure(self):
        for output in (np.zeros((200,2,2)),np.zeros((199,4)),np.zeros((200,4),dtype=np.int64),
                       {"truth":np.zeros((200,4))},
                       m.ModelPrediction(point(),arm_order=("00","10","01","11"))):
            result = run_pair(predictions=(Mock(return_value=output),Mock(return_value=point(1))))
            self.assertEqual(result.output_status["ate"],("FAILED",)*3)
            self.assertTrue(np.isnan(result.arm_results[0].ate).all())
            self.assertEqual(result.arm_results[1].output_status["ate"],("OK",)*3)

    def test_compose_preserves_not_estimable_and_engineering_priority(self):
        original = run_pair()
        left = original.arm_results[0]
        right = original.arm_results[1]
        left_ate = left.ate.copy();left_ate[0] = np.nan
        left = replace(left,ate=left_ate,
            output_status={**left.output_status,"ate":("NOT_ESTIMABLE","OK","OK")},
            failure_type="UPSTREAM_ESTIMABILITY",failure_message="direct not estimable")
        task,request,children = setup_pair()
        result = m.compose_pair(task,request,"kernel-old",(left,right))
        self.assertEqual(result.output_status["ate"],("NOT_ESTIMABLE","OK","OK"))
        self.assertEqual(result.execution_status,"NOT_ESTIMABLE")
        right_ate = right.ate.copy();right_ate[0] = np.nan
        right = replace(right,ate=right_ate,
            output_status={**right.output_status,"ate":("FAILED","OK","OK")},
            failure_type="NUMERIC",failure_message="direct numeric failure")
        result = m.compose_pair(task,request,"kernel-old",(left,right))
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.output_status["ate"],("FAILED","OK","OK"))
        self.assertEqual(result.arm_results[0].failure_message,"direct not estimable")
        self.assertEqual(result.arm_results[1].failure_message,"direct numeric failure")

    def test_context_intervals_success_no_pair_interval(self):
        task,request,children = setup_pair(outputs=("mu","ite","ate","interval"))
        callback = Mock(return_value=m.ModelIntervals(
            np.array([1.,2.,3.]),np.array([9.,10.,11.]),diagnostics={"kind":"artificial"}))
        result = m.estimate_pair(task,request,"j",context_requests=children,
                                predictors=(Mock(return_value=point()),Mock(return_value=point(1))),
                                interval_predictors=(callback,callback))
        self.assertEqual(callback.call_count,2)
        self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
        self.assertEqual(result.intervals,())
        self.assertEqual(tuple(len(v) for v in result.context_intervals),(3,3))
        for arm in result.arm_results:
            self.assertEqual(arm.output_status["interval"],("OK",)*3)
            self.assertEqual(arm.sampling_uncertainty.source,"model_posterior")
            self.assertEqual(arm.sampling_uncertainty.statistical_unit,"context")
            self.assertIsNone(arm.sampling_uncertainty.effective_sample_count)
            self.assertEqual(arm.intervals[0].context_id,arm.context_ids[0])

    def test_interval_only_exception_keeps_valid_point_and_pair_ate(self):
        task,request,children = setup_pair(outputs=("mu","ite","ate","interval"))
        intervals = (Mock(side_effect=ValueError("interval failure only")),
                     Mock(side_effect=FloatingPointError("second interval reason")))
        result = m.estimate_pair(task,request,"j",context_requests=children,
                                predictors=(Mock(return_value=point()),Mock(return_value=point(1))),
                                interval_predictors=intervals)
        self.assertEqual(result.execution_status,"OK")
        self.close("interval_failure_keeps_pair",result.ate,[33.375,61.75,95.125])
        for arm in result.arm_results:
            self.assertEqual(arm.execution_status,"FAILED")
            self.assertEqual(arm.output_status["ate"],("OK",)*3)
            self.assertEqual(arm.output_status["interval"],("FAILED",)*3)
            self.assertEqual(len(arm.intervals),3)
        self.assertEqual(result.arm_results[0].failure_provenance["intervals"]["Direct"]["message"],
                         "interval failure only")

    def test_invalid_interval_effect_isolated_from_valid_intervals(self):
        task,request,children = setup_pair(outputs=("mu","ite","ate","interval"))
        callback = Mock(return_value=m.ModelIntervals(np.array([1.,np.nan,4.]),np.array([2.,5.,3.])))
        result = m.estimate_context(task.context_refs[0],children[0],"j",Mock(return_value=point()),
                                    interval_predictor=callback)
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.output_status["interval"],("OK","FAILED","FAILED"))
        self.assertEqual(result.sampling_uncertainty.status,"OK")
        self.assertIsNone(result.intervals[1].lower)
        self.assertEqual(result.intervals[0].lower,1.)

    def test_missing_or_wrong_level_interval_remains_failed_record(self):
        task,request,children = setup_pair(outputs=("ate","interval"))
        for interval in (None,Mock(return_value=m.ModelIntervals(np.zeros(3),np.ones(3),level=.9))):
            result = m.estimate_context(task.context_refs[0],children[0],"j",
                                        Mock(return_value=point()),interval_predictor=interval)
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            self.assertEqual(result.output_status["interval"],("FAILED",)*3)
            self.assertEqual(len(result.intervals),3)

    def test_unsupported_or_not_requested_intervals_are_never_called(self):
        for method,outputs,expected in (
                ("kernel-old",("mu","ite","ate","interval"),"UNSUPPORTED"),
                ("j",("mu","ite","ate"),"NOT_REQUESTED")):
            task,request,children = setup_pair(outputs=outputs)
            interval = Mock()
            result = m.estimate_context(task.context_refs[0],children[0],method,
                                        Mock(return_value=point()),interval_predictor=interval)
            interval.assert_not_called()
            self.assertEqual(result.output_status["interval"],(expected,)*3)
            self.assertEqual(result.sampling_uncertainty.status,expected)

    def test_point_diagnostics_are_immutable_json_like(self):
        task,request,children = setup_pair()
        source = {"weights":[.25,.75],"nested":{"version":1}}
        prediction = m.ModelPrediction(point(),diagnostics=source)
        result = m.estimate_context(task.context_refs[0],children[0],"j",Mock(return_value=prediction))
        source["weights"][0] = 999
        self.assertEqual(result.diagnostics["point"]["weights"],(.25,.75))
        with self.assertRaises(TypeError):
            result.diagnostics["point"]["nested"]["version"] = 2

    def test_privileged_or_tensor_like_diagnostics_rejected(self):
        for diagnostics in ({"truth":[1.]},{"beta":[1.]},{"h":[1.]},
                            {"noise":[1.]},{"full_y":[1.]},{"weights":np.ones(4)}):
            result = run_pair(predictions=(
                Mock(return_value=m.ModelPrediction(point(),diagnostics=diagnostics)),
                Mock(return_value=point(1))))
            self.assertEqual(result.arm_results[0].execution_status,"FAILED")
            self.assertEqual(result.output_status["ate"],("FAILED",)*3)
            self.assertNotIn("point",result.arm_results[0].diagnostics)

    def test_support_and_empirical_extrapolation_independent_of_finite_models(self):
        task,request,children = setup_pair()
        support = support_for_pair(task)
        result = run_pair(task)
        for name in ("population_support","design_support","empirical_support_mask"):
            np.testing.assert_array_equal(getattr(result,name),getattr(support,name))
        self.assertTrue(result.population_support.all())
        self.assertFalse(result.empirical_support_mask.all())
        self.assertTrue(np.isfinite(result.ate).all())
        self.assertIn("context.mu",result.method_capability.outputs)
        for arm in result.arm_results:
            query = tuple(map(int,np.argwhere(~arm.empirical_support_mask)[0]))
            self.assertTrue(arm.extrapolation_flags["empirical"][query])
            with self.assertRaises(PluginSupportError):
                require_plugin_support(arm.support,(query,))
            self.assertTrue(require_plugin_support(
                arm.support,(query,),extrapolation_authorized=True).allowed)

    def test_original_hashes_float64_and_bytes_unchanged(self):
        task,request,children = setup_pair()
        arrays = [getattr(context,k) for context in task.context_refs for k in c.FACTUAL_KEYS]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        hashes = tuple(context.input_hash for context in task.context_refs)
        result = run_pair(task)
        self.assertEqual(result.input_hashes,hashes)
        self.assertEqual(before,[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])
        for context in task.context_refs:
            self.assertEqual(context.outcomes_observed.dtype,np.float64)
            c.validate_task(context)
        with self.assertRaises(ValueError):
            result.ate.setflags(write=True)

    def test_repeated_execution_deterministic_without_rng_or_assets(self):
        task,request,children = setup_pair()
        with patch.object(np,"load",side_effect=AssertionError("no asset deserialization")),\
             patch.object(np.random,"default_rng",side_effect=AssertionError("no RNG")):
            first = run_pair(task)
            second = run_pair(task)
        self.close("repeat",first.ate,second.ate)
        self.assertEqual(dict(first.diagnostics),dict(second.diagnostics))
        self.assertEqual(first.input_hashes,second.input_hashes)

    def test_output_float32_can_be_canonicalized_without_changing_inputs(self):
        task,request,children = setup_pair()
        result = m.estimate_context(task.context_refs[0],children[0],"j",
                                    Mock(return_value=point().astype(np.float32)))
        self.assertEqual(result.mu.dtype,np.float64)
        self.close("float32_output_only",result.ate,expected_ate())
        self.assertEqual(task.context_refs[0].outcomes_observed.dtype,np.float64)

    def test_not_requested_node_fields_are_not_fabricated(self):
        task,request,children = setup_pair(outputs=("ate",))
        result = run_pair(task,outputs=("ate",))
        for arm in result.arm_results:
            self.assertEqual(arm.output_status["mu"],"NOT_REQUESTED")
            self.assertEqual(arm.output_status["ite"],("NOT_REQUESTED",)*3)
            self.assertTrue(np.isnan(arm.mu).all())
            self.assertTrue(np.isnan(arm.ite).all())
        self.close("ate_only_request",result.ate,[33.375,61.75,95.125])

    def test_invalid_pair_assembly_or_model_method_rejected(self):
        task,request,children = setup_pair()
        good = run_pair(task)
        for method in ("state-mean","oracle-context","unknown",None):
            with self.assertRaises(c.ContractError):
                m.estimate_context(task.context_refs[0],children[0],method,Mock())
        for arms in ((good.arm_results[1],good.arm_results[0]),(None,good.arm_results[1])):
            with self.assertRaises(c.ContractError):
                m.compose_pair(task,request,"kernel-old",arms)
        with self.assertRaises(c.ContractError):
            m.compose_pair(task,request,"kernel-new",good.arm_results)

    def test_invalid_predictor_configuration_rejected_before_calls(self):
        task,request,children = setup_pair()
        callback = Mock()
        with self.assertRaises(c.ContractError):
            m.estimate_pair(task,request,"j",context_requests=children,predictors=(callback,None))
        callback.assert_not_called()

    def test_input_hash_tamper_detected_before_predictor(self):
        task,request,children = setup_pair()
        data = task.context_refs[0].outcomes_observed.copy();data[0]+=1
        object.__setattr__(task.context_refs[0],"outcomes_observed",data)
        callback = Mock()
        with self.assertRaises(c.ContractError):
            m.estimate_pair(task,request,"j",context_requests=children,predictors=(callback,callback))
        callback.assert_not_called()


def tearDownModule():
    print("S4.2_REFERENCE_RESIDUALS "+json.dumps(ERRORS,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
