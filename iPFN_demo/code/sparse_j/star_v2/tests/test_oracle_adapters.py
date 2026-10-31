"""6.1.1 deterministic Oracle bridge acceptance, fixed ATOL=1e-12 RTOL=0."""
import unittest
from dataclasses import replace
from unittest.mock import Mock
from collections.abc import Mapping

import numpy as np

from star_v2 import oracle_adapters as bridge
from star_v2 import oracle_context as context, oracle_pair as joint
from star_v2.estimation import PairEstimateBundle, IntervalRecord
from star_v2.support import support_for_pair, support_for_task
from test_kernel_new_adapter import tasks_fixture
from test_j_adapter import signature

ATOL = 1e-12
REFERENCE_RESIDUALS = []


def hidden():
    return np.column_stack((np.sin(np.arange(200)/7),np.cos(np.arange(200)/11),
                            np.arange(200)/256)).astype(np.float64)


def context_result(arm):
    n = np.arange(200,dtype=np.float64)
    mu = np.empty((200,2,2))
    mu[:,0,0],mu[:,0,1] = n+2*arm,5*n+90+3*arm
    mu[:,1,0],mu[:,1,1] = 2*n+3+8*arm,4*n+8+18*arm
    ite = np.column_stack((n+3+6*arm,2*n+5+10*arm,3*n+8+16*arm))
    ate = ite.mean(0)
    intervals = tuple(context.OracleInterval(effect,"OK",float(ate[k]-(k+1)),
        float(ate[k]+(k+1)),float(k+1)) for k,effect in enumerate(context.EFFECT_ORDER))
    return context.OracleContextResult(
        np.zeros(25),np.eye(25),mu,ite,ate,intervals,"OK","OK",{},
        {"prior_cholesky":"OK"},1.,{})


def joint_result():
    intervals = tuple(context.OracleInterval(effect,"OK",lower,upper,.25,
        source="joint_model_posterior",statistical_unit="pair")
        for effect,lower,upper in zip(context.EFFECT_ORDER,(6.,-3.,4.),(8.5,-1.,6.5)))
    return joint.OraclePairResult(
        "OK",{},"OK",np.zeros(25),np.eye(25),"OK",None,"OK",
        np.ones((200,2,2)),np.ones((200,3)),np.array([7.25,-2.,5.25]),
        intervals,{"average_noise_variance":1.},{},1.)


class OracleAdapterTests(unittest.TestCase):
    def setUp(self):
        self.task,self.parent,self.children = tasks_fixture()
        self.h = hidden()
        self.before = tuple(signature(t) for t in self.task.context_refs)
        self.hashes = tuple(t.input_hash for t in self.task.context_refs)
        self.core = Mock(side_effect=lambda **kw:context_result(int(kw["treatment"][0])))
        self.joint_core = Mock(return_value=joint_result())

    def tearDown(self):
        self.assertEqual(tuple(signature(t) for t in self.task.context_refs),self.before)
        self.assertEqual(tuple(t.input_hash for t in self.task.context_refs),self.hashes)

    def context_pair(self,**kwargs):
        options = dict(true_h=self.h,noise_sd=1.,context_requests=self.children,core=self.core)
        options.update(kwargs)
        return bridge.estimate_oracle_context_pair(self.task,self.parent,**options)

    def pair(self,**kwargs):
        options = dict(true_h=self.h,noise_sd=1.,core=self.joint_core)
        options.update(kwargs)
        return bridge.estimate_oracle_pair(self.task,self.parent,**options)

    def interval_requests(self):
        from star_v2.model_adapters import context_requests_from_parent
        self.parent = replace(self.parent,requested_outputs=("mu","ite","ate","interval"))
        self.children = context_requests_from_parent(
            self.task,self.parent,request_ids=("bridge-0","bridge-1"))

    def test_not_identified_zero_context_pair_and_joint_numerical_calls(self):
        task,parent,children = tasks_fixture(status="NOT_IDENTIFIED")
        for result in (
            bridge.estimate_oracle_context(task.context_refs[0],children[0],true_h=None,
                                          noise_sd=1.,core=self.core),
            bridge.estimate_oracle_context_pair(task,parent,true_h=None,noise_sd=1.,
                                               context_requests=children,core=self.core),
            bridge.estimate_oracle_pair(task,parent,true_h=None,noise_sd=1.,core=self.joint_core)):
            self.assertEqual(result.status,"NOT_IDENTIFIED")
            self.assertEqual(result.execution_status,"NOT_IDENTIFIED")
            self.assertEqual(result.failure_message,"exact upstream reason")
            self.assertEqual(dict(result.failure_provenance),dict(parent.failure_provenance))
            self.assertTrue(np.isnan(result.ate).all())
        self.core.assert_not_called()
        self.joint_core.assert_not_called()

    def test_context_two_independent_calls_and_scalar_pair_mean(self):
        result = self.context_pair()
        self.assertEqual(self.core.call_count,2)
        self.assertEqual([c.kwargs["treatment"][0] for c in self.core.call_args_list],[0.,1.])
        expected_arms = []
        for arm in (0,1):
            totals = [0.,0.,0.]
            for i in range(200):
                y00,y10,y11 = i+2*arm,2*i+3+8*arm,4*i+8+18*arm
                for k,value in enumerate((y10-y00,y11-y10,y11-y00)):
                    totals[k] += value
            expected_arms.append([v/200 for v in totals])
        expected = np.array([(expected_arms[0][k]+expected_arms[1][k])/2 for k in range(3)])
        np.testing.assert_allclose(result.ate,expected,atol=ATOL,rtol=0)
        REFERENCE_RESIDUALS.append(float(np.max(np.abs(result.ate-expected))))
        self.assertIs(type(result),PairEstimateBundle)
        self.assertEqual(result.status,"POINT")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertFalse(result.diagnostics["arms_independent"])

    def test_context_mu_ite_canonical_axis_independent_scalar_reference(self):
        result = bridge.estimate_oracle_context(self.task.context_refs[0],self.children[0],
            true_h=self.h,noise_sd=1.,core=self.core)
        for i in range(200):
            expected = np.array([[float(i),float(5*i+90)],[float(2*i+3),float(4*i+8)]])
            np.testing.assert_allclose(result.mu[i],expected,atol=ATOL,rtol=0)
            np.testing.assert_allclose(result.ite[i],[i+3,2*i+5,3*i+8],atol=ATOL,rtol=0)
        self.assertEqual(result.arm_order,("00","01","10","11"))

    def test_both_original_context_intervals_preserved_no_pair_interval(self):
        self.interval_requests()
        result = self.context_pair()
        self.assertEqual(result.intervals,())
        self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
        self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
        for arm,bundle in enumerate(result.arm_results):
            self.assertEqual(len(bundle.intervals),3)
            for k,record in enumerate(bundle.intervals):
                expected_ate = (102.5+6*arm,204.+10*arm,306.5+16*arm)[k]
                self.assertEqual(record.lower,expected_ate-(k+1))
                self.assertEqual(record.upper,expected_ate+(k+1))
                self.assertEqual(record.context_id,self.task.context_refs[arm].context_id)
                self.assertEqual(record.source,"model_posterior")
            self.assertEqual(result.context_intervals[arm],bundle.intervals)

    def test_joint_ate_interval_scope_and_no_public_node_capability(self):
        self.interval_requests()
        result = self.pair()
        np.testing.assert_array_equal(result.ate,[7.25,-2.,5.25])
        self.assertEqual(result.arm_results,(None,None))
        self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
        self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)
        self.assertTrue(np.isnan(result.mu).all() and np.isnan(result.ite).all())
        self.assertNotIn("context.mu",result.method_capability.outputs)
        for k,record in enumerate(result.intervals):
            self.assertIs(type(record),IntervalRecord)
            self.assertIsNone(record.context_id)
            self.assertEqual(record.source,"joint_model_posterior")
            self.assertEqual(record.level,.95)
            self.assertEqual(record.lower,(6.,-3.,4.)[k])
            self.assertEqual(record.upper,(8.5,-1.,6.5)[k])
        self.assertEqual(result.sampling_uncertainty.source,"joint_model_posterior")
        self.assertEqual(result.sampling_uncertainty.statistical_unit,"pair")
        self.assertIsNone(result.sampling_uncertainty.effective_sample_count)

    def test_internal_failed_status_does_not_overwrite_identification_or_ate(self):
        raw = joint_result()
        records = tuple(replace(r,status="FAILED",lower=None,upper=None,variance=None,
                                failure_reason="posterior interval unavailable") for r in raw.intervals)
        self.joint_core.return_value = replace(raw,status="FAILED",intervals=records,
                                              failure_reasons={"intervals":"exact interval reason"})
        # Interval not requested: internal FAILED must not invalidate valid ATE.
        result = self.pair()
        self.assertEqual(result.status,"POINT")
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.output_status["interval"],("NOT_REQUESTED",)*3)
        self.assertEqual(result.intervals,())
        self.assertEqual(result.sampling_uncertainty.status,"NOT_REQUESTED")
        self.interval_requests()
        result = self.pair()
        self.assertEqual(result.status,"POINT")
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.output_status["interval"],("FAILED",)*3)

    def test_non_j_all_failed_uncertainty_retains_no_forbidden_metadata(self):
        self.interval_requests()
        raw = joint_result()
        self.joint_core.return_value = replace(raw,status="FAILED",intervals=tuple(
            replace(r,status="FAILED",lower=None,upper=None,variance=None,
                    failure_reason="specific numerical failure") for r in raw.intervals))
        result = self.pair()
        uncertainty = result.sampling_uncertainty
        self.assertEqual(uncertainty.status,"FAILED")
        self.assertIsNone(uncertainty.source)
        self.assertIsNone(uncertainty.statistical_unit)
        self.assertIsNone(uncertainty.effective_sample_count)
        self.assertTrue(all(r.source == "joint_model_posterior" for r in result.intervals))
        self.assertEqual(result.failure_provenance["uncertainty_origin"]["statistical_unit"],"pair")
        self.assertEqual(result.failure_provenance["intervals"]["Direct"],"specific numerical failure")

    def test_partial_interval_failure_keeps_other_intervals_and_all_point_effects(self):
        self.interval_requests()
        raw = joint_result();records = list(raw.intervals)
        records[1] = replace(records[1],status="FAILED",lower=None,upper=None,variance=None,
                             failure_reason="only spillover interval failed")
        self.joint_core.return_value = replace(raw,status="FAILED",intervals=tuple(records))
        result = self.pair()
        self.assertEqual(result.output_status["interval"],("OK","FAILED","OK"))
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.sampling_uncertainty.status,"OK")
        self.assertEqual(result.sampling_uncertainty.statistical_unit,"pair")
        self.assertEqual(result.intervals[0].lower,6.)
        self.assertEqual(result.intervals[2].upper,6.5)

    def test_context_interval_only_failure_does_not_invalidate_pair_ate(self):
        self.interval_requests()
        def fail_intervals(**kw):
            raw = context_result(int(kw["treatment"][0]))
            return replace(raw,intervals=tuple(replace(r,status="FAILED",lower=None,upper=None,
                failure_reason="context posterior interval failure") for r in raw.intervals))
        self.core.side_effect = fail_intervals
        result = self.context_pair()
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        for arm in result.arm_results:
            self.assertEqual(arm.execution_status,"FAILED")
            self.assertEqual(arm.sampling_uncertainty.status,"FAILED")
            self.assertIsNone(arm.sampling_uncertainty.source)
            self.assertEqual(arm.failure_provenance["intervals"]["Total"],"context posterior interval failure")

    def test_single_arm_effect_failure_propagates_only_necessary_pair_effect(self):
        def failure(**kw):
            arm = int(kw["treatment"][0]);raw = context_result(arm)
            if arm:
                ate = raw.ate.copy();ate[1] = np.nan
                raw = replace(raw,ate=ate,failure_reasons={"point":"specific arm1 effect failure"})
            return raw
        self.core.side_effect = failure
        result = self.context_pair()
        self.assertEqual(result.output_status["ate"],("OK","FAILED","OK"))
        self.assertTrue(np.isnan(result.ate[1]))
        self.assertEqual(result.ate[0],105.5)
        self.assertEqual(result.ate[2],314.5)
        self.assertEqual(result.arm_results[1].failure_provenance["oracle_failure_reasons"]["point"],
                         "specific arm1 effect failure")

    def test_joint_single_effect_failure_independent_status(self):
        self.interval_requests()
        raw = joint_result();ate = raw.ate.copy();ate[0] = np.nan
        self.joint_core.return_value = replace(raw,status="FAILED",ate=ate)
        result = self.pair()
        self.assertEqual(result.output_status["ate"],("FAILED","OK","OK"))
        self.assertEqual(result.output_status["interval"],("FAILED","OK","OK"))
        self.assertTrue(np.isnan(result.ate[0]))
        self.assertEqual(result.ate[1],-2.)

    def test_unavailable_posterior_or_inconsistent_constraints_no_fake_ate(self):
        raw = joint_result()
        self.joint_core.return_value = replace(raw,status="FAILED",posterior_status="FAILED",
            posterior_mean=None,posterior_covariance=None,covariance_status="FAILED",
            point_status="FAILED",mu=None,ite=None,ate=None,
            failure_reasons={"posterior":"Inconsistent exact constraints"})
        result = self.pair()
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertTrue(np.isnan(result.ate).all())
        self.assertEqual(result.failure_provenance["oracle_failure_reasons"]["posterior"],
                         "Inconsistent exact constraints")

    def test_core_exception_has_failure_result_and_other_arm_is_retained(self):
        def fail(**kw):
            if kw["treatment"][0] == 0: raise ArithmeticError("exact arm0 core failure")
            return context_result(1)
        self.core.side_effect = fail
        result = self.context_pair()
        self.assertEqual(self.core.call_count,2)
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertEqual(result.arm_results[1].output_status["ate"],("OK",)*3)
        self.assertIn("exact arm0 core failure",result.arm_results[0].failure_message)

    def test_support_all_masks_flags_identity_hashes_and_parent_linkage(self):
        for result in (self.context_pair(),self.pair()):
            self.assertEqual(result.request,self.parent)
            self.assertEqual(result.pair_id,self.task.pair_id)
            self.assertEqual(result.input_hashes,self.hashes)
            self.assertEqual(result.target_id,self.parent.target_id)
            np.testing.assert_array_equal(result.node_ids,self.parent.node_ids)
            bundles = (result,)+tuple(a for a in result.arm_results if a is not None)
            refs = (support_for_pair(self.task),)+tuple(
                support_for_task(t) for t in self.task.context_refs) if len(bundles) == 3 else (support_for_pair(self.task),)
            for bundle,reference in zip(bundles,refs):
                for key in ("population_support","design_support","empirical_support_mask"):
                    a,b = getattr(bundle,key),getattr(reference,key)
                    self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape)
                    np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(bundle.extrapolation_flags["empirical"],reference.empirical_extrapolation)
                expected = (tuple(s.context_design_extrapolation for s in reference.context_supports)
                            if bundle is result else reference.context_design_extrapolation)
                np.testing.assert_array_equal(bundle.extrapolation_flags["context_design"],expected)
            for k,arm in enumerate(result.arm_results):
                if arm is not None:
                    self.assertEqual(arm.request,self.children[k])
                    self.assertEqual(arm.request.parent_request_id,self.parent.request_id)

    def test_strict_allowed_inputs_and_detached_privileged_h(self):
        self.context_pair();self.pair()
        expected = {"x","adjacency","treatment","observed_indices","outcomes_observed",
                    "true_h","noise_sd","context_identity"}
        for call in self.core.call_args_list:
            self.assertEqual(set(call.kwargs),expected)
            self.assertEqual(call.kwargs["outcomes_observed"].dtype,np.dtype("float64"))
            self.assertFalse(np.shares_memory(call.kwargs["true_h"],self.h))
            self.assertFalse(call.kwargs["true_h"].flags.writeable)
            for key in ("x","adjacency","treatment","observed_indices","outcomes_observed"):
                self.assertFalse(call.kwargs[key].flags.writeable)
        self.assertEqual(set(self.joint_core.call_args.kwargs),{
            "true_h","noise_sd","adjacency","treatment0","treatment1","observed_indices0",
            "observed_indices1","outcomes_observed0","outcomes_observed1","pair_identity"})
        for key in ("beta","truth","full_y","noise","private_pair"):
            with self.assertRaises(TypeError):
                bridge.estimate_oracle_pair(self.task,self.parent,true_h=self.h,noise_sd=1.,**{key:object()})

    def test_bad_metadata_rejected_and_no_arrays_or_objects_in_public_metadata(self):
        raw = joint_result()
        self.joint_core.return_value = replace(raw,diagnostics={"beta":np.ones(25)})
        rejected = self.pair()
        self.assertEqual(rejected.output_status["ate"],("FAILED",)*3)
        self.joint_core.return_value = joint_result()
        result = self.pair()
        def check(value):
            self.assertNotIsInstance(value,np.ndarray)
            if isinstance(value,Mapping):
                for key,item in value.items():
                    self.assertIs(type(key),str);check(item)
            elif isinstance(value,(tuple,list)):
                for item in value: check(item)
            else: self.assertIn(type(value),(str,int,float,bool,type(None)))
        check(result.diagnostics);check(result.failure_provenance)

    def test_binding_and_child_link_conflicts_precede_numerical_calls(self):
        wrong = replace(self.parent,input_hashes=("0"*64,"1"*64))
        with self.assertRaises(ValueError):
            bridge.estimate_oracle_pair(self.task,wrong,true_h=self.h,noise_sd=1.,core=self.joint_core)
        children = (replace(self.children[0],parent_request_id="wrong"),self.children[1])
        with self.assertRaises(ValueError): self.context_pair(context_requests=children)
        self.core.assert_not_called();self.joint_core.assert_not_called()

    def test_bound_ids_and_explicit_child_ids_preserved(self):
        task,parent,children = tasks_fixture(bound=True)
        result = bridge.estimate_oracle_context_pair(task,parent,true_h=self.h,noise_sd=1.,
                                                    context_requests=children,core=self.core)
        self.assertEqual(tuple(a.request_id for a in result.arm_results),("bound","bound"))
        other = bridge.estimate_oracle_context_pair(self.task,self.parent,true_h=self.h,noise_sd=1.,
                                                    request_ids=("explicit-0","explicit-1"),core=self.core)
        self.assertEqual(tuple(a.request_id for a in other.arm_results),("explicit-0","explicit-1"))

    def test_actual_numerical_cores_connect_without_simulator_or_model(self):
        # Hand-built public factual records; no mechanism/Pair generator or RNG.
        context_bundle = bridge.estimate_oracle_context(self.task.context_refs[0],self.children[0],
                                                        true_h=self.h,noise_sd=1.)
        joint_bundle = bridge.estimate_oracle_pair(self.task,self.parent,true_h=self.h,noise_sd=1.)
        self.assertEqual(context_bundle.output_status["ate"],("OK",)*3)
        self.assertEqual(joint_bundle.output_status["ate"],("OK",)*3)
        self.assertEqual(context_bundle.status,"POINT")
        self.assertEqual(joint_bundle.status,"POINT")

    def test_repeated_execution_is_deterministic(self):
        first,second = self.context_pair(),self.context_pair()
        np.testing.assert_array_equal(first.ate,second.ate)
        for a,b in zip(first.arm_results,second.arm_results):
            np.testing.assert_array_equal(a.mu,b.mu)
            self.assertEqual(a.intervals,b.intervals)


if __name__ == "__main__":
    unittest.main()
