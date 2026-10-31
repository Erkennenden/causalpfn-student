"""S4.4-C: data-only contract tests; no predictors, models or RNG draws.

Task views reuse the hand-built public fixture, never simulator generation.
Method IDs below test metadata validation only; no Oracle algorithm executes.
"""
from dataclasses import replace
import unittest

import numpy as np

from star_v2 import estimation as e, contracts as c
from test_kernel_new_adapter import tasks_fixture


class SamplingProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = tasks_fixture()[0].context_refs[0]

    def bundle(self, states=("FAILED",)*3, uncertainty=None, *, method="j"):
        request = e.EstimationRequest.from_task(
            self.task,status="POINT",target_id=e.CONFIG.target_id,
            request_id="explicit-provenance-request",requested_outputs=("ate","interval"))
        records = tuple(e.IntervalRecord(
            self.task.context_id,effect,state,
            0. if state=="OK" else None,4. if state=="OK" else None,
            source="model_posterior") for effect,state in zip(e.EFFECT_ORDER,states))
        return e.EstimateBundle.from_outputs(
            request,self.task,method,output_status={"ate":("OK",)*3,"interval":states},
            ate=np.array([1.,2.,3.]),intervals=records,sampling_uncertainty=uncertainty,
            failure_type="INTERVAL_FIXTURE" if "FAILED" in states else None,
            failure_message="original exact interval failure" if "FAILED" in states else None,
            failure_provenance={"intervals":{"reason":"original reason code"}})

    def test_j_all_failed_retains_complete_provenance_and_point_values(self):
        bundle = self.bundle(uncertainty=e.SamplingUncertainty(
            "FAILED","model_posterior","context",None))
        self.assertEqual(bundle.status,"POINT")
        self.assertEqual(bundle.execution_status,"FAILED")
        self.assertEqual(bundle.output_status["interval"],("FAILED",)*3)
        self.assertEqual(bundle.output_status["ate"],("OK",)*3)
        np.testing.assert_array_equal(bundle.ate,[1.,2.,3.])
        self.assertEqual(bundle.sampling_uncertainty.status,"FAILED")
        self.assertEqual(bundle.sampling_uncertainty.source,"model_posterior")
        self.assertEqual(bundle.sampling_uncertainty.statistical_unit,"context")
        self.assertIsNone(bundle.sampling_uncertainty.effective_sample_count)
        self.assertEqual(tuple(i.effect for i in bundle.intervals),("Direct","Spillover","Total"))
        for interval in bundle.intervals:
            self.assertEqual(interval.status,"FAILED")
            self.assertEqual(interval.source,"model_posterior")
            self.assertIsNone(interval.lower);self.assertIsNone(interval.upper)
            self.assertEqual(interval.level,.95)
        self.assertEqual(bundle.failure_message,"original exact interval failure")

    def test_post_bundle_normalization_preserves_identity_and_failure_evidence(self):
        original = self.bundle()
        result = replace(original,sampling_uncertainty=e.SamplingUncertainty(
            "FAILED","model_posterior","context",None))
        for name in ("status","execution_status","request_id","target_id","pair_id","context_ids",
                     "input_hashes","protocol_hash","failure_type","failure_message"):
            self.assertEqual(getattr(result,name),getattr(original,name))
        self.assertIs(result.request,original.request)
        self.assertEqual(dict(result.failure_provenance),dict(original.failure_provenance))
        self.assertEqual(dict(result.output_status),dict(original.output_status))
        self.assertEqual(result.intervals,original.intervals)
        np.testing.assert_array_equal(result.ate,original.ate)
        np.testing.assert_array_equal(result.node_ids,original.node_ids)
        self.assertEqual(result.input_hash,self.task.input_hash)
        c.validate_task(self.task)

    def test_j_partial_intervals_keep_ok_uncertainty_and_each_effect_state(self):
        for states in (("OK","FAILED","OK"),("FAILED","OK","FAILED"),("FAILED","FAILED","OK")):
            bundle = self.bundle(states,e.SamplingUncertainty(
                "OK","model_posterior","context",None))
            self.assertEqual(bundle.sampling_uncertainty.status,"OK")
            self.assertEqual(bundle.output_status["interval"],states)
            self.assertEqual(tuple(record.status for record in bundle.intervals),states)
            self.assertEqual(bundle.execution_status,"FAILED")
            self.assertEqual(bundle.output_status["ate"],("OK",)*3)
            self.assertEqual(bundle.sampling_uncertainty.source,"model_posterior")
            self.assertEqual(bundle.sampling_uncertainty.statistical_unit,"context")
            self.assertIsNone(bundle.sampling_uncertainty.effective_sample_count)

    def test_j_all_successful_analytic_uncertainty_has_no_sample_count(self):
        bundle = self.bundle(("OK",)*3,e.SamplingUncertainty(
            "OK","model_posterior","context",None))
        self.assertEqual(bundle.execution_status,"OK")
        self.assertEqual(bundle.output_status["interval"],("OK",)*3)
        self.assertEqual(bundle.sampling_uncertainty.status,"OK")
        self.assertIsNone(bundle.sampling_uncertainty.effective_sample_count)
        for record in bundle.intervals:
            self.assertEqual(record.source,"model_posterior")
            self.assertEqual((record.lower,record.upper,record.level),(0.,4.,.95))

    def test_source_without_statistical_unit_rejected(self):
        for status in ("FAILED","OK"):
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty(status,"model_posterior",None,None)

    def test_statistical_unit_without_source_rejected(self):
        for status in ("FAILED","OK"):
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty(status,None,"context",None)

    def test_wrong_j_source_or_unit_rejected_at_bundle_layer(self):
        for status,states in (("FAILED",("FAILED",)*3),("OK",("OK",)*3)):
            for source,unit in (("wrong_source","context"),("model_posterior","mechanism"),
                                ("joint_model_posterior","pair")):
                # Structurally valid object does not infer method identity.
                value = e.SamplingUncertainty(status,source,unit,None)
                with self.assertRaises(c.ContractError):
                    self.bundle(states,value)

    def test_j_non_none_effective_sample_count_rejected_including_success(self):
        for count in (1,100):
            value = e.SamplingUncertainty("OK","model_posterior","context",count)
            with self.assertRaises(c.ContractError):
                self.bundle(("OK",)*3,value)
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty("FAILED","model_posterior","context",count)
        with self.assertRaises(c.ContractError):
            e.SamplingUncertainty("FAILED",None,None,100)

    def test_all_failed_cannot_be_relabelled_ok_or_other_status(self):
        for status in ("OK","NOT_ESTIMABLE","NOT_REQUESTED","UNSUPPORTED"):
            value = (e.SamplingUncertainty("OK","model_posterior","context",None)
                     if status=="OK" else e.SamplingUncertainty(status))
            with self.assertRaises(c.ContractError):
                self.bundle(("FAILED",)*3,value)

    def test_partial_success_cannot_be_relabelled_failed(self):
        for value in (e.SamplingUncertainty("FAILED"),
                      e.SamplingUncertainty("FAILED","model_posterior","context",None)):
            with self.assertRaises(c.ContractError):
                self.bundle(("OK","FAILED","OK"),value)

    def test_non_j_failed_source_metadata_cannot_bypass_old_rule(self):
        for method in ("kernel-old","kernel-new","oracle-context"):
            value = e.SamplingUncertainty("FAILED","model_posterior","context",None)
            with self.assertRaises(c.ContractError):
                self.bundle(("FAILED",)*3,value,method=method)

    def test_non_j_valid_posterior_sample_count_and_source_rules_unchanged(self):
        value = e.SamplingUncertainty("OK","model_posterior","context",100)
        bundle = self.bundle(("OK",)*3,value,method="oracle-context")
        self.assertEqual(bundle.execution_status,"OK")
        self.assertEqual(bundle.sampling_uncertainty.effective_sample_count,100)
        for source,unit in (("wrong","context"),("model_posterior","pair"),
                            ("mechanism_bootstrap","mechanism")):
            with self.assertRaises(c.ContractError):
                self.bundle(("OK",)*3,e.SamplingUncertainty("OK",source,unit,100),
                            method="oracle-context")

    def test_existing_unavailable_uncertainty_without_provenance_still_valid(self):
        bundle = self.bundle()
        self.assertEqual(bundle.execution_status,"FAILED")
        self.assertEqual(bundle.sampling_uncertainty,e.SamplingUncertainty("FAILED"))
        legacy = self.bundle(("FAILED",)*3,e.SamplingUncertainty("FAILED"),method="oracle-context")
        self.assertEqual(legacy.execution_status,"FAILED")
        self.assertIsNone(legacy.sampling_uncertainty.source)

    def test_unsupported_not_requested_not_estimable_metadata_rules_unchanged(self):
        for status in ("UNSUPPORTED","NOT_REQUESTED","NOT_ESTIMABLE"):
            self.assertEqual(e.SamplingUncertainty(status).status,status)
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty(status,"model_posterior","context",None)
        request = e.EstimationRequest.from_task(
            self.task,status="POINT",target_id=e.CONFIG.target_id,
            request_id="point-only",requested_outputs=("ate",))
        result = e.EstimateBundle.from_outputs(
            request,self.task,"kernel-old",output_status={"ate":("OK",)*3},
            ate=np.array([1.,2.,3.]))
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.sampling_uncertainty.status,"NOT_REQUESTED")

    def test_object_requires_nonempty_paired_provenance_without_method_inference(self):
        for source,unit in (("","context"),("model_posterior",""),(1,"context")):
            with self.assertRaises(c.ContractError):
                e.SamplingUncertainty("FAILED",source,unit,None)
        # A generic object can carry structurally valid failed provenance;
        # only Bundle decides whether its method is eligible.
        value = e.SamplingUncertainty("FAILED","other_source","other_unit",None)
        self.assertEqual((value.source,value.statistical_unit),("other_source","other_unit"))
        with self.assertRaises(c.ContractError):
            self.bundle(uncertainty=value)


if __name__=="__main__":
    unittest.main()
