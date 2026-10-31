"""S3.4 integration; reuses existing independent references and fake pairs.

Absolute comparison tolerance fixed before execution: 1e-9 (ATE/matrix),
1e-12 (Fixture A). Different estimators are never assumed equal.
No formal cohort, persistence, prediction, Bootstrap or numerical core here.
"""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import baselines as b, contracts as c
from star_v2 import baselines_state_ht as state, baselines_regression as regression
from star_v2.estimation import PairEstimateBundle
from star_v2.support import support_for_pair
from test_state_mean_hajek import req, hand_records
from test_f_z2_lin_z2 import task_fixture, reference_design, reference_fit
from test_pair_ht import fixture_layer

ATOL = 1e-9
FIXTURE_ATOL = 1e-12
EVIDENCE = {}


class Stage3AcceptanceTests(unittest.TestCase):
    def close(self,label,actual,expected):
        actual,expected = np.asarray(actual),np.asarray(expected)
        finite = np.isfinite(actual)&np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        EVIDENCE[label] = max(EVIDENCE.get(label,0.),error)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=ATOL,equal_nan=True)

    def test_all_four_canonical_bundles_identities_and_order(self):
        task = task_fixture()
        request = req(task)
        self.assertEqual(b.METHOD_IDS,("state-mean","ht","f-z2","lin-z2"))
        for method in b.METHOD_IDS:
            with self.subTest(method=method):
                result = b.estimate_pair(task,request,method)
                self.assertIs(type(result),PairEstimateBundle)
                self.assertEqual(result.method_id,method)
                for name in ("status","target_id","request_id","pair_id","context_ids",
                             "input_hashes","protocol_hash","protocol_version"):
                    self.assertEqual(getattr(result,name),getattr(request,name))
                self.assertEqual(result.status,"POINT")
                self.assertEqual(result.execution_status,"OK")
                self.assertEqual(result.output_status["ate"],("OK",)*3)
                self.assertEqual(result.arm_order,("00","01","10","11"))
                self.assertEqual(result.effect_order,("Direct","Spillover","Total"))
                np.testing.assert_array_equal(result.node_ids,np.arange(200))
                self.assertEqual(result.arm_results,(None,None))
                self.assertEqual(result.support.n_unique_observed_nodes,50)
                self.assertEqual(result.support.n_factual_records,100)
                self.close(method+"_total_identity",result.ate[2],result.ate[0]+result.ate[1])

    def test_dispatch_matches_each_original_function_and_preserves_diagnostics(self):
        task = task_fixture()
        request = req(task)
        original = {"state-mean":state.estimate_state_mean,"ht":state.estimate_pair_ht,
                    "f-z2":regression.estimate_f_z2,"lin-z2":regression.estimate_lin_z2}
        for method,entry in original.items():
            result,direct = b.estimate_pair(task,request,method),entry(request,task)
            self.close(method+"_direct_dispatch",result.ate,direct.ate)
            self.assertEqual(dict(result.output_status),dict(direct.output_status))
            self.assertEqual(dict(result.diagnostics),dict(direct.diagnostics))
            self.assertEqual(dict(result.estimability),dict(direct.estimability))
            self.assertEqual(dict(result.failure_provenance),dict(direct.failure_provenance))
            self.assertEqual(dict(result.asset_hashes),dict(direct.asset_hashes))

    def test_independent_state_mean_and_ht_hand_references(self):
        task = task_fixture()
        groups,counts,means = hand_records(task)
        sm = b.estimate_pair(task,req(task),"state-mean")
        ht = b.estimate_pair(task,req(task),"ht")
        expected_ht = np.array([
            sum(groups[label])/((50/200)*.5)/200 for label in state.ARM_ORDER])
        self.close("sm_independent_reference",sm.ate,
                   [means[2]-means[0],means[3]-means[2],means[3]-means[0]])
        self.close("ht_independent_reference",ht.ate,
                   [expected_ht[2]-expected_ht[0],expected_ht[3]-expected_ht[2],
                    expected_ht[3]-expected_ht[0]])
        self.assertEqual(sm.diagnostics["states"]["00"]["count"],counts[0])
        self.assertEqual(ht.diagnostics["design"]["weight"],8.)
        # No cross-estimator equality assertion: SM and HT have different denominators.

    def test_independent_regression_references_and_full_graph_queries(self):
        task = task_fixture()
        for method in ("f-z2","lin-z2"):
            result = b.estimate_pair(task,req(task),method)
            matrix,y,contrasts,_ = reference_design(task,method)
            expected,statuses,rank,singular,_ = reference_fit(matrix,y,contrasts)
            self.close(method+"_independent_reference",result.ate,expected)
            self.assertEqual(result.output_status["ate"],statuses)
            self.assertEqual(result.diagnostics["design_rank"],rank)
            self.assertEqual(result.diagnostics["query_units"],200)
            self.assertTrue(result.diagnostics["full_population_centering"])
            self.assertTrue(result.diagnostics["full_population_rms_scaling"])
            self.close(method+"_reference_singular",result.diagnostics["singular_values"],singular)
            self.assertEqual(result.asset_hashes["feature_transform.npz"],
                             regression.TRANSFORM_SHA256)
            self.assertEqual(result.diagnostics["rcond"],1e-6)
            self.assertEqual(result.diagnostics["row_space_relative_tolerance"],10*1e-6)
            for k,effect in enumerate(result.effect_order):
                evidence = result.estimability[effect]
                self.assertEqual(evidence["query_scope"],"all_200_nodes")
                self.assertLessEqual(evidence["row_space_residual"],
                                     evidence["row_space_tolerance"])

    def test_missing_01_keeps_all_effects_in_all_methods(self):
        task = task_fixture("01")
        for method in b.METHOD_IDS:
            result = b.estimate_pair(task,req(task),method)
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            self.assertEqual(result.execution_status,"OK")
            self.assertTrue(np.isfinite(result.ate).all())
            if method in ("f-z2","lin-z2"):
                self.assertEqual(result.diagnostics["state_counts"][1],0)
                self.assertLess(result.diagnostics["design_rank"],result.diagnostics["design_columns"])
            else:
                self.assertEqual(result.diagnostics["states"]["01"]["count"],0)
        EVIDENCE["missing01_all_methods"] = "three_effects_OK"

    def test_other_missing_states_fail_only_dependent_effects(self):
        for missing,statuses in (
                ("00",("NOT_ESTIMABLE","OK","NOT_ESTIMABLE")),
                ("10",("NOT_ESTIMABLE","NOT_ESTIMABLE","OK")),
                ("11",("OK","NOT_ESTIMABLE","NOT_ESTIMABLE"))):
            task = task_fixture(missing)
            for method in b.METHOD_IDS:
                result = b.estimate_pair(task,req(task),method)
                expected = ("OK",)*3 if method=="ht" else statuses
                self.assertEqual(result.output_status["ate"],expected)
                if method=="ht":
                    self.assertEqual(result.execution_status,"OK")
                    self.assertFalse(result.diagnostics["states"][missing]["empirical_supported"])
                    continue
                self.assertEqual(result.execution_status,"NOT_ESTIMABLE")
                for k,status in enumerate(statuses):
                    if status=="OK":
                        self.assertTrue(np.isfinite(result.ate[k]))
                    else:
                        self.assertTrue(np.isnan(result.ate[k]))
                        effect = result.effect_order[k]
                        self.assertIn(effect,result.failure_provenance["effects"])
                        self.assertEqual(result.estimability[effect]["status"],"NOT_ESTIMABLE")
                self.assertIsNotNone(result.failure_type)

    def test_identification_blocks_all_numeric_paths(self):
        task = task_fixture()
        request = req(task,status="NOT_IDENTIFIED",failure_type="UPSTREAM",
                      failure_message="verbatim identification reason",
                      failure_provenance={"code":"NI_TEST"})
        with patch.object(state,"_compute",side_effect=AssertionError("no numeric core")) as sm,\
             patch.object(regression,"_compute",side_effect=AssertionError("no numeric core")) as reg:
            results = {method:b.estimate_pair(task,request,method) for method in b.METHOD_IDS}
        sm.assert_not_called();reg.assert_not_called()
        self.assertEqual(len(results),4)
        for result in results.values():
            self.assertEqual((result.status,result.execution_status),("NOT_IDENTIFIED",)*2)
            self.assertEqual(result.failure_message,request.failure_message)
            self.assertEqual(dict(result.failure_provenance),dict(request.failure_provenance))
            self.assertTrue(np.isnan(result.ate).all())
            self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
            self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)

    def test_single_method_engineering_failure_preserves_all_requests_and_other_results(self):
        task = task_fixture()
        request = req(task)
        control = {method:b.estimate_pair(task,request,method) for method in b.METHOD_IDS}
        original = regression._compute
        def injected(inputs,method):
            if method=="f-z2":
                raise ArithmeticError("isolated F-z2 fixture failure")
            return original(inputs,method)
        with patch.object(regression,"_compute",side_effect=injected):
            results = {method:b.estimate_pair(task,request,method) for method in b.METHOD_IDS}
        self.assertEqual(tuple(results),b.METHOD_IDS)
        self.assertEqual(len({(v.request_id,k) for k,v in results.items()}),4)
        failure = results["f-z2"]
        self.assertEqual(failure.status,"POINT")
        self.assertEqual(failure.execution_status,"FAILED")
        self.assertEqual(failure.output_status["ate"],("FAILED",)*3)
        self.assertEqual(failure.failure_type,"ArithmeticError")
        self.assertEqual(failure.failure_message,"isolated F-z2 fixture failure")
        self.assertEqual(failure.failure_provenance["attempts"],1)
        self.assertTrue(np.isnan(failure.ate).all())
        for method in ("state-mean","ht","lin-z2"):
            self.assertEqual(results[method].execution_status,"OK")
            self.close(method+"_failure_isolation",results[method].ate,control[method].ate)
        EVIDENCE["isolated_failure_method"] = "f-z2"

    def test_unknown_method_explicitly_rejected(self):
        task = task_fixture()
        for method in ("unknown","j","oracle-pair","State Mean",None,1,[]):
            with self.subTest(method=method),self.assertRaises(c.ContractError) as error:
                b.estimate_pair(task,req(task),method)
            self.assertEqual(error.exception.code,"METHOD")
            self.assertEqual(error.exception.evidence["allowed"],b.METHOD_IDS)

    def test_five_arrays_only_for_every_numeric_method(self):
        task = task_fixture()
        state_parser = state._public_records
        reg_builder = regression.build_regression_design
        def check_inputs(inputs):
            self.assertEqual(len(inputs),2)
            for payload in inputs:
                self.assertEqual(tuple(payload),c.FACTUAL_KEYS)
                c.validate_predictor_inputs(payload)
                for value in payload.values():
                    for context in task.context_refs:
                        for name in c.FACTUAL_KEYS:
                            self.assertFalse(np.shares_memory(value,getattr(context,name)))
        def checked_state(inputs,design):
            check_inputs(inputs)
            return state_parser(inputs,design)
        def checked_reg(inputs,method):
            check_inputs(inputs)
            return reg_builder(inputs,method)
        with patch.object(state,"_public_records",side_effect=checked_state) as sm,\
             patch.object(regression,"build_regression_design",side_effect=checked_reg) as reg:
            for method in b.METHOD_IDS:
                self.assertEqual(b.estimate_pair(task,req(task),method).execution_status,"OK")
        self.assertEqual(sm.call_count,2)
        self.assertEqual(reg.call_count,2)
        for key in ("truth","beta","h","full_y","noise"):
            with self.assertRaises(TypeError):
                b.estimate_pair(task,req(task),"ht",**{key:np.zeros(200)})

    def test_raw_arrays_hashes_and_support_unchanged(self):
        task = task_fixture()
        original_support = support_for_pair(task)
        arrays = [getattr(context,key) for context in task.context_refs for key in c.FACTUAL_KEYS]
        arrays += [getattr(original_support,key) for key in (
            "population_support","design_support","empirical_support_mask")]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        hashes = tuple(context.input_hash for context in task.context_refs)
        for method in b.METHOD_IDS:
            result = b.estimate_pair(task,req(task),method)
            self.assertEqual(result.input_hashes,hashes)
            self.assertEqual(result.context_ids,task.context_ids)
            for key in ("population_support","design_support","empirical_support_mask"):
                np.testing.assert_array_equal(getattr(result,key),getattr(original_support,key))
            self.assertFalse(result.ate.flags.writeable)
            for context in task.context_refs:
                self.assertEqual(context.outcomes_observed.dtype,np.float64)
                c.validate_task(context)
        self.assertEqual(before,[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])

    def test_ate_only_capability_and_support_remain_separate(self):
        task = task_fixture()
        request = req(task,requested_outputs=("mu","ite","ate","interval"))
        for method in b.METHOD_IDS:
            result = b.estimate_pair(task,request,method)
            self.assertEqual(result.execution_status,"OK")
            self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
            self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)
            self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
            self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
            self.assertTrue(np.isnan(result.mu).all())
            self.assertTrue(np.isnan(result.ite).all())
            self.assertTrue(result.population_support.all())
            self.assertTrue(result.design_support.all())
            self.assertFalse(result.empirical_support_mask.all())
            self.assertEqual(result.method_capability.outputs,frozenset(("pair.ate",)))

    def test_existing_ht_production_and_fixture_references_reused(self):
        self.assertEqual(state.StarPairDesign(200,50).weight,8.)
        self.assertEqual(state.StarPairDesign(4,2).weight,4.)
        for kind in ("A","B"):
            # Reuse S3.2's existing reference harness, no new exhaustive core.
            info = fixture_layer(kind)
            self.assertEqual(info["combinations"],48)
            self.assertLessEqual(info["state_residual"],FIXTURE_ATOL)
            self.assertLessEqual(info["effect_residual"],FIXTURE_ATOL)
            EVIDENCE["reused_ht_layer_"+kind] = info

    def test_identity_conflict_rejected_before_any_computation(self):
        task = task_fixture()
        bad_request = replace(req(task),input_hashes=("0"*64,)*2)
        with patch.object(state,"_compute") as sm,patch.object(regression,"_compute") as reg:
            for method in b.METHOD_IDS:
                with self.assertRaises(c.ContractError):
                    b.estimate_pair(task,bad_request,method)
        sm.assert_not_called();reg.assert_not_called()


def tearDownModule():
    print("STAGE3_ACCEPTANCE_METRICS "+json.dumps(EVIDENCE,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
