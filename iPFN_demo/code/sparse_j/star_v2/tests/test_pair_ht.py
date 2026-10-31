"""Pair HT: independent exhaustive references and test-only Monte Carlo.

Predeclared BEFORE execution: 8192 independent PAIRS, seed below, absolute
fixture tolerance 1e-12; Monte Carlo tolerance 6 empirical pair-level SE
(+1e-12 rounding allowance). Probability checks use 6 binomial SE.
No tuning, production cohort RNG, independent-arm SE or Bootstrap.
"""
from dataclasses import replace
import itertools
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import baselines_state_ht as b, contracts as c, dgp, rng as registry_rng
from test_state_mean_hajek import req, missing_fixture, hand_records
from test_stage2_acceptance import (
    fixture_a_design, conditional_truth_by_enumeration, scalar_structural,
    contrasts, local_ht_layer)
from test_dgp_truth import fixed_arrays, reference_truth

ATOL = 1e-12
MC_PAIRS = 8192
MC_SIGMA = 6.0
MC_SEED = (0x533332, 0x4854, 20261030)
EVIDENCE = {}


def fixture_layer(kind):
    a, assignments, subsets, q, r, joint, weights = fixture_a_design()
    table = np.array([[[1.25,2.5],[4.,7.75]], [[-1.5,.5],[3.25,6.]],
                      [[2.,1.],[5.,8.]], [[.75,2.25],[6.25,10.]]], dtype=np.float64)
    h = np.array([[.1,-.3,.5],[.7,-.2,.4],[-.4,.6,.8],[.2,.1,-.5]])
    beta = np.linspace(-.5,.6,25)
    target = table if kind == "A" else conditional_truth_by_enumeration(a,h,beta)
    estimates, effects = [], []
    empty_cells, structural_residual = 0, 0.
    all_reason_counts = np.zeros(4, dtype=np.int64)
    for treatments, states, inclusion in assignments:
        if kind == "A":
            y = tuple(np.array([table[node,t,s] for node,(t,s) in enumerate(arm)])
                      for arm in states)
        else:
            y = tuple(dgp.structural_mean(a,h,beta,treatment) for treatment in treatments)
            for treatment, outcome in zip(treatments, y):
                structural_residual = max(structural_residual,
                    float(np.max(np.abs(outcome-scalar_structural(a,h,beta,treatment)))))
        for obs in subsets:
            indices = np.asarray(obs, dtype=np.int64)
            codes = np.array([[2*arm[node][0]+arm[node][1] for node in obs]
                              for arm in states], dtype=np.int64)
            observed = np.stack([arm[indices] for arm in y])
            # Actual production numeric core, with design-derived weight 4.
            result = b.pair_ht_core(codes, observed, indices, b.StarPairDesign(4,2))
            if not np.array_equal(result.empirical_state_support, result.counts > 0):
                raise AssertionError("Empirical support/count mismatch")
            empty = result.counts == 0
            if not np.all(result.state_values[empty] == 0.):
                raise AssertionError("Empty HT sum changed mathematical semantics")
            for k in np.flatnonzero(empty):
                if result.state_reason[k] != "empty_sum_zero_contribution_not_zero_response":
                    raise AssertionError("Missing zero contribution diagnostic")
            empty_cells += int(empty.sum())
            inputs = tuple({"x": np.zeros((4,20), dtype=np.float32), "adjacency": a,
                            "treatment": treatments[arm], "observed_indices": indices,
                            "outcomes_observed": observed[arm]} for arm in (0,1))
            derived_codes, _, _, diagnostic = b._public_records(inputs,b.StarPairDesign(4,2))
            np.testing.assert_array_equal(derived_codes, codes)
            for counts in diagnostic["support_reason_counts"]:
                np.testing.assert_array_equal(counts, [8,4,2,2])
                all_reason_counts += counts
            estimates.append(result.state_values.reshape(2,2))
            effects.append(result.ate)
    independent_info, independent_estimates, _, _ = local_ht_layer(kind)
    # Secondary reference: existing Stage-2 local HT implementation, unchanged.
    np.testing.assert_allclose(estimates,independent_estimates,rtol=0,atol=ATOL)
    mean = np.mean(estimates,axis=0)
    effect_mean = np.mean(effects,axis=0)
    expected = target.mean(0)
    return {
        "combinations": len(estimates),
        "state_residual": float(np.max(np.abs(mean-expected))),
        "effect_residual": float(np.max(np.abs(effect_mean-contrasts(expected)))),
        "total_identity_residual": max(abs(v[2]-v[0]-v[1]) for v in effects),
        "structural_reference_residual": structural_residual,
        "empty_state_cells": empty_cells,
        "support_reason_counts": all_reason_counts.tolist(),
        "state_estimate_mean": mean.tolist(), "independent_target": expected.tolist(),
        "stage2_reference_state_residual": independent_info["state_mean_max_abs_residual"]}


def monte_carlo():
    random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(MC_SEED)))
    n, m = 200, 50
    a, h, beta = fixed_arrays(n)
    truth = reference_truth(a,h,beta).mean(0).reshape(4)
    # Vectorized independent scalar-Star structural equation. No truth/DGP
    # production function generates these facts; no private data enter core.
    leaves = random.binomial(1,.5,size=(MC_PAIRS,n-1))
    noise = random.normal(0.,1.,size=(MC_PAIRS,n))
    indices = np.stack([random.choice(n,size=m,replace=False) for _ in range(MC_PAIRS)])
    hetero = np.column_stack([np.ones(n),h])
    neighbor_h = np.tile(h[0],(n,1));neighbor_h[0] = h[1:].mean(0)
    base = hetero@beta[:4]+neighbor_h@beta[4:7]
    tau, gamma, eta = (hetero@beta[start:start+4] for start in (7,11,15))
    center_fraction = leaves.mean(1)
    center_j = leaves@h[1:]/(n-1)
    codes, observed = [], []
    full_codes = []
    for arm in (0,1):
        own = np.column_stack([np.full(MC_PAIRS,arm),leaves])
        exposure = np.full((MC_PAIRS,n),float(arm))
        exposure[:,0] = center_fraction
        j_base = np.full((MC_PAIRS,n),float(arm)*(h[0]@beta[19:22]))
        j_interaction = np.full((MC_PAIRS,n),float(arm)*(h[0]@beta[22:25]))
        j_base[:,0] = center_j@beta[19:22]
        j_interaction[:,0] = center_j@beta[22:25]
        y = base+own*tau+exposure*(gamma+own*eta)+j_base+own*j_interaction+noise
        code = (2*own+arm).astype(np.int64)
        code[:,0] = 2*arm+(center_fraction>.5).astype(np.int64)
        full_codes.append(code)
        codes.append(np.take_along_axis(code,indices,axis=1))
        observed.append(np.take_along_axis(y,indices,axis=1))
    codes = np.stack(codes,axis=1)
    observed = np.stack(observed,axis=1)
    design = b.StarPairDesign(n,m)
    states, effects = [], []
    joint_counts = np.zeros((n,4),dtype=np.int64)
    q_counts = np.zeros((n,4),dtype=np.int64)
    r_counts = np.bincount(indices.ravel(),minlength=n)
    for arm in (0,1):
        for state in range(4):
            q_counts[:,state] += (full_codes[arm]==state).sum(0)
        np.add.at(joint_counts, (indices.ravel(),codes[:,arm].ravel()),1)
    for k in range(MC_PAIRS):
        result = b.pair_ht_core(codes[k],observed[k],indices[k],design)
        states.append(result.state_values)
        effects.append(result.ate)
    states, effects = np.asarray(states), np.asarray(effects)
    target_effects = np.array([truth[2]-truth[0],truth[3]-truth[2],truth[3]-truth[0]])
    state_error = states.mean(0)-truth
    effect_error = effects.mean(0)-target_effects
    # SE units are independent pairs, including both-arm covariance.
    state_se = states.std(0,ddof=1)/math.sqrt(MC_PAIRS)
    effect_se = effects.std(0,ddof=1)/math.sqrt(MC_PAIRS)
    return {
        "pairs": MC_PAIRS, "seed": MC_SEED, "sigma_limit": MC_SIGMA,
        "state_errors": state_error.tolist(), "effect_errors": effect_error.tolist(),
        "state_se": state_se.tolist(), "effect_se": effect_se.tolist(),
        "max_state_standardized_error": float(np.max(np.abs(state_error)/state_se)),
        "max_effect_standardized_error": float(np.max(np.abs(effect_error)/effect_se)),
        "q_max_error": float(np.max(np.abs(q_counts/MC_PAIRS-.5))),
        "r_max_error": float(np.max(np.abs(r_counts/MC_PAIRS-.25))),
        "joint_max_error": float(np.max(np.abs(joint_counts/MC_PAIRS-.125))),
        "state_pass": bool(np.all(np.abs(state_error)<=MC_SIGMA*state_se+ATOL)),
        "effect_pass": bool(np.all(np.abs(effect_error)<=MC_SIGMA*effect_se+ATOL))}


class PairHTTests(unittest.TestCase):
    def test_design_probabilities_and_weights_not_hardcoded(self):
        for n,m,r,weight in ((200,50,.25,8.),(4,2,.5,4.),(6,2,1/3,6.),(4,4,1.,2.)):
            design = b.StarPairDesign(n,m)
            self.assertEqual(design.r,r)
            self.assertEqual(design.q,.5)
            self.assertEqual(design.joint_inclusion_probability,r*.5)
            self.assertEqual(design.weight,weight)

    def test_inclusion_probabilities_by_independent_enumeration(self):
        a,assignments,subsets,q,r,joint,weight = fixture_a_design()
        self.assertEqual(len(assignments)*len(subsets),48)
        np.testing.assert_array_equal(q,.5*np.ones((4,2,2)))
        np.testing.assert_array_equal(r,.5*np.ones(4))
        np.testing.assert_allclose(joint,.25,rtol=0,atol=ATOL)
        np.testing.assert_array_equal(weight,4*np.ones((4,2,2)))
        for _,_,inclusion in assignments:
            self.assertTrue(np.all(inclusion<=1))
            np.testing.assert_array_equal(inclusion.sum((1,2)),2)
        EVIDENCE["design"]={"fixture_weight":4., "formal_weight":8.,
                            "q":.5,"fixture_joint":.25,"formal_joint":.125}

    def test_fixture_layer_a_48_fixed_table_independent_truth(self):
        info = fixture_layer("A")
        EVIDENCE["layer_A"] = info
        self.assertEqual(info["combinations"],48)
        for key in ("state_residual","effect_residual","total_identity_residual"):
            self.assertLessEqual(info[key],ATOL,key)
        self.assertGreater(info["empty_state_cells"],0)

    def test_fixture_layer_b_48_continuous_dgp_independent_truth(self):
        info = fixture_layer("B")
        EVIDENCE["layer_B"] = info
        self.assertEqual(info["combinations"],48)
        for key in ("state_residual","effect_residual","total_identity_residual",
                    "structural_reference_residual"):
            self.assertLessEqual(info[key],ATOL,key)

    def test_formal_design_monte_carlo_predeclared_tolerance(self):
        with patch.object(registry_rng,"make_generator",
                          side_effect=AssertionError("Formal RNG forbidden")):
            info = monte_carlo()
        EVIDENCE["monte_carlo"] = info
        self.assertTrue(info["state_pass"],json.dumps(info))
        self.assertTrue(info["effect_pass"],json.dumps(info))
        for key,p in (("q_max_error",.5),("r_max_error",.25),("joint_max_error",.125)):
            self.assertLessEqual(info[key],MC_SIGMA*math.sqrt(p*(1-p)/MC_PAIRS),key)

    def test_formal_hand_sum_and_one_pair_factor(self):
        task = missing_fixture()
        result = b.estimate_pair_ht(req(task),task)
        groups,counts,_ = hand_records(task)
        expected = np.array([math.fsum(groups[state])/(.25*.5)/200 for state in b.ARM_ORDER])
        for k,state in enumerate(b.ARM_ORDER):
            self.assertEqual(result.diagnostics["states"][state]["count"],counts[k])
            self.assertAlmostEqual(result.diagnostics["states"][state]["value"],expected[k],delta=ATOL)
        np.testing.assert_allclose(result.ate,
            [expected[2]-expected[0],expected[3]-expected[2],expected[3]-expected[0]],
            rtol=0,atol=ATOL)
        self.assertEqual(result.diagnostics["design"]["weight"],8.)

    def test_empty_state_zero_sum_is_not_zero_response(self):
        for missing in b.ARM_ORDER:
            task = missing_fixture(missing)
            result = b.estimate_pair_ht(req(task),task)
            state = result.diagnostics["states"][missing]
            self.assertEqual(state["count"],0)
            self.assertEqual(state["value"],0.)
            self.assertEqual(state["status"],"OK")
            self.assertFalse(state["empirical_supported"])
            self.assertEqual(state["reason"],"empty_sum_zero_contribution_not_zero_response")
            self.assertIsNone(state["hajek_diagnostic"])
            self.assertEqual(result.execution_status,"OK")
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            self.assertTrue(np.isfinite(result.ate).all())
            self.assertFalse(result.empirical_support_mask[:,:,int(missing[1])][
                :,int(missing[0])].any())

    def test_state_codes_and_no_extra_arm_factor_small_core(self):
        codes = np.array([[0,2],[2,3]],dtype=np.int64)
        y = np.array([[1.,4.],[5.,9.]])
        result = b.pair_ht_core(codes,y,np.array([0,1]),b.StarPairDesign(4,2))
        np.testing.assert_array_equal(result.state_values,[1.,0.,9.,9.])
        np.testing.assert_array_equal(result.ate,[8.,0.,8.])
        self.assertEqual(result.counts.sum(),4)
        self.assertEqual(result.state_status,("OK",)*4)

    def test_invalid_design_and_pair_records_rejected(self):
        for args in ((5,2),(200,0),(200,201),(True,1),(4,2,.4),(4,2,.5,.4)):
            with self.subTest(args=args),self.assertRaises(c.ContractError):
                b.StarPairDesign(*args)
        codes = np.array([[0,2],[2,3]],dtype=np.int64)
        y = np.array([[1.,4.],[5.,9.]])
        indices = np.array([0,1])
        for bad in (np.array([[0,2],[0,3]]),np.array([[0,0],[2,3]]),
                    np.array([[4,2],[2,3]]),codes.astype(np.float64),codes[:,:1]):
            with self.assertRaises(c.ContractError):
                b.pair_ht_core(bad,y,indices,b.StarPairDesign(4,2))
        for bad in (y.astype(np.float32),np.array([[np.nan,4.],[5.,9.]])):
            with self.assertRaises(c.ContractError):
                b.pair_ht_core(codes,bad,indices,b.StarPairDesign(4,2))
        for bad in (np.array([0,0]),np.array([0,4]),np.array([-1,1])):
            with self.assertRaises(c.ContractError):
                b.pair_ht_core(codes,y,bad,b.StarPairDesign(4,2))

    def test_overflow_rejected_without_silent_zero_or_resampling(self):
        with self.assertRaises(c.ContractError):
            b.pair_ht_core(np.array([[0,2],[2,3]]),np.full((2,2),1e308),
                           np.array([0,1]),b.StarPairDesign(4,2))

    def test_not_identified_prevents_numeric_core(self):
        task = missing_fixture()
        request = req(task,status="NOT_IDENTIFIED",failure_type="UPSTREAM",
                      failure_message="original cause",failure_provenance={"code":"N17"})
        with patch.object(b,"pair_ht_core",side_effect=AssertionError("must not run")) as core:
            result = b.estimate_pair_ht(request,task)
        core.assert_not_called()
        self.assertEqual(result.execution_status,"NOT_IDENTIFIED")
        self.assertEqual(result.failure_message,"original cause")
        self.assertEqual(dict(result.failure_provenance),{"code":"N17"})
        self.assertTrue(np.isnan(result.ate).all())

    def test_identity_unsupported_fields_and_input_immutability(self):
        task = missing_fixture()
        request = req(task,requested_outputs=("mu","ite","ate","interval"))
        before = tuple(t.input_hash for t in task.context_refs)
        arrays = [getattr(t,name).tobytes() for t in task.context_refs for name in c.FACTUAL_KEYS]
        result = b.estimate_pair_ht(request,task)
        self.assertEqual(result.status,"POINT")
        self.assertEqual(result.request_id,request.request_id)
        self.assertEqual(result.target_id,request.target_id)
        self.assertEqual(result.pair_id,task.pair_id)
        self.assertEqual(result.context_ids,task.context_ids)
        self.assertEqual(result.input_hashes,before)
        self.assertEqual(result.protocol_hash,request.protocol_hash)
        self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
        self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)
        self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
        self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
        self.assertEqual(result.arm_results,(None,None))
        self.assertEqual(result.support.n_unique_observed_nodes,50)
        self.assertEqual(result.support.n_factual_records,100)
        for k,t in enumerate(task.context_refs):
            self.assertEqual(t.outcomes_observed.dtype,np.float64)
            c.validate_task(t)
        self.assertEqual(arrays,[getattr(t,name).tobytes()
                                for t in task.context_refs for name in c.FACTUAL_KEYS])
        self.assertFalse(result.ate.flags.writeable)

    def test_production_task_cannot_be_four_nodes_or_private_pair(self):
        with self.assertRaises(c.ContractError):
            b.estimate_pair_ht(None,{"N":4,"M":2})
        task = missing_fixture()
        with self.assertRaises(c.ContractError):
            b.estimate_pair_ht(replace(req(task),pair_id="wrong"),task)

    def test_shared_constraints_and_privileged_payload_negative(self):
        task = missing_fixture()
        inputs = tuple(dict(c.predictor_inputs(t)) for t in task.context_refs)
        changed = dict(inputs[1]); leaves = changed["treatment"].copy()
        leaves[1] = 1-leaves[1];changed["treatment"] = leaves
        with self.assertRaises(c.ContractError):
            b._public_records((inputs[0],changed),b.StarPairDesign(200,50))
        for key in ("truth","beta","h","full_y","noise"):
            changed = dict(inputs[0]);changed[key] = np.ones(200)
            with self.assertRaises(c.ContractError):
                b._public_records((changed,inputs[1]),b.StarPairDesign(200,50))


def tearDownModule():
    print("S3.2 HT EVIDENCE "+json.dumps(EVIDENCE,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
