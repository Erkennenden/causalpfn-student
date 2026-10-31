"""Support-only deterministic in-memory fixtures; no estimator or formal cohort."""
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import contracts as c, pair, support as s
from test_pair_data import mechanism_fixture, scripted_rng, fixture_pair, readonly


def task_pair(leaf=None, include_center=True):
    if leaf is None:
        private=fixture_pair(include_center)
    else:
        with scripted_rng(include_center,leaf=leaf):
            private=pair.generate_pair(mechanism_fixture(),pair_repeat_id=2)
    tasks=tuple(c.task_from_context(ctx,pair_repeat_id=2,rng_identity=private.rng_identities)
                for ctx in private.contexts)
    return c.pair_from_tasks(tasks)


def query_from_mask(mask):
    return tuple(map(int,np.argwhere(mask)[0]))


class SupportTests(unittest.TestCase):
    def test_population_and_design_exact_for_both_centers(self):
        specs=task_pair().context_refs
        for task in specs:
            result=s.support_for_task(task)
            self.assertEqual(result.population_support.shape,(200,2,2))
            self.assertEqual(result.population_support.dtype,np.bool_)
            self.assertTrue(result.population_support.all())
            for node,t,state in itertools.product(range(200),(0,1),(0,1)):
                expected=(t==task.center_t) if node==0 else (state==task.center_t)
                self.assertEqual(bool(result.design_support[node,t,state]),expected)
            self.assertEqual(result.arm_order,("00","01","10","11"))
            np.testing.assert_array_equal(result.node_ids,np.arange(200))

    def test_center_majority_all_zero_all_one_and_boundary_counts(self):
        for count in (0,99,100,199):
            leaf=np.zeros(199,dtype=np.int64); leaf[:count]=1
            for task in task_pair(leaf).context_refs:
                result=s.support_for_task(task)
                self.assertTrue(result.realized_state_mask[0,task.center_t,int(count>99.5)])
                self.assertEqual(result.realized_state_mask[0].sum(),1)
                expected=np.zeros((199,2,2),dtype=bool)
                expected[np.arange(199),leaf,task.center_t]=True
                np.testing.assert_array_equal(result.realized_state_mask[1:],expected)

    def test_strict_half_exposure_pure_fixture(self):
        a=np.array([[0,1,1],[1,0,0],[1,0,0]])
        fractions,states=s.majority_exposure(a,np.array([1,1,0]))
        self.assertEqual(fractions[0],0.5)
        self.assertEqual(states[0],0)
        np.testing.assert_array_equal(fractions[1:],1)
        with self.assertRaises(ValueError):
            s.majority_exposure(a,np.array([1,2,0]))

    def test_reason_priority_against_independent_node_state_reference(self):
        for task in task_pair().context_refs:
            actual=s.support_for_task(task)
            observed=set(task.observed_indices.tolist())
            counts={name:0 for name in s.REASONS}
            for node,t,state in itertools.product(range(200),(0,1),(0,1)):
                feasible=t==task.center_t if node==0 else state==task.center_t
                neighbors=[k for k in range(200) if task.adjacency[node,k]]
                exposure_state=int(sum(int(task.treatment[k]) for k in neighbors)
                                   /len(neighbors)>0.5)
                realized=int(task.treatment[node])==t and exposure_state==state
                if not feasible:
                    reason="conditional_structural_zero"
                elif not realized:
                    reason="assignment_zero"
                elif node not in observed:
                    reason="observation_sampling_zero"
                else:
                    reason="covered"
                self.assertEqual(actual.support_reason[node,t,state],reason)
                self.assertEqual(bool(actual.empirical_support_mask[node,t,state]),
                                 realized and node in observed)
                counts[reason]+=1
            self.assertEqual(counts,{"conditional_structural_zero":400,"assignment_zero":200,
                                     "observation_sampling_zero":150,"covered":50})

    def test_reasons_mutually_exclusive_complete_and_population_independent(self):
        for task in task_pair().context_refs:
            result=s.support_for_task(task)
            indicators=np.stack([result.support_reason==name for name in s.REASONS])
            np.testing.assert_array_equal(indicators.sum(0),1)
            self.assertEqual(indicators.sum(),800)
            np.testing.assert_array_equal(indicators[0],~result.design_support)
            np.testing.assert_array_equal(indicators[3],result.empirical_support_mask)
            self.assertTrue(result.population_support[~result.design_support].all())

    def test_center_observed_flag_not_four_state_coverage(self):
        for include in (False,True):
            for task in task_pair(include_center=include).context_refs:
                result=s.support_for_task(task)
                self.assertEqual(result.center_observed,include)
                self.assertEqual(result.empirical_support_mask[0].sum(),int(include))
                self.assertEqual(result.observed_count[0].sum(),int(include))

    def test_center_leaf_counts_conservation_and_state_order(self):
        for include in (False,True):
            for task in task_pair(include_center=include).context_refs:
                result=s.support_for_task(task)
                full=np.zeros((2,2,2),dtype=np.int64)
                observed=np.zeros_like(full)
                obs=set(task.observed_indices.tolist())
                for node in range(200):
                    neighbors=np.flatnonzero(task.adjacency[node])
                    exposure=int(sum(task.treatment[neighbors])/len(neighbors)>0.5)
                    group=0 if node==0 else 1
                    own=int(task.treatment[node])
                    full[group,own,exposure]+=1
                    if node in obs:
                        observed[group,own,exposure]+=1
                np.testing.assert_array_equal(result.full_graph_count,full)
                np.testing.assert_array_equal(result.observed_count,observed)
                self.assertEqual(result.group_order,("center","leaf"))
                np.testing.assert_array_equal(result.full_graph_count.sum((1,2)),[1,199])
                self.assertEqual(result.full_graph_count.sum(),200)
                self.assertEqual(result.observed_count.sum(),50)

    def test_pair_design_realization_and_records_are_distinct(self):
        specs=task_pair()
        result=s.support_for_pair(specs)
        self.assertTrue(result.design_support.all())
        self.assertTrue(result.population_support.all())
        self.assertEqual(result.design_support.sum(),800)
        self.assertEqual(result.realized_state_mask.sum(),400)
        self.assertEqual(result.empirical_support_mask.sum(),100)
        np.testing.assert_array_equal(result.realized_state_mask.sum((1,2)),2)
        self.assertEqual((result.n_unique_observed_nodes,result.n_factual_records),(50,100))
        np.testing.assert_array_equal(result.observed_node_ids,specs.context_refs[0].observed_indices)
        for i,context in enumerate(result.context_supports):
            self.assertEqual(context.context_id,specs.context_ids[i])
            self.assertEqual(context.full_graph_count.sum(),200)
            self.assertEqual(context.observed_count.sum(),50)

    def test_pair_rejects_invalid_context_collection_via_existing_contract(self):
        specs=task_pair()
        with self.assertRaises(c.ContractError):
            s.support_for_pair(replace(specs,context_refs=specs.context_refs[::-1]))
        with self.assertRaises(c.ContractError):
            s.support_for_pair(fixture_pair())

    def test_extrapolation_flags_separated(self):
        for task in task_pair().context_refs:
            result=s.support_for_task(task)
            np.testing.assert_array_equal(result.empirical_extrapolation,
                                           ~result.empirical_support_mask)
            np.testing.assert_array_equal(result.context_design_extrapolation,
                                           ~result.design_support)
            self.assertEqual(result.empirical_extrapolation.sum(),750)
            self.assertEqual(result.context_design_extrapolation.sum(),400)
            self.assertTrue(np.any(result.empirical_extrapolation & result.design_support))

    def test_plugin_gate_denies_unsupported_keeps_predictions(self):
        result=s.support_for_task(task_pair().context_refs[0])
        supported=query_from_mask(result.empirical_support_mask)
        unsupported=query_from_mask(~result.design_support)
        prediction=np.full((200,2,2),3.141592653589793)
        before=prediction.tobytes()
        self.assertTrue(s.require_plugin_support(result,(supported,)).allowed)
        check=s.check_plugin_use(result,(supported,unsupported))
        self.assertFalse(check.allowed)
        self.assertEqual(check.unsupported_queries,(unsupported,))
        self.assertEqual(check.design_extrapolation_queries,(unsupported,))
        with self.assertRaises(s.PluginSupportError) as caught:
            s.require_plugin_support(result,(unsupported,))
        self.assertFalse(caught.exception.check.allowed)
        self.assertEqual(prediction.tobytes(),before)

    def test_plugin_explicit_authorization_preserves_extrapolation_diagnostics(self):
        result=s.support_for_pair(task_pair())
        query=query_from_mask(~result.empirical_support_mask)
        check=s.require_plugin_support(result,(query,),extrapolation_authorized=True)
        self.assertTrue(check.allowed)
        self.assertEqual(check.unsupported_queries,(query,))
        self.assertEqual(check.design_extrapolation_queries,())
        self.assertFalse(result.empirical_support_mask[query])

    def test_plugin_rejects_invalid_query_and_authorization_types(self):
        result=s.support_for_task(task_pair().context_refs[0])
        for queries in ([(0,0,0)],((200,0,0),),((0,2,0),),((True,0,0),),((0,0),)):
            with self.subTest(queries=queries):
                with self.assertRaises((ValueError,TypeError)):
                    s.check_plugin_use(result,queries)
        with self.assertRaises(TypeError):
            s.check_plugin_use(result,(),extrapolation_authorized=1)

    def test_seven_ordinary_capabilities_ate_only_and_intervals(self):
        self.assertEqual(len(s.ORDINARY_METHOD_IDS),7)
        for name in ("State Mean","HT","F-z2","Lin-z2"):
            capability=s.method_capability(name)
            self.assertEqual(capability.outputs,frozenset(("pair.ate",)))
            self.assertTrue(capability.ordinary_ranking)
        for name in ("kernel-old","kernel-new"):
            outputs=s.method_capability(name).outputs
            self.assertTrue({"context.mu","context.ite","pair.ate"}<=outputs)
            self.assertFalse(any("interval" in output for output in outputs))
        outputs=s.method_capability("iPFN-J").outputs
        self.assertIn("context.ate_interval",outputs)
        self.assertNotIn("pair.ate_interval",outputs)

    def test_capability_not_requested_unsupported_and_independent_support(self):
        self.assertEqual(s.output_capability("HT","context.mu",requested=True),"UNSUPPORTED")
        self.assertEqual(s.output_capability("HT","context.mu",requested=False),"NOT_REQUESTED")
        self.assertEqual(s.output_capability("HT","pair.ate",requested=True),"SUPPORTED")
        capabilities=tuple(s.METHOD_CAPABILITIES.items())
        for task in task_pair().context_refs:
            s.support_for_task(task)
        self.assertEqual(capabilities,tuple(s.METHOD_CAPABILITIES.items()))
        with self.assertRaises(ValueError):
            s.output_capability("HT","unknown",requested=True)

    def test_oracles_reference_privileged_and_outside_ordinary_ranking(self):
        for name in ("oracle-context","oracle-pair"):
            capability=s.method_capability(name)
            self.assertTrue(capability.privileged)
            self.assertTrue(capability.reference_only)
            self.assertFalse(capability.ordinary_ranking)
            self.assertNotIn(name,s.ORDINARY_METHOD_IDS)
        self.assertIn("context.ate_interval",s.method_capability("oracle-context").outputs)
        self.assertEqual(s.method_capability("oracle-pair").outputs,
                         frozenset(("pair.ate","pair.ate_interval")))

    def test_outputs_and_capability_metadata_are_immutable(self):
        result=s.support_for_task(task_pair().context_refs[0])
        for f in fields(result):
            value=getattr(result,f.name)
            if isinstance(value,np.ndarray):
                self.assertFalse(value.flags.writeable)
                with self.assertRaises(ValueError):
                    value.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            result.center_observed=False
        with self.assertRaises(TypeError):
            s.METHOD_CAPABILITIES["new"]=None
        with self.assertRaises(FrozenInstanceError):
            s.method_capability("HT").privileged=True

    def test_only_canonical_facts_used_no_privileged_computation(self):
        specs=task_pair()
        with patch("star_v2.pair.validate_pair",side_effect=AssertionError("private data")), \
             patch("star_v2.dgp.structural_mean",side_effect=AssertionError("DGP")), \
             patch("star_v2.rng.make_generator",side_effect=AssertionError("RNG")):
            result=s.support_for_pair(specs)
        self.assertEqual(result.empirical_support_mask.sum(),100)
        with self.assertRaises(c.ContractError):
            s.support_for_task(fixture_pair().contexts[0])
        self.assertFalse(hasattr(result,"noise"))
        self.assertFalse(hasattr(result,"method_capability"))

    def test_support_values_independent_of_factual_outcome_numbers(self):
        private=fixture_pair()
        original=c.task_from_context(private.contexts[0],pair_repeat_id=2,
                                      rng_identity=private.rng_identities)
        context=private.contexts[0]
        full=readonly(context.full_outcomes+123.0)
        altered=replace(context,full_outcomes=full,
                        outcomes_observed=readonly(full[private.observed_indices]))
        other=c.task_from_context(altered,pair_repeat_id=2,rng_identity=private.rng_identities)
        left,right=s.support_for_task(original),s.support_for_task(other)
        self.assertNotEqual(left.source_input_hash,right.source_input_hash)
        for f in fields(left):
            a,b=getattr(left,f.name),getattr(right,f.name)
            if isinstance(a,np.ndarray):
                np.testing.assert_array_equal(a,b)

    def test_input_hashes_and_predictor_budget_unchanged(self):
        private=fixture_pair()
        before=private.metadata()
        tasks=tuple(c.task_from_context(ctx,pair_repeat_id=2,rng_identity=private.rng_identities)
                    for ctx in private.contexts)
        specs=c.pair_from_tasks(tasks)
        before_hashes=[t.input_hash for t in tasks]
        before_bytes=[tuple(c.predictor_inputs(t)[name].tobytes() for name in c.FACTUAL_KEYS)
                      for t in tasks]
        s.support_for_pair(specs)
        self.assertEqual(before,private.metadata())
        self.assertEqual(before_hashes,[t.input_hash for t in tasks])
        for t,values in zip(tasks,before_bytes):
            c.validate_task(t)
            inputs=c.predictor_inputs(t)
            self.assertEqual(tuple(inputs),c.FACTUAL_KEYS)
            self.assertEqual(values,tuple(inputs[name].tobytes() for name in c.FACTUAL_KEYS))

    def test_frozen_contracts_and_no_cache_or_formal_artifacts(self):
        directory=Path(__file__).resolve().parents[1]
        expected={
            "protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md":"fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "contracts.py":"bec8ce9087f4ffe5586589c29300dc8c2332698d2674d3534f6b87bd23d2a16e",
            "pair.py":"7cfa1d7e80b5c079850d09ca6c3d399d32412074593fe5098239a6cc817bfcc8"}
        for rel,digest in expected.items():
            self.assertEqual(hashlib.sha256((directory/rel).read_bytes()).hexdigest(),digest)
        self.assertFalse(list(directory.rglob("__pycache__")))
        self.assertFalse((directory.parents[2]/"results/e1_5_star_v2").exists())


if __name__=="__main__":
    unittest.main()
