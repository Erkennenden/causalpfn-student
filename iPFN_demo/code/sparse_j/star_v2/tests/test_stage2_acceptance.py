"""Stage 2 acceptance only: domain-separated fixtures and local HT references.

No baseline module, model execution, formal cohort, persistence or code repair.
The registered interface IDs are bookkeeping only: all mechanism/pair draws are
mocked to TEST_DOMAIN, which is outside the registered six-field entropy domain.
"""
from contextlib import contextmanager
from dataclasses import fields, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from star_v2.config import CONFIG
from star_v2 import dgp, pair, contracts, support
from star_v2.rng import RNGIdentity
from test_dgp_truth import reference_truth
from test_pair_data import readonly

ATOL = 1e-12  # Fixed fixture absolute tolerance; never the Oracle SVD threshold.
TEST_DOMAIN = (0x53544732, 0x41434337)
METRICS = {}


@contextmanager
def domain_separated_rng():
    records = []
    def factory(identity):
        if identity.channel not in ("population","x","coefficient",
                                     "leaf_assignment","noise","observation_index"):
            raise AssertionError("Unregistered test draw")
        entropy = [*TEST_DOMAIN,identity.cohort_code,identity.mechanism_id,
                   identity.entropy[4],identity.pair_repeat_id]
        if entropy[:2] == [CONFIG.root_seed,CONFIG.protocol_code]:
            raise AssertionError("Formal entropy consumed")
        records.append((identity,tuple(entropy)))
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    with patch.object(dgp,"make_generator",side_effect=factory), \
         patch.object(pair,"make_generator",side_effect=factory):
        yield records


def chain():
    with domain_separated_rng() as records:
        mechanism = dgp.generate_star_mechanism(cohort_code=1,mechanism_id=1)
        private = pair.generate_pair(mechanism,pair_repeat_id=1)
    tasks = contracts.pair_from_private(private)
    diagnostics = support.support_for_pair(tasks)
    inputs = tuple(contracts.predictor_inputs(t) for t in tasks.context_refs)
    return mechanism,private,tasks,diagnostics,inputs,records


def scalar_structural(a,h,beta,t):
    """Independent scalar form of protocol equation; no production DGP calls."""
    values=[]
    for i in range(len(a)):
        neighbors=[k for k in range(len(a)) if a[i,k]]
        hb=[sum(float(h[k,j]) for k in neighbors)/len(neighbors) for j in range(3)]
        hetero=[1.0,*map(float,h[i])]
        b=sum(v*float(c) for v,c in zip([*hetero,*hb],beta[:7]))
        tau=sum(v*float(c) for v,c in zip(hetero,beta[7:11]))
        gamma=sum(v*float(c) for v,c in zip(hetero,beta[11:15]))
        eta=sum(v*float(c) for v,c in zip(hetero,beta[15:19]))
        e=sum(float(t[k]) for k in neighbors)/len(neighbors)
        j=[sum(float(t[k])*float(h[k,q]) for k in neighbors)/len(neighbors)
           for q in range(3)]
        own=float(t[i])
        values.append(b+own*tau+(gamma+own*eta)*e
                      +sum(j[q]*float(beta[19+q]) for q in range(3))
                      +own*sum(j[q]*float(beta[22+q]) for q in range(3)))
    return np.asarray(values,dtype=np.float64)


def actual_states(a,t):
    return [(int(t[i]),int(sum(float(t[k]) for k in range(len(a)) if a[i,k])
                          /sum(a[i])>0.5)) for i in range(len(a))]


def fixture_a_design():
    a=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype=np.float64)
    assignments=[]
    for leaves in itertools.product((0,1),repeat=3):
        treatments=tuple(np.array([center,*leaves],dtype=np.float64) for center in (0,1))
        states=tuple(actual_states(a,t) for t in treatments)
        inclusion=np.zeros((4,2,2))
        for state_arm in states:
            for node,(t,s) in enumerate(state_arm):
                inclusion[node,t,s]+=1
        assignments.append((treatments,states,inclusion))
    subsets=list(itertools.combinations(range(4),2))
    q=sum(item[2] for item in assignments)/len(assignments)
    r=np.array([sum(i in obs for obs in subsets)/len(subsets) for i in range(4)])
    joint=np.zeros_like(q)
    for _,_,inc in assignments:
        for obs in subsets:
            mask=np.array([i in obs for i in range(4)])
            joint+=inc*mask[:,None,None]/(len(assignments)*len(subsets))
    weights=1.0/(r[:,None,None]*q)
    return a,assignments,subsets,q,r,joint,weights


def conditional_truth_by_enumeration(a,h,beta):
    """Independently enumerate T_-i conditional on each majority exposure."""
    truth=np.empty((4,2,2))
    for node,own in itertools.product(range(4),(0,1)):
        others=[i for i in range(4) if i!=node]
        groups={0:[],1:[]}
        for bits in itertools.product((0,1),repeat=3):
            treatment=np.zeros(4);treatment[node]=own;treatment[others]=bits
            exposure=actual_states(a,treatment)[node][1]
            groups[exposure].append(scalar_structural(a,h,beta,treatment)[node])
        for state in (0,1):
            truth[node,own,state]=sum(groups[state])/len(groups[state])
    return truth


def contrasts(mu):
    return np.array([mu[1,0]-mu[0,0],mu[1,1]-mu[1,0],mu[1,1]-mu[0,0]])


def local_ht_layer(kind):
    """48 paired records, no second division by number of center arms."""
    a,assignments,subsets,q,r,joint,weights=fixture_a_design()
    table=np.array([[[1.25,2.5],[4.0,7.75]],[[-1.5,0.5],[3.25,6.0]],
                    [[2.0,1.0],[5.0,8.0]],[[0.75,2.25],[6.25,10.0]]])
    h=np.array([[0.1,-0.3,0.5],[0.7,-0.2,0.4],
                [-0.4,0.6,0.8],[0.2,0.1,-0.5]],dtype=np.float64)
    beta=np.linspace(-0.5,0.6,25,dtype=np.float64)
    target=table if kind=="A" else conditional_truth_by_enumeration(a,h,beta)
    estimates=[];effects=[];empty_cells=0;empty_combinations=0;max_dgp_error=0.0
    reason_totals={name:0 for name in support.REASONS}
    for treatments,states,inclusion in assignments:
        if kind=="A":
            outcomes=tuple(np.array([table[i,t,s] for i,(t,s) in enumerate(state_arm)])
                           for state_arm in states)
        else:
            outcomes=tuple(dgp.structural_mean(a,h,beta,t) for t in treatments)
            for t,y in zip(treatments,outcomes):
                max_dgp_error=max(max_dgp_error,float(np.max(np.abs(y-scalar_structural(a,h,beta,t)))))
        for obs in subsets:
            estimate=np.zeros((2,2));counts=np.zeros((2,2),dtype=np.int64)
            # Both contexts share these exact leaves and this one observation set.
            np.testing.assert_array_equal(treatments[0][1:],treatments[1][1:])
            for arm in (0,1):
                for node in obs:
                    t,s=states[arm][node]
                    estimate[t,s]+=outcomes[arm][node]*weights[node,t,s]/len(a)
                    counts[t,s]+=1
                # Small-fixture Support is a local mathematical reference, not
                # a change to the production N=200 contract.
                reasons=[]
                for node,t,s in itertools.product(range(4),(0,1),(0,1)):
                    feasible=(t==arm) if node==0 else (s==arm)
                    realized=states[arm][node]==(t,s)
                    if not feasible: reason="conditional_structural_zero"
                    elif not realized: reason="assignment_zero"
                    elif node not in obs: reason="observation_sampling_zero"
                    else: reason="covered"
                    reasons.append(reason);reason_totals[reason]+=1
                    if reason=="covered":
                        assert realized and node in obs
                assert len(reasons)==16 and reasons.count("covered")==2
                assert reasons.count("conditional_structural_zero")==8
                assert reasons.count("assignment_zero")==4
                assert reasons.count("observation_sampling_zero")==2
            assert counts.sum()==4 and len(set(obs))==2
            empty=counts==0
            assert np.all(estimate[empty]==0)  # Zero HT contribution, NOT zero truth.
            empty_cells+=int(empty.sum());empty_combinations+=int(empty.any())
            estimates.append(estimate);effects.append(contrasts(estimate))
    averaged=np.mean(estimates,axis=0)
    expected=target.mean(0)
    effect_mean=np.mean(effects,axis=0)
    total_residual=max(abs(e[2]-e[0]-e[1]) for e in effects)
    info={"combinations":len(estimates),"state_mean_max_abs_residual":
          float(np.max(np.abs(averaged-expected))),
          "effect_max_abs_residual":float(np.max(np.abs(effect_mean-contrasts(expected)))),
          "total_identity_max_abs_residual":float(total_residual),
          "factual_dgp_reference_max_abs_residual":max_dgp_error,
          "empty_state_cells":empty_cells,"empty_combinations":empty_combinations,
          "reason_totals":reason_totals,"mean_state_estimates":averaged.tolist(),
          "target_state_means":expected.tolist()}
    return info,estimates,target,(a,h,beta)


class Stage2AcceptanceTests(unittest.TestCase):
    def close(self,label,actual,expected):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        METRICS[label]=max(METRICS.get(label,0.0),error)
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=0)

    def test_chain_identity_rng_protocol_and_shared_arrays(self):
        m,p,tasks,diag,inputs,records=chain()
        self.assertEqual(len(records),6)
        for index,(identity,test_entropy) in enumerate(records,1):
            self.assertEqual(identity.entropy,
                             (20261030,2026103001,1,1,index,0 if index<4 else 1))
            self.assertEqual(test_entropy[:2],TEST_DOMAIN)
        self.assertIs(p.mechanism,m)
        for ctx in p.contexts:
            self.assertIs(ctx.mechanism,m)
            self.assertIs(ctx.noise,p.noise)
            self.assertIs(ctx.observed_indices,p.observed_indices)
            self.assertIs(ctx.leaf_treatment,p.leaf_treatment)
        self.assertEqual((p.contexts[0].center_t,p.contexts[1].center_t),(0,1))
        for t in tasks.context_refs:
            self.assertEqual(t.protocol_hash,CONFIG.protocol_sha256)
            self.assertEqual(t.arm_order,("00","01","10","11"))
            self.assertEqual(t.pair_id,contracts.pair_id_for(1,1,1))
            self.assertEqual(t.context_id,contracts.context_id_for(1,1,1,t.center_t))
            np.testing.assert_array_equal(t.node_ids,np.arange(200))
            metadata=json.loads(t.rng_identity)
            self.assertEqual([r["entropy"] for r in metadata],
                             [list(i.entropy) for i,_ in records])
        self.assertEqual(diag.pair_id,tasks.pair_id)

    def test_chain_truth_effects_float64_and_shared_noise(self):
        m,p,tasks,diag,inputs,_=chain()
        self.close("chain_truth_reference",m.mu,reference_truth(m.adjacency,m.h,m.coefficients))
        ite=np.column_stack((m.mu[:,1,0]-m.mu[:,0,0],m.mu[:,1,1]-m.mu[:,1,0],
                             m.mu[:,1,1]-m.mu[:,0,0]))
        np.testing.assert_array_equal(m.ite,ite)
        self.close("chain_full_ate",m.ate,ite.mean(0))
        self.close("chain_total_identity",ite[:,2],ite[:,0]+ite[:,1])
        for ctx in p.contexts:
            for name in ("structural_mean","noise","full_outcomes","outcomes_observed"):
                self.assertEqual(getattr(ctx,name).dtype,np.float64)
            reference=scalar_structural(m.adjacency,m.h,m.coefficients,ctx.treatment)
            self.close("chain_structural_reference",ctx.structural_mean,reference)
        left,right=p.contexts
        residual=(right.full_outcomes-left.full_outcomes)-(right.structural_mean-left.structural_mean)
        self.close("chain_full_shared_noise",residual,0)
        self.close("chain_observed_shared_noise",residual[p.observed_indices],0)

    def test_chain_support_hash_and_predictor_correspondence(self):
        m,p,tasks,diag,inputs,_=chain()
        self.assertEqual((diag.n_unique_observed_nodes,diag.n_factual_records),(50,100))
        self.assertEqual(diag.design_support.sum(),800)
        self.assertEqual(diag.realized_state_mask.sum(),400)
        self.assertEqual(diag.empirical_support_mask.sum(),100)
        for task,ctx,report,payload in zip(tasks.context_refs,p.contexts,diag.context_supports,inputs):
            self.assertEqual(report.context_id,task.context_id)
            self.assertEqual(report.source_input_hash,task.input_hash)
            self.assertEqual(report.full_graph_count.sum(),200)
            self.assertEqual(report.observed_count.sum(),50)
            self.assertEqual(report.empirical_support_mask.sum(),50)
            for key in contracts.FACTUAL_KEYS:
                source=getattr(m,key) if key in ("x","adjacency") else getattr(ctx,key)
                np.testing.assert_array_equal(payload[key],source)
            self.assertEqual(tuple(payload),contracts.FACTUAL_KEYS)

    def test_chain_repeatability_and_metadata_hashes(self):
        a,b=chain(),chain()
        self.assertEqual(a[0].metadata(),b[0].metadata())
        self.assertEqual(a[1].metadata(),b[1].metadata())
        self.assertEqual([t.input_hash for t in a[2].context_refs],
                         [t.input_hash for t in b[2].context_refs])
        for t in a[2].context_refs:
            with self.assertRaises(contracts.ContractError):
                replace(t,input_hash="0"*64)

    def test_chain_negative_shapes_dtypes_identity_and_sharing(self):
        m,p,tasks,_,_,_=chain()
        t=tasks.context_refs[0]
        for kwargs in ({"x":readonly(t.x[:199])},
                       {"outcomes_observed":readonly(t.outcomes_observed.astype(np.float32))},
                       {"center_t":1},{"node_ids":readonly(t.node_ids[::-1])},
                       {"rng_identity":"[]"},{"context_id":"wrong"}):
            with self.assertRaises(contracts.ContractError):
                replace(t,**kwargs)
        bad=replace(p.contexts[1],noise=readonly(p.noise.copy()))
        broken=replace(p,contexts=(p.contexts[0],bad))
        with self.assertRaises(pair.PairValidationError) as caught:
            contracts.pair_from_private(broken)
        self.assertIs(caught.exception.evidence,broken)

    def test_five_keys_reject_privileged_and_support_metadata(self):
        _,_,tasks,diag,payloads,_=chain()
        payload=dict(payloads[0])
        for key in ("h","beta","truth","full_y","noise","rng_identity","input_hash","support"):
            with self.assertRaises(contracts.ContractError):
                contracts.validate_predictor_inputs({**payload,key:diag})
        with self.assertRaises(contracts.ContractError):
            contracts.predictor_inputs(tasks)

    def test_base_chains_and_storage_have_only_factual_budgets(self):
        m,p,tasks,diag,payloads,_=chain()
        privileged=[m.h,m.coefficients,m.mu,m.ite,m.ate,p.noise]
        privileged += [ctx.full_outcomes for ctx in p.contexts]
        for payload in payloads:
            for key,a in payload.items():
                current=a
                while type(current) is np.ndarray:
                    self.assertFalse(current.flags.writeable)
                    self.assertEqual(current.nbytes,a.nbytes)
                    current=current.base
                self.assertIs(type(current),bytes)
                self.assertEqual(len(current),a.nbytes)
                for hidden in privileged:
                    self.assertFalse(np.shares_memory(a,hidden))
            self.assertEqual(payload["outcomes_observed"].nbytes,400)

    def test_private_truth_beta_mutation_not_direct_predictor_input(self):
        m,p,tasks,_,payloads,_=chain()
        altered_m=replace(m,h=readonly(m.h+1),coefficients=readonly(m.coefficients+2),
                          mu=readonly(m.mu+3),ite=readonly(m.ite+4),ate=readonly(m.ate+5))
        # Deliberately hold facts fixed: this probes projection/isolation, not
        # physical validity of a changed DGP. No private-noise proof is asserted.
        changed_contexts=tuple(replace(ctx,mechanism=altered_m) for ctx in p.contexts)
        for ctx,old_task,old_payload in zip(changed_contexts,tasks.context_refs,payloads):
            task=contracts.task_from_context(ctx,pair_repeat_id=1,rng_identity=p.rng_identities)
            self.assertEqual(task.input_hash,old_task.input_hash)
            new=contracts.predictor_inputs(task)
            for key in contracts.FACTUAL_KEYS:
                np.testing.assert_array_equal(new[key],old_payload[key])
        invalid_private=replace(p,mechanism=altered_m,contexts=changed_contexts)
        with self.assertRaises(pair.PairValidationError):
            contracts.pair_from_private(invalid_private)

    def test_recomputed_facts_legitimately_change_after_beta_change(self):
        m,p,tasks,_,payloads,_=chain()
        beta=readonly(m.coefficients+np.linspace(0.1,0.3,25))
        mu=dgp.four_state_truth(m.adjacency,m.h,beta)
        ite,ate=dgp.effects_from_mu(mu)
        changed=replace(m,coefficients=beta,
                        components=readonly(dgp.structural_components(m.adjacency,m.h,beta)),
                        mu=readonly(mu),ite=readonly(ite),ate=readonly(ate))
        with domain_separated_rng():
            new_pair=pair.generate_pair(changed,pair_repeat_id=1)
        new_tasks=contracts.pair_from_private(new_pair)
        np.testing.assert_array_equal(p.noise,new_pair.noise)
        np.testing.assert_array_equal(p.observed_indices,new_pair.observed_indices)
        self.assertFalse(np.array_equal(payloads[0]["outcomes_observed"],
                                         contracts.predictor_inputs(new_tasks.context_refs[0])["outcomes_observed"]))
        self.assertNotEqual(tasks.context_refs[0].input_hash,new_tasks.context_refs[0].input_hash)

    def test_noise_proof_public_private_permission_difference(self):
        _,_,private_evidence,_,_,_=chain()
        public=contracts.pair_from_tasks(private_evidence.context_refs)
        self.assertEqual(private_evidence.noise_verification,"VERIFIED_SIMULATOR_PRIVATE")
        self.assertEqual(public.noise_verification,"NOT_VERIFIED_FROM_FACTUAL_INPUTS")
        self.assertIn("raw_noise",dict(private_evidence.shared_input_hashes))
        self.assertNotIn("raw_noise",dict(public.shared_input_hashes))
        self.assertFalse(hasattr(public,"noise"))

    def test_support_gate_method_capability_and_unbound_identification(self):
        _,_,tasks,diag,_,_=chain()
        report=diag.context_supports[0]
        query=tuple(map(int,np.argwhere(~report.empirical_support_mask)[0]))
        retained=np.full((200,2,2),1.2345)
        original=retained.tobytes()
        with self.assertRaises(support.PluginSupportError):
            support.require_plugin_support(report,(query,))
        self.assertEqual(retained.tobytes(),original)
        self.assertFalse(hasattr(report,"method_capability"))
        self.assertEqual(support.method_capability("HT").outputs,frozenset(("pair.ate",)))
        for task in tasks.context_refs:
            self.assertIsNone(task.target_id)
            self.assertIsNone(task.request_id)
            self.assertFalse(hasattr(task,"status"))
            with self.assertRaises(TypeError):
                replace(task,status="POINT")

    def test_original_arrays_and_hashes_remain_unchanged(self):
        m,p,tasks,diag,payloads,_=chain()
        before=m.metadata(),p.metadata(),[t.input_hash for t in tasks.context_refs]
        support.support_for_pair(tasks)
        for t in tasks.context_refs:
            contracts.predictor_inputs(t)
        self.assertEqual(before,(m.metadata(),p.metadata(),[t.input_hash for t in tasks.context_refs]))

    def test_fixture_a_probabilities_weights_and_pair_unit(self):
        a,assignments,subsets,q,r,joint,weights=fixture_a_design()
        self.assertEqual(len(assignments)*len(subsets),48)
        np.testing.assert_array_equal(q,0.5)
        np.testing.assert_array_equal(r,0.5)
        np.testing.assert_allclose(joint,0.25,atol=ATOL,rtol=0)
        np.testing.assert_array_equal(weights,4.0)
        self.assertTrue(all(np.max(inc)<=1 for _,_,inc in assignments))
        center_q=sum(math.comb(199,k) for k in range(100,200))/2**199
        self.assertEqual(center_q,0.5)
        formal_r=CONFIG.M/CONFIG.N
        self.assertEqual(formal_r,0.25)
        self.assertEqual(1.0/(formal_r*center_q),8.0)
        METRICS["fixture_design"]={"q":0.5,"r":0.5,"joint":0.25,"weight":4.0,
                                  "formal_r":formal_r,"formal_weight":8.0,
                                  "pair_units":48,"contexts_are_not_independent":True}

    def test_fixture_a_layer_a_48_fixed_state_tables(self):
        info,_,_,_=local_ht_layer("A")
        self.assertEqual(info["combinations"],48)
        self.assertLessEqual(info["state_mean_max_abs_residual"],ATOL)
        self.assertLessEqual(info["effect_max_abs_residual"],ATOL)
        self.assertLessEqual(info["total_identity_max_abs_residual"],ATOL)
        METRICS["fixture_layer_A"]=info

    def test_fixture_a_layer_b_48_continuous_exposure_dgp(self):
        info,_,target,(a,h,beta)=local_ht_layer("B")
        self.assertEqual(info["combinations"],48)
        self.assertLessEqual(info["state_mean_max_abs_residual"],ATOL)
        self.assertLessEqual(info["effect_max_abs_residual"],ATOL)
        self.assertLessEqual(info["factual_dgp_reference_max_abs_residual"],ATOL)
        self.close("fixture_B_production_truth_vs_independent",
                   dgp.four_state_truth(a,h,beta),target)
        # Same node/state can have different outcomes under continuous exposure.
        t0=np.array([0,0,0,0]);t1=np.array([0,1,0,0])
        self.assertEqual(actual_states(a,t0)[0],actual_states(a,t1)[0])
        self.assertGreater(abs(scalar_structural(a,h,beta,t0)[0]
                               -scalar_structural(a,h,beta,t1)[0]),ATOL)
        METRICS["fixture_layer_B"]=info

    def test_fixture_a_empty_state_reasons_and_total_identity(self):
        for kind in ("A","B"):
            info,estimates,target,_=local_ht_layer(kind)
            self.assertGreater(info["empty_combinations"],0)
            self.assertGreater(info["empty_state_cells"],0)
            self.assertEqual(info["reason_totals"],{
                "conditional_structural_zero":768,"assignment_zero":384,
                "observation_sampling_zero":192,"covered":192})
            for estimate in estimates:
                effect=contrasts(estimate)
                self.assertLessEqual(abs(effect[2]-effect[0]-effect[1]),ATOL)

    def test_no_real_model_or_legacy_generator_called(self):
        import torch
        import pfn_social.phase5_large_network as legacy
        dgp.common_projection()  # Fixed representation seed is not a cohort draw.
        with patch.object(torch,"load",side_effect=AssertionError("checkpoint")), \
             patch.object(legacy,"generate_mechanism",side_effect=AssertionError("legacy")), \
             patch.object(RNGIdentity,"generator",side_effect=AssertionError("formal RNG")):
            chain()

    def test_frozen_source_files_and_no_cache_or_formal_results(self):
        directory=Path(__file__).resolve().parents[1]
        expected={
            "protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md":"fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "dgp.py":"3509ba90b8278eb9fad142646001e450998088786a0326a869d7b2ad33be20c9",
            "pair.py":"7cfa1d7e80b5c079850d09ca6c3d399d32412074593fe5098239a6cc817bfcc8",
            "contracts.py":"bec8ce9087f4ffe5586589c29300dc8c2332698d2674d3534f6b87bd23d2a16e",
            "support.py":"f646e770bc740cbb15b06186e162f111084b4b454e7d5dc5778ff7c4f050bced"}
        for rel,digest in expected.items():
            self.assertEqual(hashlib.sha256((directory/rel).read_bytes()).hexdigest(),digest)
        self.assertFalse(list(directory.rglob("__pycache__")))
        self.assertFalse((directory.parents[2]/"results/e1_5_star_v2").exists())


def tearDownModule():
    print("STAGE2_ACCEPTANCE_METRICS "+json.dumps(METRICS,sort_keys=True))


if __name__=="__main__":
    unittest.main()
