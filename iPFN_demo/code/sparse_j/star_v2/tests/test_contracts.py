"""Deterministic contracts tests; reuses only S2.4 artificial/mock fixtures."""
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import contracts as c
from star_v2.config import CONFIG
from star_v2.pair import PairValidationError
from test_pair_data import fixture_pair, readonly


def tasks(private=None, **links):
    private = fixture_pair() if private is None else private
    return tuple(c.task_from_context(
        ctx, pair_repeat_id=private.pair_repeat_id,
        rng_identity=private.rng_identities, **links) for ctx in private.contexts)


def base_bytes(array):
    value = array
    while type(value) is np.ndarray:
        value = value.base
    return value


class ContractTests(unittest.TestCase):
    def test_task_fields_shapes_types_unbound_request(self):
        t = tasks()[0]
        self.assertEqual((t.protocol_id,t.protocol_version,t.protocol_hash),
                         (CONFIG.protocol_id,CONFIG.protocol_version,CONFIG.protocol_sha256))
        self.assertEqual((t.cohort,t.cohort_code,t.mechanism_id,t.pair_repeat_id,t.center_t),
                         ("test",2,3,2,0))
        self.assertEqual(t.reference_assignment,"iid Bernoulli(0.5)")
        self.assertEqual(t.arm_order,("00","01","10","11"))
        self.assertIsNone(t.target_id)
        self.assertIsNone(t.request_id)
        self.assertFalse(hasattr(t,"status"))
        self.assertEqual(t.task_id,t.context_id)
        np.testing.assert_array_equal(t.node_ids,np.arange(200))
        expected={"x":((200,20),"<f4"),"adjacency":((200,200),"<f8"),
                  "treatment":((200,),"<f8"),"observed_indices":((50,),"<i8"),
                  "outcomes_observed":((50,),"<f8")}
        for name,(shape,dtype) in expected.items():
            self.assertEqual(getattr(t,name).shape,shape)
            self.assertEqual(getattr(t,name).dtype.str,dtype)
        rng = json.loads(t.rng_identity)
        self.assertEqual(len(rng),6)
        self.assertEqual(rng[-1]["entropy"],[20261030,2026103001,2,3,6,2])
        self.assertEqual(rng[0]["seed_sequence"]["spawn_key"],[])

    def test_explicit_target_request_links_without_identification_status(self):
        t = tasks(target_id=CONFIG.target_id,request_id="upstream-request-17")[0]
        self.assertEqual(t.target_id,CONFIG.target_id)
        self.assertEqual(t.request_id,"upstream-request-17")
        for links in ({"target_id":"wrong"},{"request_id":"unbound"},
                      {"target_id":CONFIG.target_id,"request_id":""}):
            with self.subTest(links=links):
                with self.assertRaises(c.ContractError):
                    tasks(**links)
        with self.assertRaises(TypeError):
            replace(t,status="POINT")

    def test_reject_shape_dtype_indices_nodes_and_protocol(self):
        t = tasks()[0]
        bad_ids = t.observed_indices.copy(); bad_ids[1]=bad_ids[0]
        bad_node = readonly(t.node_ids[::-1])
        bad_treatment = t.treatment.copy(); bad_treatment[3]=2
        for changes in (
            {"node_ids":bad_node},{"x":t.x.astype(np.float64)},
            {"outcomes_observed":t.outcomes_observed.astype(np.float32)},
            {"observed_indices":readonly(bad_ids)},{"x":readonly(t.x[:199])},
            {"treatment":readonly(bad_treatment)},{"protocol_hash":"0"*64},
            {"root_seed":True},{"center_t":1},{"cohort":"pilot"},
            {"arm_order":("00","10","01","11")},
            {"adjacency":readonly(np.zeros((200,200)))},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(c.ContractError):
                    replace(t,**changes)

    def test_task_is_immutable_and_has_no_privileged_object(self):
        t = tasks()[0]
        with self.assertRaises(FrozenInstanceError):
            t.center_t=1
        names={f.name for f in fields(t)}
        self.assertFalse(names & {"mechanism","context","noise","h","coefficients",
                                  "full_outcomes","mu","ite","ate","truth"})
        for name in ("node_ids",)+c.FACTUAL_KEYS:
            a=getattr(t,name)
            self.assertFalse(a.flags.writeable)
            self.assertEqual(len(base_bytes(a)),a.nbytes)
            with self.assertRaises(ValueError):
                a.setflags(write=True)

    def test_pair_identity_counts_references_and_shared_evidence(self):
        p=c.pair_from_private(fixture_pair())
        self.assertEqual(p.context_ids,tuple(t.context_id for t in p.context_refs))
        self.assertIs(p.node_ids,p.context_refs[0].node_ids)
        self.assertEqual((p.cohort_code,p.mechanism_id,p.pair_repeat_id),(2,3,2))
        self.assertEqual((p.n_unique_observed_nodes,p.n_factual_records),(50,100))
        self.assertEqual(p.arm_order,CONFIG.arm_order)
        self.assertEqual(p.protocol_hash,CONFIG.protocol_sha256)
        self.assertEqual(set(dict(p.shared_input_hashes)),
                         {"x","adjacency","leaf_treatment","observed_indices","raw_noise"})
        self.assertIs(c.pair_from_tasks(p.context_refs).context_refs,p.context_refs)

    def test_observable_pair_never_claims_raw_noise_verification(self):
        p=c.pair_from_tasks(tasks())
        self.assertEqual(p.noise_verification,"NOT_VERIFIED_FROM_FACTUAL_INPUTS")
        self.assertNotIn("raw_noise",dict(p.shared_input_hashes))
        with self.assertRaises(TypeError):
            c.pair_from_tasks(p.context_refs,noise_verified=True)

    def test_private_noise_evidence_checked_bound_and_hash_only(self):
        private=fixture_pair()
        p=c.pair_from_private(private)
        self.assertEqual(p.noise_verification,"VERIFIED_SIMULATOR_PRIVATE")
        self.assertIsInstance(dict(p.shared_input_hashes)["raw_noise"],str)
        self.assertFalse(hasattr(p,"noise"))
        bad_context=replace(private.contexts[1],noise=readonly(private.noise.copy()))
        broken=replace(private,contexts=(private.contexts[0],bad_context))
        with self.assertRaises(PairValidationError) as caught:
            c.pair_from_private(broken)
        self.assertIs(caught.exception.evidence,broken)
        changed=c.pair_from_private(private,target_id=CONFIG.target_id,request_id="another")
        with self.assertRaises(c.ContractError):
            replace(changed,_noise_proof=p._noise_proof)

    def test_pair_rejects_missing_duplicate_reordered_or_conflicting_arms(self):
        a,b=tasks()
        for contexts in ((a,), (a,a), (b,a), [a,b]):
            with self.subTest(context_count=len(contexts)):
                with self.assertRaises(c.ContractError):
                    c.pair_from_tasks(contexts)
        linked=tasks(target_id=CONFIG.target_id,request_id="upstream")[1]
        with self.assertRaises(c.ContractError):
            c.pair_from_tasks((a,linked))

    def test_pair_rejects_shared_x_leaf_and_observation_conflicts(self):
        private=fixture_pair()
        a=tasks(private)[0]
        right=private.contexts[1]
        altered_x=readonly((private.mechanism.x+np.float32(0.125)).astype(np.float32))
        changed_m=replace(private.mechanism,x=altered_x)
        altered_t=right.treatment.copy(); altered_t[2]=1-altered_t[2]
        reordered=readonly(private.observed_indices[::-1])
        candidates=[
            replace(right,mechanism=changed_m),
            replace(right,treatment=readonly(altered_t),leaf_treatment=readonly(altered_t[1:])),
            replace(right,observed_indices=reordered,
                    outcomes_observed=readonly(right.full_outcomes[reordered])),
        ]
        for context in candidates:
            task=c.task_from_context(context,pair_repeat_id=2,
                                     rng_identity=private.rng_identities)
            with self.assertRaises(c.ContractError):
                c.pair_from_tasks((a,task))

    def test_input_hash_repeatability_changes_and_tamper_detection(self):
        private=fixture_pair()
        a,b=tasks(private)[0],tasks(private)[0]
        self.assertEqual(a.input_hash,b.input_hash)
        self.assertEqual(a.context_id,b.context_id)
        ctx=private.contexts[0]
        full=readonly(ctx.full_outcomes+np.float64(0.125))
        altered=replace(ctx,full_outcomes=full,
                        outcomes_observed=readonly(full[private.observed_indices]))
        changed=c.task_from_context(altered,pair_repeat_id=2,rng_identity=private.rng_identities)
        self.assertNotEqual(a.input_hash,changed.input_hash)
        self.assertEqual(a.context_id,changed.context_id)
        for kwargs in ({"input_hash":"0"*64},
                       {"outcomes_observed":changed.outcomes_observed},
                       {"rng_identity":"[]"},{"pair_id":"wrong"}):
            with self.assertRaises(c.ContractError):
                replace(a,**kwargs)
        with self.assertRaises(c.ContractError):
            replace(c.pair_from_private(private),shared_input_hashes=(("x","0"*64),))

    def test_hash_serialization_independent_reference(self):
        task=tasks()[0]
        meta_names=("protocol_id","protocol_version","protocol_hash","protocol_code","root_seed",
                    "cohort","cohort_code","mechanism_id","pair_repeat_id","center_t","pair_id",
                    "context_id","target_id","request_id","reference_assignment","arm_order","rng_identity")
        order=("node_ids","x","adjacency","treatment","observed_indices","outcomes_observed")
        arrays=[(name,getattr(task,name)) for name in order]
        doc={"schema":"star-v2-task-input-v1",
             "metadata":{name:getattr(task,name) for name in meta_names},
             "arrays":[]}
        doc["arrays"]=[{"name":name,"dtype":a.dtype.str,"shape":list(a.shape),
                        "order":"C","nbytes":a.nbytes} for name,a in arrays]
        header=json.dumps(doc,sort_keys=True,separators=(",",":"),ensure_ascii=True,
                          allow_nan=False).encode("ascii")
        h=hashlib.sha256()
        for data in [header]+[a.tobytes(order="C") for _,a in arrays]:
            h.update(len(data).to_bytes(8,"big"))
            h.update(data)
        self.assertEqual(h.hexdigest(),task.input_hash)

    def test_ids_independent_reference_and_cross_process(self):
        expected="star_v2_pair_"+hashlib.sha256(
            b'["e1_5_star_v2",2,3,2]').hexdigest()
        self.assertEqual(c.pair_id_for(2,3,2),expected)
        self.assertNotEqual(c.context_id_for(2,3,2,0),c.context_id_for(2,3,2,1))
        root=str(Path(__file__).resolve().parents[2])
        script=("import sys;sys.path.insert(0,"+repr(root)+");"
                "from star_v2.contracts import pair_id_for;"
                "print(pair_id_for(2,3,2))")
        actual=subprocess.check_output([sys.executable,"-I","-B","-c",script],
                                        text=True).strip()
        self.assertEqual(actual,expected)

    def test_predictor_exact_keys_no_metadata_or_task_body(self):
        t=tasks()[0]
        inputs=c.predictor_inputs(t)
        self.assertEqual(tuple(inputs),c.FACTUAL_KEYS)
        c.validate_predictor_inputs(inputs)
        with self.assertRaises(TypeError):
            inputs["truth"]=np.zeros(200)
        for obj in (fixture_pair(),c.pair_from_tasks(tasks()),{},t.rng_identity):
            with self.assertRaises(c.ContractError):
                c.predictor_inputs(obj)

    def test_privileged_dictionary_and_array_subclass_injection_rejected(self):
        inputs=dict(c.predictor_inputs(tasks()[0]))
        for name in ("h","beta","full_y","truth","noise","rng_identity","evaluation"):
            with self.subTest(name=name):
                with self.assertRaises(c.ContractError):
                    c.validate_predictor_inputs({**inputs,name:"privileged"})
        class Decorated(np.ndarray):
            pass
        bad=dict(inputs); bad["x"]=inputs["x"].view(Decorated)
        bad["x"].hidden_h=np.zeros((200,3))
        with self.assertRaises(c.ContractError):
            c.validate_predictor_inputs(bad)

    def test_predictor_base_chain_detached_from_every_private_array(self):
        private=fixture_pair()
        t=tasks(private)[0]
        inputs=c.predictor_inputs(t)
        priv=[getattr(private.mechanism,f.name) for f in fields(private.mechanism)
              if isinstance(getattr(private.mechanism,f.name),np.ndarray)]
        priv += [private.noise,private.leaf_treatment,private.observed_indices]
        for context in private.contexts:
            priv += [context.full_outcomes,context.outcomes_observed,context.treatment,
                     context.structural_mean]
        for name,a in inputs.items():
            self.assertIs(type(a),np.ndarray)
            self.assertEqual(type(base_bytes(a)),bytes)
            self.assertEqual(len(base_bytes(a)),a.nbytes)
            self.assertFalse(np.shares_memory(a,getattr(t,name)))
            for original in priv:
                self.assertFalse(np.shares_memory(a,original))
            with self.assertRaises(ValueError):
                a.setflags(write=True)

    def test_full_y_backing_view_is_copied_and_unsafe_payload_rejected(self):
        private=fixture_pair()
        ctx=private.contexts[0]
        ids=readonly(np.arange(50,dtype=np.int64))
        source=replace(ctx,observed_indices=ids,outcomes_observed=ctx.full_outcomes[:50])
        self.assertTrue(np.shares_memory(source.outcomes_observed,ctx.full_outcomes))
        task=c.task_from_context(source,pair_repeat_id=2,rng_identity=private.rng_identities)
        payload=c.predictor_inputs(task)
        self.assertEqual(payload["outcomes_observed"].nbytes,400)
        self.assertEqual(len(base_bytes(payload["outcomes_observed"])),400)
        self.assertFalse(np.shares_memory(payload["outcomes_observed"],ctx.full_outcomes))
        bad=dict(payload);bad["outcomes_observed"]=source.outcomes_observed
        with self.assertRaises(c.ContractError):
            c.validate_predictor_inputs(bad)

    def test_float64_originals_unchanged_and_no_runtime_generation(self):
        private=fixture_pair()
        before=private.metadata()
        original=[ctx.outcomes_observed.tobytes() for ctx in private.contexts]
        with patch("star_v2.pair.make_generator",side_effect=AssertionError("RNG")), \
             patch("star_v2.dgp.generate_star_mechanism",side_effect=AssertionError("DGP")):
            spec=c.pair_from_private(private)
            for task in spec.context_refs:
                inputs=c.predictor_inputs(task)
                self.assertEqual(inputs["outcomes_observed"].dtype,np.float64)
        self.assertEqual(before,private.metadata())
        self.assertEqual(original,[ctx.outcomes_observed.tobytes() for ctx in private.contexts])

    def test_context_alignment_identity_and_injection_errors_keep_evidence(self):
        private=fixture_pair()
        ctx=private.contexts[0]
        bad=replace(ctx,outcomes_observed=readonly(ctx.outcomes_observed[::-1]))
        with self.assertRaises(c.ContractError) as caught:
            c.task_from_context(bad,pair_repeat_id=2,rng_identity=private.rng_identities)
        self.assertIs(caught.exception.evidence,bad)
        with self.assertRaises(c.ContractError):
            c.task_from_context(ctx,pair_repeat_id=1,rng_identity=private.rng_identities)
        with self.assertRaises(TypeError):
            replace(tasks(private)[0],truth=private.mechanism.mu)

    def test_revalidation_detects_in_memory_field_tampering(self):
        t=tasks()[0]
        object.__setattr__(t,"input_hash","0"*64)
        with self.assertRaises(c.ContractError):
            c.predictor_inputs(t)

    def test_canonical_layout_and_endian_rules(self):
        private=fixture_pair()
        ctx=private.contexts[0]
        original=c.task_from_context(ctx,pair_repeat_id=2,rng_identity=private.rng_identities)
        m=replace(private.mechanism,x=np.asfortranarray(private.mechanism.x))
        alternate=c.task_from_context(replace(ctx,mechanism=m),pair_repeat_id=2,
                                      rng_identity=private.rng_identities)
        self.assertEqual(original.input_hash,alternate.input_hash)
        self.assertTrue(alternate.x.flags.c_contiguous)
        with self.assertRaises(c.ContractError):
            replace(original,outcomes_observed=original.outcomes_observed.astype(">f8"))


if __name__=="__main__":
    unittest.main()
