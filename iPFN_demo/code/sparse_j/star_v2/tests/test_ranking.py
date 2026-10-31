"""6.2.3 qualification tests; no formal universe, sorting score or model calls."""
from dataclasses import replace
import unittest

import numpy as np

from star_v2 import ranking
from star_v2.coverage import AttemptSelectionAmbiguous
from star_v2.task_registry import TaskRegistry
from star_v2.support import ORDINARY_METHOD_IDS
from star_v2.estimation import PairEstimateBundle, EstimateBundle, EFFECT_ORDER
from star_v2.model_adapters import context_requests_from_parent
from star_v2.metrics import Prediction, MechanismTruth
from test_coverage import make_pair, request

ERRORS={}


def point_bundle(task,req,method,states=("OK","OK","OK"),arms=None):
    values=np.array([1.,2.,3.])
    for k,state in enumerate(states):
        if state!="OK":values[k]=np.nan
    failed=states!=("OK",)*3
    kwargs=dict(ate=values,output_status={"ate":states},
        failure_type="SCRIPTED_POINT" if failed else None,
        failure_message="effect-specific synthetic failure" if failed else None,
        failure_provenance={"stage":"synthetic_point"} if failed else {})
    if req.scope=="pair":
        kwargs["arm_results"]=arms if arms is not None else (None,None)
        cls=PairEstimateBundle
    else:cls=EstimateBundle
    return cls.from_outputs(req,task,method,**kwargs)


def ranking_fixture(*,failures=None,child_failure=None,omit_pair=None,omit_child=None,oracles=False):
    failures=failures or {};reg=TaskRegistry();declarations=[];specs=[]
    methods=ORDINARY_METHOD_IDS+(("oracle-context","oracle-pair") if oracles else ())
    for repeat in range(2):
        task=make_pair(0,repeat)
        for data in (task,)+task.context_refs:reg.register_task(data)
        for method in methods:
            parent=request(task,f"synthetic-ranking-{repeat}",outputs=("ate",))
            reg.register_request(method,parent)
            declarations.append((method,"pair",task.pair_id,parent.request_id))
            children=()
            if method in ("kernel-old","kernel-new","j","oracle-context"):
                children=context_requests_from_parent(task,parent,request_ids=("shared","shared"))
                for data,child in zip(task.context_refs,children):
                    reg.register_request(method,child)
                    declarations.append((method,"context",data.context_id,child.request_id))
            specs.append((method,repeat,task,parent,children))
    reg.seal(expected_task_ids=tuple(reg.tasks),expected_request_keys=tuple(declarations))
    predictions=[];parent_bundles={}
    for method,repeat,task,parent,children in specs:
        arms=tuple(point_bundle(data,child,method) for data,child in zip(task.context_refs,children))
        for arm,(data,child) in enumerate(zip(task.context_refs,children)):
            states=("OK",)*3
            if child_failure and (method,repeat,arm)==child_failure[:3]:
                values=["OK"]*3;values[child_failure[3]]="FAILED";states=tuple(values)
            bundle=point_bundle(data,child,method,states)
            if omit_child!=(method,repeat,arm):reg.append_attempt(bundle,attempt_no=1)
        # A deliberately stale success parent can coexist with a failed child
        # Attempt in the corruption fixture: qualification must check both.
        states=failures.get((method,repeat),("OK",)*3)
        bundle=point_bundle(task,parent,method,states,arms if children else None)
        parent_bundles[(method,repeat)]=bundle
        if omit_pair!=(method,repeat):
            reg.append_attempt(bundle,attempt_no=1)
            predictions.append(Prediction.from_bundle(bundle))
    return reg,tuple(predictions),(MechanismTruth(0,np.array([1.,2.,3.])),),parent_bundles


class RankingTests(unittest.TestCase):
    def test_all_seven_ordinary_methods_successful_no_ordinal_ranks(self):
        reg,predictions,truths,_=ranking_fixture()
        out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        self.assertEqual(set(out["methods"]),set(ORDINARY_METHOD_IDS))
        self.assertFalse(out["ordinal_ranks_produced"])
        self.assertEqual(out["evidence_scope"],"SYNTHETIC")
        for method in ORDINARY_METHOD_IDS:
            self.assertTrue(out["methods"][method]["all_effects_rankable"])
            for effect in EFFECT_ORDER:
                row=out["methods"][method]["effects"][effect]
                self.assertEqual(row["rankability_status"],"RANKABLE")
                self.assertEqual((row["requested_count"],row["valid_count"],row["failed_count"],row["missing_count"]),
                                 (2,2,0,0))
                self.assertEqual(row["statistics"]["pooled_rmse"],0.)
                ERRORS["complete_rmse"]=0.

    def test_one_pair_effect_failure_does_not_invalidate_other_effects(self):
        reg,predictions,truths,_=ranking_fixture(failures={("state-mean",0):("FAILED","OK","OK")})
        out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        effect=out["methods"]["state-mean"]["effects"]
        self.assertEqual(effect["Direct"]["rankability_status"],"NOT_RANKABLE")
        self.assertEqual((effect["Direct"]["requested_count"],effect["Direct"]["valid_count"],
                          effect["Direct"]["failed_count"]),(2,1,1))
        self.assertIsNone(effect["Direct"]["statistics"]["pooled_rmse"])
        self.assertEqual(effect["Direct"]["statistics"]["success_subset_pooled_rmse"],0.)
        self.assertEqual(effect["Spillover"]["rankability_status"],"RANKABLE")
        self.assertEqual(effect["Total"]["rankability_status"],"RANKABLE")
        self.assertFalse(out["methods"]["state-mean"]["all_effects_rankable"])

    def test_regression_effect_estimability_and_irrelevant_01_not_inferred(self):
        failed=("NOT_ESTIMABLE","OK","NOT_ESTIMABLE")
        reg,predictions,truths,_=ranking_fixture(failures={("f-z2",0):failed,("lin-z2",0):failed})
        out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        for method in ("f-z2","lin-z2"):
            effects=out["methods"][method]["effects"]
            self.assertEqual(effects["Direct"]["rankability_status"],"NOT_RANKABLE")
            self.assertEqual(effects["Spillover"]["rankability_status"],"RANKABLE")
            self.assertEqual(effects["Total"]["rankability_status"],"NOT_RANKABLE")
            self.assertEqual(effects["Direct"]["reason_distribution"]["NOT_ESTIMABLE"],1)
        # Eligibility consumes per-effect evidence, never state-01 counts.
        self.assertTrue(out["methods"]["ht"]["all_effects_rankable"])

    def test_kernel_and_j_necessary_context_failures_are_effect_specific(self):
        for method in ("kernel-old","kernel-new","j"):
            reg,predictions,truths,_=ranking_fixture(child_failure=(method,0,1,1))
            before=tuple(a.content for a in reg.attempts)
            out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
            effects=out["methods"][method]["effects"]
            self.assertEqual(effects["Direct"]["rankability_status"],"RANKABLE")
            self.assertEqual(effects["Spillover"]["rankability_status"],"NOT_RANKABLE")
            self.assertEqual(effects["Total"]["rankability_status"],"RANKABLE")
            self.assertEqual(effects["Spillover"]["reason_distribution"]["NECESSARY_CONTEXT_FAILED"],1)
            self.assertEqual(tuple(a.content for a in reg.attempts),before)

    def test_missing_context_or_unexecuted_pair_remains_in_denominator(self):
        for mode in ("context","pair"):
            options={"omit_child":("j",0,0)} if mode=="context" else {"omit_pair":("ht",0)}
            reg,predictions,truths,_=ranking_fixture(**options)
            out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
            method="j" if mode=="context" else "ht"
            for row in out["methods"][method]["effects"].values():
                self.assertEqual(row["requested_count"],2)
                self.assertEqual(row["valid_count"],1)
                self.assertEqual(row["missing_count"],1)
                self.assertEqual(row["rankability_status"],"NOT_RANKABLE")

    def test_missing_point_artifact_not_recovered_from_success_subset(self):
        reg,predictions,truths,_=ranking_fixture()
        missing=next(p.key for p in predictions if p.key[0]=="ht")
        remaining=tuple(p for p in predictions if p.key!=missing)
        out=ranking.evaluate_rankability(reg,remaining,truths,formal=False)
        row=out["methods"]["ht"]["effects"]["Direct"]
        self.assertEqual(row["rankability_status"],"NOT_RANKABLE")
        self.assertEqual(row["reason_distribution"]["MISSING_POINT_OUTPUT"],1)
        self.assertEqual(row["statistics"]["success_subset_pooled_rmse"],0.)
        self.assertIsNone(row["statistics"]["pooled_rmse"])

    def test_oracles_excluded_and_hajek_not_an_eighth_method(self):
        reg,predictions,truths,_=ranking_fixture(oracles=True)
        out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        self.assertEqual(set(out["methods"]),set(ORDINARY_METHOD_IDS))
        self.assertEqual(len(out["excluded_oracle_predictions"]),4)
        self.assertNotIn("oracle-context",out["methods"])
        self.assertNotIn("oracle-pair",out["methods"])
        self.assertNotIn("hajek",out["methods"])
        fake=replace(predictions[0],key=("hajek",)+predictions[0].key[1:])
        with self.assertRaises(ValueError):
            ranking.evaluate_rankability(reg,predictions+(fake,),truths,formal=False)

    def test_duplicate_extra_and_wrong_hash_prediction_rejected(self):
        reg,predictions,truths,_=ranking_fixture()
        with self.assertRaises(ValueError):
            ranking.evaluate_rankability(reg,predictions+(predictions[0],),truths,formal=False)
        bad=replace(predictions[0],key=predictions[0].key[:3]+("unknown",))
        with self.assertRaises(ValueError):
            ranking.evaluate_rankability(reg,predictions+(bad,),truths,formal=False)
        bad=replace(predictions[0],input_hashes=("f"*64,"e"*64))
        with self.assertRaises(ValueError):
            ranking.evaluate_rankability(reg,(bad,)+predictions[1:],truths,formal=False)

    def test_formal_count_validation_without_generating_4200_tasks(self):
        counts={method:600 for method in ORDINARY_METHOD_IDS}
        self.assertTrue(ranking.validate_test_ranking_counts(mechanism_count=200,pair_counts=counts))
        for bad in ({**counts,"ht":599},{**counts,"hajek":600},{**counts,"oracle-pair":600}):
            with self.assertRaises(ValueError):
                ranking.validate_test_ranking_counts(mechanism_count=200,pair_counts=bad)
        reg,predictions,truths,_=ranking_fixture()
        with self.assertRaises(ValueError):ranking.evaluate_rankability(reg,predictions,truths)

    def test_multiple_attempts_never_select_latest_success(self):
        reg,predictions,truths,bundles=ranking_fixture(failures={("ht",0):("FAILED",)*3})
        old=bundles[("ht",0)]
        success=PairEstimateBundle.from_outputs(old.request,make_pair(0,0),"ht",
            ate=np.array([1.,2.,3.]),output_status={"ate":("OK",)*3},arm_results=(None,None))
        reg.append_attempt(success,attempt_no=2,authorization="explicit synthetic retry")
        before=tuple(a.content for a in reg.attempts)
        with self.assertRaises(AttemptSelectionAmbiguous) as caught:
            ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        self.assertEqual(caught.exception.code,"ATTEMPT_SELECTION_AMBIGUOUS")
        self.assertEqual(tuple(a.content for a in reg.attempts),before)

    def test_qualification_does_not_modify_execution_status_or_input_arrays(self):
        reg,predictions,truths,_=ranking_fixture(failures={("f-z2",0):("NOT_ESTIMABLE","OK","NOT_ESTIMABLE")})
        before=tuple(a.content for a in reg.attempts)
        arrays=tuple(p.ate.tobytes() for p in predictions)
        out=ranking.evaluate_rankability(reg,predictions,truths,formal=False)
        self.assertEqual(out["methods"]["f-z2"]["effects"]["Direct"]["rankability_status"],"NOT_RANKABLE")
        self.assertEqual(tuple(a.content for a in reg.attempts),before)
        self.assertEqual(tuple(p.ate.tobytes() for p in predictions),arrays)
        self.assertTrue(any(a.document["execution_status"]=="NOT_ESTIMABLE" for a in reg.attempts))


if __name__=="__main__":
    unittest.main()
