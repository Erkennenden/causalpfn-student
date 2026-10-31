"""6.2.3 coverage hand references; small metadata-only synthetic plans.

No formal task universe, RNG stream, DGP, model or checkpoint. ATOL=1e-12.
"""
from dataclasses import replace
import unittest

import numpy as np

from star_v2 import contracts as c
from star_v2 import coverage
from star_v2.task_registry import TaskRegistry
from star_v2.estimation import (EstimationRequest, EstimateBundle, PairEstimateBundle,
    IntervalRecord, SamplingUncertainty, EFFECT_ORDER)
from star_v2.model_adapters import context_requests_from_parent
from star_v2.metrics import MechanismTruth
from star_v2.config import CONFIG
from test_kernel_new_adapter import tasks_fixture

ERRORS={}


def make_pair(mechanism=0,repeat=0):
    original,_,_=tasks_fixture()
    arms=[]
    for old in original.context_refs:
        changes={"mechanism_id":mechanism,"pair_repeat_id":repeat,
            "pair_id":c.pair_id_for(2,mechanism,repeat),
            "context_id":c.context_id_for(2,mechanism,repeat,old.center_t),
            "rng_identity":c._rng(2,mechanism,repeat)}
        metadata={name:changes.get(name,getattr(old,name)) for name in c._META}
        arrays=[(name,getattr(old,name)) for name in c._ARRAY_ORDER]
        digest=c._digest("star-v2-task-input-v1",metadata,arrays)
        arms.append(replace(old,**changes,input_hash=digest))
    return c.pair_from_tasks(tuple(arms))


def request(task,identity,outputs=("ate","interval")):
    return EstimationRequest.from_task(task,status="POINT",target_id=CONFIG.target_id,
        request_id=identity,requested_outputs=outputs)


def coverage_bundle(task,req,index,mode="mixed"):
    point=["OK"]*3;interval=["OK"]*3;values=np.array([1.,2.,3.]);records=[]
    for k,effect in enumerate(EFFECT_ORDER):
        failed=mode=="all_failed" or mode=="mixed" and k==1 and index>=8
        if failed:
            interval[k]="FAILED"
            if mode=="mixed":point[k]="FAILED";values[k]=np.nan
            records.append(IntervalRecord(task.context_id,effect,"FAILED",source="model_posterior"))
        else:
            bounds=((2.,3.) if index==9 else (0.,1.) if index==1 else (1.,2.)) if k==0 else (
                (1.,3.) if k==1 else (3.,3.))
            records.append(IntervalRecord(task.context_id,effect,"OK",*bounds,source="model_posterior"))
    unavailable="FAILED" in interval
    uncertainty=(SamplingUncertainty("FAILED","model_posterior","context",None)
        if "OK" not in interval else SamplingUncertainty("OK","model_posterior","context",None))
    return EstimateBundle.from_outputs(req,task,"j",ate=values,output_status={
        "ate":tuple(point),"interval":tuple(interval)},intervals=tuple(records),
        sampling_uncertainty=uncertainty,
        failure_type="SCRIPTED_DEPENDENCY" if unavailable else None,
        failure_message="scripted point/interval dependency" if unavailable else None,
        failure_provenance={"stage":"synthetic_interval_fixture"} if unavailable else {})


def coverage_fixture(mode="mixed",skip_index=None,references=False):
    reg=TaskRegistry();declarations=[];context_rows=[];pair_objects=[]
    for m,repeats in ((0,3),(1,2)):
        for r in range(repeats):
            task=make_pair(m,r);pair_objects.append(task)
            for data in (task,)+task.context_refs:reg.register_task(data)
            parent=request(task,f"synthetic-parent-{m}-{r}")
            reg.register_request("j",parent);declarations.append(("j","pair",task.pair_id,parent.request_id))
            children=context_requests_from_parent(task,parent,
                request_ids=(f"synthetic-{m}-{r}-0",f"synthetic-{m}-{r}-1"))
            for data,child in zip(task.context_refs,children):
                reg.register_request("j",child)
                key=("j","context",data.context_id,child.request_id)
                declarations.append(key);context_rows.append((data,child))
    if references:
        first=pair_objects[0]
        rc=request(first.context_refs[0],"oracle-context-reference")
        rp=request(first,"oracle-pair-reference")
        reg.register_request("oracle-context",rc);reg.register_request("oracle-pair",rp)
        declarations.extend((("oracle-context","context",rc.task_id,rc.request_id),
                             ("oracle-pair","pair",rp.task_id,rp.request_id)))
    expected_data=tuple(tid for p in pair_objects for tid in (p.pair_id,)+p.context_ids)
    reg.seal(expected_task_ids=expected_data,expected_request_keys=tuple(declarations))
    bundles=[]
    for i,(data,child) in enumerate(context_rows):
        bundle=coverage_bundle(data,child,i,mode);bundles.append(bundle)
        if i!=skip_index:reg.append_attempt(bundle,attempt_no=1)
    if references:
        for method,req,data in (("oracle-context",rc,first.context_refs[0]),("oracle-pair",rp,first)):
            source="joint_model_posterior" if method=="oracle-pair" else "model_posterior"
            scope="pair" if method=="oracle-pair" else "context"
            records=tuple(IntervalRecord(None if scope=="pair" else req.task_id,e,"OK",0.,4.,source=source)
                          for e in EFFECT_ORDER)
            cls=PairEstimateBundle if scope=="pair" else EstimateBundle
            kwargs={"arm_results":(None,None)} if scope=="pair" else {}
            bundle=cls.from_outputs(req,data,method,ate=np.array([1.,2.,3.]),
                output_status={"ate":("OK",)*3,"interval":("OK",)*3},intervals=records,
                sampling_uncertainty=SamplingUncertainty("OK",source,scope,None),**kwargs)
            reg.append_attempt(bundle,attempt_no=1)
    truths=tuple(MechanismTruth(m,np.array([1.,2.,3.])) for m in (0,1))
    return reg,tuple(bundles),truths,tuple(context_rows)


class CoverageTests(unittest.TestCase):
    def close(self,name,actual,expected):
        self.assertAlmostEqual(actual,expected,delta=1e-12)
        ERRORS[name]=max(ERRORS.get(name,0.),abs(actual-expected))

    def test_three_effects_independent_rates_widths_and_90pct_boundary(self):
        reg,_,truths,_=coverage_fixture()
        out=coverage.evaluate_j_coverage(reg,truths,formal=False)
        expected=((.9,.9,1.,1.,True),(1.,.8,.8,2.,False),(1.,1.,1.,0.,True))
        for effect,(valid,all_requested,success,width,passed) in zip(EFFECT_ORDER,expected):
            row=out["effects"][effect]
            self.assertEqual(row["all_requested"],10)
            self.close(effect+"_valid",row["coverage_valid"],valid)
            self.close(effect+"_overall",row["coverage_including_failures"],all_requested)
            self.close(effect+"_success",row["interval_success_rate"],success)
            self.close(effect+"_width",row["mean_interval_width"],width)
            self.assertEqual(row["meets_90pct_target"],passed)
        self.assertFalse(out["all_effects_meet_90pct_target"])
        self.assertEqual(out["evidence_scope"],"SYNTHETIC")
        self.assertEqual(len(out["records"]),30)

    def test_center_arm_denominators_diagnostic_only(self):
        reg,_,truths,_=coverage_fixture()
        out=coverage.evaluate_j_coverage(reg,truths,formal=False)
        for arm in (0,1):
            for effect in EFFECT_ORDER:
                self.assertEqual(out["center_t"][arm][effect]["all_requested"],5)
                self.assertNotIn("meets_90pct_target",out["center_t"][arm][effect])
        self.close("arm0_direct",out["center_t"][0]["Direct"]["coverage_including_failures"],1.)
        self.close("arm1_direct",out["center_t"][1]["Direct"]["coverage_including_failures"],.8)
        self.close("arm0_spillover",out["center_t"][0]["Spillover"]["coverage_including_failures"],.8)

    def test_closed_interval_boundaries_and_dependent_point_failures(self):
        reg,_,truths,_=coverage_fixture()
        rows=coverage.evaluate_j_coverage(reg,truths,formal=False)["records"]
        direct=[r for r in rows if r["effect"]=="Direct"]
        self.assertTrue(direct[0]["covered"])
        self.assertTrue(direct[1]["covered"])
        failures=[r for r in rows if r["failure_code"]=="DEPENDENT_POINT_FAILURE"]
        self.assertEqual(len(failures),2)
        self.assertTrue(all(not r["covered"] and r["truth"]==2. for r in failures))
        self.assertTrue(all("scripted" in r["failure_reason"]["execution_failure"] for r in failures))

    def test_zero_valid_returns_na_and_keeps_full_denominator(self):
        reg,_,truths,_=coverage_fixture(mode="all_failed")
        out=coverage.evaluate_j_coverage(reg,truths,formal=False)
        for row in out["effects"].values():
            self.assertEqual(row["all_requested"],10)
            self.assertEqual(row["valid_intervals"],0)
            self.assertIsNone(row["coverage_valid"])
            self.assertIsNone(row["mean_interval_width"])
            self.close("all_fail_main",row["coverage_including_failures"],0.)
            self.close("all_fail_success",row["interval_success_rate"],0.)

    def test_missing_attempt_and_missing_output_count_as_uncovered(self):
        reg,bundles,truths,_=coverage_fixture(skip_index=0)
        out=coverage.evaluate_j_coverage(reg,truths,formal=False)
        self.assertEqual(sum(r["failure_code"]=="UNEXECUTED" for r in out["records"]),3)
        self.assertTrue(all(r["all_requested"]==10 for r in out["effects"].values()))
        submissions=tuple(coverage.IntervalSubmission.from_bundle(b) for b in bundles[2:])
        out=coverage.evaluate_j_coverage(reg,truths,submissions=submissions,formal=False)
        self.assertEqual(sum(r["failure_code"]=="MISSING_INTERVAL_OUTPUT" for r in out["records"]),3)

    def test_nonfinite_reversed_and_damaged_bounds_are_invalid_not_deleted(self):
        for lower,upper in ((float("nan"),2.),(3.,2.),([1.],2.)):
            reg,bundles,truths,_=coverage_fixture()
            submissions=[coverage.IntervalSubmission.from_bundle(b) for b in bundles]
            original=submissions[0]
            records=[dict(r) for r in original.intervals]
            records[0].update(lower=lower,upper=upper)
            submissions[0]=replace(original,intervals=tuple(records))
            out=coverage.evaluate_j_coverage(reg,truths,submissions=submissions,formal=False)
            row=out["effects"]["Direct"]
            self.assertEqual(row["all_requested"],10)
            self.assertEqual(row["valid_intervals"],9)
            self.assertEqual(out["records"][0]["failure_code"],"INVALID_INTERVAL_BOUNDS")
            self.assertFalse(out["records"][0]["covered"])

    def test_missing_truth_is_retained_as_unassessable_failure(self):
        reg,_,truths,_=coverage_fixture()
        out=coverage.evaluate_j_coverage(reg,truths[:1],formal=False)
        self.assertEqual(sum(r["failure_code"]=="MISSING_TRUTH" for r in out["records"]),10)
        self.assertEqual(len(out["records"]),30)

    def test_oracle_scopes_separate_and_excluded_from_j_denominators(self):
        reg,_,truths,_=coverage_fixture(references=True)
        out=coverage.evaluate_j_coverage(reg,truths,formal=False)
        self.assertEqual(len(out["oracle_reference_records"]),2)
        self.assertEqual({r["scope"] for r in out["oracle_reference_records"]},{"context","pair"})
        self.assertTrue(all(row["all_requested"]==10 for row in out["effects"].values()))
        self.assertTrue(all(row["key"][0]=="j" for row in out["records"]))

    def test_identity_duplicates_extra_and_scope_forgery_rejected(self):
        reg,bundles,truths,_=coverage_fixture()
        records=[coverage.IntervalSubmission.from_bundle(b) for b in bundles]
        with self.assertRaises(ValueError):
            coverage.evaluate_j_coverage(reg,truths,submissions=records+[records[0]],formal=False)
        wrong=replace(records[0],input_hashes=("f"*64,))
        with self.assertRaises(ValueError):
            coverage.evaluate_j_coverage(reg,truths,submissions=[wrong],formal=False)
        damaged=[dict(r) for r in records[0].intervals];damaged[0]["context_id"]=None
        with self.assertRaises(ValueError):
            coverage.evaluate_j_coverage(reg,truths,
                submissions=[replace(records[0],intervals=tuple(damaged))],formal=False)
        wrong=replace(records[0],key=("j","context","unknown","request"))
        with self.assertRaises(ValueError): coverage.evaluate_j_coverage(reg,truths,submissions=[wrong],formal=False)

    def test_formal_count_gate_without_formal_task_generation(self):
        args=dict(mechanism_count=200,context_request_count=1200,
                  center_counts={0:600,1:600},effect_counts={e:1200 for e in EFFECT_ORDER})
        self.assertTrue(coverage.validate_test_coverage_counts(**args))
        for changed in (dict(context_request_count=1199),dict(center_counts={0:599,1:601}),
                        dict(effect_counts={"Direct":1200,"Spillover":1199,"Total":1200})):
            with self.assertRaises(ValueError):
                coverage.validate_test_coverage_counts(**{**args,**changed})
        reg,_,truths,_=coverage_fixture()
        with self.assertRaises(ValueError): coverage.evaluate_j_coverage(reg,truths)

    def test_multiple_attempts_block_instead_of_selecting_latest_success(self):
        reg,bundles,truths,contexts=coverage_fixture(mode="all_failed")
        success=coverage_bundle(contexts[0][0],contexts[0][1],0,mode="success")
        reg.append_attempt(success,attempt_no=2,authorization="explicit synthetic retry")
        before=tuple(a.content for a in reg.attempts)
        with self.assertRaises(coverage.AttemptSelectionAmbiguous) as caught:
            coverage.evaluate_j_coverage(reg,truths,formal=False)
        self.assertEqual(caught.exception.code,"ATTEMPT_SELECTION_AMBIGUOUS")
        self.assertEqual(tuple(a.content for a in reg.attempts),before)


if __name__=="__main__":
    unittest.main()
