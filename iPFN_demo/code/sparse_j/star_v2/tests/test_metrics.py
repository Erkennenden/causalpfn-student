"""6.2.2 hand references only; no model, RNG, DGP, cohort or file outputs."""
from dataclasses import replace
import hashlib
import math
import unittest

import numpy as np

from star_v2 import metrics as m
from star_v2.estimation import EFFECT_ORDER
from star_v2.support import ContextSupport, REASONS, ORDINARY_METHOD_IDS
from test_task_registry import registry_fixture, bundle_fixture

ATOL = 1e-12
MAX_ERRORS = {}
BASE_ERRORS = ((1.,2.,3.),(-4.,-2.,0.),(0.,1.,2.))
FACTORS = (1.,2.,-1.)


def identity_hash(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def expected(method,mechanism,repeat,center=None,n=4):
    pair = f"synthetic-pair-{mechanism}-{repeat}"
    scope = "pair" if center is None else "context"
    task = pair if center is None else f"synthetic-context-{mechanism}-{repeat}-{center}"
    hashes = (identity_hash(pair+"-0"),identity_hash(pair+"-1"))
    return m.ExpectedTask((method,scope,task,"synthetic-request"),pair,mechanism,repeat,center,
        hashes if center is None else (hashes[center],),tuple(range(n)),
        ("ate",) if center is None else ("mu","ite","ate"))


def truth(mechanism,n=4):
    base = np.arange(n,dtype=np.float64)+mechanism
    mu = np.zeros((n,2,2));mu[:,0,1] = base/2;mu[:,1,0] = base;mu[:,1,1] = 3*base
    ite = np.column_stack((base,2*base,3*base))
    return m.MechanismTruth(mechanism,ite.mean(0),mu,ite)


def pair_prediction(task,true,error):
    defaults = "UNSUPPORTED" if task.key[0] in ("state-mean","ht","f-z2","lin-z2") else "NOT_REQUESTED"
    statuses = {"mu":defaults,"ite":(defaults,)*3,"ate":("OK",)*3}
    return m.Prediction(task.key,task.input_hashes,task.node_ids,statuses,
                        ate=true.ate+np.array([error,2*error,3*error]))


def pair_fixture(methods=("state-mean",),repeats=(3,3,3)):
    tasks=[];records=[];truths=tuple(truth(i) for i in range(len(repeats)))
    for method in methods:
        for mechanism,count in enumerate(repeats):
            for repeat in range(count):
                task = expected(method,mechanism,repeat)
                tasks.append(task)
                records.append(pair_prediction(task,truths[mechanism],BASE_ERRORS[mechanism][repeat]))
    return m.EvaluationPlan(tuple(tasks),synthetic=True),tuple(records),truths


def context_prediction(task,true):
    profile = np.array([2.,-1.,-1.,-1.])*FACTORS[task.mechanism_id]
    mu = true.mu.copy()
    mu[:,0,1] += 4*profile;mu[:,1,0] += profile;mu[:,1,1] += 3*profile
    ite = true.ite+np.column_stack((profile,2*profile,3*profile))
    return m.Prediction(task.key,task.input_hashes,task.node_ids,
        {"mu":"OK","ite":("OK",)*3,"ate":("OK",)*3},mu,ite,ite.mean(0))


def supplied_support(task):
    n = 4
    treatment = np.array([task.center_t,0,1,0])
    observed = np.array([True,True,False,False]) if task.pair_repeat_id%2 == 0 else np.array([False,False,True,True])
    design = np.zeros((n,2,2),dtype=bool)
    design[0,task.center_t,:] = True;design[1:,:,task.center_t] = True
    realized = np.zeros_like(design)
    reason = np.empty((n,2,2),dtype="<U32")
    for node in range(n):
        exposure = 0 if node == 0 else task.center_t
        realized[node,treatment[node],exposure] = True
        for t in (0,1):
            for s in (0,1):
                if not design[node,t,s]: category = REASONS[0]
                elif (t,s) != (treatment[node],exposure): category = REASONS[1]
                elif not observed[node]: category = REASONS[2]
                else: category = REASONS[3]
                reason[node,t,s] = category
    empirical = realized&observed[:,None,None]
    full = np.stack((realized[:1].sum(0),realized[1:].sum(0))).astype(np.int64)
    counts = np.stack((empirical[:1].sum(0),empirical[1:].sum(0))).astype(np.int64)
    return ContextSupport(task.key[2],task.input_hashes[0],task.center_t,np.arange(n,dtype=np.int64),
        np.ones_like(design),design,realized,observed,empirical,reason,full,counts,
        bool(observed[0]),~empirical,~design)


def node_fixture(methods=("j",),repeats=(3,3,3)):
    tasks=[];records=[];supports=[]
    truths=tuple(truth(i) for i in range(len(repeats)))
    for method in methods:
        for mechanism,count in enumerate(repeats):
            for repeat in range(count):
                for center in (0,1):
                    task = expected(method,mechanism,repeat,center)
                    tasks.append(task);records.append(context_prediction(task,truths[mechanism]))
                    supports.append((task.key,supplied_support(task)))
    return m.EvaluationPlan(tuple(tasks),synthetic=True),tuple(records),truths,tuple(supports)


def fingerprint(values):
    return tuple((v.dtype.str,v.shape,hashlib.sha256(v.tobytes()).hexdigest()) for v in values)


class MetricsTests(unittest.TestCase):
    def close(self,label,actual,expected):
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=0)
        MAX_ERRORS[label] = max(MAX_ERRORS.get(label,0.),float(np.max(
            np.abs(np.asarray(actual)-np.asarray(expected)))))

    def test_pair_six_metrics_independent_hand_values_for_seven_methods(self):
        plan,predictions,truths = pair_fixture(ORDINARY_METHOD_IDS)
        result = m.pair_ate_metrics(plan,predictions,truths)
        self.assertEqual(set(result["ordinary"]),set(ORDINARY_METHOD_IDS))
        self.assertEqual(dict(result["oracle_reference"]),{})
        for method in ORDINARY_METHOD_IDS:
            for k,effect in enumerate(EFFECT_ORDER):
                row = result["ordinary"][method][effect];scale=k+1
                self.assertEqual(row["status"],"COMPLETE")
                self.assertEqual((row["n_requested"],row["n_valid"]),(9,9))
                self.close("pair_rmse",row["pooled_rmse"],math.sqrt(13/3)*scale)
                self.close("pair_bias",row["signed_bias"],scale/3)
                self.close("pair_absolute_bias",row["absolute_bias"],scale/3)
                self.close("pair_acb",row["absolute_conditional_bias"],5*scale/3)
                self.close("pair_within",row["within_mechanism_variance"],2*scale**2)
                self.close("pair_between",row["between_mechanism_variability"],13*scale**2/3)
                self.assertNotEqual(row["absolute_bias"],row["absolute_conditional_bias"])
        row = result["ordinary"]["state-mean"]["Direct"]
        self.close("per_mechanism_bias",[v["mean_error"] for v in row["per_mechanism"]],[2.,-2.,1.])
        self.close("per_mechanism_variance",[v["within_variance"] for v in row["per_mechanism"]],[1.,4.,1.])

    def test_mechanism_equal_weighting_not_naive_pair_pooling(self):
        plan,records,truths = pair_fixture(repeats=(1,3))
        adjusted=[]
        for task,record in zip(plan.tasks,records):
            error = 0. if task.mechanism_id == 0 else 2.
            adjusted.append(pair_prediction(task,truths[task.mechanism_id],error))
        row = m.pair_ate_metrics(plan,adjusted,truths)["ordinary"]["state-mean"]["Direct"]
        self.close("unequal_rmse",row["pooled_rmse"],math.sqrt(2))
        self.close("unequal_bias",row["signed_bias"],1.)
        self.assertNotEqual(row["signed_bias"],1.5)
        self.assertIsNone(row["within_mechanism_variance"])  # ddof=1 unavailable for one replicate

    def test_single_effect_failure_keeps_other_effects_complete(self):
        plan,records,truths = pair_fixture()
        records=list(records);record=records[0]
        ate=record.ate.copy();ate[0]=np.nan
        records[0]=replace(record,ate=ate,output_status={
            **record.output_status,"ate":("FAILED","OK","OK")},
            failure_reasons={"reason":"specific Direct failure"},execution_status="FAILED")
        result=m.pair_ate_metrics(plan,records,truths)["ordinary"]["state-mean"]
        direct=result["Direct"]
        self.assertEqual(direct["status"],"INCOMPLETE")
        self.assertEqual((direct["n_requested"],direct["n_valid"]),(9,8))
        for field in m.PAIR_FIELDS: self.assertIsNone(direct[field])
        self.assertIsNotNone(direct["success_subset_pooled_rmse"])
        self.assertEqual(direct["failures"][0]["reason"]["reason"],"specific Direct failure")
        self.assertEqual(result["Spillover"]["status"],"COMPLETE")
        self.assertEqual(result["Total"]["status"],"COMPLETE")

    def test_missing_entire_mechanism_never_becomes_full_cohort(self):
        plan,records,truths=pair_fixture()
        records=tuple(p for p in records if "-2-" not in p.key[2])
        row=m.pair_ate_metrics(plan,records,truths)["ordinary"]["state-mean"]["Direct"]
        self.assertEqual((row["n_requested"],row["n_valid"],row["n_failed_or_missing"]),(9,6,3))
        self.assertEqual(row["n_mechanisms_requested"],3)
        self.assertEqual(row["n_mechanisms_with_valid"],2)
        self.assertIsNone(row["pooled_rmse"])
        self.close("subset_bias",row["success_subset_signed_bias"],0.)
        self.assertTrue(all(r["status"]=="MISSING" for r in row["failures"]))

    def test_oracle_reports_separate_and_ate_only_node_outputs_unsupported(self):
        methods=("state-mean","oracle-context","oracle-pair")
        plan,records,truths=pair_fixture(methods)
        pair=m.pair_ate_metrics(plan,records,truths)
        self.assertEqual(set(pair["ordinary"]),{"state-mean"})
        self.assertEqual(set(pair["oracle_reference"]),{"oracle-context","oracle-pair"})
        nodes=m.node_metrics(plan,records,truths)
        delta=m.center_leaf_metrics(plan,records,truths)
        for method,bucket in (("state-mean","ordinary"),("oracle-pair","oracle_reference")):
            self.assertEqual(nodes[bucket][method]["status"],"UNSUPPORTED")
            self.assertEqual(delta[bucket][method]["status"],"UNSUPPORTED")
        self.assertEqual(nodes["oracle_reference"]["oracle-context"]["groups"]["all"]["mu"]["status"],
                         "NOT_REQUESTED")

    def test_node_mu_ite_center_leaf_all_mse_and_rmse_hand_reference(self):
        plan,records,truths,_=node_fixture(("j","kernel-old","kernel-new","oracle-context"))
        report=m.node_metrics(plan,records,truths)
        for method in ("j","kernel-old","kernel-new","oracle-context"):
            bucket="oracle_reference" if method=="oracle-context" else "ordinary"
            groups=report[bucket][method]["groups"]
            for group,expected_mu,expected_direct in (("center",52.,8.),("leaf",13.,2.),("all",22.75,3.5)):
                self.close("node_mu_mse",groups[group]["mu"]["mse"],expected_mu)
                self.close("node_mu_rmse",groups[group]["mu"]["pooled_rmse"],math.sqrt(expected_mu))
                self.assertEqual(groups[group]["mu"]["n_requested"],18)
                for k,effect in enumerate(EFFECT_ORDER):
                    expected=expected_direct*(k+1)**2
                    self.close("node_ite_mse",groups[group]["ite"][effect]["mse"],expected)
                    self.close("node_ite_rmse",groups[group]["ite"][effect]["pooled_rmse"],math.sqrt(expected))

    def test_group_ate_independent_group_means_and_context_aggregation(self):
        plan,records,truths,_=node_fixture()
        groups=m.node_metrics(plan,records,truths)["ordinary"]["j"]["groups"]
        for group,rmse,bias,acb in (
            ("center",math.sqrt(8),4/3,8/3),
            ("leaf",math.sqrt(2),-2/3,4/3),
            ("all",math.sqrt(.125),-1/6,1/3)):
            for k,effect in enumerate(EFFECT_ORDER):
                row=groups[group]["group_ate"][effect]
                self.close("group_ate_rmse",row["pooled_rmse"],rmse*(k+1))
                self.close("group_ate_bias",row["signed_bias"],bias*(k+1))
                self.close("group_ate_acb",row["absolute_conditional_bias"],acb*(k+1))
        first=groups["center"]["group_ate"]["Direct"]["raw_rows"][0]
        self.close("group_true",first["true_group_effect"],0.)
        self.close("group_predicted",first["predicted_group_effect"],2.)

    def test_center_leaf_context_errors_and_mechanism_statistics(self):
        plan,records,truths,_=node_fixture(("j","oracle-context"))
        report=m.center_leaf_metrics(plan,records,truths)
        for bucket,method in (("ordinary","j"),("oracle_reference","oracle-context")):
            output=report[bucket][method]
            self.assertEqual(len(output["context_records"]),18)
            first=output["context_records"][0]["effects"]["Direct"]
            self.close("delta_true",first["true_difference"],-2.)
            self.close("delta_predicted",first["predicted_difference"],1.)
            self.close("delta_raw_error",first["error"],3.)
            for k,effect in enumerate(EFFECT_ORDER):
                row=output["effects"][effect];scale=k+1
                self.close("delta_rmse",row["pooled_rmse"],math.sqrt(18)*scale)
                self.close("delta_bias",row["signed_bias"],2*scale)
                self.close("delta_abs",row["absolute_bias"],2*scale)
                self.close("delta_acb",row["absolute_conditional_bias"],4*scale)
                self.close("delta_between_sd",row["between_mechanism_standard_deviation"],math.sqrt(21)*scale)

    def test_failed_ite_effect_keeps_raw_delta_record_and_valid_other_effects(self):
        plan,records,truths,_=node_fixture()
        records=list(records);first=records[0]
        ite=first.ite.copy();ite[:,1]=np.nan
        records[0]=replace(first,ite=ite,output_status={
            **first.output_status,"ite":("OK","FAILED","OK")},
            execution_status="FAILED",failure_reasons={"reason":"Spillover failure"})
        output=m.center_leaf_metrics(plan,records,truths)["ordinary"]["j"]
        self.assertIsNone(output["context_records"][0]["effects"]["Spillover"]["error"])
        self.assertEqual(len(output["context_records"]),18)
        self.assertEqual(output["effects"]["Spillover"]["status"],"INCOMPLETE")
        self.assertIsNone(output["effects"]["Spillover"]["pooled_rmse"])
        self.assertEqual(output["effects"]["Direct"]["status"],"COMPLETE")
        self.assertEqual(output["effects"]["Total"]["status"],"COMPLETE")


    def test_support_strata_independent_queries_counts_and_equal_mechanisms(self):
        plan,records,truths,supports=node_fixture()
        records=list(records)
        missing=records.pop(6)  # m1 repeat0 arm0
        failed_index=next(i for i,p in enumerate(records) if p.key[2]=="synthetic-context-2-1-1")
        old=records[failed_index]
        records[failed_index]=replace(old,mu=None,output_status={**old.output_status,"mu":"FAILED"},
            execution_status="FAILED",failure_reasons={"reason":"mu unavailable"})
        output=m.support_stratification(plan,records,truths,supports)["ordinary"]["j"]
        self.assertEqual(len(output["raw_queries"]),288)
        self.assertEqual(sum(r["status"]=="OK" for r in output["raw_queries"]),256)
        self.assertEqual(len(output["support_counts"]),18)
        self.assertEqual({r["reason"] for r in output["raw_queries"]},set(REASONS))
        for count in output["support_counts"]:
            self.assertEqual(sum(np.asarray(count["full_graph_count"]).flat),4)
            self.assertEqual(sum(np.asarray(count["observed_count"]).flat),2)
        coefficient=((0.,4.),(1.,3.))
        valid_keys={p.key for p in records if p.output_status["mu"]=="OK"}
        for row in output["strata"]:
            all_errors={i:[] for i in range(3)}
            requested=0
            for task,(_,support) in zip(plan.tasks,supports):
                if row["center_t"] is not None and task.center_t!=row["center_t"]: continue
                for node in range(4):
                    if row["group"]=="center" and node!=0: continue
                    if row["group"]=="leaf" and node==0: continue
                    for t in (0,1):
                        for s in (0,1):
                            if support.support_reason[node,t,s]!=row["reason"]: continue
                            requested+=1
                            if task.key in valid_keys:
                                all_errors[task.mechanism_id].append(
                                    (2. if node==0 else -1.)*FACTORS[task.mechanism_id]*coefficient[t][s])
            valid=sum(len(v) for v in all_errors.values())
            self.assertEqual(row["n_requested"],requested)
            self.assertEqual(row["n_valid"],valid)
            self.assertEqual(row["n_failed_or_missing"],requested-valid)
            active=[v for v in all_errors.values() if v]
            if active:
                expected_mse=sum(sum(x*x for x in v)/len(v) for v in active)/len(active)
                expected_bias=sum(sum(v)/len(v) for v in active)/len(active)
                self.close("support_mse",row["mse"],expected_mse)
                self.close("support_rmse",row["pooled_rmse"],math.sqrt(expected_mse))
                self.close("support_bias",row["signed_bias"],expected_bias)
            else:
                self.assertIsNone(row["mse"]);self.assertIsNone(row["signed_bias"])
        self.assertTrue(output["conditional_on_valid_predictions"])

    def test_empty_support_layer_is_na_not_zero(self):
        plan,records,truths,supports=node_fixture(repeats=(1,1,1))
        output=m.support_stratification(plan,records,truths,supports)["ordinary"]["j"]
        empty=[r for r in output["strata"] if r["group"]=="center"
               and r["reason"]=="observation_sampling_zero"]
        self.assertEqual(len(empty),3)
        for row in empty:
            self.assertEqual(row["status"],"EMPTY")
            self.assertEqual(row["n_requested"],0)
            self.assertIsNone(row["mse"]);self.assertIsNone(row["pooled_rmse"])

    def test_missing_all_predictions_preserves_support_denominator(self):
        plan,_,truths,supports=node_fixture()
        output=m.support_stratification(plan,(),truths,supports)["ordinary"]["j"]
        rows=[r for r in output["strata"] if r["group"]=="all" and r["center_t"] is None]
        self.assertEqual(sum(r["n_requested"] for r in rows),288)
        self.assertEqual(sum(r["n_valid"] for r in rows),0)
        self.assertTrue(all(r["mse"] is None for r in rows))
        self.assertTrue(all(r["status"]=="NO_VALID_PREDICTIONS" for r in rows))

    def test_duplicate_extra_identity_wrong_hash_and_truth_universe_rejected(self):
        plan,records,truths=pair_fixture()
        with self.assertRaises(ValueError): m.pair_ate_metrics(plan,records+(records[0],),truths)
        extra=replace(records[0],key=records[0].key[:3]+("extra-request",))
        with self.assertRaises(ValueError): m.pair_ate_metrics(plan,records+(extra,),truths)
        wrong=replace(records[0],input_hashes=("f"*64,"e"*64))
        with self.assertRaises(ValueError): m.pair_ate_metrics(plan,(wrong,)+records[1:],truths)
        with self.assertRaises(ValueError): m.pair_ate_metrics(plan,records,truths[:-1])
        with self.assertRaises(ValueError): m.pair_ate_metrics(plan,records,truths+(truths[0],))
        with self.assertRaises(ValueError): m.EvaluationPlan(plan.tasks+(plan.tasks[0],),synthetic=True)

    def test_wrong_shapes_nonfinite_values_and_illegal_status_rejected(self):
        plan,records,truths,_=node_fixture()
        p=records[0]
        for bad in (np.zeros((4,4)),np.zeros((4,2,2),dtype=np.float32),
                    np.full((4,2,2),np.nan),np.full((4,2,2),np.inf)):
            with self.assertRaises(ValueError): replace(p,mu=bad)
        with self.assertRaises(ValueError): replace(p,output_status={**p.output_status,"mu":"NOT_RANKABLE"})
        with self.assertRaises(ValueError): m.MechanismTruth(0,np.array([0.,np.inf,0.]))
        with self.assertRaises(ValueError): replace(p,node_ids=(1,0,2,3))
        # A changed node order is rejected on binding even for otherwise valid arrays.
        with self.assertRaises(ValueError):
            m.node_metrics(plan,(replace(p,input_hashes=("f"*64,)),)+records[1:],truths)

    def test_illegal_support_labels_counts_and_binding_rejected(self):
        plan,records,truths,supports=node_fixture()
        key,support=supports[0]
        labels=support.support_reason.copy();labels[0,0,0]="invented"
        bad=replace(support,support_reason=labels)
        with self.assertRaises(ValueError):
            m.support_stratification(plan,records,truths,((key,bad),)+supports[1:])
        bad=replace(support,source_input_hash="f"*64)
        with self.assertRaises(ValueError):
            m.support_stratification(plan,records,truths,((key,bad),)+supports[1:])
        counts=support.full_graph_count.copy();counts[0,0,0]+=1
        with self.assertRaises(ValueError):
            m.support_stratification(plan,records,truths,((key,replace(support,full_graph_count=counts)),)+supports[1:])
        with self.assertRaises(ValueError): m.support_stratification(plan,records,truths,supports[:-1])
        with self.assertRaises(ValueError): m.support_stratification(plan,records,truths,supports+(supports[0],))

    def test_pair_context_association_and_missing_expected_arm_rejected(self):
        plan,_,_,_=node_fixture()
        with self.assertRaises(ValueError): m.EvaluationPlan(plan.tasks[1:],synthetic=True)
        changed=replace(plan.tasks[1],pair_id="wrong-pair")
        with self.assertRaises(ValueError):
            m.EvaluationPlan((plan.tasks[0],changed)+plan.tasks[2:],synthetic=True)

    def test_existing_sealed_registry_and_public_bundle_projection(self):
        registry,task,request,_=registry_fixture()
        plan=m.EvaluationPlan.from_registry(registry)
        record=m.Prediction.from_bundle(bundle_fixture(task,request))
        report=m.pair_ate_metrics(plan,(record,),(m.MechanismTruth(0,np.array([1.,2.,3.])),))
        self.assertEqual(plan.source_seal,registry.seal_sha256)
        self.assertFalse(plan.synthetic)
        self.assertEqual(len(plan.tasks),4)
        self.assertEqual(report["ordinary"]["state-mean"]["Direct"]["status"],"COMPLETE")
        self.close("bundle_projection",report["ordinary"]["state-mean"]["Direct"]["pooled_rmse"],0.)
        self.assertEqual(report["ordinary"]["j"]["Direct"]["status"],"INCOMPLETE")
        self.assertEqual(registry.attempts,())
        self.assertEqual(len(registry.unexecuted_requests()),4)

    def test_unsupported_node_capability_cannot_be_bypassed(self):
        plan,records,truths=pair_fixture(("state-mean","oracle-pair"))
        p=records[0]
        forged=replace(p,mu=np.zeros((4,2,2)),output_status={**p.output_status,"mu":"OK"})
        with self.assertRaises(ValueError): m.node_metrics(plan,(forged,records[1]),truths)

    def test_all_input_hashes_and_output_immutability(self):
        plan,records,truths,supports=node_fixture()
        arrays=[]
        for p in records: arrays.extend((p.mu,p.ite,p.ate))
        for t in truths: arrays.extend((t.mu,t.ite,t.ate))
        for _,s in supports:
            arrays.extend((s.support_reason,s.full_graph_count,s.observed_count,s.empirical_support_mask))
        before=fingerprint(arrays)
        node=m.node_metrics(plan,records,truths)
        m.center_leaf_metrics(plan,records,truths)
        m.support_stratification(plan,records,truths,supports)
        self.assertEqual(fingerprint(arrays),before)
        with self.assertRaises(TypeError): node["ordinary"]["j"]["groups"]["all"]["mu"]["mse"]=0


if __name__=="__main__":
    unittest.main()
