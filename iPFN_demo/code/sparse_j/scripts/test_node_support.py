"""Deterministic, in-memory fixtures; no DGP or model inference."""
import os
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import unittest

import numpy as np

import analyze_node_support as a

PRIVATE_INPUT = os.environ.get('E1_5_SOURCE_RUN')
PUBLIC = Path(__file__).resolve().parents[3]/'results/e1_5_star_ate_submission_v1/node_support'


def fixture(center=0, observed_center=True):
    adjacency=np.zeros((200,200));adjacency[0,1:]=1;adjacency[1:,0]=1
    treatment=np.zeros(200);treatment[0]=center;treatment[100:]=1
    inputs=dict(x=np.zeros((200,20)),adjacency=adjacency,treatment=treatment,
                observed_indices=np.arange(0 if observed_center else 1,50 if observed_center else 51,dtype=np.int64),
                outcomes_observed=np.zeros(50))
    truth=np.tile([2.,4.,8.,16.],(200,1));truth[0]=[1.,2.,4.,8.]
    leaf_error=np.r_[np.tile([1.,-1.],99),2.]
    prediction=truth.copy();prediction[0]+=[1.,-2.,3.,-4.]
    prediction[1:]+=leaf_error[:,None]*np.arange(1,5)[None,:]
    return inputs,truth,prediction


def identity(mid=1000,rid=0,center=0):
    return dict(context_id=a.context_id(mid,rid,center),mechanism_id=mid,
                pair_repeat_id=rid,center_t=center,method='j')


class NodeSupportFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PRIVATE_INPUT:
            cls.frozen,cls.provenance=a.load_frozen(Path(PRIVATE_INPUT))
        else:
            # Public fixtures use only the four public pure-function bodies.
            # This temporary test bundle is not the original experimental snapshot.
            with tempfile.TemporaryDirectory(prefix='e1_5_public_fixture_') as tmp:
                run=Path(tmp);hashes={}
                with zipfile.ZipFile(run/'source_snapshot.zip','w') as archive:
                    for name in ('e1_5_star.py','e1_5_star_analysis.py'):
                        rel='iPFN_demo/code/sparse_j/scripts/'+name
                        raw=(Path(__file__).parent/name).read_bytes()
                        hashes[rel]=hashlib.sha256(raw).hexdigest()
                        archive.writestr(rel,raw)
                (run/'run_manifest.json').write_text(json.dumps({'source_sha256':hashes}),encoding='utf-8')
                cls.frozen,cls.provenance=a.load_frozen(run)

    def scored(self,center=0,observed_center=True,mid=1000,rid=0):
        inputs,truth,pred=fixture(center,observed_center)
        return a.score_context(self.frozen,inputs,truth,pred,identity(mid,rid,center),
                               np.ones(4,dtype=bool),np.ones(3,dtype=bool))

    def test_frozen_extraction_only_selected_functions(self):
        self.assertEqual(set(self.provenance),{'effects','score','support_reason','summarize'})
        self.assertFalse(hasattr(self.frozen,'build_world'))
        self.assertFalse(hasattr(self.frozen,'make_pair'))
        self.assertFalse(hasattr(self.frozen,'predict_record'))

    def test_hand_response_formula_and_grouping(self):
        response,*_=self.scored()
        c=next(r for r in response if r['category']=='center' and r['arm']=='00')
        l=next(r for r in response if r['category']=='leaf' and r['arm']=='00')
        self.assertEqual((c['true_mean'],c['pred_mean'],c['bias'],c['mae'],c['rmse']),(1.,2.,1.,1.,1.))
        self.assertEqual((c['n_nodes'],l['n_nodes']),(1,199))
        self.assertAlmostEqual(l['true_mean'],2.)
        self.assertAlmostEqual(l['pred_mean'],2+2/199)
        self.assertAlmostEqual(l['bias'],2/199)
        self.assertAlmostEqual(l['mae'],200/199)
        self.assertAlmostEqual(l['mse'],202/199)
        self.assertAlmostEqual(l['rmse'],np.sqrt(202/199))

    def test_effect_order_and_center_leaf_contrast(self):
        _,effects,differences,*_=self.scored()
        center={r['effect']:r for r in effects if r['category']=='center'}
        leaf={r['effect']:r for r in effects if r['category']=='leaf'}
        self.assertEqual([center[e]['true_mean'] for e in a.EFFECTS],[3.,4.,7.])
        self.assertEqual([center[e]['bias'] for e in a.EFFECTS],[2.,-7.,-5.])
        self.assertAlmostEqual(leaf['direct']['bias'],4/199)
        self.assertAlmostEqual(leaf['spillover']['bias'],2/199)
        self.assertAlmostEqual(leaf['total']['bias'],6/199)
        d=next(r for r in differences if r['effect']=='direct')
        self.assertEqual(d['true_mean'],-3.)
        self.assertAlmostEqual(d['bias'],2-4/199)
        self.assertAlmostEqual(d['mae'],abs(2-4/199))
        self.assertAlmostEqual(d['center_minus_leaf_bias'],d['bias'])
        self.assertAlmostEqual(d['center_minus_leaf_rmse'],2-2*np.sqrt(202/199))

    def test_zero_error(self):
        inputs,truth,_=fixture()
        response,effects,difference,*_=a.score_context(self.frozen,inputs,truth,truth,
            identity(),np.ones(4,dtype=bool),np.ones(3,dtype=bool))
        self.assertTrue(all(r[k]==0 for r in response+effects+difference
                            for k in ('bias','mae','mse','rmse')))

    def test_support_four_reasons_and_center_observation_separate(self):
        *_,support,_,_=self.scored()
        by={(r['category'],r['arm']):r for r in support}
        self.assertEqual(by['leaf','01']['support_reason'],self.frozen.STRUCTURAL)
        self.assertEqual(by['center','00']['support_reason'],self.frozen.ASSIGNMENT)
        self.assertEqual(by['leaf','10']['support_reason'],self.frozen.SAMPLING)
        self.assertEqual(by['leaf','00']['support_reason'],self.frozen.COVERED)
        self.assertTrue(by['leaf','10']['structurally_allowed'])
        self.assertTrue(by['leaf','10']['assignment_produced'])
        self.assertFalse(by['leaf','10']['observed_covered'])
        self.assertTrue(by['leaf','10']['model_arm_supported'])
        other=self.scored(observed_center=False)[3]
        row=next(r for r in other if r['category']=='center' and r['arm']=='01')
        self.assertFalse(row['center_observed'])
        self.assertEqual(row['support_reason'],self.frozen.SAMPLING)
        with self.assertRaises(ValueError):
            self.frozen.support_reason('leaf',0,1,0,1,0)

    def test_missing_duplicate_records_rejected(self):
        contexts=[{'context_id':a.context_id(1000,0,c)} for c in (0,1)]
        receipts=[dict(context_id=r['context_id'],method=m) for r in contexts for m in a.METHODS]
        a.check_task_grid(contexts,receipts,[1000],[0])
        for ctx,rr in ((contexts[:-1],receipts),(contexts,receipts[:-1]),
                       (contexts,receipts+[receipts[0]])):
            with self.assertRaises(ValueError):a.check_task_grid(ctx,rr,[1000],[0])

    def test_nonfinite_and_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):a.error_statistics(np.array([0.]),np.array([np.nan]))
        with self.assertRaises(ValueError):a.error_statistics(np.zeros(2),np.zeros(1))
        inputs,truth,pred=fixture()
        point=dict(arm_means=pred,ite_effects=self.frozen.effects(pred),
                   ate_effects=self.frozen.effects(pred).mean(0),arm_supported=np.ones(4),
                   ite_supported=np.ones(3),ate_supported=np.ones(3))
        a.validate_arrays(inputs,truth,point)
        point['arm_means']=pred[:199]
        with self.assertRaises(ValueError):a.validate_arrays(inputs,truth,point)

    def test_mechanism_equal_aggregation_and_frozen_rmse_semantics(self):
        rows=[]
        for mid,errors in ((1000,[1.,3.]),(1001,[2.,6.])):
            for rid,error in enumerate(errors):
                for c in (0,1):
                    for cat,n in (('center',1),('leaf',199)):
                        for arm in a.ARMS:
                            value=error if cat=='center' else (10. if mid==1000 else 20.)
                            rows.append(dict(identity(mid,rid,c),category=cat,n_nodes=n,arm=arm,
                                             **a.error_statistics(np.zeros(n),np.full(n,value))))
        mech=a.error_aggregate(rows,'arm','mechanism',[1000,1001],[0,1],methods=('j',))
        overall=a.error_aggregate(mech,'arm','overall',[1000,1001],[0,1],methods=('j',))
        c=next(r for r in overall if r['category']=='center' and r['center_t']==0 and r['arm']=='00')
        l=next(r for r in overall if r['category']=='leaf' and r['center_t']==0 and r['arm']=='00')
        self.assertAlmostEqual(c['rmse'],3.)
        self.assertAlmostEqual(c['pooled_rmse'],np.sqrt(12.5))
        self.assertAlmostEqual(l['rmse'],15.)
        with self.assertRaises(ValueError):a.error_aggregate(rows[:-1],'arm','mechanism',[1000,1001],[0,1],methods=('j',))
        with self.assertRaises(ValueError):a.error_aggregate(mech[:-1],'arm','overall',[1000,1001],[0,1],methods=('j',))

    def test_support_aggregation_and_frozen_paired_summary(self):
        metrics=[];support=[];rich=[]
        for mid in (1000,1001):
            for rid in (0,1):
                for c in (0,1):
                    *_,ss,nm,ns=self.scored(center=c,mid=mid,rid=rid)
                    rich+=ss;metrics+=nm
                    for row in ns:
                        row=dict(row)
                        if row['category']=='leaf':row['arm_rmse']=7. if int(row['arm'][1])!=c else 2.
                        support.append(row)
        contrasts,mechanisms,secondary,summary=self.frozen.summarize(support,metrics,[1000,1001],[0,1])
        self.assertEqual(len(contrasts),16)
        self.assertEqual(summary['mean_delta'],5.)
        self.assertEqual(summary['ci95'],[5.,5.])
        self.assertEqual(summary['bootstrap_seed'],2026102502)
        self.assertEqual(summary['independent_units'],2)
        mech=a.support_aggregate(rich,'mechanism',[1000,1001],[0,1],methods=('j',))
        overall=a.support_aggregate(mech,'overall',[1000,1001],[0,1],methods=('j',))
        r=next(r for r in overall if r['center_t']==0 and r['category']=='leaf' and r['arm']=='10')
        self.assertEqual(r['observation_sampling_zero_count'],4)
        self.assertEqual(r['observed_covered_rate'],0.)
        self.assertEqual(r['model_supported_rate'],1.)
        missing_leaf=[r for i,r in enumerate(support) if i!=next(
            j for j,s in enumerate(support) if s['category']=='leaf')]
        with self.assertRaises(ValueError):self.frozen.summarize(missing_leaf,metrics,[1000,1001],[0,1])



class PublicSummaryFixtures(unittest.TestCase):
    @staticmethod
    def rows(name):
        with (PUBLIC/(name+'.csv')).open(encoding='utf-8',newline='') as handle:
            return list(csv.DictReader(handle))

    def test_public_error_macro_and_pooled(self):
        for stem,target in [('response','arm'),('effects','effect'),('effect_center_leaf','effect')]:
            mech=self.rows(stem+'_mechanism');overall=self.rows(stem+'_overall_by_center')
            for r in overall:
                rr=[v for v in mech if all(v[k]==r[k] for k in ('method','center_t','category',target))]
                self.assertEqual(len(rr),20)
                self.assertEqual({int(v['mechanism_id']) for v in rr},set(range(1000,1020)))
                self.assertTrue(all(v['contexts']=='10' for v in rr))
                self.assertEqual(r['contexts'],'200')
                for field in a.ERROR_FIELDS:
                    self.assertAlmostEqual(float(r[field]),np.mean([float(v[field]) for v in rr]))
                self.assertAlmostEqual(float(r['mean_context_rmse']),float(r['rmse']))
                self.assertAlmostEqual(float(r['pooled_rmse']),np.sqrt(float(r['mse'])))

    def test_public_support_counts_and_model_capability(self):
        mech=self.rows('support_mechanism');overall=self.rows('support_overall_by_center')
        for r in overall:
            rr=[v for v in mech if all(v[k]==r[k] for k in ('method','center_t','category','arm'))]
            self.assertEqual(len(rr),20)
            for field in r:
                if field.endswith('_count') and not field.startswith('mean_'):
                    self.assertEqual(int(r[field]),sum(int(v[field]) for v in rr))
                elif field.endswith('_rate') or field.startswith('mean_'):
                    self.assertAlmostEqual(float(r[field]),np.mean([float(v[field]) for v in rr]))
            self.assertEqual(sum(int(r[s+'_count']) for s in a.REASONS),200)
            self.assertEqual(float(r['model_supported_rate']),1.)
        for r in self.rows('support_reason_counts_by_center'):
            rr=[v for v in overall if all(v[k]==r[k] for k in ('method','center_t','category'))]
            self.assertEqual(int(r['count']),sum(int(v[r['support_reason']+'_count']) for v in rr))
            self.assertAlmostEqual(float(r['mechanism_equal_rate']),int(r['count'])/800)

    def test_public_paired_mechanism_bootstrap(self):
        rows=self.rows('response_mechanism');primary=self.rows('support_primary_by_mechanism')
        summaries=self.rows('support_primary_summary')
        indexed={(r['method'],int(r['mechanism_id']),int(r['center_t']),r['arm']):float(r['rmse'])
                 for r in rows if r['category']=='leaf'}
        for method in a.METHODS:
            values=[]
            for mid in range(1000,1020):
                value=np.mean([indexed[method,mid,1-int(arm[1]),arm]-indexed[method,mid,int(arm[1]),arm]
                               for arm in a.ARMS])
                recorded=next(r for r in primary if r['method']==method and int(r['mechanism_id'])==mid)
                self.assertAlmostEqual(value,float(recorded['primary_delta']))
                self.assertEqual(recorded['repeats'],'10');values.append(value)
            rng=np.random.default_rng(2026102502)
            draws=rng.choice(np.array(values),size=(10000,20),replace=True).mean(axis=1)
            summary=next(r for r in summaries if r['method']==method)
            self.assertEqual(summary['independent_units'],'20')
            self.assertEqual(summary['bootstrap_unit'],'whole mechanism')
            self.assertAlmostEqual(float(summary['mean_delta']),np.mean(values))
            for field,wanted in zip(('ci95_lower','ci95_upper'),np.quantile(draws,[.025,.975])):
                self.assertAlmostEqual(float(summary[field]),wanted)

    def test_public_frozen_effect_mapping_and_combined_response(self):
        native=self.rows('frozen_metrics_mechanism');effects=self.rows('effects_mechanism')
        self.assertEqual(len(native),240)
        for row in effects:
            n=next(v for v in native if all(v[k]==row[k] for k in ('method','mechanism_id','center_t','category')))
            for field,suffix in [('bias','bias'),('rmse','rmse'),('true_mean','ate_true'),('pred_mean','ate_pred')]:
                self.assertAlmostEqual(float(row[field]),float(n[row['effect']+'_'+suffix]))
        expected={(0,'j'):(.468063,.602225),(0,'kernel-old'):(.467046,.638793),
                  (0,'kernel-new'):(.488948,.643881),(1,'j'):(.431337,.524968),
                  (1,'kernel-old'):(.385468,.511814),(1,'kernel-new'):(.395020,.508793)}
        for (center,method),values in expected.items():
            for cat,value in zip(('center','leaf'),values):
                rows=[r for r in native if r['method']==method and int(r['center_t'])==center and r['category']==cat]
                self.assertEqual(len(rows),20)
                self.assertAlmostEqual(np.mean([float(r['arm_rmse']) for r in rows]),value,places=6)

if __name__=='__main__':
    unittest.main()
