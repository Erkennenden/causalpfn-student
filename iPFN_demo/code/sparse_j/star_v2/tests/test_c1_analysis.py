"""C.1 final Synthetic end-to-end acceptance. ATOL=1e-12, RTOL=0."""
from pathlib import Path
from unittest.mock import patch
import unittest,copy,csv,json,math,shutil,tempfile
import numpy as np
from star_v2 import analysis_loader as loader,run_analysis as pipeline,analysis_acceptance as accept
from star_v2 import analysis_tables as tables,analysis_schema as wire,bootstrap
from c1_analysis_fixture import make_synthetic,validation_patches,load,MANUAL_INDICES

ATOL=1e-12
class C1AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner=unittest.TestCase()
        try:
            cls.f=make_synthetic(cls.owner)
            cls.loaded=load(cls.f)
            cls.output=cls.f.base/"analysis"
            cls.receipt=pipeline._write_loaded(cls.loaded,cls.output,synthetic_indices=MANUAL_INDICES)
        except BaseException:
            cls.owner.doCleanups()
            raise
    @classmethod
    def tearDownClass(cls):cls.owner.doCleanups()

    def test_full_persisted_recovery_and_support_linkage(self):
        x=self.loaded
        self.assertEqual((len(x.registry.tasks),len(x.registry.requests),len(x.registry.attempts)),(18,102,102))
        self.assertEqual(len(x.truths),2);self.assertEqual(len(x.context_supports),12)
        self.assertEqual(len(x.predictions),102)
        for key,bundle in x.bundles.items():
            if key[1]=="context":
                self.assertEqual(bundle.request.identity["cohort_code"],2)
                self.assertEqual(bundle.input_hash,bundle.support.source_input_hash)
            elif key[0] in ("j","kernel-old","kernel-new","oracle-context"):
                self.assertEqual(len(bundle.arm_results),2)
                self.assertTrue(all(a.request.parent_request_id==bundle.request_id for a in bundle.arm_results))
        for s in x.context_supports.values():
            np.testing.assert_array_equal(s.empirical_extrapolation,~s.empirical_support_mask)
            np.testing.assert_array_equal(s.context_design_extrapolation,~s.design_support)

    def test_independent_analysis_acceptance_all_outputs(self):
        result=accept._validate_loaded(self.loaded,self.output,
            manifest_sha256=self.receipt["manifest_sha256"],report_sha256=self.receipt["report_sha256"],
            synthetic_indices=MANUAL_INDICES)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual((result["comparisons"],result["effect_comparisons"],result["j_coverage_distributions"]),(21,63,9))
        self.assertTrue(result["oracle_separate"]);self.assertFalse(result["hajek_ranked"])
        self.assertTrue(all((self.output/"figures"/(n+".svg")).stat().st_size>100 for n in
            ("pair_ate","comparisons","j_coverage","node_error","support_strata","oracle_context","oracle_pair")))

    def test_manual_bootstrap_rmse_difference_and_shared_indices(self):
        arrays,_=__import__("star_v2.data_artifacts",fromlist=["load_arrays"]).load_arrays(
            self.output,tuple({k:v[k] for k in ("path","bytes","sha256")} for v in
                wire.read_json(self.output,"analysis_manifest.json")["files"] if v["path"] in
                ("bootstrap/distributions.npz","bootstrap/distributions.json"))[::-1])
        # Select explicitly by path (not file ordering) below.
        self.assertEqual(arrays["indices"].dtype,np.dtype("int64"))
        np.testing.assert_array_equal(arrays["indices"],MANUAL_INDICES)
        meta=wire.read_json(self.output,"bootstrap/metadata.json")
        j=next(i for i,c in enumerate(meta["comparisons"]) if (c["method_a"],c["method_b"])==("ht","f-z2"))
        reference=[]
        for draw in MANUAL_INDICES:
            mse=[]
            for method in (1,2):
                per_m=[sum((method+1+int(m)/4+r/8)**2 for r in range(3))/3 for m in draw]
                mse.append(math.sqrt(sum(per_m)/2))
            reference.append(mse[0]-mse[1])
        for k in range(3):
            np.testing.assert_allclose(arrays["paired_rmse_differences"][:,j,k],
                np.array(reference)*(k+1),atol=ATOL,rtol=0)

    def test_j_failure_denominator_oracle_and_rankability(self):
        cov=wire.read_json(self.output,"evaluation/j_coverage.json")
        self.assertEqual(cov["effects"]["Direct"]["all_requested"],12)
        self.assertEqual(cov["effects"]["Spillover"]["failed_or_missing"],1)
        self.assertEqual(cov["effects"]["Spillover"]["coverage_including_failures"],11/12)
        self.assertEqual(cov["center_t"]["1"]["Spillover"]["coverage_including_failures"],5/6)
        rank=wire.read_json(self.output,"evaluation/rankability.json")
        self.assertEqual(set(rank["methods"]),{"state-mean","ht","f-z2","lin-z2","kernel-old","kernel-new","j"})
        self.assertEqual(rank["methods"]["state-mean"]["effects"]["Direct"]["rankability_status"],"NOT_RANKABLE")
        self.assertEqual(rank["methods"]["state-mean"]["effects"]["Spillover"]["rankability_status"],"RANKABLE")
        for key,bundle in self.loaded.bundles.items():
            if key[0]=="oracle-pair":
                self.assertTrue(np.isnan(bundle.mu).all())
                self.assertEqual(bundle.output_status["mu"],"NOT_REQUESTED")
                self.assertNotIn("pair.mu",bundle.method_capability.outputs)
                self.assertTrue(all(i.source=="joint_model_posterior" and i.context_id is None for i in bundle.intervals))

    def test_center_leaf_independent_scalar_reference(self):
        report=wire.read_json(self.output,"evaluation/center_leaf_metrics.json")
        for method in ("kernel-old","kernel-new","j"):
            for k,effect in enumerate(("Direct","Spillover","Total")):
                row=report["ordinary"][method]["effects"][effect]
                self.assertAlmostEqual(row["pooled_rmse"],.5*(k+1),delta=ATOL)
                self.assertAlmostEqual(row["signed_bias"],.5*(k+1),delta=ATOL)
                self.assertAlmostEqual(row["absolute_conditional_bias"],.5*(k+1),delta=ATOL)
                self.assertAlmostEqual(row["between_mechanism_standard_deviation"],0.,delta=ATOL)

    def test_fixed_bootstrap_seed_independent_pcg64(self):
        draws=bootstrap.generate_bootstrap_indices()
        gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261030,2026103001,2,0,7,0])))
        self.assertEqual(draws.indices.shape,(10000,200))
        for row in draws.indices:np.testing.assert_array_equal(row,gen.integers(0,200,200,dtype=np.int64))

    def test_missing_corrupt_incomplete_and_identity_rejected(self):
        from star_v2 import data_artifacts as data
        import dataclasses
        for mode in ("missing","corrupt","incomplete","pilot","duplicate","identity"):
            with self.subTest(mode=mode):
                root=self.f.base/("bad_"+mode);shutil.copytree(self.f.out,root)
                if mode in ("missing","corrupt"):
                    p=next((root/"predictions").rglob("*.npz"))
                    if mode=="missing":p.unlink()
                    else:p.write_bytes(b"damaged")
                elif mode=="incomplete":(root/"RUN_INCOMPLETE.json").write_text('{"status":"INCOMPLETE"}')
                elif mode=="pilot":
                    p=root/"run_manifest.json";doc=json.loads(p.read_text());doc["run_kind"]="pilot";p.write_bytes(data.json_bytes(doc))
                elif mode=="duplicate":
                    p=root/"ledger_final/attempts.jsonl";p.write_bytes(p.read_bytes()+p.read_bytes().splitlines(keepends=True)[0])
                else:
                    p=root/"identification_run_link.json";doc=json.loads(p.read_text());doc["run_id"]="wrong";p.write_bytes(data.json_bytes(doc))
                with validation_patches(self.f),self.assertRaises((ValueError,FileNotFoundError)):
                    loader.load_accepted_test(root,**self.f.anchors)

    def test_unaccepted_and_synthetic_as_formal_rejected(self):
        anchors=dict(self.f.anchors);anchors["acceptance_sha256"]="0"*64
        with self.assertRaises(ValueError):loader.load_accepted_test(self.f.out,**anchors)
        with self.assertRaisesRegex(ValueError,"synthetic"):
            accept._validate_loaded(self.loaded,self.output,
                manifest_sha256=self.receipt["manifest_sha256"],report_sha256=self.receipt["report_sha256"])

    def test_deterministic_repeat(self):
        out=self.f.base/"analysis_repeat"
        receipt=pipeline._write_loaded(self.loaded,out,synthetic_indices=MANUAL_INDICES)
        before={p.relative_to(self.output).as_posix():p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        after={p.relative_to(out).as_posix():p.read_bytes() for p in out.rglob("*") if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(receipt["manifest_sha256"],self.receipt["manifest_sha256"])

if __name__=="__main__":unittest.main()
