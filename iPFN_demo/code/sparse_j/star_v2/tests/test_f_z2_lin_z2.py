"""S3.3 only; fixed asset and artificial/mocked pairs, no formal cohorts.

PREDECLARED tolerances: numerical reference comparisons 1e-9 absolute;
orthogonality/centering 1e-11. Rank 1e-6 and row residual 1e-5*||c||
are fixed-source rules, not tuned against outcomes. No old module import.
"""
import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from star_v2 import baselines_regression as b, contracts as c, pair as p
from star_v2.config import CONFIG
from test_pair_data import mechanism_fixture, readonly, scripted_rng
from test_state_mean_hajek import req

ATOL = 1e-9
ORTHO_ATOL = 1e-11
TEST_ENTROPY = (0x533333,0x5A32,99881)
EVIDENCE = {}


def symmetric_x():
    # Center exactly equals full-population X mean; enables an estimable
    # singleton state in the missing-01 fixture for interacted regression.
    random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(TEST_ENTROPY)))
    half = random.normal(size=(99,20)).astype(np.float32)
    return np.vstack([np.zeros((1,20),np.float32),half,-half,
                      np.zeros((1,20),np.float32)])


def task_fixture(missing=None, x=None):
    mechanism = replace(mechanism_fixture(), x=readonly(symmetric_x() if x is None else x))
    if missing is None:
        leaves = (np.arange(199)%3 == 0).astype(np.int64)
    else:
        leaves = np.zeros(199,dtype=np.int64)
        if missing in ("00","01"):
            leaves[:49] = 1
        if missing in ("00","10"):
            leaves[49:] = 1
    with scripted_rng(leaf=leaves,indices=np.arange(50,dtype=np.int64)):
        private = p.generate_pair(mechanism,pair_repeat_id=2)
    return c.pair_from_private(private)


def reference_features(x):
    # Reads frozen asset directly, independently of production loader/basis.
    blob = b.TRANSFORM_PATH.read_bytes()
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        raw = ((x.astype(np.float64)-data["x_mean"])/data["x_scale"])@data["projection"]
    center = raw-np.mean(raw,axis=0)
    rms = np.sqrt(np.mean(center**2,axis=0))
    active = rms>0
    normalized = center[:,active]/rms[active]
    if normalized.shape[1]:
        u,s,_ = np.linalg.svd(normalized,full_matrices=False)
        z = u[:,s>1e-6*s[0]]*np.sqrt(200)
    else:
        s,z = np.empty(0),np.empty((200,0))
    return raw,center,rms,s,z


def reference_row(z, state, method):
    intercept = [float(k==state) for k in range(4)]
    if method == "f-z2":
        return intercept+list(z)
    return intercept+[float(z[j]) if state==k else 0.
                      for k in range(4) for j in range(len(z))]


def reference_design(task,method):
    z = reference_features(task.context_refs[0].x)[-1]
    matrix,y = [],[]
    for context in task.context_refs:
        treatment = context.treatment
        exposure = int(sum(map(float,treatment[1:]))/199>.5)
        for node,outcome in zip(context.observed_indices,context.outcomes_observed):
            state = 2*int(treatment[node])+(exposure if node==0 else int(treatment[0]))
            matrix.append(reference_row(z[node],state,method))
            y.append(float(outcome))
    queries = np.array([[reference_row(z[node],state,method) for node in range(200)]
                        for state in range(4)]).mean(1)
    contrasts = np.stack([queries[2]-queries[0],queries[3]-queries[2],queries[3]-queries[0]])
    return np.asarray(matrix),np.asarray(y),contrasts,z


def reference_fit(matrix,y,contrasts,weight=8.):
    # Independent lstsq fits + transposed-system contrast reconstruction;
    # production instead explicitly projects using retained right singulars.
    fw = matrix*np.sqrt(weight)
    response = y*np.sqrt(weight)
    coefficient,_,rank,singular = np.linalg.lstsq(fw,response,rcond=1e-6)
    values,statuses,residuals = [],[],[]
    for contrast in contrasts:
        representation = np.linalg.lstsq(fw.T,contrast,rcond=1e-6)[0]
        residual = np.linalg.norm(fw.T@representation-contrast)
        valid = residual <= 1e-5*np.linalg.norm(contrast)
        residuals.append(residual)
        values.append(float(contrast@coefficient) if valid else np.nan)
        statuses.append("OK" if valid else "NOT_ESTIMABLE")
    return np.asarray(values),tuple(statuses),int(rank),singular,np.asarray(residuals)


class RegressionTests(unittest.TestCase):
    def close(self,name,actual,expected,tolerance=ATOL):
        actual,expected = np.asarray(actual),np.asarray(expected)
        finite = np.isfinite(actual)&np.isfinite(expected)
        residual = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        EVIDENCE[name] = max(EVIDENCE.get(name,0.),residual)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=tolerance,equal_nan=True)

    def test_frozen_transform_identity_and_immutable_arrays(self):
        blob = b.TRANSFORM_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(blob).hexdigest(),
            "b261372fea1cb241e4acbd9734eab5b9506f6478e90d90c0f5f3e633b7a3a737")
        transform = b.load_frozen_transform()
        self.assertEqual(transform.projection.shape,(20,2))
        self.assertEqual(transform.sha256,b.TRANSFORM_SHA256)
        for value in (transform.x_mean,transform.x_scale,transform.projection):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_hash_mismatch_rejected_before_np_load(self):
        fake = Mock()
        blob = b.TRANSFORM_PATH.read_bytes()
        fake.read_bytes.return_value = blob+b"invalid"
        with patch.object(b,"TRANSFORM_PATH",fake),patch.object(b.np,"load") as loader:
            with self.assertRaises(c.ContractError) as error:
                b.load_frozen_transform()
        loader.assert_not_called()
        self.assertEqual(error.exception.code,"ASSET_HASH")
        self.assertEqual(error.exception.evidence["expected"],b.TRANSFORM_SHA256)

    def test_full_population_features_numpy_reference(self):
        x = symmetric_x()
        result = b.full_population_features(x)
        raw,center,rms,singular,z = reference_features(x)
        self.close("feature_raw",result.raw,raw)
        self.close("feature_center",result.centered,center)
        self.close("feature_rms",result.rms,rms)
        self.close("feature_singular",result.singular_values,singular)
        self.close("feature_basis",result.z,z)
        self.close("feature_center_mean",result.z.mean(0),np.zeros(2),ORTHO_ATOL)
        self.close("feature_orthogonality",result.z.T@result.z/200,np.eye(2),ORTHO_ATOL)
        self.assertEqual(result.rank,2)
        self.assertFalse(result.z.flags.writeable)

    def test_fixed_state_design_covariate_principles_without_import(self):
        source = b.TRANSFORM_PATH.parent.parent/"pfn_social"/"phase6_sparse_classical.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(v for v in tree.body if isinstance(v,ast.FunctionDef)
                        and v.name=="state_design")
        selected = []
        started = False
        for statement in function.body:
            if isinstance(statement,ast.Assign):
                names = [t.id for t in statement.targets if isinstance(t,ast.Name)]
                if "z" in names: started = True
                if "one" in names: break
            if started: selected.append(statement)
        self.assertTrue(selected)
        raw = reference_features(symmetric_x())[0]
        namespace = {"np":np,"auxiliary":raw,"n":200,"rcond":1e-6}
        # Only the centering/RMS/SVD AST; no legacy propensity/model code.
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),"exec"),namespace)
        self.close("fixed_source_covariate_basis",
                   b.full_population_features(symmetric_x()).z,namespace["z"])

    def test_collinear_projection_direction_is_removed(self):
        half = np.linspace(-2.,2.,100,dtype=np.float32)
        x = np.tile(half[:,None],(1,20))
        x = np.concatenate([x,-x])
        feature = b.full_population_features(x)
        self.assertEqual(feature.rank,1)
        ref = reference_features(x)
        self.close("collinear_basis",feature.z,ref[-1])
        self.assertLess(feature.singular_values[1],1e-6*feature.singular_values[0])

    def test_full_n_not_observed_subset_normalization(self):
        task = task_fixture()
        x,ids = task.context_refs[0].x,task.context_refs[0].observed_indices
        feature = b.full_population_features(x)
        # Reference full-N mean is center raw value; sample mean differs.
        self.assertGreater(np.linalg.norm(feature.raw[ids].mean(0)-feature.raw.mean(0)),1e-3)
        self.assertGreater(np.linalg.norm(feature.z[ids].mean(0)),1e-2)
        self.close("all_node_centering",feature.centered.mean(0),np.zeros(2),ORTHO_ATOL)

    def test_f_design_state_order_and_joint_rows(self):
        self.check_design("f-z2")

    def test_lin_design_state_order_and_joint_rows(self):
        self.check_design("lin-z2")

    def check_design(self,method):
        task = task_fixture()
        inputs = tuple(c.predictor_inputs(t) for t in task.context_refs)
        design = b.build_regression_design(inputs,method)
        matrix,y,contrasts,z = reference_design(task,method)
        self.assertEqual(design.matrix.shape,(100,6 if method=="f-z2" else 12))
        self.close(method+"_design",design.matrix,matrix)
        self.close(method+"_response",design.outcomes,y)
        self.close(method+"_query_contrasts",design.contrasts,contrasts)
        self.close(method+"_all_node_query",design.contrasts[:,:4],
                   [[-1.,0.,1.,0.],[0.,0.,-1.,1.],[-1.,0.,0.,1.]])
        self.assertTrue(np.all(design.weights==8.))
        self.assertEqual(int(design.counts.sum()),100)
        self.assertFalse(design.matrix.flags.writeable)
        if method=="lin-z2":
            sample_mean = z[task.context_refs[0].observed_indices].mean(0)
            wrong = np.zeros_like(contrasts)
            for k,(pos,neg) in enumerate(((2,0),(3,2),(3,0))):
                wrong[k] = np.asarray(reference_row(sample_mean,pos,method))-np.asarray(
                    reference_row(sample_mean,neg,method))
            self.assertGreater(np.linalg.norm(contrasts-wrong),1e-2)

    def test_f_fit_against_independent_lstsq_and_ols(self):
        self.check_fit("f-z2",b.estimate_f_z2)

    def test_lin_fit_against_independent_lstsq_and_ols(self):
        self.check_fit("lin-z2",b.estimate_lin_z2)

    def check_fit(self,method,entry):
        task = task_fixture()
        matrix,y,contrasts,_ = reference_design(task,method)
        expected,statuses,rank,singular,residuals = reference_fit(matrix,y,contrasts)
        result = entry(req(task),task)
        self.assertEqual(result.output_status["ate"],statuses)
        self.assertEqual(statuses,("OK",)*3)
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.diagnostics["design_rank"],matrix.shape[1])
        self.assertEqual(rank,matrix.shape[1])
        self.assertIsNotNone(result.diagnostics["condition_number"])
        self.close(method+"_fit",result.ate,expected)
        self.close(method+"_singular",result.diagnostics["singular_values"],singular)
        ols = b.fit_estimable_contrasts(matrix,y,contrasts,np.ones(len(y)))
        self.close(method+"_constant_wls_ols",result.ate,ols.ate)
        independent_ols = reference_fit(matrix,y,contrasts,weight=1.)[0]
        self.close(method+"_independent_ols",result.ate,independent_ols)
        self.close(method+"_total_identity",[result.ate[2]],
                   [result.ate[0]+result.ate[1]])

    def test_rank_deficiency_all_contrasts_still_estimable(self):
        matrix = np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        y = np.array([2.,3.,5.])
        contrasts = np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        fit = b.fit_estimable_contrasts(matrix,y,contrasts,np.full(3,8.))
        self.assertEqual(fit.rank,2)
        self.assertEqual(fit.columns,3)
        self.assertEqual(fit.output_status,("OK",)*3)
        self.assertIsNone(fit.condition_number)
        self.assertIsNotNone(fit.effective_condition_number)
        self.close("rank_deficient_valid",fit.ate,[2.,3.,5.])

    def test_nonestimable_contrast_not_minimum_norm_fake_zero(self):
        matrix = np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        y = np.array([2.,3.,5.])
        contrasts = np.array([[1.,0.,0.],[0.,0.,1.],[1.,0.,1.]])
        fit = b.fit_estimable_contrasts(matrix,y,contrasts,np.ones(3))
        self.assertEqual(fit.output_status,("OK","NOT_ESTIMABLE","NOT_ESTIMABLE"))
        self.assertAlmostEqual(fit.ate[0],2.,delta=ATOL)
        self.assertTrue(np.isnan(fit.ate[1:]).all())
        self.assertGreater(fit.residuals[1],fit.tolerances[1])

    def test_numeric_rank_cutoff_and_row_space_gate_are_predeclared(self):
        self.assertEqual(b.RCOND,1e-6)
        self.assertEqual(b.ROW_SPACE_RTOL,10*1e-6)
        matrix = np.diag([1.,.5e-6])
        contrasts = np.array([[1.,.5e-5],[1.,2e-5],[1.,0.]])
        y = np.array([1.,2.])
        fit = b.fit_estimable_contrasts(matrix,y,contrasts,np.ones(2))
        self.assertEqual(fit.rank,1)
        self.assertEqual(fit.output_status,("OK","NOT_ESTIMABLE","OK"))
        full = b.fit_estimable_contrasts(np.diag([1.,2e-6]),y,contrasts,np.ones(2))
        self.assertEqual(full.rank,2)
        self.assertEqual(full.output_status,("OK",)*3)

    def test_missing_01_alone_does_not_invalidate_effects(self):
        for method,entry in (("f-z2",b.estimate_f_z2),("lin-z2",b.estimate_lin_z2)):
            with self.subTest(method=method):
                task = task_fixture("01")
                result = entry(req(task),task)
                self.assertEqual(result.diagnostics["state_counts"][1],0)
                self.assertEqual(result.output_status["ate"],("OK",)*3)
                self.assertEqual(result.execution_status,"OK")
                self.assertLess(result.diagnostics["design_rank"],result.diagnostics["design_columns"])
                reference = reference_fit(*reference_design(task,method)[:3])
                self.close(method+"_missing01",result.ate,reference[0])
                EVIDENCE[method+"_missing01_rank"] = result.diagnostics["design_rank"]

    def test_partial_effect_failures_by_required_state(self):
        for missing,statuses in (
                ("00",("NOT_ESTIMABLE","OK","NOT_ESTIMABLE")),
                ("10",("NOT_ESTIMABLE","NOT_ESTIMABLE","OK")),
                ("11",("OK","NOT_ESTIMABLE","NOT_ESTIMABLE"))):
            for method,entry in (("f-z2",b.estimate_f_z2),("lin-z2",b.estimate_lin_z2)):
                with self.subTest(missing=missing,method=method):
                    task = task_fixture(missing)
                    result = entry(req(task),task)
                    self.assertEqual(result.output_status["ate"],statuses)
                    self.assertEqual(result.execution_status,"NOT_ESTIMABLE")
                    self.assertEqual(result.failure_type,"CONTRAST_NOT_ESTIMABLE")
                    matrix,y,contrasts,_ = reference_design(task,method)
                    expected = reference_fit(matrix,y,contrasts)
                    self.assertEqual(result.output_status["ate"],expected[1])
                    self.close(method+"_missing"+missing,result.ate,expected[0])
                    for k,state in enumerate(statuses):
                        effect = b.EFFECT_ORDER[k]
                        if state!="OK":
                            self.assertTrue(np.isnan(result.ate[k]))
                            self.assertIn(effect,result.failure_provenance["effects"])
                            self.assertGreater(result.estimability[effect]["row_space_residual"],
                                               result.estimability[effect]["row_space_tolerance"])
                        else:
                            self.assertTrue(np.isfinite(result.ate[k]))

    def test_nonidentified_never_reads_asset_or_fits(self):
        task = task_fixture()
        request = req(task,status="NOT_IDENTIFIED",failure_type="UPSTREAM",
                      failure_message="keep original reason",failure_provenance={"code":"ID17"})
        for entry in (b.estimate_f_z2,b.estimate_lin_z2):
            with patch.object(b,"load_frozen_transform",side_effect=AssertionError("no load")) as loader,\
                 patch.object(b,"fit_estimable_contrasts",side_effect=AssertionError("no fit")) as fitter:
                result = entry(request,task)
            loader.assert_not_called()
            fitter.assert_not_called()
            self.assertEqual((result.status,result.execution_status),("NOT_IDENTIFIED",)*2)
            self.assertEqual(result.failure_message,request.failure_message)
            self.assertEqual(dict(result.failure_provenance),dict(request.failure_provenance))
            self.assertTrue(np.isnan(result.ate).all())

    def test_five_array_isolation_and_privileged_injection_rejected(self):
        task = task_fixture()
        inputs = tuple(dict(c.predictor_inputs(t)) for t in task.context_refs)
        with patch.object(b,"build_regression_design",wraps=b.build_regression_design) as build:
            result = b.estimate_f_z2(req(task),task)
        for payload in build.call_args.args[0]:
            self.assertEqual(tuple(payload),c.FACTUAL_KEYS)
            for value in payload.values():
                self.assertFalse(value.flags.writeable)
                self.assertFalse(np.shares_memory(value,task.context_refs[0].outcomes_observed))
        self.assertEqual(result.execution_status,"OK")
        for key in ("truth","beta","h","full_y","noise","rng_identity","input_hash","support"):
            changed = dict(inputs[0]);changed[key] = np.zeros(200)
            with self.subTest(key=key),self.assertRaises(c.ContractError):
                b.build_regression_design((changed,inputs[1]),"f-z2")

    def test_raw_float64_arrays_and_hashes_unchanged(self):
        task = task_fixture()
        arrays = [getattr(t,key) for t in task.context_refs for key in c.FACTUAL_KEYS]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        hashes = tuple(t.input_hash for t in task.context_refs)
        for entry in (b.estimate_f_z2,b.estimate_lin_z2):
            result = entry(req(task),task)
            self.assertEqual(result.input_hashes,hashes)
            self.assertEqual(result.protocol_hash,CONFIG.protocol_sha256)
            self.assertEqual(result.asset_hashes["feature_transform.npz"],b.TRANSFORM_SHA256)
        self.assertEqual(before,[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])
        for task_context in task.context_refs:
            c.validate_task(task_context)
            self.assertEqual(task_context.outcomes_observed.dtype,np.float64)

    def test_identity_ate_only_support_and_no_false_standard_errors(self):
        task = task_fixture()
        request = req(task,requested_outputs=("mu","ite","ate","interval"))
        for method,entry in (("f-z2",b.estimate_f_z2),("lin-z2",b.estimate_lin_z2)):
            result = entry(request,task)
            self.assertEqual(result.method_id,method)
            for name in ("status","target_id","request_id","pair_id","context_ids","input_hashes"):
                self.assertEqual(getattr(result,name),getattr(request,name))
            self.assertEqual(result.output_status["mu"],"UNSUPPORTED")
            self.assertEqual(result.output_status["ite"],("UNSUPPORTED",)*3)
            self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
            self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
            self.assertEqual(result.arm_results,(None,None))
            self.assertEqual(result.diagnostics["n_factual_records"],100)
            self.assertEqual(result.diagnostics["n_unique_observed_nodes"],50)
            self.assertEqual(result.diagnostics["observation_clusters"],50)
            self.assertFalse(result.diagnostics["standard_errors_reported"])
            self.assertFalse(result.diagnostics["arms_independent"])
            self.assertTrue(result.population_support.all())
            self.assertFalse(result.empirical_support_mask.all())
            self.assertTrue(np.isnan(result.mu).all())
            self.assertTrue(np.isnan(result.ite).all())

    def test_invalid_arrays_weights_and_shapes_rejected(self):
        matrix = np.eye(3)
        y = np.ones(3)
        contrasts = np.eye(3)
        for args in (
                (matrix.astype(np.float32),y,contrasts,np.ones(3)),
                (matrix,np.array([np.nan,1.,1.]),contrasts,np.ones(3)),
                (matrix,y,contrasts,np.array([8.,1.,8.])),
                (matrix,y,contrasts,np.zeros(3)),
                (matrix,y,contrasts[:2],np.ones(3)),
                (matrix,y.reshape(3,1),contrasts,np.ones(3))):
            with self.assertRaises(c.ContractError):
                b.fit_estimable_contrasts(*args)
        for x in (symmetric_x().astype(np.float64),np.zeros((50,20),np.float32),
                  np.full((200,20),np.nan,np.float32)):
            with self.assertRaises(c.ContractError):
                b.full_population_features(x)

    def test_identity_conflict_and_tamper_rejected_before_asset_load(self):
        task = task_fixture()
        with patch.object(b,"load_frozen_transform") as loader:
            with self.assertRaises(c.ContractError):
                b.estimate_f_z2(replace(req(task),pair_id="other"),task)
            with self.assertRaises(c.ContractError):
                b.estimate_lin_z2(req(task),task.context_refs[0])
        loader.assert_not_called()
        request = req(task)
        changed = task.context_refs[0].outcomes_observed.copy();changed[0]+=1.
        object.__setattr__(task.context_refs[0],"outcomes_observed",changed)
        with self.assertRaises(c.ContractError):
            b.estimate_lin_z2(request,task)

    def test_linalg_failure_keeps_failure_evidence_without_retry(self):
        task = task_fixture()
        with patch.object(b.np.linalg,"svd",side_effect=np.linalg.LinAlgError("fixture failure")) as svd:
            result = b.estimate_f_z2(req(task),task)
        svd.assert_called_once()
        self.assertEqual(result.status,"POINT")
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.failure_type,"LinAlgError")
        self.assertEqual(result.failure_message,"fixture failure")
        self.assertEqual(result.failure_provenance["attempts"],1)
        self.assertTrue(np.isnan(result.ate).all())


def tearDownModule():
    print("S3.3 EVIDENCE "+json.dumps(EVIDENCE,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
