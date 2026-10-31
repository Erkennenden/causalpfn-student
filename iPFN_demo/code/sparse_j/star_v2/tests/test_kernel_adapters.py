"""Kernel-old injection tests only; actual assets/models remain S4.5.

PREDECLARED atol=1e-12, rtol=0. The simulated query cache is not evidence
that the real FrozenMixture/checkpoints pass caching or numerical checks.
"""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from star_v2 import contracts as c, kernel_adapters as k
from star_v2 import model_adapters as common
from star_v2.support import require_plugin_support, PluginSupportError
from test_model_adapters import setup_pair
from test_pair_data import readonly
from test_contracts import base_bytes

ATOL = 1e-12
ERRORS = {}


def native_dict(means):
    # Toy producer only. Expected public ATEs in tests use separate scalar
    # formulas, never this packer or any production estimator.
    ite = np.array([[row[2]-row[0],row[3]-row[2],row[3]-row[0]] for row in means])
    with np.errstate(invalid="ignore"):
        ate = ite.mean(0)
    return {
        "arm_means":means,"ite_effects":ite,"ate_effects":ate,
        "arm_supported":np.isfinite(means).all(0),
        "ite_supported":np.isfinite(ite).all(0),
        "ate_supported":np.isfinite(ate),
        "diagnostics":{"observed_rows":50,"query_units":200,
            "joint_weights":[.125]*8,
            "likelihood":"Normalized M-dimensional Gaussian density; no division of log likelihood by M or N"}}


class ToyFrozenMixture:
    """Explicit query-only cache; fresh T/indices/Y term on EVERY call."""
    def __init__(self, *, dynamic=False, corrupt=None, mean_change=None):
        self.identity = k.FROZEN_KERNEL_OLD
        self.caches = [{},{}]
        self.calls = self.rebuilds = 0
        self.dynamic,self.corrupt,self.mean_change = dynamic,corrupt,mean_change
        self.hidden_stale = False
        self.cached_dynamic = None

    def predict(self,**payload):
        self.calls += 1
        # Mirror the original FrozenMixture.validate() conversion, not the adapter.
        normalized_x = np.asarray(payload["x"], dtype=float)
        normalized_adjacency = np.asarray(payload["adjacency"], dtype=float)
        key = hashlib.sha256(normalized_x.tobytes()+normalized_adjacency.tobytes()).digest()
        if self.caches[0].get("key") != key:
            self.rebuilds += 1
            geometry = float(normalized_x[0,0])+float(normalized_adjacency.sum())/1024
            for cache in self.caches:
                cache.clear()
                cache.update(key=key,query_mean=np.full((200,4),geometry),
                             query_covariance=np.ones((200,4,1)))
        geometry = self.caches[0]["query_mean"][0,0] if self.dynamic else 0.
        if self.dynamic:
            dynamic = (float(payload["treatment"][0])/4+float(payload["treatment"][1])/8
                       +int(payload["observed_indices"][0])/1024
                       +float(np.rint(payload["outcomes_observed"][0]*8))/1024)
            if self.hidden_stale:
                if self.cached_dynamic is None:
                    self.cached_dynamic = dynamic
                dynamic = self.cached_dynamic
            i = np.arange(200,dtype=np.float64)
            means = np.column_stack([1+i/8,3+i/2+3*geometry-2*dynamic,
                8+3*i/8+5*geometry+4*dynamic,17+7*i/8+9*geometry+12*dynamic])
        else:
            arm = int(payload["treatment"][0])
            i = np.arange(200,dtype=np.float64)
            means = np.column_stack([1+i/8+2*arm,3+i/2-7*arm,
                                     8+3*i/8+5*arm,17+7*i/8+11*arm])
        if self.mean_change:
            self.mean_change(means)
        result = native_dict(means)
        if self.corrupt:
            self.corrupt(result,self,payload)
        return result

    def fit(self,*args,**kwargs):
        raise AssertionError("refit forbidden")
    def calibrate(self,*args,**kwargs):
        raise AssertionError("calibration forbidden")


def injected(model=None, **kw):
    model = ToyFrozenMixture(**kw) if model is None else model
    loader = Mock(return_value=model)
    reader = Mock(side_effect=lambda value:value.identity)
    return k.KernelOldAdapter(loader,identity_reader=reader),model,loader,reader


def scalar_dynamic_reference(payload):
    # Independent formulas for the three full-N effect means in the toy model.
    graph_term = (float(payload["x"][0,0])
                  +sum(float(v) for row in payload["adjacency"] for v in row)/1024)
    dynamic_term = (float(payload["treatment"][0])/4+float(payload["treatment"][1])/8
                    +int(payload["observed_indices"][0])/1024
                    +round(float(payload["outcomes_observed"][0])*8)/1024)
    return np.array([
        sum(7+node/4+5*graph_term+4*dynamic_term for node in range(200))/200,
        sum(9+node/2+4*graph_term+8*dynamic_term for node in range(200))/200,
        sum(16+3*node/4+9*graph_term+12*dynamic_term for node in range(200))/200])


def changed_inputs(payload,field,value):
    values = dict(payload)
    values[field] = readonly(value)
    c.validate_predictor_inputs(values)
    return values


class KernelAdapterTests(unittest.TestCase):
    def close(self,name,actual,expected):
        actual,expected = np.asarray(actual),np.asarray(expected)
        finite = np.isfinite(actual)&np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        ERRORS[name] = max(ERRORS.get(name,0.),error)
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=0,equal_nan=True)


    def test_legacy_cache_key_matches_float64_native_normalization_for_two_contexts(self):
        adapter,model,loader,_ = injected(dynamic=True)
        task,_,children = setup_pair()
        source_arrays = [getattr(context,name) for context in task.context_refs
                         for name in c.FACTUAL_KEYS]
        before = [(id(a),a.dtype.str,a.tobytes(),hashlib.sha256(a.tobytes()).hexdigest())
                  for a in source_arrays]
        task_hashes = tuple(context.input_hash for context in task.context_refs)
        for arm,context in enumerate(task.context_refs):
            payload = c.predictor_inputs(context)
            self.assertEqual(payload["x"].dtype,np.dtype("float32"))
            # Independent reference: exact documented native dtype=float conversion.
            native_x = np.asarray(payload["x"],dtype=float)
            native_a = np.asarray(payload["adjacency"],dtype=float)
            expected = hashlib.sha256(native_x.tobytes()+native_a.tobytes()).digest()
            raw_key = hashlib.sha256(payload["x"].tobytes()+payload["adjacency"].tobytes()).digest()
            self.assertNotEqual(expected,raw_key)
            self.assertEqual(k._legacy_cache_key(payload["x"],payload["adjacency"]),expected)
            result = adapter.estimate_context(context,children[arm])
            self.assertEqual(result.execution_status,"OK")
            self.assertEqual(result.output_status["ate"],("OK",)*3)
            self.assertTrue(all(cache["key"]==expected for cache in model.caches))
        self.assertEqual((model.calls,model.rebuilds,loader.call_count),(2,1,1))
        self.assertEqual(before,[(id(a),a.dtype.str,a.tobytes(),hashlib.sha256(a.tobytes()).hexdigest())
                                 for a in source_arrays])
        self.assertEqual(task_hashes,tuple(context.input_hash for context in task.context_refs))

    def test_normalized_legacy_key_excludes_dynamic_inputs_and_rebuilds_changed_x(self):
        adapter,model,_,_ = injected(dynamic=True)
        task,_,_ = setup_pair()
        base = dict(c.predictor_inputs(task.context_refs[0]))
        expected = hashlib.sha256(np.asarray(base["x"],dtype=float).tobytes()+
                                  np.asarray(base["adjacency"],dtype=float).tobytes()).digest()
        treatment = base["treatment"].copy();treatment[0] = 1
        indices = base["observed_indices"].copy();indices[0] = 199
        outcomes = base["outcomes_observed"].copy();outcomes[0] += 8
        payloads = [base,changed_inputs(base,"treatment",treatment),
                    changed_inputs(base,"observed_indices",indices),
                    changed_inputs(base,"outcomes_observed",outcomes)]
        results = []
        for payload in payloads:
            results.append(adapter(**payload).arm_means)
            self.assertTrue(all(cache["key"]==expected for cache in model.caches))
        self.assertEqual(model.rebuilds,1)
        self.assertTrue(all(not np.array_equal(results[0],value) for value in results[1:]))
        changed_x = base["x"].copy();changed_x[0,0] += 1
        adapter(**changed_inputs(base,"x",changed_x))
        self.assertEqual(model.rebuilds,2)
        self.assertNotEqual(model.caches[0]["key"],expected)

    def test_legacy_fix_still_rejects_raw_dtype_key_and_noncanonical_graph(self):
        def wrong_key(raw,model,payload):
            model.caches[0]["key"] = hashlib.sha256(
                payload["x"].tobytes()+payload["adjacency"].tobytes()).digest()
        adapter,model,_,_ = injected(corrupt=wrong_key)
        task,_,children = setup_pair()
        result = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(result.execution_status,"FAILED")
        self.assertIn("Query cache key",result.failure_message)
        payload = dict(c.predictor_inputs(task.context_refs[0]))
        adjacency = payload["adjacency"].copy();adjacency[1,2]=adjacency[2,1]=1
        self.assertNotEqual(k.query_cache_identity(payload["x"],payload["adjacency"]),
                            k.query_cache_identity(payload["x"],adjacency))
        calls = model.calls
        with self.assertRaises(c.ContractError):
            adapter(**{**payload,"adjacency":readonly(adjacency)})
        self.assertEqual(model.calls,calls)

    def test_lazy_constructor_and_identification_zero_loader_predict_reader(self):
        adapter,model,loader,reader = injected()
        loader.assert_not_called();reader.assert_not_called()
        task,request,children = setup_pair(status="NOT_IDENTIFIED",
            failure_type="UPSTREAM",failure_message="unchanged upstream reason")
        result = adapter.estimate_pair(task,request,context_requests=children)
        loader.assert_not_called();reader.assert_not_called()
        self.assertEqual(model.calls,0)
        self.assertEqual((result.status,result.execution_status),("NOT_IDENTIFIED",)*2)
        self.assertEqual(result.failure_message,"unchanged upstream reason")
        self.assertEqual(result.asset_hashes,dict(k.FROZEN_KERNEL_OLD.asset_hashes))
        for arm in result.arm_results:
            self.assertEqual(arm.asset_hashes,dict(k.FROZEN_KERNEL_OLD.asset_hashes))
            self.assertTrue(arm.diagnostics["kernel_old_identity"]["identification_blocked"])
        self.assertFalse(result.diagnostics["kernel_old_identity"]["runtime_asset_bytes_verified"])

    def test_hand_reference_context_axis_effects_and_pair_average(self):
        adapter,model,loader,reader = injected()
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.close("pair_hand",result.ate,[33.375,61.75,95.125])
        for arm_id,arm in enumerate(result.arm_results):
            expected = np.array([[7+node/4+3*arm_id,9+node/2+6*arm_id,
                                  16+3*node/4+9*arm_id] for node in range(200)])
            self.close("arm"+str(arm_id)+"_ite",arm.ite,expected)
            self.close("arm"+str(arm_id)+"_ate",arm.ate,
                       [sum(row[k] for row in expected)/200 for k in range(3)])
            self.close("non_symmetric_01",arm.mu[:,0,1],
                       [3+node/2-7*arm_id for node in range(200)])
            self.close("identity_total",arm.ate[2],arm.ate[0]+arm.ate[1])
        self.assertEqual(result.arm_order,("00","01","10","11"))
        self.assertEqual(loader.call_count,1)
        self.assertEqual(model.calls,2)
        self.assertEqual(model.rebuilds,1)

    def test_asset_chain_legal_on_all_bundles_without_claiming_runtime_verification(self):
        adapter,_,_,_ = injected()
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        for bundle in (result,*result.arm_results):
            self.assertEqual(bundle.asset_hashes,dict(k.FROZEN_KERNEL_OLD.asset_hashes))
            self.assertEqual(bundle.diagnostics["kernel_old_identity"]["asset_identity_source"],
                             "protocol_expected_not_runtime_verified")
            self.assertFalse(bundle.diagnostics["kernel_old_identity"]["runtime_asset_bytes_verified"])
            self.assertEqual(bundle.input_hashes,
                             request.input_hashes if bundle is result else bundle.request.input_hashes)
        self.assertEqual(result.target_id,request.target_id)
        self.assertEqual(result.request_id,request.request_id)
        self.assertEqual(result.context_ids,task.context_ids)

    def test_strict_five_arrays_and_original_input_protection(self):
        task,request,children = setup_pair()
        original = [getattr(t,key) for t in task.context_refs for key in c.FACTUAL_KEYS]
        before = [hashlib.sha256(a.tobytes()).hexdigest() for a in original]
        model = ToyFrozenMixture()
        original_predict = model.predict
        def observed(**payload):
            self.assertEqual(tuple(payload),c.FACTUAL_KEYS)
            self.assertEqual(payload["outcomes_observed"].dtype,np.float64)
            for array in payload.values():
                self.assertFalse(array.flags.writeable)
                self.assertIs(type(base_bytes(array)),bytes)
                self.assertEqual(len(base_bytes(array)),array.nbytes)
                for source in original:
                    self.assertFalse(np.shares_memory(array,source))
            return original_predict(**payload)
        model.predict = observed
        adapter,_,_,_ = injected(model)
        adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(before,[hashlib.sha256(a.tobytes()).hexdigest() for a in original])
        for context in task.context_refs:
            self.assertEqual(context.outcomes_observed.dtype,np.float64)
            c.validate_task(context)

    def test_extra_privileged_or_writable_inputs_rejected_before_loader(self):
        adapter,model,loader,reader = injected()
        task,_,_ = setup_pair()
        payload = dict(c.predictor_inputs(task.context_refs[0]))
        for field in ("truth","h","beta","full_y","noise","support","input_hash"):
            with self.assertRaises(c.ContractError):
                adapter(**{**payload,field:np.zeros(200)})
        with self.assertRaises(c.ContractError):
            adapter(**{**payload,"outcomes_observed":payload["outcomes_observed"].copy()})
        loader.assert_not_called();reader.assert_not_called()
        self.assertEqual(model.calls,0)

    def test_frozen_identity_exact_components_weights_grid_and_no_prior_noise_substitution(self):
        identity = k.FROZEN_KERNEL_OLD
        self.assertEqual(identity.components,(("models/kernel_subspace_prior.npz",1.),
                                              ("models/kernel_linear_prior.npz",.3)))
        self.assertEqual(identity.noise_variances,
            (.24760918477397018,.9904367390958807,3.961746956383523,15.846987825534091))
        conflicts = (
            replace(identity,components=identity.components[::-1]),
            replace(identity,components=(identity.components[0],("models/kernel_linear_prior.npz",1.))),
            replace(identity,representation_weights=(.5,.5)),
            replace(identity,noise_weights=(.25,)*4),
            replace(identity,noise_variances=(.25,1.,4.,16.)),
            replace(identity,arm_order=("00","10","01","11")),
            replace(identity,asset_hashes=((identity.asset_hashes[0][0],"0"*64),)+identity.asset_hashes[1:]))
        for conflict in conflicts:
            with self.assertRaises(c.ContractError):
                k.validate_kernel_old_identity(conflict)
        self.assertIs(k.validate_kernel_old_identity(identity),identity)

    def test_loader_identity_conflict_is_failed_once_no_retry(self):
        model = ToyFrozenMixture()
        model.identity = replace(model.identity,noise_variances=(.25,1.,4.,16.))
        adapter,_,loader,_ = injected(model)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(loader.call_count,1)
        self.assertEqual(model.calls,0)
        self.assertEqual(result.execution_status,"FAILED")
        self.assertIn("KERNEL_IDENTITY",result.arm_results[0].failure_message)
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertEqual(result.asset_hashes,dict(k.FROZEN_KERNEL_OLD.asset_hashes))

    def test_runtime_configuration_change_rejected_before_predict(self):
        adapter,model,_,_ = injected()
        task,request,children = setup_pair()
        adapter.estimate_context(task.context_refs[0],children[0])
        model.identity = replace(model.identity,representation_weights=(.5,.5))
        result = adapter.estimate_context(task.context_refs[1],children[1])
        self.assertEqual(model.calls,1)
        self.assertEqual(result.execution_status,"FAILED")

    def test_native_missing_extra_keys_shapes_and_dtypes_rejected(self):
        corruptions = (
            lambda r:r.pop("ite_effects"),
            lambda r:r.update(beta=np.zeros(25)),
            lambda r:r.update(arm_means=np.zeros((200,2,2))),
            lambda r:r.update(arm_means=r["arm_means"].astype(np.float32)),
            lambda r:r.update(ite_effects=np.zeros((200,2))),
            lambda r:r.update(ate_effects=np.zeros(4)),
            lambda r:r.update(arm_supported=np.ones(4,dtype=int)))
        task,request,children = setup_pair()
        for mutation in corruptions:
            adapter,_,_,_ = injected(corrupt=lambda raw,model,payload:mutation(raw))
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertTrue(np.isnan(result.ate).all())

    def test_native_ite_ate_masks_are_crosschecks_not_authoritative_values(self):
        task,request,children = setup_pair()
        corruptions = (
            lambda raw:raw["ite_effects"].__setitem__((0,0),999.),
            lambda raw:raw["ate_effects"].__setitem__(0,999.),
            lambda raw:raw["arm_supported"].__setitem__(0,False),
            lambda raw:raw["ite_supported"].__setitem__(1,False),
            lambda raw:raw["ate_supported"].__setitem__(2,False))
        for mutation in corruptions:
            adapter,_,_,_ = injected(corrupt=lambda raw,model,payload:mutation(raw))
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertIn("KERNEL_CONSISTENCY",result.failure_message)

    def test_native_diagnostic_inconsistency_rejected(self):
        mutations = (
            {"observed_rows":100},{"query_units":50},
            {"joint_weights":[.125]*7},{"joint_weights":[1.]*8},
            {"joint_weights":[np.nan]+[.125]*7},
            {"likelihood":"wrong normalizer"},{"truth":[0.]})
        task,request,children = setup_pair()
        for changes in mutations:
            adapter,_,_,_ = injected(corrupt=lambda raw,model,payload:raw["diagnostics"].update(changes))
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertIn("KERNEL_DIAGNOSTICS",result.failure_message)

    def test_unused_01_nonfinite_preserves_three_valid_pair_effects(self):
        def change(means):
            means[150,1] = np.nan
        adapter,_,_,_ = injected(mean_change=change)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.close("missing01_hand",result.ate,[33.375,61.75,95.125])
        self.assertEqual(result.arm_results[0].output_status["mu"],"FAILED")
        self.assertEqual(result.arm_results[0].diagnostics["point"]["native_cross_checks"],"passed")

    def test_nonfinite_10_preserves_total_only(self):
        def change(means):
            means[99,2] = np.nan
        adapter,_,_,_ = injected(mean_change=change)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.output_status["ate"],("FAILED","FAILED","OK"))
        self.assertTrue(np.isnan(result.ate[:2]).all())
        self.assertEqual(result.ate[2],95.125)

    def test_infinite_11_rejects_dependent_effects_not_direct(self):
        def change(means):
            means[99,3] = np.inf
        adapter,_,_,_ = injected(mean_change=change)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.output_status["ate"],("OK","FAILED","FAILED"))
        self.assertEqual(result.ate[0],33.375)

    def test_one_arm_exception_preserves_other_arm_and_cause(self):
        def fail(raw,model,payload):
            if payload["treatment"][0] == 1:
                raise ArithmeticError("arm1 isolated error")
        adapter,model,loader,_ = injected(corrupt=fail)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.arm_results[0].output_status["ate"],("OK",)*3)
        self.assertEqual(result.arm_results[1].failure_message,"arm1 isolated error")
        self.assertEqual(model.calls,2)
        self.assertEqual(loader.call_count,1)

    def test_same_instance_arm0_arm1_arm0_fresh_dynamic_reference(self):
        adapter,model,loader,_ = injected(dynamic=True)
        task,request,children = setup_pair()
        results = []
        for arm in (0,1,0):
            result = adapter.estimate_context(task.context_refs[arm],children[arm])
            self.assertEqual(result.execution_status,"OK")
            payload = c.predictor_inputs(task.context_refs[arm])
            self.close("dynamic_arm"+str(arm),result.ate,scalar_dynamic_reference(payload))
            results.append(result)
        self.close("arm0_repeat",results[0].ate,results[2].ate)
        self.assertGreater(np.linalg.norm(results[0].ate-results[1].ate),.01)
        self.assertEqual(model.rebuilds,1)
        self.assertEqual(model.calls,3)
        self.assertEqual(loader.call_count,1)

    def test_dynamic_treatment_indices_y_changes_do_not_rebuild_query_cache(self):
        adapter,model,_,_ = injected(dynamic=True)
        task,request,children = setup_pair()
        base = dict(c.predictor_inputs(task.context_refs[0]))
        modified_t = base["treatment"].copy();modified_t[1] = 1-modified_t[1]
        modified_ids = base["observed_indices"].copy();modified_ids[0] = 199
        modified_y = base["outcomes_observed"].copy();modified_y[0]+=8.
        payloads = [base,changed_inputs(base,"treatment",modified_t),
                    changed_inputs(base,"observed_indices",modified_ids),
                    changed_inputs(base,"outcomes_observed",modified_y),base]
        predictions = []
        for payload in payloads:
            value = adapter(**payload)
            predictions.append(value.arm_means)
            # Independent per-node formula: recover public ATE through the
            # common layer only, with injected already-checked point output.
            result = common.estimate_context(task.context_refs[0],children[0],"kernel-old",
                                              lambda **unused:value)
            self.close("changed_dynamic",result.ate,scalar_dynamic_reference(payload))
        self.assertEqual(model.rebuilds,1)
        self.assertEqual(model.calls,5)
        self.assertFalse(np.array_equal(predictions[0],predictions[1]))
        self.assertFalse(np.array_equal(predictions[0],predictions[2]))
        self.assertFalse(np.array_equal(predictions[0],predictions[3]))
        self.close("dynamic_reset",predictions[0],predictions[-1])

    def test_changed_x_rebuilds_geometry_and_graph_key_is_qualified(self):
        adapter,model,_,_ = injected(dynamic=True)
        task,_,_ = setup_pair()
        base = dict(c.predictor_inputs(task.context_refs[0]))
        adapter(**base)
        x = base["x"].copy();x[0,0]+=1.
        changed = changed_inputs(base,"x",x)
        adapter(**changed);adapter(**base)
        self.assertEqual(model.rebuilds,3)
        self.assertNotEqual(k.query_cache_identity(base["x"],base["adjacency"]),
                            k.query_cache_identity(x,base["adjacency"]))
        a = base["adjacency"].copy();a[1,2]=a[2,1]=1.
        # A different graph is only a geometry-key unit check. Production
        # TaskSpec/predictor inputs retain the canonical Star restriction.
        self.assertNotEqual(k.query_cache_identity(base["x"],base["adjacency"]),
                            k.query_cache_identity(base["x"],a))
        with self.assertRaises(c.ContractError):
            adapter(**{**base,"adjacency":readonly(a)})

    def test_explicit_dynamic_cache_pollution_detected(self):
        def contaminate(raw,model,payload):
            model.caches[0]["residual"] = np.zeros(50)
        adapter,_,_,_ = injected(corrupt=contaminate)
        task,request,children = setup_pair()
        result = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(result.execution_status,"FAILED")
        self.assertIn("KERNEL_CACHE",result.failure_message)

    def test_same_geometry_query_cache_changed_by_y_is_detected(self):
        task,request,children = setup_pair()
        calls = [0]
        def contaminate(raw,model,payload):
            calls[0]+=1
            if calls[0]>1:
                model.caches[0]["query_mean"][0,0]+=float(payload["outcomes_observed"][0])
        adapter,_,_,_ = injected(corrupt=contaminate)
        first = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(first.execution_status,"OK")
        second = adapter.estimate_context(task.context_refs[1],children[1])
        self.assertEqual(second.execution_status,"FAILED")
        self.assertIn("Same-geometry",second.failure_message)

    def test_hidden_stale_mock_is_detectable_by_independent_reference(self):
        # Black-box self-consistency cannot prove absence of arbitrary hidden
        # state. This deliberately bad mock fools native redundancy checks,
        # but the independent dynamic reference MUST detect its stale output.
        model = ToyFrozenMixture(dynamic=True);model.hidden_stale = True
        adapter,_,_,_ = injected(model)
        task,request,children = setup_pair()
        first = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(first.execution_status,"OK")
        stale = adapter.estimate_context(task.context_refs[1],children[1])
        expected = scalar_dynamic_reference(c.predictor_inputs(task.context_refs[1]))
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(stale.ate,expected,atol=ATOL,rtol=0)
        self.assertGreater(np.linalg.norm(stale.ate-expected),.01)

    def test_loader_exception_not_retried_and_assets_still_declared(self):
        loader = Mock(side_effect=RuntimeError("injected load failed"))
        reader = Mock()
        adapter = k.KernelOldAdapter(loader,identity_reader=reader)
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(loader.call_count,1)
        reader.assert_not_called()
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertIn("RuntimeError: injected load failed",result.arm_results[0].failure_message)
        self.assertEqual(result.asset_hashes,dict(k.FROZEN_KERNEL_OLD.asset_hashes))

    def test_support_extrapolation_not_native_finiteness(self):
        adapter,_,_,_ = injected()
        task,request,children = setup_pair()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertTrue(np.isfinite(result.ate).all())
        self.assertTrue(result.population_support.all())
        self.assertFalse(result.empirical_support_mask.all())
        for arm in result.arm_results:
            query = tuple(map(int,np.argwhere(~arm.empirical_support_mask)[0]))
            self.assertTrue(arm.extrapolation_flags["empirical"][query])
            with self.assertRaises(PluginSupportError):
                require_plugin_support(arm.support,(query,))


    def test_complete_pair_context_support_and_extrapolation_inheritance(self):
        from star_v2.support import support_for_pair, support_for_task

        task,request,children = setup_pair()
        contexts = task.context_refs
        self.assertEqual(tuple(context.center_t for context in contexts),(0,1))
        original = tuple(getattr(context,name) for context in contexts for name in c.FACTUAL_KEYS)
        before_bytes = tuple(array.tobytes() for array in original)
        before_dtypes = tuple(array.dtype for array in original)
        before_hashes = tuple(context.input_hash for context in contexts)
        pair_reference = support_for_pair(task)
        context_references = tuple(support_for_task(context) for context in contexts)

        adapter,_,_,_ = injected()
        with patch.object(np,"load",side_effect=AssertionError("No real asset deserialization")):
            result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"OK")

        def exact_array(actual,expected):
            self.assertEqual(actual.shape,expected.shape)
            self.assertEqual(actual.dtype,expected.dtype)
            np.testing.assert_array_equal(actual,expected)

        mask_names = ("population_support","design_support","empirical_support_mask")
        for name in mask_names:
            with self.subTest(level="pair",field=name):
                exact_array(getattr(result,name),getattr(pair_reference,name))

        pair_expected_flags = {
            "empirical":pair_reference.empirical_extrapolation,
            "context_design":tuple(reference.context_design_extrapolation
                                   for reference in context_references)}
        pair_flags = result.extrapolation_flags
        self.assertEqual(set(pair_flags),set(pair_expected_flags))
        exact_array(pair_flags["empirical"],pair_expected_flags["empirical"])
        self.assertIs(type(pair_flags["context_design"]),tuple)
        self.assertEqual(len(pair_flags["context_design"]),len(context_references))
        for actual,expected in zip(pair_flags["context_design"],pair_expected_flags["context_design"]):
            exact_array(actual,expected)

        self.assertEqual(len(result.arm_results),len(contexts))
        self.assertEqual(len(result.support.context_supports),len(contexts))
        for arm_id,(arm,reference) in enumerate(zip(result.arm_results,context_references)):
            self.assertEqual(arm.context_ids,(contexts[arm_id].context_id,))
            self.assertEqual(arm.input_hash,before_hashes[arm_id])
            for name in mask_names:
                with self.subTest(level="context",arm=arm_id,field=name):
                    exact_array(getattr(arm,name),getattr(reference,name))
                    exact_array(getattr(result.support.context_supports[arm_id],name),
                                getattr(reference,name))
            expected_flags = {
                "empirical":reference.empirical_extrapolation,
                "context_design":reference.context_design_extrapolation}
            self.assertEqual(set(arm.extrapolation_flags),set(expected_flags))
            for name,expected in expected_flags.items():
                with self.subTest(level="context_flags",arm=arm_id,field=name):
                    exact_array(arm.extrapolation_flags[name],expected)
            # Finite model responses cannot replace empirical coverage.
            finite_model_mask = np.isfinite(arm.mu)
            self.assertTrue(finite_model_mask.all())
            self.assertFalse(np.array_equal(arm.empirical_support_mask,finite_model_mask))

        self.assertEqual(result.input_hashes,before_hashes)
        self.assertEqual(tuple(context.input_hash for context in contexts),before_hashes)
        after_arrays = tuple(getattr(context,name) for context in contexts for name in c.FACTUAL_KEYS)
        for after,before in zip(after_arrays,original):
            self.assertIs(after,before)
        self.assertEqual(tuple(array.dtype for array in after_arrays),before_dtypes)
        self.assertEqual(tuple(array.tobytes() for array in after_arrays),before_bytes)
        for context in contexts:
            self.assertEqual(context.outcomes_observed.dtype,np.float64)
            c.validate_task(context)

    def test_no_assets_refit_calibration_or_legacy_import_called(self):
        adapter,model,loader,_ = injected()
        task,request,children = setup_pair()
        with patch.object(np,"load",side_effect=AssertionError("NPZ forbidden")),\
             patch.object(model,"fit",side_effect=AssertionError("refit forbidden")) as fit,\
             patch.object(model,"calibrate",side_effect=AssertionError("calibration forbidden")) as calibrate:
            result = adapter.estimate_pair(task,request,context_requests=children)
        fit.assert_not_called();calibrate.assert_not_called()
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(loader.call_count,1)


def tearDownModule():
    print("S4.3A_REFERENCE_RESIDUALS "+json.dumps(ERRORS,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
