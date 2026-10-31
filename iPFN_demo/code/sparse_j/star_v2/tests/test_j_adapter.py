"""Mock-only J.1 acceptance. Fixed atol=1e-12, rtol=0; no real loader."""
import hashlib
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from star_v2.contracts import FACTUAL_KEYS, predictor_inputs
from star_v2.j_adapter import JAdapter
from test_kernel_new_adapter import tasks_fixture


def signature(task):
    return {key: (id(getattr(task, key)), getattr(task, key).dtype.str,
                  getattr(task, key).shape,
                  hashlib.sha256(getattr(task, key).tobytes()).hexdigest())
            for key in FACTUAL_KEYS}


class PointMock:
    def __init__(self):
        self.calls = []

    def predict_arm_means(self, **inputs):
        self.calls.append(inputs)
        n = np.arange(200, dtype=np.float64)
        # Asymmetric canonical 00,01,10,11, with nonconstant node contrasts.
        return np.column_stack((n, n + 90, 2*n + 3, 4*n + 8))


class JAdapterTests(unittest.TestCase):
    def setUp(self):
        pair, _, children = tasks_fixture()
        self.task, self.request = pair.context_refs[0], children[0]
        self.model = PointMock()
        self.loader = Mock(return_value=self.model)
        self.adapter = JAdapter(loader=self.loader)
        self.before = signature(self.task)
        self.input_hash = self.task.input_hash
        # Any accidental torch checkpoint deserialization must fail the test.
        self.load_patch = patch("torch.load", side_effect=AssertionError("real load forbidden"))
        self.torch_load = self.load_patch.start()
        self.addCleanup(self.load_patch.stop)

    def tearDown(self):
        self.assertEqual(signature(self.task), self.before)
        self.assertEqual(self.task.input_hash, self.input_hash)
        self.torch_load.assert_not_called()

    def test_construction_does_not_load(self):
        self.loader.assert_not_called()
        self.assertEqual(self.model.calls, [])

    def test_default_loader_stays_unimported_at_construction(self):
        # No execution of the frozen loader or legacy module is needed.
        before = sys.modules.get("predict")
        with patch("star_v2.j_adapter._default_loader",
                   side_effect=AssertionError("default loader must remain lazy")) as loader:
            JAdapter()
            loader.assert_not_called()
        self.assertIs(sys.modules.get("predict"), before)

    def test_not_identified_blocks_loader_and_predict(self):
        pair, _, children = tasks_fixture(status="NOT_IDENTIFIED")
        result = self.adapter.estimate_context(pair.context_refs[0], children[0])
        self.loader.assert_not_called()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(result.status, "NOT_IDENTIFIED")
        self.assertEqual(result.failure_message, "exact upstream reason")

    def test_point_loads_once_and_reuses_same_model(self):
        a = self.adapter.estimate_context(self.task, self.request)
        b = self.adapter.estimate_context(self.task, self.request)
        self.loader.assert_called_once_with()
        self.assertEqual(len(self.model.calls), 2)
        self.assertIs(self.adapter._model, self.model)
        np.testing.assert_array_equal(a.ate, b.ate)
        self.assertEqual(a.status, "POINT")

    def test_exact_five_keys_and_canonical_dtypes(self):
        self.adapter.estimate_context(self.task, self.request)
        inputs = self.model.calls[0]
        self.assertEqual(set(inputs), set(FACTUAL_KEYS))
        expected = (np.float32, np.float64, np.float64, np.int64, np.float32)
        for key, dtype in zip(FACTUAL_KEYS, expected):
            self.assertEqual(inputs[key].dtype, np.dtype(dtype))

    def test_outcomes_are_independent_float32_copy(self):
        public = predictor_inputs(self.task)
        before = public["outcomes_observed"].tobytes()
        self.adapter(**public)
        converted = self.model.calls[0]["outcomes_observed"]
        self.assertIsNot(converted, public["outcomes_observed"])
        self.assertTrue(converted.flags.owndata)
        self.assertIsNone(converted.base)
        self.assertFalse(np.shares_memory(converted, self.task.outcomes_observed))
        np.testing.assert_array_equal(converted, public["outcomes_observed"].astype(np.float32))
        converted[:] = -999
        self.assertEqual(public["outcomes_observed"].tobytes(), before)

    def test_other_four_arrays_preserved_and_isolated(self):
        public = predictor_inputs(self.task)
        self.adapter(**public)
        for key in FACTUAL_KEYS[:-1]:
            received = self.model.calls[0][key]
            self.assertIs(received, public[key])
            self.assertFalse(received.flags.writeable)
            self.assertFalse(np.shares_memory(received, getattr(self.task, key)))
            self.assertEqual(received.dtype, getattr(self.task, key).dtype)
            self.assertEqual(received.tobytes(), getattr(self.task, key).tobytes())

    def test_extra_or_missing_keys_rejected_before_loader(self):
        public = dict(predictor_inputs(self.task))
        for extra in ("truth", "h", "beta", "full_y", "noise", "support"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.adapter(**dict(public, **{extra: object()}))
        for key in FACTUAL_KEYS:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.adapter(**{k:v for k,v in public.items() if k != key})
        self.loader.assert_not_called()
        self.assertEqual(self.model.calls, [])

    def test_cpu_rng_restored_after_loader_consumes_rng(self):
        before = torch.random.get_rng_state().clone()
        def load():
            torch.rand(37)
            return self.model
        self.loader.side_effect = load
        result = self.adapter.estimate_context(self.task, self.request)
        self.assertEqual(result.execution_status, "OK")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.loader.assert_called_once_with()

    def test_cpu_rng_restored_when_loader_raises(self):
        before = torch.random.get_rng_state().clone()
        def load():
            torch.rand(13)
            raise RuntimeError("mock construction failed")
        self.loader.side_effect = load
        result = self.adapter.estimate_context(self.task, self.request)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(result.execution_status, "FAILED")
        self.assertIn("mock construction failed", result.failure_message)
        self.assertEqual(self.model.calls, [])

    def test_canonical_contrasts_identity_and_full_graph_mean(self):
        result = self.adapter.estimate_context(self.task, self.request)
        n = np.arange(200, dtype=np.float64)
        expected = np.column_stack((n+3, 2*n+5, 3*n+8))
        np.testing.assert_allclose(result.ite, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result.ate, [102.5, 204., 306.5], atol=1e-12, rtol=0)
        self.assertEqual(result.mu[0,0,1], 90.)
        self.assertEqual(result.mu[0,1,0], 3.)
        self.assertEqual(result.request, self.request)
        self.assertEqual(result.method_id, "j")
        self.assertEqual(result.output_status["ate"], ("OK", "OK", "OK"))

    def test_default_not_identified_never_imports_or_loads(self):
        pair, _, children = tasks_fixture(status="NOT_IDENTIFIED")
        before = sys.modules.get("predict")
        with patch("star_v2.j_adapter._default_loader",
                   side_effect=AssertionError("forbidden load")) as loader:
            result = JAdapter().estimate_context(pair.context_refs[0], children[0])
            loader.assert_not_called()
        self.assertEqual(result.status, "NOT_IDENTIFIED")
        self.assertIs(sys.modules.get("predict"), before)




# J.2 additions: J.1 tests above are retained unchanged.
from collections.abc import Mapping
from dataclasses import replace
import builtins
import importlib.util

from star_v2.j_adapter import INTERVAL_KIND
from star_v2.model_adapters import ModelPrediction, ModelIntervals

ATOL = 1e-12
RTOL = 0
REFERENCE_RESIDUALS = []


def interval_result():
    return {
        "ate_lower": np.array([100., 200., 300.]),
        "ate_upper": np.array([105., 208., 312.]),
        "interval_level": np.array(.95),
        "interval_kind": np.array("frozen_J_marginal_posterior_credible"),
    }


class JContextCompletionTests(unittest.TestCase):
    setUp = JAdapterTests.setUp
    tearDown = JAdapterTests.tearDown

    def with_intervals(self, callback=None):
        self.interval_callback = Mock(return_value=interval_result()) if callback is None else callback
        self.adapter = JAdapter(loader=self.loader, interval_predictor=self.interval_callback)
        return replace(self.request, requested_outputs=("mu", "ite", "ate", "interval"))

    def assert_posterior(self, result, statuses):
        self.assertEqual(result.output_status["interval"], statuses)
        self.assertEqual(tuple(r.status for r in result.intervals), statuses)
        for record in result.intervals:
            self.assertEqual(record.source, "model_posterior")
            self.assertEqual(record.level, .95)
            self.assertEqual(record.context_id, self.task.context_id)
        uncertainty = result.sampling_uncertainty
        self.assertEqual(uncertainty.source, "model_posterior")
        self.assertEqual(uncertainty.statistical_unit, "context")
        self.assertIsNone(uncertainty.effective_sample_count)

    def test_scalar_reference_mu_ite_ate_all_200_nodes(self):
        result = self.adapter.estimate_context(self.task, self.request)
        mu = np.empty((200,2,2))
        ite = np.empty((200,3))
        totals = [0., 0., 0.]
        for i in range(200):
            y00, y01, y10, y11 = float(i), float(i+90), float(2*i+3), float(4*i+8)
            mu[i,0,0], mu[i,0,1] = y00, y01
            mu[i,1,0], mu[i,1,1] = y10, y11
            direct, spillover, total = y10-y00, y11-y10, y11-y00
            ite[i] = direct, spillover, total
            for k, value in enumerate((direct,spillover,total)):
                totals[k] += value
            self.assertEqual(total, direct+spillover)
        ate = np.array([value/200 for value in totals])
        for actual, expected in ((result.mu,mu), (result.ite,ite), (result.ate,ate)):
            np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
            REFERENCE_RESIDUALS.append(float(np.max(np.abs(actual-expected))))
        self.assertFalse(np.allclose(result.ate, ite[:50].mean(0), atol=ATOL, rtol=RTOL))
        self.assertEqual(result.request, self.request)

    def test_point_bridge_returns_canonical_model_prediction(self):
        result = self.adapter(**predictor_inputs(self.task))
        self.assertIs(type(result), ModelPrediction)
        self.assertEqual(result.arm_order, ("00","01","10","11"))
        self.assertEqual(result.arm_means.shape, (200,4))
        self.assertEqual(result.arm_means[0].tolist(), [0.,90.,3.,8.])

    def test_wrong_shape_and_dtype_fail_without_partial_success(self):
        for output in (np.zeros((199,4)), np.zeros((200,2,2)),
                       np.zeros((200,4),dtype=np.int64)):
            with self.subTest(shape=output.shape, dtype=output.dtype):
                model = Mock()
                model.predict_arm_means.return_value = output
                adapter = JAdapter(loader=Mock(return_value=model))
                result = adapter.estimate_context(self.task, self.request)
                self.assertEqual(result.execution_status, "FAILED")
                self.assertEqual(result.output_status["ate"], ("FAILED",)*3)
                self.assertTrue(np.isnan(result.ate).all())

    def test_nonfinite_states_preserve_independent_effects(self):
        expected = (
            ("FAILED","OK","FAILED"), ("OK","OK","OK"),
            ("FAILED","FAILED","OK"), ("OK","FAILED","FAILED"))
        for column in range(4):
            with self.subTest(column=column):
                output = np.tile(np.array([0.,90.,3.,8.]),(200,1))
                output[17,column] = np.nan
                model = Mock()
                model.predict_arm_means.return_value = output
                result = JAdapter(loader=Mock(return_value=model)).estimate_context(
                    self.task, self.request)
                self.assertEqual(result.output_status["mu"], "FAILED")
                self.assertEqual(result.output_status["ate"], expected[column])
                for k, status in enumerate(expected[column]):
                    if status == "OK":
                        self.assertEqual(result.ate[k], (3.,5.,8.)[k])
                    else:
                        self.assertTrue(np.isnan(result.ate[k]))

    def test_point_exception_blocks_interval_and_all_point_outputs(self):
        request = self.with_intervals()
        self.model.predict_arm_means = Mock(side_effect=RuntimeError("exact point failure"))
        result = self.adapter.estimate_context(self.task, request)
        self.interval_callback.assert_not_called()
        self.assertEqual(result.output_status["ate"], ("FAILED",)*3)
        self.assertEqual(result.output_status["mu"], "FAILED")
        self.assertTrue(np.isnan(result.ate).all())
        self.assertIn("exact point failure", result.failure_message)

    def test_successful_95_percent_intervals_and_kind(self):
        request = self.with_intervals()
        result = self.adapter.estimate_context(self.task, request)
        self.assert_posterior(result, ("OK",)*3)
        self.assertEqual(result.sampling_uncertainty.status, "OK")
        self.assertEqual(result.execution_status, "OK")
        self.assertEqual([r.lower for r in result.intervals], [100.,200.,300.])
        self.assertEqual([r.upper for r in result.intervals], [105.,208.,312.])
        self.assertEqual(tuple(r.effect for r in result.intervals), ("Direct","Spillover","Total"))
        self.assertEqual(result.diagnostics["interval"]["interval_kind"], INTERVAL_KIND)
        self.assertEqual(result.request, request)
        self.assertEqual(result.request.input_hashes, self.request.input_hashes)

    def test_interval_bridge_returns_public_model_intervals(self):
        self.with_intervals()
        result = self.adapter.predict_intervals(**predictor_inputs(self.task))
        self.assertIs(type(result), ModelIntervals)
        self.assertEqual(result.level, .95)
        self.assertEqual(result.diagnostics, {"interval_kind": INTERVAL_KIND})

    def test_interval_exception_preserves_points_and_exact_failure(self):
        callback = Mock(side_effect=FloatingPointError("mock negative posterior variance"))
        request = self.with_intervals(callback)
        result = self.adapter.estimate_context(self.task, request)
        self.assert_posterior(result, ("FAILED",)*3)
        self.assertEqual(result.sampling_uncertainty.status, "FAILED")
        self.assertEqual(result.execution_status, "FAILED")
        self.assertEqual(result.output_status["mu"], "OK")
        self.assertEqual(result.output_status["ite"], ("OK",)*3)
        self.assertEqual(result.output_status["ate"], ("OK",)*3)
        np.testing.assert_allclose(result.ate,[102.5,204.,306.5],atol=ATOL,rtol=RTOL)
        for failure in result.failure_provenance["intervals"].values():
            self.assertEqual(failure["type"], "FloatingPointError")
            self.assertEqual(failure["message"], "mock negative posterior variance")
        for record in result.intervals:
            self.assertIsNone(record.lower)
            self.assertIsNone(record.upper)

    def test_partial_invalid_interval_is_local_and_uncertainty_remains_ok(self):
        native = interval_result()
        native["ate_upper"][1] = np.nan
        request = self.with_intervals(Mock(return_value=native))
        result = self.adapter.estimate_context(self.task, request)
        self.assert_posterior(result, ("OK","FAILED","OK"))
        self.assertEqual(result.sampling_uncertainty.status, "OK")
        self.assertEqual(result.output_status["ate"], ("OK",)*3)
        self.assertEqual(result.execution_status, "FAILED")
        self.assertEqual(len(result.failure_provenance["intervals"]), 1)
        self.assertEqual(result.intervals[0].lower, 100.)
        self.assertEqual(result.intervals[2].upper, 312.)

    def test_point_and_interval_share_model_and_conversion_rule(self):
        request = self.with_intervals()
        result = self.adapter.estimate_context(self.task, request)
        self.loader.assert_called_once_with()
        args, kwargs = self.interval_callback.call_args
        self.assertEqual(args, (self.model,))
        self.assertEqual(kwargs.pop("level"), .95)
        self.assertEqual(set(kwargs), set(FACTUAL_KEYS))
        points = self.model.calls[0]
        for key in FACTUAL_KEYS:
            np.testing.assert_array_equal(points[key],kwargs[key])
            self.assertEqual(points[key].dtype,kwargs[key].dtype)
        py, iy = points["outcomes_observed"],kwargs["outcomes_observed"]
        self.assertEqual(py.dtype,np.dtype("float32"))
        self.assertFalse(np.shares_memory(py,iy))
        for y in (py,iy):
            self.assertTrue(y.flags.owndata)
            self.assertFalse(np.shares_memory(y,self.task.outcomes_observed))
        self.assertEqual(result.output_status["ate"],("OK",)*3)

    def test_metadata_contains_only_json_like_values(self):
        request = self.with_intervals()
        result = self.adapter.estimate_context(self.task,request)
        def check(value):
            self.assertNotIsInstance(value,(np.ndarray,torch.Tensor,np.generic))
            if isinstance(value,Mapping):
                for key,item in value.items():
                    self.assertIs(type(key),str)
                    check(item)
            elif isinstance(value,(tuple,list)):
                for item in value: check(item)
            else:
                self.assertIn(type(value),(str,int,float,bool,type(None)))
        check(result.diagnostics)
        check(result.failure_provenance)
        check(result.estimability)

    def test_not_identified_with_intervals_has_zero_calls(self):
        pair,_,children = tasks_fixture(status="NOT_IDENTIFIED")
        request = replace(children[0],requested_outputs=("mu","ite","ate","interval"))
        callback = Mock(side_effect=AssertionError("interval must not run"))
        adapter = JAdapter(loader=self.loader,interval_predictor=callback)
        result = adapter.estimate_context(pair.context_refs[0],request)
        self.loader.assert_not_called()
        callback.assert_not_called()
        self.assertEqual(self.model.calls,[])
        self.assertEqual(result.status,"NOT_IDENTIFIED")
        self.assertEqual(result.failure_message,"exact upstream reason")

    def test_no_interval_request_never_calls_interval_callback(self):
        callback = Mock(side_effect=AssertionError("not requested"))
        result = JAdapter(loader=self.loader,interval_predictor=callback).estimate_context(
            self.task,self.request)
        callback.assert_not_called()
        self.assertEqual(result.output_status["interval"],("NOT_REQUESTED",)*3)

    def test_import_itself_does_not_import_real_j_modules(self):
        import star_v2.j_adapter as module
        original = builtins.__import__
        def guarded(name,*args,**kwargs):
            if name == "predict" or name.startswith("pfn_social"):
                raise AssertionError("premature real J import: "+name)
            return original(name,*args,**kwargs)
        spec = importlib.util.spec_from_file_location("star_v2._j_import_probe",module.__file__)
        isolated = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__",side_effect=guarded):
            spec.loader.exec_module(isolated)
            isolated.JAdapter()

    def test_wrong_native_level_or_kind_fails_intervals_only(self):
        for key,value in (("interval_level",np.array(.90)),
                          ("interval_kind",np.array("incorrect_kind"))):
            with self.subTest(field=key):
                native = interval_result()
                native[key] = value
                request = self.with_intervals(Mock(return_value=native))
                result = self.adapter.estimate_context(self.task,request)
                self.assert_posterior(result,("FAILED",)*3)
                self.assertEqual(result.output_status["ate"],("OK",)*3)




# J.3 pair and complete-state mock acceptance; tolerances remain unchanged.
from star_v2.j_adapter import model_state_fingerprint
from star_v2.model_adapters import context_requests_from_parent
from star_v2.estimation import PairEstimateBundle
from star_v2.support import support_for_pair, support_for_task

PAIR_REFERENCE_RESIDUALS = []
STATE_EVIDENCE = []


def state_snapshot(model):
    # Independent snapshots of actual parameters and buffers, not production hash.
    owner = model.model
    return {
        category: {name: (tensor.detach().cpu().numpy().dtype.str,
                          tuple(tensor.shape),
                          hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest())
                   for name,tensor in iterator()}
        for category,iterator in (("parameters",owner.named_parameters),
                                  ("buffers",owner.named_buffers))
    }


class StatefulPointMock:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.register_parameter("weights",torch.nn.Parameter(
            torch.tensor([1.,2.],dtype=torch.float64),requires_grad=False))
        self.model.register_buffer("prior",torch.tensor([[3.,4.]],dtype=torch.float64))
        self.model.eval()
        self.calls = []
        self.mutate_point = False
        self.overflow_arm1 = False

    def predict_arm_means(self,**inputs):
        self.calls.append(inputs)
        if self.mutate_point:
            self.model.prior.add_(1.)
        a = float(inputs["treatment"][0])
        result = np.empty((200,4))
        for i in range(200):
            result[i] = (i+7*a, i+90-3*a, 2*i+3+11*a, 4*i+8+19*a)
        if a == 1 and self.overflow_arm1:
            # Only Direct overflows. Other state contrasts/means remain finite.
            result[:] = 0.
            result[0] = (-1e308,0.,1e308,0.)
        return result


class JPairCompletionTests(unittest.TestCase):
    def setUp(self):
        self.pair,parent,_ = tasks_fixture()
        self.parent = replace(parent,requested_outputs=("mu","ite","ate","interval"))
        self.children = context_requests_from_parent(
            self.pair,self.parent,request_ids=("j-child-0","j-child-1"))
        self.model = StatefulPointMock()
        self.original = tuple(signature(t) for t in self.pair.context_refs)
        self.hashes = self.pair.input_hashes if hasattr(self.pair,"input_hashes") else tuple(
            t.input_hash for t in self.pair.context_refs)
        self.loader = Mock(return_value=self.model)
        def intervals(model,**inputs):
            self.assertIs(model,self.model)
            self.assertEqual(inputs["level"],.95)
            a = float(inputs["treatment"][0])
            return dict(ate_lower=np.array([100.+a,200.+a,300.+a]),
                        ate_upper=np.array([105.+a,208.+a,312.+a]),
                        interval_level=np.array(.95),interval_kind=np.array(INTERVAL_KIND))
        self.interval_callback = Mock(side_effect=intervals)
        self.adapter = JAdapter(loader=self.loader,interval_predictor=self.interval_callback)
        self.patch_load = patch("torch.load",side_effect=AssertionError("real load forbidden"))
        self.torch_load = self.patch_load.start()
        self.addCleanup(self.patch_load.stop)

    def tearDown(self):
        self.assertEqual(tuple(signature(t) for t in self.pair.context_refs),self.original)
        self.assertEqual(tuple(t.input_hash for t in self.pair.context_refs),self.hashes)
        self.torch_load.assert_not_called()

    def estimate(self):
        return self.adapter.estimate_pair(self.pair,self.parent,context_requests=self.children)

    def test_pair_scalar_reference_different_arms_all_nodes(self):
        result = self.estimate()
        self.assertIs(type(result),PairEstimateBundle)
        arm_expected = []
        for a in (0.,1.):
            totals = [0.,0.,0.]
            for i in range(200):
                y00,y10,y11 = i+7*a,2*i+3+11*a,4*i+8+19*a
                for k,value in enumerate((y10-y00,y11-y10,y11-y00)):
                    totals[k] += value
            arm_expected.append([value/200 for value in totals])
        expected = [(arm_expected[0][k]+arm_expected[1][k])/2 for k in range(3)]
        for arm,values in zip(result.arm_results,arm_expected):
            np.testing.assert_allclose(arm.ate,values,atol=ATOL,rtol=RTOL)
        np.testing.assert_allclose(result.ate,expected,atol=ATOL,rtol=RTOL)
        PAIR_REFERENCE_RESIDUALS.append(float(np.max(np.abs(result.ate-expected))))
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertNotEqual(result.arm_results[0].ate[0],result.arm_results[1].ate[0])
        self.assertEqual(result.ate[2],result.ate[0]+result.ate[1])
        self.loader.assert_called_once_with()

    def test_context_bundles_original_intervals_and_identity_retained(self):
        result = self.estimate()
        self.assertEqual(result.request,self.parent)
        self.assertEqual(result.request_id,self.parent.request_id)
        self.assertEqual(result.target_id,self.parent.target_id)
        self.assertEqual(result.pair_id,self.pair.pair_id)
        np.testing.assert_array_equal(result.node_ids,self.parent.node_ids)
        self.assertEqual(result.input_hashes,tuple(t.input_hash for t in self.pair.context_refs))
        self.assertEqual(result.intervals,())
        self.assertEqual(result.output_status["interval"],("UNSUPPORTED",)*3)
        self.assertEqual(result.sampling_uncertainty.status,"UNSUPPORTED")
        for k,arm in enumerate(result.arm_results):
            self.assertEqual(arm.request,self.children[k])
            self.assertEqual(arm.request.parent_request_id,self.parent.request_id)
            self.assertEqual(arm.context_ids,(self.pair.context_refs[k].context_id,))
            self.assertEqual(arm.request_id,self.children[k].request_id)
            self.assertEqual(arm.input_hashes,(self.pair.context_refs[k].input_hash,))
            self.assertEqual(result.context_intervals[k],arm.intervals)
            self.assertEqual([r.lower for r in arm.intervals],[100.+k,200.+k,300.+k])
            self.assertEqual([r.upper for r in arm.intervals],[105.+k,208.+k,312.+k])
            self.assertEqual(arm.diagnostics["interval"]["interval_kind"],INTERVAL_KIND)
            self.assertEqual(arm.sampling_uncertainty.source,"model_posterior")
            self.assertEqual(arm.sampling_uncertainty.statistical_unit,"context")
            self.assertIsNone(arm.sampling_uncertainty.effective_sample_count)

    def test_single_arm_single_effect_failure_is_local(self):
        self.model.overflow_arm1 = True
        result = self.estimate()
        self.assertEqual(result.arm_results[0].output_status["ate"],("OK",)*3)
        self.assertEqual(result.arm_results[1].output_status["ate"],("FAILED","OK","OK"))
        self.assertEqual(result.output_status["ate"],("FAILED","OK","OK"))
        self.assertTrue(np.isnan(result.ate[0]))
        self.assertTrue(np.isfinite(result.ate[1:]).all())
        self.assertEqual(result.failure_type,"NECESSARY_ARM_FAILURE")
        self.assertTrue(result.arm_results[1].failure_provenance["effects"]["Direct"])

    def test_whole_interval_exception_retains_all_three_failures_and_pair_ate(self):
        def intervals(model,**inputs):
            if inputs["treatment"][0] == 1:
                raise FloatingPointError("exact arm1 posterior failure")
            return interval_result()
        self.interval_callback.side_effect = intervals
        result = self.estimate()
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(result.arm_results[1].execution_status,"FAILED")
        arm = result.arm_results[1]
        self.assertEqual(tuple(r.status for r in arm.intervals),("FAILED",)*3)
        self.assertTrue(all(r.source == "model_posterior" for r in arm.intervals))
        self.assertEqual(arm.sampling_uncertainty.status,"FAILED")
        self.assertEqual(arm.sampling_uncertainty.source,"model_posterior")
        self.assertEqual(arm.sampling_uncertainty.statistical_unit,"context")
        self.assertIsNone(arm.sampling_uncertainty.effective_sample_count)
        for evidence in arm.failure_provenance["intervals"].values():
            self.assertEqual(evidence["message"],"exact arm1 posterior failure")
            self.assertEqual(evidence["type"],"FloatingPointError")
        self.assertEqual(result.intervals,())

    def test_pair_and_both_context_support_and_all_extrapolation_flags(self):
        result = self.estimate()
        refs = (support_for_pair(self.pair),)+tuple(
            support_for_task(t) for t in self.pair.context_refs)
        for bundle,reference in zip((result,)+result.arm_results,refs):
            for name in ("population_support","design_support","empirical_support_mask"):
                actual,expected = getattr(bundle,name),getattr(reference,name)
                self.assertEqual(actual.shape,expected.shape)
                self.assertEqual(actual.dtype,expected.dtype)
                np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(bundle.extrapolation_flags["empirical"],
                                          reference.empirical_extrapolation)
            expected_design = (tuple(s.context_design_extrapolation for s in reference.context_supports)
                               if bundle is result else reference.context_design_extrapolation)
            np.testing.assert_array_equal(bundle.extrapolation_flags["context_design"],expected_design)
            self.assertFalse(bundle.empirical_support_mask.all())
            self.assertTrue(bundle.extrapolation_flags["empirical"].any())

    def test_pair_not_identified_zero_loader_predict_and_interval_calls(self):
        pair,parent,_ = tasks_fixture(status="NOT_IDENTIFIED")
        parent = replace(parent,requested_outputs=("mu","ite","ate","interval"))
        result = self.adapter.estimate_pair(
            pair,parent,request_ids=("ni-child-0","ni-child-1"))
        self.loader.assert_not_called()
        self.interval_callback.assert_not_called()
        self.assertEqual(self.model.calls,[])
        self.assertEqual(result.status,"NOT_IDENTIFIED")
        self.assertEqual(result.failure_message,"exact upstream reason")
        for arm in result.arm_results:
            self.assertEqual(arm.status,"NOT_IDENTIFIED")
            self.assertEqual(arm.request.parent_request_id,parent.request_id)

    def test_explicit_child_ids_bound_requests_and_invalid_linkage(self):
        result = self.adapter.estimate_pair(
            self.pair,self.parent,request_ids=("explicit-0","explicit-1"))
        self.assertEqual(tuple(a.request_id for a in result.arm_results),("explicit-0","explicit-1"))
        pair,parent,children = tasks_fixture(bound=True)
        bound = self.adapter.estimate_pair(pair,parent,context_requests=children)
        self.assertEqual(tuple(a.request_id for a in bound.arm_results),("bound","bound"))
        loader = Mock(side_effect=AssertionError("must reject before loader"))
        wrong = (replace(self.children[0],parent_request_id="wrong-parent"),self.children[1])
        with self.assertRaises(ValueError):
            JAdapter(loader=loader).estimate_pair(self.pair,self.parent,context_requests=wrong)
        loader.assert_not_called()

    def test_metadata_never_retains_model_tensors_or_arrays(self):
        result = self.estimate()
        def check(value):
            self.assertNotIsInstance(value,(torch.Tensor,np.ndarray,np.generic,StatefulPointMock))
            if isinstance(value,Mapping):
                for key,item in value.items():
                    self.assertIs(type(key),str)
                    check(item)
            elif isinstance(value,(tuple,list)):
                for item in value: check(item)
            else:
                self.assertIn(type(value),(str,int,float,bool,type(None)))
        for bundle in (result,)+result.arm_results:
            check(bundle.diagnostics)
            check(bundle.failure_provenance)
            check(bundle.estimability)

    def test_parameters_buffers_fingerprint_point_interval_pair_unchanged(self):
        before = state_snapshot(self.model)
        initial = model_state_fingerprint(self.model)
        self.assertTrue(before["parameters"])
        self.assertTrue(before["buffers"])
        self.adapter(**predictor_inputs(self.pair.context_refs[0]))
        after_point = model_state_fingerprint(self.model)
        self.assertEqual(state_snapshot(self.model),before)
        self.adapter.predict_intervals(**predictor_inputs(self.pair.context_refs[0]))
        after_interval = model_state_fingerprint(self.model)
        self.assertEqual(state_snapshot(self.model),before)
        result = self.estimate()
        final = model_state_fingerprint(self.model)
        self.assertEqual(state_snapshot(self.model),before)
        self.assertEqual((initial,after_point,after_interval,final),(initial,)*4)
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        STATE_EVIDENCE.append({"before":initial,"after_point":after_point,
                               "after_interval":after_interval,"after_pair":final,
                               "independent_parameters_buffers":before})

    def test_point_buffer_mutation_is_detected(self):
        self.model.mutate_point = True
        result = self.estimate()
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertIn("parameters/buffers state changed",result.arm_results[0].failure_message)
        self.interval_callback.assert_not_called()

    def test_interval_parameter_mutation_preserves_already_valid_point(self):
        def mutate(model,**inputs):
            with torch.no_grad():
                model.model.weights.add_(1.)
            return interval_result()
        self.interval_callback.side_effect = mutate
        result = self.adapter.estimate_context(self.pair.context_refs[0],self.children[0])
        self.assertEqual(result.output_status["ate"],("OK",)*3)
        self.assertEqual(result.output_status["interval"],("FAILED",)*3)
        for evidence in result.failure_provenance["intervals"].values():
            self.assertIn("parameters/buffers state changed",evidence["message"])

    def test_external_state_mutation_rejected_before_next_prediction(self):
        self.adapter.estimate_context(self.pair.context_refs[0],self.children[0])
        calls = len(self.model.calls)
        self.model.model.prior.add_(1.)
        result = self.adapter.estimate_context(self.pair.context_refs[1],self.children[1])
        self.assertEqual(len(self.model.calls),calls)
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)

    def test_pair_rng_restored_and_repeated_execution_deterministic(self):
        before = torch.random.get_rng_state().clone()
        def load():
            torch.rand(29)
            return self.model
        self.loader.side_effect = load
        first = self.estimate()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        second = self.estimate()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.loader.assert_called_once_with()
        np.testing.assert_array_equal(first.ate,second.ate)
        for a,b in zip(first.arm_results,second.arm_results):
            np.testing.assert_array_equal(a.mu,b.mu)
            self.assertEqual(a.intervals,b.intervals)


if __name__ == "__main__":
    unittest.main()
