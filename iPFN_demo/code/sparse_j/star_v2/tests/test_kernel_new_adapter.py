"""S4.3-B pure mock acceptance; no torch, checkpoint or simulator calls.

atol=1e-12, rtol=0 fixed BEFORE first execution. Public TaskSpec fixtures are
hand-built; RNG identities are metadata only, no Generator is constructed.
"""
import builtins
import copy
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from star_v2 import contracts as c, model_adapters as common, kernel_new_adapter as k
from star_v2.config import CONFIG
from star_v2.estimation import EstimationRequest
from star_v2.support import support_for_pair, support_for_task, require_plugin_support, PluginSupportError

ATOL = 1e-12
ERRORS = {}


def tasks_fixture(*, status="POINT", bound=False):
    x = np.arange(4000,dtype=np.float32).reshape(200,20)/np.float32(4096)
    adjacency = np.zeros((200,200));adjacency[0,1:] = adjacency[1:,0] = 1.
    nodes = np.arange(200,dtype=np.int64)
    indices = np.arange(50,dtype=np.int64)
    public = []
    for arm in (0,1):
        treatment = np.r_[float(arm),(np.arange(199)%3 == 0).astype(np.float64)]
        fields = dict(protocol_id=CONFIG.protocol_id,protocol_version=CONFIG.protocol_version,
            protocol_hash=CONFIG.protocol_sha256,protocol_code=CONFIG.protocol_code,
            root_seed=CONFIG.root_seed,cohort="test",cohort_code=2,mechanism_id=0,
            pair_repeat_id=0,center_t=arm,pair_id=c.pair_id_for(2,0,0),
            context_id=c.context_id_for(2,0,0,arm),target_id=CONFIG.target_id if bound else None,
            request_id="bound" if bound else None,reference_assignment="iid Bernoulli(0.5)",
            arm_order=("00","01","10","11"),rng_identity=c._rng(2,0,0))
        arrays = dict(node_ids=nodes,x=x,adjacency=adjacency,treatment=treatment,
                      observed_indices=indices,outcomes_observed=np.arange(50,dtype=np.float64)/16+arm/8)
        digest = c._digest("star-v2-task-input-v1",{n:fields[n] for n in c._META},
                           list(arrays.items()))
        public.append(c.TaskSpec(**fields,**arrays,input_hash=digest))
    pair = c.pair_from_tasks(tuple(public))
    kwargs = dict(failure_type="UPSTREAM_ID",failure_message="exact upstream reason",
                  failure_provenance={"code":"NI17"}) if status=="NOT_IDENTIFIED" else {}
    parent = EstimationRequest.from_task(pair,status=status,target_id=CONFIG.target_id,
        request_id="bound" if bound else "mock-parent",requested_outputs=("mu","ite","ate"),**kwargs)
    children = common.context_requests_from_parent(pair,parent,
        request_ids=("bound","bound") if bound else ("mock-arm-0","mock-arm-1"))
    return pair,parent,children


class TensorMock:
    def __init__(self,value,requires_grad=False):
        self.array = np.array(value,dtype=np.float64)
        self.requires_grad = requires_grad
        self.freeze_calls = 0
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array
    def requires_grad_(self,value):
        self.requires_grad = value
        self.freeze_calls += 1
        return self


class ModuleMock:
    def __init__(self): self.training = True


class CalibrationMock:
    """Source-shaped tensors, independent toy prediction, no internal cache API."""
    def __init__(self, *, dynamic=False, mutation=None, output_change=None, exception=False):
        self.identity = "phase5"
        self.design_kinds = ["phase5","phase5"]
        self.specs = [
            {"kind":"polynomial","x_mean":np.zeros(20),"x_scale":np.ones(20),
             "projection":np.arange(40,dtype=np.float64).reshape(20,2)/64},
            {"kind":"linear","input_kind":"raw20","xmean":np.zeros(20),"xscale":np.ones(20)}]
        self.params = {
            "log_covariance_scales":TensorMock([0.,0.],True),
            "representation_logits":TensorMock([.25,-.25],True),
            "noise_logits":TensorMock([0.,1.,2.,3.],True),
            "log_grid_scale":TensorMock(.125,True),
            "log1p_gamma":TensorMock(.5,True)}
        self.buffers = {
            "noise_variance_grid":TensorMock([.25,1.,4.,16.]),
            "priors.0.mean":TensorMock(np.zeros(39)),
            "priors.0.factor":TensorMock(np.eye(39)),
            "priors.1.mean":TensorMock(np.zeros(144)),
            "priors.1.factor":TensorMock(np.eye(144))}
        self.training = True
        self.child = ModuleMock()
        self.calls = self.eval_calls = 0
        self.dynamic,self.mutation,self.output_change,self.exception = dynamic,mutation,output_change,exception
        self.fit = Mock(side_effect=AssertionError("fit forbidden"))
        self.refit = Mock(side_effect=AssertionError("refit forbidden"))
        self.calibrate = Mock(side_effect=AssertionError("calibration forbidden"))
        self.backward = Mock(side_effect=AssertionError("backward forbidden"))
        self.optimizer = Mock()
        self.optimizer.step.side_effect = AssertionError("optimizer forbidden")
        self.forward_cached = Mock(side_effect=AssertionError("internal cache entry forbidden"))
        self.make_cache = Mock(side_effect=AssertionError("external cache construction forbidden"))

    def named_parameters(self): return tuple(self.params.items())
    def named_buffers(self): return tuple(self.buffers.items())
    def state_dict(self): return {**self.params,**self.buffers}
    def modules(self): return (self,self.child)
    def eval(self):
        self.eval_calls+=1
        for module in self.modules(): module.training = False
        return self

    def predict_arm_means(self,*,x,adjacency,treatment,observed_indices,outcomes_observed):
        self.calls+=1
        assert not self.training and not self.child.training
        assert all(not value.requires_grad for value in self.params.values())
        i = np.arange(200,dtype=np.float64)
        arm = int(treatment[0])
        d = (float(treatment[0])/4+float(treatment[1])/8
             +int(observed_indices[0])/1024+float(np.rint(outcomes_observed[0]*8))/1024) if self.dynamic else 0.
        means = np.column_stack([1+i/8+2*arm,3+i/2-7*arm-2*d,
                                 8+3*i/8+5*arm+4*d,17+7*i/8+11*arm+12*d])
        if self.mutation: self.mutation(self)
        if self.exception: raise ArithmeticError("whole mock prediction failed")
        if self.output_change: return self.output_change(means)
        return means


def subset_signature(model):
    # Mock stand-in for the legacy subset reader. Intentionally excludes
    # every calibration parameter so the independent FULL hash is essential.
    return k.state_fingerprint({"specs":model.specs,"design_kinds":model.design_kinds,
                               "priors":{n:v for n,v in model.buffers.items()}})


def injection(model=None):
    model = CalibrationMock() if model is None else model
    checkpoint = {"format":k.FORMAT,"identity":model.identity,
                  "specs":copy.deepcopy(model.specs),"design_kinds":list(model.design_kinds),
                  "model_state":{name:value.numpy().copy() for name,value in model.state_dict().items()},
                  "frozen_state_sha256":subset_signature(model)}
    manifest = k.KernelNewManifest(identity=model.identity,design_kinds=tuple(model.design_kinds),
        specs_sha256=k.state_fingerprint(checkpoint["specs"]),
        model_state_sha256=k.state_fingerprint(checkpoint["model_state"]),
        frozen_state_sha256=checkpoint["frozen_state_sha256"])
    loader = Mock(return_value=(model,checkpoint))
    reader = Mock(side_effect=subset_signature)
    adapter = k.KernelNewAdapter(loader,manifest=manifest,frozen_state_reader=reader)
    return adapter,model,checkpoint,manifest,loader,reader


def expected_effects(arm, payload=None):
    d = (int(payload["treatment"][0])/4+int(payload["treatment"][1])/8
         +int(payload["observed_indices"][0])/1024
         +round(float(payload["outcomes_observed"][0])*8)/1024) if payload is not None else 0.
    return np.array([[7+node/4+3*arm+4*d,9+node/2+6*arm+8*d,
                      16+3*node/4+9*arm+12*d] for node in range(200)])


def expected_ate(arm,payload=None):
    effects = expected_effects(arm,payload)
    return np.array([sum(float(row[k]) for row in effects)/200 for k in range(3)])


class KernelNewAdapterTests(unittest.TestCase):
    def close(self,label,actual,expected):
        actual,expected = np.asarray(actual),np.asarray(expected)
        finite = np.isfinite(actual)&np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        ERRORS[label] = max(ERRORS.get(label,0.),error)
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=0,equal_nan=True)

    def test_lazy_and_not_identified_zero_loader_predict_subset_reader(self):
        adapter,model,_,_,loader,reader = injection()
        loader.assert_not_called();reader.assert_not_called()
        task,request,children = tasks_fixture(status="NOT_IDENTIFIED")
        result = adapter.estimate_pair(task,request,context_requests=children)
        loader.assert_not_called();reader.assert_not_called()
        self.assertEqual(model.calls,0)
        self.assertEqual(model.eval_calls,0)
        self.assertEqual((result.status,result.execution_status),("NOT_IDENTIFIED",)*2)
        self.assertEqual(result.failure_message,request.failure_message)
        self.assertEqual(dict(result.failure_provenance),dict(request.failure_provenance))
        for bundle in (result,*result.arm_results):
            self.assertEqual(bundle.asset_hashes,{k.CHECKPOINT:k.CHECKPOINT_SHA256})
            self.assertFalse(bundle.diagnostics["kernel_new_identity"]["runtime_checkpoint_bytes_verified"])

    def test_checkpoint_path_sha_format_and_state_order_declaration_rejected(self):
        _,_,_,manifest,_,_ = injection()
        for changes in ({"checkpoint":"other.pt"},{"checkpoint_sha256":"0"*64},
                        {"format":"other"},{"identity":"unknown"},
                        {"design_kinds":("phase4","invalid")},{"design_kinds":("phase5",)},
                        {"arm_order":("00","10","01","11")}):
            with self.subTest(changes=changes),self.assertRaises(c.ContractError):
                replace(manifest,**changes)

    def test_missing_checkpoint_contract_fields_rejected_before_predict(self):
        task,request,children = tasks_fixture()
        for field in ("format","identity","specs","design_kinds","model_state","frozen_state_sha256"):
            adapter,model,checkpoint,_,loader,_ = injection()
            checkpoint.pop(field)
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertEqual(model.calls,0)
            self.assertEqual(loader.call_count,1)

    def test_checkpoint_format_identity_specs_kinds_and_model_state_conflicts(self):
        task,request,children = tasks_fixture()
        for field,value in (("format","other"),("identity","population"),
                            ("specs",[{"kind":"linear"}]),("design_kinds",["phase4","phase4"]),
                            ("frozen_state_sha256","0"*64)):
            adapter,model,checkpoint,_,_,_ = injection()
            checkpoint[field] = value
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertEqual(model.calls,0)
        adapter,model,checkpoint,_,_,_ = injection()
        checkpoint["model_state"]["noise_logits"][0]+=1.
        result = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(model.calls,0)

    def test_model_spec_kind_identity_or_subset_conflict_before_predict(self):
        task,request,children = tasks_fixture()
        for change in (
                lambda model:setattr(model,"identity","population"),
                lambda model:model.design_kinds.__setitem__(0,"phase4"),
                lambda model:model.specs[0]["projection"].__setitem__((0,0),999.)):
            adapter,model,_,_,_,_ = injection()
            change(model)
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertEqual(model.calls,0)
        adapter,model,_,_,_,reader = injection()
        reader.side_effect = lambda model:"0"*64
        self.assertEqual(adapter.estimate_context(task.context_refs[0],children[0]).execution_status,"FAILED")
        self.assertEqual(model.calls,0)

    def test_eval_and_all_calibration_parameters_gradients_disabled(self):
        adapter,model,_,manifest,_,_ = injection()
        before = k.state_fingerprint(model.state_dict())
        self.assertTrue(all(v.requires_grad for v in model.params.values()))
        task,request,children = tasks_fixture()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"OK")
        self.assertFalse(model.training);self.assertFalse(model.child.training)
        self.assertEqual(model.eval_calls,1)
        self.assertEqual(set(model.params),{
            "log_covariance_scales","representation_logits","noise_logits","log_grid_scale","log1p_gamma"})
        for parameter in model.params.values():
            self.assertFalse(parameter.requires_grad)
            self.assertEqual(parameter.freeze_calls,1)
        self.assertEqual(before,manifest.model_state_sha256)
        self.assertEqual(k.state_fingerprint(model.state_dict()),before)
        for arm in result.arm_results:
            self.assertEqual(arm.diagnostics["point"]["full_state_before"],before)
            self.assertEqual(arm.diagnostics["point"]["full_state_after"],before)

    def test_every_calibration_parameter_mutation_detected_with_subset_unchanged(self):
        task,request,children = tasks_fixture()
        for name in ("log_covariance_scales","representation_logits","noise_logits","log_grid_scale","log1p_gamma"):
            def mutate(model):
                model.params[name].array.flat[0]+=1.
            adapter,model,_,manifest,_,_ = injection(CalibrationMock(mutation=mutate))
            subset = subset_signature(model)
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED",name)
            self.assertIn("MODEL_STATE_CHANGED",result.failure_message)
            self.assertEqual(subset_signature(model),subset)
            self.assertNotEqual(k.state_fingerprint(model.state_dict()),manifest.model_state_sha256)
            self.assertFalse(result.diagnostics["kernel_new_identity"]["legacy_subset_is_complete_state"])

    def test_every_buffer_mutation_detected_even_with_unchanged_subset_reader(self):
        task,request,children = tasks_fixture()
        names = ("noise_variance_grid","priors.0.mean","priors.0.factor","priors.1.mean","priors.1.factor")
        for name in names:
            def mutate(model):
                model.buffers[name].array.flat[0]+=.125
            adapter,model,_,manifest,_,reader = injection(CalibrationMock(mutation=mutate))
            reader.side_effect = lambda model:manifest.frozen_state_sha256
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED",name)
            self.assertIn("MODEL_STATE_CHANGED",result.failure_message)

    def test_training_or_gradient_reenable_detected_without_silent_repair(self):
        task,request,children = tasks_fixture()
        for mutation in (lambda model:setattr(model.child,"training",True),
                         lambda model:setattr(model.params["noise_logits"],"requires_grad",True)):
            adapter,model,_,_,_,_ = injection(CalibrationMock(mutation=mutation))
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertTrue("MODEL_EVAL" in result.failure_message or "MODEL_GRAD" in result.failure_message)
            self.assertEqual(model.eval_calls,1)

    def test_mutation_before_next_call_prevents_predict_and_reload(self):
        adapter,model,_,_,loader,_ = injection()
        task,request,children = tasks_fixture()
        self.assertEqual(adapter.estimate_context(task.context_refs[0],children[0]).execution_status,"OK")
        model.params["representation_logits"].array[0]+=1.
        result = adapter.estimate_context(task.context_refs[1],children[1])
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(model.calls,1)
        self.assertEqual(loader.call_count,1)

    def test_state_inventory_and_source_shape_conflicts_rejected(self):
        task,request,children = tasks_fixture()
        for mutate in (lambda model:model.params.pop("log1p_gamma"),
                       lambda model:model.buffers.update(extra_buffer=TensorMock(0.)),
                       lambda model:setattr(model.params["noise_logits"],"array",np.ones(3))):
            model = CalibrationMock();mutate(model)
            adapter,_,_,_,_,_ = injection(model)
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertEqual(model.calls,0)

    def test_complete_fingerprint_key_order_layout_and_parameter_sensitivity(self):
        _,model,_,_,_,_ = injection()
        state = model.state_dict()
        self.assertEqual(k.state_fingerprint(state),k.state_fingerprint(dict(reversed(list(state.items())))))
        self.assertEqual(k.state_fingerprint(np.arange(12.).reshape(3,4)),
                         k.state_fingerprint(np.asfortranarray(np.arange(12.).reshape(3,4))))
        self.assertEqual(k.state_fingerprint(np.array([1.,2.],dtype="<f8")),
                         k.state_fingerprint(np.array([1.,2.],dtype=">f8")))
        original = k.state_fingerprint(state)
        model.params["log_grid_scale"].array[...]+=.01
        self.assertNotEqual(k.state_fingerprint(state),original)

    def test_strict_five_arrays_float64_isolated_from_original_storage(self):
        adapter,model,_,_,_,_ = injection()
        task,request,children = tasks_fixture()
        arrays = [getattr(t,name) for t in task.context_refs for name in c.FACTUAL_KEYS]
        original = model.predict_arm_means
        def checked(**payload):
            self.assertEqual(tuple(payload),c.FACTUAL_KEYS)
            self.assertEqual(payload["outcomes_observed"].dtype,np.float64)
            for array in payload.values():
                self.assertFalse(array.flags.writeable)
                root = array
                while type(root) is np.ndarray: root=root.base
                self.assertIs(type(root),bytes)
                self.assertEqual(len(root),array.nbytes)
                for source in arrays:
                    self.assertFalse(np.shares_memory(array,source))
            return original(**payload)
        model.predict_arm_means = checked
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"OK")
        model.forward_cached.assert_not_called();model.make_cache.assert_not_called()

    def test_extra_privileged_inputs_refused_before_loader(self):
        adapter,model,_,_,loader,_ = injection()
        task,request,children = tasks_fixture()
        payload = dict(c.predictor_inputs(task.context_refs[0]))
        for name in ("truth","h","beta","full_y","noise","support","rng_identity"):
            with self.assertRaises(c.ContractError):
                adapter(**{**payload,name:np.zeros(200)})
        loader.assert_not_called()
        self.assertEqual(model.calls,0)

    def test_independent_non_symmetric_four_state_and_pair_ate_reference(self):
        adapter,_,_,_,_,_ = injection()
        task,request,children = tasks_fixture()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.close("pair_hand",result.ate,[33.375,61.75,95.125])
        self.assertEqual(result.arm_order,("00","01","10","11"))
        for arm_id,arm in enumerate(result.arm_results):
            self.close("ite_"+str(arm_id),arm.ite,expected_effects(arm_id))
            self.close("ate_"+str(arm_id),arm.ate,expected_ate(arm_id))
            self.close("mu01_"+str(arm_id),arm.mu[:,0,1],
                       [3+node/2-7*arm_id for node in range(200)])
            self.close("total_"+str(arm_id),arm.ate[2],arm.ate[0]+arm.ate[1])

    def test_nonfinite_state_effect_dependency_and_unused01(self):
        task,request,children = tasks_fixture()
        for column,statuses in ((0,("FAILED","OK","FAILED")),(1,("OK",)*3),(2,("FAILED","FAILED","OK")),
                                (3,("OK","FAILED","FAILED"))):
            def damage(means):
                means[75,column] = np.nan
                return means
            adapter,_,_,_,_,_ = injection(CalibrationMock(output_change=damage))
            result = adapter.estimate_pair(task,request,context_requests=children)
            self.assertEqual(result.output_status["ate"],statuses)
            if column==1:
                self.close("unused01",result.ate,[33.375,61.75,95.125])
                self.assertEqual(result.arm_results[0].output_status["mu"],"FAILED")
            for index,state in enumerate(statuses):
                self.assertEqual(bool(np.isfinite(result.ate[index])),state=="OK")

    def test_wrong_shape_or_dtype_complete_prediction_failure(self):
        task,request,children = tasks_fixture()
        for change in (lambda means:means[:199],lambda means:means.reshape(200,2,2),
                       lambda means:means.astype(int),lambda means:{"arm_means":means}):
            adapter,_,_,_,_,_ = injection(CalibrationMock(output_change=change))
            result = adapter.estimate_context(task.context_refs[0],children[0])
            self.assertEqual(result.execution_status,"FAILED")
            self.assertTrue(np.isnan(result.ate).all())

    def test_whole_prediction_exception_no_fake_partial_success(self):
        adapter,model,_,_,loader,_ = injection(CalibrationMock(exception=True))
        task,request,children = tasks_fixture()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"FAILED")
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        for arm in result.arm_results:
            self.assertEqual(arm.failure_message,"whole mock prediction failed")
            self.assertTrue(np.isnan(arm.mu).all())
            self.assertTrue(np.isnan(arm.ite).all())
            self.assertTrue(np.isnan(arm.ate).all())
        self.assertEqual(loader.call_count,1)
        self.assertEqual(model.calls,2)

    def test_exception_after_state_mutation_still_detected(self):
        def mutate(model): model.params["noise_logits"].array[0]+=1.
        adapter,_,_,_,_,_ = injection(CalibrationMock(exception=True,mutation=mutate))
        task,request,children = tasks_fixture()
        result = adapter.estimate_context(task.context_refs[0],children[0])
        self.assertEqual(result.execution_status,"FAILED")
        self.assertIn("MODEL_STATE_CHANGED",result.failure_message)

    def test_same_model_t0_t1_t0_dynamic_reference_and_repeat_determinism(self):
        adapter,model,_,manifest,loader,_ = injection(CalibrationMock(dynamic=True))
        task,request,children = tasks_fixture()
        results = []
        for arm in (0,1,0):
            result = adapter.estimate_context(task.context_refs[arm],children[arm])
            self.assertEqual(result.execution_status,"OK")
            payload = c.predictor_inputs(task.context_refs[arm])
            self.close("dynamic_"+str(arm),result.ate,expected_ate(arm,payload))
            results.append(result)
        self.close("dynamic_repeat",results[0].ate,results[2].ate)
        self.assertFalse(np.array_equal(results[0].ate,results[1].ate))
        self.assertEqual(model.calls,3);self.assertEqual(loader.call_count,1)
        self.assertEqual(k.state_fingerprint(model.state_dict()),manifest.model_state_sha256)
        model.forward_cached.assert_not_called();model.make_cache.assert_not_called()

    def test_full_support_and_extrapolation_inherited_elementwise(self):
        adapter,_,_,_,_,_ = injection()
        task,request,children = tasks_fixture()
        pair_reference = support_for_pair(task)
        references = tuple(support_for_task(t) for t in task.context_refs)
        result = adapter.estimate_pair(task,request,context_requests=children)
        def exact(actual,expected):
            self.assertEqual(actual.shape,expected.shape)
            self.assertEqual(actual.dtype,expected.dtype)
            np.testing.assert_array_equal(actual,expected)
        for name in ("population_support","design_support","empirical_support_mask"):
            exact(getattr(result,name),getattr(pair_reference,name))
            for arm,reference in zip(result.arm_results,references):
                exact(getattr(arm,name),getattr(reference,name))
        self.assertEqual(set(result.extrapolation_flags),{"empirical","context_design"})
        exact(result.extrapolation_flags["empirical"],pair_reference.empirical_extrapolation)
        for index,(arm,reference) in enumerate(zip(result.arm_results,references)):
            self.assertEqual(set(arm.extrapolation_flags),{"empirical","context_design"})
            exact(arm.extrapolation_flags["empirical"],reference.empirical_extrapolation)
            exact(arm.extrapolation_flags["context_design"],reference.context_design_extrapolation)
            exact(result.extrapolation_flags["context_design"][index],reference.context_design_extrapolation)
            self.assertFalse(np.array_equal(np.isfinite(arm.mu),arm.empirical_support_mask))
            query = tuple(map(int,np.argwhere(~arm.empirical_support_mask)[0]))
            with self.assertRaises(PluginSupportError):
                require_plugin_support(arm.support,(query,))

    def test_input_bytes_dtype_hash_and_bound_links_unchanged(self):
        for bound in (False,True):
            adapter,_,_,_,_,_ = injection()
            task,request,children = tasks_fixture(bound=bound)
            arrays = tuple(getattr(t,name) for t in task.context_refs for name in c.FACTUAL_KEYS)
            before = tuple((v.dtype,hashlib.sha256(v.tobytes()).hexdigest()) for v in arrays)
            hashes = tuple(t.input_hash for t in task.context_refs)
            result = adapter.estimate_pair(task,request,context_requests=children)
            self.assertEqual(before,tuple((v.dtype,hashlib.sha256(v.tobytes()).hexdigest()) for v in arrays))
            self.assertEqual(result.input_hashes,hashes)
            self.assertEqual(result.request_id,request.request_id)
            self.assertEqual(result.target_id,request.target_id)
            for t in task.context_refs: c.validate_task(t)

    def test_no_real_loader_assets_training_optimizer_backward_or_legacy_paths(self):
        adapter,model,_,_,loader,_ = injection()
        original_import = builtins.__import__
        def checked_import(name,*args,**kwargs):
            if name in ("torch","benchmark","predict") or name.startswith("pfn_social."):
                raise AssertionError("Real/legacy model path forbidden")
            return original_import(name,*args,**kwargs)
        with patch.object(builtins,"__import__",side_effect=checked_import),\
             patch.object(np,"load",side_effect=AssertionError("Asset deserialization forbidden")):
            task,request,children = tasks_fixture()
            result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(result.execution_status,"OK")
        self.assertEqual(loader.call_count,1)
        for trap in (model.fit,model.refit,model.calibrate,model.backward,
                     model.optimizer.step,model.forward_cached,model.make_cache):
            trap.assert_not_called()

    def test_loader_failure_never_retried_or_stores_context_traceback(self):
        _,model,checkpoint,manifest,_,reader = injection()
        loader = Mock(side_effect=RuntimeError("injected lazy failure"))
        adapter = k.KernelNewAdapter(loader,manifest=manifest,frozen_state_reader=reader)
        task,request,children = tasks_fixture()
        result = adapter.estimate_pair(task,request,context_requests=children)
        self.assertEqual(loader.call_count,1);reader.assert_not_called()
        self.assertEqual(result.output_status["ate"],("FAILED",)*3)
        self.assertEqual(adapter._load_error,("RuntimeError","injected lazy failure"))

    def test_identity_or_input_hash_conflict_precedes_loading(self):
        adapter,_,_,_,loader,_ = injection()
        task,request,children = tasks_fixture()
        bad = replace(request,input_hashes=("0"*64,)*2)
        with self.assertRaises(c.ContractError):
            adapter.estimate_pair(task,bad,context_requests=children)
        loader.assert_not_called()


def tearDownModule():
    print("S4.3B_REFERENCE_RESIDUALS "+json.dumps(ERRORS,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
