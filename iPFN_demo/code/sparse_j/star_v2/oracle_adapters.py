"""Oracle-only request bridges; no ordinary callback bypass or new state codes."""
from collections.abc import Mapping
import numpy as np

from .contracts import TaskSpec, PairTaskSpec, predictor_inputs
from .estimation import (EstimateBundle, PairEstimateBundle, IntervalRecord,
    SamplingUncertainty, validate_binding, metadata, EFFECT_ORDER)
from .support import method_capability
from .model_adapters import context_requests_from_parent, _validate_children
from .oracle_context import OracleContextResult, estimate_oracle_context as _context_core
from .oracle_pair import OraclePairResult, estimate_oracle_pair as _pair_core
from .oracle_numerics import noise_scale



def _json_scalars(value):
    # Numerical diagnostics contain NumPy scalars; arrays/objects stay forbidden.
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,Mapping):
        return {k:_json_scalars(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return tuple(_json_scalars(v) for v in value)
    return value


def _h(value):
    if (type(value) is not np.ndarray or value.shape != (200,3)
            or value.dtype.kind not in "fi" or not np.isfinite(value).all()):
        raise ValueError("Explicit finite true_h[200,3] required, never a simulator object")
    copy = np.array(value,dtype=np.float64,copy=True)
    return np.frombuffer(copy.tobytes(),dtype=np.float64).reshape(200,3)


def _failure(task,request,method,reason=None):
    cap = method_capability(method)
    statuses = {}
    for name in request.requested_outputs:
        key = request.scope+"."+("ate_interval" if name == "interval" else name)
        if key in cap.outputs:
            statuses[name] = "FAILED" if name == "mu" else ("FAILED",)*3
    identified = request.status != "NOT_IDENTIFIED"
    source = "joint_model_posterior" if method == "oracle-pair" else "model_posterior"
    intervals = tuple(IntervalRecord(
        None if request.scope == "pair" else request.context_ids[0],effect,"FAILED",source=source)
        for effect in EFFECT_ORDER) if "interval" in statuses else ()
    outputs = dict(output_status=statuses,intervals=intervals,
        failure_type="ORACLE_BRIDGE_FAILURE" if identified else request.failure_type,
        failure_message=reason if identified else request.failure_message,
        failure_provenance={"stage":"oracle_bridge","attempts":1,"reason":reason}
            if identified else request.failure_provenance)
    cls = EstimateBundle if request.scope == "context" else PairEstimateBundle
    if request.scope == "pair":
        outputs["arm_results"] = (None,None)
    return cls.from_outputs(request,task,method,**outputs)


def _field(value,shape,valid,scalar=False):
    out = np.full(shape,np.nan,dtype=np.float64)
    states = ["FAILED"]*(1 if scalar else 3)
    if valid and type(value) is np.ndarray and value.dtype == np.dtype("float64") and value.shape == shape:
        for k in range(len(states)):
            part = value if scalar else value[...,k]
            if np.isfinite(part).all():
                states[k] = "OK"
                if scalar: out[:] = value
                else: out[...,k] = part
    return out,states[0] if scalar else tuple(states)


def _mapped(task,request,method,raw):
    joint = method == "oracle-pair"
    if type(raw) is not (OraclePairResult if joint else OracleContextResult):
        raise TypeError("Expected the typed Oracle numerical result")
    mean = raw.posterior_mean
    valid = (raw.point_status == "OK" and type(mean) is np.ndarray
             and mean.shape == (25,) and mean.dtype == np.dtype("float64")
             and np.isfinite(mean).all())
    values,states = {},{}
    ate,ate_states = _field(raw.ate,(3,),valid)
    for name,shape,scalar in (("mu",(200,2,2),True),("ite",(200,3),False),("ate",(3,),False)):
        if name not in request.requested_outputs:
            continue
        if request.scope+"."+name not in method_capability(method).outputs:
            continue
        value,status = (ate,ate_states) if name == "ate" else _field(getattr(raw,name),shape,valid,scalar)
        values[name],states[name] = value,status
    source,unit = ("joint_model_posterior","pair") if joint else ("model_posterior","context")
    interval_states,records,interval_reasons = [],[],{}
    uncertainty = None
    if "interval" in request.requested_outputs:
        if type(raw.intervals) is not tuple:
            raise TypeError("Oracle intervals must be an explicit tuple")
        lookup = {}
        for record in raw.intervals:
            if record.effect in lookup:
                raise ValueError("Duplicate Oracle interval effect")
            lookup[record.effect] = record
        for k,effect in enumerate(EFFECT_ORDER):
            record = lookup.get(effect)
            reason = getattr(record,"failure_reason",None) if record else "Missing Oracle interval"
            ok = (record is not None and record.status == "OK" and ate_states[k] == "OK"
                  and record.source == source and record.statistical_unit == unit
                  and record.level == .95
                  and type(record.lower) in (int,float) and type(record.upper) in (int,float)
                  and np.isfinite([record.lower,record.upper]).all() and record.lower <= record.upper)
            state = "OK" if ok else "FAILED"
            if not ok:
                reason = reason or ("Necessary point effect unavailable" if ate_states[k] != "OK"
                                    else "Invalid Oracle interval status/source/scope/endpoints")
                interval_reasons[effect] = reason
            records.append(IntervalRecord(None if joint else request.context_ids[0],effect,state,
                float(record.lower) if ok else None,float(record.upper) if ok else None,source=source))
            interval_states.append(state)
        states["interval"] = tuple(interval_states)
        # Non-J unavailable uncertainty must contain no provenance metadata.
        uncertainty = (SamplingUncertainty("OK",source,unit,None)
                       if "OK" in interval_states else SamplingUncertainty("FAILED"))
    diagnostics = {"oracle_numerics":dict(raw.diagnostics if joint else raw.covariance_diagnostics),
                   "internal_status":raw.status if joint else raw.point_status,
                   "point_status":raw.point_status,"reference_only":True}
    if joint and raw.constraint is not None:
        diagnostics["constraint"] = dict(raw.constraint.diagnostics)
        if raw.constraint.singular_values is not None:
            diagnostics["constraint"]["singular_values"] = tuple(float(v) for v in raw.constraint.singular_values)
    provenance = {"oracle_failure_reasons":dict(raw.failure_reasons)}
    if interval_reasons:
        provenance["intervals"] = interval_reasons
        provenance["uncertainty_origin"] = {"source":source,"statistical_unit":unit,
                                           "effective_sample_count":None}
    # Reject privileged/non-JSON metadata rather than serializing objects/repr.
    diagnostics,provenance = metadata(_json_scalars(diagnostics)),metadata(_json_scalars(provenance))
    failed = any(v == "FAILED" for state in states.values()
                 for v in ((state,) if type(state) is str else state))
    kwargs = dict(**values,output_status=states,intervals=tuple(records),
        sampling_uncertainty=uncertainty,diagnostics=diagnostics,
        estimability={effect:{"status":ate_states[k],
            "reason":"finite_oracle_point" if ate_states[k] == "OK" else "Oracle point unavailable"}
            for k,effect in enumerate(EFFECT_ORDER)},
        failure_type="ORACLE_OUTPUT_FAILURE" if failed else None,
        failure_message="One or more requested Oracle outputs unavailable" if failed else None,
        failure_provenance=provenance)
    cls = PairEstimateBundle if joint else EstimateBundle
    if joint: kwargs["arm_results"] = (None,None)
    return cls.from_outputs(request,task,method,**kwargs)


def estimate_oracle_context(task,request,*,true_h,noise_sd,core=None):
    """One public Context request plus explicit permitted privileged knowledge."""
    if type(task) is not TaskSpec: raise TypeError("Expected public TaskSpec")
    validate_binding(request,task)
    if request.status == "NOT_IDENTIFIED":
        return _failure(task,request,"oracle-context")
    hidden,sigma = _h(true_h),noise_scale(noise_sd)
    inputs = predictor_inputs(task)
    callback = _context_core if core is None else core
    try:
        raw = callback(**inputs,true_h=hidden,noise_sd=sigma,context_identity={
            "context_id":task.context_id,"pair_id":task.pair_id,"request_id":request.request_id,
            "target_id":request.target_id,"protocol_hash":request.protocol_hash})
        result = _mapped(task,request,"oracle-context",raw)
    except Exception as exc:
        result = _failure(task,request,"oracle-context",type(exc).__name__+": "+str(exc))
    validate_binding(request,task)
    return result


def estimate_oracle_pair(task,request,*,true_h,noise_sd,core=None):
    """Joint Oracle-pair, never two invented independent context estimates."""
    if type(task) is not PairTaskSpec: raise TypeError("Expected public PairTaskSpec")
    validate_binding(request,task)
    if request.status == "NOT_IDENTIFIED":
        return _failure(task,request,"oracle-pair")
    hidden,sigma = _h(true_h),noise_scale(noise_sd)
    a,b = tuple(predictor_inputs(t) for t in task.context_refs)
    callback = _pair_core if core is None else core
    try:
        raw = callback(true_h=hidden,noise_sd=sigma,adjacency=a["adjacency"],
            treatment0=a["treatment"],treatment1=b["treatment"],
            observed_indices0=a["observed_indices"],observed_indices1=b["observed_indices"],
            outcomes_observed0=a["outcomes_observed"],outcomes_observed1=b["outcomes_observed"],
            pair_identity={"pair_id":task.pair_id,"context_ids":request.context_ids,
                "request_id":request.request_id,"target_id":request.target_id,
                "protocol_hash":request.protocol_hash})
        result = _mapped(task,request,"oracle-pair",raw)
    except Exception as exc:
        result = _failure(task,request,"oracle-pair",type(exc).__name__+": "+str(exc))
    validate_binding(request,task)
    return result


def estimate_oracle_context_pair(task,request,*,true_h,noise_sd,
                                 context_requests=None,request_ids=None,core=None):
    """Two independent posterior calls; only corresponding point ATEs are averaged."""
    if type(task) is not PairTaskSpec: raise TypeError("Expected public PairTaskSpec")
    if context_requests is None:
        context_requests = context_requests_from_parent(task,request,request_ids=request_ids)
    elif request_ids is not None:
        raise TypeError("Supply explicit context_requests or request_ids, not both")
    _validate_children(task,request,context_requests)
    if "ate" not in request.requested_outputs:
        raise ValueError("Context-derived Pair requires requested ATE")
    arms = tuple(estimate_oracle_context(t,context_requests[k],
        true_h=true_h,noise_sd=noise_sd,core=core) for k,t in enumerate(task.context_refs))
    values = np.full(3,np.nan)
    states = []
    for k in range(3):
        arm_states = tuple(arm.output_status["ate"][k] for arm in arms)
        state = ("FAILED" if request.status == "NOT_IDENTIFIED" else
                 "OK" if arm_states == ("OK","OK") else
                 "FAILED" if "FAILED" in arm_states else "NOT_ESTIMABLE")
        if state == "OK":
            values[k] = sum(arm.ate[k]/2 for arm in arms)
            if not np.isfinite(values[k]): state = "FAILED";values[k] = np.nan
        states.append(state)
    ni = request.status == "NOT_IDENTIFIED"
    failed = any(state != "OK" for state in states)
    provenance = request.failure_provenance if ni else {
        "arms":tuple({"context_id":a.context_ids[0],"execution_status":a.execution_status,
                      "output_status":dict(a.output_status),"failure_provenance":a.failure_provenance,
                      "failure_type":a.failure_type,"failure_message":a.failure_message} for a in arms)}
    return PairEstimateBundle.from_outputs(request,task,"oracle-context",
        ate=values,output_status={"ate":tuple(states)},arm_results=arms,
        diagnostics={"aggregation":"mean_of_two_context_effects","arms_independent":False,
                     "interval_scope":"context_only"},
        failure_type=request.failure_type if ni else "NECESSARY_ARM_FAILURE" if failed else None,
        failure_message=request.failure_message if ni else "Necessary context ATE unavailable" if failed else None,
        failure_provenance=provenance)
