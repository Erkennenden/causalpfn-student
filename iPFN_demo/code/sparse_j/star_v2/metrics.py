"""Pure deterministic metrics for frozen section 9; no ranking/coverage/bootstrap.

Plans come from sealed ledger identities, never successful predictions. Explicit
synthetic plans allow small tests without weakening production TaskSpec rules.
None is N/A. Failed public Bundle NaNs are unavailable sentinels, never errors
used in arithmetic. Duplicate attempts need upstream selection, never best-of.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .task_registry import TaskRegistry
from .estimation import (EstimateBundle, PairEstimateBundle, EFFECT_ORDER,
                         OUTPUT_STATES, EXECUTION_STATES, metadata)
from .support import method_capability, ContextSupport, REASONS

GROUPS = ("center","leaf","all")
NODE_METHODS = ("kernel-old","kernel-new","j","oracle-context")
PAIR_FIELDS = ("pooled_rmse","signed_bias","absolute_bias","absolute_conditional_bias",
               "within_mechanism_variance","between_mechanism_variability")


def _check(condition,message):
    if not condition: raise ValueError(message)


def _immutable(value):
    if isinstance(value,Mapping):
        return MappingProxyType({k:_immutable(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):
        return tuple(_immutable(v) for v in value)
    return value


def _copy(value,shape,name,finite=True):
    _check(type(value) is np.ndarray and value.dtype == np.dtype("float64")
           and value.shape == shape,"Invalid float64 shape/dtype: "+name)
    if finite: _check(np.isfinite(value).all(),"Nonfinite valid data: "+name)
    return np.frombuffer(value.tobytes(order="C"),dtype=np.float64).reshape(shape)


@dataclass(frozen=True,slots=True)
class ExpectedTask:
    key: tuple  # unchanged (method_id, scope, task_id, request_id)
    pair_id: str
    mechanism_id: int
    pair_repeat_id: int
    center_t: int | None
    input_hashes: tuple
    node_ids: tuple
    requested_outputs: tuple
    identification_status: str = "POINT"

    def __post_init__(self):
        _check(type(self.key) is tuple and len(self.key) == 4
               and all(type(v) is str and v for v in self.key),"Invalid execution key")
        method,scope,task_id,_ = self.key
        cap = method_capability(method)
        _check(cap.method_id == method and scope in ("pair","context"),"Invalid method/scope")
        _check(type(self.pair_id) is str and bool(self.pair_id),"Missing pair_id")
        _check(all(type(v) is int and v >= 0 for v in (self.mechanism_id,self.pair_repeat_id)),
               "Invalid mechanism/repeat")
        _check((scope == "pair" and self.center_t is None and task_id == self.pair_id)
               or (scope == "context" and type(self.center_t) is int and self.center_t in (0,1)),
               "Pair/context identity conflict")
        _check(type(self.node_ids) is tuple and len(self.node_ids) >= 2
               and all(type(v) is int and v == i for i,v in enumerate(self.node_ids)),"Node order mismatch")
        _check(type(self.input_hashes) is tuple and len(self.input_hashes) == (2 if scope == "pair" else 1)
               and all(type(v) is str and len(v) == 64
                       and all(c in "0123456789abcdef" for c in v) for v in self.input_hashes),
               "Invalid input hashes")
        _check(type(self.requested_outputs) is tuple and self.requested_outputs
               and len(set(self.requested_outputs)) == len(self.requested_outputs)
               and set(self.requested_outputs) <= {"mu","ite","ate","interval"},"Invalid requested outputs")
        _check(self.identification_status in ("POINT","NOT_IDENTIFIED"),"Invalid Identification")


@dataclass(frozen=True,slots=True)
class EvaluationPlan:
    tasks: tuple
    synthetic: bool = False
    source_seal: str | None = None

    def __post_init__(self):
        _check(type(self.tasks) is tuple and self.tasks,"Explicit nonempty expected universe required")
        _check(all(type(t) is ExpectedTask for t in self.tasks),"ExpectedTask records required")
        _check(len({t.key for t in self.tasks}) == len(self.tasks),"Duplicate expected execution identity")
        n = self.tasks[0].node_ids
        _check(all(t.node_ids == n for t in self.tasks),"Inconsistent node order")
        _check(self.synthetic or len(n) == 200,"Production node budget remains 200")
        _check(self.synthetic or type(self.source_seal) is str and len(self.source_seal) == 64,
               "Production plan requires sealed registry provenance")
        slots = set();pair_ids = {};context_ids = {}
        for t in self.tasks:
            slot = (t.key[0],t.key[1],t.mechanism_id,t.pair_repeat_id,t.center_t)
            _check(slot not in slots,"Ambiguous repeated point request; explicit upstream selection required")
            slots.add(slot)
            pair_slot = (t.mechanism_id,t.pair_repeat_id)
            _check(pair_slot not in pair_ids or pair_ids[pair_slot] == t.pair_id,"Pair association conflict")
            pair_ids[pair_slot] = t.pair_id
            if t.key[1] == "context":
                context_slot = pair_slot+(t.center_t,)
                _check(context_slot not in context_ids or context_ids[context_slot] == t.key[2],
                       "Context identity conflict")
                context_ids[context_slot] = t.key[2]
        for method,_,m,r,center in tuple(slots):
            if center is not None:
                _check((method,"context",m,r,1-center) in slots,"Expected context arm missing")

    @classmethod
    def from_registry(cls,registry):
        _check(type(registry) is TaskRegistry and registry.sealed,"Sealed TaskRegistry required")
        tasks = []
        for key,record in registry.requests.items():
            request = record.document["request"]
            if not set(request["requested_outputs"]) & {"mu","ite","ate"}:
                continue  # interval-only requests are not point-metric denominators
            data = registry.tasks[key[2]].document
            ident = data["identity"]
            tasks.append(ExpectedTask(key,data["pair_id"],ident["mechanism_id"],ident["pair_repeat_id"],
                ident["center_t"][0] if key[1] == "context" else None,
                tuple(data["input_hashes"]),tuple(data["node_ids"]),
                tuple(request["requested_outputs"]),request["status"]))
        return cls(tuple(tasks),source_seal=registry.seal_sha256)


@dataclass(frozen=True,slots=True)
class Prediction:
    key: tuple
    input_hashes: tuple
    node_ids: tuple
    output_status: Mapping
    mu: np.ndarray | None = None
    ite: np.ndarray | None = None
    ate: np.ndarray | None = None
    failure_reasons: Mapping = None
    identification_status: str = "POINT"
    execution_status: str = "OK"

    def __post_init__(self):
        _check(type(self.key) is tuple and len(self.key) == 4
               and all(type(v) is str and bool(v) for v in self.key), "Invalid prediction key")
        _check(type(self.node_ids) is tuple and len(self.node_ids) >= 2
               and all(type(v) is int and v == i for i,v in enumerate(self.node_ids)),
               "Prediction node order mismatch")
        _check(isinstance(self.output_status,Mapping)
               and set(self.output_status) == {"mu","ite","ate"},"Complete point output statuses required")
        status = dict(self.output_status)
        _check(type(status["mu"]) is str and status["mu"] in OUTPUT_STATES,"Invalid mu status")
        for name in ("ite","ate"):
            _check(type(status[name]) is tuple and len(status[name]) == 3
                   and all(s in OUTPUT_STATES for s in status[name]),"Invalid effect statuses")
        _check(self.execution_status in EXECUTION_STATES,"Invalid execution status")
        _check(self.identification_status in ("POINT","NOT_IDENTIFIED"),"Invalid Identification")
        n = len(self.node_ids)
        for name,shape in (("mu",(n,2,2)),("ite",(n,3)),("ate",(3,))):
            value = getattr(self,name)
            states = (status[name],) if name == "mu" else status[name]
            if value is None:
                _check("OK" not in states,"Missing OK field: "+name)
                continue
            copy = _copy(value,shape,name,finite=False)
            for k,s in enumerate(states):
                part = copy if name == "mu" else copy[...,k]
                if s == "OK": _check(np.isfinite(part).all(),"Nonfinite OK output: "+name)
                else: _check(np.isnan(part).all(),"Unavailable output needs None/NaN sentinel: "+name)
            object.__setattr__(self,name,copy)
        if self.identification_status == "NOT_IDENTIFIED":
            _check(not any(s == "OK" for v in status.values()
                           for s in ((v,) if type(v) is str else v)),"NI cannot contain valid predictions")
        object.__setattr__(self,"output_status",_immutable(status))
        object.__setattr__(self,"failure_reasons",metadata(self.failure_reasons or {}))

    @classmethod
    def from_bundle(cls,bundle):
        _check(type(bundle) in (EstimateBundle,PairEstimateBundle),"Public Bundle required")
        return cls((bundle.method_id,bundle.request.scope,bundle.task_id,bundle.request_id),
            bundle.input_hashes,tuple(int(v) for v in bundle.node_ids),
            {k:bundle.output_status[k] for k in ("mu","ite","ate")},
            bundle.mu,bundle.ite,bundle.ate,
            {"failure_type":bundle.failure_type,"failure_message":bundle.failure_message,
             "provenance":bundle.failure_provenance},bundle.status,bundle.execution_status)


@dataclass(frozen=True,slots=True)
class MechanismTruth:
    mechanism_id: int
    ate: np.ndarray
    mu: np.ndarray | None = None
    ite: np.ndarray | None = None

    def __post_init__(self):
        _check(type(self.mechanism_id) is int and self.mechanism_id >= 0,"Invalid truth mechanism")
        object.__setattr__(self,"ate",_copy(self.ate,(3,),"truth.ate"))
        for name in ("mu","ite"):
            value = getattr(self,name)
            if value is not None:
                _check(type(value) is np.ndarray and value.ndim == (3 if name == "mu" else 2),
                       "Truth node shape mismatch")
                shape = (len(value),2,2) if name == "mu" else (len(value),3)
                object.__setattr__(self,name,_copy(value,shape,"truth."+name))


def _index(plan,predictions,truths):
    _check(type(plan) is EvaluationPlan,"Explicit EvaluationPlan required")
    expected = {t.key:t for t in plan.tasks};found = {}
    for record in predictions:
        _check(type(record) is Prediction and record.key in expected,"Extra/invalid prediction identity")
        _check(record.key not in found,"Duplicate prediction identity")
        task = expected[record.key]
        _check(record.node_ids == task.node_ids and record.input_hashes == task.input_hashes,
               "Prediction input identity mismatch")
        _check(record.identification_status == task.identification_status,"Identification mismatch")
        for name,status in record.output_status.items():
            active = name in task.requested_outputs and task.key[1]+"."+name in method_capability(task.key[0]).outputs
            _check(active or all(s != "OK" for s in ((status,) if type(status) is str else status)),
                   "Unrequested/unsupported prediction capability")
        found[record.key] = record
    actual_truths = {}
    for truth in truths:
        _check(type(truth) is MechanismTruth and truth.mechanism_id not in actual_truths,"Duplicate/invalid truth")
        actual_truths[truth.mechanism_id] = truth
    _check(set(actual_truths) == {t.mechanism_id for t in plan.tasks},"Missing or extra mechanism truth")
    for truth in actual_truths.values():
        for name in ("mu","ite"):
            value = getattr(truth,name)
            _check(value is None or len(value) == len(plan.tasks[0].node_ids),"Truth node order/shape mismatch")
    return found,actual_truths


def _valid(record,name,k=None):
    if record is None: return False
    status = record.output_status[name]
    return (status if k is None else status[k]) == "OK"


def _failure(task,record,name,k=None):
    status = "MISSING" if record is None else (
        record.output_status[name] if k is None else record.output_status[name][k])
    return {"key":task.key,"status":status,
            "reason":"Missing prediction record" if record is None else record.failure_reasons}


def _numeric(rows,pair_variance):
    groups = {}
    for row in rows: groups.setdefault(row["mechanism_id"],[]).append(row)
    if not groups: return None,()
    mechanism = []
    for m,values in groups.items():
        errors = np.array([r["error"] for r in values])
        mse = float(np.mean([r.get("mse",r["error"]**2) for r in values]))
        mechanism.append({"mechanism_id":m,"n_valid":len(values),"mean_error":float(errors.mean()),
                          "mse":mse,"within_variance":float(errors.var(ddof=1)) if len(values)>1 else None})
    means = np.array([r["mean_error"] for r in mechanism])
    mse = float(np.mean([r["mse"] for r in mechanism]))
    _check(np.isfinite([mse,*means]).all(),"Nonfinite metric calculation")
    between = float(means.var(ddof=1)) if len(means)>1 else None
    values = {"mse":mse,"pooled_rmse":float(np.sqrt(mse)),"signed_bias":float(means.mean()),
              "absolute_bias":float(abs(means.mean())),"absolute_conditional_bias":float(np.abs(means).mean()),
              "between_mechanism_standard_deviation":float(np.sqrt(between)) if between is not None else None}
    if pair_variance:
        values["within_mechanism_variance"] = (float(np.mean([r["within_variance"] for r in mechanism]))
            if all(r["within_variance"] is not None for r in mechanism) else None)
        values["between_mechanism_variability"] = between
    return values,tuple(mechanism)


def _summary(expected,rows,failures,pair_variance=False):
    values,mechanism = _numeric(rows,pair_variance)
    complete = len(rows) == len(expected) and bool(expected)
    names = ("mse","pooled_rmse","signed_bias","absolute_bias","absolute_conditional_bias",
             "between_mechanism_standard_deviation")+(
                 ("within_mechanism_variance","between_mechanism_variability") if pair_variance else ())
    out = {"status":"COMPLETE" if complete else "INCOMPLETE" if expected else "NOT_REQUESTED",
           "n_requested":len(expected),"n_valid":len(rows),"n_failed_or_missing":len(failures),
           "n_mechanisms_requested":len({t.mechanism_id for t in expected}),
           "n_mechanisms_with_valid":len(mechanism),"raw_rows":tuple(rows),
           "failures":tuple(failures),"per_mechanism":mechanism}
    out.update({name:values[name] if complete else None for name in names})
    out.update({"success_subset_"+name:values[name] if values is not None else None for name in names})
    return out


def _report(): return {"ordinary":{},"oracle_reference":{}}
def _bucket(method): return "oracle_reference" if method_capability(method).reference_only else "ordinary"
def _methods(plan): return tuple(dict.fromkeys(t.key[0] for t in plan.tasks))
def _group_nodes(n,group): return (0,) if group == "center" else tuple(range(1,n)) if group == "leaf" else tuple(range(n))


def pair_ate_metrics(plan,predictions,truths):
    found,true = _index(plan,predictions,truths);out = _report()
    for method in _methods(plan):
        tasks = [t for t in plan.tasks if t.key[0] == method and t.key[1] == "pair" and "ate" in t.requested_outputs]
        effects = {}
        for k,effect in enumerate(EFFECT_ORDER):
            rows=[];failures=[]
            for task in tasks:
                record = found.get(task.key)
                if _valid(record,"ate",k):
                    error = float(record.ate[k]-true[task.mechanism_id].ate[k])
                    rows.append({"key":task.key,"mechanism_id":task.mechanism_id,
                                 "pair_repeat_id":task.pair_repeat_id,"error":error})
                else: failures.append(_failure(task,record,"ate",k))
            effects[effect] = _summary(tasks,rows,failures,True)
        out[_bucket(method)][method] = effects
    return _immutable(out)


def node_metrics(plan,predictions,truths):
    found,true = _index(plan,predictions,truths);out = _report()
    for method in _methods(plan):
        if method not in NODE_METHODS:
            out[_bucket(method)][method] = {"status":"UNSUPPORTED"}
            continue
        contexts = [t for t in plan.tasks if t.key[0] == method and t.key[1] == "context"]
        groups = {}
        for group in GROUPS:
            nodes = _group_nodes(len(plan.tasks[0].node_ids),group)
            groups[group] = {}
            for name in ("mu","ite"):
                tasks = [t for t in contexts if name in t.requested_outputs]
                effects = {}
                for k,effect in ((None,"mu"),) if name == "mu" else enumerate(EFFECT_ORDER):
                    rows=[];failures=[];group_rows=[]
                    for task in tasks:
                        truth = getattr(true[task.mechanism_id],name)
                        _check(truth is not None,"Missing requested node truth: "+name)
                        record = found.get(task.key)
                        if _valid(record,name,k):
                            predicted = getattr(record,name)[list(nodes)]
                            actual = truth[list(nodes)]
                            if k is not None: predicted,actual = predicted[:,k],actual[:,k]
                            errors = predicted-actual
                            row = {"key":task.key,"mechanism_id":task.mechanism_id,
                                   "pair_repeat_id":task.pair_repeat_id,"center_t":task.center_t,
                                   "error":float(errors.mean()),"mse":float(np.mean(errors**2))}
                            rows.append(row)
                            if k is not None:
                                group_rows.append({**row,"error":float(predicted.mean()-actual.mean()),
                                    "mse":float((predicted.mean()-actual.mean())**2),
                                    "predicted_group_effect":float(predicted.mean()),
                                    "true_group_effect":float(actual.mean())})
                        else: failures.append(_failure(task,record,name,k))
                    effects[effect] = _summary(tasks,rows,failures)
                    if name == "ite":
                        groups[group].setdefault("group_ate",{})[effect] = _summary(tasks,group_rows,failures)
                groups[group][name] = effects["mu"] if name == "mu" else effects
        out[_bucket(method)][method] = {"groups":groups}
    return _immutable(out)


def center_leaf_metrics(plan,predictions,truths):
    found,true = _index(plan,predictions,truths);out = _report()
    for method in _methods(plan):
        if method not in NODE_METHODS:
            out[_bucket(method)][method] = {"status":"UNSUPPORTED"}
            continue
        tasks = [t for t in plan.tasks if t.key[0] == method and t.key[1] == "context"
                 and "ite" in t.requested_outputs]
        rows_by_effect = {effect:[] for effect in EFFECT_ORDER}
        failed_by_effect = {effect:[] for effect in EFFECT_ORDER}
        raw_records = []
        for task in tasks:
            truth = true[task.mechanism_id].ite
            _check(truth is not None,"Missing ITE truth")
            record = found.get(task.key);effects = {}
            for k,effect in enumerate(EFFECT_ORDER):
                true_center,true_leaf = float(truth[0,k]),float(truth[1:,k].mean())
                true_delta = true_center-true_leaf
                valid = _valid(record,"ite",k)
                center = float(record.ite[0,k]) if valid else None
                leaf = float(record.ite[1:,k].mean()) if valid else None
                predicted_delta = center-leaf if valid else None
                error = predicted_delta-true_delta if valid else None
                effects[effect] = {"true_center":true_center,"true_leaf":true_leaf,
                    "predicted_center":center,"predicted_leaf":leaf,"true_difference":true_delta,
                    "predicted_difference":predicted_delta,"error":error}
                if valid:
                    rows_by_effect[effect].append({"key":task.key,"mechanism_id":task.mechanism_id,
                        "pair_repeat_id":task.pair_repeat_id,"center_t":task.center_t,
                        "error":error,**effects[effect]})
                else: failed_by_effect[effect].append(_failure(task,record,"ite",k))
            raw_records.append({"key":task.key,"mechanism_id":task.mechanism_id,"effects":effects})
        out[_bucket(method)][method] = {
            "effects":{e:_summary(tasks,rows_by_effect[e],failed_by_effect[e]) for e in EFFECT_ORDER},
            "context_records":raw_records}
    return _immutable(out)


def _support_index(plan,supports):
    expected = {t.key:t for t in plan.tasks if t.key[0] in NODE_METHODS
                and t.key[1] == "context" and "mu" in t.requested_outputs}
    found = {}
    for key,s in supports:
        _check(key in expected and key not in found,"Extra/duplicate Support identity")
        t = expected[key];n = len(t.node_ids)
        _check(type(s) is ContextSupport and s.context_id == key[2]
               and s.source_input_hash == t.input_hashes[0] and s.center_t == t.center_t
               and tuple(int(v) for v in s.node_ids) == t.node_ids,"Support binding conflict")
        _check(s.arm_order == ("00","01","10","11") and s.group_order == ("center","leaf"),
               "Support axis ordering conflict")
        for name in ("population_support","design_support","realized_state_mask","empirical_support_mask"):
            v = getattr(s,name)
            _check(type(v) is np.ndarray and v.shape == (n,2,2) and v.dtype == np.dtype("bool"),
                   "Invalid Support mask")
        _check(s.observed_node_mask.shape == (n,) and s.observed_node_mask.dtype == np.dtype("bool"),
               "Invalid observation mask")
        _check(s.support_reason.shape == (n,2,2) and set(np.unique(s.support_reason)) <= set(REASONS),
               "Illegal Support classification")
        _check(s.population_support.all() and np.all(s.realized_state_mask.sum((1,2)) == 1),
               "Invalid population/realized Support")
        _check(np.array_equal(s.empirical_support_mask,s.realized_state_mask&s.observed_node_mask[:,None,None]),
               "Empirical Support conflict")
        # Validate the supplied Support's existing partition, never recompute
        # exposure/design from treatment or predictions.
        partitions = (~s.design_support,
            s.design_support&~s.realized_state_mask,
            s.design_support&s.realized_state_mask&~s.observed_node_mask[:,None,None],
            s.empirical_support_mask)
        for reason,mask in zip(REASONS,partitions):
            _check(np.array_equal(s.support_reason == reason,mask),"Support reason/mask conflict")
        full = np.stack((s.realized_state_mask[:1].sum(0),s.realized_state_mask[1:].sum(0)))
        observed = np.stack((s.empirical_support_mask[:1].sum(0),s.empirical_support_mask[1:].sum(0)))
        _check(s.full_graph_count.shape == (2,2,2) and s.full_graph_count.dtype == np.dtype("int64")
               and np.array_equal(full,s.full_graph_count),"Full graph count conflict")
        _check(s.observed_count.shape == (2,2,2) and s.observed_count.dtype == np.dtype("int64")
               and np.array_equal(observed,s.observed_count),"Observed count conflict")
        _check(s.center_observed == bool(s.observed_node_mask[0]),"Center observed flag conflict")
        found[key] = s
    _check(set(found) == set(expected),"Missing pre-requested Support")
    return found


def support_stratification(plan,predictions,truths,supports):
    found,true = _index(plan,predictions,truths)
    supplied = _support_index(plan,supports)
    out = _report()
    for method in _methods(plan):
        if method not in NODE_METHODS:
            out[_bucket(method)][method] = {"status":"UNSUPPORTED"}
            continue
        tasks = [t for t in plan.tasks if t.key[0] == method and t.key[1] == "context"
                 and "mu" in t.requested_outputs]
        if not tasks:
            out[_bucket(method)][method] = {"status":"NOT_REQUESTED"}
            continue
        queries=[];counts=[]
        for task in tasks:
            s = supplied[task.key];record = found.get(task.key)
            truth = true[task.mechanism_id].mu
            _check(truth is not None,"Missing mu truth")
            valid = _valid(record,"mu")
            counts.append({"key":task.key,"mechanism_id":task.mechanism_id,"center_t":task.center_t,
                "center_observed":s.center_observed,"full_graph_count":s.full_graph_count.tolist(),
                "observed_count":s.observed_count.tolist()})
            for i in range(len(task.node_ids)):
                for t in (0,1):
                    for exposure in (0,1):
                        error = float(record.mu[i,t,exposure]-truth[i,t,exposure]) if valid else None
                        queries.append({"key":task.key,"mechanism_id":task.mechanism_id,
                            "center_t":task.center_t,"node_id":i,"t":t,"s":exposure,
                            "group":"center" if i == 0 else "leaf",
                            "reason":str(s.support_reason[i,t,exposure]),"error":error,
                            "status":"OK" if valid else "MISSING" if record is None else record.output_status["mu"],
                            "failure_reason":None if valid else _failure(task,record,"mu")["reason"]})
        strata = []
        for group in GROUPS:
            for center in (None,0,1):
                for reason in REASONS:
                    selected = [r for r in queries if (group == "all" or r["group"] == group)
                                and (center is None or r["center_t"] == center) and r["reason"] == reason]
                    per_mechanism = []
                    for m in dict.fromkeys(t.mechanism_id for t in tasks):
                        rows = [r for r in selected if r["mechanism_id"] == m]
                        errors = [r["error"] for r in rows if r["status"] == "OK"]
                        mse = float(np.mean(np.square(errors))) if errors else None
                        bias = float(np.mean(errors)) if errors else None
                        per_mechanism.append({"mechanism_id":m,"n_requested":len(rows),
                            "n_valid":len(errors),"n_failed_or_missing":len(rows)-len(errors),
                            "mse":mse,"signed_bias":bias})
                    valid_mechanisms = [r for r in per_mechanism if r["n_valid"] > 0]
                    mse = float(np.mean([r["mse"] for r in valid_mechanisms])) if valid_mechanisms else None
                    bias = float(np.mean([r["signed_bias"] for r in valid_mechanisms])) if valid_mechanisms else None
                    if mse is not None: _check(np.isfinite([mse,bias]).all(),"Nonfinite stratified metric")
                    valid_count = sum(r["n_valid"] for r in per_mechanism)
                    strata.append({"group":group,"center_t":center,"reason":reason,
                        "status":"EMPTY" if not selected else "DESCRIPTIVE" if valid_count else "NO_VALID_PREDICTIONS",
                        "n_requested":len(selected),"n_valid":valid_count,
                        "n_failed_or_missing":len(selected)-valid_count,
                        "n_mechanisms_with_valid":len(valid_mechanisms),
                        "mse":mse,"pooled_rmse":float(np.sqrt(mse)) if mse is not None else None,
                        "signed_bias":bias,"per_mechanism":per_mechanism})
        out[_bucket(method)][method] = {"status":"DESCRIPTIVE_ONLY",
            "aggregation":"within_mechanism_valid_queries_then_equal_mechanisms",
            "conditional_on_valid_predictions":True,"strata":strata,
            "raw_queries":queries,"support_counts":counts}
    return _immutable(out)
