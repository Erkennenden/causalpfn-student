"""Seven-method per-effect rankability only; no ordinal score or new metrics."""
from dataclasses import replace
from collections import Counter

import numpy as np

from .task_registry import TaskRegistry
from .support import ORDINARY_METHOD_IDS, method_capability
from .metrics import EvaluationPlan, Prediction, pair_ate_metrics
from .estimation import EFFECT_ORDER
from .coverage import check, freeze, unique_attempts, formal_grid

CONTEXT_METHODS=("kernel-old","kernel-new","j")


def validate_test_ranking_counts(*,mechanism_count,pair_counts):
    """Formal aggregate gate without constructing the formal task universe."""
    check(type(mechanism_count) is int and mechanism_count == 200,"Test requires 200 mechanisms")
    check(set(pair_counts) == set(ORDINARY_METHOD_IDS),"Exactly seven ordinary methods required")
    check(all(type(v) is int and v == 600 for v in pair_counts.values()),"600 Pair requests per method required")
    check(sum(pair_counts.values()) == 4200,"4200 Pair-method requests required")
    return True


def evaluate_rankability(registry,predictions,truths,*,formal=True):
    check(type(registry) is TaskRegistry and registry.sealed,"Sealed registry required")
    pairs={k:r for k,r in registry.requests.items() if k[0] in ORDINARY_METHOD_IDS
           and k[1] == "pair" and "ate" in r.document["request"]["requested_outputs"]}
    check({k[0] for k in pairs} == set(ORDINARY_METHOD_IDS),"All seven ordinary methods must be pre-requested")
    counts={method:sum(k[0] == method for k in pairs) for method in ORDINARY_METHOD_IDS}
    mechanisms={registry.tasks[k[2]].document["identity"]["mechanism_id"] for k in pairs}
    if formal:
        validate_test_ranking_counts(mechanism_count=len(mechanisms),pair_counts=counts)
        for method in ORDINARY_METHOD_IDS: formal_grid(registry,[k for k in pairs if k[0] == method])
    dependencies={}
    relevant=set(pairs)
    for key,execution in pairs.items():
        if key[0] not in CONTEXT_METHODS: continue
        request=execution.document["request"]
        children=[]
        for center,cid in enumerate(request["context_ids"]):
            candidates=[ck for ck,cr in registry.requests.items()
                if ck[:2] == (key[0],"context") and ck[2] == cid
                and cr.document["request"]["parent_request_id"] == key[3]]
            check(len(candidates)<=1,"Ambiguous necessary context request")
            children.append(candidates[0] if candidates else None)
            relevant.update(candidates)
        dependencies[key]=tuple(children)
        if formal:
            check(all(k is not None for k in children),"Formal plan missing necessary context request")
    selected=unique_attempts(registry,relevant)
    supplied={};excluded=[]
    for prediction in predictions:
        check(type(prediction) is Prediction and prediction.key in registry.requests,"Unknown prediction identity")
        key=prediction.key
        check(key not in supplied,"Duplicate prediction identity")
        check(key[1] == "pair","Ranking accepts Pair predictions only")
        request=registry.requests[key].document["request"]
        check(prediction.input_hashes == tuple(request["input_hashes"])
              and prediction.node_ids == tuple(request["node_ids"])
              and prediction.identification_status == request["status"],"Prediction/request binding conflict")
        supplied[key]=prediction
        if method_capability(key[0]).reference_only:
            excluded.append(key)
        else:
            check(key in pairs,"Unexpected ordinary Pair prediction")
            check(selected[key] is not None,"Prediction has no recorded execution Attempt")
    rows=[];effective=[]
    for key,execution in pairs.items():
        request=execution.document["request"];attempt=selected[key];prediction=supplied.get(key)
        states=[];reasons=[]
        for k,effect in enumerate(EFFECT_ORDER):
            code=None;detail=None;state="OK"
            if attempt is None:
                code="UNEXECUTED";state="MISSING"
            elif attempt["status"] != "POINT":
                code="NOT_IDENTIFIED";state="NOT_IDENTIFIED";detail=attempt["failure_message"]
            elif attempt["output_status"]["ate"][k] != "OK":
                state=attempt["output_status"]["ate"][k];code=state;detail=attempt["failure_message"]
            elif prediction is None:
                code="MISSING_POINT_OUTPUT";state="MISSING"
            elif prediction.output_status["ate"][k] != "OK":
                code="OUTPUT_STATUS_CONFLICT";state="FAILED";detail=prediction.failure_reasons
            if state == "OK" and key in dependencies:
                for child in dependencies[key]:
                    child_attempt=selected.get(child)
                    if child is None or child_attempt is None:
                        code="MISSING_NECESSARY_CONTEXT";state="MISSING";break
                    child_state=child_attempt["output_status"]["ate"][k]
                    if child_attempt["status"] != "POINT" or child_state != "OK":
                        code="NECESSARY_CONTEXT_"+child_state;state="FAILED"
                        detail={"key":child,"state":child_state,"reason":child_attempt["failure_message"],
                                "provenance":child_attempt["failure_provenance"]};break
            states.append("OK" if state == "OK" else "FAILED")
            reasons.append(code)
            rows.append({"key":key,"method":key[0],"effect":effect,"pair_id":request["pair_id"],
                "mechanism_id":request["identity"]["mechanism_id"],"request_id":key[3],
                "status":state,"valid":state == "OK","failure_code":code,"failure_reason":detail,
                "attempt_failure_provenance":attempt["failure_provenance"] if attempt else None})
        if prediction is not None:
            values=prediction.ate.copy() if prediction.ate is not None else np.full(3,np.nan)
            for k,state in enumerate(states):
                if state != "OK": values[k]=np.nan
            effective.append(replace(prediction,ate=values,
                output_status={**prediction.output_status,"ate":tuple(states)},
                execution_status=("NOT_IDENTIFIED" if prediction.identification_status == "NOT_IDENTIFIED" else
                                  "FAILED" if "FAILED" in states else prediction.execution_status),
                failure_reasons={"qualification_input_failures":tuple(reasons)}))
    base_plan=EvaluationPlan.from_registry(registry)
    plan=EvaluationPlan(tuple(t for t in base_plan.tasks if t.key[0] in ORDINARY_METHOD_IDS),
                        source_seal=registry.seal_sha256)
    stats=pair_ate_metrics(plan,effective,truths)["ordinary"]
    methods={}
    for method in ORDINARY_METHOD_IDS:
        effects={}
        for effect in EFFECT_ORDER:
            requests=[r for r in rows if r["method"] == method and r["effect"] == effect]
            valid=sum(r["valid"] for r in requests)
            missing=sum(r["status"] == "MISSING" for r in requests)
            failed=len(requests)-valid-missing
            effects[effect]={"requested_count":len(requests),"valid_count":valid,
                "failed_count":failed,"missing_count":missing,
                "rankability_status":"RANKABLE" if valid == len(requests) else "NOT_RANKABLE",
                "reason_distribution":dict(Counter(r["failure_code"] for r in requests if not r["valid"])),
                "failure_reasons":tuple(r for r in requests if not r["valid"]),
                "records":requests,"statistics":stats[method][effect]}
        methods[method]={"effects":effects,"all_effects_rankable":
            all(r["rankability_status"] == "RANKABLE" for r in effects.values())}
    return freeze({"evidence_scope":"FORMAL_TEST_PLAN" if formal else "SYNTHETIC",
        "registry_seal":registry.seal_sha256,"methods":methods,
        "excluded_oracle_predictions":excluded,"ordinal_ranks_produced":False})
