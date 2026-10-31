"""Frozen J coverage only. No model calls, interval construction or Bootstrap."""
from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType
import math

from .task_registry import TaskRegistry
from .estimation import EstimateBundle, PairEstimateBundle, EFFECT_ORDER
from .metrics import MechanismTruth
from .support import method_capability


def check(condition,message):
    if not condition: raise ValueError(message)


def freeze(value):
    if isinstance(value,Mapping): return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)): return tuple(freeze(v) for v in value)
    return value


class AttemptSelectionAmbiguous(ValueError):
    code = "ATTEMPT_SELECTION_AMBIGUOUS"
    def __init__(self,keys):
        self.keys = tuple(keys)
        super().__init__(self.code+": explicit authoritative selection is required")


def unique_attempts(registry,keys):
    check(type(registry) is TaskRegistry and registry.sealed,"Sealed registry required")
    relevant = set(keys);grouped = {key:[] for key in relevant}
    for attempt in registry.attempts:
        if attempt.execution_key in relevant: grouped[attempt.execution_key].append(attempt)
    ambiguous = [key for key,values in grouped.items() if len(values)>1]
    if ambiguous: raise AttemptSelectionAmbiguous(sorted(ambiguous))
    return {key:values[0].document if values else None for key,values in grouped.items()}


def validate_test_coverage_counts(*,mechanism_count,context_request_count,center_counts,effect_counts):
    """Aggregate formal-plan gate; tests need not generate 1200 actual tasks."""
    check(type(mechanism_count) is int and mechanism_count == 200,"Test requires 200 mechanisms")
    check(type(context_request_count) is int and context_request_count == 1200,"Test requires 1200 J contexts")
    check(type(center_counts) is dict and all(type(k) is int and type(v) is int
              for k,v in center_counts.items()) and center_counts == {0:600,1:600},
          "Test requires 600 requests per center arm")
    check(type(effect_counts) is dict and all(type(v) is int for v in effect_counts.values()),
          "Effect counts must be integers")
    check(effect_counts == {effect:1200 for effect in EFFECT_ORDER},"Test effect denominator mismatch")
    return True


def formal_grid(registry,keys):
    slots = {}
    for key in keys:
        data = registry.tasks[key[2]].document
        identity = data["identity"]
        check(identity["cohort_code"] == 2,"Formal evaluation requires Test cohort")
        slots.setdefault(identity["mechanism_id"],{}).setdefault(identity["pair_repeat_id"],[]).append(
            identity["center_t"][0] if key[1] == "context" else None)
    check(set(slots) == set(range(200)),"Formal mechanism IDs must be 0..199")
    for repeats in slots.values():
        check(set(repeats) == {0,1,2},"Formal repeat IDs must be 0..2")
        for centers in repeats.values():
            check(sorted(centers) == [0,1] if keys[0][1] == "context" else centers == [None],
                  "Formal request grid duplicate/missing identity")


@dataclass(frozen=True,slots=True)
class IntervalSubmission:
    key: tuple
    input_hashes: tuple
    intervals: tuple

    @classmethod
    def from_bundle(cls,bundle):
        check(type(bundle) in (EstimateBundle,PairEstimateBundle),"Public Bundle required")
        return cls((bundle.method_id,bundle.request.scope,bundle.task_id,bundle.request_id),
            bundle.input_hashes,tuple({f.name:getattr(r,f.name) for f in fields(type(r))}
                                     for r in bundle.intervals))


def _finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def _safe_bound(value):
    return value if _finite(value) else None


def _summary(rows,target=False):
    requested=len(rows);valid=[r for r in rows if r["valid"]]
    covered=sum(r["covered"] for r in valid)
    out={"all_requested":requested,"valid_intervals":len(valid),"covered_valid":covered,
         "failed_or_missing":requested-len(valid),
         "coverage_valid":covered/len(valid) if valid else None,
         "coverage_including_failures":covered/requested if requested else None,
         "interval_success_rate":len(valid)/requested if requested else None,
         "mean_interval_width":sum(r["upper"]-r["lower"] for r in valid)/len(valid) if valid else None}
    if target:
        out["meets_90pct_target"] = bool(requested and covered/requested >= .90)
    return out


def evaluate_j_coverage(registry,truths,*,submissions=None,formal=True):
    """Evaluate all sealed J context interval requests; malformed outputs fail closed.

    Interval endpoints are closed: lower <= truth <= upper. Explicit submissions
    may audit external result records; omitted submissions use the unique stored
    Attempt intervals. No attempt is implicitly preferred over another.
    """
    check(type(registry) is TaskRegistry and registry.sealed,"Sealed registry required")
    expected={k:v for k,v in registry.requests.items() if k[:2] == ("j","context")
              and "interval" in v.document["request"]["requested_outputs"]}
    check(expected,"No pre-requested J context intervals")
    selected=unique_attempts(registry,expected)
    centers={arm:sum(registry.tasks[k[2]].document["identity"]["center_t"] == [arm]
                     for k in expected) for arm in (0,1)}
    mechanisms={registry.tasks[k[2]].document["identity"]["mechanism_id"] for k in expected}
    if formal:
        validate_test_coverage_counts(mechanism_count=len(mechanisms),context_request_count=len(expected),
            center_counts=centers,effect_counts={e:len(expected) for e in EFFECT_ORDER})
        formal_grid(registry,list(expected))
    truth_map={}
    for truth in truths:
        check(type(truth) is MechanismTruth and truth.mechanism_id not in truth_map,
              "Duplicate/invalid mechanism truth")
        truth_map[truth.mechanism_id]=truth
    check(set(truth_map) <= mechanisms,"Extra mechanism truth")
    by_key={};references=[]
    if submissions is None:
        # Oracle history is reported separately without selecting an Attempt.
        for reference_attempt in registry.attempts:
            if reference_attempt.execution_key[0].startswith("oracle-"):
                doc=reference_attempt.document
                if doc["intervals"]:
                    references.append({"key":reference_attempt.execution_key,
                        "scope":reference_attempt.execution_key[1],"attempt_no":doc["attempt_no"],
                        "intervals":tuple(doc["intervals"])})
        submissions=tuple(IntervalSubmission(k,tuple(registry.requests[k].document["request"]["input_hashes"]),
                                             tuple(a["intervals"]))
                          for k,a in selected.items() if a is not None)
    for submission in submissions:
        check(type(submission) is IntervalSubmission and type(submission.key) is tuple
              and submission.key in registry.requests,"Unknown interval request identity")
        key=submission.key;request=registry.requests[key].document["request"]
        check(submission.input_hashes == tuple(request["input_hashes"]),"Interval input hash mismatch")
        check(key not in by_key,"Duplicate interval submission")
        check(type(submission.intervals) is tuple,"Invalid interval record container")
        method=key[0];cap=method_capability(method)
        check("interval" in request["requested_outputs"] and key[1]+".ate_interval" in cap.outputs,
              "Method/scope does not provide requested intervals")
        expected_source="joint_model_posterior" if method == "oracle-pair" else "model_posterior"
        seen=set()
        for record in submission.intervals:
            check(isinstance(record,Mapping),"Damaged interval without effect identity")
            effect=record.get("effect")
            check(effect in EFFECT_ORDER and effect not in seen,"Unknown/duplicate interval effect")
            seen.add(effect)
            check(record.get("context_id") == (request["context_ids"][0] if key[1] == "context" else None),
                  "Interval context/pair scope conflict")
            check(record.get("source") in (None,expected_source),"Unreliable interval source/scope")
            check(record.get("status") != "OK" or record.get("source") == expected_source,
                  "Missing successful interval provenance")
        by_key[key]=submission
        if method.startswith("oracle-"):
            references.append({"key":key,"scope":key[1],"intervals":tuple(dict(r) for r in submission.intervals)})
        else:
            check(key in expected,"Only pre-requested J context intervals enter coverage")
            check(selected[key] is not None,"Result has no recorded execution Attempt")
    rows=[]
    for key,execution in expected.items():
        request=execution.document["request"];data=registry.tasks[key[2]].document
        identity=data["identity"];attempt=selected[key];submission=by_key.get(key)
        records={r["effect"]:r for r in submission.intervals} if submission else {}
        for k,effect in enumerate(EFFECT_ORDER):
            record=records.get(effect);truth=truth_map.get(identity["mechanism_id"])
            target=float(truth.ate[k]) if truth is not None else None
            lower=record.get("lower") if record else None;upper=record.get("upper") if record else None
            reason=None
            if attempt is None: reason="UNEXECUTED"
            elif attempt["status"] != "POINT": reason="NOT_IDENTIFIED"
            elif "ate" in request["requested_outputs"] and attempt["output_status"]["ate"][k] != "OK":
                reason="DEPENDENT_POINT_FAILURE"
            elif submission is None or record is None: reason="MISSING_INTERVAL_OUTPUT"
            elif attempt["output_status"]["interval"][k] != "OK": reason="RECORDED_INTERVAL_FAILURE"
            elif record.get("status") != "OK": reason="INTERVAL_OUTPUT_FAILURE"
            elif record.get("level") != .95: reason="INVALID_INTERVAL_LEVEL"
            elif not _finite(lower) or not _finite(upper) or lower>upper: reason="INVALID_INTERVAL_BOUNDS"
            elif target is None: reason="MISSING_TRUTH"
            else:
                stored={r["effect"]:r for r in attempt["intervals"]}.get(effect)
                if stored is None or any(record.get(f) != stored.get(f) for f in (
                        "context_id","effect","status","lower","upper","level","source")):
                    reason="INTERVAL_ATTEMPT_CONTENT_CONFLICT"
            valid=reason is None
            rows.append({"key":key,"request_id":key[3],"pair_id":request["pair_id"],
                "context_id":request["context_ids"][0],"mechanism_id":identity["mechanism_id"],
                "pair_repeat_id":identity["pair_repeat_id"],"center_t":identity["center_t"][0],
                "effect":effect,"interval_status":record.get("status") if record else "MISSING",
                "effective_status":"OK" if valid else "FAILED","failure_code":reason,
                "failure_reason":None if valid else {
                    "code":reason,"record_reason":record.get("failure_reason") if record else None,
                    "execution_failure":attempt["failure_message"] if attempt else None,
                    "provenance":attempt["failure_provenance"] if attempt else None},
                "lower":_safe_bound(lower),"upper":_safe_bound(upper),"truth":target,
                "valid":valid,"covered":bool(valid and lower<=target<=upper)})
    overall={e:_summary([r for r in rows if r["effect"] == e],True) for e in EFFECT_ORDER}
    strata={arm:{e:_summary([r for r in rows if r["effect"] == e and r["center_t"] == arm])
                 for e in EFFECT_ORDER} for arm in (0,1)}
    return freeze({"evidence_scope":"FORMAL_TEST_PLAN" if formal else "SYNTHETIC",
        "registry_seal":registry.seal_sha256,"effects":overall,"center_t":strata,"records":rows,
        "oracle_reference_records":references,
        "all_effects_meet_90pct_target":all(r["meets_90pct_target"] for r in overall.values())})
