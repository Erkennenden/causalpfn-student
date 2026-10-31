"""Three-layer ledger per approved 6.2.1 uniqueness decision; no execution."""
from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json
import os
import threading

import numpy as np

from .config import CONFIG
from . import contracts as c
from .estimation import (EstimationRequest, EstimateBundle, PairEstimateBundle,
    EFFECT_ORDER, IntervalRecord, SamplingUncertainty, normalized_status,
    aggregate_execution, metadata, task_contexts)
from .support import method_capability


class LedgerError(ValueError):
    pass


def _plain(value):
    if isinstance(value,Mapping):
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(_plain(metadata(value)),sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False)


def checksum(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def strict_json(text):
    def unique(pairs):
        out = {}
        for k,v in pairs:
            if k in out: raise LedgerError("Duplicate JSON key: "+k)
            out[k] = v
        return out
    def invalid(value):
        raise LedgerError("Nonfinite JSON constant: "+value)
    try:
        return json.loads(text,object_pairs_hook=unique,parse_constant=invalid)
    except (ValueError,TypeError) as exc:
        raise LedgerError("Invalid ledger JSON: "+str(exc)) from exc


def _require(condition,message):
    if not condition: raise LedgerError(message)


def _nodes(value):
    _require(type(value) is list and len(value) == 200
             and all(type(v) is int and v == i for i,v in enumerate(value)),
             "Canonical node order required")


def request_document(request):
    return {f.name:(request.node_ids.tolist() if f.name == "node_ids"
                    else _plain(getattr(request,f.name))) for f in fields(EstimationRequest)}


def request_from_document(doc):
    _require(type(doc) is dict and set(doc) == {f.name for f in fields(EstimationRequest)},
             "Request schema mismatch")
    _nodes(doc["node_ids"])
    kwargs = dict(doc)
    for name in ("context_ids","input_hashes","requested_outputs","arm_order","effect_order"):
        kwargs[name] = tuple(kwargs[name])
    kwargs["node_ids"] = np.array(kwargs["node_ids"],dtype=np.int64)
    return EstimationRequest(**kwargs)


DATA_KEYS = {"task_id","scope","pair_id","context_ids","input_hashes","node_ids",
             "identity","protocol_id","protocol_version","protocol_hash",
             "root_seed","protocol_code","bound_links"}


@dataclass(frozen=True,slots=True)
class DataTask:
    content: str

    def __post_init__(self):
        d = strict_json(self.content)
        _require(type(d) is dict and set(d) == DATA_KEYS,"Data task schema mismatch")
        _nodes(d["node_ids"])
        ident = d["identity"]
        _require(type(ident) is dict and set(ident) == {
            "cohort","cohort_code","mechanism_id","pair_repeat_id","rng_identity","center_t"},
            "Data identity schema mismatch")
        cc,mid,r = (ident[k] for k in ("cohort_code","mechanism_id","pair_repeat_id"))
        expected_pair = c.pair_id_for(cc,mid,r)
        _require(ident["cohort"] == {1:"pilot",2:"test"}.get(cc),"Cohort mismatch")
        centers = ident["center_t"]
        _require(d["scope"] in ("context","pair"),"Invalid data scope")
        _require(centers == [0,1] if d["scope"] == "pair"
                 else type(centers) is list and len(centers) == 1
                      and type(centers[0]) is int and centers[0] in (0,1),"Center ordering mismatch")
        contexts = [c.context_id_for(cc,mid,r,t) for t in centers]
        _require(d["pair_id"] == expected_pair and d["context_ids"] == contexts,"Pair/context ID conflict")
        _require(d["task_id"] == (expected_pair if d["scope"] == "pair" else contexts[0]),
                 "Do not replace existing task_id semantics")
        _require(ident["rng_identity"] == c._rng(cc,mid,r),"Frozen RNG identity conflict")
        _require((d["protocol_id"],d["protocol_version"],d["protocol_hash"],
                  d["root_seed"],d["protocol_code"]) ==
                 (CONFIG.protocol_id,CONFIG.protocol_version,CONFIG.protocol_sha256,
                  CONFIG.root_seed,CONFIG.protocol_code),"Frozen protocol/seed conflict")
        _require(type(d["input_hashes"]) is list and len(d["input_hashes"]) == len(contexts)
                 and all(type(h) is str and len(h) == 64 and all(x in "0123456789abcdef" for x in h)
                         for h in d["input_hashes"]),"Input hash schema mismatch")
        _require(type(d["bound_links"]) is list and len(d["bound_links"]) == len(contexts),
                 "Bound link count mismatch")
        for link in d["bound_links"]:
            _require(type(link) is dict and set(link) == {"target_id","request_id"},"Bound link schema")
            _require(link["target_id"] in (None,CONFIG.target_id),"Bound target mismatch")
            _require(link["request_id"] is None or
                     (type(link["request_id"]) is str and link["request_id"].strip()
                      and link["target_id"] == CONFIG.target_id),"Bound request mismatch")
        object.__setattr__(self,"content",canonical(d))

    @property
    def document(self): return strict_json(self.content)
    @property
    def task_id(self): return self.document["task_id"]

    @classmethod
    def from_document(cls,doc): return cls(canonical(doc))

    @classmethod
    def from_task(cls,task):
        contexts = task_contexts(task)
        first = contexts[0]
        scope = "context" if type(task) is c.TaskSpec else "pair"
        doc = {k:getattr(first,k) for k in ("protocol_id","protocol_version","protocol_hash")}
        doc.update(task_id=first.context_id if scope == "context" else task.pair_id,
            scope=scope,pair_id=task.pair_id,context_ids=[t.context_id for t in contexts],
            input_hashes=[t.input_hash for t in contexts],node_ids=task.node_ids.tolist(),
            root_seed=CONFIG.root_seed,protocol_code=CONFIG.protocol_code,
            identity={**{k:getattr(first,k) for k in (
                "cohort","cohort_code","mechanism_id","pair_repeat_id","rng_identity")},
                "center_t":[t.center_t for t in contexts]},
            bound_links=[{"target_id":t.target_id,"request_id":t.request_id} for t in contexts])
        return cls.from_document(doc)


@dataclass(frozen=True,slots=True)
class ExecutionRequest:
    content: str

    def __post_init__(self):
        d = strict_json(self.content)
        _require(type(d) is dict and set(d) == {"method_id","effects","request"},"Execution schema mismatch")
        request = request_from_document(d["request"])
        cap = method_capability(d["method_id"])
        _require(cap.method_id == d["method_id"],"Use canonical method_id")
        _require(d["effects"] == list(EFFECT_ORDER),"Frozen effect order mismatch")
        _require(any(request.scope+"."+("ate_interval" if out == "interval" else out)
                     in cap.outputs for out in request.requested_outputs),"No applicable method output")
        object.__setattr__(self,"content",canonical(d))

    @property
    def document(self): return strict_json(self.content)
    @property
    def key(self):
        d = self.document;r = request_from_document(d["request"])
        return (d["method_id"],r.scope,r.task_id,r.request_id)
    @classmethod
    def from_document(cls,d): return cls(canonical(d))
    @classmethod
    def from_request(cls,method,request):
        return cls.from_document({"method_id":method,"effects":list(EFFECT_ORDER),
                                  "request":request_document(request)})


ATTEMPT_KEYS = {"execution_request","task_id","attempt_no","authorization",
    "status","execution_status","output_status","failure_type","failure_message",
    "failure_provenance","diagnostics","estimability","intervals",
    "sampling_uncertainty","asset_hashes"}


@dataclass(frozen=True,slots=True)
class Attempt:
    content: str

    def __post_init__(self):
        d = strict_json(self.content)
        _require(type(d) is dict and set(d) == ATTEMPT_KEYS,"Attempt schema mismatch")
        record = ExecutionRequest.from_document(d["execution_request"])
        request = request_from_document(record.document["request"])
        method = record.key[0];cap = method_capability(method)
        _require(d["task_id"] == request.task_id,"Attempt task identity mismatch")
        _require(type(d["attempt_no"]) is int and d["attempt_no"] >= 1,"Invalid attempt_no")
        auth = d["authorization"]
        _require(auth is None or type(auth) is str and bool(auth.strip()),"Invalid authorization reference")
        _require(d["attempt_no"] == 1 or auth is not None,"Explicit retry authorization required")
        _require(d["status"] == request.status,"Identification status changed")
        supplied = {k:tuple(v) if type(v) is list else v for k,v in d["output_status"].items()}
        statuses = normalized_status(request,cap,supplied)
        _require(set(supplied) == {"mu","ite","ate","interval"},"Complete output status evidence required")
        _require(d["execution_status"] == aggregate_execution(request,statuses),
                 "Execution status mismatch; NOT_RANKABLE is not execution")
        if d["execution_status"] == "NOT_IDENTIFIED":
            _require((d["failure_type"],d["failure_message"],d["failure_provenance"]) ==
                     (request.failure_type,request.failure_message,_plain(request.failure_provenance)),
                     "Identification failure must be inherited verbatim")
        elif d["execution_status"] != "OK":
            _require(all(type(d[k]) is str and d[k].strip() for k in ("failure_type","failure_message")),
                     "Unavailable execution requires explicit failure reason")
        records = [IntervalRecord(**v) for v in d["intervals"]]
        _require(len({r.effect for r in records}) == len(records),"Duplicate interval effect")
        expected_source = "joint_model_posterior" if method == "oracle-pair" else "model_posterior"
        for record in records:
            _require(record.context_id == (request.context_ids[0] if request.scope == "context" else None),
                     "Interval scope/identity mismatch")
            _require(record.status == statuses["interval"][EFFECT_ORDER.index(record.effect)],
                     "Interval status conflict")
            _require(record.status != "OK" or record.source == expected_source,"Interval source conflict")
        required = {EFFECT_ORDER[i] for i,s in enumerate(statuses["interval"])
                    if s in ("OK","FAILED","NOT_ESTIMABLE")}
        _require({r.effect for r in records} == required,"Missing or extra requested interval record")
        u = SamplingUncertainty(**d["sampling_uncertainty"])
        key = request.scope+".ate_interval"
        expected_u = ("NOT_REQUESTED" if "interval" not in request.requested_outputs else
            "UNSUPPORTED" if key not in cap.outputs else
            "OK" if "OK" in statuses["interval"] else
            "FAILED" if "FAILED" in statuses["interval"] else "NOT_ESTIMABLE")
        _require(u.status == expected_u,"Uncertainty status conflict")
        if u.status == "OK" or method == "j" and u.source is not None:
            _require((u.source,u.statistical_unit) == (expected_source,request.scope),
                     "Uncertainty source/unit conflict")
        elif any(v is not None for v in (u.source,u.statistical_unit,u.effective_sample_count)):
            raise LedgerError("Unavailable non-J uncertainty cannot fabricate provenance")
        _require(method != "j" or u.effective_sample_count is None,"J analytic uncertainty count must be None")
        object.__setattr__(self,"content",canonical(d))

    @property
    def document(self): return strict_json(self.content)
    @property
    def execution_key(self): return ExecutionRequest.from_document(self.document["execution_request"]).key
    @property
    def key(self): return (self.execution_key,self.document["attempt_no"])

    @classmethod
    def from_document(cls,d): return cls(canonical(d))

    @classmethod
    def from_bundle(cls,bundle,*,attempt_no,authorization=None):
        _require(type(bundle) in (EstimateBundle,PairEstimateBundle),"Typed public Bundle required")
        request = ExecutionRequest.from_request(bundle.method_id,bundle.request)
        doc = {k:_plain(getattr(bundle,k)) for k in ("status","execution_status","output_status",
            "failure_type","failure_message","failure_provenance","diagnostics","estimability","asset_hashes")}
        doc.update(execution_request=request.document,task_id=bundle.task_id,attempt_no=attempt_no,
            authorization=authorization,intervals=[{f.name:getattr(r,f.name) for f in fields(IntervalRecord)}
                                                   for r in bundle.intervals],
            sampling_uncertainty={f.name:getattr(bundle.sampling_uncertainty,f.name)
                                  for f in fields(SamplingUncertainty)})
        return cls.from_document(doc)


class TaskRegistry:
    """Append-only records; seal fixes inventory, not execution success/ranking."""
    def __init__(self):
        self._tasks = {};self._requests = {};self._attempts = []
        self._sealed = False;self._seal = None
        # Process/thread-affine single writer; NOT a cross-process lock.
        self._writer = (os.getpid(), threading.get_ident())
        self._recording = False
        self._last_attempt_no = {}
        self._attempt_by_key = {}
        self._indexed_count = 0
        self._indexed_executions = 0
        self._indexed_tail = None
        self._history_digest = hashlib.sha256(b"star-v2-attempt-index-v1").hexdigest()
    @property
    def tasks(self): return MappingProxyType(self._tasks)
    @property
    def requests(self): return MappingProxyType(self._requests)
    @property
    def attempts(self):
        self.validate_attempt_indexes()  # Full-history boundary, never a hot lookup.
        return tuple(self._attempts)
    @property
    def sealed(self): return self._sealed
    @property
    def seal_sha256(self): return self._seal

    def register_task(self,task):
        _require(not self._sealed,"Task inventory is sealed")
        record = task if type(task) is DataTask else DataTask.from_task(task)
        if record.task_id in self._tasks:
            raise LedgerError("DUPLICATE_TASK" if self._tasks[record.task_id] == record else "TASK_CONFLICT")
        self._tasks[record.task_id] = record
        return record

    def _bind(self,record):
        d = record.document;r = d["request"];key = record.key
        _require(key[2] in self._tasks,"Request refers to unregistered data task")
        task = self._tasks[key[2]].document
        for name in ("scope","pair_id","context_ids","input_hashes","node_ids","identity",
                     "protocol_id","protocol_version","protocol_hash"):
            _require(r[name] == task[name],"Request/data binding conflict: "+name)
        for link in task["bound_links"]:
            _require(link["target_id"] is None or link["target_id"] == r["target_id"],"Bound target changed")
            _require(link["request_id"] is None or link["request_id"] == r["request_id"],"Bound request changed")

    def register_request(self,method,request):
        return self.register_execution(ExecutionRequest.from_request(method,request))

    def register_execution(self,record):
        _require(not self._sealed,"Execution inventory is sealed")
        _require(type(record) is ExecutionRequest,"ExecutionRequest required")
        self._bind(record)
        if record.key in self._requests:
            raise LedgerError("DUPLICATE_REQUEST" if self._requests[record.key] == record else "REQUEST_CONFLICT")
        self._requests[record.key] = record
        return record

    def inventory(self,*,expected_task_ids,expected_request_keys):
        tids,keys = tuple(expected_task_ids),tuple(tuple(k) for k in expected_request_keys)
        _require(len(set(tids)) == len(tids) and len(set(keys)) == len(keys),"Duplicate expected inventory")
        return immutable_inventory({
            "missing_tasks":sorted(set(tids)-set(self._tasks)),
            "extra_tasks":sorted(set(self._tasks)-set(tids)),
            "missing_requests":sorted(set(keys)-set(self._requests)),
            "extra_requests":sorted(set(self._requests)-set(keys))})

    def seal(self,*,expected_task_ids,expected_request_keys):
        _require(not self._sealed,"Already sealed")
        evidence = self.inventory(expected_task_ids=expected_task_ids,expected_request_keys=expected_request_keys)
        _require(not any(evidence.values()),"Incomplete inventory: "+canonical(evidence))
        for task in self._tasks.values():
            d = task.document
            _require(d["pair_id"] in self._tasks and self._tasks[d["pair_id"]].document["scope"] == "pair",
                     "Pair data task missing")
            parent = self._tasks[d["pair_id"]].document
            for i,cid in enumerate(parent["context_ids"]):
                _require(cid in self._tasks,"Context data task missing")
                child = self._tasks[cid].document
                _require(child["scope"] == "context" and child["pair_id"] == parent["pair_id"]
                         and child["input_hashes"] == [parent["input_hashes"][i]],"Pair/context link conflict")
        for record in self._requests.values():
            d = record.document;r = d["request"]
            if r["parent_request_id"] is not None:
                parent_key = (d["method_id"],"pair",r["pair_id"],r["parent_request_id"])
                _require(parent_key in self._requests,"Parent execution request missing")
                parent = self._requests[parent_key].document["request"]
                for field in ("status","target_id","requested_outputs","failure_type",
                              "failure_message","failure_provenance"):
                    _require(r[field] == parent[field],"Parent/child request conflict: "+field)
        self._seal = checksum({"tasks":[self._tasks[k].document for k in sorted(self._tasks)],
                              "requests":[self._requests[k].document for k in sorted(self._requests)]})
        self._sealed = True
        return self._seal

    def _check_writer(self):
        _require(self._writer == (os.getpid(), threading.get_ident()),
                 "Registry requires its single owning process/thread")

    @staticmethod
    def _extend_history(previous, content):
        return hashlib.sha256(bytes.fromhex(previous)+content.encode("ascii")).hexdigest()

    def _check_index_header(self):
        self._check_writer()
        _require(isinstance(self._attempts,list)
            and type(self._last_attempt_no) is dict and type(self._attempt_by_key) is dict
            and self._indexed_count == len(self._attempts) == len(self._attempt_by_key)
            and self._indexed_executions == len(self._last_attempt_no)
            and (self._indexed_tail is self._attempts[-1] if self._attempts
                 else self._indexed_tail is None),
            "Attempt index/header is missing, stale or inconsistent")

    def _entry(self, key, no):
        entry = self._attempt_by_key.get((key,no))
        _require(type(entry) is tuple and len(entry)==5, "Attempt index entry missing/corrupt")
        attempt,position,content,stored_key,stored_no = entry
        _require(type(position) is int and 0 <= position < self._indexed_count
            and type(attempt) is Attempt and self._attempts[position] is attempt
            and attempt.content==content and stored_key==key and stored_no==no,
            "Attempt index disagrees with authoritative history")
        return entry

    def last_attempt_no(self, key):
        """O(1) local consistency checks; full audit remains explicit."""
        self._check_index_header()
        _require(type(key) is tuple and len(key)==4 and all(type(v) is str for v in key)
                 and key in self._requests, "Unregistered execution key")
        no = self._last_attempt_no.get(key)
        if no is None:
            _require((key,1) not in self._attempt_by_key, "Last-Attempt index missing")
            return 0
        _require(type(no) is int and no>=1, "Invalid last-Attempt index")
        last = self._entry(key,no)
        _require((key,no+1) not in self._attempt_by_key, "Last-Attempt index is stale")
        if no>1:
            _require(self._entry(key,no-1)[1]<last[1], "Attempt order index conflict")
        return no

    def next_attempt_no(self, key):
        return self.last_attempt_no(key)+1

    def get_attempt(self, key, attempt_no):
        last = self.last_attempt_no(key)
        _require(type(attempt_no) is int and attempt_no>=1, "Invalid Attempt lookup number")
        if attempt_no>last:
            return None
        return self._entry(key,attempt_no)[0]

    @property
    def attempt_count(self):
        self._check_index_header()
        return self._indexed_count

    @property
    def latest_attempt(self):
        self._check_index_header()
        return self._indexed_tail

    def _build_attempt_indexes(self):
        """Linear, fully validated rebuild from immutable authoritative records."""
        self._check_writer()
        last,by_key = {},{}
        digest = hashlib.sha256(b"star-v2-attempt-index-v1").hexdigest()
        for position,attempt in enumerate(self._attempts):
            _require(type(attempt) is Attempt, "Invalid Attempt history entry")
            # Recheck complete schema/status/provenance at the audit boundary.
            _require(Attempt.from_document(attempt.document)==attempt, "Invalid Attempt content")
            key,no = attempt.key
            _require(key in self._requests
                and attempt.document["execution_request"]==self._requests[key].document,
                "Attempt history/request identity conflict")
            _require(no==last.get(key,0)+1 and (key,no) not in by_key,
                "Duplicate, missing or out-of-order Attempt history")
            by_key[(key,no)] = (attempt,position,attempt.content,key,no)
            last[key] = no
            digest = self._extend_history(digest,attempt.content)
        return last,by_key,digest

    def rebuild_attempt_indexes(self):
        """Explicit repair of indexes only; never repair or replace history."""
        self._check_writer()
        _require(not self._recording, "Registry write already in progress")
        last,by_key,digest = self._build_attempt_indexes()
        _require(digest==self._history_digest, "Authoritative Attempt history changed")
        self._last_attempt_no,self._attempt_by_key = last,by_key
        self._indexed_count = len(self._attempts)
        self._indexed_executions = len(last)
        self._indexed_tail = self._attempts[-1] if self._attempts else None
        return self._indexed_count

    def validate_attempt_indexes(self):
        """Full O(N) integrity check for exports, final audit and diagnostics."""
        self._check_index_header()
        last,by_key,digest = self._build_attempt_indexes()
        _require(last==self._last_attempt_no and by_key==self._attempt_by_key
                 and digest==self._history_digest, "Attempt index/full history mismatch")
        return self._indexed_count

    def record_attempt(self,attempt):
        self._check_writer()
        _require(not self._recording, "Registry write already in progress")
        self._recording = True
        try:
            self._check_index_header()
            _require(self._sealed,"Seal the expected universe before execution records")
            _require(type(attempt) is Attempt,"Typed Attempt required")
            key,no = attempt.key
            _require(key in self._requests,"Attempt refers to unregistered execution")
            _require(attempt.document["execution_request"] == self._requests[key].document,
                     "Attempt changed frozen request/task/RNG identity")
            _require(no == self.next_attempt_no(key),
                     "Duplicate, missing or out-of-order attempt_no")
            # All legal-state checks precede any history/index mutation.
            digest = self._extend_history(self._history_digest,attempt.content)
            position = self._indexed_count
            self._attempts.append(attempt)
            self._attempt_by_key[(key,no)] = (attempt,position,attempt.content,key,no)
            if key not in self._last_attempt_no:
                self._indexed_executions += 1
            self._last_attempt_no[key] = no
            self._indexed_count += 1
            self._indexed_tail = attempt
            self._history_digest = digest
            return attempt
        finally:
            self._recording = False

    def append_attempt(self,bundle,*,attempt_no,authorization=None):
        return self.record_attempt(Attempt.from_bundle(bundle,attempt_no=attempt_no,authorization=authorization))

    def unexecuted_requests(self):
        self.validate_attempt_indexes()
        return tuple(k for k in self._requests if k not in self._last_attempt_no)


def immutable_inventory(value):
    return MappingProxyType({k:tuple(tuple(x) if type(x) in (list,tuple) else x for x in v)
                             for k,v in value.items()})
