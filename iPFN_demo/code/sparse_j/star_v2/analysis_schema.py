"""Deterministic v2 analysis wire format; no scientific estimators."""
from collections.abc import Mapping
from dataclasses import is_dataclass,fields
import csv,io,json,math,os
from pathlib import Path
import numpy as np
from . import data_artifacts as data

SCHEMA_VERSION=1
COMMON_COLUMNS=("row_id","method_id","effect","group","center_t","support_reason",
 "metric","value","status","unit","n_requested","n_valid","n_failed_or_missing",
 "ci_status","ci_lower","ci_upper","ci_level","ci_interpretation","source_selector",
 "source_run_id","source_snapshot_id","source_manifest_sha256","protocol_sha256",
 "registry_seal","reference_only","details")
PAIR_METRICS=("pooled_rmse","signed_bias","absolute_bias","absolute_conditional_bias",
              "within_mechanism_variance","between_mechanism_variability")
DELTA_METRICS=("pooled_rmse","signed_bias","absolute_bias","absolute_conditional_bias",
               "between_mechanism_standard_deviation")
GROUP_METRICS=("pooled_rmse","signed_bias","absolute_conditional_bias")

def plain(value):
    if isinstance(value,Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return plain(value.item())
    if is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,np.ndarray):raise ValueError("Arrays must be explicitly persisted in NPZ")
    if type(value) is float and not math.isfinite(value):raise ValueError("Nonfinite JSON result")
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError("Non-JSON analysis metadata")

def blob(value):
    return (json.dumps(plain(value),sort_keys=True,separators=(",",":"),ensure_ascii=True,
                       allow_nan=False)+"\n").encode("ascii")

def digest(value):return data.sha(blob(value))

def write_json(root,name,value):return data._write(root,name,blob(value))

def read_json(root,name):return data.strict_json((Path(root)/name).read_text(encoding="ascii"))

def write_csv(root,name,columns,rows):
    path=data._target(root,name);path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n",extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row)!=set(columns):raise ValueError("CSV schema mismatch: "+name)
            writer.writerow({key:json.dumps(plain(value),sort_keys=True,separators=(",",":"),
                ensure_ascii=True,allow_nan=False) if isinstance(value,(Mapping,tuple,list))
                else "" if value is None else value for key,value in row.items()})
            count+=1
        stream.flush();os.fsync(stream.fileno())
    raw=path.read_bytes()
    return {"path":name,"bytes":len(raw),"sha256":data.sha(raw),"rows":count,"columns":list(columns)}

def table_row(provenance,*,method,effect,metric,value,status,n_requested,n_valid,
              group="all",center_t=None,reason=None,unit="outcome",
              ci=None,ci_status="NOT_REQUESTED",details=None,selector=None,reference=False):
    result=dict(zip(COMMON_COLUMNS,(None,)*len(COMMON_COLUMNS)))
    result.update(method_id=method,effect=effect,group=group,center_t=center_t,
        support_reason=reason,metric=metric,value=value,status=status,unit=unit,
        n_requested=n_requested,n_valid=n_valid,n_failed_or_missing=n_requested-n_valid,
        ci_status=ci_status,ci_lower=None if ci is None else ci[0],
        ci_upper=None if ci is None else ci[1],ci_level=None if ci is None else .95,
        ci_interpretation="pointwise percentile; linear quantile; no multiplicity adjustment" if ci else None,
        source_selector=selector or {"method_id":method,"effect":effect,"group":group,"center_t":center_t},
        source_run_id=provenance["run_id"],source_snapshot_id=provenance["source_snapshot_id"],
        source_manifest_sha256=provenance["source_manifest_sha256"],
        protocol_sha256=provenance["protocol_sha256"],registry_seal=provenance["registry_seal"],
        reference_only=reference,details=details or {})
    result["row_id"]=digest({k:v for k,v in result.items() if k not in ("row_id","value",
        "n_valid","n_failed_or_missing","ci_lower","ci_upper")})
    return result

def metric_unit(metric):
    return "outcome^2" if metric in ("mse","within_mechanism_variance",
                                    "between_mechanism_variability") else "outcome"
