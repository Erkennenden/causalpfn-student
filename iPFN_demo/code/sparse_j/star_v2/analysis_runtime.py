"""Frozen section 9.1.1 runtime summaries; no prediction or selection."""
from collections import defaultdict
import numpy as np
from .config import CONFIG
from .support import ORDINARY_METHOD_IDS

def summarize_runtime(loaded):
    rows=[];summaries={}
    for method in ORDINARY_METHOD_IDS+("oracle-context","oracle-pair"):
        grouped=defaultdict(list)
        for key,record in loaded.runtimes.items():
            if key[:2]!=(method,"pair"):continue
            request=loaded.registry.requests[key].document["request"]
            identity=request["identity"]
            value=record.get("runtime_seconds")
            if value is not None and (type(value) not in (int,float) or not np.isfinite(value) or value<0):
                raise ValueError("Invalid runtime observation")
            row={"key":key,"method_id":method,"mechanism_id":identity["mechanism_id"],
                "pair_repeat_id":identity["pair_repeat_id"],"runtime_seconds":value,
                "peak_rss_bytes":record.get("peak_rss_bytes"),"boundary":record["boundary"],
                "j_interval_records":record.get("interval_runtime_seconds",()),
                "execution_status":loaded.bundles[key].execution_status}
            grouped[identity["mechanism_id"]].append(row);rows.append(row)
        mechanisms=[]
        for mid,values in sorted(grouped.items()):
            valid=all(r["runtime_seconds"] is not None for r in values)
            mechanisms.append({"mechanism_id":mid,"n_pairs":len(values),
                "seconds":sum(r["runtime_seconds"] for r in values) if valid else None,
                "failed_output_pairs":sum(r["execution_status"]!="OK" for r in values)})
        complete=bool(mechanisms) and all(r["seconds"] is not None for r in mechanisms)
        values=np.array([r["seconds"] for r in mechanisms if r["seconds"] is not None],dtype=np.float64)
        summaries[method]={"status":"COMPLETE" if complete else "UNAVAILABLE",
            "unit":"seconds per mechanism (sum of its Pair point runtimes)",
            "n_requested":len(mechanisms),"n_valid":len(values),
            "n_failed_or_missing":len(mechanisms)-len(values),
            "mean":float(values.mean()) if complete else None,
            "standard_deviation":float(values.std(ddof=1)) if complete and len(values)>1 else None,
            "median":float(np.median(values)) if complete else None,
            "p95":float(np.quantile(values,.95,method="linear")) if complete else None,
            "mechanisms":mechanisms,
            "definition":"Mechanism-equal; sample SD ddof=1; P95 linear quantile; failed outputs retain incurred runtime"}
    return {"methods":summaries,"raw_pairs":rows,"run_setup_and_timing":loaded.resources,
            "rss_definition":"Retain original per-request sampled windows, not a claimed process lifetime peak",
            "context_point_times":"Not independently measured; never assign half of Pair time"}
