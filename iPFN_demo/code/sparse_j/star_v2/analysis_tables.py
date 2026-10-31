"""Frozen metric invocation and v2 tidy tables. No estimator or new score."""
from itertools import combinations
import numpy as np
from . import metrics,coverage,bootstrap
from .support import ORDINARY_METHOD_IDS,method_capability,REASONS
from .estimation import EFFECT_ORDER
from .analysis_runtime import summarize_runtime
from . import analysis_schema as wire

def evaluate_core(loaded,*,synthetic_indices=None):
    pairs=tuple(p for p in loaded.predictions if p.key[1]=="pair")
    if synthetic_indices is None:
        evidence=bootstrap.mechanism_block_bootstrap(loaded.registry,loaded.predictions,
            loaded.truths,tuple(loaded.context_supports.items()))
        cov=coverage.evaluate_j_coverage(loaded.registry,loaded.truths)
    else:
        collection=bootstrap.build_mechanism_blocks(loaded.registry,loaded.predictions,
            loaded.truths,tuple(loaded.context_supports.items()),formal=False)
        evidence=bootstrap.bootstrap_from_synthetic_indices(collection,synthetic_indices)
        cov=collection.coverage
    return {"bootstrap":evidence,"coverage":cov,"pair":evidence.point_metrics,
        "rankability":evidence.rankability,
        "nodes":metrics.node_metrics(loaded.plan,loaded.predictions,loaded.truths),
        "center_leaf":metrics.center_leaf_metrics(loaded.plan,loaded.predictions,loaded.truths),
        "runtime":summarize_runtime(loaded)}

def _metric_rows(provenance,method,effect,summary,names,*,group="all",family="pair_ate"):
    reference=method_capability(method).reference_only
    for metric in names:
        yield wire.table_row(provenance,method=method,effect=effect,metric=metric,
            value=summary[metric],status=summary["status"],
            n_requested=summary["n_requested"],n_valid=summary["n_valid"],group=group,
            unit=wire.metric_unit(metric),reference=reference,
            selector={"method_id":method,"scope":"pair" if family=="pair_ate" else "context",
                      "effect":effect,"group":group,"family":family},
            details={"statistical_unit":"mechanism","n_mechanisms_requested":summary["n_mechanisms_requested"],
                     "n_mechanisms_with_valid":summary["n_mechanisms_with_valid"]})
        if summary["status"]=="INCOMPLETE":
            yield wire.table_row(provenance,method=method,effect=effect,
                metric="success_subset_"+metric,value=summary["success_subset_"+metric],
                status="DESCRIPTIVE_SUCCESS_SUBSET_ONLY",n_requested=summary["n_requested"],
                n_valid=summary["n_valid"],group=group,unit=wire.metric_unit(metric),reference=reference,
                selector={"method_id":method,"scope":"pair" if family=="pair_ate" else "context",
                          "effect":effect,"group":group,"family":family,"conditional_on_valid":True})

def summary_tables(loaded,core):
    p=loaded.provenance;evidence=core["bootstrap"]
    tables={name:[] for name in ("pair_ate","node_error","group_ate","center_leaf","runtime",
        "rankability","comparisons","comparison_index","j_coverage","capability",
        "oracle_context","oracle_pair")}
    for bucket,methods in core["pair"].items():
        for method,effects in methods.items():
            rows=[row for effect,summary in effects.items()
                  for row in _metric_rows(p,method,effect,summary,wire.PAIR_METRICS)]
            name="pair_ate" if bucket=="ordinary" else "oracle_context" if method=="oracle-context" else "oracle_pair"
            tables[name].extend(rows)
    for bucket,methods in core["nodes"].items():
        for method,report in methods.items():
            if report.get("status")=="UNSUPPORTED":
                tables["capability"].append(wire.table_row(p,method=method,effect="all",metric="node_outputs",
                    value=None,status="UNSUPPORTED",n_requested=0,n_valid=0,unit="not applicable",
                    reference=method_capability(method).reference_only))
                continue
            for group,items in report["groups"].items():
                for output in ("mu","ite"):
                    effects={"mu":items["mu"]} if output=="mu" else items["ite"]
                    for effect,summary in effects.items():
                        rows=list(_metric_rows(p,method,effect,summary,("mse","pooled_rmse"),
                            group=group,family="node_"+output))
                        tables["node_error"].extend(rows)
                for effect,summary in items["group_ate"].items():
                    tables["group_ate"].extend(_metric_rows(p,method,effect,summary,
                        wire.GROUP_METRICS,group=group,family="group_ate"))
    for bucket,methods in core["center_leaf"].items():
        for method,report in methods.items():
            if report.get("status")=="UNSUPPORTED":continue
            for effect,summary in report["effects"].items():
                tables["center_leaf"].extend(_metric_rows(p,method,effect,summary,
                    wire.DELTA_METRICS,group="center-minus-leaf",family="center_leaf_difference"))
    for method,report in core["runtime"]["methods"].items():
        for metric in ("mean","standard_deviation","median","p95"):
            tables["runtime"].append(wire.table_row(p,method=method,effect="all",metric=metric,
                value=report[metric],status=report["status"],unit=report["unit"],
                n_requested=report["n_requested"],n_valid=report["n_valid"],
                reference=method_capability(method).reference_only,
                details={"formula":report["definition"],"bootstrap_ci":"NOT_REQUESTED",
                         "failed_estimations_retain_incurred_runtime":True}))
    rank=core["rankability"]["methods"]
    for method in ORDINARY_METHOD_IDS:
        for effect in EFFECT_ORDER:
            item=rank[method]["effects"][effect]
            tables["rankability"].append(wire.table_row(p,method=method,effect=effect,
                metric="rankability",value=None,status=item["rankability_status"],unit="qualification",
                n_requested=item["requested_count"],n_valid=item["valid_count"],
                details={"failure_reasons":item["failure_reasons"],"reason_distribution":item["reason_distribution"],
                         "ordinal_rank":"NOT_PRODUCED_BY_FROZEN_CORE","hajek":"DIAGNOSTIC_ONLY"}))
    for comparison in evidence.comparisons:
        valid_effects=0
        for effect,dist in comparison.effects.items():
            a=rank[comparison.method_a]["effects"][effect];b=rank[comparison.method_b]["effects"][effect]
            valid_a={r["pair_id"] for r in a["records"] if r["valid"]}
            valid_b={r["pair_id"] for r in b["records"] if r["valid"]}
            ok=dist.status=="OK";valid_effects+=ok
            value=a["statistics"]["pooled_rmse"]-b["statistics"]["pooled_rmse"] if ok else None
            tables["comparisons"].append(wire.table_row(p,method=comparison.method_a+" minus "+comparison.method_b,
                effect=effect,metric="paired_rmse_difference",value=value,status=dist.status,
                n_requested=a["requested_count"],n_valid=len(valid_a&valid_b),
                ci=dist.interval,ci_status=dist.status,unit="outcome",
                selector={"methods":[comparison.method_a,comparison.method_b],"scope":"pair","effect":effect},
                details={"method_a":comparison.method_a,"method_b":comparison.method_b,
                    "statistical_unit":"mechanism","requested_a":a["requested_count"],"requested_b":b["requested_count"],
                    "valid_a":a["valid_count"],"valid_b":b["valid_count"],"failure_reasons":dist.failure_reasons}))
        tables["comparison_index"].append(wire.table_row(p,
            method=comparison.method_a+" minus "+comparison.method_b,effect="all",
            metric="effect_comparison_inventory",value=None,status="COMPLETE" if valid_effects==3 else "CONTAINS_NOT_COMPARABLE",
            n_requested=3,n_valid=valid_effects,unit="effect-specific records",
            details={"method_a":comparison.method_a,"method_b":comparison.method_b,
                "role":"index only, not a score","effect_records":3}))
    for scope,effects in evidence.j_coverage.items():
        for effect,dist in effects.items():
            point=core["coverage"]["effects"][effect] if scope=="overall" else core["coverage"]["center_t"][
                0 if scope=="center_t=0" else 1][effect]
            center=None if scope=="overall" else 0 if scope=="center_t=0" else 1
            tables["j_coverage"].append(wire.table_row(p,method="j",effect=effect,
                metric="coverage_including_failures",value=point["coverage_including_failures"],
                status="COMPLETE",unit="proportion",n_requested=point["all_requested"],
                n_valid=point["valid_intervals"],center_t=center,ci=dist.interval,ci_status=dist.status,
                details={**dict(point),"scope":scope,"threshold_overall_only":.90 if center is None else None,
                    "interval_level":.95,"statistical_unit":"mechanism",
                    "coverage_valid_ci":"NOT_REQUESTED","oracle_intervals_in_denominator":False}))
    for name in ("node_error","group_ate","center_leaf","runtime","capability"):
        ordinary=[]
        for row in tables[name]:
            if row["reference_only"]:
                tables["oracle_context" if row["method_id"]=="oracle-context" else "oracle_pair"].append(row)
            else:ordinary.append(row)
        tables[name]=ordinary
    for name,rows in tables.items():
        ids=[r["row_id"] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate derived table row: "+name)
    return tables

def support_reports(loaded):
    """Method-at-a-time avoids retaining four full node-state dictionaries."""
    for method in ("kernel-old","kernel-new","j","oracle-context"):
        plan=metrics.EvaluationPlan(tuple(t for t in loaded.plan.tasks if t.key[0]==method),
                                   source_seal=loaded.registry.seal_sha256)
        predictions=tuple(p for p in loaded.predictions if p.key[0]==method)
        supports=tuple((key,s) for key,s in loaded.method_supports if key[0]==method)
        result=metrics.support_stratification(plan,predictions,loaded.truths,supports)
        bucket="oracle_reference" if method=="oracle-context" else "ordinary"
        report=result[bucket][method]
        rows=[]
        for stratum in report["strata"]:
            for metric in ("mse","pooled_rmse","signed_bias"):
                rows.append(wire.table_row(loaded.provenance,method=method,effect="mu_state",
                    metric=metric,value=stratum[metric],status=stratum["status"],
                    n_requested=stratum["n_requested"],n_valid=stratum["n_valid"],
                    group=stratum["group"],center_t=stratum["center_t"],reason=stratum["reason"],
                    unit=wire.metric_unit(metric),reference=method=="oracle-context",
                    details={"statistical_unit":"mechanism","descriptive_only":True,
                        "conditional_on_valid_predictions":True,"per_mechanism":stratum["per_mechanism"]}))
        yield method,rows,report

def bootstrap_arrays(core):
    e=core["bootstrap"];n=len(e.replicate_ids)
    comparisons=np.full((n,21,3),np.nan,dtype=np.float64)
    for j,comparison in enumerate(e.comparisons):
        for k,effect in enumerate(EFFECT_ORDER):
            if comparison.effects[effect].status=="OK":comparisons[:,j,k]=comparison.effects[effect].values
    cover=np.stack([np.column_stack([e.j_coverage[scope][effect].values for effect in EFFECT_ORDER])
                    for scope in bootstrap.COVERAGE_SCOPES],axis=1)
    return {"indices":e.indices,"replicate_ids":e.replicate_ids,"draw_positions":e.draw_positions,
            "method_rmse":e.method_rmse,"paired_rmse_differences":comparisons,"j_coverage":cover}

def bootstrap_metadata(core):
    e=core["bootstrap"]
    return {"schema_version":1,"evidence_scope":e.evidence_scope,
        "rng_metadata":e.rng_metadata,"numpy_version":e.numpy_version,
        "method_order":e.method_order,"effect_order":e.effect_order,"coverage_scopes":bootstrap.COVERAGE_SCOPES,
        "comparisons":[{"method_a":c.method_a,"method_b":c.method_b,
            "effects":{effect:{"status":d.status,"interval":d.interval,"failure_reasons":d.failure_reasons}
                for effect,d in c.effects.items()}} for c in e.comparisons],
        "replicate_count":len(e.replicate_ids),"mechanism_draw_count":e.indices.shape[1],
        "ci_level":.95,"quantile_method":"linear","interpretation":e.interval_interpretation,
        "coverage_ci_including_failures_only":True,"registry_seal":e.registry_seal}
