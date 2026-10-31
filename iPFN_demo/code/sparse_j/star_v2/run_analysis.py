"""Frozen v2 analysis CLI. Reads independently accepted Test only; no models."""
from pathlib import Path
import argparse,io,json
import numpy as np
from . import analysis_loader as loader,analysis_schema as wire,analysis_tables as tables
from . import analysis_figures as figures,data_artifacts as data,provenance,run_test
from .estimation import EFFECT_ORDER

def _write_npz(root,core):
    arrays=tables.bootstrap_arrays(core);output=io.BytesIO()
    np.savez(output,**arrays)
    receipt=data._write(root,"bootstrap/distributions.npz",output.getvalue())
    header={"schema_version":1,"arrays":{name:{"dtype":a.dtype.str,"shape":list(a.shape),
        "sha256_c_bytes":data.sha(a.tobytes(order="C"))} for name,a in arrays.items()},
        "metadata":wire.plain(tables.bootstrap_metadata(core))}
    meta=wire.write_json(root,"bootstrap/distributions.json",header)
    return receipt,meta

def _attempt_rows(loaded):
    for attempt in loaded.registry.attempts:
        doc=attempt.document
        yield {"execution_key":attempt.execution_key,"attempt_no":doc["attempt_no"],
            "identification_status":doc["status"],"execution_status":doc["execution_status"],
            "output_status":doc["output_status"],"failure_type":doc["failure_type"],
            "failure_message":doc["failure_message"],"failure_provenance":doc["failure_provenance"]}

def _raw_pair_rows(loaded):
    truth={t.mechanism_id:t for t in loaded.truths}
    for key,bundle in sorted(loaded.bundles.items()):
        if key[1]!="pair":continue
        mid=bundle.request.identity["mechanism_id"]
        for k,effect in enumerate(EFFECT_ORDER):
            valid=bundle.output_status["ate"][k]=="OK"
            yield {"execution_key":key,"mechanism_id":mid,"pair_repeat_id":bundle.request.identity["pair_repeat_id"],
                "effect":effect,"status":bundle.output_status["ate"][k],
                "predicted":float(bundle.ate[k]) if valid else None,
                "truth":float(truth[mid].ate[k]),"error":float(bundle.ate[k]-truth[mid].ate[k]) if valid else None,
                "failure_provenance":bundle.failure_provenance,"reference_only":bundle.method_capability.reference_only}

def _oracle_intervals(loaded):
    for key,bundle in sorted(loaded.bundles.items()):
        if not key[0].startswith("oracle-"):continue
        for record in bundle.intervals:
            yield {"execution_key":key,"scope":key[1],"effect":record.effect,"context_id":record.context_id,
                "status":record.status,"lower":record.lower,"upper":record.upper,"level":record.level,
                "source":record.source,"failure_reason":bundle.failure_message if record.status!="OK" else None,
                "bundle_failure_provenance":bundle.failure_provenance,"enters_j_denominator":False}

def _json_lines(root,name,rows):
    path=data._target(root,name);path.parent.mkdir(parents=True,exist_ok=True);count=0
    import os
    with path.open("xb") as stream:
        for row in rows:stream.write(wire.blob(row));count+=1
        stream.flush();os.fsync(stream.fileno())
    raw=path.read_bytes()
    return {"path":name,"bytes":len(raw),"sha256":data.sha(raw),"rows":count}

def _write_loaded(loaded,output,*,synthetic_indices=None):
    """Private synthetic seam; production CLI never accepts synthetic indices."""
    target=provenance._fresh_directory(output)
    target.mkdir()
    receipts=[]
    try:
        core=tables.evaluate_core(loaded,synthetic_indices=synthetic_indices)
        tab=tables.summary_tables(loaded,core)
        tab["support_strata"]=[]
        for method,rows,report in tables.support_reports(loaded):
            if method=="oracle-context":tab["oracle_context"].extend(rows)
            else:tab["support_strata"].extend(rows)
            receipts.append(_json_lines(target,"raw/support_"+method+".jsonl",report["raw_queries"]))
            receipts.append(wire.write_json(target,"raw/support_counts_"+method+".json",report["support_counts"]))
        table_receipts={}
        for name,rows in sorted(tab.items()):
            receipt=wire.write_csv(target,"tables/"+name+".csv",wire.COMMON_COLUMNS,rows)
            table_receipts[name]=receipt;receipts.append(receipt)
        for effect in EFFECT_ORDER:
            rows=[r for r in tab["pair_ate"] if r["effect"]==effect]
            receipts.append(wire.write_csv(target,"tables/effect_"+effect+".csv",wire.COMMON_COLUMNS,rows))
        receipts.extend(_write_npz(target,core))
        receipts.append(wire.write_json(target,"bootstrap/metadata.json",tables.bootstrap_metadata(core)))
        receipts.append(_json_lines(target,"raw/attempts.jsonl",_attempt_rows(loaded)))
        receipts.append(_json_lines(target,"raw/pair_errors.jsonl",_raw_pair_rows(loaded)))
        receipts.append(_json_lines(target,"raw/j_intervals.jsonl",core["coverage"]["records"]))
        receipts.append(_json_lines(target,"raw/oracle_intervals.jsonl",_oracle_intervals(loaded)))
        receipts.append(_json_lines(target,"raw/source_records.jsonl",loaded.input_records))
        receipts.append(wire.write_json(target,"evaluation/pair_metrics.json",core["pair"]))
        receipts.append(wire.write_json(target,"evaluation/node_metrics.json",core["nodes"]))
        receipts.append(wire.write_json(target,"evaluation/center_leaf_metrics.json",core["center_leaf"]))
        receipts.append(wire.write_json(target,"evaluation/j_coverage.json",core["coverage"]))
        receipts.append(wire.write_json(target,"evaluation/rankability.json",core["rankability"]))
        receipts.append(wire.write_json(target,"evaluation/runtime.json",core["runtime"]))
        receipts.extend(figures.render_figures(target,tab,table_receipts,loaded.provenance))
        receipts.append(wire.write_json(target,"input_provenance.json",loaded.provenance))
        text=("# Frozen v2 Analysis\n\nEngineering output only; independently validate before acceptance.\n"
              "No scientific performance PASS is asserted by artifact completeness.\n"
              "Oracle reports are independent; Hajek is diagnostics only.\n"
              "Rankability is per effect; no composite score or ordinal ranks are invented.\n"
              "Coverage denominator includes every pre-requested interval, including failures.\n"
              "Only paired RMSE differences and J coverage have pointwise 95% Bootstrap intervals.\n"
              "Other frozen metrics are point estimates; no additional CIs, p-values or multiplicity adjustments.\n")
        receipts.append(data._write(target,"ANALYSIS_REPORT.md",text.encode("ascii")))
        paths=[r["path"] for r in receipts]
        if len(paths)!=len(set(paths)):raise ValueError("Duplicate analysis output path")
        manifest={"schema_version":1,"kind":"V2_ANALYSIS_MANIFEST","scope":core["bootstrap"].evidence_scope,
            "input_provenance":loaded.provenance,"registry_seal":loaded.registry.seal_sha256,
            "files":sorted(receipts,key=lambda r:r["path"]),
            "table_rows":{name:len(rows) for name,rows in tab.items()},
            "comparisons":21,"effect_comparisons":63,"j_coverage_distributions":9,
            "method_order":core["bootstrap"].method_order,"effect_order":EFFECT_ORDER,
            "replicate_count":len(core["bootstrap"].replicate_ids),
            "mechanism_draw_count":core["bootstrap"].indices.shape[1]}
        m=wire.write_json(target,"analysis_manifest.json",manifest)
        report=wire.write_json(target,"acceptance_report.json",{"schema_version":1,
            "analysis_status":"COMPLETE","engineering_acceptance":"PENDING_INDEPENDENT_VALIDATION",
            "manifest_sha256":m["sha256"],"input_provenance":loaded.provenance,
            "scientific_performance_pass_claimed":False})
        return {"manifest_sha256":m["sha256"],"report_sha256":report["sha256"],
                "scope":core["bootstrap"].evidence_scope,"output":str(target)}
    except BaseException as error:
        wire.write_json(target,"ANALYSIS_INCOMPLETE.json",{"status":"INCOMPLETE",
            "failure_type":type(error).__name__,"failure_message":str(error)})
        raise

def analyze_test(root,output,**anchors):
    target=run_test.validate_output_isolation(output,runtime_root=Path(__file__).resolve().parents[1],
                                             source_delivery=anchors["source_delivery"])
    if target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Analysis outputs cannot be inside immutable input Run")
    loaded=loader.load_accepted_test(root,**anchors)
    return _write_loaded(loaded,target)

def add_input_arguments(parser):
    for name in ("run-directory","run-manifest-sha256","completion-sha256","source-delivery",
                 "source-manifest-sha256","execution-receipt","execution-receipt-sha256",
                 "run-acceptance","run-acceptance-sha256"):
        parser.add_argument("--"+name,required=True)

def input_arguments(args):
    return dict(run_manifest_sha256=args.run_manifest_sha256,completion_sha256=args.completion_sha256,
        source_delivery=args.source_delivery,source_manifest_sha256=args.source_manifest_sha256,
        execution_receipt_path=args.execution_receipt,execution_receipt_sha256=args.execution_receipt_sha256,
        acceptance_path=args.run_acceptance,acceptance_sha256=args.run_acceptance_sha256)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);add_input_arguments(parser)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(analyze_test(args.run_directory,args.output,**input_arguments(args)),sort_keys=True))
    return 0
if __name__=="__main__":main()
