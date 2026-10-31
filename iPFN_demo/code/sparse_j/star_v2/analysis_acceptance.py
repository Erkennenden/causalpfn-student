"""Independent frozen analysis validation; engineering PASS is not scientific PASS."""
from pathlib import Path
from itertools import zip_longest
from collections.abc import Mapping
import argparse,csv,hashlib,json
import numpy as np
from . import analysis_schema as wire,analysis_loader as loader,analysis_tables as tables
from . import data_artifacts as data,run_test
from .run_analysis import add_input_arguments,input_arguments

def require(value,message):
    if not value:raise ValueError(message)

def _verify_file(root,record):
    path=data._target(root,record["path"])
    with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
    require(path.stat().st_size==record["bytes"] and digest==record["sha256"],"Changed analysis output: "+record["path"])

def _cells(row):
    return {key:json.dumps(wire.plain(v),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)
        if isinstance(v,(Mapping,tuple,list)) else "" if v is None else str(v) for key,v in row.items()}

def _csv_equals(root,name,rows):
    with (root/name).open(encoding="utf-8",newline="") as stream:
        reader=csv.DictReader(stream)
        require(tuple(reader.fieldnames)==wire.COMMON_COLUMNS,"CSV columns changed")
        seen=set();count=0
        for actual,expected in zip_longest(reader,rows):
            require(actual is not None and expected is not None and actual==_cells(expected),
                    "Frozen metric/table mismatch: "+name)
            require(actual["row_id"] not in seen,"Duplicate table row identity")
            seen.add(actual["row_id"]);count+=1
        return count

def _json_lines_equals(root,name,rows):
    with (root/name).open(encoding="ascii") as stream:
        count=0
        for actual,expected in zip_longest(stream,rows):
            require(actual is not None and expected is not None
                and data.strict_json(actual)==wire.plain(expected),"Raw derived record mismatch: "+name)
            count+=1
        return count

def _validate_loaded(loaded,root,*,manifest_sha256,report_sha256,synthetic_indices=None):
    """Private explicit synthetic seam; public CLI cannot validate toy as Test."""
    root=Path(root)
    require(not (root/"ANALYSIS_INCOMPLETE.json").exists(),"Analysis is INCOMPLETE")
    require(data.sha((root/"acceptance_report.json").read_bytes())==report_sha256,"Analysis report SHA256 mismatch")
    report=wire.read_json(root,"acceptance_report.json")
    require(report["analysis_status"]=="COMPLETE" and report["manifest_sha256"]==manifest_sha256
        and report["scientific_performance_pass_claimed"] is False,"Complete engineering report required")
    require(data.sha((root/"analysis_manifest.json").read_bytes())==manifest_sha256,"Analysis manifest SHA256 mismatch")
    manifest=wire.read_json(root,"analysis_manifest.json")
    scope="FORMAL_TEST_PLAN" if synthetic_indices is None else "SYNTHETIC"
    require(manifest["scope"]==scope,"Formal validator rejects synthetic analysis")
    require(manifest["input_provenance"]==wire.plain(loaded.provenance)
        and report["input_provenance"]==wire.plain(loaded.provenance)
        and manifest["registry_seal"]==loaded.registry.seal_sha256,"Analysis input provenance mismatch")
    paths=[r["path"] for r in manifest["files"]]
    require(len(paths)==len(set(p.casefold() for p in paths)),"Duplicate analysis file")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual==set(paths)|{"analysis_manifest.json","acceptance_report.json"},"Missing/extra analysis files")
    for record in manifest["files"]:_verify_file(root,record)
    # Recover accepted inputs independently and recompute through the immutable
    # scientific modules, not from values written in the derived CSV.
    core=tables.evaluate_core(loaded,synthetic_indices=synthetic_indices)
    expected=tables.summary_tables(loaded,core);expected["support_strata"]=[]
    for method,rows,support in tables.support_reports(loaded):
        expected["oracle_context" if method=="oracle-context" else "support_strata"].extend(rows)
        _json_lines_equals(root,"raw/support_"+method+".jsonl",support["raw_queries"])
        require(wire.read_json(root,"raw/support_counts_"+method+".json")==wire.plain(support["support_counts"]),
                "Support count table mismatch")
    row_counts={name:_csv_equals(root,"tables/"+name+".csv",rows) for name,rows in expected.items()}
    require(row_counts==manifest["table_rows"],"Analysis row inventory mismatch")
    for effect in core["bootstrap"].effect_order:
        _csv_equals(root,"tables/effect_"+effect+".csv",(r for r in expected["pair_ate"] if r["effect"]==effect))
    require(len(expected["comparison_index"])==21 and len(expected["comparisons"])==63
        and len(expected["j_coverage"])==9 and len(expected["rankability"])==21,
        "21 comparisons/63 effects/9 coverage/21 qualification inventory mismatch")
    ordinary=set(core["bootstrap"].method_order)
    require(ordinary==set(("state-mean","ht","f-z2","lin-z2","kernel-old","kernel-new","j"))
        and all(r["method_id"] in ordinary for r in expected["rankability"])
        and all(r["reference_only"] for r in expected["oracle_context"]+expected["oracle_pair"]),
        "Oracle/Hajek ranking contamination")
    required={"tables/"+name+".csv" for name in expected}
    required.update("tables/effect_"+effect+".csv" for effect in core["bootstrap"].effect_order)
    figure_names=("pair_ate","comparisons","j_coverage","node_error","group_ate","center_leaf",
                  "support_strata","runtime","oracle_context","oracle_pair")
    required.update("figures/"+name+suffix for name in figure_names for suffix in (".svg",".png",".metadata.json"))
    required.update("raw/support_"+method+".jsonl" for method in ("kernel-old","kernel-new","j","oracle-context"))
    required.update("raw/support_counts_"+method+".json" for method in ("kernel-old","kernel-new","j","oracle-context"))
    required.update(("bootstrap/distributions.npz","bootstrap/distributions.json","bootstrap/metadata.json",
        "raw/attempts.jsonl","raw/pair_errors.jsonl","raw/j_intervals.jsonl","raw/oracle_intervals.jsonl",
        "raw/source_records.jsonl","input_provenance.json","ANALYSIS_REPORT.md"))
    required.update("evaluation/"+name+".json" for name in ("pair_metrics","node_metrics","center_leaf_metrics",
                                                       "j_coverage","rankability","runtime"))
    require(set(paths)==required,"Frozen analysis file schema mismatch")
    records={r["path"]:r for r in manifest["files"]}
    descriptors=tuple({k:records[name][k] for k in ("path","bytes","sha256")}
        for name in ("bootstrap/distributions.npz","bootstrap/distributions.json"))
    arrays,metadata=data.load_arrays(root,descriptors)
    expected_arrays=tables.bootstrap_arrays(core)
    require(set(arrays)==set(expected_arrays),"Bootstrap array inventory mismatch")
    for name,value in expected_arrays.items():
        require(arrays[name].dtype==value.dtype and arrays[name].shape==value.shape
            and np.array_equal(arrays[name],value,equal_nan=True),"Bootstrap result mismatch: "+name)
    require(metadata==wire.plain(tables.bootstrap_metadata(core))
        and wire.read_json(root,"bootstrap/metadata.json")==metadata,"Bootstrap metadata mismatch")
    if synthetic_indices is None:
        require(arrays["indices"].shape==(10000,200),"Frozen Bootstrap dimensions changed")
        reference=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261030,2026103001,2,0,7,0])))
        for i in range(10000):
            require(np.array_equal(arrays["indices"][i],reference.integers(0,200,200,dtype=np.int64)),
                    "Independent PCG64 reference mismatch")
        for effect,item in core["coverage"]["effects"].items():
            require(item["all_requested"]==1200,"Coverage failed-interval denominator changed")
    for name,value in (("pair_metrics",core["pair"]),("node_metrics",core["nodes"]),
                       ("center_leaf_metrics",core["center_leaf"]),("j_coverage",core["coverage"]),
                       ("rankability",core["rankability"]),("runtime",core["runtime"])):
        require(wire.read_json(root,"evaluation/"+name+".json")==wire.plain(value),
                "Frozen result JSON mismatch: "+name)
    _json_lines_equals(root,"raw/j_intervals.jsonl",core["coverage"]["records"])
    _json_lines_equals(root,"raw/source_records.jsonl",loaded.input_records)
    _json_lines_equals(root,"raw/attempts.jsonl",(
        {"execution_key":a.execution_key,"attempt_no":a.document["attempt_no"],
         "identification_status":a.document["status"],"execution_status":a.document["execution_status"],
         "output_status":a.document["output_status"],"failure_type":a.document["failure_type"],
         "failure_message":a.document["failure_message"],"failure_provenance":a.document["failure_provenance"]}
         for a in loaded.registry.attempts))
    # Figure metadata must point to the actual validated table and its rows;
    # binary figure integrity is anchored by the external manifest receipt.
    for name in ("pair_ate","comparisons","j_coverage","node_error","group_ate","center_leaf",
                 "support_strata","runtime","oracle_context","oracle_pair"):
        metric={"comparisons":"paired_rmse_difference","j_coverage":"coverage_including_failures",
                "runtime":"mean"}.get(name,"pooled_rmse")
        selected=[r for r in expected[name] if r["metric"]==metric]
        if name in ("oracle_context","oracle_pair"):
            selected=[r for r in selected if r["source_selector"].get("family")=="pair_ate"]
        if name=="support_strata":
            selected=[r for r in selected if r["group"]=="all" and r["center_t"] is None]
        meta=wire.read_json(root,"figures/"+name+".metadata.json")
        require(meta["input_provenance"]==wire.plain(loaded.provenance)
            and meta["source_table"]==records["tables/"+name+".csv"]
            and meta["source_row_ids"]==[r["row_id"] for r in selected]
            and meta["rows"]==wire.plain(selected)
            and meta["scientific_performance_pass_claimed"] is False,
            "Figure provenance/denominator/CI mismatch")
    return {"status":"PASS","scope":scope,"scientific_performance_pass_claimed":False,
        "source_run_id":loaded.provenance["run_id"],"source_snapshot_id":loaded.provenance["source_snapshot_id"],
        "analysis_manifest_sha256":manifest_sha256,"analysis_report_sha256":report_sha256,
        "files_verified":len(paths)+2,"table_rows":row_counts,"comparisons":21,"effect_comparisons":63,
        "j_coverage_distributions":9,"oracle_separate":True,"hajek_ranked":False}

def validate_analysis(root,*,analysis_directory,manifest_sha256,report_sha256,**anchors):
    target=run_test.validate_output_isolation(analysis_directory,
        runtime_root=Path(__file__).resolve().parents[1],source_delivery=anchors["source_delivery"])
    if target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Analysis cannot be placed in immutable input Run")
    loaded=loader.load_accepted_test(root,**anchors)
    return _validate_loaded(loaded,target,manifest_sha256=manifest_sha256,report_sha256=report_sha256)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);add_input_arguments(parser)
    for name in ("analysis-directory","manifest-sha256","report-sha256"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(validate_analysis(args.run_directory,analysis_directory=args.analysis_directory,
        manifest_sha256=args.manifest_sha256,report_sha256=args.report_sha256,**input_arguments(args)),sort_keys=True))
    return 0
if __name__=="__main__":main()
