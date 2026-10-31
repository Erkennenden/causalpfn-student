"""Presentation-only views of the accepted E1.5 Star v2 analysis.

No scientific estimators, model loaders or bootstrap modules are imported.
Existing statistics and raw coordinates are copied, never re-estimated.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import sys
import tempfile
import time
import unittest

RUN = "e1_5_star_v2_test_20261031_01"
SNAPSHOT = "99c8b9c9a1e75a9fac4d86d491aaf4505a47cdc5518e11525fc3092e1f73113d"
MANIFEST_SHA = "8e7ed35b92b360ef29f6290c4a1a4ff200f354d27626986ec76fb99e9db3b08b"
REPORT_SHA = "a393b2190dc60486f045ecb211a3ad1a391fdff7f60481e3a201940dbf58ab34"
ACCEPTANCE_SHA = "56875e3b0e5014de887a72b7ed0cd3362aa4747e59ae7429e6b2662726077d12"
TEMPLATES = {
    "make_figures.py": "a6f955591d69316d96c44912360337b14ac03ff6d2fa24e82ec6850ec2b063ca",
    "make_table.py": "78bd9b24f7385942ae4dfb054c1e1f82ede41a93bc5097253db203327a2e45b3",
}
NORMALIZED_TEMPLATES = {
    "make_figures.py": "f07191b11e776df5e97096a2f73b8f8f3abde2c02aea8394c90b95cf2bdb1e9a",
    "make_table.py": "e78f6d493839bcd81ac7c12f59d8a94b4389e494beb8ef31ae4f20694d02e527",
}
METHODS = ("state-mean", "ht", "f-z2", "lin-z2", "kernel-old", "kernel-new", "j")
NODE_METHODS = ("j", "kernel-old", "kernel-new")
EFFECTS = ("Direct", "Spillover", "Total")
METRICS = {
    "rmse": "pooled_rmse", "signed_bias": "signed_bias",
    "absolute_bias": "absolute_bias", "acb": "absolute_conditional_bias",
    "within_variance": "within_mechanism_variance",
    "between_variability": "between_mechanism_variability",
}
COORDINATES = ("true_center", "predicted_center", "true_leaf", "predicted_leaf",
               "true_difference", "predicted_difference", "error")
MAIN_COLUMNS = ("method_id", "effect", *METRICS, "n_requested", "n_valid",
                "n_failed_or_missing", "rankability", "source_row_ids",
                "source_run_id", "source_snapshot_id")
J_COLUMNS = ("effect", "scope", "center_t", "coverage", "ci_lower", "ci_upper",
             "ci_level", "ci_status", "mean_interval_width", "width_ci_status",
             "nominal_interval_level", "covered", "n_requested", "n_valid",
             "n_failed_or_missing", "interval_success_rate", "meets_90pct_target",
             "source_row_id", "source_run_id", "source_snapshot_id")
COORD_COLUMNS = ("method_id", "effect", "context_id", "request_id", "mechanism_id",
                 "pair_repeat_id", "center_t", *COORDINATES, "source_json_pointer",
                 "source_run_id", "source_snapshot_id")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def bad_constant(value):
        raise ValueError("Nonfinite JSON: " + value)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=bad_constant, object_pairs_hook=unique_pairs)


def relative_path(name):
    require(type(name) is str and name and "\\" not in name and ":" not in name
            and "\0" not in name, "Unsafe relative path")
    parsed = PurePosixPath(name)
    require(not parsed.is_absolute() and str(parsed) == name
            and all(p not in ("", ".", "..") for p in name.split("/")),
            "Traversal/noncanonical relative path")
    return name


def safe_path(path, outside_git=False):
    path = Path(os.path.abspath(path))
    for ancestor in (path, *path.parents):
        if os.path.lexists(ancestor):
            info = ancestor.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not
                    (getattr(info, "st_file_attributes", 0) & 1024),
                    "Link/reparse point: " + str(ancestor))
        if outside_git:
            require(not os.path.lexists(ancestor / ".git"),
                    "Git worktree ancestor: " + str(ancestor))
    return path


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "Duplicate/missing CSV columns")
        rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            "Malformed CSV row")
    return rows


def identity(row):
    require(row["source_run_id"] == RUN and row["source_snapshot_id"] == SNAPSHOT,
            "Formal identity mismatch")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_rows_unique(rows):
    require(len({row["row_id"] for row in rows}) == len(rows), "Duplicate source row ID")
    for row in rows:
        identity(row)


def extract_main(rows, ranks):
    source_rows_unique(rows)
    source_rows_unique(ranks)
    expected = {(m, e, metric) for m in METHODS for e in EFFECTS for metric in METRICS.values()}
    keys = [(r["method_id"], r["effect"], r["metric"]) for r in rows]
    require(len(rows) == 126 and len(set(keys)) == 126 and set(keys) == expected,
            "Main metric inventory must be 126")
    rankkeys = [(r["method_id"], r["effect"]) for r in ranks]
    require(len(ranks) == 21 and len(set(rankkeys)) == 21
            and set(rankkeys) == {(m, e) for m in METHODS for e in EFFECTS},
            "Rankability inventory must be 21")
    lookup = dict(zip(keys, rows))
    ranklookup = dict(zip(rankkeys, ranks))
    result = []
    for method in METHODS:
        for effect in EFFECTS:
            members = [lookup[method, effect, metric] for metric in METRICS.values()]
            counts = {(r["n_requested"], r["n_valid"], r["n_failed_or_missing"]) for r in members}
            require(counts == {("600", "600", "0")}
                    and all(r["status"] == "COMPLETE" and r["ci_status"] == "NOT_REQUESTED"
                            and not r["ci_lower"] and not r["ci_upper"] for r in members),
                    "Main cohort/count/status mismatch")
            require(ranklookup[method, effect]["status"] == "RANKABLE",
                    "Unexpected ranking qualification; do not repair")
            result.append(dict(method_id=method, effect=effect,
                **{dest: lookup[method, effect, metric]["value"] for dest, metric in METRICS.items()},
                n_requested="600", n_valid="600", n_failed_or_missing="0",
                rankability="RANKABLE",
                source_row_ids={dest: lookup[method, effect, metric]["row_id"]
                                for dest, metric in METRICS.items()},
                source_run_id=RUN, source_snapshot_id=SNAPSHOT))
            require(all(finite(float(result[-1][metric])) for metric in METRICS),
                    "Invalid main metric value")
    return result


def extract_intervals(rows):
    source_rows_unique(rows)
    expected = {(e, c) for e in EFFECTS for c in ("", "0", "1")}
    keys = [(r["effect"], r["center_t"]) for r in rows]
    require(len(rows) == 9 and len(set(keys)) == 9 and set(keys) == expected,
            "J interval inventory must be 9")
    lookup = dict(zip(keys, rows))
    result = []
    for center in ("", "0", "1"):
        for effect in EFFECTS:
            row = lookup[effect, center]
            details = json.loads(row["details"])
            n = 1200 if center == "" else 600
            require(row["method_id"] == "j" and row["metric"] == "coverage_including_failures"
                    and row["status"] == "COMPLETE" and row["ci_status"] == "OK"
                    and row["ci_level"] == "0.95" and row["n_requested"] == str(n)
                    and row["n_valid"] == str(n) and row["n_failed_or_missing"] == "0",
                    "J interval identity/count/status mismatch")
            require(details["all_requested"] == n and details["valid_intervals"] == n
                    and details["failed_or_missing"] == 0 and details["interval_level"] == .95
                    and details["coverage_including_failures"] == float(row["value"])
                    and finite(details["mean_interval_width"]) and details["mean_interval_width"] >= 0,
                    "J interval detail mismatch")
            require(0 <= float(row["ci_lower"]) <= float(row["ci_upper"]) <= 1,
                    "Invalid existing coverage CI")
            require(details["threshold_overall_only"] == (.9 if center == "" else None),
                    "Coverage threshold scope changed")
            result.append(dict(effect=effect, scope=details["scope"], center_t=center,
                coverage=row["value"], ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                ci_level=row["ci_level"], ci_status=row["ci_status"],
                mean_interval_width=details["mean_interval_width"], width_ci_status="NOT_REQUESTED",
                nominal_interval_level=details["interval_level"], covered=details["covered_valid"],
                n_requested=row["n_requested"], n_valid=row["n_valid"],
                n_failed_or_missing=row["n_failed_or_missing"],
                interval_success_rate=details["interval_success_rate"],
                meets_90pct_target=details.get("meets_90pct_target"),
                source_row_id=row["row_id"], source_run_id=RUN, source_snapshot_id=SNAPSHOT))
    return result


def extract_coordinates(document):
    result = []
    common_truth = {}
    for method in NODE_METHODS:
        item = document["ordinary"][method]
        contexts = item["context_records"]
        require(len(contexts) == 1200, "Expected 1200 saved contexts per model")
        by_context = {tuple(r["key"]): r for r in contexts}
        require(len(by_context) == 1200, "Duplicate context record")
        for effect in EFFECTS:
            rows = item["effects"][effect]["raw_rows"]
            require(len(rows) == 1200, "Expected 1200 existing effect records")
            seen = set()
            for index, row in enumerate(rows):
                key = tuple(row["key"])
                require(len(key) == 4 and key[:2] == (method, "context") and key not in seen,
                        "Duplicate/malformed context identity")
                seen.add(key)
                require(key[3].startswith(RUN + ":" + method + ":")
                        and type(row["mechanism_id"]) is int and row["mechanism_id"] in range(200)
                        and type(row["pair_repeat_id"]) is int and row["pair_repeat_id"] in range(3)
                        and type(row["center_t"]) is int and row["center_t"] in (0, 1),
                        "Context Run/mechanism/repeat/center mismatch")
                parent = by_context[key]
                require(parent["mechanism_id"] == row["mechanism_id"]
                        and all(parent["effects"][effect][field] == row[field] for field in COORDINATES)
                        and all(finite(row[field]) for field in COORDINATES),
                        "Saved coordinate records disagree")
                # Binding check only; all output coordinates remain the original saved values.
                for actual, expected in (
                    (row["predicted_difference"], row["predicted_center"] - row["predicted_leaf"]),
                    (row["true_difference"], row["true_center"] - row["true_leaf"]),
                    (row["error"], row["predicted_difference"] - row["true_difference"])):
                    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                            "Saved coordinate arithmetic mismatch")
                truth_key = (row["mechanism_id"], row["pair_repeat_id"], row["center_t"], effect)
                truth = tuple(row[f] for f in ("true_center", "true_leaf", "true_difference"))
                require(truth_key not in common_truth or common_truth[truth_key] == truth,
                        "Cross-method truth mismatch")
                common_truth[truth_key] = truth
                result.append(dict(method_id=method, effect=effect, context_id=key[2], request_id=key[3],
                    mechanism_id=row["mechanism_id"], pair_repeat_id=row["pair_repeat_id"],
                    center_t=row["center_t"], **{field: row[field] for field in COORDINATES},
                    source_json_pointer=f"/ordinary/{method}/effects/{effect}/raw_rows/{index}",
                    source_run_id=RUN, source_snapshot_id=SNAPSHOT))
            grid = {(r["mechanism_id"], r["pair_repeat_id"], r["center_t"]) for r in rows}
            require(len(grid) == 1200 and grid == {(m, r, c) for m in range(200)
                                                  for r in range(3) for c in (0, 1)},
                    "Incomplete context grid")
            require(seen == set(by_context), "Effect/context inventory mismatch")
    require(len(result) == 10800, "Expected exactly 10800 original records")
    return result



def _template_payload(path, name):
    """Accept only the original bytes or their exact CRLF-to-LF representation."""
    require(name in TEMPLATES, "Unknown reference template")
    path = safe_path(path, True)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    normalized = payload.replace(b"\r\n", b"\n")
    normalized_sha = hashlib.sha256(normalized).hexdigest()
    require(normalized_sha == NORMALIZED_TEMPLATES[name],
            "Reference template normalized identity mismatch: " + name)
    require(actual in (TEMPLATES[name], NORMALIZED_TEMPLATES[name]),
            "Reference template representation mismatch: " + name)
    evidence = dict(
        path=str(path.resolve()), original_sha256=TEMPLATES[name],
        git_normalized_sha256=NORMALIZED_TEMPLATES[name], actual_sha256=actual,
        representation="original_bytes" if actual == TEMPLATES[name] else "git_lf_normalized",
        normalization_rule="CRLF_TO_LF_ONLY; no encoding, whitespace or other changes")
    return evidence, payload


def template_provenance(runtime):
    runtime = safe_path(runtime, True)
    return {name: _template_payload(runtime / name, name)[0] for name in TEMPLATES}

def load_inputs(analysis, receipt, runtime):
    analysis = safe_path(analysis, True)
    receipt = safe_path(receipt, True)
    runtime = safe_path(runtime, True)
    require(sha(receipt) == ACCEPTANCE_SHA, "Independent acceptance receipt SHA mismatch")
    acceptance = read_json(receipt)
    require(acceptance["status"] == "PASS" and acceptance["scope"] == "FORMAL_TEST_PLAN"
            and acceptance["source_run_id"] == RUN and acceptance["source_snapshot_id"] == SNAPSHOT
            and acceptance["analysis_manifest_sha256"] == MANIFEST_SHA
            and acceptance["analysis_report_sha256"] == REPORT_SHA
            and acceptance["files_verified"] == 72 and acceptance["comparisons"] == 21
            and acceptance["effect_comparisons"] == 63 and acceptance["j_coverage_distributions"] == 9
            and acceptance["oracle_separate"] is True and acceptance["hajek_ranked"] is False
            and acceptance["scientific_performance_pass_claimed"] is False,
            "Independent acceptance binding mismatch")
    require(sha(analysis / "analysis_manifest.json") == MANIFEST_SHA
            and sha(analysis / "acceptance_report.json") == REPORT_SHA, "Analysis anchor mismatch")
    manifest = read_json(analysis / "analysis_manifest.json")
    records = manifest["files"]
    names = [relative_path(r["path"]) for r in records]
    require(len(names) == 70 and len(set(n.casefold() for n in names)) == 70,
            "Frozen analysis file inventory mismatch")
    actual = []
    for path in analysis.rglob("*"):
        safe_path(path, True)
        if path.is_file():
            actual.append(path.relative_to(analysis).as_posix())
    require(set(actual) == set(names) | {"analysis_manifest.json", "acceptance_report.json"}
            and len(actual) == 72, "Accepted analysis contains missing/extra files")
    for record in records:
        path = analysis / record["path"]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                "Changed accepted input: " + record["path"])
    template_identity = template_provenance(runtime)
    selected = ("tables/pair_ate.csv", "tables/rankability.csv", "tables/j_coverage.csv",
                "evaluation/center_leaf_metrics.json")
    sources = {name: next(r for r in records if r["path"] == name) for name in selected}
    return dict(main=extract_main(csv_rows(analysis / selected[0]), csv_rows(analysis / selected[1])),
                intervals=extract_intervals(csv_rows(analysis / selected[2])),
                coordinates=extract_coordinates(read_json(analysis / selected[3])),
                sources=sources, analysis=analysis, receipt=receipt, runtime=runtime,
                template_identity=template_identity)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def wire_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return str(value)


def write_csv(path, columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        require(set(row) == set(columns), "Presentation CSV schema mismatch")
        writer.writerow({k: wire_cell(v) for k, v in row.items()})
    write_bytes(path, stream.getvalue().encode("utf-8"))


def load_templates(runtime, *, expected_identity=None):
    """Execute the verified byte buffers; never reopen an unchecked template."""
    runtime = safe_path(runtime, True)
    verified = {name: _template_payload(runtime / name, name) for name in TEMPLATES}
    if expected_identity is not None:
        require({name: value[0] for name, value in verified.items()} == expected_identity,
                "Template identity changed after input verification")
    names = (("make_table", "make_table.py"), ("_presentation_make_figures", "make_figures.py"))
    require(all(name not in sys.modules for name, _ in names), "Template module name collision")
    loaded = []
    try:
        for name, filename in names:
            evidence, payload = verified[filename]
            spec = importlib.util.spec_from_file_location(name, evidence["path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded.append(name)
            exec(compile(payload, evidence["path"], "exec"), module.__dict__)
    except BaseException:
        for name in loaded:
            sys.modules.pop(name, None)
        raise
    return sys.modules["make_table"], sys.modules["_presentation_make_figures"]


def render_main_tex(rows, intervals, template):
    escape = template.latex_escape
    names = {**template.METHOD_NAMES, "state-mean": "State mean"}
    labels = ("RMSE", "Signed Bias", "Absolute Bias", "ACB", "Within variance", "Between variability")
    lines = [r"% Requires booktabs; presentation only, no ordinal ranking."]
    for effect in EFFECTS:
        lines += [r"\begin{table}[tb]", r"\centering", r"\scriptsize",
                  r"\caption{Pair ATE accuracy: " + effect + r". $N=200$, $M=50$.}",
                  r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                  "Method & " + " & ".join(labels) + r" & Valid/requested \\", r"\midrule"]
        for row in rows:
            if row["effect"] == effect:
                lines.append(escape(names[row["method_id"]]) + " & "
                             + " & ".join(format(float(row[m]), ".4f") for m in METRICS)
                             + r" & " + row["n_valid"] + "/" + row["n_requested"] + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\par\smallskip\begin{minipage}{\linewidth}\scriptsize",
                  "200 independent mechanisms; 3 nested pair repeats each. "
                  "RMSE and bias have outcome units; both variance columns have squared outcome units. "
                  "Absolute Bias is the absolute overall signed bias; ACB is mean absolute mechanism bias. "
                  "Within variance uses R-1; between variability uses K-1. "
                  "All entries are RANKABLE qualifications, without ordinal ranks or rank shading. "
                  "Point estimates only; confidence intervals were not requested.",
                  r"\end{minipage}\end{table}"]
    lines += [r"\begin{table}[tb]\centering\scriptsize",
              r"\caption{J nominal 95\% posterior interval summary.}",
              r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Scope & Effect & Coverage (\%) & Coverage 95\% CI (\%) & Mean width & Valid/requested \\",
              r"\midrule"]
    for row in intervals:
        lines.append(escape(row["scope"]) + " & " + row["effect"] + " & "
                     + format(100 * float(row["coverage"]), ".2f") + " & ["
                     + format(100 * float(row["ci_lower"]), ".2f") + ", "
                     + format(100 * float(row["ci_upper"]), ".2f") + "] & "
                     + format(row["mean_interval_width"], ".4f") + " & "
                     + row["n_valid"] + "/" + row["n_requested"] + r" \\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\begin{minipage}{\linewidth}\scriptsize",
              r"Coverage CIs are existing pointwise 95\% mechanism-block Bootstrap intervals, "
              r"without multiplicity adjustment. The 90\% target applies to overall point coverage only; "
              "center strata are diagnostic. Failed intervals remain in coverage denominators. "
              "Mean widths condition on valid context posterior intervals and have no confidence interval.",
              r"\end{minipage}\end{table}"]
    return "\n".join(lines) + "\n"


def render_figures(output, data, template):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"j": "#2166AC", "kernel-old": "#999999", "kernel-new": "#762A83"}
    names = {**template.METHOD_NAMES, "state-mean": "State mean"}
    common = dict(source_run_id=RUN, source_snapshot_id=SNAPSHOT, presentation_only=True,
                  scientific_performance_pass_claimed=False)
    def save(fig, name, meta):
        for extension in ("svg", "png", "pdf"):
            stream = io.BytesIO()
            metadata = ({"Date": None, "Creator": "E1.5 Star v2 presentation adapter"}
                        if extension == "svg" else
                        {"CreationDate": None, "ModDate": None, "Creator": "E1.5 Star v2 presentation adapter"}
                        if extension == "pdf" else {"Software": "E1.5 Star v2 presentation adapter"})
            fig.savefig(stream, format=extension, dpi=300, metadata=metadata)
            write_bytes(output / "figures" / (name + "." + extension), stream.getvalue())
        write_bytes(output / "figures" / (name + ".metadata.json"), json_bytes({**common, **meta}))
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    scopes = ("overall", "center_t=0", "center_t=1")
    for axis, effect in zip(axes, EFFECTS):
        for index, scope in enumerate(scopes):
            row = next(r for r in data["intervals"] if r["effect"] == effect and r["scope"] == scope)
            axis.plot(row["mean_interval_width"], index, "D", color="#2166AC")
        axis.set_title(effect)
        axis.set_xlabel("Mean posterior interval width (outcome)")
        axis.set_yticks(range(3), ["Overall", "Center T=0", "Center T=1"])
        axis.set_ylim(2.5, -.5)
        axis.set_xlim(0, max(r["mean_interval_width"] for r in data["intervals"]
                             if r["effect"] == effect) * 1.12)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle("J: mean width of existing context-level 95% posterior intervals", fontsize=10)
    fig.text(.5, .015, "200 mechanisms; N=200, M=50; widths condition on valid intervals. No width CI.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .07, 1, .91))
    save(fig, "j_interval_width", dict(source_table="tables/j_interval_summary.csv",
         source_row_ids=[r["source_row_id"] for r in data["intervals"]],
         rows=data["intervals"], width_ci_status="NOT_REQUESTED", statistical_unit="mechanism",
         n_requested_overall=1200, n_requested_per_center=600))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), squeeze=False)
    groups = (("center", "Center"), ("leaf", "Leaf mean"), ("difference", "Center minus leaf"))
    for group_index, (group, label) in enumerate(groups):
        for effect_index, effect in enumerate(EFFECTS):
            axis = axes[group_index, effect_index]
            values = []
            for method in NODE_METHODS:
                for center, marker in ((0, "o"), (1, "^")):
                    rows = [r for r in data["coordinates"] if r["method_id"] == method
                            and r["effect"] == effect and r["center_t"] == center]
                    x = [r["true_" + group] for r in rows]
                    y = [r["predicted_" + group] for r in rows]
                    require(len(rows) == 600, "Plot group inventory mismatch")
                    axis.scatter(x, y, s=6, alpha=.22, linewidths=0, marker=marker,
                                 color=colors[method], rasterized=False)
                    values.extend(x)
                    values.extend(y)
            low, high = min(values), max(values)
            padding = max((high - low) * .04, .01)
            axis.plot([low-padding, high+padding], [low-padding, high+padding],
                      color="#333333", linestyle="--", linewidth=.7)
            axis.set_xlim(low-padding, high+padding)
            axis.set_ylim(low-padding, high+padding)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(effect + " / " + label)
            axis.set_xlabel("True effect (outcome)")
            axis.set_ylabel("Predicted effect (outcome)")
    handles = [Line2D([], [], linestyle="", marker="o", color=colors[m], label=names[m])
               for m in NODE_METHODS]
    handles += [Line2D([], [], linestyle="", marker=marker, color="#333333", label="Center T=" + str(c))
                for c, marker in ((0, "o"), (1, "^"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=5, fontsize=8)
    fig.suptitle("Saved true and predicted effects: all contexts; dashed line y=x", fontsize=11, y=.995)
    fig.text(.5, .012, "200 independent mechanisms; 3 nested pairs x 2 arms; N=200, M=50. No fitted line or new CI.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .92))
    save(fig, "effects_true_vs_predicted", dict(source_table="sources/effects_true_vs_predicted.csv",
         source_json="evaluation/center_leaf_metrics.json", original_records=10800,
         displayed_points=32400, panels=9, points_per_panel=3600,
         statistical_unit="mechanism", markers={"0": "circle", "1": "triangle"},
         methods=NODE_METHODS, effects=EFFECTS, confidence_intervals="NOT_REQUESTED",
         selection="All 1200 contexts per model/effect; no sampling, fitting or ordinal ranking"))


def tree_receipt(root):
    """Full content SHA256 protection receipt; no timestamps or statistics changed."""
    root = safe_path(root)
    digest = hashlib.sha256()
    count = total = 0
    for path in sorted(root.rglob("*")):
        safe_path(path)
        if path.is_file():
            name = path.relative_to(root).as_posix()
            size = path.stat().st_size
            digest.update((name + "\t" + str(size) + "\t" + sha(path) + "\n").encode())
            count += 1
            total += size
    return dict(files=count, bytes=total, tree_sha256=digest.hexdigest(), verification="full file-content SHA256")


def verify_output_data(output, data):
    for name, columns, expected in (
        ("tables/main_results.csv", MAIN_COLUMNS, data["main"]),
        ("tables/j_interval_summary.csv", J_COLUMNS, data["intervals"]),
        ("sources/effects_true_vs_predicted.csv", COORD_COLUMNS, data["coordinates"])):
        actual = csv_rows(output / name)
        require(actual == [{k: wire_cell(row[k]) for k in columns} for row in expected],
                "Presentation data changed: " + name)
    require((output / "tables/main_results.tex").read_text(encoding="utf-8").count(r"\begin{table}") == 4,
            "Expected three main effect tables plus interval summary")
    width = read_json(output / "figures/j_interval_width.metadata.json")
    coords = read_json(output / "figures/effects_true_vs_predicted.metadata.json")
    require(width["rows"] == data["intervals"] and width["width_ci_status"] == "NOT_REQUESTED",
            "Width figure provenance mismatch")
    require(coords["original_records"] == 10800 and coords["displayed_points"] == 32400
            and coords["source_run_id"] == RUN and coords["source_snapshot_id"] == SNAPSHOT,
            "Coordinate figure provenance mismatch")
    for name in ("j_interval_width", "effects_true_vs_predicted"):
        for ext, prefix in (("png", b"\x89PNG\r\n\x1a\n"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
            require((output / "figures" / (name + "." + ext)).read_bytes().startswith(prefix),
                    "Invalid figure format")


def run_tests(test_path, output):
    """Run portable synthetic unit tests; output is retained for API compatibility."""
    spec = importlib.util.spec_from_file_location("test_presentation_v2_adapter", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.testsRun > 0, "No portable presentation tests collected")
    return dict(scope="PORTABLE_SYNTHETIC_UNIT_TESTS",
                tests=result.testsRun,
                passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                successful=result.wasSuccessful(), log=log.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "acceptance", "runtime", "source-delivery", "run-directory",
                 "original-analysis", "output", "test-file"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    output = safe_path(args.output, True)
    require(not os.path.lexists(output), "Presentation destination exists; never overwrite")
    test_path = safe_path(args.test_file)
    roots = dict(R=args.runtime, S=args.source_delivery, T=args.run_directory,
                 A=args.original_analysis, V=args.analysis, code=Path(__file__).resolve().parent)
    for root in roots.values():
        root = safe_path(root)
        require(not output.resolve().is_relative_to(root.resolve())
                and not root.resolve().is_relative_to(output.resolve()), "Protected/output overlap")
    started = time.monotonic()
    print("PRECHECK: verifying full accepted V and formal anchors", flush=True)
    data = load_inputs(args.analysis, args.acceptance, args.runtime)
    print("DATA=PASS: main 21, interval 9, original records 10800; no metric recomputation", flush=True)
    before = {}
    for name, root in roots.items():
        print("PROTECTION_BASELINE=" + name, flush=True)
        before[name] = tree_receipt(root)
    output.mkdir(exist_ok=False)
    write_csv(output / "tables/main_results.csv", MAIN_COLUMNS, data["main"])
    write_csv(output / "tables/j_interval_summary.csv", J_COLUMNS, data["intervals"])
    write_csv(output / "sources/effects_true_vs_predicted.csv", COORD_COLUMNS, data["coordinates"])
    old_mpl = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix=".render-cache-", dir=output) as cache:
            os.environ["MPLCONFIGDIR"] = cache
            import matplotlib as mpl
            with mpl.rc_context():
                table_template, figure_template = load_templates(data["runtime"],
                                                               expected_identity=data["template_identity"])
                write_bytes(output / "tables/main_results.tex",
                            render_main_tex(data["main"], data["intervals"], table_template).encode("utf-8"))
                render_figures(output, data, table_template)
    finally:
        if old_mpl is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl
    readme = (
        "# E1.5 Star v2 presentation-only supplement\n\n"
        f"Run ID: {RUN}\nSnapshot ID: {SNAPSHOT}\n\n"
        "Inputs are the Gate 1.3 independently accepted verification copy. "
        "No formal metric, bootstrap distribution, model prediction or DGP was recomputed.\n\n"
        "main_results.csv has 21 method-effect rows carrying 126 original metric values. "
        "main_results.tex contains three effect panels and a 9-row J interval summary. "
        "Absolute Bias differs from ACB; variance has squared outcome units. "
        "No ordinal ranking or rank shading is added.\n\n"
        "J width is conditional on valid context posterior intervals. Width CIs are NOT_REQUESTED. "
        "Coverage CIs are the existing pointwise 95% mechanism-block Bootstrap CIs. "
        "The 90% target applies only to overall coverage point estimates, not center strata or CI bounds.\n\n"
        "effects_true_vs_predicted.csv copies all 10800 saved method-context-effect records. "
        "Each supplies center, leaf mean and center-minus-leaf coordinates (32400 displayed points). "
        "The 200 mechanisms are independent; nested contexts are not independent mechanisms. "
        "No sampling, fitted regression or new confidence interval is used. Oracle and Hajek are not ranked.\n\n"
        "Original make_figures.py styling and make_table.py names/latex_escape are reused unchanged. "
        "Their main/read_results/ate_table/build_figure/render_table entrypoints are not called: "
        "they require a different schema or would recompute statistics/add rankings. "
        "These are adapted views, not direct outputs of the legacy template runners.\n"
    )
    write_bytes(output / "README.md", readme.encode("utf-8"))
    print("RENDER=COMPLETE; running only new presentation tests", flush=True)
    tests = run_tests(test_path, output)
    write_bytes(output / "tests/test_results.json", json_bytes(tests))
    require(tests["successful"] and tests["failures"] == tests["errors"] == tests["skipped"] == 0,
            "Presentation tests failed; preserve outputs and stop")
    verify_output_data(output, data)
    after = {}
    for name, root in roots.items():
        print("PROTECTION_RECHECK=" + name, flush=True)
        after[name] = tree_receipt(root)
    require(before == after, "Original file protection failed; preserve outputs and stop")
    require(not any(n == "star_v2" or n.startswith("star_v2.") or n == "predict"
                    or n == "benchmark" or n.startswith("pfn_social") for n in sys.modules),
            "Forbidden scientific module imported")
    generated = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            generated.append(dict(path=path.relative_to(output).as_posix(),
                                  bytes=path.stat().st_size, sha256=sha(path)))
    require(len(generated) == 14, "Unexpected presentation output inventory")
    provenance = dict(kind="PRESENTATION_ONLY", source_run_id=RUN, source_snapshot_id=SNAPSHOT,
        analysis_directory=str(args.analysis), analysis_manifest_sha256=MANIFEST_SHA,
        analysis_report_sha256=REPORT_SHA, independent_acceptance=dict(
            path=str(args.acceptance), sha256=ACCEPTANCE_SHA),
        sources=data["sources"], templates=TEMPLATES,
        template_identity=data["template_identity"],
        template_reuse="Unchanged style/constants/latex_escape; no legacy statistic or ranking functions",
        implementation=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        test_implementation=dict(path=str(test_path), sha256=sha(test_path)),
        source_protection=dict(status="PASS", before=before, after=after),
        statistical_recomputation=False, bootstrap_executed=False, models_executed=False,
        counts=dict(main_rows=21, original_metric_values=126, j_interval_rows=9,
                    original_coordinate_records=10800, plotted_coordinate_points=32400),
        files=generated)
    write_bytes(output / "presentation_manifest.json", json_bytes(provenance))
    acceptance = dict(status="PASS", scope="PRESENTATION_ONLY", source_run_id=RUN,
        source_snapshot_id=SNAPSHOT, counts=provenance["counts"], tests=tests,
        original_file_protection="PASS", original_file_protection_mode="full content SHA256",
        presentation_manifest_sha256=sha(output / "presentation_manifest.json"),
        artifacts=16, figure_groups=2, figure_files=6, metadata_files=2,
        scientific_performance_pass_claimed=False,
        elapsed_seconds=time.monotonic()-started)
    write_bytes(output / "presentation_acceptance.json", json_bytes(acceptance))
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    require(actual == {r["path"] for r in generated} | {"presentation_manifest.json", "presentation_acceptance.json"},
            "Final presentation file universe mismatch")
    for record in generated:
        path = output / record["path"]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                "Final presentation hash mismatch")
    print(json.dumps(acceptance, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

