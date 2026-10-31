"""Reproduce E1.5 paired star-graph experiments without executing the notebook."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

# Support both `python -m scripts.e1_5_star` and an absolute script path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMPORT_STARTED = time.perf_counter()
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from pfn_social.contracts import arm_mean_exposure
from pfn_social.phase3_hierarchical import _components, structural_outcome
from pfn_social.phase6_limited_outcomes import generate_mechanism
from predict import CHECKPOINT_SHA256, load_selected_model
from scripts.e1_5_star_analysis import support_reason, summarize, validate_run
from scripts.e1_5_star_resources import memory_usage

N, M = 200, 50
DEFAULT_SEED = 20261024
REPEAT_NAMESPACE = 2026102501
ARMS = ("00", "01", "10", "11")
EFFECTS = ("direct", "spillover", "total")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets():
    """Keep the existing raw-byte integrity contract, including text configs."""
    config = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
    expected = dict(config["frozen_assets_sha256"])
    regional = json.loads((ROOT / "regional_selected.json").read_text(encoding="utf-8"))
    expected["regional_inference.py"] = regional["inference_sha256"]
    actual = {name: sha256(ROOT / name) for name in expected}
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError(f"Integrity mismatch: {name}; check LF checkout before running")
    if actual["models/iPFN_J.pt"] != CHECKPOINT_SHA256:
        raise ValueError("Experiment config and frozen-J loader hashes disagree")
    return actual


def effects(mu):
    return np.stack((mu[:, 2] - mu[:, 0], mu[:, 3] - mu[:, 2],
                     mu[:, 3] - mu[:, 0]), axis=-1)


def build_world(mechanism_id):
    base = generate_mechanism(0, mechanism_id).parent
    a = np.zeros((N, N), dtype=np.float32)
    a[0, 1:] = a[1:, 0] = 1
    world = SimpleNamespace(adjacency=a, h=base.h, coefficients=base.coefficients,
                            components=_components(a, base.h, base.coefficients))
    degree = a.sum(1)
    em = np.array([[arm_mean_exposure(int(d), .5, s) for s in (0, 1)] for d in degree])
    hbar = a @ base.h / degree[:, None]
    b, tau, gamma, eta = world.components.T
    beta = base.coefficients
    truth = np.column_stack([
        b + t*tau + em[:, s] * (gamma + t*eta + hbar @ (beta[19:22] + t*beta[22:25]))
        for t, s in ((0, 0), (0, 1), (1, 0), (1, 1))])
    # Independent structural checks: leaves have only the hub as neighbor;
    # hub expectation is a linear interpolation of all-off/all-on neighbors.
    for t in (0, 1):
        off = structural_outcome(world, np.zeros(N), own_override=t)[0]
        on_t = np.ones(N)
        on_t[0] = 0
        on = structural_outcome(world, on_t, own_override=t)[0]
        for s in (0, 1):
            assigned = np.zeros(N)
            assigned[0] = s
            factual = structural_outcome(world, assigned, own_override=t)
            np.testing.assert_allclose(truth[1:, 2*t+s], factual[1:], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(truth[0, 2*t+s], off + em[0, s]*(on-off), atol=1e-10)
    return base, world, truth


def pair_rng(seed, mechanism_id, repeat_id):
    # Preserve the notebook exactly for repeat 0; separate namespace thereafter.
    if repeat_id == 0:
        entropy = seed + mechanism_id
    else:
        entropy = [seed, REPEAT_NAMESPACE, mechanism_id, repeat_id]
    return np.random.default_rng(entropy), entropy


def make_pair(base, world, mechanism_id, repeat_id, seed):
    rng, entropy = pair_rng(seed, mechanism_id, repeat_id)
    leaf_t = rng.binomial(1, .5, N - 1).astype(np.float32)
    noise = rng.normal(0, base.setting.noise_sd, N)
    indices = rng.permutation(N)[:M].astype(np.int64)
    pair = []
    for center_t in (0, 1):
        treatment = np.concatenate(([center_t], leaf_t)).astype(np.float32)
        y = (structural_outcome(world, treatment) + noise).astype(np.float32)
        pair.append(dict(x=base.x.copy(), adjacency=world.adjacency,
                         treatment=treatment, observed_indices=indices.copy(),
                         outcomes_observed=y[indices].copy()))
    return pair, entropy


def score(inputs, truth, pred, mechanism_id, repeat_id, center_t):
    true_effects, pred_effects = effects(truth), effects(pred)
    error = pred_effects - true_effects
    rmse = lambda value: float(np.sqrt(np.mean(np.square(value))))
    common = dict(mechanism_id=mechanism_id, pair_repeat_id=repeat_id, center_t=center_t,
                  center_observed=bool(0 in inputs["observed_indices"]))
    metrics, support = [], []
    t = inputs["treatment"]
    s = (inputs["adjacency"] @ t / inputs["adjacency"].sum(1) > .5).astype(int)
    observed = inputs["observed_indices"]
    for category, nodes in (("center", np.arange(1)), ("leaf", np.arange(1, N))):
        row = dict(common, category=category, arm_rmse=rmse(pred[nodes] - truth[nodes]))
        for j, name in enumerate(EFFECTS):
            row[name + "_rmse"] = rmse(error[nodes, j])
            row[name + "_bias"] = float(error[nodes, j].mean())
            row[name + "_ate_true"] = float(true_effects[nodes, j].mean())
            row[name + "_ate_pred"] = float(pred_effects[nodes, j].mean())
        metrics.append(row)
        obs = observed[observed == 0] if category == "center" else observed[observed != 0]
        counts = np.bincount(2*t[obs].astype(int) + s[obs], minlength=4)
        full_counts = np.bincount(2*t[nodes].astype(int) + s[nodes], minlength=4)
        for arm, label in enumerate(ARMS):
            support.append(dict(common, category=category, arm=label,
                                full_graph_count=int(full_counts[arm]),
                                observed_count=int(counts[arm]),
                                empirical_support=bool(counts[arm] > 0),
                                support_reason=support_reason(category, int(label[0]), int(label[1]),
                                                              center_t, int(full_counts[arm]), int(counts[arm])),
                                arm_rmse=rmse(pred[nodes, arm] - truth[nodes, arm])))
    return metrics, support


def compare_notebook(path, metrics, support):
    """Compare to rounded saved outputs; never run or rewrite notebook cells."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    comparisons = []
    for marker, rows, pattern, keys in (
        ("display(pair_rmse_df.head(6)", metrics,
         r"^\s*\d+\s+0\s+([01])\s+(中心|叶子)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
         ("arm_rmse", "direct_rmse", "spillover_rmse", "total_rmse")),
        ("display(support_df.head(8)", support,
         r"^\s*\d+\s+0\s+([01])\s+(00|01|10|11)\s+(\d+)\s+(True|False)\s+([\d.]+)\s*$",
         ("observed_count", "empirical_support", "arm_rmse")),
    ):
        cells = [c for c in nb["cells"] if marker in "".join(c.get("source", []))]
        if len(cells) != 1:
            raise ValueError(f"Missing or ambiguous notebook reference: {marker}")
        text = "\n".join("".join(o.get("data", {}).get("text/plain", []))
                         for o in cells[0].get("outputs", []))
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        expected_count = 4 if rows is metrics else 8
        matches = [m for m in matches if rows is support or m[2] in ("中心", "叶子")]
        if len(matches) != expected_count:
            raise ValueError(f"Expected {expected_count} reference rows, found {len(matches)}")
        for match in matches:
            center_t = int(match[1])
            category = {"中心": "center", "叶子": "leaf"}.get(match[2], "leaf")
            candidates = [r for r in rows if r["mechanism_id"] == 0 and r["pair_repeat_id"] == 0
                          and r["center_t"] == center_t and r["category"] == category
                          and (rows is metrics or r["arm"] == match[2])]
            row, = candidates
            for key, saved in zip(keys, match.groups()[2:]):
                expected = float(saved) if saved not in ("True", "False") else float(saved == "True")
                difference = abs(float(row[key]) - expected)
                tolerance = 5.1e-5 if "rmse" in key else 0.
                comparisons.append(dict(center_t=center_t, category=category,
                                        arm=row.get("arm"), metric=key, notebook_rounded=expected,
                                        reproduced=float(row[key]), absolute_difference=difference,
                                        tolerance=tolerance, passed=difference <= tolerance))
    report = dict(notebook_sha256=sha256(path), comparisons=comparisons,
                  explanation="Notebook prints 4 decimal places; RMSE tolerance is 0.000051.")
    report["passed"] = all(c["passed"] for c in comparisons)
    return report


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_memory_usage():
    """Record unavailable Windows memory counters explicitly on macOS."""
    if sys.platform == "darwin":
        return dict(status="unavailable",
                    reason="Native Windows process memory APIs are unavailable on macOS",
                    peak_working_set_bytes=None, working_set_bytes=None,
                    private_bytes=None, process_pid=None)
    return memory_usage()


def source_snapshot(directory):
    """Archive complete package source/config bytes, tests and experiment notes."""
    repo = ROOT.parents[2]
    files = set(ROOT.rglob("*.py")) | set(ROOT.rglob("*.json")) | set(ROOT.rglob("requirements*.txt"))
    files.update((repo / ".gitattributes", repo / "iPFN_demo/E1_5_STAR.md",
                  repo / "iPFN_demo/requirements-notebook.txt"))
    digests = {}
    with zipfile.ZipFile(directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = path.relative_to(repo).as_posix()
            data = path.read_bytes()
            archive.writestr(name, data)
            digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def git_record():
    def run(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout
    return dict(commit=run("rev-parse", "HEAD").strip(), branch=run("branch", "--show-current").strip(),
                status=run("status", "--porcelain=v1", "--untracked-files=all"),
                diff=run("diff", "--no-ext-diff"), staged_diff=run("diff", "--cached", "--no-ext-diff"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite an existing run")
    parser.add_argument("--mechanism-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--repeat-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compare-notebook", type=Path)
    parser.add_argument("--purpose", choices=("smoke", "timing_pilot", "expanded_dev"), default="smoke")
    args = parser.parse_args()
    for name, values in (("mechanism", args.mechanism_ids), ("repeat", args.repeat_ids)):
        if min(values) < 0 or len(values) != len(set(values)):
            parser.error(f"{name} IDs must be unique nonnegative integers")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.compare_notebook and (args.mechanism_ids != [0] or args.repeat_ids != [0] or args.seed != DEFAULT_SEED):
        parser.error("Notebook comparison requires mechanism 0, repeat 0 and seed 20261024")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new run name")
    if args.purpose == "timing_pilot" and (args.mechanism_ids != list(range(900, 904)) or
                                           args.repeat_ids != list(range(5)) or args.seed != DEFAULT_SEED):
        parser.error("Timing pilot requires mechanisms 900-903, repeats 0-4, seed 20261024")
    if args.purpose == "expanded_dev" and (args.mechanism_ids != list(range(1000, 1020)) or
                                           args.repeat_ids != list(range(10)) or args.seed != DEFAULT_SEED):
        parser.error("Expanded dev v1 requires mechanisms 1000-1019, repeats 0-9, seed 20261024")

    started = time.perf_counter()
    assets = verify_assets()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        load_started = time.perf_counter()
        model = load_selected_model()
        load_seconds = time.perf_counter() - load_started
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema_version=2, protocol="expanded_dev_star_v1", purpose=args.purpose,
                        status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                        n=N, m=M, family_id=0, seed=args.seed, mechanism_ids=args.mechanism_ids,
                        pair_repeat_ids=args.repeat_ids, arm_order=ARMS, effect_order=EFFECTS,
                        model_config=asdict(model.model.config), model_identity=model.identity,
                        assets_sha256=assets, python=platform.python_version(),
                        versions={k: metadata.version(k) for k in ("numpy", "torch", "threadpoolctl", "scipy", "networkx")},
                        installed_packages=sorted({(d.metadata["Name"], d.version) for d in metadata.distributions()}),
                        threadpools=threadpool_info(), torch_build=torch.__config__.show(),
                        command_line=sys.argv, platform=platform.platform(), machine=platform.machine(),
                        executable=sys.executable, logical_cpus=os.cpu_count(),
                        source_sha256=source_snapshot(args.output),
                        cpu_threads=1, model_load_seconds=load_seconds,
                        import_seconds=started - IMPORT_STARTED, mechanisms=[], pairs=[], tasks=[])
        manifest["git"] = git_record()
        manifest["git_commit"] = manifest["git"]["commit"]
        write_json(args.output / "manifest.json", manifest)
        all_metrics, all_support, generation_times, pair_times = [], [], [], []
        for mid in args.mechanism_ids:
            generation_started = time.perf_counter()
            base, world, truth = build_world(mid)
            generation_times.append(time.perf_counter() - generation_started)
            manifest["mechanisms"].append(dict(mechanism_id=mid, config=asdict(base.config),
                                                parent_mechanism_sha256=base.mechanism_sha256,
                                                noise_sd=base.setting.noise_sd))
            for rid in args.repeat_ids:
                pair_started = time.perf_counter()
                pair, entropy = make_pair(base, world, mid, rid, args.seed)
                manifest["pairs"].append(dict(mechanism_id=mid, pair_repeat_id=rid, rng_entropy=entropy,
                                              bit_generator="PCG64", shared="leaf treatment, noise, observed indices"))
                for center_t, inputs in enumerate(pair):
                    name = f"m{mid}_r{rid}_t{center_t}"
                    inference_started = time.perf_counter()
                    pred = model.predict_arm_means(**inputs)
                    inference_seconds = time.perf_counter() - inference_started
                    metrics, support = score(inputs, truth, pred, mid, rid, center_t)
                    all_metrics.extend(metrics)
                    all_support.extend(support)
                    # Factual file has exactly the deployment API's five keys.
                    np.savez_compressed(args.output / (name + "_input.npz"), **inputs)
                    np.savez_compressed(args.output / (name + "_evaluation.npz"),
                                        truth_mu=truth, pred_mu=pred, true_effects=effects(truth),
                                        pred_effects=effects(pred), arm_errors=pred-truth,
                                        effect_errors=effects(pred)-effects(truth))
                    manifest["tasks"].append(dict(mechanism_id=mid, pair_repeat_id=rid,
                                                  center_t=center_t, task_id=name,
                                                  observed_indices=inputs["observed_indices"].tolist(),
                                                  center_observed=bool(0 in inputs["observed_indices"]),
                                                  inference_seconds=inference_seconds,
                                                  process_cpu_seconds=time.process_time(),
                                                  resources=task_memory_usage()))
                pair_times.append(time.perf_counter() - pair_started)
        write_csv(args.output / "metrics.csv", all_metrics)
        write_csv(args.output / "support.csv", all_support)
        contrasts, primary, secondary, summary = summarize(all_support, all_metrics, args.mechanism_ids, args.repeat_ids)
        write_csv(args.output / "paired_contrasts.csv", contrasts)
        write_csv(args.output / "mechanism_primary.csv", primary)
        write_csv(args.output / "mechanism_metrics.csv", secondary)
        write_json(args.output / "summary.json", summary)
        comparison = compare_notebook(args.compare_notebook, all_metrics, all_support) if args.compare_notebook else None
        if comparison is not None:
            write_json(args.output / "notebook_comparison.json", comparison)
        manifest["timing"] = dict(mechanism_generation_seconds=generation_times, pair_seconds=pair_times,
                                  estimated_20_mechanisms_10_pairs_seconds=load_seconds + 20*float(np.mean(generation_times)) + 200*float(np.mean(pair_times)),
                                  estimate_note="One model load, 20 mechanism generations, 200 pairs/400 predictions. Import/process startup excluded; smoke estimate, not a benchmark.")
        manifest["wall_seconds_after_import"] = time.perf_counter() - started
        manifest["output_sha256"] = {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.name != "manifest.json"}
        manifest["status"] = "validating" if comparison is None or comparison["passed"] else "comparison_failed"
        write_json(args.output / "manifest.json", manifest)
        if manifest["status"] == "comparison_failed":
            raise SystemExit("Notebook comparison failed; inspect notebook_comparison.json")
        # Only a fully accepted run can become complete. Failures retain evidence.
        try:
            manifest["acceptance"] = validate_run(args.output, candidate=True)
        except Exception as exc:
            manifest["status"] = "validation_failed"
            manifest["validation_error"] = str(exc)
            write_json(args.output / "manifest.json", manifest)
            raise
        manifest["status"] = "complete"
        manifest["wall_seconds_after_import"] = time.perf_counter() - started
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(dict(status=manifest["status"], tasks=len(manifest["tasks"]),
                              wall_seconds_after_import=manifest["wall_seconds_after_import"],
                              timing=manifest["timing"]), indent=2))
        if manifest["status"] != "complete":
            raise SystemExit("Notebook comparison failed; inspect notebook_comparison.json")


if __name__ == "__main__":
    main()
