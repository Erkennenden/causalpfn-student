"""Mechanism-level summaries and strict artifact acceptance for E1.5.

The endpoint is a conditional association, not a causal effect of Support.
No inference or model fitting is performed here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BOOTSTRAP_SEED = 2026102502
BOOTSTRAP_DRAWS = 10000
STRUCTURAL = "conditional_structural_zero"
ASSIGNMENT = "assignment_zero"
SAMPLING = "observation_sampling_zero"
COVERED = "covered"


def support_reason(category, own, exposure, center_t, full_count, observed_count):
    # Under fixed hub T=c: leaves must have S=c, and the hub must have T=c.
    impossible = exposure != center_t if category == "leaf" else own != center_t
    if impossible:
        if full_count or observed_count:
            raise ValueError("A conditionally impossible state has factual observations")
        return STRUCTURAL
    if full_count == 0:
        if observed_count:
            raise ValueError("Observed count exceeds full-graph count")
        return ASSIGNMENT
    if observed_count == 0:
        return SAMPLING
    if not 0 < observed_count <= full_count:
        raise ValueError("Invalid Support counts")
    return COVERED


def summarize(support, metrics, mechanism_ids, repeat_ids):
    """Equal weights: 4 arms -> repeats within mechanism -> mechanisms."""
    indexed = {}
    for row in support:
        if row["category"] != "leaf":
            continue
        key = (row["mechanism_id"], row["pair_repeat_id"], row["center_t"], row["arm"])
        if key in indexed:
            raise ValueError(f"Duplicate Support key: {key}")
        indexed[key] = row
    expected = {(m, r, c, a) for m in mechanism_ids for r in repeat_ids
                for c in (0, 1) for a in ("00", "01", "10", "11")}
    if set(indexed) != expected:
        raise ValueError("Missing or unexpected paired leaf states")
    contrasts, mechanisms = [], []
    for mid in mechanism_ids:
        repeat_means = []
        for rid in repeat_ids:
            differences = []
            for arm in ("00", "01", "10", "11"):
                allowed = indexed[mid, rid, int(arm[1]), arm]
                blocked = indexed[mid, rid, 1-int(arm[1]), arm]
                if blocked["support_reason"] != STRUCTURAL or allowed["support_reason"] == STRUCTURAL:
                    raise ValueError("Conditional coverage labels disagree with paired design")
                delta = float(blocked["arm_rmse"] - allowed["arm_rmse"])
                contrasts.append(dict(mechanism_id=mid, pair_repeat_id=rid, arm=arm,
                                      structural_rmse=blocked["arm_rmse"], allowed_rmse=allowed["arm_rmse"],
                                      allowed_empirical_support=allowed["empirical_support"], delta=delta))
                differences.append(delta)
            repeat_means.append(float(np.mean(differences)))
        mechanisms.append(dict(mechanism_id=mid, repeats=len(repeat_ids),
                               primary_delta=float(np.mean(repeat_means))))
    values = np.array([r["primary_delta"] for r in mechanisms])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    summary = dict(protocol="expanded_dev_star_v1", interpretation="Conditional association; not a causal effect of Support",
                   aggregation="mean four arm differences; mean repeats within mechanism; mean mechanisms",
                   independent_units=len(values), bootstrap_seed=BOOTSTRAP_SEED,
                   bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_unit="whole mechanism",
                   mean_delta=float(values.mean()),
                   ci95=np.quantile(draws, [.025, .975]).tolist() if len(values) > 1 else None,
                   ci_note="Pilot intervals are exploratory; one mechanism cannot estimate between-mechanism uncertainty.")
    secondary = []
    for mid in mechanism_ids:
        for center_t in (0, 1):
            for category in ("center", "leaf"):
                group = [r for r in metrics if r["mechanism_id"] == mid and
                         r["center_t"] == center_t and r["category"] == category]
                if len(group) != len(repeat_ids) or {r["pair_repeat_id"] for r in group} != set(repeat_ids):
                    raise ValueError("Incomplete mechanism metric group")
                row = dict(mechanism_id=mid, center_t=center_t, category=category)
                for key in group[0]:
                    if key not in {"mechanism_id", "pair_repeat_id", "center_t", "category", "center_observed"}:
                        row[key] = float(np.mean([r[key] for r in group]))
                secondary.append(row)
    return contrasts, mechanisms, secondary, summary


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, label):
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    except AssertionError as exc:
        raise ValueError(f"Mismatch: {label}: {exc}") from exc


def _check_rows(path, expected, keys):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        _require(set(reader.fieldnames or []) == set(expected[0]), f"Wrong CSV columns: {path.name}")
    indexed = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        _require(key not in indexed, f"Duplicate CSV row: {path.name} {key}")
        indexed[key] = row
    _require(len(rows) == len(expected), f"Wrong row count: {path.name}")
    for wanted in expected:
        key = tuple(str(wanted[k]) for k in keys)
        _require(key in indexed, f"Missing CSV row: {path.name} {key}")
        for field, value in wanted.items():
            stored = indexed[key][field]
            if isinstance(value, bool):
                _require(stored == str(value), f"Wrong boolean: {path.name} {key} {field}")
            elif isinstance(value, int):
                _require(stored == str(value), f"Wrong integer: {path.name} {key} {field}")
            elif isinstance(value, float):
                number = float(stored)
                _require(np.isfinite(number), f"Nonfinite CSV value: {path.name} {field}")
                _same(number, value, f"{path.name} {key} {field}")
            else:
                _require(stored == str(value), f"Wrong value: {path.name} {key} {field}")


def validate_run(directory, *, candidate=False):
    """Raise on ANY mismatch. Regenerate factual data, never re-run PFN.

    candidate=True is internal: only accepts 'validating'; the public command
    requires 'complete'. Source snapshots and all saved arrays are checked.
    """
    from scripts.e1_5_star import (ROOT, N, M, ARMS, EFFECTS, build_world, make_pair,
                                   score, effects, sha256, verify_assets)
    from pfn_social.sparse_frozen_gaussian import validate_factual_inputs

    began = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 2, "Only schema 2 runs support strict acceptance")
    _require(manifest["status"] == ("validating" if candidate else "complete"), "Run is not in required completion state")
    _require((manifest["n"], manifest["m"], manifest["family_id"]) == (N, M, 0), "Wrong N/M/family")
    _require(manifest["arm_order"] == list(ARMS) and manifest["effect_order"] == list(EFFECTS), "Wrong state/effect order")
    mids, rids = manifest["mechanism_ids"], manifest["pair_repeat_ids"]
    _require(bool(mids) and bool(rids) and len(set(mids)) == len(mids) and len(set(rids)) == len(rids), "Invalid mechanism/repeat IDs")
    _require(all(type(i) is int and i >= 0 for i in mids+rids), "Invalid ID type or range")
    _require(type(manifest["seed"]) is int and manifest["seed"] >= 0, "Invalid seed")
    purpose = manifest.get("purpose")
    _require(purpose in {"smoke", "timing_pilot", "expanded_dev"}, "Missing/invalid run purpose")
    if purpose in {"timing_pilot", "expanded_dev"}:
        wanted_mids = list(range(900, 904)) if purpose == "timing_pilot" else list(range(1000, 1020))
        wanted_rids = list(range(5)) if purpose == "timing_pilot" else list(range(10))
        _require(mids == wanted_mids and rids == wanted_rids and manifest["seed"] == 20261024,
                 "Frozen protocol IDs/repeats/seed mismatch")
    _require(manifest["assets_sha256"] == verify_assets(), "Asset hashes differ from run")
    _require(manifest["model_identity"]["sha256"] == manifest["assets_sha256"]["models/iPFN_J.pt"], "Wrong model identity")
    _require(manifest["model_config"] == dict(architecture="J", raw_p=20, hidden=64), "Wrong model config")
    _require(manifest["cpu_threads"] == 1, "Wrong CPU thread setting")
    for relative, digest in manifest["source_sha256"].items():
        _require(sha256(ROOT.parents[2] / relative) == digest, f"Runtime source changed: {relative}")
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        _require(set(archive.namelist()) == set(manifest["source_sha256"]), "Source snapshot membership mismatch")
        for relative, digest in manifest["source_sha256"].items():
            _require(hashlib.sha256(archive.read(relative)).hexdigest() == digest, f"Source snapshot mismatch: {relative}")
    receipts = {"runtime_measurement.json", "acceptance_report.json", "resource_samples.csv", "stdout.txt", "stderr.txt"}
    files = {p.name for p in directory.iterdir() if p.is_file()} - receipts - {"manifest.json"}
    required_files = {"source_snapshot.zip", "metrics.csv", "support.csv", "paired_contrasts.csv",
                      "mechanism_primary.csv", "mechanism_metrics.csv", "summary.json"}
    expected_tasks = {(m, r, c) for m in mids for r in rids for c in (0, 1)}
    for m, r, c in expected_tasks:
        required_files.update({f"m{m}_r{r}_t{c}_input.npz", f"m{m}_r{r}_t{c}_evaluation.npz"})
    if "notebook_comparison.json" in manifest["output_sha256"]:
        required_files.add("notebook_comparison.json")
        comparison = json.loads((directory / "notebook_comparison.json").read_text(encoding="utf-8"))
        _require(comparison.get("passed") is True, "Notebook comparison failed")
    _require(files == required_files == set(manifest["output_sha256"]), "Missing or unexpected output files")
    for name, digest in manifest["output_sha256"].items():
        _require(sha256(directory / name) == digest, f"Output hash mismatch: {name}")
    tasks, pairs, mechanisms = {}, {}, {}
    for row in manifest["tasks"]:
        key = (row["mechanism_id"], row["pair_repeat_id"], row["center_t"])
        _require(key not in tasks, f"Duplicate task: {key}")
        tasks[key] = row
    _require(set(tasks) == expected_tasks, "Missing/unexpected tasks")
    for row in manifest["pairs"]:
        key = (row["mechanism_id"], row["pair_repeat_id"])
        _require(key not in pairs, f"Duplicate pair: {key}")
        pairs[key] = row
    _require(set(pairs) == {(m, r) for m in mids for r in rids}, "Missing/unexpected pairs")
    for row in manifest["mechanisms"]:
        _require(row["mechanism_id"] not in mechanisms, "Duplicate mechanism receipt")
        mechanisms[row["mechanism_id"]] = row
    _require(set(mechanisms) == set(mids), "Missing/unexpected mechanism receipts")
    metrics, support = [], []
    from dataclasses import asdict
    for mid in mids:
        base, world, truth = build_world(mid)
        _require(mechanisms[mid]["config"] == asdict(base.config) and
                 mechanisms[mid]["parent_mechanism_sha256"] == base.mechanism_sha256 and
                 mechanisms[mid]["noise_sd"] == base.setting.noise_sd, "Mechanism provenance mismatch")
        for rid in rids:
            regenerated, entropy = make_pair(base, world, mid, rid, manifest["seed"])
            _require(pairs[mid, rid]["rng_entropy"] == entropy and pairs[mid, rid]["bit_generator"] == "PCG64", "Pair random stream mismatch")
            for center_t, expected_input in enumerate(regenerated):
                task = tasks[mid, rid, center_t]
                name = f"m{mid}_r{rid}_t{center_t}"
                _require(task["task_id"] == name, "Wrong task filename mapping")
                with np.load(directory / (name + "_input.npz"), allow_pickle=False) as saved:
                    _require(set(saved.files) == set(expected_input), f"Wrong factual input keys: {name}")
                    inputs = {k: saved[k] for k in saved.files}
                validate_factual_inputs(**inputs)
                _require(inputs["outcomes_observed"].shape == (M,), f"Wrong observation count: {name}")
                for key, value in inputs.items():
                    _require(value.dtype == expected_input[key].dtype and np.array_equal(value, expected_input[key]),
                             f"Factual regeneration/pair shared random quantities mismatch: {name} {key}")
                _require(task["observed_indices"] == inputs["observed_indices"].tolist() and
                         task["center_observed"] == bool(0 in inputs["observed_indices"]), "Observation receipt mismatch")
                with np.load(directory / (name + "_evaluation.npz"), allow_pickle=False) as saved:
                    evaluation = {k: saved[k] for k in saved.files}
                _require(set(evaluation) == {"truth_mu", "pred_mu", "true_effects", "pred_effects", "arm_errors", "effect_errors"}, "Wrong evaluation keys")
                for key, value in evaluation.items():
                    shape = (N, 4) if key in {"truth_mu", "pred_mu", "arm_errors"} else (N, 3)
                    _require(value.shape == shape and np.isfinite(value).all(), f"Invalid prediction/evaluation array: {name} {key}")
                _same(evaluation["truth_mu"], truth, f"{name} mechanism truth across repeats")
                pred = evaluation["pred_mu"]
                for key, value in dict(true_effects=effects(truth), pred_effects=effects(pred),
                                       arm_errors=pred-truth, effect_errors=effects(pred)-effects(truth)).items():
                    _same(evaluation[key], value, f"{name} {key}")
                met, sup = score(inputs, truth, pred, mid, rid, center_t)
                metrics.extend(met)
                support.extend(sup)
    task_keys = ("mechanism_id", "pair_repeat_id", "center_t", "category")
    _check_rows(directory / "metrics.csv", metrics, task_keys)
    _check_rows(directory / "support.csv", support, task_keys + ("arm",))
    contrasts, primary, secondary, summary = summarize(support, metrics, mids, rids)
    _check_rows(directory / "paired_contrasts.csv", contrasts, ("mechanism_id", "pair_repeat_id", "arm"))
    _check_rows(directory / "mechanism_primary.csv", primary, ("mechanism_id",))
    _check_rows(directory / "mechanism_metrics.csv", secondary, ("mechanism_id", "center_t", "category"))
    _require(json.loads((directory / "summary.json").read_text(encoding="utf-8")) == summary, "Mechanism bootstrap summary mismatch")
    return dict(passed=True, tasks=len(tasks), pairs=len(pairs), mechanisms=len(mids),
                metric_rows=len(metrics), support_rows=len(support),
                checks="IDs, indices, shapes, finite values, regenerated paired random quantities, truth, CSV/NPZ, bootstrap, sources, assets, output hashes, status",
                seconds=time.perf_counter()-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.directory), indent=2))


if __name__ == "__main__":
    main()
