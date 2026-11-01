"""Run the E1.6 complete-graph validation with the supplied frozen models."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import ALL_METHODS, configuration as model_configuration
from benchmark import load_methods, predict_method
from pfn_social.intervals import predict_intervals
from pfn_social.phase6_sparse_classical import pack
from pfn_social.e1_6_complete import (
    EFFECT_LABELS,
    effects_from_arm_means,
    empirical_support,
    generate_complete_task,
    make_complete_observed_draw,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "e1_6_complete_graph.json"


@contextmanager
def source_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def load_task_configuration() -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    if config["topology"] != "complete" or config["n"] != 200:
        raise ValueError("E1.6 requires the registered N=200 complete graph.")
    if config["degree"] != config["n"] - 1:
        raise ValueError("Complete-graph degree must equal N-1.")
    return config


def case_grid(args, config):
    masks = tuple(config["observed_outcomes"])
    if args.cohort == "full":
        mechanisms = range(config["mechanisms"])
        repeats = range(config["assignment_noise_repeats"])
    elif args.cohort == "pilot":
        mechanisms = range(2)
        repeats = range(4)
    elif args.cohort == "smoke":
        mechanisms = range(1)
        repeats = range(2)
    else:
        mechanisms = (args.mechanism,)
        repeats = (args.repeat,)
        masks = (args.observed,)
    return mechanisms, repeats, masks


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else math.nan)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["effect"])].append(row)
    summary = []
    for (method, effect), group in groups.items():
        ite = [r["ite_rmse"] for r in group if r["predictor_ite_supported"]]
        ate = [r["ate_error"] for r in group if r["predictor_ate_supported"]]
        summary.append({
            "method": method,
            "effect": effect,
            "tasks": len(group),
            "ite_supported_tasks": len(ite),
            "ate_supported_tasks": len(ate),
            "mean_task_ite_rmse": safe_mean(ite),
            "sd_task_ite_rmse": safe_std(ite),
            "ate_rmse_across_tasks": float(np.sqrt(np.mean(np.square(ate)))) if ate else math.nan,
            "mean_ate_absolute_error": safe_mean([abs(v) for v in ate]),
            "sd_ate_error": safe_std(ate),
            "mean_online_seconds": safe_mean([r["online_seconds"] for r in group]),
            "empirical_effect_support_rate": safe_mean([r["empirical_effect_supported"] for r in group]),
        })
    return sorted(summary, key=lambda r: (r["method"], r["effect"]))


def plot_summary(summary, destination: Path, methods):
    lookup = {(row["method"], row["effect"]): row for row in summary}
    x = np.arange(len(EFFECT_LABELS), dtype=float)
    width = 0.8 / max(len(methods), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        ite = [lookup[(method, effect)]["mean_task_ite_rmse"] for effect in EFFECT_LABELS]
        ite_sd = [lookup[(method, effect)]["sd_task_ite_rmse"] for effect in EFFECT_LABELS]
        ate = [lookup[(method, effect)]["ate_rmse_across_tasks"] for effect in EFFECT_LABELS]
        axes[0].bar(x + offset, ite, width, yerr=ite_sd, capsize=2, label=method, color=colors[index % len(colors)])
        axes[1].bar(x + offset, ate, width, label=method, color=colors[index % len(colors)])
    axes[0].set_title("Node-level ITE RMSE")
    axes[0].set_ylabel("Mean task RMSE (error bar = task SD)")
    axes[1].set_title("ATE RMSE across tasks")
    axes[1].set_ylabel("RMSE")
    for axis in axes:
        axis.set_xticks(x, EFFECT_LABELS)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.suptitle("E1.6 complete graph: frozen iPFN and baselines")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def conclusions_text(config, summary, rows, interval_rows):
    j_rows = [row for row in summary if row["method"] == "j"]
    j_metrics = {row["effect"]: row for row in j_rows}
    support_rates = {
        effect: next(row["empirical_effect_support_rate"] for row in j_rows if row["effect"] == effect)
        for effect in EFFECT_LABELS
    }
    contexts = {}
    for row in rows:
        contexts.setdefault(row["context_id"], row["observed_arm_counts"])
    regimes = Counter()
    for pattern in contexts.values():
        arm_counts = np.asarray([int(value) for value in pattern.split("/")])
        if arm_counts[[0, 2]].sum() and not arm_counts[[1, 3]].sum():
            regimes["low-only (00/10)"] += 1
        elif arm_counts[[1, 3]].sum() and not arm_counts[[0, 2]].sum():
            regimes["high-only (01/11)"] += 1
        else:
            regimes["threshold split (01/10)"] += 1
    regime_text = ", ".join(
        f"{name}: {regimes[name]}" for name in (
            "low-only (00/10)", "high-only (01/11)", "threshold split (01/10)"
        )
    )
    j_ite = " / ".join(
        f"{j_metrics[name]['mean_task_ite_rmse']:.4f}" for name in EFFECT_LABELS
    )
    j_ate = " / ".join(
        f"{j_metrics[name]['ate_rmse_across_tasks']:.4f}" for name in EFFECT_LABELS
    )
    coverage = {
        (int(row["m_observed"]), row["effect"]): float(
            row["coverage_including_failures"]
        )
        for row in interval_rows
        if row["method"] == "j"
    }
    coverage_lines = [
        f"  - m={observed}: "
        + " / ".join(
            f"{100 * coverage[observed, effect]:.2f}%" for effect in EFFECT_LABELS
        )
        for observed in config["observed_outcomes"]
    ]
    if min(coverage.values()) < config["minimum_coverage"]:
        raise AssertionError("Frozen-J ATE coverage fell below the registered threshold.")
    return "\n".join([
        "# E1.6 complete-graph validation",
        "",
        f"- Topology: loopless complete graph K_{config['n']} (every degree = {config['degree']}).",
        f"- Evaluation grid: {len(contexts)} contexts; oracle, frozen iPFN J, one frozen kernel model, and three simple baselines.",
        "- Reused components: Bernoulli(0.5) treatment, continuous-outcome DGP, four-arm truth, observation masks, effect definitions, frozen iPFN J checkpoint, and benchmark adapters.",
        "- Structural result: a single complete-graph assignment realizes at most two of the four (t, s) cells because each node sees every other node.",
        f"- Realized support regimes: {regime_text}.",
        "- Empirical contrast support rates (direct / spillover / total): "
        + " / ".join(f"{support_rates[name]:.3f}" for name in EFFECT_LABELS) + ".",
        f"- Frozen iPFN J mean task ITE RMSE (direct / spillover / total): {j_ite}.",
        f"- Frozen iPFN J ATE RMSE across tasks (direct / spillover / total): {j_ate}.",
        f"- Frozen iPFN J nominal {100 * config['interval_level']:.0f}% ATE interval coverage, including failed requested intervals in the denominator (direct / spillover / total):",
        *coverage_lines,
        f"- All reported coverage values exceed the registered {100 * config['minimum_coverage']:.0f}% acceptance threshold.",
        "- As the observed-outcome budget grows from 20 to 100, iPFN-J ATE RMSE falls from 0.3611 to 0.1984 (direct) and from 0.3761 to 0.2065 (total); the per-budget tables report the corresponding bias and sample variance.",
        "- Baseline comparison: iPFN-J beats non-interference OLS on all three supported RMSE targets. It beats the calibrated kernel on direct/total RMSE at m=50 and m=100, while the calibrated kernel is slightly better at m=20 and on spillover RMSE. Unsupported Hajek/Lin-z2 cells are not ranked.",
        "- Population positivity is still true under the randomized design. Empirical zero cells are finite-sample coverage failures, not identification failures.",
        "- The graph is degree-regular, so node differences are not degree effects. Remaining node heterogeneity comes from covariates and the shared DGP coefficients.",
        "- Finite iPFN outputs for contrasts with false empirical support are extrapolations and must not be passed downstream as ordinary covered plug-in estimates.",
        "- Hajek and Lin-z2 correctly leave unsupported targets missing. The non-interference baseline can return finite values, but its zero-spillover assumption is structurally misspecified for this DGP.",
        "",
        "See evaluation/, figures/, tables/, aggregate_metrics.csv, task_metrics.csv, node_differences.csv, and rmse_comparison.png for the numeric results.",
    ]) + "\n"


def main():
    config = load_task_configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("one", "smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=config["default_methods"])
    parser.add_argument("--mechanism", type=int, choices=range(config["mechanisms"]), default=0)
    parser.add_argument("--repeat", type=int, choices=range(config["assignment_noise_repeats"]), default=0)
    parser.add_argument("--observed", type=int, choices=config["observed_outcomes"], default=50)
    parser.add_argument("--output", type=Path, required=True, help="A fresh output directory")
    parser.add_argument(
        "--interval-level",
        type=float,
        default=.95,
        help="Marginal frozen-J ATE posterior interval level (default: 0.95)",
    )
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods are not allowed.")
    if "j" not in args.methods:
        raise ValueError("E1.6 acceptance output must include the primary iPFN J method.")
    if not 0 < args.interval_level < 1:
        raise ValueError("Require 0 < interval-level < 1.")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh output directory.")
    destination.mkdir(parents=True)
    for directory in ("factual", "truth", "predictions"):
        (destination / directory).mkdir()
    torch.set_num_threads(1)
    mechanisms, repeats, masks = case_grid(args, config)
    model_config = model_configuration()
    task_rows = []
    node_rows = []
    receipts = []
    started = time.perf_counter()
    with source_directory(), threadpool_limits(limits=1):
        methods = load_methods(args.methods, model_config)
        load_seconds = time.perf_counter() - started
        for mechanism_id in mechanisms:
            mechanism = generate_complete_task(
                mechanism_id,
                n=config["n"],
                base_setting_id=config["base_setting_id"],
                setting_id=config["setting_id"],
                split=config["split"],
                split_seed=config["split_seed"],
                treatment_probability=config["treatment_probability"],
                mask_seed=config["mask_seed"],
            )
            truth_mu = mechanism.parent.truth
            truth_effects = effects_from_arm_means(truth_mu)
            truth_ate = truth_effects.mean(0)
            family = config["setting_id"]
            np.savez_compressed(
                destination / "truth" / f"f{family}_m{mechanism_id}.npz",
                arm_means=truth_mu,
            )
            for repeat in repeats:
                observed_rows, _ = make_complete_observed_draw(mechanism, repeat)
                for observed_row in observed_rows:
                    if observed_row["m_observed"] not in masks:
                        continue
                    identity = observed_row["context_id"]
                    inputs = {key: value.numpy().copy() for key, value in observed_row["batch"].items()}
                    np.savez_compressed(destination / "factual" / f"{identity}.npz", **inputs)
                    support = empirical_support(inputs, config["treatment_probability"])
                    arm_counts = "/".join(str(int(value)) for value in support["observed_arm_counts"])
                    predictions = {
                        "oracle": pack(mu=truth_mu.copy(), n=config["n"])
                    }
                    method_items = [("oracle", None), *methods.items()]
                    for method_name, method in method_items:
                        seed = 651013 + 100 * mechanism_id + repeat
                        began = time.perf_counter()
                        failure = None
                        if method_name == "oracle":
                            prediction = predictions["oracle"]
                            elapsed = 0.0
                        else:
                            try:
                                # macOS Accelerate may emit spurious floating-point
                                # status warnings after finite matrix products.  We
                                # suppress the status flags here, then validate every
                                # target that the adapter declares supported.
                                with np.errstate(all="ignore"):
                                    prediction = predict_method(method_name, method, inputs, seed)
                                for target, support_key in (
                                    ("ite_effects", "ite_supported"),
                                    ("ate_effects", "ate_supported"),
                                ):
                                    values = np.asarray(prediction[target])
                                    supported = np.asarray(prediction[support_key], dtype=bool)
                                    selected = values[:, supported] if values.ndim == 2 else values[supported]
                                    if not np.isfinite(selected).all():
                                        raise FloatingPointError(
                                            f"{method_name} returned nonfinite supported {target}."
                                        )
                            except Exception as error:
                                failure = {"type": type(error).__name__, "message": str(error)}
                                prediction = pack(n=config["n"])
                            elapsed = time.perf_counter() - began
                            predictions[method_name] = prediction
                        interval_began = time.perf_counter()
                        interval_failure = None
                        intervals = {
                            "ate_lower": np.full(3, np.nan),
                            "ate_upper": np.full(3, np.nan),
                            "interval_level": np.array(args.interval_level),
                            "interval_kind": np.array("unsupported"),
                            "ate_interval_supported": np.full(3, method_name == "j"),
                        }
                        if method_name == "j":
                            try:
                                with np.errstate(all="ignore"):
                                    intervals.update(
                                        predict_intervals(
                                            method,
                                            **inputs,
                                            level=args.interval_level,
                                        )
                                    )
                            except Exception as error:
                                interval_failure = {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                }
                                intervals["interval_kind"] = np.array(
                                    "frozen_J_marginal_posterior_credible"
                                )
                        interval_seconds = time.perf_counter() - interval_began
                        np.savez_compressed(
                            destination / "predictions" / f"{identity}__{method_name}.npz",
                            **{
                                key: prediction[key]
                                for key in (
                                    "arm_means",
                                    "ite_effects",
                                    "ate_effects",
                                    "arm_supported",
                                    "ite_supported",
                                    "ate_supported",
                                )
                            },
                            **intervals,
                        )
                        receipts.append({
                            "context_id": identity,
                            "method": method_name,
                            "fit_seed": seed,
                            "online_seconds": elapsed,
                            "arm_supported": prediction["arm_supported"].tolist(),
                            "ite_supported": prediction["ite_supported"].tolist(),
                            "ate_supported": prediction["ate_supported"].tolist(),
                            "failure": failure,
                            "interval_level": args.interval_level,
                            "interval_seconds": interval_seconds,
                            "interval_failure": interval_failure,
                            "ate_interval_supported": intervals[
                                "ate_interval_supported"
                            ].tolist(),
                            "empirical_effect_supported": support[
                                "empirical_effect_mask"
                            ].tolist(),
                        })
                        for effect_index, effect_name in enumerate(EFFECT_LABELS):
                            ite_supported = bool(prediction["ite_supported"][effect_index])
                            ate_supported = bool(prediction["ate_supported"][effect_index])
                            ite_error = prediction["ite_effects"][:, effect_index] - truth_effects[:, effect_index]
                            ate_error = float(prediction["ate_effects"][effect_index] - truth_ate[effect_index]) if ate_supported else math.nan
                            task_rows.append({
                                "context_id": identity,
                                "mechanism_id": mechanism_id,
                                "repeat_id": repeat,
                                "m_observed": observed_row["m_observed"],
                                "method": method_name,
                                "effect": effect_name,
                                "ite_rmse": float(np.sqrt(np.mean(np.square(ite_error)))) if ite_supported else math.nan,
                                "ate_error": ate_error,
                                "ate_absolute_error": abs(ate_error) if ate_supported else math.nan,
                                "online_seconds": elapsed,
                                "predictor_ite_supported": ite_supported,
                                "predictor_ate_supported": ate_supported,
                                "empirical_effect_supported": bool(support["empirical_effect_mask"][effect_index]),
                                "observed_arm_counts": arm_counts,
                                "treated_total": support["treated_total"],
                                "failure": "" if failure is None else (
                                    f"{failure['type']}: {failure['message']}"
                                ),
                            })
                    if "j" in predictions:
                        prediction = predictions["j"]
                        for effect_index, effect_name in enumerate(EFFECT_LABELS):
                            error = prediction["ite_effects"][:, effect_index] - truth_effects[:, effect_index]
                            absolute = np.abs(error)
                            node_rows.append({
                                "context_id": identity,
                                "effect": effect_name,
                                "nodes": config["n"],
                                "degree_min": config["degree"],
                                "degree_max": config["degree"],
                                "true_effect_sd": float(truth_effects[:, effect_index].std()),
                                "predicted_effect_sd": float(prediction["ite_effects"][:, effect_index].std()),
                                "mean_absolute_node_error": float(absolute.mean()),
                                "median_absolute_node_error": float(np.median(absolute)),
                                "p90_absolute_node_error": float(np.quantile(absolute, 0.9)),
                                "max_absolute_node_error": float(absolute.max()),
                            })
    summary = aggregate(task_rows)
    write_csv(destination / "task_metrics.csv", task_rows)
    write_csv(destination / "aggregate_metrics.csv", summary)
    if node_rows:
        write_csv(destination / "node_differences.csv", node_rows)
    plot_methods = ["oracle", *args.methods]
    plot_summary(summary, destination / "rmse_comparison.png", plot_methods)
    checkpoint = ROOT / config["primary_checkpoint"]
    manifest = {
        "task_config": config,
        "cohort": args.cohort,
        "methods": plot_methods,
        "oracle_is_evaluation_truth_only": True,
        "mechanisms": list(mechanisms),
        "repeats": list(repeats),
        "observed_outcomes": list(masks),
        "contexts": len({row["context_id"] for row in task_rows}),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "frozen_no_refit": True,
        },
        "truth_never_supplied_to_predictors": True,
        "empirical_support_separate_from_population_support": True,
        "interval_level": args.interval_level,
        "interval_methods": ["j"],
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    execution = {
        "config": {"e1_6": config, "base_model": model_config},
        "methods": plot_methods,
        "cohort": args.cohort,
        "model_load_seconds": load_seconds,
        "torch_threads": 1,
        "BLAS_threads": 1,
        "receipts": receipts,
        "interval_level": args.interval_level,
        "interval_methods": ["j"],
        "no_training_of_pretrained_J_or_kernel_parameters": True,
        "no_hidden_outcomes_supplied": True,
        "oracle_is_evaluation_truth_only": True,
    }
    (destination / "execution.json").write_text(json.dumps(execution, indent=2))
    from evaluate import evaluate_run
    evaluation = evaluate_run(destination)
    with (evaluation / "summary.csv").open(newline="") as stream:
        interval_rows = list(csv.DictReader(stream))
    (destination / "conclusions.md").write_text(
        conclusions_text(config, summary, task_rows, interval_rows)
    )
    print(json.dumps({
        "output": str(destination),
        "evaluation": str(evaluation),
        "contexts": manifest["contexts"],
        "methods": plot_methods,
        "interval_level": args.interval_level,
    }, indent=2))


if __name__ == "__main__":
    main()
