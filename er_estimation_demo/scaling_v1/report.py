"""Collect per-run metrics; optionally plot with matplotlib (no power-law fit)."""
import csv
import json
from pathlib import Path

EFFECTS = ("direct", "spillover", "total")


def summarize(directory, *, plot=False):
    directory = Path(directory)
    results = []
    for path in sorted(directory.rglob("results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        status_path, manifest_path = path.parent/"status.json", path.parent/"manifest.json"
        if not status_path.exists() or not manifest_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (result.get("version") == "causalfm_er_scaling_cepo_v2" and result.get("status") == "complete"
                and status.get("status") == "complete" and manifest.get("config") == result.get("config")
                and manifest.get("data_identity") == result.get("data_identity")):
            results.append((path.parent, result))
    if not results:
        raise ValueError(f"No completed scaling runs found in {directory}.")
    rows = []
    for run, result in results:
        config = result["config"]
        for effect in result["effects"]:
            rows.append(dict(run=str(run.relative_to(directory)), model_size=config["model_size"],
                train_tasks=config["train_tasks"], train_seed=config["train_seed"], steps=config["steps"],
                evaluated_step=result["evaluated_step"], parameters=result["parameters"],
                unique_train_tasks_seen=result["unique_train_tasks_seen"],
                task_presentations=result["task_presentations"], batch_size=config["batch_size"],
                validation_tasks=config["validation_tasks"], test_tasks=config["test_tasks"],
                data_identity=result["data_identity"], device=result["device"],
                training_seconds=result["timing"]["training_seconds"],
                data_seconds=result["timing"]["data_seconds"],
                validation_seconds=result["timing"]["validation_seconds"],
                best_validation_nll=result["best_validation_nll"], **effect))
    with (directory/"summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        _plots(directory, results)
    print(f"Summary: {(directory/'summary.csv').resolve()}", flush=True)
    return rows


def _plots(directory, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("CSV was saved. Plotting additionally needs matplotlib: python -m pip install matplotlib") from exc
    # Keep incompatible budgets/data/LRs in separate panels/files instead of joining them.
    cohorts = {}
    ignored = {"model_size", "train_tasks", "graph_file", "cache_dir", "threads", "log_every"}
    for path, result in results:
        identity = {k: v for k, v in result["config"].items() if k not in ignored}
        identity.update(data_identity=result["data_identity"], actual_device=result["device"])
        key = json.dumps(identity, sort_keys=True)
        cohorts.setdefault(key, []).append((path, result))
    for group_id, runs in enumerate(cohorts.values(), 1):
        for axis_name in ("tasks", "parameters"):
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
            if axis_name == "tasks":
                selected = [r for _, r in runs if r["config"]["model_size"] == "medium"]
                get_x = lambda r: r["config"]["train_tasks"]
                xlabel = "Distinct training task bank size"
            else:
                count = max(r["config"]["train_tasks"] for _, r in runs)
                selected = [r for _, r in runs if r["config"]["train_tasks"] == count]
                get_x = lambda r: r["parameters"]
                xlabel = f"PFN parameters (task bank={count})"
            selected.sort(key=get_x)
            for i, effect in enumerate(EFFECTS):
                axes[i].plot([get_x(r) for r in selected], [r["effects"][i]["ite_rmse"] for r in selected], "o-")
                axes[i].set(xlabel=xlabel, ylabel="Aggregated ITE RMSE", title=effect)
                axes[i].set_xscale("log", base=2)
                axes[i].grid(alpha=.25)
            seen=[r["unique_train_tasks_seen"] for r in selected]
            partial=any(r["unique_train_tasks_seen"] < r["config"]["train_tasks"] for r in selected)
            fig.suptitle(("PARTIAL TRAINING / preview only | " if partial else "") + "Actual distinct tasks seen: " + ", ".join(map(str,seen)), fontsize=10)
            fig.savefig(directory/f"scaling_{axis_name}_group{group_id}.png", dpi=180)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for path, result in runs:
            with (path/"history.csv").open() as handle:
                records = [row for row in csv.DictReader(handle) if row["validation_nll"]]
            label = f"{result['config']['model_size']}, D={result['config']['train_tasks']}"
            values = [float(row["validation_nll"]) for row in records]
            axes[0].plot([int(row["step"]) for row in records], values, marker=".", label=label)
            axes[1].plot([float(row["training_seconds"]) for row in records], values, marker=".", label=label)
        axes[0].set(xlabel="Optimizer updates", ylabel="Validation CEPO GMM NLL")
        axes[1].set(xlabel="Training compute time (seconds)", ylabel="Validation CEPO GMM NLL")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.savefig(directory/f"training_curves_group{group_id}.png", dpi=180)
        plt.close(fig)
