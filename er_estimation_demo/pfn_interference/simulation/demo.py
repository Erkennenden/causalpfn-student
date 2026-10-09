"""导出可查看的 DGP 样例：python -m pfn_interference.simulation.demo。"""
import argparse
import csv
import json
from pathlib import Path

import torch

from pfn_interference.graph import _decode_graph, adjacency_sha256
from .dgp import generate_task
from .outcomes import OUTCOME_FAMILIES


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-file", type=Path, default=root / "data/fixed_er_N1000_p0.02_seed12345.json")
    parser.add_argument("--output", type=Path, default=root / "runs" / "dgp_example")
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-sd", type=float, default=4.)
    parser.add_argument("--family", choices=("mixture", *OUTCOME_FAMILIES), default="mixture")
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    record = json.loads(args.graph_file.read_text(encoding="utf-8"))
    graph = _decode_graph(record["n_units"], record["upper_hex"])
    if adjacency_sha256(graph) != record["adjacency_sha256"]:
        raise ValueError("Graph file SHA256 does not match its adjacency.")
    tasks = [generate_task(graph, task_id=i, seed=args.seed, noise_sd=args.noise_sd, outcome_family=args.family)
             for i in range(args.tasks)]
    batch = {key: torch.cat([task.batch[key] for task in tasks], dim=0) for key in tasks[0].batch}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output / "batch.pt")

    columns = ["task_id", "node_id", "x", "neighbor_x", "treatment", "exposure", "y_obs", "epsilon",
               "mu_t0_low", "mu_t0_high", "mu_t1_low", "mu_t1_high",
               "direct_effect", "spillover_effect", "total_effect"]
    with (args.output / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for task in tasks:
            b = {key: value[0] for key, value in task.batch.items()}
            for i in range(graph.shape[0]):
                observed = [float(b[key][i]) for key in
                            ("x", "neighbor_x", "observed_treatment", "observed_exposure", "y_obs", "noise")]
                writer.writerow([task.metadata["task_id"], i, *observed,
                                 *b["oracle_arm_means"][i].flatten().tolist(), *b["oracle_ite"][i].tolist()])

    mechanisms = [dict(metadata=task.metadata, function=task.outcome.to_dict()) for task in tasks]
    (args.output / "mechanisms.json").write_text(
        json.dumps(mechanisms, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "graph_file": str(args.graph_file.resolve()), "graph_sha256": record["adjacency_sha256"],
        "tasks": args.tasks, "units_per_task": graph.shape[0],
        "effect_order": ["direct", "spillover", "total"],
        "ate_by_task": batch["oracle_ate"].tolist(),
        "max_total_minus_direct_minus_spillover": float((batch["oracle_ite"][..., 2]
            - batch["oracle_ite"][..., 0] - batch["oracle_ite"][..., 1]).abs().max()),
        "model_training_run": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
