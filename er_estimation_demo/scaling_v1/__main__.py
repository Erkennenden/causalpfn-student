"""Usage: python -m scaling_v1 {smoke,train,sweep,evaluate,report}."""
import argparse
from dataclasses import asdict, fields, replace
import os
from pathlib import Path
import time

from .data import ROOT, write_json
from .runner import MODEL_SIZES, RunConfig, evaluate_checkpoint, train
from .report import summarize

SWEEP = (("A_medium_D256", "medium", 256), ("B_medium_D1024", "medium", 1024),
         ("C_medium_D4096", "medium", 4096), ("D_small_D4096", "small", 4096),
         ("E_large_D4096", "large", 4096))


def training_arguments(parser, *, sweep=False):
    default = RunConfig()
    for field in fields(default):
        name = field.name
        if sweep and name in {"model_size", "train_tasks", "train_seed"}:
            continue
        value = getattr(default, name)
        flags = ["--"+name.replace("_", "-")]
        if name == "validation_tasks":
            flags.append("--val-tasks")
        kwargs = dict(default=value, type=type(value))
        if name == "model_size":
            kwargs["choices"] = tuple(MODEL_SIZES)
        parser.add_argument(*flags, **kwargs)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="End-to-end small run on the unchanged N1000 graph")
    training_arguments(smoke)
    smoke.set_defaults(model_size="small", train_tasks=8, validation_tasks=2, test_tasks=2,
                       steps=20, batch_size=1, validation_every=10, checkpoint_every=10, log_every=5)
    training_arguments(commands.add_parser("train", help="Train one configuration"))
    sweep = commands.add_parser("sweep", help="Run the five configurations sequentially")
    training_arguments(sweep, sweep=True)
    sweep.add_argument("--seeds", nargs="+", type=int, default=[2026])
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on the fixed test stream")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--cache-dir", default=str(ROOT/"runs/scaling_cache"))
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--test-tasks", type=int, default=128)
    evaluation.add_argument("--threads", type=int, default=4)
    evaluation.add_argument("--query-chunk-size", type=int, default=16)
    report = commands.add_parser("report", help="Collect CSV and optional PNG curves")
    report.add_argument("directory", type=Path)
    report.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "report":
        summarize(args.directory, plot=args.plot)
        return
    if args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint, args.output, cache_dir=args.cache_dir, device=args.device,
                            test_tasks=args.test_tasks, threads=args.threads, query_chunk_size=args.query_chunk_size)
        return
    config = RunConfig(**{field.name: getattr(args, field.name) for field in fields(RunConfig) if hasattr(args, field.name)})
    output = args.output or ROOT/"runs"/f"scaling_{args.command}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    print(f"Output: {output.resolve()}", flush=True)
    if args.command in {"train", "smoke"}:
        train(config, output, resume=args.resume)
        summarize(output)
        return
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**63 for seed in args.seeds):
        parser.error("--seeds must contain distinct nonnegative signed-64-bit integers")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Sweep output already contains files; use --resume or a new --output.")
    jobs = [(f"{name}_seed{seed}", replace(config, model_size=size, train_tasks=count, train_seed=seed))
            for seed in args.seeds for name, size, count in SWEEP]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"sweep_plan.json", [dict(name=name, config=asdict(cfg)) for name, cfg in jobs])
    for i, (name, cfg) in enumerate(jobs, 1):
        print(f"Run {i}/{len(jobs)}: {name}", flush=True)
        target = output/name
        train(cfg, target, resume=args.resume and (target/"model_last.pt").exists())
        summarize(output)


if __name__ == "__main__":
    main()
