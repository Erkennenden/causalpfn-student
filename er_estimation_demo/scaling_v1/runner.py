"""Fixed-update Adam training and exact three-effect evaluation on one device."""
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch

from train_local_network_interference import (LocalNetworkQueryTransformer, ModelConfig,
    compute_training_losses, resolve_device as legacy_device, seed_everything)
from . import VERSION
from .data import ROOT, TaskBank, atomic_save, batch_indices, write_json

MODEL_SIZES = {"small": (2, 32), "medium": (4, 64), "large": (8, 96), "xlarge": (12, 128)}
EFFECTS = ("direct", "spillover", "total")


@dataclass(frozen=True)
class RunConfig:
    graph_file: str = str(ROOT / "data/fixed_er_N1000_p0.02_seed12345.json")
    cache_dir: str = str(ROOT / "runs/scaling_cache")
    model_size: str = "medium"
    train_tasks: int = 1024
    validation_tasks: int = 64
    test_tasks: int = 128
    steps: int = 2000
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    noise_sd: float = 4.
    data_seed: int = 12345
    train_seed: int = 2026
    validation_every: int = 250
    checkpoint_every: int = 500
    log_every: int = 50
    query_chunk_size: int = 16
    threads: int = 4
    device: str = "auto"

    def __post_init__(self):
        for name in ("train_tasks", "validation_tasks", "test_tasks", "steps", "batch_size",
                     "validation_every", "checkpoint_every", "log_every", "query_chunk_size", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.model_size not in MODEL_SIZES:
            raise ValueError("Unknown model_size.")
        for name in ("learning_rate", "weight_decay", "noise_sd"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive.")
        for name in ("data_seed", "train_seed"):
            if not isinstance(getattr(self, name), int) or not 0 <= getattr(self, name) < 2**63:
                raise ValueError(f"{name} must be a nonnegative signed-64-bit integer.")


def device_for(name):
    if name.startswith("cuda") or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda" if name == "auto" else name)
    return legacy_device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "musa":
        torch.musa.synchronize()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_config(size):
    depth, width = MODEL_SIZES[size]
    return ModelConfig(num_layers=depth, d_model=width, num_heads=4, ffn_dim=4*width, dropout=0.)


def effect_metrics(predicted, truth):
    predicted, truth = np.asarray(predicted, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("Effect arrays must have matching [task,node,3] shapes.")
    if not predicted.size or not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("Effect arrays must be nonempty and finite.")
    error = predicted-truth
    ate_error = error.mean(axis=1)
    return [dict(effect=name, ite_rmse=float(np.sqrt(np.mean(error[..., i]**2))),
        ite_mae=float(np.mean(np.abs(error[..., i]))),
        ate_rmse=float(np.sqrt(np.mean(ate_error[:, i]**2))), ate_bias=float(ate_error[:, i].mean()),
        zero_predictor_ite_rmse=float(np.sqrt(np.mean(truth[..., i]**2))))
        for i, name in enumerate(EFFECTS)]


def validation_nll(model, bank, count, device):
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.no_grad():
            for task_id in range(count):
                batch = to_device(bank.batch("validation", [task_id]), device)
                prediction = model(batch["tokens"], batch["queries"], batch["adjacency"])
                values.append(float(compute_training_losses(model, prediction, batch)["total_loss"].cpu()))
    finally:
        model.train(was_training)
    value = float(np.mean(values))
    if not math.isfinite(value):
        raise FloatingPointError("Validation NLL is nonfinite.")
    return value


def evaluate_model(model, bank, count, device, output, query_chunk_size=16):
    # Reuse the three CEPO means for all coherent effects.
    from evaluation.unified import predict_pfn_majority_ite
    predictions, truths, rows = [], [], []
    start = time.perf_counter()
    for task_id in range(count):
        batch = to_device(bank.batch("test", [task_id]), device)
        predicted = predict_pfn_majority_ite(model=model, batch=batch, treatment_prob=.5,
                                            query_chunk_size=query_chunk_size).detach().cpu()
        truth = batch["oracle_ite"].detach().cpu()
        predictions.append(predicted)
        truths.append(truth)
        error = (predicted-truth).double().numpy()[0]
        for i, name in enumerate(EFFECTS):
            rows.append(dict(task_id=task_id, effect=name, ite_mse=float(np.mean(error[:, i]**2)),
                             ate_error=float(np.mean(error[:, i]))))
        if (task_id+1) % 16 == 0 or task_id+1 == count:
            print(f"exact test aggregation: {task_id+1}/{count} tasks", flush=True)
    prediction, truth = torch.cat(predictions), torch.cat(truths)
    metrics = effect_metrics(prediction.numpy(), truth.numpy())
    atomic_save(dict(predicted=prediction, truth=truth, task_ids=torch.arange(count)), Path(output)/"test_predictions.pt")
    with (Path(output)/"test_task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(effects=metrics, test_tasks=count, test_seconds=time.perf_counter()-start)


def resume_identity(config):
    # Paths/device may change when moving a checkpoint; data hashes are checked separately.
    return {k: v for k, v in asdict(config).items() if k not in {"steps", "graph_file", "cache_dir", "device", "threads"}}


def train(config: RunConfig, output, *, resume=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Run directory already contains files: {output}. Use --resume or a new --output.")
    resume_path = output/"model_last.pt"
    if resume and not resume_path.exists():
        resume_path = output/"model_initial.pt"
        if not resume_path.exists():
            raise FileNotFoundError("--resume requires model_last.pt or model_initial.pt in the run directory.")
    torch.set_num_threads(config.threads)
    device = device_for(config.device)
    seed_everything(config.train_seed)
    bank = TaskBank(config.graph_file, config.cache_dir, data_seed=config.data_seed, noise_sd=config.noise_sd)
    mconfig = model_config(config.model_size)
    model = LocalNetworkQueryTransformer(mconfig).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    step, best_step, best_nll, best_state = 0, 0, math.inf, None
    scheduled_best = dict(step=0, nll=math.inf, state=None)
    seen = set()
    timing = dict(data_seconds=0., training_seconds=0., validation_seconds=0.)
    previous_wall = 0.
    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_path.name == "model_initial.pt" and state["step"] != 0:
            raise ValueError("Initial checkpoint must contain step zero.")
        if state["version"] != VERSION or resume_identity(RunConfig(**state["config"])) != resume_identity(config):
            raise ValueError("Resume config mismatch; use a new output directory.")
        if state["data_identity"] != bank.fingerprint:
            raise ValueError("Resume data identity mismatch.")
        if config.steps < state["step"]:
            raise ValueError("Requested steps is smaller than the saved checkpoint step.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        step, best_step, best_nll = state["step"], state["best_step"], state["best_validation_nll"]
        best_state, seen = state["best_model_state"], set(state["seen_task_ids"])
        scheduled_best = state["scheduled_best"]
        if config.steps > step:
            # A former endpoint is no longer a validation candidate unless it was
            # on the fixed cadence. This makes extension select the same model.
            best_step, best_nll, best_state = (scheduled_best["step"],
                scheduled_best["nll"], scheduled_best["state"])
        timing, previous_wall = state["timing"], state["training_wall_seconds"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"fixed_graph.json", bank.record)
    write_json(output/"manifest.json", dict(version=VERSION, config=asdict(config),
        model_config=asdict(mconfig), parameters=sum(p.numel() for p in model.parameters()),
        data_identity=bank.fingerprint, data_config=bank.identity, device=str(device), torch_version=str(torch.__version__)))
    write_json(output/"status.json", dict(status="training", step=step, target_steps=config.steps))
    (output/"results.json").unlink(missing_ok=True)
    wall_start = time.perf_counter()

    def checkpoint(current_step):
        return dict(version=VERSION, config=asdict(config), model_config=asdict(mconfig),
            data_identity=bank.fingerprint, graph_record=bank.record, model_state=cpu_tree(model.state_dict()),
            optimizer_state=cpu_tree(optimizer.state_dict()), step=current_step, best_step=best_step,
            best_validation_nll=best_nll, best_model_state=best_state, seen_task_ids=sorted(seen),
            scheduled_best=scheduled_best,
            timing=dict(timing), training_wall_seconds=previous_wall+time.perf_counter()-wall_start)

    if not resume:
        atomic_save(checkpoint(0), output/"model_initial.pt")
    history = output/"history.csv"
    fields = ["step", "train_nll", "validation_nll", "unique_train_tasks_seen", "task_presentations",
              "training_seconds", "data_seconds", "validation_seconds"]
    kept = []
    if resume and history.exists():
        with history.open() as handle:
            kept = [row for row in csv.DictReader(handle) if int(row["step"]) <= step]
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
        handle.flush()
        while step < config.steps:
            ids = batch_indices(step, config.train_tasks, config.batch_size, config.train_seed)
            before = time.perf_counter()
            batch = to_device(bank.batch("train", ids), device)
            synchronize(device)
            timing["data_seconds"] += time.perf_counter()-before
            before = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["tokens"], batch["queries"], batch["adjacency"])
            loss = compute_training_losses(model, prediction, batch)["total_loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss at step {step+1}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            timing["training_seconds"] += time.perf_counter()-before
            step += 1
            seen.update(ids)
            val = None
            validate = step % config.validation_every == 0 or step == config.steps
            if validate:
                before = time.perf_counter()
                val = validation_nll(model, bank, config.validation_tasks, device)
                timing["validation_seconds"] += time.perf_counter()-before
                if step % config.validation_every == 0 and val < scheduled_best["nll"]:
                    scheduled_best = dict(step=step, nll=val, state=cpu_tree(model.state_dict()))
                if val < best_nll:
                    best_step, best_nll, best_state = step, val, cpu_tree(model.state_dict())
            if step % config.log_every == 0 or validate:
                row = dict(step=step, train_nll=float(loss.detach().cpu()), validation_nll=val,
                    unique_train_tasks_seen=len(seen), task_presentations=step*config.batch_size, **timing)
                writer.writerow(row)
                handle.flush()
                print(f"step {step}/{config.steps} train_nll={row['train_nll']:.5f} val_nll={val} seen={len(seen)}", flush=True)
            if validate or step % config.checkpoint_every == 0:
                state = checkpoint(step)
                atomic_save(state, output/"model_last.pt")
                if best_state is not None:
                    best_payload = {**state, "step": best_step, "model_state": best_state}
                    # This checkpoint is for inference; optimizer belongs to the latest step.
                    best_payload.pop("optimizer_state")
                    atomic_save(best_payload, output/"model_best.pt")
                if step % config.checkpoint_every == 0 or step == config.steps:
                    atomic_save(state, output/f"model_step_{step:06d}.pt")
                write_json(output/"status.json", dict(status="training", step=step, target_steps=config.steps))
    # Always rebuild the inference checkpoint from the last committed best state on resume.
    if best_state is None:
        raise RuntimeError("No validated checkpoint was produced.")
    atomic_save(dict(version=VERSION, config=asdict(config), model_config=asdict(mconfig),
        data_identity=bank.fingerprint, graph_record=bank.record, step=best_step,
        model_state=best_state, best_validation_nll=best_nll), output/"model_best.pt")
    model.load_state_dict(best_state)
    write_json(output/"status.json", dict(status="evaluating", step=step, evaluated_step=best_step))
    evaluation = evaluate_model(model, bank, config.test_tasks, device, output, config.query_chunk_size)
    result = dict(version=VERSION, status="complete", config=asdict(config),
        data_identity=bank.fingerprint, graph_sha256=bank.record["adjacency_sha256"],
        parameters=sum(p.numel() for p in model.parameters()), evaluated_step=best_step,
        best_validation_nll=best_nll, unique_train_tasks_seen=len(seen), task_presentations=step*config.batch_size,
        device=str(device), timing=timing, **evaluation)
    write_json(output/"results.json", result)
    write_json(output/"status.json", dict(status="complete", step=step, evaluated_step=best_step))
    print(f"Finished: {output.resolve()}", flush=True)
    return result


def evaluate_checkpoint(checkpoint_path, output, *, cache_dir, device="auto", test_tasks=128, threads=4, query_chunk_size=16):
    if test_tasks < 1 or threads < 1 or query_chunk_size < 1:
        raise ValueError("test_tasks, threads and query_chunk_size must be positive.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output already contains files; choose a new directory.")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("version") != VERSION:
        raise ValueError("Old effect-loss scaling checkpoint; retrain with CEPO")
    config = RunConfig(**state["config"])
    torch.set_num_threads(threads)
    actual_device = device_for(device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"fixed_graph.json", state["graph_record"])
    bank = TaskBank(output/"fixed_graph.json", cache_dir, data_seed=config.data_seed, noise_sd=config.noise_sd)
    if bank.fingerprint != state["data_identity"]:
        raise ValueError("Checkpoint data identity differs from the current code or graph.")
    model = LocalNetworkQueryTransformer(ModelConfig(**state["model_config"])).to(actual_device)
    model.load_state_dict(state["model_state"])
    result = dict(evaluated_step=state["step"], data_identity=bank.fingerprint,
                  **evaluate_model(model, bank, test_tasks, actual_device, output, query_chunk_size))
    write_json(output/"evaluation.json", result)
    return result
