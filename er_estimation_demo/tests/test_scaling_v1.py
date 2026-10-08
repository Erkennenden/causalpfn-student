"""End-to-end contracts for the new scaling runner; no mocked training."""
import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    assert importlib.util.find_spec("scaling_v1") is not None, "scaling v1 not implemented"
    return importlib.import_module("scaling_v1." + name)


@pytest.fixture
def graph_file(tmp_path):
    from fixed_er_graph import adjacency_sha256
    graph = torch.ones(3, 3) - torch.eye(3)
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(dict(n_units=3, upper_hex="e0", adjacency_sha256=adjacency_sha256(graph))))
    return path


def config(graph_file, tmp_path, **kwargs):
    cls = module("runner").RunConfig
    return cls(graph_file=str(graph_file), cache_dir=str(tmp_path / "cache"),
               train_tasks=4, validation_tasks=2, test_tasks=2, batch_size=2,
               steps=kwargs.pop("steps", 4), model_size="small", device="cpu", threads=1,
               validation_every=2, checkpoint_every=2, log_every=1, **kwargs)


def test_cache_has_one_graph_and_split_tasks_are_reproducible(graph_file, tmp_path):
    bank = module("data").TaskBank(graph_file, tmp_path / "cache", data_seed=37)
    batch = bank.batch("train", [0, 1])
    again = bank.batch("train", [1, 0])
    torch.testing.assert_close(batch["tokens"][0], again["tokens"][1], rtol=0, atol=0)
    assert batch["adjacency"].shape == (2, 3, 3)
    assert batch["adjacency"].untyped_storage().nbytes() == 3*3*4
    task_files = list(bank.directory.glob("train/*.pt"))
    assert len(task_files) == 2
    payload = torch.load(task_files[0], weights_only=True)
    assert "adjacency" not in payload["tensors"]
    assert not torch.equal(batch["tokens"][0], bank.batch("validation", [0])["tokens"][0])
    reloaded = module("data").TaskBank(graph_file, tmp_path / "cache", data_seed=37)
    torch.testing.assert_close(batch["tokens"], reloaded.batch("train", [0,1])["tokens"], rtol=0, atol=0)
    changed = module("data").TaskBank(graph_file, tmp_path / "cache", data_seed=38)
    assert changed.directory != bank.directory


def test_sampler_covers_each_epoch_and_does_not_depend_on_call_order():
    sampler = module("data").batch_indices
    sequence = sum([sampler(step, 3, 2, 123) for step in range(3)], [])
    assert sorted(sequence[:3]) == [0, 1, 2]
    assert sorted(sequence[3:]) == [0, 1, 2]
    assert sampler(2, 3, 2, 123) == sequence[4:]


def test_effect_metrics_keep_ite_and_ate_distinct():
    # Opposite node errors cancel in ATE but must remain visible in ITE RMSE.
    predicted = np.array([[[1., 2., 3.], [-1., -2., -3.]]])
    result = module("runner").effect_metrics(predicted, np.zeros_like(predicted))
    assert [row["ite_rmse"] for row in result] == [1., 2., 3.]
    assert [row["ate_rmse"] for row in result] == [0., 0., 0.]


def test_train_updates_parameters_and_evaluates_saved_best_model(graph_file, tmp_path):
    runner = module("runner")
    cfg = config(graph_file, tmp_path)
    result = runner.train(cfg, tmp_path / "run")
    last = torch.load(tmp_path / "run/model_last.pt", weights_only=True)
    initial = torch.load(tmp_path / "run/model_initial.pt", weights_only=True)
    assert last["step"] == 4
    assert any(not torch.equal(last["model_state"][k], initial["model_state"][k]) for k in last["model_state"])
    assert result["status"] == "complete"
    assert result["unique_train_tasks_seen"] == 4
    assert len(result["effects"]) == 3
    assert all(np.isfinite(row["ite_rmse"]) for row in result["effects"])
    saved = torch.load(tmp_path / "run/model_best.pt", weights_only=True)
    assert result["evaluated_step"] == saved["step"]
    assert (tmp_path / "run/test_task_metrics.csv").exists()
    assert (tmp_path / "run/test_predictions.pt").exists()
    with pytest.raises(FileExistsError):
        runner.train(cfg, tmp_path / "run")


def test_resume_matches_uninterrupted_training_and_rejects_changed_data(graph_file, tmp_path):
    runner = module("runner")
    cfg = config(graph_file, tmp_path)
    runner.train(cfg, tmp_path / "full")
    runner.train(replace(cfg, steps=2), tmp_path / "resumed")
    runner.train(cfg, tmp_path / "resumed", resume=True)
    full = torch.load(tmp_path / "full/model_last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/model_last.pt", weights_only=True)
    for name in full["model_state"]:
        torch.testing.assert_close(full["model_state"][name], resumed["model_state"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="config|identity"):
        runner.train(replace(cfg, train_tasks=5), tmp_path / "resumed", resume=True)


def test_resume_repairs_inference_checkpoint_after_interrupted_save(graph_file, tmp_path):
    runner = module("runner")
    cfg = config(graph_file, tmp_path, steps=2)
    output = tmp_path / "run"
    runner.train(cfg, output)
    # Simulate interruption after model_last was committed but before model_best.
    (output/"model_best.pt").unlink()
    result = runner.train(cfg, output, resume=True)
    assert (output/"model_best.pt").exists()
    best = torch.load(output/"model_best.pt", weights_only=True)
    assert best["step"] == result["evaluated_step"]


def test_resume_off_schedule_endpoint_preserves_checkpoint_selection(graph_file, tmp_path):
    runner = module("runner")
    cfg = replace(config(graph_file, tmp_path, steps=6, learning_rate=.02, train_seed=0),
                  validation_every=6, checkpoint_every=1, test_tasks=1)
    full = runner.train(cfg, tmp_path/"full")
    runner.train(replace(cfg, steps=1), tmp_path/"resumed")
    resumed = runner.train(cfg, tmp_path/"resumed", resume=True)
    assert full["evaluated_step"] == resumed["evaluated_step"]
    assert full["effects"] == resumed["effects"]
    a = torch.load(tmp_path/"full/model_best.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/model_best.pt", weights_only=True)
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name], b["model_state"][name], rtol=0, atol=0)


def test_interrupted_extension_does_not_report_previous_results(graph_file, tmp_path, monkeypatch):
    runner = module("runner")
    cfg = config(graph_file, tmp_path, steps=2)
    output = tmp_path/"run"
    runner.train(cfg, output)
    def interrupted_validation(*args, **kwargs):
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(runner, "validation_nll", interrupted_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train(replace(cfg, steps=4), output, resume=True)
    with pytest.raises(ValueError, match="No completed"):
        module("report").summarize(output)


@pytest.mark.parametrize("override", [{"steps":0}, {"batch_size":0}, {"train_tasks":0},
    {"validation_tasks":0}, {"test_tasks":0}, {"learning_rate":-1}, {"noise_sd":-1},
    {"validation_every":0}, {"checkpoint_every":0}])
def test_invalid_run_config_fails_before_training(graph_file, tmp_path, override):
    with pytest.raises(ValueError):
        cfg = replace(config(graph_file, tmp_path), **override)
        module("runner").train(cfg, tmp_path / "bad")
    assert not (tmp_path / "bad/model_last.pt").exists()


def test_cli_smoke_and_independent_checkpoint_evaluation(graph_file, tmp_path):
    module("runner")
    output = tmp_path / "cli"
    common = [sys.executable, "-m", "scaling_v1"]
    subprocess.run(common + ["smoke", "--graph-file", str(graph_file), "--output", str(output),
        "--cache-dir", str(tmp_path/"cache"), "--steps", "2", "--device", "cpu"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "complete" and report["config"]["steps"] == 2
    subprocess.run(common + ["evaluate", "--checkpoint", str(output/"model_best.pt"),
        "--output", str(tmp_path/"evaluation"), "--cache-dir", str(tmp_path/"cache"),
        "--test-tasks", "2", "--device", "cpu"], cwd=ROOT, check=True, capture_output=True, text=True)
    evaluated = json.loads((tmp_path/"evaluation/evaluation.json").read_text())
    assert evaluated["evaluated_step"] == report["evaluated_step"]
    assert evaluated["effects"] == report["effects"]


def test_five_configuration_sweep_exports_consistent_summary(graph_file, tmp_path):
    module("runner")
    output = tmp_path / "sweep"
    subprocess.run([sys.executable, "-m", "scaling_v1", "sweep", "--graph-file", str(graph_file),
        "--output", str(output), "--cache-dir", str(tmp_path/"cache"), "--steps", "1",
        "--batch-size", "1", "--validation-tasks", "1", "--test-tasks", "1", "--device", "cpu"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    results = [json.loads(path.read_text()) for path in sorted(output.glob("*/results.json"))]
    assert len(results) == 5 and all(r["status"] == "complete" for r in results)
    assert len({r["data_identity"] for r in results}) == 1
    assert {(r["config"]["model_size"], r["config"]["train_tasks"]) for r in results} == {
        ("medium",256), ("medium",1024), ("medium",4096), ("small",4096), ("large",4096)}
    with (output/"summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert {row["effect"] for row in rows} == {"direct","spillover","total"}
