"""Behavioral regressions for the September 2026 code-review fixes."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import pfn_interference.model as demo


ROOT = Path(__file__).resolve().parents[1]


def prepared_csv(tmp_path):
    config = demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')
    args = SimpleNamespace(data_csv=tmp_path/'episodes.csv', csv_datasets=8,
        data_seed=42, validation_fraction=.25, split_seed=7, overwrite_data_csv=False)
    demo.prepare_episode_csv_files(config, data_csv=args.data_csv,
        num_datasets=args.csv_datasets, data_seed=args.data_seed,
        validation_fraction=args.validation_fraction, split_seed=args.split_seed)
    return config, args


def change_label_without_changing_file_size(data_csv):
    train_path, _ = demo.episode_split_paths(data_csv)
    original = train_path.read_bytes()
    lines = original.splitlines(keepends=True)
    headers = lines[0].rstrip(b'\r\n').split(b',')
    fields = lines[1].rstrip(b'\r\n').split(b',')
    index = headers.index(b'direct_effect')
    value = fields[index]
    offset = int(value.startswith(b'-'))
    digit = b'8' if value[offset:offset+1] != b'8' else b'7'
    fields[index] = value[:offset] + digit + value[offset+1:]
    ending = b'\r\n' if lines[1].endswith(b'\r\n') else b'\n'
    lines[1] = b','.join(fields) + ending
    modified = b''.join(lines)
    assert len(original) == len(modified) and original != modified
    train_path.write_bytes(modified)


def test_direct_ddp_loader_rejects_same_size_csv_label_change(tmp_path):
    config, args = prepared_csv(tmp_path)
    change_label_without_changing_file_size(args.data_csv)
    with pytest.raises(ValueError, match='SHA256'):
        demo.load_or_prepare_training_data(args, config, demo.DistributedRuntime(rank=1, world_size=2))


def test_ddp_loader_uses_cpu_control_group_for_verification_collective(tmp_path, monkeypatch):
    config, args = prepared_csv(tmp_path)
    accelerator = torch.device('privateuseone:1')
    monkeypatch.setattr(demo.dist, 'is_initialized', lambda: True)
    def transport(values, src, *, device=None, group=None):
        assert src == 0
        assert device is None and group is None  # default Gloo/CPU group
    monkeypatch.setattr(demo.dist, 'broadcast_object_list', transport)
    loaded = demo.load_or_prepare_training_data(args, config,
        demo.DistributedRuntime(rank=0, world_size=2), device=accelerator)
    assert loaded['train_batch']['tokens'].shape == (6, 24, 5)


@pytest.mark.parametrize('corrupt', [False, True])
def test_ddp_ranks_share_csv_verification_result(tmp_path, monkeypatch, corrupt):
    import pfn_interference.simulation.csv_tasks as adapter
    config, args = prepared_csv(tmp_path)
    if corrupt:
        change_label_without_changing_file_size(args.data_csv)
    # Replace only the process-group transport; validate and load real files.
    mailbox = []
    monkeypatch.setattr(demo.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(demo.dist, 'broadcast_object_list', lambda values, src: mailbox.extend(values))
    if corrupt:
        with pytest.raises(ValueError, match='SHA256'):
            demo.load_or_prepare_training_data(args, config, demo.DistributedRuntime(rank=0, world_size=2))
    else:
        loaded = demo.load_or_prepare_training_data(args, config, demo.DistributedRuntime(rank=0, world_size=2))
        assert loaded['train_batch']['tokens'].shape == (6, 24, 5)
    def receive(values, src):
        values[:] = mailbox
    def should_not_rescan(*args, **kwargs):
        raise AssertionError('Worker must consume rank-zero verification, not rescan the CSV bank.')
    monkeypatch.setattr(demo.dist, 'broadcast_object_list', receive)
    monkeypatch.setattr(adapter, 'validate_prepared_random_csv', should_not_rescan)
    if corrupt:
        with pytest.raises(ValueError, match='SHA256'):
            demo.load_or_prepare_training_data(args, config, demo.DistributedRuntime(rank=1, world_size=2))
    else:
        loaded = demo.load_or_prepare_training_data(args, config, demo.DistributedRuntime(rank=1, world_size=2))
        assert loaded['train_batch']['tokens'].shape == (6, 24, 5)


def test_ddp_rank_zero_logs_finite_validation_diagnostics(tmp_path, monkeypatch, capsys):
    config = demo.DataConfig(n_units=24, graph_family='mixed', dgp_version='random_lpe_v1')
    batch = demo.generate_batch(config, 2, seed=4, device=torch.device('cpu'))
    # Real model, optimizer, validation and logging; only collective transport
    # is replaced so this regression can run without opening network sockets.
    monkeypatch.setattr(demo, 'DistributedDataParallel', lambda model, **kwargs: model)
    monkeypatch.setattr(demo.dist, 'all_reduce', lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, 'broadcast', lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, 'barrier', lambda *args, **kwargs: None)
    demo.train(data_config=config,
        model_config=demo.ModelConfig(d_model=16, num_layers=1, ffn_dim=32),
        train_config=demo.TrainConfig(epochs=1, batch_size=1),
        train_batch=batch, validation_batch=batch, device=torch.device('cpu'),
        checkpoint_path=tmp_path/'model.pt', runtime=demo.DistributedRuntime(rank=0, world_size=2))
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('epoch='))
    metrics = dict(item.split('=', 1) for item in line.split())
    for effect in ('direct','spillover','total'):
        assert math.isfinite(float(metrics[f'validation_{effect}_rmse']))
    for arm in ('mu00','mu10','mu11'):
        assert math.isfinite(float(metrics[f'validation_{arm}_nll']))








def test_rank0_training_diagnostic_is_named_as_local_shard(tmp_path, monkeypatch, capsys):
    config = demo.DataConfig(n_units=24, graph_family="mixed", dgp_version="random_lpe_v1")
    batch = demo.generate_batch(config, 2, seed=31, device=torch.device("cpu"))
    monkeypatch.setattr(demo, "DistributedDataParallel", lambda model, **kwargs: model)
    monkeypatch.setattr(demo.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, "barrier", lambda *args, **kwargs: None)
    demo.train(
        data_config=config,
        model_config=demo.ModelConfig(d_model=16, num_layers=1, ffn_dim=32),
        train_config=demo.TrainConfig(epochs=1, batch_size=1),
        train_batch=batch,
        validation_batch=batch,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "diag.pt",
        runtime=demo.DistributedRuntime(rank=0, world_size=2),
        rank_sharded_train=True,
    )
    output = capsys.readouterr().out
    assert "rank0_training_label_diagnostic=" in output
    assert "\ntraining_label_diagnostic=" not in "\n" + output
