"""Regression tests for scalable rank-local CSV loading."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_local_network_interference as demo
import csv_random_dgp as random_csv


def cfg() -> demo.DataConfig:
    return demo.DataConfig(n_units=24, graph_family="mixed", dgp_version="random_lpe_v1")


def prepared_bank(tmp_path: Path):
    data_csv = tmp_path / "episodes.csv"
    demo.prepare_episode_csv_files(
        cfg(),
        data_csv=data_csv,
        num_datasets=8,
        data_seed=42,
        validation_fraction=0.25,
        split_seed=7,
        generation_batch_size=4,
    )
    return data_csv


def csv_dataset_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["dataset_id"]) for row in csv.DictReader(handle)}


def test_prepare_ddp_train_shards_preserves_split_and_balances_ranks(tmp_path):
    data_csv = prepared_bank(tmp_path)
    shard_dir = tmp_path / "shards"

    metadata = demo.prepare_ddp_train_shards(
        data_csv,
        shard_dir=shard_dir,
        world_size=2,
    )

    manifest = json.loads(Path(str(data_csv) + ".manifest.json").read_text())
    expected_train = set(manifest["training_dataset_ids"])
    expected_validation = set(manifest["validation_dataset_ids"])
    rank0 = csv_dataset_ids(shard_dir / "train_rank0.csv")
    rank1 = csv_dataset_ids(shard_dir / "train_rank1.csv")

    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == expected_train
    assert len(rank0) == len(rank1) == 3
    assert metadata["training_datasets"] == 6
    assert metadata["validation_datasets"] == 2
    assert metadata["train_datasets_per_rank"] == 3
    assert csv_dataset_ids(demo.episode_split_paths(data_csv)[1]) == expected_validation


def test_rank_sharded_loader_loads_only_local_train_and_only_rank0_validation(tmp_path):
    data_csv = prepared_bank(tmp_path)
    shard_dir = tmp_path / "shards"
    demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)
    args = SimpleNamespace(
        data_csv=data_csv,
        csv_datasets=8,
        data_seed=42,
        validation_fraction=0.25,
        split_seed=7,
        overwrite_data_csv=False,
        ddp_shard_dir=shard_dir,
    )

    rank0 = demo.load_or_prepare_training_data(
        args,
        cfg(),
        demo.DistributedRuntime(rank=0, world_size=2),
    )
    rank1 = demo.load_or_prepare_training_data(
        args,
        cfg(),
        demo.DistributedRuntime(rank=1, world_size=2),
    )

    assert rank0["rank_sharded_train"] is True
    assert rank1["rank_sharded_train"] is True
    assert rank0["train_batch"]["tokens"].shape[0] == 3
    assert rank1["train_batch"]["tokens"].shape[0] == 3
    assert rank0["validation_batch"]["tokens"].shape[0] == 2
    assert rank1["validation_batch"]["tokens"].shape[0] == 0
    assert set(rank0["train_dataset_ids"]).isdisjoint(rank1["train_dataset_ids"])
    assert set(rank0["train_dataset_ids"]) | set(rank1["train_dataset_ids"]) == csv_dataset_ids(
        demo.episode_split_paths(data_csv)[0]
    )


def test_rank_sharded_train_epoch_visits_every_local_episode(monkeypatch):
    episode_count = 6
    tokens = torch.zeros(episode_count, 2, demo.TOKEN_DIM)
    tokens[:, 0, 0] = torch.arange(episode_count, dtype=torch.float32)
    train_batch = {
        "tokens": tokens,
        "oracle_ite": torch.zeros(episode_count, 2, 3),
    }
    visited: list[int] = []

    def fake_step(model, optimizer, batch):
        visited.extend(int(value) for value in batch["tokens"][:, 0, 0].tolist())
        metrics = {"total_loss": 1.0}
        for effect in ("direct", "spillover", "total"):
            metrics.update({effect + "_effect_mse": 0., effect + "_effect_mae": 0.})
        return metrics

    monkeypatch.setattr(demo, "train_step_metrics", fake_step)
    monkeypatch.setattr(demo.dist, "all_reduce", lambda *args, **kwargs: None)

    demo.train_epoch(
        object(),
        object(),
        train_batch,
        batch_size=2,
        training_seed=0,
        epoch=1,
        device=torch.device("cpu"),
        runtime=demo.DistributedRuntime(rank=1, world_size=2),
        data_is_rank_sharded=True,
    )

    assert sorted(visited) == list(range(episode_count))


def test_chunked_neighbor_covariate_score_matches_dense():
    adjacency = torch.tensor(
        [
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
        ],
        dtype=torch.uint8,
    )
    x = torch.tensor([[0.2, -0.3, 1.0], [0.5, 0.1, -0.4], [-1.0, 0.4, 0.8]])
    score = torch.sigmoid(x)
    degree = adjacency.sum(-1).to(torch.float32)
    dense = torch.bmm(adjacency.to(torch.float32), score.unsqueeze(-1)).squeeze(-1) / degree

    chunked = demo.compute_neighbor_covariate_score_chunked(
        adjacency,
        score,
        degree,
        chunk_size=2,
    )

    torch.testing.assert_close(chunked, dense)




def test_prepare_ddp_train_shards_can_rebuild_from_master_without_split_files(tmp_path):
    data_csv = prepared_bank(tmp_path)
    train_path, validation_path = demo.episode_split_paths(data_csv)
    train_path.unlink()
    validation_path.unlink()
    shard_dir = tmp_path / "shards_from_master"

    metadata = demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)

    assert metadata["training_datasets"] == 6
    assert metadata["validation_datasets"] == 2
    assert (shard_dir / "validation.csv").is_file()
    assert len(csv_dataset_ids(shard_dir / "validation.csv")) == 2
    assert len(csv_dataset_ids(shard_dir / "train_rank0.csv")) == 3
    assert len(csv_dataset_ids(shard_dir / "train_rank1.csv")) == 3


def test_rank_sharded_random_loader_needs_master_manifest_not_derived_splits(tmp_path):
    data_csv = prepared_bank(tmp_path)
    shard_dir = tmp_path / "shards"
    demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)
    train_path, validation_path = demo.episode_split_paths(data_csv)
    train_path.unlink()
    validation_path.unlink()
    args = SimpleNamespace(
        data_csv=data_csv,
        csv_datasets=8,
        data_seed=42,
        validation_fraction=0.25,
        split_seed=7,
        overwrite_data_csv=False,
        ddp_shard_dir=shard_dir,
    )

    rank0 = demo.load_or_prepare_training_data(
        args,
        cfg(),
        demo.DistributedRuntime(rank=0, world_size=2),
    )

    assert rank0["train_batch"]["tokens"].shape[0] == 3
    assert rank0["validation_batch"]["tokens"].shape[0] == 2


def test_nonmain_rank_accepts_empty_validation_when_train_is_physically_sharded(tmp_path, monkeypatch):
    config = cfg()
    batch = demo.generate_batch(config, 2, seed=5, device=torch.device("cpu"))
    empty_validation = {"tokens": torch.empty(0, config.n_units, demo.TOKEN_DIM)}
    monkeypatch.setattr(demo, "DistributedDataParallel", lambda model, **kwargs: model)
    monkeypatch.setattr(demo.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo.dist, "barrier", lambda *args, **kwargs: None)

    demo.train(
        data_config=config,
        model_config=demo.ModelConfig(d_model=16, num_layers=1, ffn_dim=32),
        train_config=demo.TrainConfig(epochs=1, batch_size=1),
        train_batch=batch,
        validation_batch=empty_validation,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "unused.pt",
        runtime=demo.DistributedRuntime(rank=1, world_size=2),
        rank_sharded_train=True,
    )




def test_large_task_bank_keeps_extended_graph_resample_budget():
    assert demo.ER_MAX_RESAMPLE_ATTEMPTS >= 10_000


def test_known_count_csv_loader_matches_legacy_loader(tmp_path):
    batch = demo.generate_batch(cfg(), 3, seed=19, device=torch.device("cpu"))
    path = tmp_path / "known_count.csv"
    demo.save_episode_batch_csv(batch, path)

    legacy, legacy_ids = demo.load_episode_batch_csv(path, device=torch.device("cpu"))
    known, known_ids = demo.load_episode_batch_csv(
        path,
        device=torch.device("cpu"),
        expected_dataset_count=3,
    )

    assert known_ids == legacy_ids
    assert known.keys() == legacy.keys()
    for key in known:
        assert torch.equal(known[key], legacy[key]), key


def test_known_count_csv_loader_rejects_wrong_episode_count(tmp_path):
    batch = demo.generate_batch(cfg(), 2, seed=23, device=torch.device("cpu"))
    path = tmp_path / "wrong_count.csv"
    demo.save_episode_batch_csv(batch, path)
    with pytest.raises(ValueError, match="expected_dataset_count|episode count"):
        demo.load_episode_batch_csv(
            path,
            device=torch.device("cpu"),
            expected_dataset_count=3,
        )



def _rewrite_first_data_row(path: Path, *, field: str, value: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rank_sharded_loader_rejects_mutated_label_content(tmp_path):
    data_csv = prepared_bank(tmp_path)
    shard_dir = tmp_path / "shards"
    demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)
    rank0_path = shard_dir / "train_rank0.csv"
    _rewrite_first_data_row(rank0_path, field="direct_effect", value="123456.0")

    with pytest.raises(ValueError, match="integrity|SHA256|sha256"):
        demo.load_rank_sharded_episode_csv_splits(
            data_csv,
            shard_dir=shard_dir,
            runtime=demo.DistributedRuntime(rank=0, world_size=2),
        )


def test_rank_sharded_loader_rejects_duplicate_rank_file(tmp_path):
    data_csv = prepared_bank(tmp_path)
    shard_dir = tmp_path / "shards"
    demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)
    shutil.copyfile(shard_dir / "train_rank1.csv", shard_dir / "train_rank0.csv")

    with pytest.raises(ValueError, match="integrity|SHA256|sha256"):
        demo.load_rank_sharded_episode_csv_splits(
            data_csv,
            shard_dir=shard_dir,
            runtime=demo.DistributedRuntime(rank=0, world_size=2),
        )


def test_stratified_split_keeps_exact_global_validation_count(tmp_path):
    source = tmp_path / "split_source.csv"
    counts = [9, 9, 9, 13]  # old per-group round() gives 9 validation; global target is 8
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "graph_type"])
        writer.writeheader()
        dataset_id = 0
        for graph_type, count in enumerate(counts):
            for _ in range(count):
                writer.writerow({"dataset_id": dataset_id, "graph_type": graph_type})
                dataset_id += 1
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"

    summary = demo.split_episode_csv(
        source,
        train,
        validation,
        validation_fraction=0.2,
        split_seed=42,
        stratify_by_graph_type=True,
    )

    assert summary == {
        "total_datasets": 40,
        "training_datasets": 32,
        "validation_datasets": 8,
    }
    assert summary["training_datasets"] % 8 == 0


def test_existing_random_master_can_be_resplit_without_regeneration(tmp_path):
    data_csv = tmp_path / "resplit.csv"
    config = cfg()
    demo.prepare_episode_csv_files(
        config,
        data_csv=data_csv,
        num_datasets=16,
        data_seed=42,
        validation_fraction=0.25,
        split_seed=7,
        generation_batch_size=4,
    )
    before_hash = random_csv.file_sha256(data_csv)
    old_manifest = json.loads(Path(str(data_csv) + ".manifest.json").read_text())

    updated = random_csv.ensure_prepared_random_master_split(
        config,
        data_csv=data_csv,
        num_datasets=16,
        data_seed=42,
        validation_fraction=0.125,
        split_seed=7,
        verify_hashes=True,
    )

    assert random_csv.file_sha256(data_csv) == before_hash
    assert updated["summary"]["validation_datasets"] == 2
    assert updated["summary"]["training_datasets"] == 14
    assert updated["identity"]["validation_fraction"] == 0.125
    assert old_manifest["files"]["master"]["sha256"] == updated["files"]["master"]["sha256"]





def test_random_master_resplit_uses_manifest_task_metadata_not_full_csv_scan(tmp_path, monkeypatch):
    data_csv = tmp_path / "metadata_resplit.csv"
    config = cfg()
    demo.prepare_episode_csv_files(
        config,
        data_csv=data_csv,
        num_datasets=16,
        data_seed=43,
        validation_fraction=0.25,
        split_seed=7,
        generation_batch_size=4,
    )

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("resplitting an existing random bank must not rescan the giant master CSV")

    monkeypatch.setattr(demo, "select_episode_split_ids", forbidden_scan)
    updated = random_csv.ensure_prepared_random_master_split(
        config,
        data_csv=data_csv,
        num_datasets=16,
        data_seed=43,
        validation_fraction=0.125,
        split_seed=9,
        verify_hashes=False,
    )
    assert updated["summary"]["validation_datasets"] == 2

def test_validation_dataset_id_digest_is_order_insensitive_for_legacy_split(tmp_path):
    data_csv = prepared_bank(tmp_path)
    manifest_path = Path(str(data_csv) + ".manifest.json")
    manifest_path.unlink()
    _, validation_path = demo.episode_split_paths(data_csv)

    with validation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    blocks = {}
    order = []
    for row in rows:
        dataset_id = int(row["dataset_id"])
        if dataset_id not in blocks:
            blocks[dataset_id] = []
            order.append(dataset_id)
        blocks[dataset_id].append(row)
    assert len(order) >= 2
    reordered = []
    for dataset_id in reversed(order):
        reordered.extend(blocks[dataset_id])
    with validation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reordered)

    shard_dir = tmp_path / "legacy_order_shards"
    demo.prepare_ddp_train_shards(data_csv, shard_dir=shard_dir, world_size=2)
    loaded = demo.load_rank_sharded_episode_csv_splits(
        data_csv,
        shard_dir=shard_dir,
        runtime=demo.DistributedRuntime(rank=0, world_size=2),
    )

    assert set(loaded["validation_dataset_ids"]) == set(order)
