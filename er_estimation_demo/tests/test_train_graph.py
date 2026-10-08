from pathlib import Path
import argparse
import csv
import json
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_local_network_interference as demo  # noqa: E402

from train_local_network_interference import (  # noqa: E402
    ER_GRAPH,
    CONFIGURATION_GRAPH,
    QUERY_DIRECT,
    QUERY_SPILLOVER,
    QUERY_TOTAL,
    DataConfig,
    LocalNetworkQueryTransformer,
    ModelConfig,
    TrainConfig,
    apply_algorithm_2,
    checkpoint_paths,
    compute_query_metrics,
    compute_training_losses,
    diagnose_query_effect_support,
    generate_batch,
    generate_graphs,
    load_checkpoint,
    load_episode_batch_csv,
    prepare_episode_csv_splits,
    save_algorithm_2_report,
    save_checkpoint,
    save_episode_batch_csv,
    select_episode_batch,
    split_episode_csv,
    episode_split_paths,
    train_step,
)


def small_model_config(**overrides):
    values = dict(
        input_dim=5,
        query_dim=2,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=32,
        dropout=0.0,
        gmm_n_components=5,
        gmm_min_sigma=1e-3,
        gmm_pi_temperature=1.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_process_group_init_kwargs_bind_accelerator_device() -> None:
    runtime = demo.DistributedRuntime(rank=3, world_size=8)
    device = torch.device("privateuseone:3")

    kwargs = demo.process_group_init_kwargs(runtime, device)

    assert kwargs["backend"] == "mccl"
    assert kwargs["rank"] == 3
    assert kwargs["world_size"] == 8
    assert kwargs["device_id"] == device


def test_distributed_training_requires_prepared_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        data_csv=tmp_path / "episodes.csv",
        csv_datasets=8,
        data_seed=12345,
        validation_fraction=0.2,
        split_seed=0,
        overwrite_data_csv=True,
    )
    runtime = demo.DistributedRuntime(rank=0, world_size=8)

    with pytest.raises(RuntimeError, match="prepare-data-only"):
        demo.load_or_prepare_training_data(
            args,
            DataConfig(n_units=10, er_edge_probability=0.31),
            runtime,
        )


def test_prepare_data_only_writes_splits_without_loading_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_csv = tmp_path / "episodes.csv"

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("prepare-only must not load split tensors")

    monkeypatch.setattr(demo, "load_episode_batch_csv", fail_if_loaded)
    prepared = demo.prepare_episode_csv_files(
        DataConfig(n_units=10, er_edge_probability=0.31),
        data_csv=data_csv,
        num_datasets=8,
        data_seed=12345,
        validation_fraction=0.25,
        split_seed=0,
        overwrite=True,
    )

    assert Path(prepared["data_csv"]).is_file()
    assert Path(prepared["train_path"]).is_file()
    assert Path(prepared["validation_path"]).is_file()
    assert prepared["summary"]["total_datasets"] == 8



def test_prepare_data_generates_large_csv_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = demo.generate_batch
    calls: list[int] = []

    def record_generate(config, batch_size, **kwargs):
        calls.append(batch_size)
        return original(config, batch_size, **kwargs)

    monkeypatch.setattr(demo, "generate_batch", record_generate)
    data_csv = tmp_path / "chunked.csv"
    prepared = demo.prepare_episode_csv_files(
        DataConfig(n_units=10, er_edge_probability=0.31),
        data_csv=data_csv,
        num_datasets=9,
        data_seed=321,
        validation_fraction=0.2,
        split_seed=0,
        overwrite=True,
        generation_batch_size=4,
    )

    assert calls == [4, 4, 1]
    restored, dataset_ids = load_episode_batch_csv(
        Path(prepared["data_csv"]), device=torch.device("cpu")
    )
    assert dataset_ids == list(range(9))
    assert restored["tokens"].shape == (9, 10, 5)


def test_csv_schema_version_rejects_stale_files(tmp_path: Path) -> None:
    assert demo.CSV_SCHEMA_VERSION == "majority_arm_v12_noise_free_cepo"
    source = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=2,
        seed=404,
        device=torch.device("cpu"),
    )
    csv_path = tmp_path / "current.csv"
    save_episode_batch_csv(source, csv_path)
    assert demo.csv_schema_matches(csv_path, expected_n_units=10)
    assert not demo.csv_schema_matches(csv_path, expected_n_units=300)

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        current_rows = list(csv.DictReader(handle))
    assert current_rows
    assert "counterfactual_neighbor_bits" not in current_rows[0]
    assert "majority_arm" in current_rows[0]

    rows[1] = rows[1].replace(demo.CSV_SCHEMA_VERSION, "rooted_reference_cf_p002_v4", 1)
    stale_path = tmp_path / "stale.csv"
    stale_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert not demo.csv_schema_matches(stale_path, expected_n_units=10)
    with pytest.raises(ValueError, match="stale"):
        load_episode_batch_csv(stale_path, device=torch.device("cpu"))

def test_every_generated_graph_is_valid_reproducible_er() -> None:
    config = DataConfig(n_units=10, er_edge_probability=0.31)
    first = generate_graphs(
        config,
        2048,
        generator=torch.Generator().manual_seed(11),
    )
    second = generate_graphs(
        config,
        2048,
        generator=torch.Generator().manual_seed(11),
    )

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    adjacency, graph_type, star_center = first

    assert adjacency.shape == (2048, 10, 10)
    assert torch.all(graph_type == ER_GRAPH)
    assert torch.all(star_center == -1)
    assert torch.equal(adjacency, adjacency.transpose(1, 2))
    assert torch.equal(
        torch.diagonal(adjacency, dim1=1, dim2=2),
        torch.zeros(2048, 10),
    )
    assert torch.all((adjacency == 0) | (adjacency == 1))
    assert torch.all(adjacency.sum(dim=-1) >= 1)
    assert 2.8 < float(adjacency.sum(dim=-1).mean()) < 3.2


def test_default_n1000_p002_graph_has_about_twenty_neighbors() -> None:
    adjacency, _, _ = generate_graphs(
        DataConfig(),
        32,
        generator=torch.Generator().manual_seed(111),
    )

    realized_average_degree = float(adjacency.sum(dim=-1).mean())
    assert 19.7 < realized_average_degree < 20.3


def test_data_config_exposes_four_graph_family_prior() -> None:
    config = DataConfig()

    assert config.graph_family == "er"
    assert config.er_edge_probability == pytest.approx(0.02)
    assert config.sbm_between_probability == pytest.approx(0.006)
    assert config.configuration_degree_sigma == pytest.approx(1.0)
    assert config.configuration_max_degree == 25
    assert config.rgg_torus is True
    with pytest.raises(ValueError, match="graph_family"):
        DataConfig(graph_family="unknown")
    with pytest.raises(ValueError, match="er_edge_probability"):
        DataConfig(er_edge_probability=1.0)



def test_mixed_graph_prior_samples_degree_before_random_family_reproducibly() -> None:
    config = DataConfig(n_units=80, er_edge_probability=6.0 / 79.0, graph_family="mixed")
    batch_size = 12
    seed = 2026

    expected_rng = torch.Generator().manual_seed(seed)
    expected_target_degree = 4.0 + 6.0 * torch.rand(
        batch_size, generator=expected_rng
    )
    family_index = torch.randint(
        0, len(demo.GRAPH_FAMILY_ORDER), (batch_size,), generator=expected_rng
    )
    family_lookup = torch.tensor(demo.GRAPH_FAMILY_ORDER, dtype=torch.long)
    expected_graph_type = family_lookup[family_index]

    first = generate_graphs(
        config,
        batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    second = generate_graphs(
        config,
        batch_size,
        generator=torch.Generator().manual_seed(seed),
    )

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    adjacency, graph_type, star_center = first
    assert torch.equal(graph_type, expected_graph_type)
    assert torch.all(star_center == -1)
    assert torch.equal(adjacency, adjacency.transpose(1, 2))
    assert torch.count_nonzero(torch.diagonal(adjacency, dim1=1, dim2=2)) == 0
    assert torch.all(adjacency.sum(dim=-1) >= 1)

    realized_degree = adjacency.sum(dim=-1).mean(dim=-1)
    assert torch.all(torch.abs(realized_degree - expected_target_degree) < 2.0), (
        expected_target_degree,
        realized_degree,
    )


def test_mixed_configuration_graphs_use_each_episode_target_degree() -> None:
    config = DataConfig(n_units=80, er_edge_probability=6.0 / 79.0, graph_family="mixed")
    batch_size = 6
    seed = 31415
    requested = torch.full(
        (batch_size,), demo.CONFIGURATION_GRAPH, dtype=torch.long
    )

    expected_rng = torch.Generator().manual_seed(seed)
    expected_target_degree = 4.0 + 6.0 * torch.rand(
        batch_size, generator=expected_rng
    )

    adjacency, graph_type, _ = generate_graphs(
        config,
        batch_size,
        generator=torch.Generator().manual_seed(seed),
        graph_types=requested,
    )

    assert torch.equal(graph_type, requested)
    realized_degree = adjacency.sum(dim=-1).mean(dim=-1)
    assert torch.all(torch.abs(realized_degree - expected_target_degree) <= 0.05), (
        expected_target_degree,
        realized_degree,
    )


def test_generate_graphs_accepts_explicit_family_schedule() -> None:
    config = DataConfig(n_units=40, er_edge_probability=6.0 / 39.0, graph_family="mixed")
    requested = torch.tensor(
        [
            demo.ER_GRAPH,
            demo.CONFIGURATION_GRAPH,
            demo.RGG_GRAPH,
            demo.SBM_GRAPH,
        ],
        dtype=torch.long,
    )

    adjacency, graph_type, _ = generate_graphs(
        config,
        4,
        generator=torch.Generator().manual_seed(9),
        graph_types=requested,
    )

    assert torch.equal(graph_type, requested)
    assert torch.all(adjacency.sum(dim=-1) >= 1)


def test_prepare_csv_leaves_mixed_family_sampling_to_generate_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = demo.generate_batch
    graph_type_arguments: list[object] = []

    def record_generate(config, batch_size, **kwargs):
        graph_type_arguments.append(kwargs.get("graph_types"))
        return original(config, batch_size, **kwargs)

    monkeypatch.setattr(demo, "generate_batch", record_generate)
    data_csv = tmp_path / "four_families.csv"
    prepared = demo.prepare_episode_csv_files(
        DataConfig(n_units=40, er_edge_probability=6.0 / 39.0, graph_family="mixed"),
        data_csv=data_csv,
        num_datasets=20,
        data_seed=12345,
        validation_fraction=0.2,
        split_seed=7,
        overwrite=True,
        generation_batch_size=6,
    )

    assert graph_type_arguments == [None, None, None, None]
    master, _ = load_episode_batch_csv(
        Path(prepared["data_csv"]), device=torch.device("cpu")
    )
    assert set(master["graph_type"].tolist()).issubset(set(demo.GRAPH_FAMILY_ORDER))
    assert prepared["summary"]["total_datasets"] == 20
    assert (
        prepared["summary"]["training_datasets"]
        + prepared["summary"]["validation_datasets"]
        == 20
    )
    assert abs(prepared["summary"]["validation_datasets"] - 4) <= 1

def test_er_resampling_stops_after_maximum_attempts() -> None:
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        demo._generate_er_adjacency(
            n_units=10,
            batch_size=4,
            edge_probability=1e-12,
            generator=torch.Generator().manual_seed(12),
            max_attempts=2,
        )


def test_er_resampling_returns_when_remaining_rows_succeed_on_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter(
        [
            torch.stack([torch.zeros(45), torch.ones(45)]),
            torch.zeros(1, 45),
        ]
    )

    def controlled_rand(*shape, generator):
        sample = next(samples)
        assert tuple(sample.shape) == shape
        return sample

    monkeypatch.setattr(torch, "rand", controlled_rand)
    adjacency = demo._generate_er_adjacency(
        n_units=10,
        batch_size=2,
        edge_probability=0.5,
        generator=torch.Generator().manual_seed(13),
        max_attempts=2,
    )

    expected = (torch.ones(10, 10) - torch.eye(10)).expand(2, -1, -1)
    assert torch.equal(adjacency, expected)


def test_batch_contains_scalar_x_and_majority_arm_three_query_types() -> None:
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31, noise_sd=1.0),
        batch_size=32,
        seed=22,
        device=torch.device("cpu"),
    )

    assert batch["x"].shape == (32, 10)
    assert batch["tokens"].shape == (32, 10, 5)
    assert batch["queries"].shape == (32, 10, 3, 4)
    assert batch["query_effect"].shape == (32, 10, 3)
    assert batch["oracle_ite"].shape == (32, 10, 3)
    assert batch["oracle_ate"].shape == (32, 3)
    assert torch.equal(batch["query_effect"], batch["outcome_a"] - batch["outcome_b"])

    assert "counterfactual_exposure" not in batch
    assert "counterfactual_neighbor_treatment" not in batch
    assert "reference_is_low" not in batch

    queries = batch["queries"]
    low_exposure = batch["low_sampled_exposure"]
    high_exposure = batch["high_sampled_exposure"]
    degree = batch["degree"]
    low_count = torch.round(low_exposure * degree).to(torch.int64)
    high_count = torch.round(high_exposure * degree).to(torch.int64)
    threshold = torch.floor(degree / 2).to(torch.int64)

    assert bool((low_count <= threshold).all())
    assert bool((high_count > threshold).all())
    assert torch.equal(queries[..., QUERY_DIRECT, 1], low_exposure)
    assert torch.equal(queries[..., QUERY_DIRECT, 3], low_exposure)
    assert torch.equal(queries[..., QUERY_SPILLOVER, 1], high_exposure)
    assert torch.equal(queries[..., QUERY_SPILLOVER, 3], low_exposure)
    assert torch.equal(queries[..., QUERY_TOTAL, 1], high_exposure)
    assert torch.equal(queries[..., QUERY_TOTAL, 3], low_exposure)

    observed_counts = torch.round(batch["observed_exposure"] * degree).to(torch.int64)
    expected_arm = (observed_counts > threshold).to(torch.int64)
    assert torch.equal(batch["majority_arm"], expected_arm)
    torch.testing.assert_close(batch["oracle_ate"], batch["oracle_ite"].mean(dim=1))

def test_majority_query_sampler_uses_low_and_high_exact_support() -> None:
    config = DataConfig(n_units=10, er_edge_probability=0.31)
    batch = generate_batch(
        config,
        batch_size=16,
        seed=23,
        device=torch.device("cpu"),
    )

    degree = batch["degree"]
    low = batch["low_sampled_exposure"]
    high = batch["high_sampled_exposure"]
    low_count = torch.round(low * degree).to(torch.int64)
    high_count = torch.round(high * degree).to(torch.int64)
    threshold = torch.floor(degree / 2).to(torch.int64)

    assert bool((low_count <= threshold).all())
    assert bool((high_count > threshold).all())
    assert torch.all(high > low)
    assert torch.allclose(low * degree, low_count.to(torch.float32))
    assert torch.allclose(high * degree, high_count.to(torch.float32))


def test_radius_one_rooted_configuration_recovers_exposure() -> None:
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=1,
        seed=24,
        device=torch.device("cpu"),
    )
    root = 8
    configuration = demo.extract_radius_one_rooted_configuration(
        adjacency=batch["adjacency"][0],
        x=batch["x"][0],
        neighbor_treatment=(
            batch["adjacency"][0, root] * batch["observed_treatment"][0]
        ),
        root_index=root,
    )

    assert configuration.root_index == root
    assert int(configuration.node_indices[0].item()) == root
    assert configuration.root_indicator[0].item() == 1.0
    assert configuration.root_indicator[1:].sum().item() == 0.0
    expected_nodes = torch.cat(
        [
            torch.tensor([root]),
            torch.nonzero(batch["adjacency"][0, root] > 0, as_tuple=False)
            .flatten()
            .sort()
            .values,
        ]
    )
    assert torch.equal(configuration.node_indices, expected_nodes)
    assert torch.equal(
        configuration.adjacency,
        batch["adjacency"][0][expected_nodes][:, expected_nodes],
    )
    assert torch.equal(configuration.x, batch["x"][0, expected_nodes])
    assert configuration.treatment_marks[0].item() == 0.0
    assert configuration.exposure == pytest.approx(
        float(batch["observed_exposure"][0, root].item())
    )


def test_tnet_linear_dgp_uses_own_and_neighbor_covariate_scores() -> None:
    batch = generate_batch(
        DataConfig(
            n_units=10,
            er_edge_probability=0.31,
            dgp_version="tnet_linear_neighbor_covariate",
            x_outcome_coefficient=0.0,
            baseline_sd=0.0,
            unit_sd=0.0,
            own_score_baseline_coefficient=1.0,
            neighbor_score_baseline_coefficient=0.5,
            tau_mean=0.4,
            tau_episode_sd=0.0,
            tau_unit_sd=0.0,
            tau_own_score_slope=0.2,
            tau_neighbor_score_slope=0.2,
            gamma_mean=0.2,
            gamma_sd=0.0,
            gamma_own_score_slope=0.1,
            gamma_neighbor_score_slope=0.2,
            eta_mean=0.0,
            eta_sd=0.0,
            noise_sd=0.0,
        ),
        batch_size=16,
        seed=33,
        device=torch.device("cpu"),
    )

    x = batch["x"]
    adjacency = batch["adjacency"].to(torch.float32)
    degree = batch["degree"]
    treatment = batch["observed_treatment"]
    exposure = batch["observed_exposure"]

    own_score = torch.sigmoid(x)
    neighbor_score = torch.bmm(
        adjacency, own_score.unsqueeze(-1)
    ).squeeze(-1) / degree
    own_centered = own_score - 0.5
    neighbor_centered = neighbor_score - 0.5
    expected_baseline = own_score + 0.5 * neighbor_score
    expected_tau = 0.4 + 0.2 * own_centered + 0.2 * neighbor_centered
    expected_gamma = 0.2 + 0.1 * own_centered + 0.2 * neighbor_centered

    assert torch.allclose(batch["covariate_score"], own_score)
    assert torch.allclose(batch["neighbor_covariate_score"], neighbor_score)
    assert torch.allclose(batch["structural_baseline"], expected_baseline)
    assert torch.allclose(batch["tau"], expected_tau)
    assert torch.allclose(batch["gamma"], expected_gamma)
    assert torch.equal(batch["eta"], torch.zeros_like(batch["eta"]))
    assert torch.allclose(
        batch["y_obs"],
        expected_baseline + expected_tau * treatment + expected_gamma * exposure,
    )

    exposure_a = batch["queries"][..., QUERY_SPILLOVER, 1]
    exposure_b = batch["queries"][..., QUERY_SPILLOVER, 3]
    exposure_gap = exposure_a - exposure_b
    effects = batch["query_effect"]
    assert torch.allclose(effects[..., QUERY_DIRECT], expected_tau)
    assert torch.allclose(
        effects[..., QUERY_SPILLOVER], expected_gamma * exposure_gap
    )
    assert torch.allclose(
        effects[..., QUERY_TOTAL], expected_tau + expected_gamma * exposure_gap
    )


def test_default_dgp_matches_gao_ding_design_2_outcome_equation() -> None:
    config = DataConfig(n_units=10, er_edge_probability=0.31, noise_sd=0.0)
    batch = generate_batch(
        config,
        batch_size=128,
        seed=2033,
        device=torch.device("cpu"),
    )

    x = batch["x"]
    adjacency = batch["adjacency"].to(torch.float32)
    degree = batch["degree"]
    treatment = batch["observed_treatment"]
    exposure = batch["observed_exposure"]
    neighbor_x = torch.bmm(adjacency, x.unsqueeze(-1)).squeeze(-1) / degree

    expected_tau = 6.0 + 0.2 * torch.exp(x)
    expected_baseline_without_epsilon = 1.0 - x - 3.0 * neighbor_x
    torch.testing.assert_close(batch["structural_baseline"], expected_baseline_without_epsilon)
    epsilon = batch["y_obs"] - (expected_baseline_without_epsilon
        + expected_tau * treatment - .9 * exposure)
    expected_y = (
        1.0
        - 0.9 * exposure
        + 6.0 * treatment
        - x
        + 0.2 * treatment * torch.exp(x)
        - 3.0 * neighbor_x
        + epsilon
    )

    assert config.dgp_version == "gao_ding_design2_outcome"
    assert torch.allclose(batch["tau"], expected_tau)
    assert torch.equal(batch["gamma"], torch.full_like(x, -0.9))
    assert torch.equal(batch["eta"], torch.zeros_like(x))
    assert float(epsilon.mean()) == pytest.approx(0.0, abs=0.3)
    assert float(epsilon.std()) == pytest.approx(4.0, abs=0.3)
    assert torch.allclose(batch["y_obs"], expected_y, atol=2e-6, rtol=1e-5)


def test_tnet_dgp_effect_heterogeneity_tracks_network_covariates() -> None:
    batch = generate_batch(
        DataConfig(
            n_units=10,
            er_edge_probability=0.31,
            dgp_version="tnet_linear_neighbor_covariate",
            baseline_sd=0.0,
            unit_sd=0.0,
            noise_sd=0.0,
        ),
        batch_size=32,
        seed=34,
        device=torch.device("cpu"),
    )

    assert batch["covariate_score"].shape == (32, 10)
    assert batch["neighbor_covariate_score"].shape == (32, 10)
    assert batch["tau"].shape == (32, 10)
    assert batch["gamma"].shape == (32, 10)
    assert torch.std(batch["tau"]) > 0
    assert torch.std(batch["gamma"]) > 0
    assert torch.equal(batch["eta"], torch.zeros_like(batch["eta"]))


def test_old_checkpoint_data_config_mapping_uses_legacy_dgp() -> None:
    legacy = demo.data_config_from_mapping(
        {
            "n_units": 10,
            "er_edge_probability": 0.31,
            "treatment_prob": 0.5,
            "x_sd": 1.0,
            "x_outcome_coefficient": 0.5,
            "baseline_sd": 1.0,
            "unit_sd": 0.2,
            "tau_mean": 0.5,
            "tau_episode_sd": 0.35,
            "tau_unit_sd": 0.0,
            "gamma_mean": 0.3,
            "gamma_sd": 0.15,
            "eta_mean": 0.0,
            "eta_sd": 0.0,
            "noise_sd": 0.0,
        }
    )

    assert legacy.dgp_version == "legacy_v7"
    assert legacy.own_score_baseline_coefficient == 0.0
    assert legacy.neighbor_score_baseline_coefficient == 0.0
    assert legacy.tau_own_score_slope == 0.0
    assert legacy.tau_neighbor_score_slope == 0.0
    assert legacy.gamma_own_score_slope == 0.0
    assert legacy.gamma_neighbor_score_slope == 0.0


def test_observation_noise_changes_context_only() -> None:
    common = dict(
        n_units=10,
        er_edge_probability=0.31,
        tau_episode_sd=0.2,
        tau_unit_sd=0.1,
        gamma_sd=0.2,
        eta_sd=0.0,
    )
    clean = generate_batch(
        DataConfig(**common, noise_sd=0.0),
        batch_size=16,
        seed=44,
        device=torch.device("cpu"),
    )
    noisy = generate_batch(
        DataConfig(**common, noise_sd=1.0),
        batch_size=16,
        seed=44,
        device=torch.device("cpu"),
    )

    assert torch.equal(clean["queries"], noisy["queries"])
    assert torch.equal(clean["low_sampled_exposure"], noisy["low_sampled_exposure"])
    assert torch.equal(clean["high_sampled_exposure"], noisy["high_sampled_exposure"])
    assert torch.equal(clean["oracle_ite"], noisy["oracle_ite"])
    assert torch.equal(clean["query_effect"], noisy["query_effect"])
    assert not torch.equal(clean["y_obs"], noisy["y_obs"])


def test_compact_observed_tokens_keep_all_factual_outcomes_visible() -> None:
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31, noise_sd=1.0),
        batch_size=4,
        seed=54,
        device=torch.device("cpu"),
    )

    assert not hasattr(demo, "CONTEXT_FRACTION")
    assert not hasattr(demo, "N_CONTEXT_UNITS")
    assert not hasattr(demo, "N_QUERY_UNITS")
    assert not hasattr(demo, "context_query_counts")
    assert batch["tokens"].shape == (4, 10, 5)
    assert torch.equal(batch["tokens"][..., 0], batch["x"])
    assert torch.equal(batch["tokens"][..., 1], batch["observed_treatment"])
    assert torch.equal(batch["tokens"][..., 2], batch["y_obs"])
    assert torch.equal(batch["tokens"][..., 3], batch["observed_exposure"])
    expected_degree_norm = batch["adjacency"].sum(dim=-1) / float(10 - 1)
    assert torch.equal(batch["tokens"][..., 4], expected_degree_norm)



def _configure_embedding_to_expose_treated_neighbor_signal(
    model: LocalNetworkQueryTransformer,
) -> None:
    """Make the embedding output depend only on the first interference channel."""

    with torch.no_grad():
        for parameter in model.self_encoder.parameters():
            parameter.zero_()
        for parameter in model.neighbor_encoder.parameters():
            parameter.zero_()
        model.neighbor_encoder[-1].bias.fill_(1.0)

        for parameter in model.embedding_fusion.parameters():
            parameter.zero_()
        # Fusion input is [self(d), interference(d), exposure, degree].
        model.embedding_fusion[0].weight[0, model.config.d_model] = 1.0
        model.embedding_fusion[2].weight[0, 0] = 1.0


def _configure_embedding_to_expose_local_topology(
    model: LocalNetworkQueryTransformer,
) -> None:
    """Make the root embedding expose only the first topology-message channel."""

    with torch.no_grad():
        for parameter in model.self_encoder.parameters():
            parameter.zero_()
        for parameter in model.neighbor_encoder.parameters():
            parameter.zero_()
        model.neighbor_encoder[-1].bias.fill_(1.0)

        for parameter in model.topology_encoder.parameters():
            parameter.zero_()
        model.topology_encoder[0].weight[0, model.config.d_model] = 1.0
        model.topology_encoder[2].weight[0, 0] = 1.0

        for parameter in model.embedding_fusion.parameters():
            parameter.zero_()
        # Fusion input is [self(d), interference(d), exposure, degree].
        model.embedding_fusion[0].weight[0, model.config.d_model] = 1.0
        model.embedding_fusion[2].weight[0, 0] = 1.0


def test_network_aware_embedding_uses_edges_among_root_neighbors() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    _configure_embedding_to_expose_local_topology(model)

    # Root 0 has the same three direct neighbors in both graphs, with the same
    # treatment vector, explicit exposure, and root degree. Only the edge
    # among neighbors changes: treated-treated (1--2) versus treated-untreated
    # (1--3). A topology-aware root embedding must distinguish them.
    tokens = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 2.0 / 3.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    linked_treated = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    linked_untreated = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        ]
    )

    with torch.no_grad():
        treated_topology = model._network_aware_embedding(tokens, linked_treated)
        untreated_topology = model._network_aware_embedding(tokens, linked_untreated)

    assert not torch.allclose(
        treated_topology[:, 0], untreated_topology[:, 0]
    )


def test_root_topology_embedding_excludes_outside_two_hop_nodes() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    _configure_embedding_to_expose_local_topology(model)

    # Node 3 is not a direct neighbor of root 0. Adding edge 1--3 must not
    # enter root 0's induced one-hop topology message.
    tokens = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 2.0 / 3.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0, 0.0],
                [9.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    base = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    outside_edge = base.clone()
    outside_edge[0, 1, 3] = 1.0
    outside_edge[0, 3, 1] = 1.0

    with torch.no_grad():
        base_embedding = model._network_aware_embedding(tokens, base)
        outside_embedding = model._network_aware_embedding(tokens, outside_edge)

    assert torch.allclose(base_embedding[:, 0], outside_embedding[:, 0])


def test_network_aware_embedding_uses_one_hop_treated_neighbors() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    _configure_embedding_to_expose_treated_neighbor_signal(model)

    # [X, T, Y_obs, E_obs, degree_norm]. Keep the explicit exposure fixed so
    # any change below must come from the new treated-neighbor branch.
    untreated = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.5], [2.0, 0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    treated = untreated.clone()
    treated[0, 1, 1] = 1.0

    connected = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    disconnected = torch.zeros_like(connected)

    with torch.no_grad():
        untreated_connected = model._network_aware_embedding(untreated, connected)
        treated_connected = model._network_aware_embedding(treated, connected)
        untreated_disconnected = model._network_aware_embedding(
            untreated, disconnected
        )
        treated_disconnected = model._network_aware_embedding(treated, disconnected)

    assert not torch.allclose(
        untreated_connected[:, 0], treated_connected[:, 0]
    )
    assert torch.allclose(
        untreated_disconnected[:, 0], treated_disconnected[:, 0]
    )


def test_isolated_neighbor_embedding_is_finite() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    tokens = torch.randn(2, 4, 5)
    tokens[..., 1] = torch.randint(0, 2, (2, 4), dtype=torch.int64).float()
    adjacency = torch.zeros(2, 4, 4)

    with torch.no_grad():
        embedded = model._network_aware_embedding(tokens, adjacency)

    assert embedded.shape == (2, 4, model.config.d_model)
    assert torch.isfinite(embedded).all()


def test_gmm_head_outputs_query_units_with_graph_aware_global_attention() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=4,
        seed=55,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        original = model(batch["tokens"], demo.cepo_queries(batch), batch["adjacency"])

        permutation = torch.tensor([3, 0, 7, 5, 2, 6, 1, 4, 9, 8])
        permuted = model(
            batch["tokens"][:, permutation],
            demo.cepo_queries(batch)[:, permutation],
            batch["adjacency"][:, permutation][:, :, permutation],
        )

    for key in ("gmm_pi", "gmm_mu", "gmm_sigma"):
        assert original[key].shape == (4, 10, 4, 5)
        assert torch.allclose(
            permuted[key],
            original[key][:, permutation],
            atol=1e-5,
            rtol=1e-5,
        )
    assert torch.allclose(
        original["gmm_pi"].sum(dim=-1),
        torch.ones(4, 10, 4),
        atol=1e-6,
    )
    assert torch.all(original["gmm_pi"] > 0.0)
    assert torch.all(original["gmm_sigma"] >= 1e-3)


def test_cepo_predictions_share_a_query_conditioned_head():
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    b=generate_batch(DataConfig(n_units=10,er_edge_probability=.31),2,seed=5,device=torch.device('cpu'))
    q=demo.cepo_queries(b).clone()
    q[...,1,:]=q[...,0,:]
    with torch.no_grad():
        out=model(b['tokens'],q,b['adjacency'])
    for value in out.values():
        torch.testing.assert_close(value[...,0,:],value[...,1,:])




def test_transformer_layer_has_no_graph_bias_and_ignores_adjacency() -> None:
    config = small_model_config()
    layer = demo.GraphBiasedTransformerEncoderLayer(config).eval()
    src = torch.randn(2, 10, config.d_model)
    empty = torch.zeros(2, 10, 10)
    complete = torch.ones_like(empty)
    diagonal = torch.arange(10)
    complete[:, diagonal, diagonal] = 0.0

    assert not hasattr(layer, "edge_bias")

    with torch.no_grad():
        first = layer(src, empty)
        second = layer(src, complete)

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_model_has_no_trainable_graph_bias_parameter() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()

    parameter_names = {name for name, _ in model.named_parameters()}

    assert not any("edge_bias" in name for name in parameter_names)


def test_gmm_nll_prefers_a_component_near_the_target() -> None:
    target = torch.tensor([[[0.25]]])
    pi = torch.tensor([[[[1.0]]]])
    sigma = torch.tensor([[[[0.1]]]])
    near_mu = torch.tensor([[[[0.25]]]])
    far_mu = torch.tensor([[[[2.0]]]])

    near = demo.gmm_nll_loss(pi, near_mu, sigma, target)
    far = demo.gmm_nll_loss(pi, far_mu, sigma, target)
    assert near < far


def test_gmm_posterior_summary_for_single_normal() -> None:
    pi = torch.ones(1, 1, 1, 1)
    mu = torch.tensor([[[[2.0]]]])
    sigma = torch.tensor([[[[0.5]]]])

    mean = demo.gmm_posterior_mean(pi, mu)
    lower, upper = demo.gmm_posterior_interval(pi, mu, sigma, mass=0.9)

    assert torch.allclose(mean, torch.tensor([[[2.0]]]))
    assert lower.item() == pytest.approx(2.0 - 1.644854 * 0.5, abs=2e-4)
    assert upper.item() == pytest.approx(2.0 + 1.644854 * 0.5, abs=2e-4)


def test_model_rejects_old_contrast_query_dimension():
    with pytest.raises(ValueError,match='CEPO'):
        small_model_config(query_dim=4)




def test_training_loss_is_equal_weight_cepo_nll():
    b=generate_batch(DataConfig(n_units=10,er_edge_probability=.31),2,seed=6,device=torch.device('cpu'))
    model=LocalNetworkQueryTransformer(small_model_config())
    p=model(b['tokens'],demo.cepo_queries(b),b['adjacency'])
    loss=compute_training_losses(model,p,b)['total_loss']
    target=demo.cepo_targets(b)
    expected=demo.gmm_nll_loss(p['gmm_pi'],p['gmm_mu'],p['gmm_sigma'],target)
    torch.testing.assert_close(loss,expected)




def test_training_loss_uses_all_unit_cepo_labels() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=4,
        seed=67,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        predictions = model(batch["tokens"], demo.cepo_queries(batch), batch["adjacency"])
    original = compute_training_losses(model, predictions, batch)["total_loss"]

    changed = {key: value.clone() for key, value in batch.items()}
    changed["structural_baseline"][:, 0] += 10_000.0
    changed_loss = compute_training_losses(model, predictions, changed)["total_loss"]

    assert changed_loss > original


def test_query_effect_targets_rejects_query_only_legacy_targets() -> None:
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=4,
        seed=68,
        device=torch.device("cpu"),
    )
    legacy = {
        "tokens": batch["tokens"],
        "query_effect": batch["query_effect"][:, 8:, :],
    }

    with pytest.raises(ValueError, match=r"\[batch, units, 3\]"):
        demo.query_effect_targets(legacy)


def test_train_step_changes_parameters_with_adam() -> None:
    model = LocalNetworkQueryTransformer(small_model_config())
    optimizer = demo.create_optimizer(model.parameters())
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=8,
        seed=77,
        device=torch.device("cpu"),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, optimizer, batch)

    assert torch.isfinite(torch.tensor(loss))
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )


def test_metrics_separate_cepo_distributions_from_effect_point_estimates():
    model=LocalNetworkQueryTransformer(small_model_config())
    b=generate_batch(DataConfig(n_units=10,er_edge_probability=.31),2,seed=88,device=torch.device('cpu'))
    with torch.no_grad(): p=model(b['tokens'],demo.cepo_queries(b),b['adjacency'])
    metrics=compute_query_metrics(model,p,b)
    for arm in ('mu00','mu01','mu10','mu11'):
        assert all(arm+'_'+key in metrics for key in ('gmm_nll','rmse','mae','bias','coverage_90'))
    for effect in ('direct','spillover','total'):
        assert all(effect+'_effect_'+key in metrics for key in ('rmse','mae','bias'))
        assert effect+'_effect_coverage_90' not in metrics
    assert metrics['validation_gmm_nll']==pytest.approx(metrics['cepo_gmm_nll_macro'])




def test_continuous_gmm_diagnostic_has_no_support_overflow() -> None:
    diagnostic = diagnose_query_effect_support(
        DataConfig(n_units=10, er_edge_probability=0.31),
        ModelConfig(),
        batch_size=128,
        seed=99,
        device=torch.device("cpu"),
    )

    for name in ("direct_effect", "spillover_effect", "total_effect"):
        assert name in diagnostic
        assert diagnostic[name]["minimum"] <= diagnostic[name]["maximum"]
        assert diagnostic[name]["overflow_rate"] == 0.0
    assert diagnostic["support"] == "continuous_gmm"
    assert diagnostic["sample_size_per_effect"] == 128 * 10


def test_epoch_episode_indices_visit_every_episode_once() -> None:
    first = demo.epoch_episode_indices(11, training_seed=7, epoch=1)
    second = demo.epoch_episode_indices(11, training_seed=7, epoch=1)
    later = demo.epoch_episode_indices(11, training_seed=7, epoch=2)

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(11))
    assert sorted(later.tolist()) == list(range(11))
    assert not torch.equal(first, later)


def test_adam_and_plateau_scheduler_match_requested_defaults() -> None:
    model = LocalNetworkQueryTransformer(small_model_config())
    optimizer = demo.create_optimizer(model.parameters())
    scheduler = demo.create_plateau_scheduler(optimizer)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert scheduler.mode == "min"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 5


def test_evaluate_reports_pooled_cepo_nll():
    model=LocalNetworkQueryTransformer(small_model_config()).eval()
    b=generate_batch(DataConfig(n_units=10,er_edge_probability=.31),5,seed=166,device=torch.device('cpu'))
    metrics=demo.evaluate(model,b,batch_size=2,device=torch.device('cpu'))
    with torch.no_grad():
        p=model(b['tokens'],demo.cepo_queries(b),b['adjacency'])
        expected=demo.gmm_nll_loss(p['gmm_pi'],p['gmm_mu'],p['gmm_sigma'],demo.cepo_targets(b))
    assert metrics['validation_gmm_nll']==pytest.approx(float(expected),rel=1e-6,abs=1e-6)




def test_checkpoint_roundtrip_and_multiple_paths(tmp_path: Path) -> None:
    data_config = DataConfig(n_units=10, er_edge_probability=0.31)
    model_config = small_model_config()
    model = LocalNetworkQueryTransformer(model_config)
    optimizer = demo.create_optimizer(model.parameters())
    scheduler = demo.create_plateau_scheduler(optimizer)
    batch = generate_batch(
        data_config,
        batch_size=4,
        seed=111,
        device=torch.device("cpu"),
    )
    train_step(model, optimizer, batch)

    paths = checkpoint_paths(tmp_path / "model.pt")
    assert paths["best_nll"].name == "model.pt"
    assert paths["best_point"].name == "model_best_point.pt"
    assert paths["latest"].name == "model_latest.pt"

    save_checkpoint(
        path=paths["best_nll"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_validation_gmm_nll=1.2,
        best_point_rmse=0.3,
        monitor_name="validation_gmm_nll",
        monitor_value=1.2,
        data_config=data_config,
        model_config=model_config,
    )
    restored, restored_optimizer, metadata = load_checkpoint(
        path=paths["best_nll"],
        device=torch.device("cpu"),
    )
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(batch["tokens"], demo.cepo_queries(batch), batch["adjacency"])
        actual = restored(batch["tokens"], demo.cepo_queries(batch), batch["adjacency"])

    for key in expected:
        assert torch.equal(expected[key], actual[key])
    assert isinstance(restored_optimizer, torch.optim.Adam)
    assert metadata["monitor_name"] == "validation_gmm_nll"
    assert metadata["best_validation_gmm_nll"] == 1.2
    assert metadata["best_point_rmse"] == 0.3
    assert metadata["data_config"]["er_edge_probability"] == pytest.approx(0.31)
    assert set(metadata["data_config"]) == set(DataConfig.__dataclass_fields__)




def test_algorithm_2_reports_four_gmm_queries_and_three_effects_per_unit() -> None:
    model = LocalNetworkQueryTransformer(small_model_config())
    first = apply_algorithm_2(
        model,
        DataConfig(n_units=10, er_edge_probability=0.31),
        num_datasets=2,
        seed=2026,
        device=torch.device("cpu"),
    )
    second = apply_algorithm_2(
        model,
        DataConfig(n_units=10, er_edge_probability=0.31),
        num_datasets=2,
        seed=2026,
        device=torch.device("cpu"),
    )

    assert first == second
    assert first["test_seed"] == 2026
    assert first["num_datasets"] == 2
    assert first["gmm_n_components"] == 5
    assert "num_context_units" not in first
    assert "num_query_units" not in first
    assert first["num_observed_units_per_dataset"] == 10
    assert first["num_queried_units_per_dataset"] == 10
    assert first["num_evaluated_units_per_dataset"] == 10
    assert len(first["units"]) == 2 * 10 * 3
    assert {row["unit_id"] for row in first["units"]} == set(range(1, 11))
    assert {row["query_type"] for row in first["units"]} == {
        "direct",
        "spillover",
        "total",
    }
    assert "cepo_gmm_nll_macro" in first["overall"]
    assert "x" in first["units"][0]
    assert "tau" in first["units"][0]
    assert "gamma" in first["units"][0]
    assert "eta" in first["units"][0]
    assert len(first["mu_predictions"]) == 2 * 10 * 4
    assert "gmm_pi" in first["mu_predictions"][0]
    assert "effect_lower_90" not in first["units"][0]


def test_algorithm_2_report_writes_json_and_csv(tmp_path: Path) -> None:
    report = apply_algorithm_2(
        LocalNetworkQueryTransformer(small_model_config()),
        DataConfig(n_units=10, er_edge_probability=0.31),
        num_datasets=2,
        seed=2026,
        device=torch.device("cpu"),
    )
    paths = save_algorithm_2_report(report, tmp_path)

    assert paths["summary_json"].is_file()
    assert paths["dataset_csv"].is_file()
    assert paths["unit_csv"].is_file()
    with paths["summary_json"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert "units" not in summary
    with paths["unit_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60


def test_episode_csv_roundtrip_preserves_training_tensors(tmp_path: Path) -> None:
    source = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=6,
        seed=123,
        device=torch.device("cpu"),
    )
    csv_path = tmp_path / "episodes.csv"

    save_episode_batch_csv(source, csv_path)
    restored, dataset_ids = load_episode_batch_csv(
        csv_path,
        device=torch.device("cpu"),
    )

    assert dataset_ids == list(range(6))
    for key in (
        "tokens",
        "queries",
        "adjacency",
        "graph_type",
        "star_center",
        "observed_treatment",
        "observed_exposure",
        "degree",
        "majority_arm",
        "low_sampled_exposure",
        "high_sampled_exposure",
        "y_obs",
        "outcome_a",
        "outcome_b",
        "query_effect",
        "oracle_ite",
        "oracle_ate",
        "x",
        "tau",
        "gamma",
        "eta",
    ):
        assert torch.equal(restored[key], source[key]), key


def test_er_graph_type_roundtrips_and_reports_er_name(tmp_path: Path) -> None:
    data_config = DataConfig(n_units=10, er_edge_probability=0.31)
    source = generate_batch(
        data_config,
        batch_size=2,
        seed=124,
        device=torch.device("cpu"),
    )
    csv_path = tmp_path / "er_episodes.csv"

    save_episode_batch_csv(source, csv_path)
    restored, _ = load_episode_batch_csv(
        csv_path,
        device=torch.device("cpu"),
    )
    report = apply_algorithm_2(
        LocalNetworkQueryTransformer(small_model_config()),
        data_config,
        num_datasets=2,
        seed=2026,
        device=torch.device("cpu"),
        batch=restored,
    )

    assert torch.all(source["graph_type"] == ER_GRAPH)
    assert torch.equal(restored["graph_type"], source["graph_type"])
    assert {row["graph_type"] for row in report["datasets"]} == {"er"}
    assert {row["graph_type"] for row in report["units"]} == {"er"}


def test_stratified_episode_split_falls_back_when_a_family_has_one_episode(
    tmp_path: Path,
) -> None:
    config = DataConfig(
        n_units=40,
        er_edge_probability=6.0 / 39.0,
        graph_family="mixed",
    )
    graph_types = torch.tensor(
        [
            demo.ER_GRAPH,
            demo.ER_GRAPH,
            demo.CONFIGURATION_GRAPH,
            demo.RGG_GRAPH,
        ],
        dtype=torch.long,
    )
    source = generate_batch(
        config,
        batch_size=4,
        seed=987,
        device=torch.device("cpu"),
        graph_types=graph_types,
    )
    source_path = tmp_path / "small_mixed.csv"
    train_path, validation_path = episode_split_paths(source_path)
    save_episode_batch_csv(source, source_path)

    summary = split_episode_csv(
        source_path,
        train_path,
        validation_path,
        validation_fraction=0.25,
        split_seed=7,
        stratify_by_graph_type=True,
    )

    _, train_ids = load_episode_batch_csv(train_path, device=torch.device("cpu"))
    _, validation_ids = load_episode_batch_csv(
        validation_path, device=torch.device("cpu")
    )
    assert summary == {
        "total_datasets": 4,
        "training_datasets": 3,
        "validation_datasets": 1,
    }
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(range(4))


def test_episode_csv_split_is_80_20_by_whole_episode(tmp_path: Path) -> None:
    source = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=10,
        seed=321,
        device=torch.device("cpu"),
    )
    source_path = tmp_path / "episodes.csv"
    train_path, validation_path = episode_split_paths(source_path)
    save_episode_batch_csv(source, source_path)

    summary = split_episode_csv(
        source_path,
        train_path,
        validation_path,
        validation_fraction=0.2,
        split_seed=7,
    )
    train_batch, train_ids = load_episode_batch_csv(
        train_path,
        device=torch.device("cpu"),
    )
    validation_batch, validation_ids = load_episode_batch_csv(
        validation_path,
        device=torch.device("cpu"),
    )

    assert summary == {
        "total_datasets": 10,
        "training_datasets": 8,
        "validation_datasets": 2,
    }
    assert len(train_ids) == 8
    assert len(validation_ids) == 2
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(range(10))
    assert train_batch["tokens"].shape[0] == 8
    assert validation_batch["tokens"].shape[0] == 2


def test_select_episode_batch_selects_only_requested_episodes() -> None:
    source = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=5,
        seed=456,
        device=torch.device("cpu"),
    )
    selected = select_episode_batch(
        source,
        torch.tensor([4, 1, 4]),
        device=torch.device("cpu"),
    )

    assert selected["tokens"].shape[0] == 3
    assert torch.equal(selected["tokens"][0], source["tokens"][4])
    assert torch.equal(selected["tokens"][1], source["tokens"][1])
    assert torch.equal(selected["tokens"][2], source["tokens"][4])


def test_train_steps_scheduler_once_per_epoch_with_validation_gmm_nll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=4,
        seed=177,
        device=torch.device("cpu"),
    )
    validation_batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=2,
        seed=178,
        device=torch.device("cpu"),
    )
    monitored = []

    class SchedulerSpy:
        def step(self, value):
            monitored.append(float(value))

        def state_dict(self):
            return {}

    monkeypatch.setattr(
        demo,
        "create_plateau_scheduler",
        lambda optimizer, **kwargs: SchedulerSpy(),
    )
    validation_values = iter([1.25, 1.0, 0.9])
    monkeypatch.setattr(demo, "train_epoch", lambda *args, **kwargs: 2.0)

    def fake_evaluate(*args, **kwargs):
        value = next(validation_values)
        return {
            "validation_gmm_nll": value,
            "effect_rmse_macro": 0.5,
            "direct_effect_rmse": 0.4,
            "spillover_effect_rmse": 0.5,
            "total_effect_rmse": 0.6,
        }

    monkeypatch.setattr(demo, "evaluate", fake_evaluate)
    monkeypatch.setattr(demo, "save_checkpoint", lambda **kwargs: None)

    demo.train(
        data_config=DataConfig(n_units=10, er_edge_probability=0.31),
        model_config=small_model_config(),
        train_config=TrainConfig(epochs=3, batch_size=2),
        train_batch=train_batch,
        validation_batch=validation_batch,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "model.pt",
    )

    assert monitored == [1.25, 1.0, 0.9]


def test_train_uses_csv_batches_without_generating_new_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_data = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31, eta_sd=0.0),
        batch_size=10,
        seed=789,
        device=torch.device("cpu"),
    )
    source_path = tmp_path / "episodes.csv"
    train_path, validation_path = episode_split_paths(source_path)
    save_episode_batch_csv(all_data, source_path)
    split_episode_csv(
        source_path,
        train_path,
        validation_path,
        validation_fraction=0.2,
        split_seed=0,
    )
    train_batch, _ = load_episode_batch_csv(
        train_path,
        device=torch.device("cpu"),
    )
    validation_batch, _ = load_episode_batch_csv(
        validation_path,
        device=torch.device("cpu"),
    )

    def fail_generate_batch(*args, **kwargs):
        raise AssertionError("training/validation must not call generate_batch")

    monkeypatch.setattr(demo, "generate_batch", fail_generate_batch)
    model = demo.train(
        data_config=DataConfig(n_units=10, er_edge_probability=0.31, eta_sd=0.0),
        model_config=small_model_config(),
        train_config=TrainConfig(epochs=1, batch_size=2, training_seed=0),
        train_batch=train_batch,
        validation_batch=validation_batch,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "model.pt",
    )

    assert isinstance(model, LocalNetworkQueryTransformer)
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "model_best_point.pt").is_file()


def test_prepare_episode_csv_splits_generates_once_and_reuses_master(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_csv = tmp_path / "training_episodes.csv"
    first = prepare_episode_csv_splits(
        DataConfig(n_units=10, er_edge_probability=0.31),
        data_csv=data_csv,
        num_datasets=10,
        data_seed=12345,
        validation_fraction=0.2,
        split_seed=0,
        overwrite=False,
    )

    assert data_csv.is_file()
    assert first["train_path"].is_file()
    assert first["validation_path"].is_file()
    assert first["summary"]["training_datasets"] == 8
    assert first["summary"]["validation_datasets"] == 2

    def fail_generate_batch(*args, **kwargs):
        raise AssertionError("existing master CSV should be reused")

    monkeypatch.setattr(demo, "generate_batch", fail_generate_batch)
    second = prepare_episode_csv_splits(
        DataConfig(n_units=10, er_edge_probability=0.31),
        data_csv=data_csv,
        num_datasets=999,
        data_seed=999,
        validation_fraction=0.2,
        split_seed=0,
        overwrite=False,
    )
    assert second["summary"] == first["summary"]






def test_current_source_has_no_v5_schema_or_undefined_paper_name_residue() -> None:
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "rooted_all_node_queries_p002_v5" not in source
    assert "PAPER_METHOD_NAMES" not in source
    assert "PAPER_EFFECT_NAMES" not in source

def test_parse_args_exposes_er_only_and_paper_baseline_controls(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_local_network_interference.py"])

    args = demo.parse_args()

    assert args.er_edge_probability == pytest.approx(0.02)
    assert not hasattr(args, "counterfactual_flip_count")
    assert args.localized_mode == "kernel"
    assert args.localized_bandwidth == pytest.approx(2.0)
    assert args.localized_min_neighbors == 20
    assert args.localized_arm_samples == 16
    assert args.standard_ite_gps_ridge == pytest.approx(1e-3)
    assert args.standard_ite_epochs == 500
    assert args.standard_ite_learning_rate == pytest.approx(1e-3)
    assert args.standard_ite_balance_weight == pytest.approx(1e-2)
    assert args.standard_ite_hidden_dim == 32
    for removed_name in (
        "star_probability",
        "er_probability",
        "sbm_probability",
        "small_world_probability",
        "random_regular_probability",
        "random_regular_degree",
    ):
        assert not hasattr(args, removed_name)
    assert args.skip_paper_baselines is False
    assert args.baseline_repetitions == 20
    assert args.baseline_datasets == 64
    assert args.baseline_bandwidth == 3
    assert args.baseline_ridge == pytest.approx(0.0)
    monkeypatch.setattr(sys,"argv",["train_local_network_interference.py","--direct-loss-weight","2"])
    with pytest.raises(SystemExit):
        demo.parse_args()


def test_removed_graph_cli_flags_are_rejected(monkeypatch) -> None:
    for flag in (
        "--star-probability",
        "--er-probability",
        "--sbm-probability",
        "--small-world-probability",
        "--random-regular-probability",
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_local_network_interference.py", flag, "1.0"],
        )
        with pytest.raises(SystemExit):
            demo.parse_args()

def test_default_configuration_uses_1000_units_with_all_nodes_queried() -> None:
    data_config = DataConfig()
    assert data_config.n_units == 1000
    assert demo.N_UNITS == 1000
    assert not hasattr(demo, "N_CONTEXT_UNITS")
    assert not hasattr(demo, "N_QUERY_UNITS")

    batch = generate_batch(
        data_config,
        batch_size=2,
        seed=100,
        device=torch.device("cpu"),
    )
    assert batch["tokens"].shape == (2, 1000, 5)
    assert batch["queries"].shape == (2, 1000, 3, 4)
    assert batch["adjacency"].shape == (2, 1000, 1000)
    assert batch["degree"].shape == (2, 1000)
    assert batch["majority_arm"].shape == (2, 1000)
    assert batch["oracle_ite"].shape == (2, 1000, 3)
    assert batch["oracle_ate"].shape == (2, 3)
    assert batch["structural_baseline"].shape == (2, 1000)
    expected_degree_norm = batch["adjacency"].sum(dim=-1) / 999.0
    assert torch.allclose(batch["tokens"][..., 4], expected_degree_norm)

    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    with torch.no_grad():
        predictions = model(
            batch["tokens"], demo.cepo_queries(batch), batch["adjacency"]
        )
    for key in ("gmm_pi", "gmm_mu", "gmm_sigma"):
        assert predictions[key].shape == (2, 1000, 4, 5)


def test_1000_unit_episode_csv_roundtrip(tmp_path: Path) -> None:
    source = generate_batch(
        DataConfig(),
        batch_size=1,
        seed=101,
        device=torch.device("cpu"),
    )
    csv_path = tmp_path / "episodes_1000.csv"
    save_episode_batch_csv(source, csv_path)
    restored, dataset_ids = load_episode_batch_csv(
        csv_path,
        device=torch.device("cpu"),
    )

    assert dataset_ids == [0]
    assert restored["tokens"].shape == (1, 1000, 5)
    assert restored["queries"].shape == (1, 1000, 3, 4)
    assert restored["adjacency"].shape == (1, 1000, 1000)
    assert restored["degree"].shape == (1, 1000)
    assert restored["oracle_ite"].shape == (1, 1000, 3)
    assert restored["structural_baseline"].shape == (1, 1000)
    for key in demo.CSV_TENSOR_KEYS:
        assert torch.equal(restored[key], source[key])
    assert torch.equal(restored["oracle_ate"], source["oracle_ate"])
    assert torch.allclose(restored["covariate_score"], source["covariate_score"])
    assert torch.allclose(
        restored["neighbor_covariate_score"], source["neighbor_covariate_score"]
    )


def test_100_unit_run_executes_unified_baselines_without_exact_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = LocalNetworkQueryTransformer(small_model_config())
    data_config = DataConfig(n_units=100, er_edge_probability=0.06)
    batch = generate_batch(
        data_config,
        batch_size=1,
        seed=102,
        device=torch.device("cpu"),
    )
    calls = {}

    def fake_evaluate(**kwargs):
        calls["batch"] = kwargs["batch"]
        calls["arm_samples"] = kwargs["localized_arm_samples"]
        calls["standard_ite_epochs"] = kwargs["standard_ite_epochs"]
        calls["standard_ite_hidden_dim"] = kwargs["standard_ite_hidden_dim"]
        return {"main_ite_methods": ["PFN with interference", "Localized DR-Lasso", "HyperSCI",], "main_ate_methods": ["PFN with interference", "HT", "Haj", "F", "L", "Tnet"]}

    def fake_save(report, output_dir):
        calls["report"] = report
        calls["output_dir"] = output_dir
        return {"benchmark_summary": Path(output_dir) / "benchmark_summary.json"}

    monkeypatch.setattr(demo, "evaluate_unified_benchmark", fake_evaluate)
    monkeypatch.setattr(demo, "save_unified_benchmark", fake_save)
    args = type(
        "Args",
        (),
        {
            "skip_paper_baselines": False,
            "skip_localized_baselines": False,
            "baseline_bandwidth": 3,
            "baseline_ridge": 1e-6,
            "localized_bandwidth": 0.24,
            "localized_min_neighbors": 20,
            "localized_mode": "kernel",
            "localized_lasso_alpha": 1e-4,
            "localized_nuisance_ridge": 1e-3,
            "localized_cross_fit_folds": 5,
            "localized_arm_samples": 1,
            "test_seed": 2026,
            "test_output_dir": tmp_path,
        },
    )()
    demo._run_paper_baselines(
        model=model,
        data_config=data_config,
        args=args,
        batch=batch,
    )
    assert calls["batch"] is batch
    assert calls["arm_samples"] == 1
    assert calls["standard_ite_epochs"] == 500
    assert calls["standard_ite_hidden_dim"] == 32
    assert calls["output_dir"] == tmp_path
    assert calls["report"]["main_ate_methods"] == ["PFN with interference", "HT", "Haj", "F", "L", "Tnet"]


def test_distributed_epoch_indices_partition_divisible_dataset() -> None:
    shards = [
        demo.distributed_epoch_episode_indices(
            32,
            training_seed=7,
            epoch=3,
            rank=rank,
            world_size=4,
            batch_size=2,
        )
        for rank in range(4)
    ]

    assert all(shard.shape == (8,) for shard in shards)
    combined = torch.cat(shards)
    assert sorted(combined.tolist()) == list(range(32))
    assert len(set(combined.tolist())) == 32


def test_distributed_epoch_indices_pad_equally_for_ddp_steps() -> None:
    shards = [
        demo.distributed_epoch_episode_indices(
            10,
            training_seed=9,
            epoch=1,
            rank=rank,
            world_size=3,
            batch_size=2,
        )
        for rank in range(3)
    ]

    assert all(shard.shape == (4,) for shard in shards)
    assert all(shard.numel() % 2 == 0 for shard in shards)
    combined = torch.cat(shards)
    assert set(range(10)).issubset(set(combined.tolist()))
    assert combined.numel() == 12


def test_distributed_runtime_defaults_to_single_process(monkeypatch) -> None:
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    runtime = demo.distributed_runtime_from_env()

    assert runtime.rank == 0
    assert runtime.world_size == 1
    assert set(runtime.__dataclass_fields__) == {"rank", "world_size"}
    assert runtime.is_distributed is False
    assert runtime.is_main is True


def test_distributed_runtime_reads_torchrun_environment(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")

    runtime = demo.distributed_runtime_from_env()

    assert runtime.rank == 5
    assert runtime.world_size == 8
    assert set(runtime.__dataclass_fields__) == {"rank", "world_size"}
    assert runtime.is_distributed is True
    assert runtime.is_main is False


def test_distributed_backend_matches_device_type() -> None:
    assert demo.distributed_backend_for_device(torch.device("cpu")) == "gloo"
    assert demo.distributed_backend_for_device(torch.device("privateuseone:0")) == "mccl"


def test_unwrap_model_returns_wrapped_module() -> None:
    model = LocalNetworkQueryTransformer(small_model_config())

    class Wrapper:
        def __init__(self, module):
            self.module = module

    assert demo.unwrap_model(model) is model
    assert demo.unwrap_model(Wrapper(model)) is model


def test_load_existing_episode_csv_splits_reuses_written_files(tmp_path: Path) -> None:
    data_csv = tmp_path / "episodes.csv"
    prepared = prepare_episode_csv_splits(
        DataConfig(n_units=10, er_edge_probability=0.31),
        data_csv=data_csv,
        num_datasets=10,
        data_seed=123,
        validation_fraction=0.2,
        split_seed=4,
        overwrite=True,
    )

    loaded = demo.load_existing_episode_csv_splits(data_csv)

    assert loaded["summary"] == prepared["summary"]
    assert torch.equal(loaded["train_batch"]["tokens"], prepared["train_batch"]["tokens"])
    assert torch.equal(
        loaded["validation_batch"]["tokens"],
        prepared["validation_batch"]["tokens"],
    )


def test_transformer_cached_context_query_path_matches_forward() -> None:
    model = LocalNetworkQueryTransformer(small_model_config()).eval()
    batch = generate_batch(
        DataConfig(n_units=10, er_edge_probability=0.31),
        batch_size=2,
        seed=313,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        expected = model(batch["tokens"], demo.cepo_queries(batch), batch["adjacency"])
        encoded = model.encode_context(batch["tokens"], batch["adjacency"])
        actual = model.predict_from_context(encoded, demo.cepo_queries(batch))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
