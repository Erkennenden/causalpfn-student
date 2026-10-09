from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

import pfn_interference.model as demo


def test_test_only_cli_writes_unified_benchmark_outputs(tmp_path: Path) -> None:
    data_config = demo.DataConfig(
        n_units=8,
        er_edge_probability=0.5,
        tau_episode_sd=0.0,
        tau_unit_sd=0.0,
        gamma_sd=0.0,
        noise_sd=0.0,
    )
    model_config = demo.ModelConfig(
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=32,
        gmm_n_components=2,
    )
    model = demo.LocalNetworkQueryTransformer(model_config)
    optimizer = demo.create_optimizer(model.parameters())
    checkpoint = tmp_path / "smoke.pt"
    demo.save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_validation_gmm_nll=0.0,
        best_point_rmse=0.0,
        monitor_name="smoke",
        monitor_value=0.0,
        data_config=data_config,
        model_config=model_config,
    )
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "-m", "pfn_interference.model",
        "--test-only",
        "--device",
        "cpu",
        "--checkpoint-path",
        str(checkpoint),
        "--test-datasets",
        "1",
        "--test-seed",
        "2026",
        "--test-output-dir",
        str(output_dir),
        "--baseline-bandwidth",
        "2",
        "--localized-mode",
        "knn",
        "--localized-min-neighbors",
        "2",
        "--localized-lasso-alpha",
        "0",
        "--localized-nuisance-ridge",
        "0",
        "--localized-cross-fit-folds",
        "2",
        "--localized-arm-samples",
        "1",
        "--nsi-pcr-rank-max",
        "4",
        "--nsi-min-effective-donors",
        "2",
        "--nsi-max-bandwidth",
        "1.0",
        "--nsi-min-support-mass",
        "0.90",
        "--nsi-svd-rcond",
        "1e-6",
        "--standard-ite-epochs",
        "5",
        "--standard-ite-hidden-dim",
        "12",
        "--tnet-epochs",
        "2",
        "--tnet-hidden-dim",
        "8",
        "--tnet-grid-size",
        "4",
        "--tnet-spline-basis",
        "7",
        "--tnet-learning-rate-1step",
        "0.0001",
        "--tnet-learning-rate-2step",
        "0.01",
        "--skip-causalpfn",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    expected = {
        "ite_unit_results.csv",
        "ite_summary.csv",
        "ate_graph_results.csv",
        "ate_summary.csv",
        "main_ite_table.csv",
        "main_ate_table.csv",
        "benchmark_summary.json",
        "benchmark_readable.txt",
        "oracle_semiparametric_bound.csv",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    with (output_dir / "benchmark_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["main_ite_methods"] == ["PFN with interference", "Localized DR-Lasso", "NSI", "HyperSCI", "Tnet"]
    assert summary["main_ate_methods"] == ["PFN with interference", "HT", "Haj", "F", "L", "reg-net", "Tnet"]

    with (output_dir / "main_ite_table.csv").open(newline="", encoding="utf-8") as handle:
        ite_methods = [row["method"] for row in csv.DictReader(handle)]
    assert ite_methods == ["PFN with interference", "Localized DR-Lasso", "NSI", "HyperSCI", "Tnet"]

    with (output_dir / "main_ate_table.csv").open(newline="", encoding="utf-8") as handle:
        methods = [row["method"] for row in csv.DictReader(handle)]
    assert methods == ["PFN with interference", "HT", "Haj", "F", "L", "reg-net", "Tnet"]


def test_causalpfn_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pfn_interference/model.py"])
    args = demo.parse_args()
    assert args.causalpfn_checkpoint == Path("checkpoints/causalpfn_v0.pt")
    assert args.nsi_pcr_rank_max == 6
    assert args.nsi_min_effective_donors == 12.0
    assert args.nsi_max_bandwidth == 0.5
    assert args.nsi_min_support_mass == 0.99
    assert args.nsi_svd_rcond == 1.0e-6
    assert args.causalpfn_query_chunk_size == 512
    assert args.skip_causalpfn is False
    assert args.tnet_epochs == 160
    assert args.tnet_hidden_dim == 64
    assert args.tnet_grid_size == 20
    assert args.tnet_spline_basis == 12
    assert args.tnet_learning_rate_1step == 1.0e-4
    assert args.tnet_learning_rate_2step == 1.0e-2
