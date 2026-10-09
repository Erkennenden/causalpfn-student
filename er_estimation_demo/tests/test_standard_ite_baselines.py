import numpy as np
import torch

import train_local_network_interference as demo
from baselines.ite_standard import (
    LITERATURE_ITE_METHOD_NAMES,
    evaluate_literature_ite_baselines_on_batch,
    integrate_majority_ite,
)
from estimands import arm_mean_exposure


def test_literature_ite_method_names_are_stable() -> None:
    assert LITERATURE_ITE_METHOD_NAMES == ("HyperSCI",)


def test_majority_adapter_matches_closed_form_design_average() -> None:
    degree = np.asarray([2, 3, 5, 6], dtype=np.int64)
    base = np.asarray([0.2, -0.1, 0.4, 0.9], dtype=np.float64)
    tau = np.asarray([0.5, 0.7, -0.2, 0.3], dtype=np.float64)
    gamma = np.asarray([0.3, -0.4, 0.8, 0.1], dtype=np.float64)
    eta = np.asarray([0.2, 0.1, -0.3, 0.5], dtype=np.float64)

    def predictor(treatment: int, exposure: np.ndarray) -> np.ndarray:
        t = float(treatment)
        return base + tau * t + gamma * exposure + eta * t * exposure

    actual = integrate_majority_ite(
        predictor=predictor,
        degree=degree,
        treatment_prob=0.4,
    )
    low = np.asarray(
        [arm_mean_exposure(int(d), 0.4, 0) for d in degree], dtype=np.float64
    )
    high = np.asarray(
        [arm_mean_exposure(int(d), 0.4, 1) for d in degree], dtype=np.float64
    )
    expected = np.column_stack(
        [
            tau + eta * low,
            (gamma + eta) * (high - low),
            tau + gamma * (high - low) + eta * high,
        ]
    )
    assert actual.shape == (4, 3)
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)






def test_literature_runner_returns_hypersci_on_same_batch() -> None:
    batch = demo.generate_batch(
        demo.DataConfig(n_units=20, er_edge_probability=0.28, noise_sd=0.05),
        batch_size=2,
        seed=789,
        device=torch.device("cpu"),
    )
    report = evaluate_literature_ite_baselines_on_batch(
        batch=batch,
        treatment_prob=0.5,
        gps_ridge=1.0e-3,
        neural_epochs=5,
        neural_learning_rate=5.0e-3,
        neural_balance_weight=1.0e-2,
        neural_hidden_dim=12,
        seed=1234,
    )
    assert tuple(report["methods"]) == ("HyperSCI",)
    assert all(row["method"] == "HyperSCI" for row in report["unit_effects"])
    assert all(np.isfinite(row["integration_mcse"]) for row in report["unit_effects"])
    assert len(report["unit_effects"]) == 2 * 20 * 3
    assert all(np.isfinite(float(row["estimate"])) for row in report["unit_effects"])
    assert all(bool(row["supported"]) for row in report["unit_effects"])
