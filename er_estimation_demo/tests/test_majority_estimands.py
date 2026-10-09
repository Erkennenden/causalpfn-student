from __future__ import annotations

import numpy as np
import torch

from pfn_interference.estimands import (
    arm_mean_exposure,
    conditional_count_weights,
    oracle_ate,
    oracle_ite_from_parameters,
    sample_majority_exposures,
)


def test_degree_six_majority_support_and_weights_sum_to_one() -> None:
    low = conditional_count_weights(6, 0.5, 0)
    high = conditional_count_weights(6, 0.5, 1)

    assert low.shape == (7,)
    assert high.shape == (7,)
    assert np.isclose(low[:4].sum(), 1.0)
    assert np.isclose(low[4:].sum(), 0.0)
    assert np.isclose(high[:4].sum(), 0.0)
    assert np.isclose(high[4:].sum(), 1.0)
    assert np.isclose(low.sum(), 1.0)
    assert np.isclose(high.sum(), 1.0)


def test_majority_exposure_means_are_ordered_for_dense_degree() -> None:
    low = arm_mean_exposure(60, 0.5, 0)
    high = arm_mean_exposure(60, 0.5, 1)

    assert 0.0 <= low <= 0.5
    assert 0.5 < high <= 1.0
    assert high > low


def test_linear_eta_zero_oracle_has_total_equal_direct_plus_spillover() -> None:
    ite = oracle_ite_from_parameters(
        degree=torch.tensor([6.0, 60.0]),
        tau=torch.tensor([0.5, 0.4]),
        gamma=torch.tensor([0.3, -0.2]),
        eta=torch.zeros(2),
        treatment_prob=0.5,
    )

    torch.testing.assert_close(ite[:, 2], ite[:, 0] + ite[:, 1])
    torch.testing.assert_close(ite[:, 0], torch.tensor([0.5, 0.4]))
    assert ite[0, 1] > 0.0
    assert ite[1, 1] < 0.0


def test_oracle_ate_is_node_average() -> None:
    ite = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]],
            [[2.0, -1.0, 1.0], [4.0, 2.0, 6.0]],
        ]
    )
    ate = oracle_ate(ite)
    torch.testing.assert_close(
        ate,
        torch.tensor([[2.0, 3.0, 5.0], [3.0, 0.5, 3.5]]),
    )


def test_sampled_majority_exposures_respect_exact_support() -> None:
    degree = torch.tensor([[1.0, 2.0, 6.0, 60.0]])
    generator = torch.Generator(device="cpu").manual_seed(123)

    low, high = sample_majority_exposures(
        degree,
        treatment_prob=0.5,
        generator=generator,
    )

    counts_low = torch.round(low * degree).to(torch.int64)
    counts_high = torch.round(high * degree).to(torch.int64)
    threshold = torch.floor(degree / 2).to(torch.int64)
    assert bool((counts_low <= threshold).all())
    assert bool((counts_high > threshold).all())
    torch.testing.assert_close(low * degree, counts_low.to(torch.float32))
    torch.testing.assert_close(high * degree, counts_high.to(torch.float32))
