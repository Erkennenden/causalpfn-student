from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pfn_social.e1_6_complete import (  # noqa: E402
    complete_adjacency,
    effects_from_arm_means,
    empirical_support,
    generate_complete_task,
    make_complete_observed_draw,
)
from pfn_social.intervals import predict_intervals  # noqa: E402
from predict import load_selected_model  # noqa: E402


def test_complete_adjacency_contract():
    adjacency = complete_adjacency(12)
    assert adjacency.dtype == np.uint8
    np.testing.assert_array_equal(adjacency, adjacency.T)
    np.testing.assert_array_equal(np.diag(adjacency), 0)
    np.testing.assert_array_equal(adjacency.sum(1), 11)


def test_complete_task_is_reproducible_and_truth_has_four_arms():
    first = generate_complete_task(0)
    second = generate_complete_task(0)
    np.testing.assert_array_equal(first.parent.adjacency, second.parent.adjacency)
    np.testing.assert_allclose(first.parent.truth, second.parent.truth)
    assert first.parent.truth.shape == (200, 4)
    assert effects_from_arm_means(first.parent.truth).shape == (200, 3)
    assert first.parent.setting.graph == "complete"
    assert first.parent.setting.degree == 199
    assert np.isfinite(first.parent.x).all()
    assert np.isfinite(first.parent.h).all()
    assert np.isfinite(first.parent.components).all()
    assert np.isfinite(first.parent.truth).all()


def test_factual_batch_keeps_truth_out_and_masks_are_nested():
    task = generate_complete_task(1)
    rows, _ = make_complete_observed_draw(task, 0)
    assert [row["m_observed"] for row in rows] == [20, 50, 100]
    assert set(rows[0]["batch"]) == {
        "x", "adjacency", "treatment", "observed_indices", "outcomes_observed"
    }
    small = rows[0]["batch"]["observed_indices"].numpy()
    medium = rows[1]["batch"]["observed_indices"].numpy()
    large = rows[2]["batch"]["observed_indices"].numpy()
    np.testing.assert_array_equal(small, medium[:20])
    np.testing.assert_array_equal(medium, large[:50])


def test_complete_graph_realizes_at_most_two_arms_but_population_has_all_four():
    task = generate_complete_task(2)
    rows, _ = make_complete_observed_draw(task, 3)
    for row in rows:
        batch = {key: value.numpy() for key, value in row["batch"].items()}
        support = empirical_support(batch)
        assert np.count_nonzero(support["population_arm_counts"]) <= 2
        assert support["population_support_mask"].all()
        np.testing.assert_allclose(
            support["population_arm_probabilities"].sum(), 1.0
        )


def test_frozen_j_returns_finite_ordered_95_percent_ate_intervals():
    task = generate_complete_task(0)
    rows, _ = make_complete_observed_draw(task, 0)
    inputs = {key: value.numpy() for key, value in rows[1]["batch"].items()}
    with np.errstate(all="ignore"):
        result = predict_intervals(load_selected_model(), **inputs, level=.95)
    assert result["ate_lower"].shape == (3,)
    assert result["ate_upper"].shape == (3,)
    assert np.isfinite(result["ate_lower"]).all()
    assert np.isfinite(result["ate_upper"]).all()
    assert np.all(result["ate_lower"] <= result["ate_upper"])
    assert float(result["interval_level"]) == .95
