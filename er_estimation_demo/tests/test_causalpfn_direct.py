from __future__ import annotations

import sys


def test_vendored_causalpfn_model_imports_without_optional_estimator_dependencies() -> None:
    from third_party.causalpfn.models import InContextModel

    assert InContextModel is not None
    assert "faiss" not in sys.modules
    assert "huggingface_hub" not in sys.modules

from pathlib import Path

import torch

from baselines.causalpfn_direct import load_causalpfn_checkpoint


def test_local_checkpoint_loads_and_predicts_finite_cepo() -> None:
    checkpoint = Path("checkpoints/causalpfn_v0.pt")
    assert checkpoint.exists()
    model = load_causalpfn_checkpoint(checkpoint, torch.device("cpu"))

    X_context = torch.tensor(
        [[[0.0, 0.00], [0.2, 0.25], [-0.3, 0.50], [0.5, 0.75]]],
        dtype=torch.float32,
    )
    t_context = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    y_context = torch.tensor([[0.0, 0.8, -0.2, 1.1]], dtype=torch.float32)
    X_query = torch.tensor([[[0.1, 0.25], [0.1, 0.25]]], dtype=torch.float32)
    t_query = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with torch.inference_mode():
        prediction = model.predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=X_query,
            t_query=t_query,
            temperature=torch.tensor([1.0]),
        )

    assert prediction.shape == (1, 1, 2)
    assert torch.isfinite(prediction).all()

from baselines.causalpfn_direct import predict_causalpfn_direct_on_batch


class CovariateSurface:
    model_config = {}

    def eval(self):
        return self

    def predict_cepo(self, *, X_context, t_context, y_context, X_query,
                     t_query, temperature, n_samples=None):
        # A second input feature changes the result, exposing an accidental
        # return to the removed exposure-augmented inference path.
        x = X_query[..., 0]
        extra = X_query[..., 1:].sum(dim=-1)
        return (2 * x + t_query * (1.25 + .75 * x + extra)).unsqueeze(1)


def test_direct_adapter_uses_covariates_only_for_every_graph_and_averages_all_nodes():
    batch = {
        "x": torch.tensor([[[0.], [1.]], [[2.], [3.]]]),
        "observed_exposure": torch.tensor([[.25, .75], [.5, 1.]]),
        "observed_treatment": torch.tensor([[0., 1.], [1., 0.]]),
        "y_obs": torch.tensor([[0., 2.5], [4., 6.]]),
        "degree": torch.tensor([[4., 3.], [2., 4.]]),
    }
    result = predict_causalpfn_direct_on_batch(
        model=CovariateSurface(), batch=batch, treatment_prob=.5, query_chunk_size=2)
    torch.testing.assert_close(result.ite_direct,
                               torch.tensor([[1.25, 2.], [2.75, 3.5]]))
    torch.testing.assert_close(result.ate_direct,
                               torch.tensor([1.625, 3.125], dtype=torch.float64))


def test_direct_adapter_does_not_need_graph_exposure_or_truth():
    batch = {
        "x": torch.tensor([[0., 1.]]),
        "observed_treatment": torch.tensor([[0., 1.]]),
        "y_obs": torch.tensor([[0., 2.5]]),
    }
    result = predict_causalpfn_direct_on_batch(
        model=CovariateSurface(), batch=batch, treatment_prob=.5, query_chunk_size=2)
    torch.testing.assert_close(result.ite_direct, torch.tensor([[1.25, 2.]]))
