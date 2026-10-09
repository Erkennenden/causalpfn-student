"""Epoch RMSE must pool squared errors before taking a square root."""
import math

import pytest
import torch

import pfn_interference.model as demo


class KnownErrorModel(torch.nn.Module):
    """A one-component GMM with known errors; real loss/backward/Adam still run."""
    def __init__(self):
        super().__init__()
        self.config = demo.ModelConfig(gmm_n_components=1)
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tokens, queries, adjacency):
        mu = tokens[..., :4].unsqueeze(-1) + self.offset
        return {"gmm_pi": torch.ones_like(mu), "gmm_mu": mu,
                "gmm_sigma": torch.ones_like(mu)}


@pytest.mark.parametrize("errors,batch_size,expected_mse", [
    ([1., 3.], 1, 5.),             # sqrt(5), not mean(1, 3) = 2
    ([1., 1., 3.], 2, 11. / 3.),  # last batch is shorter
    ([1., 3.], 2, 5.),             # batch grouping must not change metrics
])
def test_train_epoch_reports_pooled_rmse(errors, batch_size, expected_mse, monkeypatch):
    monkeypatch.setenv("PFN_TRAIN_INTERVALS", "0")
    count = len(errors)
    truth = torch.arange(1, count + 1, dtype=torch.float32).view(count, 1, 1).expand(-1, 1, 3)
    error = torch.tensor(errors).view(count, 1, 1) * torch.tensor([1., 2., 3.])
    tokens = torch.zeros(count, 1, 5)
    tokens[..., :4] = torch.cat((torch.zeros_like(truth[...,:1]),torch.zeros_like(truth[...,:1]),truth[...,:1],2*truth[...,:1]),-1) + torch.cat((torch.zeros_like(error[...,:1]),torch.zeros_like(error[...,:1]),error[...,:1],error[...,2:3]),-1)
    batch = {"tokens": tokens, "queries": torch.zeros(count, 1, 4, 2),
             "adjacency": torch.zeros(count, 1, 1), "cepo_target":torch.cat((torch.zeros_like(truth[...,:1]),torch.zeros_like(truth[...,:1]),truth[...,:1],2*truth[...,:1]),-1)}
    model = KnownErrorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
    result = demo.train_epoch(model, optimizer, batch, batch_size=batch_size,
        training_seed=0, epoch=1, device=torch.device("cpu"), return_metrics=True)
    mean_abs_truth = (count + 1) / 2.
    for factor, effect in enumerate(("direct", "spillover", "total"), 1):
        rmse = factor * math.sqrt(expected_mse)
        for alias in (effect, effect + "_effect"):
            assert result[alias + "_rmse"] == pytest.approx(rmse)
            assert result[alias + "_nrmse_pct"] == pytest.approx(100 * rmse / (mean_abs_truth*(2 if effect=="total" else 1)))
        assert result[effect + "_effect_mse"] == pytest.approx(factor ** 2 * expected_mse)
    assert result["effect_rmse_macro"] == pytest.approx(2 * math.sqrt(expected_mse))
    # Original GMM NLL remains the objective; the RMSE fix must not add MSE loss.
    expected_nll = .5 * expected_mse * (10 / 4) + .5 * math.log(2 * math.pi)
    assert result["total_loss"] == pytest.approx(expected_nll)
    assert result["cepo_gmm_nll_macro"] == pytest.approx(expected_nll)
    assert model.offset.item() == 0.
