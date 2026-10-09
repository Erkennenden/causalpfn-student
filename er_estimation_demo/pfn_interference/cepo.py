"""Exact majority-arm CEPO supervision and coherent effect point estimates.

The four marginal GMMs describe mu(0,0), mu(0,1), mu(1,0), mu(1,1). They do not
specify a joint posterior; no effect intervals are inferred from them.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pfn_interference.estimands import arm_mean_exposure

PREDICTION_PROTOCOL = 'majority_cepo_four_gmm_v2'
ARM_NAMES = ('mu00', 'mu01', 'mu10', 'mu11')
ARM_COORDINATES = ((0., 0.), (0., 1.), (1., 0.), (1., 1.))


def cepo_queries(batch):
    tokens = batch['tokens']
    return tokens.new_tensor(ARM_COORDINATES).view(1, 1, 4, 2).expand(*tokens.shape[:2], 4, 2)


def cepo_targets(batch, treatment_prob=None):
    """Noise-free E[Y_i(d,E_i)|S_i=s], averaging exact conditional counts.

    Training attaches this tensor once using the saved treatment design before
    sharding mini-batches. Oracle fields are labels only, never model inputs.
    """
    if 'cepo_target' in batch:
        target = batch['cepo_target']
        if target.ndim != 3 or target.shape[-1] != 4:
            raise ValueError('cepo_target must have shape [batch, units, 4] in order 00,01,10,11')
        return target
    if 'tokens' in batch and batch['tokens'].shape[0] == 0:
        return batch['tokens'].new_empty((0, batch['tokens'].shape[1], 4))
    design = batch.get('design_treatment_prob')
    if design is not None:
        design = design.detach().cpu().to(torch.float64)
        if design.numel() == 0 or not bool(torch.isfinite(design).all()):
            raise ValueError('Invalid design treatment probability metadata')
        inferred = float(design.flatten()[0])
        if not bool(torch.allclose(design, torch.full_like(design, inferred))):
            raise ValueError('A CEPO batch must have one treatment design probability')
        if treatment_prob is not None and not np.isclose(treatment_prob, inferred, rtol=1e-7, atol=1e-8):
            raise ValueError('Explicit treatment probability disagrees with batch design')
        if treatment_prob is None:
            treatment_prob = inferred
    if treatment_prob is None:
        raise ValueError('CEPO targets require explicit treatment_prob or batch design metadata; '
                         'attach targets using the saved DataConfig before standalone training/evaluation')
    if not np.isfinite(treatment_prob) or not 0 < treatment_prob < 1:
        raise ValueError('CEPO treatment probability must lie strictly between zero and one')
    degree = batch['degree']
    d_cpu = degree.detach().cpu().numpy()
    if np.any(d_cpu <= 0) or not np.allclose(d_cpu, np.rint(d_cpu)):
        raise ValueError('CEPO targets require positive integer degree')
    d_cpu = np.rint(d_cpu).astype(np.int64)
    low = np.empty(d_cpu.shape, np.float64)
    high = np.empty_like(low)
    for d in np.unique(d_cpu):
        mask = d_cpu == d
        low[mask] = arm_mean_exposure(int(d), treatment_prob, 0)
        high[mask] = arm_mean_exposure(int(d), treatment_prob, 1)
    low = torch.as_tensor(low, dtype=degree.dtype, device=degree.device)
    high = torch.as_tensor(high, dtype=degree.dtype, device=degree.device)
    baseline, tau, gamma = (batch[k] for k in ('structural_baseline', 'tau', 'gamma'))
    eta = batch.get('eta', torch.zeros_like(tau))
    return torch.stack((baseline + gamma*low,
                        baseline + gamma*high,
                        baseline + tau + (gamma+eta)*low,
                        baseline + tau + (gamma+eta)*high), dim=-1)


def effects_from_mu(mu):
    if mu.ndim < 2 or mu.shape[-1] != 4:
        raise ValueError('CEPO means must end in four arms: 00,01,10,11')
    return torch.stack((mu[..., 2]-mu[..., 0], mu[..., 3]-mu[..., 2],
                        mu[..., 3]-mu[..., 0]), dim=-1)


def effect_targets(batch):
    return batch['oracle_ite'] if 'oracle_ite' in batch else effects_from_mu(cepo_targets(batch))


def compute_cepo_losses(predictions, batch):
    from pfn_interference.model import gmm_nll_loss
    pi, mu, sigma = (predictions[k] for k in ('gmm_pi', 'gmm_mu', 'gmm_sigma'))
    target = cepo_targets(batch)
    if pi.shape != mu.shape or pi.shape != sigma.shape or pi.shape[:-1] != target.shape:
        raise ValueError('Each of four CEPO targets requires matching GMM pi, mu and sigma components')
    losses = {f'{name}_gmm_nll': gmm_nll_loss(pi[..., j, :], mu[..., j, :],
              sigma[..., j, :], target[..., j]) for j, name in enumerate(ARM_NAMES)}
    losses['cepo_gmm_nll_macro'] = torch.stack(list(losses.values())).mean()
    losses['total_loss'] = losses['cepo_gmm_nll_macro']
    return losses


def compute_cepo_metrics(predictions, batch, *, interval_mass=.9, include_intervals=True):
    from pfn_interference.model import (gmm_posterior_mean, gmm_posterior_interval,
                                                compute_regression_metrics)
    means = gmm_posterior_mean(predictions['gmm_pi'], predictions['gmm_mu'])
    targets = cepo_targets(batch)
    metrics = {k: float(v.detach().item()) for k, v in compute_cepo_losses(predictions, batch).items()}
    metrics['validation_gmm_nll'] = metrics['cepo_gmm_nll_macro']
    if include_intervals:
        lo, hi = gmm_posterior_interval(predictions['gmm_pi'], predictions['gmm_mu'],
                                        predictions['gmm_sigma'], mass=interval_mass)
    for j, name in enumerate(ARM_NAMES):
        reg = compute_regression_metrics(means[..., j], targets[..., j])
        for key in ('mae', 'mse', 'rmse', 'bias'):
            metrics[f'{name}_{key}'] = reg[key]
        if include_intervals:
            pct = round(interval_mass*100)
            metrics[f'{name}_coverage_{pct}'] = float(((targets[...,j]>=lo[...,j]) & (targets[...,j]<=hi[...,j])).float().mean())
            metrics[f'{name}_interval_width_{pct}'] = float((hi[...,j]-lo[...,j]).mean())
    effects = effects_from_mu(means)
    truth = effect_targets(batch)
    for j, name in enumerate(('direct', 'spillover', 'total')):
        reg = compute_regression_metrics(effects[...,j], truth[...,j])
        for prefix in (name, f'{name}_effect'):
            metrics.update({f'{prefix}_{k}': v for k,v in reg.items()})
        mean_truth, mean_pred = truth[...,j].mean(1), effects[...,j].mean(1)
        metrics[f'average_{name}_effect_true'] = float(mean_truth.mean())
        metrics[f'average_{name}_effect_pred'] = float(mean_pred.mean())
        metrics[f'average_{name}_effect_absolute_error'] = float((mean_truth-mean_pred).abs().mean())
    metrics['effect_rmse_macro'] = sum(metrics[f'{r}_rmse'] for r in ('direct','spillover','total'))/3
    return metrics


def predict_mu_distributions(model, batch):
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            return model(batch['tokens'], cepo_queries(batch), batch['adjacency'])
    finally:
        model.train(was_training)


def mu_prediction_rows(predictions, batch, treatment_prob=None):
    """Losslessly export each node/arm marginal, plus its mean and oracle label."""
    distribution = {key: predictions[key].detach().cpu()
                    for key in ('gmm_pi', 'gmm_mu', 'gmm_sigma')}
    truth = cepo_targets(batch, treatment_prob).detach().cpu()
    means = (distribution['gmm_pi'] * distribution['gmm_mu']).sum(-1)
    if means.shape != truth.shape:
        raise ValueError('GMM export requires one distribution per node and CEPO arm')
    rows = []
    for b in range(means.shape[0]):
        for i in range(means.shape[1]):
            for j, arm in enumerate(ARM_NAMES):
                row = dict(dataset_id=b+1, unit_id=i+1, arm=arm,
                           truth=float(truth[b,i,j]), estimate=float(means[b,i,j]))
                for key, value in distribution.items():
                    row[key] = json.dumps(value[b,i,j].tolist())
                rows.append(row)
    return rows
