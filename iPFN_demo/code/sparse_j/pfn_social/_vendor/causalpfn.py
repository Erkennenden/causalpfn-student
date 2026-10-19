"""CausalPFN模型与原公式预处理，含MUSA数值兼容适配。见LICENSE.causalpfn。"""

from __future__ import annotations


# ========================================================================
# transformer layer
# ========================================================================

import math
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        norm = lambda x: RMSNorm(x)
        self.attn_norm = norm(embed_dim)
        self.ff_norm = norm(embed_dim)
        self.q_norm = norm(self.head_dim)
        self.k_norm = norm(self.head_dim)

    def forward(self, x, context_length):
        x = x.transpose(0, 1)
        (B, L, _) = x.size()
        h = self.attn_norm(x)
        (q, ff_h, ff_gate) = self.q_proj(h).chunk(3, dim=-1)
        (k, v) = self.kv_proj(h[:, :context_length]).chunk(2, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        (q, k) = (self.q_norm(q), self.k_norm(k))
        q = q * math.log2(context_length) / 10
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = attn.reshape(B, L, self.num_heads * self.head_dim)
        ff_x = ff_h * F.silu(ff_gate)
        residual = self.ff_norm(self.out_proj(attn + ff_x))
        x_out = x + residual
        return x_out.transpose(0, 1)


# ========================================================================
# model
# ========================================================================

from enum import Enum
import numpy as np
import torch.nn as nn


class Task(Enum):
    REG = 1
    CLS = 2


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos):
    # torch_musa 2.7.1 produced NaNs for finite, mixed constant/variable
    # columns in this reduction. Keep the upstream formula on CPU and return
    # to the original device; trainable layers and gradients stay unchanged.
    device = data.device
    if device.type in {"musa", "privateuseone"}:
        data = data.to("cpu")
    if bool(torch.isinf(data).any()):
        raise FloatingPointError("Infinite covariates before normalization")
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-06
    data = (data - mean) / std
    if bool(torch.isinf(data).any()):
        raise FloatingPointError("Infinite covariates after normalization")
    return data.to(device)


def clip_outliers(data, eval_pos, n_sigma=4):
    assert len(data.shape) == 3, "X must be T,B,H"
    # On the same MUSA runtime, NaN-bearing inputs acquired Infs during
    # clipping. Use the unchanged CPU calculation, allowing the upstream
    # missing-value NaNs but rejecting Infs before nan_to_num can hide them.
    device = data.device
    if device.type in {"musa", "privateuseone"}:
        data = data.to("cpu")
    if bool(torch.isinf(data).any()):
        raise FloatingPointError("Infinite covariates before outlier clipping")
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    data = torch.clip(data, mean - cutoff, mean + cutoff)
    if bool(torch.isinf(data).any()):
        raise FloatingPointError("Infinite covariates after outlier clipping")
    return data.to(device)


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class TabDPTLongContextModel(nn.Module):

    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        nbins=int,
    ):
        super().__init__()
        self.n_out = n_out
        self.ninp = ninp
        self.nbins = nbins
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim=ninp, num_heads=nhead, ff_dim=nhid)
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp, bias=True)
        self.head = nn.Sequential(
            nn.Linear(ninp, nhid, bias=False),
            nn.GELU(),
            nn.Linear(nhid, n_out + nbins, bias=False),
        )
        self.xnorm = nn.LayerNorm(ninp, bias=False)
        self.ynorm = nn.LayerNorm(ninp, bias=False)

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        return_log_act_norms: bool = False,
    ) -> torch.Tensor:
        context_length = y_src.shape[0]
        B = y_src.shape[1]
        x_src = normalize_data(x_src, -1 if self.training else context_length)
        x_src = clip_outliers(
            x_src, -1 if self.training else context_length, n_sigma=10
        )
        x_src = torch.nan_to_num(x_src, nan=0)
        x_src = self.xnorm(self.encoder(x_src))
        y_src = self.ynorm(self.y_encoder(y_src.unsqueeze(-1)))
        train_x = x_src[:context_length] + y_src
        src = torch.cat([train_x, x_src[context_length:]], 0)
        log_act_norms = {}
        log_act_norms["y"] = torch.norm(y_src, dim=-1).mean()
        log_act_norms["x"] = torch.norm(x_src, dim=-1).mean()
        for l, layer in enumerate(self.transformer_encoder):
            if l in [0, 1, 3, 6, 9]:
                log_act_norms[f"layer_{l}"] = torch.norm(src, dim=-1).mean()
            src = layer(src, context_length)
        pred = self.head(src)
        pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))
        if return_log_act_norms:
            return (pred[context_length:], log_act_norms)
        else:
            return pred[context_length:]

    @classmethod
    def load(cls, model_state, config):
        assert config.model.max_num_classes > 2
        model = TabDPTLongContextModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            nbins=config.model.nbins,
        )
        module_prefix = "_orig_mod."
        model_state = {
            k.replace(module_prefix, ""): v for (k, v) in model_state.items()
        }
        model.load_state_dict(model_state)
        model.eval()
        return model


# ========================================================================
# utils
# ========================================================================


def pad_x(X: torch.Tensor, num_features=100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros(
        (*X.shape[:-1], num_features - n_features), device=X.device
    )
    return torch.cat([X, zero_feature_padding], dim=-1)


# ========================================================================
# loading
# ========================================================================


class DictToObject:

    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for (k, v) in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for (k, v) in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    else:
        return obj


def load_pretrained_tabdpt_model(
    ckpt_path: str | None, ckpt: dict | None = None
) -> TabDPTLongContextModel:
    assert (
        ckpt_path is not None or ckpt is not None
    ), "Either ckpt_path or ckpt must be provided."
    checkpoint = (
        torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ckpt is None
        else ckpt
    )
    config = DictToObject(checkpoint["cfg"])
    model = TabDPTLongContextModel.load(model_state=checkpoint["model"], config=config)
    return model


def load_pretrained_tabdpt_config(
    ckpt_path: str | None, ckpt: dict | None = None
) -> dict:
    assert (
        ckpt_path is not None or ckpt is not None
    ), "Either ckpt_path or ckpt must be provided."
    checkpoint = (
        torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ckpt is None
        else ckpt
    )
    return object_to_dict(checkpoint["cfg"])


# ========================================================================
# icl model
# ========================================================================

from typing import Tuple


class InContextModel(nn.Module):

    def __init__(
        self,
        model: TabDPTLongContextModel,
        model_config: dict,
        sigma: float = 0.5,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ):
        super().__init__()
        self.model: nn.Module = model
        self.model_config = model_config
        self.prepare_input = lambda x, y: (pad_x(x, model.num_features), y)
        self.nbins = model_config["model"]["nbins"]
        model_config["model_type"] = "tabdpt"
        model_config["sigma"] = sigma
        self.sigma = sigma
        self.vmin = vmin
        self.vmax = vmax
        bin_edges = torch.linspace(self.vmin, self.vmax, self.nbins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + 0.5 * bin_width
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_width", bin_width)
        self.register_buffer("bin_centers", bin_centers)

    def _predict_mean(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_centers, dim=-1)

    def _sample_from_logits(self, logits: torch.Tensor, n_samples: int) -> torch.Tensor:
        orig_shape = logits.shape[:-1]
        probs = torch.softmax(logits, dim=-1)
        logits_reshaped = probs.reshape(-1, self.nbins)
        sampled_indices = torch.multinomial(
            logits_reshaped, num_samples=n_samples, replacement=True
        )
        samples = self.bin_centers[sampled_indices]
        samples = samples.view(*orig_shape, n_samples)
        return samples

    def _hl_gaussian_cross_entropy_loss(
        self, logits: torch.Tensor, y_target: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """
        Calculate the cross-entropy loss between the predicted distribution and the target tensor.
        Args:
        logits: Tensor of shape (..., nbins) containing the predicted distribution
        y_target: Tensor of shape (...) containing target distribution
        vmin: Minimum value of the distribution
        vmax: Maximum value of the distribution
        sigma: Standard deviation for the Gaussian smoothing

        Note: This function construct a soft target distribution (one per element in y_target)
        by integrating a Gaussian centered at y_i over each bin. (Histogram Loss Gaussian or HL-Gauss)
        """
        assert sigma > 0, "Sigma must be positive."
        assert (
            y_target.shape == logits.shape[:-1]
        ), "y_target must have the same shape as logits except for the last dimension."
        y_target_expanded = y_target.unsqueeze(-1)

        def normal_cdf(x, mean, std):
            return 0.5 * (1.0 + torch.erf((x - mean) / (math.sqrt(2) * std)))

        lower_edges = self.bin_centers - 0.5 * self.bin_width
        upper_edges = self.bin_centers + 0.5 * self.bin_width
        cdf_upper = normal_cdf(upper_edges, y_target_expanded, sigma)
        cdf_lower = normal_cdf(lower_edges, y_target_expanded, sigma)
        p = cdf_upper - cdf_lower
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-08)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -torch.sum(p * log_probs, dim=-1).mean(dim=-1)
        return ce_loss

    def _get_y0_y1_shift_scale(self, t_context: torch.Tensor, y_context: torch.Tensor):
        treated_mask = (t_context == 1).long()
        control_mask = (t_context == 0).long()
        treated_counts = treated_mask.sum(dim=1, keepdim=True)
        control_counts = control_mask.sum(dim=1, keepdim=True)
        y1_mean = (y_context * treated_mask).sum(dim=1, keepdim=True) / treated_counts
        y0_mean = (y_context * control_mask).sum(dim=1, keepdim=True) / control_counts
        y1_std = (
            torch.sqrt(
                ((y_context - y1_mean) ** 2 * treated_mask).sum(dim=1, keepdim=True)
                / treated_counts
            )
            + 1e-20
        )
        y0_std = (
            torch.sqrt(
                ((y_context - y0_mean) ** 2 * control_mask).sum(dim=1, keepdim=True)
                / control_counts
            )
            + 1e-20
        )
        y0_scale = torch.where(control_counts > 0, y0_std, y1_std)
        y1_scale = torch.where(treated_counts > 0, y1_std, y0_std)
        y0_shift = torch.where(control_counts > 0, y0_mean, y1_mean)
        y1_shift = torch.where(treated_counts > 0, y1_mean, y0_mean)
        return (y0_shift, y0_scale, y1_shift, y1_scale)

    def cepo_losses(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        E_y0_query: torch.Tensor,
        E_y1_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        sigma = sigma or self.sigma
        (y0_shift, y0_scale, y1_shift, y1_scale) = self._get_y0_y1_shift_scale(
            t_context, y_context
        )
        y_standardized = torch.where(
            t_context == 1,
            (y_context - y1_shift) / y1_scale,
            (y_context - y0_shift) / y0_scale,
        )
        E_y1_standardized = (E_y1_query - y1_shift) / y1_scale
        E_y0_standardized = (E_y0_query - y0_shift) / y0_scale
        random_treatments = torch.randint(
            0, 2, E_y0_query.shape, device=E_y0_query.device
        )
        x_and_t_context = torch.cat([t_context.unsqueeze(-1), X_context], dim=2)
        x_and_t_query = torch.cat([random_treatments.unsqueeze(-1), X_query], dim=2)
        Ey_target = torch.where(
            random_treatments == 0, E_y0_standardized, E_y1_standardized
        )
        x_and_t = torch.cat([x_and_t_context, x_and_t_query], dim=1)
        (x_src, y_src) = self.prepare_input(x_and_t, y_standardized)
        logits = self.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
            0, 1
        )
        logits = logits[:, :, -self.model.nbins :]
        logits /= temperature
        return self._hl_gaussian_cross_entropy_loss(
            logits=logits, y_target=Ey_target, sigma=sigma
        )

    def forward(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        E_y0_query: torch.Tensor,
        E_y1_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ):
        """
        The forward method will simply call the cepo_losses method and returns the loss for training.
        This is done to support multi-GPU training.
        """
        return self.cepo_losses(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=X_query,
            E_y0_query=E_y0_query,
            E_y1_query=E_y1_query,
            sigma=sigma,
            temperature=temperature,
        )

    def predict_cepo(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        t_query: torch.Tensor,
        temperature: torch.Tensor,
        n_samples: int | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        (y0_shift, y0_scale, y1_shift, y1_scale) = self._get_y0_y1_shift_scale(
            t_context, y_context
        )
        y_standardized = torch.where(
            t_context == 1,
            (y_context - y1_shift) / y1_scale,
            (y_context - y0_shift) / y0_scale,
        )
        x_and_t_context = torch.cat([t_context.unsqueeze(-1), X_context], dim=2)
        x_and_t_query = torch.cat([t_query.unsqueeze(-1), X_query], dim=2)
        x_and_t = torch.cat([x_and_t_context, x_and_t_query], dim=1)
        (x_src, y_src) = self.prepare_input(x_and_t, y_standardized)
        logits = self.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
            0, 1
        )
        logits = logits[:, :, -self.model.nbins :]
        logits = logits.unsqueeze(1)
        (y1_scale, y1_shift) = (y1_scale.unsqueeze(1), y1_shift.unsqueeze(1))
        (y0_scale, y0_shift) = (y0_scale.unsqueeze(1), y0_shift.unsqueeze(1))
        t_query = t_query.unsqueeze(1)
        temperature = temperature[None, :, None, None]
        logits = logits / temperature
        mean = self._predict_mean(logits)
        mean_shift_scaled = torch.where(
            t_query == 1, mean * y1_scale + y1_shift, mean * y0_scale + y0_shift
        )
        if n_samples is None:
            return mean_shift_scaled
        samples = self._sample_from_logits(logits, n_samples)
        samples_shift_scaled = torch.where(
            t_query.unsqueeze(-1) == 1,
            samples * y1_scale + y1_shift,
            samples * y0_scale + y0_shift,
        )
        return (mean_shift_scaled, samples_shift_scaled)

    @classmethod
    def load(cls, model_state: dict, model_config: dict) -> "InContextModel":
        inner_model_state = {
            k.replace("model.", ""): v
            for (k, v) in model_state.items()
            if k.startswith("model.")
        }
        if model_config["model_type"] == "tabdpt":
            ckpt_loaded = {"cfg": model_config, "model": inner_model_state}
            base_model = load_pretrained_tabdpt_model(ckpt_path=None, ckpt=ckpt_loaded)
        else:
            raise ValueError(f"Unknown model. Supported model is 'tabdpt'.")
        sigma = model_config.get("sigma", 0.5)
        return InContextModel(model=base_model, model_config=model_config, sigma=sigma)
