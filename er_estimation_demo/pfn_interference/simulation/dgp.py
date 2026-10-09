"""网络版 DGP V3：随机 f → Y=f+epsilon → 现有聚合公式 → 三种效应。

f 的输入为 [X_i, mean_neighbor(X)_i, t, e]。每个任务抽一次 f；
同一任务内所有个体、观测结果和反事实查询共用它。只有输出端加 epsilon。
"""
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Callable

import numpy as np
import torch

from pfn_interference.estimands import conditional_count_weights, sample_majority_exposures
from pfn_interference.prior import _task_generator
from .outcomes import INPUT_NAMES, RandomOutcomeFunction, sample_outcome_function

VERSION = "causalfm_network_cepo_v3_no_iv"
OutcomeFunction = Callable[[np.ndarray], np.ndarray]


def _evaluate(outcome: OutcomeFunction, x, neighbor_x, t, e) -> np.ndarray:
    inputs = np.stack(np.broadcast_arrays(x, neighbor_x, t, e), axis=-1)
    value = np.asarray(outcome(inputs), dtype=np.float64)
    if value.shape != inputs.shape[:-1] or not np.isfinite(value).all():
        raise ValueError("outcome function must return one finite value per input row.")
    return value


def aggregate_effects(outcome: OutcomeFunction, *, x, neighbor_x, degree,
                      treatment_prob: float = .5) -> dict[str, np.ndarray]:
    """按已有分组和权重聚合，返回 mu_i(t,s)、三种个体效应及总体平均。

    s=0: k <= floor(d/2); s=1: k > floor(d/2).
    mu_i(t,s) = sum_k P(K=k | S=s) f_theta(X_i, neighbor_X_i, t, k/d).
    同一个体的加性 epsilon 在各潜在结果差值中抵消，因此真值直接用 f。
    """
    x, neighbor_x, degree = [np.asarray(v, dtype=np.float64) for v in (x, neighbor_x, degree)]
    if x.ndim != 1 or not x.size or neighbor_x.shape != x.shape or degree.shape != x.shape:
        raise ValueError("x, neighbor_x and degree must be nonempty vectors of the same shape.")
    if not all(np.isfinite(v).all() for v in (x, neighbor_x, degree)):
        raise ValueError("covariates and degrees must be finite.")
    if np.any(degree <= 0) or not np.array_equal(degree, np.round(degree)):
        raise ValueError("degrees must be positive integers.")
    if not math.isfinite(treatment_prob) or not 0 < treatment_prob < 1:
        raise ValueError("treatment_prob must be in (0,1).")

    mu = np.empty((x.size, 2, 2), dtype=np.float64)  # [个体, 自己的处理 t, 暴露组 s]
    for d in np.unique(degree.astype(np.int64)):
        selected = degree == d
        exposure = np.arange(d + 1, dtype=np.float64) / d
        values = _evaluate(outcome, x[selected, None, None], neighbor_x[selected, None, None],
                           np.arange(2)[None, :, None], exposure[None, None, :])
        for s in (0, 1):
            # 直接复用项目原来的权重，逐点求 f 后再加权。
            weights = conditional_count_weights(int(d), treatment_prob, s)
            mu[selected, :, s] = np.sum(values * weights[None, None, :], axis=-1)

    direct = mu[:, 1, 0] - mu[:, 0, 0]
    spillover = mu[:, 1, 1] - mu[:, 1, 0]
    total = mu[:, 1, 1] - mu[:, 0, 0]
    ite = np.stack((direct, spillover, total), axis=-1)
    return {"oracle_arm_means": mu, "oracle_ite": ite, "oracle_ate": ite.mean(axis=0)}


@dataclass
class GeneratedTask:
    batch: dict[str, torch.Tensor]
    outcome: OutcomeFunction
    metadata: dict


def generate_task(adjacency, *, task_id: int = 0, seed: int = 12345,
                  stream: str = "train", noise_sd: float = 4., treatment_prob: float = .5,
                  outcome: OutcomeFunction | None = None,
                  outcome_family: str = "mixture") -> GeneratedTask:
    """生成一个完整网络任务；batch 自带长度为 1 的任务维。

    默认 X~N(0,1), T~Bernoulli(p), epsilon~N(0,noise_sd**2)。
    outcome 可传入已知函数以核对公式；默认每任务抽样一个 CausalFM 式随机 f。
    queries/cepo_target 对应四个 majority-arm CEPO；噪声只进入 tokens 中的观测 Y。
    oracle_ite/oracle_ate 是聚合后的效应真值，供现有评估口径使用。
    """
    if not isinstance(task_id, Integral) or isinstance(task_id, bool) or not 0 <= task_id < 2**63:
        raise ValueError("task_id must be a nonnegative signed-64-bit integer.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be nonempty.")
    if not math.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("noise_sd must be finite and nonnegative.")
    if not math.isfinite(treatment_prob) or not 0 < treatment_prob < 1:
        raise ValueError("treatment_prob must be in (0,1).")
    graph = torch.as_tensor(adjacency).detach().cpu()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] < 2:
        raise ValueError("adjacency must be square with at least two nodes.")
    if not bool(((graph == 0) | (graph == 1)).all()) or not torch.equal(graph, graph.T):
        raise ValueError("adjacency must be binary and symmetric.")
    if bool(graph.diag().any()) or not bool((graph.sum(-1) > 0).all()):
        raise ValueError("adjacency must have no self-loops or isolated nodes.")
    graph = graph.to(torch.float32).clone()
    n = graph.shape[0]
    degree = graph.sum(-1)
    rng = lambda channel: _task_generator(int(seed), stream, int(task_id), channel)

    x = torch.randn(n, generator=rng("data/normal"))
    neighbor_x = torch.mv(graph, x) / degree
    treatment = (torch.rand(n, generator=rng("treatment")) < treatment_prob).float()
    exposure = torch.mv(graph, treatment) / degree
    noise = noise_sd * torch.randn(n, generator=rng("noise/realization"))
    outcome_seed = rng("outcome/causalfm").initial_seed() % 2**32
    custom_outcome = outcome is not None
    if outcome is None:
        outcome = sample_outcome_function(outcome_seed, family=outcome_family)

    xn, nxn = x.numpy(), neighbor_x.numpy()
    factual_mean = _evaluate(outcome, xn, nxn, treatment.numpy(), exposure.numpy())
    y_obs = torch.from_numpy(factual_mean).float() + noise

    truth = aggregate_effects(outcome, x=xn, neighbor_x=nxn, degree=degree.numpy(), treatment_prob=treatment_prob)
    mu = truth['oracle_arm_means']
    targets = mu.reshape(n, 4)  # 00,01,10,11

    tensors = {
        "tokens": torch.stack((x, treatment, y_obs, exposure, degree/(n-1)), -1),
        "queries": torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).expand(n,4,2).clone(),
        "cepo_target": torch.from_numpy(targets).float(),
        "design_treatment_prob": torch.tensor(treatment_prob),
        "adjacency": graph, "degree": degree, "x": x, "neighbor_x": neighbor_x,
        "y_obs": y_obs, "noise": noise,
        "observed_treatment": treatment, "observed_exposure": exposure,
        "majority_arm": (torch.round(exposure*degree) > torch.floor(degree/2)).long(),
        "task_id": torch.tensor(task_id, dtype=torch.long),
        **{key: torch.from_numpy(value).float() for key, value in truth.items()},
    }
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise FloatingPointError("DGP produced nonfinite tensors.")
    metadata = dict(version=VERSION, task_id=int(task_id), seed=int(seed), stream=stream,
                    n_units=n, treatment_prob=treatment_prob, noise_sd=noise_sd,
                    outcome_seed=None if custom_outcome else outcome_seed,
                    outcome_kind="custom" if custom_outcome else "causalfm_no_u_mixture",
                    outcome_family="custom" if custom_outcome else outcome.family,
                    formula="f_theta(X_i, mean_neighbor_X_i, t, e) + epsilon_i",
                    effect_order=["direct", "spillover", "total"],
                    arm_mean_axes=["unit", "treatment", "majority_arm"])
    return GeneratedTask({key: value.unsqueeze(0) for key, value in tensors.items()}, outcome, metadata)
