"""X-only S-learner inference using the project's loaded local checkpoint.

Adapted from CausalPFN 0.1.4's causal_estimator.py, under its bundled license; see
third_party/causalpfn/LICENSE and CAUSALPFN_LOCAL_CHECKPOINT_CN.md.
The module name is retained so existing C and unified evaluation entries work.
No causalpfn, faiss or huggingface_hub package, download, or PFN training is used.
The factual-data gradient-boosted retrieval learner is retained. NumPy performs
the one-dimensional exact nearest-neighbour search; distance ties use input
index order. This is a local implementation, not a bitwise FAISS guarantee.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np
import torch

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError as exc:
    raise ImportError(
        "Local CausalPFN retrieval requires scikit-learn; "
        "the causalpfn package and another checkpoint download are not needed."
    ) from exc


IMPLEMENTATION = "local_checkpoint_x_only_s_v1"


def _nearest_indices(context_effects, query_effects, k):
    """Exact 1D squared-L2 neighbours, float32 scores as in the upstream index.

    Work in blocks to avoid an unbounded query-by-context distance allocation.
    Stable sorting makes equally distant neighbours reproducible.
    """
    context = np.asarray(context_effects, dtype=np.float32).reshape(1, -1)
    queries = np.asarray(query_effects, dtype=np.float32)
    neighbours = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 256):
        distance = (queries[start:start + 256, None] - context) ** 2
        neighbours[start:start + 256] = np.argsort(
            distance, axis=1, kind="stable"
        )[:, :k]
    return neighbours


def _stratum_end(neighbours, query_order, start, max_query_length):
    # Preserve 0.1.4's query-window selection, including its endpoint convention.
    left = start + 1
    right = min(start + max_query_length, len(query_order))
    while right > left + 1:
        middle = (left + right) // 2
        indices = query_order[start:middle]
        if len(np.unique(neighbours[indices].reshape(-1))) > 2048:
            right = middle - 1
        else:
            left = middle
    return min(left, len(query_order))


def estimate_x_only(model, X, t, y, *, device, max_query_length):
    """Return mu0(X), mu1(X), mean(mu1-mu0) from frozen local PFN weights.

    Inputs are one graph's factual X, treatment and observed outcome only.
    The model argument is supplied by load_causalpfn_checkpoint(), which reads
    checkpoints/causalpfn_v0.pt using third_party.causalpfn.models.InContextModel.
    Each call fits a fresh auxiliary learner; nothing is cached across graphs.
    """
    if not isinstance(max_query_length, Integral) or max_query_length <= 0:
        raise ValueError("max_query_length must be a positive integer")
    X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
    if (X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0
            or t.shape != (len(X),) or y.shape != (len(X),)):
        raise ValueError("Expected nonempty X=[N,P], t=[N], y=[N]")
    if not all(np.isfinite(value).all() for value in (X, t, y)):
        raise ValueError("Factual X, t and y must be finite")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("Treatment must be binary (0/1)")
    if not ((t == 0).any() and (t == 1).any()):
        raise ValueError("CausalPFN requires both treatment groups")
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)

    config = model.model_config
    if config.get("model_type") == "tabdpt":
        max_features = config["model"]["max_num_features"] - 1
        if X.shape[1] > max_features:
            X = TruncatedSVD(n_components=max_features, algorithm="arpack").fit_transform(X)

    # For valid binary treatment, [1-t,t] is exactly the official one-hot
    # representation, including its float64 dtype. No encoder version pin needed.
    treatment_column = t.astype(np.float64).reshape(-1, 1)
    features = np.concatenate((X, 1 - treatment_column, treatment_column), axis=1)
    stratifier = GradientBoostingRegressor(
        n_estimators=100, max_depth=6,
        min_samples_leaf=max(1, int(len(X) / 100)), random_state=111,
    )
    stratifier.fit(features, y)
    zeros = np.zeros((len(X), 1), dtype=np.float64)
    ones = np.ones_like(zeros)
    effects = (
        stratifier.predict(np.concatenate((X, zeros, ones), axis=1))
        - stratifier.predict(np.concatenate((X, ones, zeros), axis=1))
    )

    control = np.flatnonzero(t == 0)
    treated = np.flatnonzero(t == 1)
    k = min(1024, len(control), len(treated))
    X_query = np.concatenate((X, X), axis=0)
    t_query = np.concatenate((np.zeros(len(X), dtype=X.dtype),
                              np.ones(len(X), dtype=X.dtype)))
    query_effects = np.concatenate((effects, effects))
    query_order = np.argsort(query_effects)
    control_neighbours = _nearest_indices(effects[control], query_effects, k)
    treated_neighbours = _nearest_indices(effects[treated], query_effects, k)
    means = np.empty(len(X_query), dtype=X.dtype)
    device = torch.device(device)

    model.eval()
    with torch.inference_mode():
        temperature = torch.tensor([1.0], device=device)
        start = 0
        while start < len(X_query):
            control_end = _stratum_end(control_neighbours, query_order, start, max_query_length)
            treated_end = _stratum_end(treated_neighbours, query_order, start, max_query_length)
            control_indices = control[np.unique(
                control_neighbours[query_order[start:control_end]].reshape(-1))]
            treated_indices = treated[np.unique(
                treated_neighbours[query_order[start:treated_end]].reshape(-1))]
            context_indices = np.concatenate((control_indices, treated_indices))
            if len(context_indices) > 4096:
                raise ValueError("Retrieved context exceeds 4096 rows")
            end = min(control_end, treated_end)
            query_indices = query_order[start:end]
            prediction = model.predict_cepo(
                X_context=torch.from_numpy(X[context_indices]).to(device).unsqueeze(0).float(),
                t_context=torch.from_numpy(t[context_indices]).to(device).unsqueeze(0).float(),
                y_context=torch.from_numpy(y[context_indices]).to(device).unsqueeze(0).float(),
                X_query=torch.from_numpy(X_query[query_indices]).to(device).unsqueeze(0).float(),
                t_query=torch.from_numpy(t_query[query_indices]).to(device).unsqueeze(0).float(),
                temperature=temperature, n_samples=None,
            )
            if not isinstance(prediction, torch.Tensor) or prediction.shape != (1, 1, len(query_indices)):
                raise ValueError("Local checkpoint returned an unexpected prediction shape")
            means[query_indices] = prediction[0, 0].detach().cpu().numpy()
            start = end

    if not np.isfinite(means).all():
        raise ValueError("Local checkpoint returned non-finite predictions")
    mu0, mu1 = means[:len(X)], means[len(X):]
    return mu0, mu1, float((mu1 - mu0).mean())
