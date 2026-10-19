"""HyperSCI原始网络及原有设备适配；仅调整同文件内部引用。来源说明见文件末尾。"""

from __future__ import annotations


# ========================================================================
# utils
# ========================================================================

import torch
import numpy as np
import torch.nn.functional as F


def pdist(sample_1, sample_2, norm=2, eps=1e-05):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    (n_1, n_2) = (sample_1.size(0), sample_2.size(0))
    norm = float(norm)
    if norm == 2.0:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1.0 / norm)


def wasserstein(
    x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False
):
    """return W dist between x and y"""
    "distance matrix M"
    nx = x.shape[0]
    ny = y.shape[0]
    x = x.reshape(nx, -1)
    y = y.reshape(ny, -1)
    M = pdist(x, y)
    "estimate lambda and delta"
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, min(10.0 / (nx * ny), 0.99))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()
    "compute new distance matrix"
    Mt = M
    row = delta * M.new_ones(M[0:1, :].shape)
    col = torch.cat([delta * M.new_ones(M[:, 0:1].shape), M.new_zeros((1, 1))], 0)
    if cuda:
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)
    "compute marginal"
    a = torch.cat([p * M.new_ones((nx, 1)) / nx, (1 - p) * M.new_ones((1, 1))], 0)
    b = torch.cat([(1 - p) * M.new_ones((ny, 1)) / ny, p * M.new_ones((1, 1))], 0)
    "compute kernel"
    Mlam = eff_lam * Mt
    temp_term = M.new_ones(1) * 1e-06
    if cuda:
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a
    u = a
    for i in range(its):
        u = 1.0 / ainvK.matmul(b / torch.t(torch.t(u).matmul(K)))
        if cuda:
            u = u.to(device)
    v = b / torch.t(torch.t(u).matmul(K))
    if cuda:
        v = v.to(device)
    upper_t = u * (torch.t(v) * K).detach()
    E = upper_t * Mt
    D = 2 * torch.sum(E)
    if cuda:
        D = D.to(device)
    return (D, Mlam)


def get_hyperedge_attr(features, hyperedge_index, type="mean"):
    if type == "mean":
        hyperedge_attr = None
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]
        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        (unique_labels, labels_count) = labels.unique(dim=0, return_counts=True)
        hyperedge_attr = torch.zeros_like(
            unique_labels, dtype=torch.float
        ).scatter_add_(0, labels, samples)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr


# ========================================================================
# model
# ========================================================================

from torch import nn
import torch_geometric.nn as gnn
import pfn_social._vendor.hypersci as utils

device = torch.device("cpu")


class HyperSCI(nn.Module):

    def __init__(self, args, x_dim):
        super(HyperSCI, self).__init__()
        self.h_dim = args.h_dim
        self.n_out = args.n_out
        self.g_dim = args.g_dim
        self.dropout = args.dropout
        self.graph_model = args.graph_model
        self.encoder = args.encoder
        self.skip_type = args.skip
        self.num_gnn_layer = args.num_gnn_layer
        self.phi_layer = args.phi_layer
        self.activate = args.activate
        self.phi_x = (
            nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device))
            if self.phi_layer == 1
            else nn.Sequential(
                nn.Linear(x_dim, self.h_dim).to(device),
                nn.ReLU().to(device),
                nn.Linear(self.h_dim, self.h_dim).to(device),
            )
        )
        self.phi_x_p = nn.Sequential(
            nn.Linear(x_dim, self.h_dim).to(device),
            nn.ReLU().to(device),
            nn.Linear(self.h_dim, self.h_dim).to(device),
        )
        if self.encoder == "gat":
            self.hgnn = gnn.HypergraphConv(
                self.h_dim,
                self.g_dim,
                use_attention=True,
                heads=2,
                concat=False,
                dropout=0.5,
            ).to(device)
            self.hgnn_more = [
                gnn.HypergraphConv(
                    self.g_dim,
                    self.g_dim,
                    use_attention=True,
                    heads=2,
                    concat=False,
                    dropout=0.5,
                ).to(device)
                for i in range(self.num_gnn_layer - 1)
            ]
        elif self.encoder == "gcn":
            self.hgnn = gnn.HypergraphConv(self.h_dim, self.g_dim).to(device)
            self.hgnn_more = [
                gnn.HypergraphConv(self.g_dim, self.g_dim).to(device)
                for i in range(self.num_gnn_layer - 1)
            ]
        else:
            self.hgnn = nn.Sequential(nn.Linear(self.h_dim, self.g_dim).to(device))
            self.hgnn_more = [
                nn.Sequential(nn.Linear(self.g_dim, self.g_dim).to(device))
                for i in range(self.num_gnn_layer - 1)
            ]
        if self.skip_type == "123":
            self.y_rep_dim = x_dim + self.h_dim + self.g_dim
        elif self.skip_type == "23":
            self.y_rep_dim = self.h_dim + self.g_dim
        self.out_t00 = [
            nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device)
            for i in range(self.n_out)
        ]
        self.out_t10 = [
            nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device)
            for i in range(self.n_out)
        ]
        self.out_t01 = nn.Linear(self.y_rep_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.y_rep_dim, 1).to(device)

    def forward(self, features, treatments, hyperedge_index):
        phi_x = self.phi_x(features)
        phi_x_t = torch.mul(treatments.view(-1, 1), phi_x)
        phi_x_p = phi_x
        hyperedge_attr = None
        if self.encoder == "gat":
            hyperedge_attr = utils.get_hyperedge_attr(
                phi_x_t, hyperedge_index, type="mean"
            )
        if self.encoder == "gat" or self.encoder == "gcn":
            rep_hgnn = self.hgnn(
                x=phi_x_t,
                hyperedge_index=hyperedge_index,
                hyperedge_attr=hyperedge_attr,
            )
            for i in range(self.num_gnn_layer - 1):
                if self.activate:
                    rep_hgnn = F.relu(rep_hgnn)
                if self.encoder == "gat":
                    hyperedge_attr = utils.get_hyperedge_attr(
                        rep_hgnn, hyperedge_index, type="mean"
                    )
                rep_hgnn = self.hgnn_more[i](
                    x=rep_hgnn,
                    hyperedge_index=hyperedge_index,
                    hyperedge_attr=hyperedge_attr,
                )
        else:
            rep_hgnn = self.hgnn(phi_x_t)
            for i in range(self.num_gnn_layer - 1):
                if self.activate:
                    rep_hgnn = F.relu(rep_hgnn)
                rep_hgnn = self.hgnn_more[i](rep_hgnn)
        if self.activate:
            rep_hgnn = F.relu(rep_hgnn)
        rep_hgnn = F.dropout(rep_hgnn, self.dropout, training=self.training)
        if self.skip_type == "123":
            rep_post_0 = torch.cat(
                [features, torch.zeros_like(phi_x_p), rep_hgnn], dim=1
            )
            rep_post_1 = torch.cat([features, phi_x_p, rep_hgnn], dim=1)
        elif self.skip_type == "23":
            rep_post_0 = torch.cat([torch.zeros_like(phi_x_p), rep_hgnn], dim=1)
            rep_post_1 = torch.cat([phi_x_p, rep_hgnn], dim=1)
        if self.n_out == 0:
            y00 = rep_post_0
            y10 = rep_post_1
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post_0))
            y10 = F.relu(self.out_t10[i](rep_post_1))
        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)
        results = {"y1_pred": y1_pred, "y0_pred": y0_pred, "rep": phi_x}
        return results


"""# HyperSCI source provenance

Jing Ma et al., Learning Causal Effects on Hypergraphs, KDD 2022.
Repository: https://github.com/jma712/HyperSCI
Pinned commit: ba33ef07b16e2110faabd127bb510aed63e29ea1

model.py contains the HyperSCI class from Model.py, with imports narrowed,
utils made package-relative. Weights are initialized on CPU, then the adapter
moves the model to the assigned evaluation device.
The model class body is unchanged. Only the supported one-layer, n_out=0,
attention, skip=123 configuration is instantiated by our adapter.
utils.py contains pdist, wasserstein and get_hyperedge_attr from utils.py.
Small-sample guards: retain a two-dimensional group of size one instead of
squeeze(); cap dropout probability at .99 when n_treated*n_control <= 10.
For the N=1000 pilot these guards do not change the original computation.
The original Wasserstein estimator (including gradient behavior) is retained.
For MUSA evaluation, auxiliary Wasserstein tensors stay on the input device;
detached scalars no longer make an unnecessary CPU round trip.
This is a pair-edge / majority-estimand adaptation, not a replication of the
paper's datasets, default hyperparameters, or evaluation protocol.

Original Model.py SHA256: a5fc0efcaa45443faa11295554481bf478244a39b313fe5667923aa7423cb874
Original utils.py SHA256: e1c4b753b093d0596ae968a8cd63dd9ad74698d7dcd67298e054201bfacb6af3
PyG version used by the pilot and this release: 2.6.1.
"""
