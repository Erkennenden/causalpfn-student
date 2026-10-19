import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ============================================================================
# Graph Convolution Layers
# ============================================================================

class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        return self.dropout(self.conv(x, edge_index))


class GINLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.gin = GINConv(self.mlp)

    def forward(self, x, edge_index):
        return self.gin(x, edge_index)


class GATLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0, heads=1):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = F.elu(x)
        return self.dropout(x)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0, aggr='mean'):
        super(GraphSAGELayer, self).__init__()
        self.sage = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        x = F.relu(x)
        return self.dropout(x)


# Registry of GNN layers usable as drop-in (in_channels, hidden_channels, dropout)
# blocks. All entries must accept the same constructor signature and produce
# output of dimension `hidden_channels`. Used by HINet / HINet_no_net_conf via
# the `gnn_layer` config setting (GNN architecture ablation).
GNN_LAYER_REGISTRY = {
    "GIN": GINLayer,
    "GAT": GATLayer,
    "GCN": GCNLayer,
    "GraphSAGE": GraphSAGELayer,
}


def get_gnn_layer(name):
    """Look up a GNN layer class by name. Raises a clear error on typo."""
    try:
        return GNN_LAYER_REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(GNN_LAYER_REGISTRY))
        raise ValueError(
            f"Unknown gnn_layer={name!r}. Valid options: {valid}."
        )


# ============================================================================
# Gradient Reversal Layer (for domain adversarial training)
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ============================================================================
# Masked Attention Layer (used by SPNet)
# ============================================================================

class MaskedAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaskedAttentionLayer, self).__init__(aggr='add')
        self.fc = nn.Linear(2 * in_channels, 1)

    def forward(self, x_cat, x, t, edge_index):
        return self.propagate(edge_index, x_cat=x_cat, x=x, t=t)

    def message(self, x_cat_i, x_cat_j, x_i, x_j, t_i, t_j, edge_index, size_i):
        r_ij = torch.cat([x_cat_i, x_cat_j], dim=-1)
        alpha = F.relu(self.fc(r_ij))
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)
        output = x_j * alpha
        return output

    def update(self, aggr_out):
        return aggr_out
