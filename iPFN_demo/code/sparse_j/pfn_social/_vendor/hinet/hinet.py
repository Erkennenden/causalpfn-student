import torch
import torch.nn as nn

from .components import Discriminator, Encoder, NetworkOutcomeMixin
from .layers import GradientReversalLayer, get_gnn_layer


class HINet(NetworkOutcomeMixin, nn.Module):
    """Heterogeneous Interference Network with network-aware balancing."""

    def __init__(
        self,
        Xshape,
        hidden,
        dropout=0,
        gnn_layer="GIN",
    ):
        super().__init__()

        GNN = get_gnn_layer(gnn_layer)
        self.gnn_layer_name = gnn_layer

        self.encoder = Encoder(Xshape, hidden, dropout)

        # Network-aware treatment-prediction branch.
        self.gnn_predict = GNN(hidden, hidden, dropout)
        self.grl = GradientReversalLayer(lambda_=1)
        self.discriminator = Discriminator(
            hidden + hidden,
            hidden,
            hidden,
            1,
            dropout,
        )

        # Treatment-conditioned outcome messages are transformed before
        # aggregation so feature-dependent spillover effects can be learned.
        self._init_network_outcome(GNN, hidden, dropout)

    def forward(self, x, t, z, edge_index):
        del z
        embed = self.encoder(x)

        reversed_embed = self.grl(embed)
        neighborhood = self.gnn_predict(reversed_embed, edge_index)
        t_pred = self.discriminator(
            torch.cat([neighborhood, reversed_embed], dim=1)
        )

        y = self._forward_network_outcome(embed, t, edge_index)
        return t_pred, y


class HINet_no_net_conf(NetworkOutcomeMixin, nn.Module):
    """HINet ablation without neighborhood context in its discriminator.

    Its encoder and outcome branch are identical to HINet. Only the treatment
    branch differs: treatment is predicted from the node's own reversed
    representation, without aggregating neighboring representations.
    """

    def __init__(
        self,
        Xshape,
        hidden,
        dropout=0,
        gnn_layer="GIN",
    ):
        super().__init__()

        GNN = get_gnn_layer(gnn_layer)
        self.gnn_layer_name = gnn_layer

        self.encoder = Encoder(Xshape, hidden, dropout)
        self.grl = GradientReversalLayer(lambda_=1)
        self.discriminator = Discriminator(
            hidden,
            hidden,
            hidden,
            1,
            dropout,
        )

        # Keep this exactly matched to HINet so the ablation isolates the
        # neighborhood-aware treatment discriminator.
        self._init_network_outcome(GNN, hidden, dropout)

    def forward(self, x, t, z, edge_index):
        del z
        embed = self.encoder(x)

        t_pred = self.discriminator(self.grl(embed))
        y = self._forward_network_outcome(embed, t, edge_index)
        return t_pred, y
