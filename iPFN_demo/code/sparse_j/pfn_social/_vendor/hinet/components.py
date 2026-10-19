import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0):
        super(Encoder, self).__init__()
        self.predict1 = nn.Linear(input_size, output_size)
        self.predict2 = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0):
        super(Predictor, self).__init__()
        self.predict1 = nn.Linear(input_size, hidden_size1)
        self.predict2 = nn.Linear(hidden_size1, hidden_size2)
        self.predict3 = nn.Linear(hidden_size2, output_size)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0):
        super(Discriminator, self).__init__()
        self.disc1 = nn.Linear(input_size, hidden_size1)
        self.disc2 = nn.Linear(hidden_size1, hidden_size2)
        self.disc3 = nn.Linear(hidden_size2, output_size)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x


class NetworkOutcomeMixin:
    """Shared treatment-conditioned network outcome branch.

    Each node''s representation and treatment are transformed before graph
    aggregation. This lets the GNN distinguish which feature profiles receive
    treatment, which is required for heterogeneous spillover effects.
    """

    def _init_network_outcome(self, gnn_cls, hidden, dropout):
        self.outcome_message = Encoder(hidden + 1, hidden, dropout)
        self.gnn_y = gnn_cls(hidden, hidden, dropout)
        self.pred_y = Predictor(
            hidden + hidden + 1,
            hidden,
            hidden,
            1,
            dropout,
        )

    def _forward_network_outcome(self, embed, treatment, edge_index):
        treatment = treatment.reshape(-1, 1).to(
            device=embed.device,
            dtype=embed.dtype,
        )
        messages = self.outcome_message(torch.cat([embed, treatment], dim=1))
        neighborhood = self.gnn_y(messages, edge_index)
        return self.pred_y(
            torch.cat([neighborhood, embed, treatment], dim=1)
        )
