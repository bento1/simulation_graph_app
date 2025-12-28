# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import coalesce



class EdgeMP(MessagePassing):
    def __init__(self, node_in, edge_in, hidden):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_in + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_in + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, m], dim=1))
        # out = x+ out
        # return self.norm(out)
        return out

    def message(self, x_j, edge_attr):
        # x_j: source node feature, edge_attr: [dx,dy,dz,dist]
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=1))


class MeshGNN(nn.Module):
    def __init__(self, in_dim, edge_dim=4, hidden=128, layers=4, out_dim=3, dropout=0.1):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([EdgeMP(hidden, edge_dim, hidden) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, data: Data):
        x = self.lin_in(data.x)
        x = F.relu(x)

        for conv in self.convs:
            x_new = conv(x, data.edge_index, data.edge_attr)
            x = F.relu(x_new + x)  # residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x)


