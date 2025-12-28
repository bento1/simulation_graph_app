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


# gnn_fem_mesh_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv,GATv2Conv
from torch_geometric.utils import coalesce

class EdgeGATBlock(nn.Module):
    def __init__(self, hidden, edge_dim, heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATConv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(x + h)   # residual

class MeshGNN_GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim=4,
        hidden=128,
        layers=6,
        heads=8,
        out_dim=3,
        dropout=0.1,
    ):
        super().__init__()

        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        self.blocks = nn.ModuleList([
            EdgeGATBlock(
                hidden=hidden,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),  # FEM displacement / stress 양수일 때
        )

    def forward(self, data: Data):
        x = self.lin_in(data.x)

        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)

        return self.head(x)


class EdgeGAT2Block(nn.Module):
    def __init__(self, hidden, edge_dim, heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(x + h)   # residual

class MeshGNN_GAT2(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim=4,
        hidden=128,
        layers=6,
        heads=8,
        out_dim=3,
        dropout=0.1,
    ):
        super().__init__()

        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.LayerNorm(hidden),
        )

        self.blocks = nn.ModuleList([
            EdgeGAT2Block(
                hidden=hidden,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU() ,
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim),
            nn.ELU() ,  # FEM displacement / stress 양수일 때
        )
        self.relu=nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden)
    def forward(self, data: Data):
        x = self.lin_in(data.x)
        x = F.elu(x)
        x = self.bn1(x)
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return self.relu(x)
    
class EdgeGAT3Block(nn.Module):
    def __init__(self, hidden, edge_dim, heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden ,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False
        )
        self.norm = nn.LayerNorm(hidden*heads)
        self.dropout = dropout
        self.lin=nn.Linear(hidden,hidden*heads)
        self.lin2=nn.Linear(hidden*heads,hidden)
        

    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x   = self.lin(x)
        x = F.tanh(x)
        x= self.norm( x+ h) 
        x = self.lin2(x)
        x = F.tanh(x)
        return x# residual
    
class MeshGNN_GAT3(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim=4,
        hidden=128,
        layers=6,
        heads=8,
        out_dim=3,
        dropout=0.1,
    ):
        super().__init__()

        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.LayerNorm(hidden),
        )

        self.blocks = nn.ModuleList([
            EdgeGAT3Block(
                hidden=hidden,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh() ,
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim),
            # nn.ELU() ,  # FEM displacement / stress 양수일 때
        )
        self.tanh=nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(hidden)
    def forward(self, data: Data):
        x = self.lin_in(data.x)
        x = self.bn1(x)
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return self.tanh(x)