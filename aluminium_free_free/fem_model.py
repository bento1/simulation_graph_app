# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing,GraphNorm
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
from torch_geometric.nn import GATConv,GATv2Conv,SAGEConv
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
        self.tanh= nn.Tanh()
    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h=  self.norm(x + h) 
        return  self.tanh(h)# residual

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
    
class MeshGNN_GAT4(nn.Module):
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
            nn.Tanh() ,
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim),
            # nn.ELU() ,  # FEM displacement / stress 양수일 때
        )
        # self.tanh=nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(hidden)
    def forward(self, data: Data):
        x = self.lin_in(data.x)
        x = self.bn1(x)
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)

        return self.head(x)
    

class FullMeshGAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden=128,
        edge_dim=4,
        out_dim=3,
        layers=4,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.lin_in = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first
        self.convs.append(
            GATv2Conv(hidden, hidden, heads=heads, edge_dim=edge_dim, concat=True, dropout=dropout)
        )
        self.norms.append(GraphNorm(hidden * heads))

        # middle
        for _ in range(layers - 2):
            self.convs.append(
                GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim,concat=True, dropout=dropout)
            )
            self.norms.append(GraphNorm(hidden * heads))

        # last
        self.convs.append(
            GATv2Conv(hidden * heads, hidden, heads=1, edge_dim=edge_dim, concat=False, dropout=dropout)
        )
        self.norms.append(GraphNorm(hidden))

        self.head = nn.Linear(hidden, out_dim)
        
        self.tanh=nn.Tanh()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_in(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr=data.edge_attr)
            h = F.elu(h)
            h = norm(h, batch)
            x = x + h if x.shape == h.shape else h  # residual (safe)
        x = self.head(x)
        return self.tanh(x)


import torch.nn as nn
import torch.nn.functional as F


class MeshInvariantDiffusion(nn.Module):
    def __init__(self, pos_dim, cond_dim, out_dim, hidden=256, T=1000):
        super().__init__()

        self.time_embed = nn.Embedding(T, hidden)

        self.net = nn.Sequential(
            nn.Linear(pos_dim + cond_dim + out_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pos, cond, x_t, t, batch):
        """
        pos   : [N,3]
        cond  : [N,Cc]
        x_t   : [N,Co]
        t     : [B]
        batch : [N]
        """
        t_emb = self.time_embed(t)      # [B,H]
        t_node = t_emb[batch]           # [N,H]

        h = torch.cat([pos, cond, x_t, t_node], dim=-1)
        return self.net(h)              # ε prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def fourier_encode(pos, L=6):
    """
    pos: [N,3]
    return: [N, 3*(2L+1)]
    """
    enc = [pos]
    for i in range(L):
        freq = 2.0 ** i
        enc.append(torch.sin(freq * pos))
        enc.append(torch.cos(freq * pos))
    return torch.cat(enc, dim=-1)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        return: [B, dim]
        """
        half = self.dim // 2
        device = t.device

        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return x + h

class MeshInvariantDiffusion_ver2(nn.Module):
    def __init__(
        self,
        pos_dim=3,
        cond_dim=8,
        out_dim=3,
        hidden=256,
        T=1000,
        fourier_L=6,
        num_blocks=4,
    ):
        super().__init__()

        # ---- positional encoding ----
        self.fourier_L = fourier_L
        pe_dim = pos_dim * (2 * fourier_L + 1)

        # ---- time embedding ----
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # ---- input projection ----
        in_dim = pe_dim + out_dim + hidden
        self.fc_in = nn.Linear(in_dim, hidden)

        # ---- FiLM conditioning ----
        self.film = FiLM(cond_dim, hidden)

        # ---- residual blocks ----
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(num_blocks)]
        )

        # ---- output ----
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, pos, cond, x_t, t, batch):
        """
        pos   : [N,3]
        cond  : [N,Cc]
        x_t   : [N,Co]
        t     : [B]
        batch : [N]
        """

        # Fourier position encoding
        pos_enc = fourier_encode(pos, self.fourier_L)

        # time embedding (mesh-wise → node-wise)
        t_emb = self.time_embed(t)       # [B,H]
        t_node = t_emb[batch]            # [N,H]

        # input concat
        h = torch.cat([pos_enc, x_t, t_node], dim=-1)
        h = self.fc_in(h)

        # FiLM conditioning
        h = self.film(h, cond)

        # residual MLP
        h = self.blocks(h)

        return self.fc_out(h)             # ε prediction


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels



def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    if t.dtype != torch.float32:
        t = t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class GroupNorm32(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__()
        g = min(num_groups, num_channels)
        self.gn = nn.GroupNorm(g, num_channels)
    def forward(self, x): return self.gn(x)

class ResBlockFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = GroupNorm32(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = torch.chunk(self.emb_proj(emb), 2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)

class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        cond_dim: int = 0,
        base: int = 64,
        time_dim: int = 256,
        ch_mult=(1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        out_channels = out_channels or in_channels
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.cond_proj = None
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # Down
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        cur = base
        self.skip_channels.append(cur)

        for level, mult in enumerate(self.ch_mult):
            out_ch = base * mult
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(ResBlockFiLM(cur, out_ch, emb_dim=time_dim, dropout=dropout))
                cur = out_ch
                self.skip_channels.append(cur)
            if level != len(self.ch_mult) - 1:
                self.downsamples.append(Downsample(cur))
                self.skip_channels.append(cur)

        # Middle
        self.mid1 = ResBlockFiLM(cur, cur, emb_dim=time_dim, dropout=dropout)
        self.mid2 = ResBlockFiLM(cur, cur, emb_dim=time_dim, dropout=dropout)

        # Up
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        skip_chs = list(self.skip_channels)

        for level, mult in reversed(list(enumerate(self.ch_mult))):
            out_ch = base * mult
            for _ in range(self.num_res_blocks):
                skip = skip_chs.pop()
                self.up_blocks.append(ResBlockFiLM(cur + skip, out_ch, emb_dim=time_dim, dropout=dropout))
                cur = out_ch
            if level != 0:
                skip = skip_chs.pop()
                self.up_blocks.append(ResBlockFiLM(cur + skip, out_ch, emb_dim=time_dim, dropout=dropout))
                cur = out_ch
                self.upsamples.append(Upsample(cur))

        self.out_norm = GroupNorm32(cur)
        self.out_conv = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, x, t, cond=None):
        # time emb
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        emb = self.time_mlp(t_emb)
        if self.cond_proj is not None and cond is not None:
            emb = emb + self.cond_proj(cond)

        h = self.in_conv(x)
        hs = [h]

        # Down staged
        dbi = 0
        dsi = 0
        for level, _mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[dbi](h, emb); dbi += 1
                hs.append(h)
            if level != len(self.ch_mult) - 1:
                h = self.downsamples[dsi](h); dsi += 1
                hs.append(h)

        # Middle
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        # Up staged
        ubi = 0
        usi = 0
        for level, _mult in reversed(list(enumerate(self.ch_mult))):
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = self.up_blocks[ubi](torch.cat([h, skip], dim=1), emb); ubi += 1
            if level != 0:
                skip = hs.pop()
                h = self.up_blocks[ubi](torch.cat([h, skip], dim=1), emb); ubi += 1
                h = self.upsamples[usi](h); usi += 1

        return self.out_conv(F.silu(self.out_norm(h)))


def kl_divergence_gaussian(mu, logvar):
    # KL( N(mu, var) || N(0, I) )
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2)
        )

    def forward(self, x, cond):
        """
        x: [B, C, H, W]
        cond: [B, D]
        """
        gamma, beta = self.net(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(out_ch)#nn.GroupNorm(4, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_ch)#nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.film = FiLM(cond_dim, out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, cond):
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = self.film(h, cond)
        return h + self.skip(x)
class ResBlockSpatialCond(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.gamma = nn.Conv2d(cond_ch, out_ch, 3, padding=1)
        self.beta  = nn.Conv2d(cond_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)

        h = h * (1 + self.gamma(cond)) + self.beta(cond)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return h + self.skip(x)

class ResBlockNoAct(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

        self.act = nn.SiLU()

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return h + self.skip(x)
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, base, depth):
        super().__init__()

        layers = []
        ch = in_channels

        for i in range(depth):
            out_ch = base * (2 ** i)
            layers.append(ResBlock(ch, out_ch))
            layers.append(nn.AvgPool2d(2))
            ch = out_ch

        self.conv = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(ch, latent_dim)
        self.fc_logvar = nn.Linear(ch, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, base, depth):
        super().__init__()

        start_ch = base * (2 ** (depth - 1))
        self.fc = nn.Linear(latent_dim, start_ch * 4 * 4)

        layers = []
        ch = start_ch

        for i in reversed(range(depth)):
            out_ch = base * (2 ** i)

            layers.append(ResBlock(ch, out_ch))
            layers.append(
                nn.ConvTranspose2d(
                    out_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            ch = out_ch

        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1))
        layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.conv(h)


class CVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim=128,
        base=64,
        depth=4
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, latent_dim, base, depth)
        self.decoder = Decoder(in_channels, latent_dim, base, depth)

    def forward(self, x, ):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

class ResidualVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim=128,
        base=64,
        depth=4
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, latent_dim, base, depth)
        self.decoder = Decoder(in_channels, latent_dim, base, depth)

    def forward(self, x, ):
        mu_s = x.mean(dim=(2, 3), keepdim=True)
        dx = x - mu_s
        mu, logvar = self.encoder(dx)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        recon = recon + mu_s
        return recon, mu, logvar, z, mu_s
    
class EncoderConvZ(nn.Module): 
    def __init__(self, in_channels, latent_dim, base, depth): 
        super().__init__() 
        layers = [] 
        ch = in_channels 
        for i in range(depth): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlock(ch, out_ch)) 
            layers.append(nn.AvgPool2d(2)) 
            ch = out_ch 
        
        self.conv = nn.Sequential(*layers) 
        self.conv_mu = nn.Conv2d( out_ch, latent_dim, kernel_size=3, stride=1, padding=1 ) 
        self.conv_logvar =nn.Conv2d( out_ch, latent_dim, kernel_size=3, stride=1, padding=1 ) 
    
    def forward(self, x):
        h = self.conv(x) 
        logvar = torch.clamp(self.conv_logvar(h), min=-0.3, max=0.3)
        return self.conv_mu(h), logvar 

def reparameterize(mu, logvar): 
    std = torch.exp(0.5 * logvar) 
    eps = torch.randn_like(std) 
    return mu + eps * std
class ScalingDecoderConvZ(nn.Module):
    def __init__(self, out_channels, latent_dim, base, depth): 
        super().__init__() 
        self.latent_dim=latent_dim 
        ch = latent_dim 
        layers = [] 
        for i in reversed(range(depth)): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlock(ch, out_ch)) 
            layers.append( nn.ConvTranspose2d( out_ch, out_ch, kernel_size=4, stride=2, padding=1 ) ) 
            layers.append(nn.BatchNorm2d(out_ch, eps=1e-8)) 
            layers.append(nn.SiLU()) 
            ch = out_ch 
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1))
        self.conv = nn.Sequential(*layers) 

    def forward(self, latent): 
        z_q = latent[:, :self.latent_dim] # [B, Z] 
        mean = latent[:, self.latent_dim:self.latent_dim+1] # [B,1] 
        std = latent[:, self.latent_dim+1:self.latent_dim+2] # [B,1] 
        scaled_image=self.conv(z_q) 
        B,C,X,Y=scaled_image.shape 
        mean=mean[:,:,0,0].unsqueeze(-1).unsqueeze(-1) 
        std=std[:,:,0,0].unsqueeze(-1).unsqueeze(-1) 
        iamge=scaled_image*std.expand(B,C,X,Y)+mean.expand(B,C,X,Y) 
        return scaled_image,iamge
    
class ScalingVAE3(nn.Module): 
    def __init__( self, in_channels, latent_dim=128, base=64, depth=4 ): 
        super().__init__() 
        self.encoder = EncoderConvZ(in_channels, latent_dim, base, depth) 
        self.decoder = ScalingDecoderConvZ(in_channels, latent_dim, base, depth) 
    
    def forward(self, x): 
        mean = x.mean(dim=(1,2,3), keepdim=True).detach() 
        std = x.std(dim=(1,2,3), keepdim=True).detach() + 1e-12 
        x_norm=(x-mean)/std 
        mu, logvar = self.encoder(x_norm) 
        z = reparameterize(mu, logvar) 
        B,C,X,Y=z.shape 
        z = torch.cat([z, mean.expand(B,1,X,Y), std.expand(B,1,X,Y)], dim=1) # [B, Z+2] 
        scaled_image,iamge = self.decoder(z) 
        return scaled_image,iamge, mu, logvar, z

class ScalingDecoderConvZ4(nn.Module):
    def __init__(self, out_channels, latent_dim, base, depth): 
        super().__init__() 
        self.latent_dim=latent_dim 
        ch = latent_dim 
        layers = [] 
        for i in reversed(range(depth)): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlock(ch, out_ch)) 
            layers.append( nn.ConvTranspose2d( out_ch, out_ch, kernel_size=4, stride=2, padding=1 ) ) 
            layers.append(nn.BatchNorm2d(out_ch, eps=1e-8)) 
            layers.append(nn.SiLU()) 
            ch = out_ch 
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1))
        self.conv = nn.Sequential(*layers) 
        self.MAGIC_MEAN_MEAN=-0.18460561
        self.MAGIC_MEAN_STD=0.0006700889
        self.MAGIC_SCALED_MEAN_MIN=-16.384098
        self.MAGIC_SCALED_MEAN_MAX=16.185783
        self.MAGIC_VAR_MEAN=-15.035463
        self.MAGIC_VAR_STD=3.5190845
        self.MAGIC_SCALED_STD_MIN=-2.7833695
        self.MAGIC_SCALED_STD_MAX=3.713802
        
    def inv_norm_mean(self, m):
        a=-1
        b=1
        m = (m - a) / (b - a)
        m = m * (self.MAGIC_SCALED_MEAN_MAX - self.MAGIC_SCALED_MEAN_MIN) + self.MAGIC_SCALED_MEAN_MIN
        m=self.MAGIC_MEAN_MEAN+m*self.MAGIC_MEAN_STD
        return m
    
    def inv_norm_std(self,s):
        a=-1
        b=1
        s = (s - a) / (b - a)
        s = s * (self.MAGIC_SCALED_STD_MAX - self.MAGIC_SCALED_STD_MIN) + self.MAGIC_SCALED_STD_MIN
        s=s*self.MAGIC_VAR_STD+self.MAGIC_VAR_MEAN
        s=torch.sqrt(torch.exp(s))
        return s
    def forward(self, latent): 
        z_q = latent[:, :self.latent_dim] # [B, Z] 
        mean = latent[:, self.latent_dim:self.latent_dim+1] # [B,1] 
        std = latent[:, self.latent_dim+1:self.latent_dim+2] # [B,1]
        mean=self.inv_norm_mean(mean)
        std=self.inv_norm_std(std)
        scaled_image=self.conv(z_q) 
        B,C,X,Y=scaled_image.shape 
        mean=mean[:,:,0,0].unsqueeze(-1).unsqueeze(-1) 
        std=std[:,:,0,0].unsqueeze(-1).unsqueeze(-1) 
        iamge=scaled_image*std.expand(B,C,X,Y)+mean.expand(B,C,X,Y) 
        return scaled_image,iamge
    
class ScalingVAE4(nn.Module): 
    def __init__( self, in_channels, latent_dim=128, base=64, depth=4 ): 
        super().__init__() 
        self.encoder = EncoderConvZ(in_channels, latent_dim, base, depth) 
        self.decoder = ScalingDecoderConvZ4(in_channels, latent_dim, base, depth) 
        self.MAGIC_MEAN_MEAN=-0.18460561
        self.MAGIC_MEAN_STD=0.0006700889
        self.MAGIC_SCALED_MEAN_MIN=-16.384098
        self.MAGIC_SCALED_MEAN_MAX=16.185783
        self.MAGIC_VAR_MEAN=-15.035463
        self.MAGIC_VAR_STD=3.5190845
        self.MAGIC_SCALED_STD_MIN=-2.7833695
        self.MAGIC_SCALED_STD_MAX=3.713802
        
        
    def norm_mean(self, m):
        a=-1
        b=1
        m=(m-self.MAGIC_MEAN_MEAN)/self.MAGIC_MEAN_STD
        m=     (m - self.MAGIC_SCALED_MEAN_MIN) / (self.MAGIC_SCALED_MEAN_MAX - self.MAGIC_SCALED_MEAN_MIN)
        m = m * (b - a) + a
        return m
    
    def norm_std(self,s):
        a=-1
        b=1
        s=(torch.log(s**2)-self.MAGIC_VAR_MEAN)/self.MAGIC_VAR_STD
        s=     (s- self.MAGIC_SCALED_STD_MIN) / (self.MAGIC_SCALED_STD_MAX - self.MAGIC_SCALED_STD_MIN)
        s = s * (b - a) + a
        return s
    
    def forward(self, x): 
        mean = x.mean(dim=(1,2,3), keepdim=True).detach() 
        std = x.std(dim=(1,2,3), keepdim=True).detach()
        x_norm=(x-mean)/std 
        mu, logvar = self.encoder(x_norm) 
        z = reparameterize(mu, logvar) 
        mean=self.norm_mean(mean)
        std=self.norm_std(std)
        B,C,X,Y=z.shape 
        z = torch.cat([z, mean.expand(B,1,X,Y), std.expand(B,1,X,Y)], dim=1) # [B, Z+2] 
        scaled_image,iamge = self.decoder(z) 
        return scaled_image,iamge, mu, logvar, z

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """
    t: (B,) int64 or float32
    return: (B, dim)
    """
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device).float() / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # (B, dim)

class DDPMScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # posterior q(x_{t-1} | x_t, x0) variance
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        # posterior mean coefficients
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    @torch.no_grad()
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        x0 = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        """
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - b * eps) / torch.clamp(a, min=1e-8)

    @torch.no_grad()
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor):
        """
        DDPM: use eps_hat to estimate x0 then compute posterior mean/var for x_{t-1}.
        """
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_hat)

        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * x0_hat + coef2 * x_t

        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        return mean, log_var, x0_hat

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)
        """
        eps_hat = model(x_t, t, cond)
        mean, log_var, _ = self.p_mean_variance(x_t, t, eps_hat)

        # t==0이면 noise를 더하지 않음
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond: torch.Tensor, device=None, return_all=False):
        """
        Start from x_T ~ N(0, I), iterate t=T-1..0.
        """
        device = device or cond.device
        x = torch.randn(shape, device=device)
        xs = [x] if return_all else None

        for ti in reversed(range(self.T)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond)
            if return_all:
                xs.append(x)

        return xs if return_all else x
    def extract(self,a, t, x_shape):
        # a: [T], t: [B]
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))
    def v_target(self, x0, noise, t):
        alpha_bars=self.alphas_cumprod
        ab = self.extract(alpha_bars, t, x0.shape)
        return (torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * x0).detach()
    
    def predict_x0_from_v(self, x_t, v, t):
        ab = self.extract(self.alphas_cumprod, t, x_t.shape)
        return (torch.sqrt(ab) * x_t - torch.sqrt(1.0 - ab) * v) 

    @torch.no_grad()
    def p_sample_v(self, model, x, t, cond):
        """
        One reverse step: x_t -> x_{t-1} using v-prediction
        """
        # model predicts v
        v_pred = model(x, t, cond)

        # x0 estimate
        x0_hat = self.predict_x0_from_v(x, v_pred, t)

        # coefficients
        beta_t = self.extract(self.betas, t, x.shape)
        alpha_t = 1.0 - beta_t
        ab_t = self.extract(self.alphas_cumprod, t, x.shape)
        ab_prev = self.extract(self.alphas_cumprod_prev, t, x.shape)

        # DDPM posterior mean
        coef1 = torch.sqrt(ab_prev) * beta_t / (1.0 - ab_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t)
        mean = coef1 * x0_hat + coef2 * x

        # noise
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        posterior_var = self.extract(self.posterior_variance, t, x.shape)
        return mean + nonzero_mask * torch.sqrt(posterior_var) * noise
    @torch.no_grad()
    def p_sample_loop_v(
        self,
        model,
        shape,
        cond: torch.Tensor,
        device=None,
        return_all=False
    ):
        """
        Start from x_T ~ N(0, I), iterate t=T-1..0 (v-prediction)
        """
        device = device or cond.device
        B = shape[0]

        # x_T
        x = torch.randn(shape, device=device)

        xs = [x] if return_all else None

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            x = self.p_sample_v(model, x, t, cond)

            if return_all:
                xs.append(x)

        return xs if return_all else x

class CondTokens(nn.Module):
    """
    cond: (B, Z)  -> tokens: (B, N, d)
    """
    def __init__(self, cond_dim: int, num_tokens: int, token_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, num_tokens * token_dim),
            nn.SiLU(),
            nn.Linear(num_tokens * token_dim, num_tokens * token_dim),
        )

    def forward(self, cond: torch.Tensor):
        B = cond.size(0)
        tok = self.proj(cond).view(B, self.num_tokens, self.token_dim)
        return tok



class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, ctx_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film1 = FiLM(ctx_dim, channels)
        self.film2 = FiLM(ctx_dim, channels)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        h = self.conv1(F.silu(self.film1(self.norm1(x), ctx)))
        h = self.conv2(F.silu(self.film2(self.norm2(h), ctx)))
        return x + h
    
class CrossAttention(nn.Module):
    def __init__(self, dim, token_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(token_dim, dim)
        self.to_v = nn.Linear(token_dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, tokens):
        """
        x: [B, HW, C]
        tokens: [B, T, D]
        """
        B, N, C = x.shape
        H = self.heads

        q = self.to_q(x).view(B, N, H, C // H).transpose(1, 2)
        k = self.to_k(tokens).view(B, -1, H, C // H).transpose(1, 2)
        v = self.to_v(tokens).view(B, -1, H, C // H).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class CrossAttnResBlock(nn.Module):
    """
    ResBlock + Cross-Attn (cond tokens) in the middle.
    We still use FiLM lightly for stability (optional but 추천).
    """
    def __init__(self, channels: int, ctx_dim: int, token_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.attn = CrossAttention(dim=channels, token_dim=token_dim, heads=num_heads, )
        self.norm_attn = nn.LayerNorm(channels)

        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.film = FiLM(ctx_dim, channels)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, cond_tokens: torch.Tensor):
        # conv
        h = self.conv1(F.silu(self.film(self.norm1(x), ctx)))

        # cross-attn over 4x4 tokens
        B, C, H, W = h.shape  # H=W=4
        q = h.flatten(2).transpose(1, 2)          # (B,16,C)
        q = self.norm_attn(q)
        q = q + self.attn(q, cond_tokens)         # (B,16,C)
        h = q.transpose(1, 2).view(B, C, H, W)    # back to (B,C,4,4)

        # conv
        h = self.conv2(F.silu(self.film(self.norm2(h), ctx)))
        return x + h
    
class TinyLatentDiffusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels:int,
        base_channels: int,
        cond_dim: int,          # Z
        num_cond_tokens: int,   # N
        token_dim: int,         # d
        time_dim: int = 128,
    ):
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ctx = concat([cond, time_emb]) -> ctx_dim
        self.ctx_dim = cond_dim + time_dim

        self.cond_tokens = CondTokens(cond_dim, num_cond_tokens, token_dim)

        self.in_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        layers=[]
        for _ in range(8):
            layers.append(FiLMResBlock(base_channels, self.ctx_dim))
            layers.append(CrossAttnResBlock(base_channels, self.ctx_dim, token_dim, num_heads=4))
            layers.append(nn.GroupNorm(8, base_channels))
            layers.append(nn.SiLU())
        self.body=nn.ModuleList(layers)
        self.out_proj = nn.Conv2d(base_channels, in_channels, out_channels, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        x_t: (B,C,4,4)
        t:   (B,) int64
        cond:(B,Z)
        return eps_hat: (B,C,4,4)
        """
        # time embedding
        te = sinusoidal_timestep_embedding(t, self.time_dim)
        te = self.time_mlp(te)

        # ctx for FiLM blocks
        ctx = torch.cat([cond, te], dim=1)  # (B, Z+time_dim)

        # cond tokens for cross-attn
        cond_tok = self.cond_tokens(cond)   # (B, N, d)

        h = self.in_proj(x_t)

        for block in self.body:
            if isinstance(block, CrossAttnResBlock):
                h = block(h, ctx,cond_tok)
            elif isinstance(block, FiLMResBlock):
                h = block(h, ctx )
            else:
                h = block(h)

        eps_hat = self.out_proj(h)
        return eps_hat


class LearnableSpatialReducer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv = nn.Conv2d(1, base, kernel_size=1)
        self.norm1 = nn.LayerNorm(base)
        self.norm2 = nn.LayerNorm(base)
        self.fc1 = nn.Linear(base, base)
        self.fc2 = nn.Linear(base, base)
        self.fc3 = nn.Linear(base, 1)

    def forward(self, x):
        # x: [B,1,H,W]
        x = x.mean(dim=(2,3))      # global average (no params)
        x = self.conv(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h+x
        h = self.fc2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = h+x
        x = self.fc3(h)
        return x.view(x.size(0),1,1,1)

class LearnableDense(nn.Module):
    def __init__(self, latent_dim,base):
        super().__init__()
        self.conv = nn.Conv2d(latent_dim, base, kernel_size=1)
        self.norm1 = nn.LayerNorm(base)
        self.norm2 = nn.LayerNorm(base)
        self.fc1 = nn.Linear(base, base)
        self.fc2 = nn.Linear(base, base)
        self.fc3 = nn.Linear(base, 1)

    def forward(self, x):
        # x: [B,1,H,W]
        x = x.mean(dim=(2,3))      # global average (no params)
        x = self.conv(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h+x
        h = self.fc2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = h+x
        x = self.fc3(h)
        return x.view(x.size(0),1,1,1)

class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base=64,
        cond_ch=16,        # 🔥 spatial condition channel
        token_dim=None
    ):
        super().__init__()

        # Encoder
        self.enc1 = ResBlockSpatialCond(in_ch, base, cond_ch)
        self.enc2 = ResBlockSpatialCond(base, base * 2, cond_ch)
        self.enc3 = ResBlockSpatialCond(base * 2, base * 4, cond_ch)

        self.down = nn.AvgPool2d(2)

        # Bottleneck
        self.mid = ResBlockSpatialCond(base * 4, base * 4, cond_ch)

        # Optional cross-attention
        self.use_cross = token_dim is not None
        if self.use_cross:
            self.cross_attn = CrossAttention(base * 4, token_dim)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Decoder
        self.dec3 = ResBlockSpatialCond(base * 8, base * 2, cond_ch)
        self.dec2 = ResBlockSpatialCond(base * 4, base, cond_ch)
        self.dec1 = ResBlockSpatialCond(base * 2, base, cond_ch)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, cond_map, cond_tokens=None):
        """
        x         : [B, C, H, W]
        cond_map  : [B, Cc, H, W]
        """

        # ---- Encoder ----
        # c1 = cond_map
        B,C,W,H=x.shape
        c1 = F.interpolate(cond_map, size=(W, H), mode="nearest")
        e1 = self.enc1(x, c1)

        c2 = self.down(c1)
        e2 = self.enc2(self.down(e1), c2)

        c3 = self.down(c2)
        e3 = self.enc3(self.down(e2), c3)

        # ---- Bottleneck ----
        c4 = self.down(c3)
        h = self.mid(self.down(e3), c4)

        if self.use_cross and cond_tokens is not None:
            B, C, Hh, Ww = h.shape
            h_flat = h.permute(0, 2, 3, 1).reshape(B, Hh * Ww, C)
            h_flat = h_flat + self.cross_attn(h_flat, cond_tokens)
            h = h_flat.view(B, Hh, Ww, C).permute(0, 3, 1, 2)

        # ---- Decoder ----
        h = self.up(h)
        h = self.dec3(torch.cat([h, e3], dim=1), c3)

        h = self.up(h)
        h = self.dec2(torch.cat([h, e2], dim=1), c2)

        h = self.up(h)
        h = self.dec1(torch.cat([h, e1], dim=1), c1)

        return self.out(h)


class ScalingDecoderConvZ5(nn.Module):
    def __init__(self, out_channels, latent_dim, base, depth): 
        super().__init__() 
        self.latent_dim=latent_dim 
        ch = latent_dim 
        layers = [] 
        for i in reversed(range(depth)): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlockNoAct(ch, out_ch)) 
            layers.append( nn.ConvTranspose2d( out_ch, out_ch, kernel_size=4, stride=2, padding=1 ) ) 
            layers.append(nn.BatchNorm2d(out_ch, eps=1e-8)) 
            layers.append(nn.SiLU()) 
            ch = out_ch 
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1)) 
        self.conv = nn.Sequential(*layers) 
        
        self.image_mean_conv=LearnableDense(latent_dim,base)

        # ch = latent_dim 
        # layers = [] 
        # for i in reversed(range(depth)): 
        #     out_ch = base * (2 ** i) 
        #     layers.append(ResBlockNoAct(ch, out_ch)) 
        #     layers.append( nn.ConvTranspose2d( out_ch, out_ch, kernel_size=4, stride=2, padding=1 ) ) 
        #     layers.append(nn.BatchNorm2d(out_ch, eps=1e-8)) 
        #     ch = out_ch 
        # layers.append(nn.Conv2d(ch, out_channels, 3, padding=1)) 
        # self.image_logvar_conv=nn.Sequential(*layers)  #LearnableSpatialReducer(base)
        
        self.std_layers=ConditionalUNet(in_ch=3,
                                        out_ch=3,
                                        base=64,
                                        cond_ch=latent_dim,
                                        token_dim=None)
        self.avg_layers=ConditionalUNet(in_ch=3,
                                    out_ch=3,
                                    base=64,
                                    cond_ch=latent_dim,
                                    token_dim=None)
    def forward(self, latent): 
        scaled_image=self.conv(latent) 
        image=scaled_image*self.std_layers(scaled_image,latent)+self.avg_layers(scaled_image,latent)
        return scaled_image,image

class EncoderConvZ5(nn.Module): 
    def __init__(self, in_channels, latent_dim, base, depth): 
        super().__init__() 
        layers = [] 
        ch = in_channels 
        for i in range(depth): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlockNoAct(ch, out_ch)) 
            layers.append( nn.SiLU())
            layers.append(nn.AvgPool2d(2)) 
            ch = out_ch 
        
        self.conv = nn.Sequential(*layers) 
        layers_conv_mu = [] 
        layers_conv_logvar = []
        ch=out_ch
        for i in range(depth): 
            layers_conv_mu.append(ResBlockNoAct(ch, latent_dim)) 
            layers_conv_mu.append(nn.SiLU()) 
            layers_conv_logvar.append(ResBlockNoAct(ch, latent_dim))
            layers_conv_logvar.append(nn.SiLU()) 
            ch = latent_dim 

        self.conv_mu = nn.Sequential(*layers_conv_mu)
        self.conv_logvar =nn.Sequential(*layers_conv_mu)
    
    def forward(self, x):
        h = self.conv(x) 
        logvar = torch.clamp(self.conv_logvar(h), min=-10, max=10)
        return self.conv_mu(h), logvar 

def reparameterize(mu, logvar): 
    std = torch.exp(0.5 * logvar) 
    eps = torch.randn_like(std) 
    return mu + eps * std
class ScalingVAE5(nn.Module): 
    def __init__( self, in_channels, latent_dim=128, base=64, depth=4 ): 
        super().__init__() 
        self.encoder = EncoderConvZ5(in_channels, latent_dim, base, depth) 
        self.decoder = ScalingDecoderConvZ5(in_channels, latent_dim, base, depth) 

    def forward(self, x ): 
        mu, logvar = self.encoder(x) 
        z = reparameterize(mu, logvar) 
        scaled_image,iamge = self.decoder(z) 
        return scaled_image,iamge, mu, logvar, z

class ResBlockForRESNET(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.silu(self.conv1(x))
        h = self.conv2(h)
        return F.silu(x + h)


class SmallResNetCond(nn.Module):
    def __init__(self, in_ch=3, cond_ch=16, out_ch=3, base=64, depth=4):
        super().__init__()

        self.in_proj = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1)

        self.blocks = nn.Sequential(
            *[ResBlockForRESNET(base) for _ in range(depth)]
        )

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, cond_map):
        if cond_map.shape[-2:] != x.shape[-2:]:
            cond_map = F.interpolate(
                cond_map, size=x.shape[-2:],
                mode="bilinear", align_corners=False
            )

        h = torch.cat([x, cond_map], dim=1)
        h = F.silu(self.in_proj(h))
        h = self.blocks(h)
        return self.out(h)


class ScalingDecoderConvZ6(nn.Module):
    def __init__(self, out_channels, latent_dim, base, depth): 
        super().__init__() 
        self.latent_dim=latent_dim 
        ch = latent_dim 
        layers = [] 
        for i in reversed(range(depth)): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlockNoAct(ch, out_ch)) 
            layers.append( nn.ConvTranspose2d( out_ch, out_ch, kernel_size=4, stride=2, padding=1 ) ) 
            layers.append(nn.BatchNorm2d(out_ch, eps=1e-8)) 
            layers.append(nn.SiLU()) 
            ch = out_ch 
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1)) 
        self.conv = nn.Sequential(*layers) 
        
        self.std_layers=SmallResNetCond(in_ch=3, cond_ch=latent_dim, out_ch=3, base=64, depth=4)
        self.avg_layers=SmallResNetCond(in_ch=3, cond_ch=latent_dim, out_ch=3, base=64, depth=4)

    def forward(self, latent): 
        scaled_image=self.conv(latent) 
        image=scaled_image*self.std_layers(scaled_image,latent)+self.avg_layers(scaled_image,latent)
        return scaled_image,image

class EncoderConvZ6(nn.Module): 
    def __init__(self, in_channels, latent_dim, base, depth): 
        super().__init__() 
        layers = [] 
        ch = in_channels 
        for i in range(depth): 
            out_ch = base * (2 ** i) 
            layers.append(ResBlockNoAct(ch, out_ch)) 
            layers.append( nn.SiLU())
            layers.append(nn.AvgPool2d(2)) 
            ch = out_ch 
        
        self.conv = nn.Sequential(*layers) 
        layers_conv_mu = [] 
        layers_conv_logvar = []
        ch=out_ch
        for i in range(depth): 
            layers_conv_mu.append(ResBlockNoAct(ch, latent_dim)) 
            layers_conv_mu.append(nn.SiLU()) 
            layers_conv_logvar.append(ResBlockNoAct(ch, latent_dim))
            layers_conv_logvar.append(nn.SiLU()) 
            ch = latent_dim 

        self.conv_mu = nn.Sequential(*layers_conv_mu)
        self.conv_logvar =nn.Sequential(*layers_conv_mu)
    
    def forward(self, x):
        h = self.conv(x) 
        logvar = torch.clamp(self.conv_logvar(h), min=-10, max=10)
        return self.conv_mu(h), logvar 

def reparameterize(mu, logvar): 
    std = torch.exp(0.5 * logvar) 
    eps = torch.randn_like(std) 
    return mu + eps * std
class ScalingVAE6(nn.Module): 
    def __init__( self, in_channels, latent_dim=128, base=64, depth=4 ): 
        super().__init__() 
        self.encoder = EncoderConvZ6(in_channels, latent_dim, base, depth) 
        self.decoder = ScalingDecoderConvZ6(in_channels, latent_dim, base, depth) 

    def forward(self, x ): 
        mu, logvar = self.encoder(x) 
        z = reparameterize(mu, logvar) 
        scaled_image,iamge = self.decoder(z) 
        return scaled_image,iamge, mu, logvar, z
    
if  __name__=='__main__':
    x = torch.randn(16, 3, 64, 64)   # [16,3,64,64]
    model_param={ 'in_channels':3, 
            'out_channels':3,
            'GRID':64,
            'loss_scale':1.0,
            'learning_rate':1e-3,
            'num_epochs':1000,
            'base':256,
            'latent_dim':16,
            'batch_size':2,
            'depth':4 } 
    model = ScalingVAE5( in_channels=model_param['in_channels'], 
            latent_dim=model_param['latent_dim'], 
            base=model_param['base'],
            depth=model_param['depth'] )
    model(x)
    # model_param={ 'in_channels':3, 
    #         'out_channels':3,
    #         'GRID':64,
    #         'loss_scale':1.0,
    #         'learning_rate':1e-3,
    #         'num_epochs':1000,
    #         'base':256,
    #         'num_cond_tokens':128,
    #         'token_dim':20,
    #         'time_dim':20,
    #         'batch_size':64,
    #         'cond_dim':20,
    #         'T':1000 } 
    
    # model = TinyLatentDiffusion(
    #     in_channels=model_param['in_channels'],
    #     base_channels=model_param['base'],
    #     cond_dim=model_param['cond_dim'],
    #     num_cond_tokens=model_param['num_cond_tokens'],  # 의미 단위로 늘릴수록 cross-attn 효과 커짐
    #     token_dim=model_param['token_dim'],
    #     time_dim=model_param['time_dim'],
    # )

    # sched = DDPMScheduler(T=model_param['T'])
    # x_t = torch.randn(2, 3, 4, 4)   # [16,3,64,64]
    # t = torch.randn(2, )   # [16,3,64,64]
    # cond = torch.randn(2, 20)   # [16,3,64,64]
    # model( x_t, t, cond)