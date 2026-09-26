import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import GCNConv


class AttentionPoolingPyG(nn.Module):
    def __init__(self, attn_dim: int):
        super().__init__()
        self.A_dim = attn_dim
        # learnable Query Q, Weight(, bias) K, V
        self.query = nn.Parameter(torch.empty(1, 1, self.A_dim))
        nn.init.xavier_uniform_(self.query.data)
        self.w_k = nn.Linear(self.A_dim, self.A_dim)
        self.w_v = nn.Linear(self.A_dim, self.A_dim)

    def forward(self, x, batch):  # x: (B, L, A_D), attention_mask: (B, L)
        K = self.w_k(x)
        V = self.w_v(x)
        scores = (K @ self.query.t()).squeeze(-1) / math.sqrt(self.A_dim)  # (N,)

        weights = pyg_softmax(scores, batch)           # (N,) graph-wise softmax

        B = int(batch.max().item()) + 1
        pooled = torch.zeros(B, self.A_dim, device=x.device)
        pooled.index_add_(0, batch, weights.unsqueeze(-1) * V)
        return pooled


@dataclass
class ModelConfigPyG:
    hidden_dim_num: int
    hidden_dim: int
    dropout: float = 0.1
    embedding_dim: int = 1152
    output_dim: int = 4
    attn_dim: int
    gcn_dim_num: int = 2
    add_self_loops_in_conv: bool = False


class Deeploc2_1_PyG(nn.Module):
    def __init__(self, cfg: ModelConfigPyG):
        super().__init__()
        E_dim = cfg.embedding_dim
        A_dim = cfg.attn_dim
        H_dim = cfg.hidden_dim
        H_dim_num = cfg.hidden_dim_num
        G_dim_num = cfg.gcn_dim_num

        self.dropout = nn.Dropout(cfg.dropout)

        # Input Layer: (B, L, E_D) > (B, L, A_D) + LN(A_D)
        self.input_layer = nn.Sequential(
            nn.LayerNorm(E_dim), nn.Linear(E_dim, A_dim), nn.LayerNorm(A_dim)
        )

        # GCN
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(
                A_dim, A_dim, add_self_loops=cfg.add_self_loops_in_conv, normalize=True
            )
        )
        for _ in range(G_dim_num - 1):
            self.convs.append(
                GCNConv(
                    A_dim,
                    A_dim,
                    add_self_loops=cfg.add_self_loops_in_conv,
                    normalize=True,
                )
            )

        # Pooling Layer: (B, L, A_D) > (B, A_D) + Dropout
        self.attention_Pooling = AttentionPoolingPyG(A_dim)
        layers = [nn.Dropout(cfg.dropout)]

        # Hidden Layer: (B, A_D) > (B, H_D)
        Dim_ = A_dim
        for _ in range(H_dim_num):
            layers.append(nn.LayerNorm(Dim_))
            layers.append(nn.Linear(Dim_, H_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(cfg.dropout))
            Dim_ = H_dim
        self.hidden_layer = nn.Sequential(*layers)

        # Output Layer: (B, H_D) > (B, 4)
        self.output_LFC = nn.Linear(Dim_, cfg.output_dim)

    def forward(self, data):
        x = self.input_layer(x)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attention_Pooling(x, data.batch)
        x = self.hidden_layer(x)
        x = self.output_LFC(x)
        return x
