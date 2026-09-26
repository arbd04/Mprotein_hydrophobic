import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field


@dataclass
class AttentionPoolingConfig:
    attn_dim: int
    kernel_size: int = 5
    sigma: float = 1.0


class AttentionPooling(nn.Module):
    def __init__(self, cfg: AttentionPoolingConfig):
        super().__init__()
        self.A_dim = cfg.attn_dim
        self.K = cfg.kernel_size
        self.R = cfg.kernel_size // 2
        self.sigma = cfg.sigma
        # learnable Query Q, Weight(, bias) K, V
        self.query = nn.Parameter(torch.empty(1, 1, self.A_dim))
        nn.init.xavier_uniform_(self.query.data)
        self.w_k = nn.Linear(self.A_dim, self.A_dim)
        self.w_v = nn.Linear(self.A_dim, self.A_dim)

    def gaussian_blur(self, scores, attention_mask, eps=1e-6):
        # 가우시안 커널: [-R, ... , R]
        x = torch.arange(-self.R, self.R + 1, device=scores.device, dtype=scores.dtype)
        G_kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
        G_kernel = (G_kernel / G_kernel.sum()).view(1, 1, -1)  # (1, 1, K)
        s = scores.unsqueeze(1)  # (B, 1, L)
        m = attention_mask.unsqueeze(1)  # (B, 1, L)
        num = F.conv1d(s * m, G_kernel, padding=self.R)
        den = F.conv1d(m, G_kernel, padding=self.R).clamp_min(eps)
        blurred = (num / den).squeeze(1)  # (B, L)
        return blurred

    def forward(self, x, attention_mask):  # x: (B, L, A_D), attention_mask: (B, L)
        B, _, _ = x.size()
        Q = self.query.expand(B, 1, -1)
        K = self.w_k(x)
        V = self.w_v(x)
        scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.A_dim)).squeeze(
            1
        )  # (B, L)
        scores_blurred = self.gaussian_blur(scores, attention_mask)  # (B, L)
        scores_masked = scores_blurred.masked_fill(attention_mask == 0, -1e9)  # (B, L)
        weights = F.softmax(scores_masked, dim=-1)
        pooled = torch.matmul(weights.unsqueeze(1), V).squeeze(1)
        return pooled


@dataclass
class ModelConfig:
    hidden_dim_num: int
    hidden_dim: int
    dropout: float = 0.1
    embedding_dim: int = 1152
    output_dim: int = 4
    pool: AttentionPoolingConfig = field(default_factory=AttentionPoolingConfig)


class Deeploc2_1(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        E_dim = cfg.embedding_dim
        A_dim = cfg.pool.attn_dim
        H_dim = cfg.hidden_dim
        H_dim_num = cfg.hidden_dim_num

        # Input Layer: (B, L, E_D) > (B, L, A_D) + LN(A_D)
        self.input_layer = nn.Sequential(
            nn.LayerNorm(E_dim), nn.Linear(E_dim, A_dim), nn.LayerNorm(A_dim)
        )

        # Pooling Layer: (B, L, A_D) > (B, A_D) + Dropout
        self.attention_Pooling = AttentionPooling(cfg.pool)
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

    def forward(self, x, attention_mask):
        x = self.input_layer(x)
        x = self.attention_Pooling(x, attention_mask)
        x = self.hidden_layer(x)
        x = self.output_LFC(x)
        return x
