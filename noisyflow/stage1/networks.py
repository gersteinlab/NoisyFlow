from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from noisyflow.nn import MLP


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, 1) or (B,)
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=t.device)
        )  # (half,)
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class VelocityField(nn.Module):
    """
    f_psi(z, t, label) -> velocity in R^d.
    """

    def __init__(
        self,
        d: int,
        num_classes: int,
        hidden: List[int] = [256, 256, 256],
        time_emb_dim: int = 64,
        label_emb_dim: int = 64,
        act: str = "silu",
    ):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        in_dim = d + time_emb_dim + label_emb_dim
        self.mlp = MLP(in_dim, d, hidden=hidden, act=act)

    def forward(self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z: (B, d)
        t: (B, 1) or (B,)
        y: (B,) int64 labels
        """
        te = self.time_emb(t)  # (B, time_emb_dim)
        le = self.label_emb(y)  # (B, label_emb_dim)
        h = torch.cat([z, te, le], dim=-1)
        return self.mlp(h)
