from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], act: str = "silu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
        }
        if act not in acts:
            raise ValueError(f"Unknown activation {act}")
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(acts[act])
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
