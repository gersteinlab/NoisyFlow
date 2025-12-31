from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from noisyflow.nn import MLP


class Classifier(nn.Module):
    def __init__(self, d: int, num_classes: int, hidden: List[int] = [256, 256]):
        super().__init__()
        self.net = MLP(d, num_classes, hidden=hidden, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
