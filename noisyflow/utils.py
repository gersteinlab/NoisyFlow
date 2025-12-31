from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader: DataLoader) -> Iterator:
    """Infinite dataloader iterator."""
    while True:
        for batch in loader:
            yield batch


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap common training wrappers (e.g., DataParallel/DistributedDataParallel via `.module`
    and Opacus GradSampleModule via `._module`) to access the underlying nn.Module.
    """
    current = module
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_module"):
            current = current._module
            continue
        break
    return current


@dataclass
class DPConfig:
    """Minimal DP-SGD config for Opacus."""

    enabled: bool = True
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    grad_sample_mode: Optional[str] = None
    secure_mode: bool = False


def dp_label_prior_from_counts(
    labels: torch.Tensor,
    num_classes: int,
    mechanism: str = "gaussian",
    sigma: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Optional DP label prior via noised counts.

    Very simple: count labels, add noise, clip to positive, normalize.
    This is *not* a full accountant; it is just the mechanism.
    """
    device = device or labels.device
    counts = torch.bincount(labels.long(), minlength=num_classes).float().to(device)

    if mechanism == "gaussian":
        noise = torch.randn_like(counts) * sigma
        noisy = counts + noise
    elif mechanism == "laplace":
        # Laplace(0, b) with b = sigma here for convenience
        u = torch.rand_like(counts) - 0.5
        noisy = counts - sigma * torch.sign(u) * torch.log1p(-2 * torch.abs(u) + 1e-12)
    else:
        raise ValueError("mechanism must be 'gaussian' or 'laplace'")

    noisy = torch.clamp(noisy, min=1e-6)
    prior = noisy / noisy.sum()
    return prior
