from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset

from noisyflow.utils import set_seed


def _random_affine(d: int, scale_logstd: float, shift_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    s = torch.diag(torch.exp(torch.randn(d) * scale_logstd))
    A = Q @ s
    b = torch.randn(d) * shift_scale
    return A, b


def make_federated_mixture_gaussians(
    K: int = 3,
    n_per_client: int = 2000,
    n_target_ref: int = 2000,
    n_target_test: int = 1000,
    d: int = 2,
    num_classes: int = 3,
    component_scale: float = 3.0,
    component_cov: float = 0.5,
    class_probs: Optional[List[float]] = None,
    scale_logstd: float = 0.2,
    shift_scale: float = 0.5,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Synthetic federated mixture-of-Gaussians classification dataset.

    Returns:
      client_datasets: list of TensorDataset(x, label)
      target_ref:      TensorDataset(y, label) (public labeled reference)
      target_test:     TensorDataset(y, label)
    """
    set_seed(seed)
    means = torch.randn(num_classes, d) * component_scale

    if class_probs is None:
        probs = torch.ones(num_classes) / float(num_classes)
    else:
        probs = torch.tensor(class_probs, dtype=torch.float32)
        if probs.numel() != num_classes:
            raise ValueError("class_probs length must match num_classes")
        probs = probs / probs.sum()

    def sample_base(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.multinomial(probs, n, replacement=True)
        x = means[labels] + torch.randn(n, d) * component_cov
        return x, labels

    client_datasets: List[TensorDataset] = []
    for _ in range(K):
        A, b = _random_affine(d, scale_logstd=scale_logstd, shift_scale=shift_scale)
        x, labels = sample_base(n_per_client)
        x = x @ A.T + b
        client_datasets.append(TensorDataset(x, labels))

    A_t, b_t = _random_affine(d, scale_logstd=scale_logstd, shift_scale=shift_scale)
    x_ref, l_ref = sample_base(n_target_ref)
    y_ref = x_ref @ A_t.T + b_t
    x_test, l_test = sample_base(n_target_test)
    y_test = x_test @ A_t.T + b_t

    target_ref = TensorDataset(y_ref, l_ref)
    target_test = TensorDataset(y_test, l_test)
    return client_datasets, target_ref, target_test
