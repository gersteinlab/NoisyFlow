from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import TensorDataset

from noisyflow.utils import set_seed


def make_toy_federated_gaussians(
    K: int = 3,
    n_per_client: int = 2000,
    n_target_ref: int = 2000,
    n_target_test: int = 1000,
    d: int = 2,
    num_classes: int = 3,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Generates a toy multi-client Gaussian mixture classification problem with domain shift.
    - Each class is a Gaussian in base space.
    - Each client applies a different affine transform (batch effect).
    - Target applies its own affine transform.

    Returns:
      client_datasets: list of TensorDataset(x, label)
      target_ref:      TensorDataset(y, label) (public labeled reference nu)
      target_test:     TensorDataset(y, label)
    """
    set_seed(seed)
    base_means = torch.randn(num_classes, d) * 3.0
    base_cov = 0.5

    def sample_base(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, num_classes, (n,))
        x = base_means[labels] + torch.randn(n, d) * base_cov
        return x, labels

    def random_affine() -> Tuple[torch.Tensor, torch.Tensor]:
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        s = torch.diag(torch.exp(torch.randn(d) * 0.2))
        A = Q @ s
        b = torch.randn(d) * 0.5
        return A, b

    client_datasets: List[TensorDataset] = []
    for _ in range(K):
        A, b = random_affine()
        x, labels = sample_base(n_per_client)
        x = x @ A.T + b
        client_datasets.append(TensorDataset(x, labels))

    A_t, b_t = random_affine()
    x_ref, l_ref = sample_base(n_target_ref)
    y_ref = x_ref @ A_t.T + b_t
    x_test, l_test = sample_base(n_target_test)
    y_test = x_test @ A_t.T + b_t

    target_ref = TensorDataset(y_ref, l_ref)
    target_test = TensorDataset(y_test, l_test)
    return client_datasets, target_ref, target_test
