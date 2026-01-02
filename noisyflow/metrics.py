from __future__ import annotations

from typing import Iterable, List, Optional

import torch


@torch.no_grad()
def _subsample_rows(x: torch.Tensor, max_rows: int, *, seed: int = 0) -> torch.Tensor:
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    n = int(x.shape[0])
    if n <= max_rows:
        return x
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen, device=x.device)[:max_rows]
    return x.index_select(0, idx)


@torch.no_grad()
def sliced_w2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 128,
    max_samples: Optional[int] = 2000,
    seed: int = 0,
) -> float:
    """
    Approximate W2 via the Sliced Wasserstein-2 distance.

    Definition: SW2^2(x,y) = E_u[ W2^2(<u,x>, <u,y>) ] where u is a random unit direction.
    We estimate the expectation with `num_projections` random projections and use the closed-form
    1D W2^2 estimator based on sorted samples.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors of shape (N,d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dim mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("x and y must be non-empty")
    num_projections = int(num_projections)
    if num_projections <= 0:
        raise ValueError("num_projections must be > 0")

    x = x.float()
    y = y.float()

    if max_samples is not None:
        x = _subsample_rows(x, int(max_samples), seed=seed)
        y = _subsample_rows(y, int(max_samples), seed=seed + 1)

    n = int(min(x.shape[0], y.shape[0]))
    if n <= 1:
        return float("nan")
    if int(x.shape[0]) != n:
        x = x[:n]
    if int(y.shape[0]) != n:
        y = y[:n]

    d = int(x.shape[1])
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    proj = torch.randn(d, num_projections, generator=gen, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min_(1e-12)

    x_proj = x @ proj  # (n, P)
    y_proj = y @ proj  # (n, P)
    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)
    w2_sq_per_proj = (x_sorted - y_sorted).pow(2).mean(dim=0)  # (P,)
    sw2_sq = w2_sq_per_proj.mean()
    return float(torch.sqrt(sw2_sq).cpu().item())


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, *, gamma: float) -> torch.Tensor:
    """
    Squared MMD with an RBF kernel k(a,b)=exp(-gamma*||a-b||^2).

    Returns a scalar tensor on the same device as inputs.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors of shape (N,d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dim mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("x and y must be non-empty")
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    x = x.float()
    y = y.float()

    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True)
    d_xx = (x_norm + x_norm.t() - 2.0 * (x @ x.t())).clamp_min_(0.0)
    d_yy = (y_norm + y_norm.t() - 2.0 * (y @ y.t())).clamp_min_(0.0)
    d_xy = (x_norm + y_norm.t() - 2.0 * (x @ y.t())).clamp_min_(0.0)

    k_xx = torch.exp(-gamma * d_xx)
    k_yy = torch.exp(-gamma * d_yy)
    k_xy = torch.exp(-gamma * d_xy)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


@torch.no_grad()
def rbf_mmd2_multi_gamma(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gammas: Iterable[float],
    max_samples: Optional[int] = 2000,
    seed: int = 0,
) -> List[float]:
    """
    Compute RBF MMD^2 for multiple gammas, optionally subsampling rows for scalability.
    """
    if max_samples is not None:
        x = _subsample_rows(x, int(max_samples), seed=seed)
        y = _subsample_rows(y, int(max_samples), seed=seed + 1)
    out: List[float] = []
    for g in gammas:
        out.append(float(rbf_mmd2(x, y, gamma=float(g)).cpu().item()))
    return out
