from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from noisyflow.stage1.networks import VelocityField
from noisyflow.utils import DPConfig


def _make_private_with_mode(
    privacy_engine,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp: DPConfig,
):
    grad_sample_mode = getattr(dp, "grad_sample_mode", None)
    if grad_sample_mode is not None:
        try:
            return privacy_engine.make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=dp.noise_multiplier,
                max_grad_norm=dp.max_grad_norm,
                grad_sample_mode=grad_sample_mode,
            )
        except TypeError:
            pass
    return privacy_engine.make_private(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
    )


def flow_matching_loss(
    f: VelocityField,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Sample z ~ N(0,I), t ~ Unif[0,1],
      x_t = (1-t) z + t x
      v*  = x - z
    Minimize || f(x_t, t, y) - v* ||^2
    """
    z = torch.randn_like(x)
    t = torch.rand(x.shape[0], 1, device=x.device)
    x_t = (1.0 - t) * z + t * x
    v_star = x - z
    v = f(x_t, t, y)
    return ((v - v_star) ** 2).sum(dim=1).mean()


@torch.no_grad()
def sample_flow_euler(
    f: VelocityField,
    labels: torch.Tensor,
    n_steps: int = 50,
    z0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solves z' = f(z,t,label), t in [0,1], Euler discretization.
    Returns z_1.

    labels: (B,) int64
    z0: (B,d) optional, else sample N(0,I)
    """
    device = labels.device
    z = torch.randn(labels.shape[0], f.d, device=device) if z0 is None else z0.to(device)
    dt = 1.0 / float(n_steps)
    for k in range(n_steps):
        t = torch.full((labels.shape[0], 1), float(k) / float(n_steps), device=device)
        z = z + dt * f(z, t, labels)
    return z


def train_flow_stage1(
    f: VelocityField,
    loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Client-side training of Stage I. If dp is provided and Opacus is installed, uses DP-SGD.

    Returns dict with final loss and (if DP) epsilon.
    """
    f.to(device)
    f.train()
    opt = torch.optim.Adam(f.parameters(), lr=lr)

    privacy_engine = None
    if dp is not None and dp.enabled:
        try:
            from opacus import PrivacyEngine
        except Exception as e:
            raise RuntimeError(
                "Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP."
            ) from e
        try:
            privacy_engine = PrivacyEngine(secure_mode=getattr(dp, "secure_mode", False))
        except TypeError:
            privacy_engine = PrivacyEngine()
        f, opt, loader = _make_private_with_mode(privacy_engine, f, opt, loader, dp)

    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            loss = flow_matching_loss(f, xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

        if ep % max(1, epochs // 5) == 0:
            print(f"[Stage I] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}")

    out: Dict[str, float] = {"flow_loss": last_loss}
    if privacy_engine is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon_flow"] = eps
        out["delta_flow"] = float(dp.delta)
        print(f"[Stage I] DP eps={eps:.3f}, delta={dp.delta:g}")
    return out
