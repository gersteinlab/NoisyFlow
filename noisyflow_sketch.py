
"""
NoisyFlow sketch implementation (single-file starter).

Implements the three stages from your draft:
- Stage I: DP Flow Matching generator (velocity field f_psi) trained with (optional) DP-SGD
- Stage II: Neural OT map via ICNN potential Phi_theta (Option A/B/C)
- Stage III: Server-side synthesis + classifier training

This is intentionally a *sketch*: it’s a runnable starting point for tabular / embedding-space
experiments, not a production-grade training pipeline.

Dependencies:
  pip install torch numpy
Optional (for DP-SGD):
  pip install opacus

Notes:
- Flow sampling uses a simple fixed-step Euler solver for the ODE z' = f(z,t,label).
- ICNN conjugate is approximated by inner gradient ascent on x (detached each step).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# -----------------------------
# Utilities
# -----------------------------

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

@dataclass
class DPConfig:
    """Minimal DP-SGD config for Opacus."""
    enabled: bool = True
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5

# -----------------------------
# Stage I: Flow Matching Generator
# -----------------------------

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
        # shape (B, half)
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb

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

class VelocityField(nn.Module):
    """
    f_psi(z, t, label) -> velocity in R^d

    Matches your Stage I description (Eq. velocity-field, flow-ode, flow-loss). fileciteturn0file0
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
        te = self.time_emb(t)                    # (B, time_emb_dim)
        le = self.label_emb(y)                   # (B, label_emb_dim)
        h = torch.cat([z, te, le], dim=-1)        # (B, d+...)
        return self.mlp(h)                        # (B, d)

def flow_matching_loss(
    f: VelocityField,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Eq. (flow-loss) from your draft. fileciteturn0file0

    Sample z ~ N(0,I), t ~ Unif[0,1],
      x_t = (1-t) z + t x
      v*  = x - z
    Minimize || f(x_t, t, y) - v* ||^2
    """
    B, d = x.shape
    z = torch.randn_like(x)
    t = torch.rand(B, 1, device=x.device)
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
    B = labels.shape[0]
    d = f.d
    z = torch.randn(B, d, device=device) if z0 is None else z0.to(device)
    dt = 1.0 / float(n_steps)
    for k in range(n_steps):
        t = torch.full((B, 1), float(k) / float(n_steps), device=device)
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
        privacy_engine = PrivacyEngine()
        f, opt, loader = privacy_engine.make_private(
            module=f,
            optimizer=opt,
            data_loader=loader,
            noise_multiplier=dp.noise_multiplier,
            max_grad_norm=dp.max_grad_norm,
        )

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
        print(f"[Stage I] DP ε={eps:.3f}, δ={dp.delta:g}")
    return out

def dp_label_prior_from_counts(
    labels: torch.Tensor,
    num_classes: int,
    mechanism: str = "gaussian",
    sigma: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Optional DP label prior via noised counts, as in your Stage I paragraph. fileciteturn0file0

    Very simple: count labels, add noise, clip to positive, normalize.
    This is *not* a full accountant; it’s just the mechanism.
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

# -----------------------------
# Stage II: ICNN + Neural OT Dual
# -----------------------------

class ICNN(nn.Module):
    """
    Simple fully-connected ICNN: convex in x by constraining U_l >= 0.

    z_{l+1} = act(W_l x + U_l z_l + b_l), with U_l elementwise >= 0.
    Output: scalar Phi(x) = w_out^T z_L + w_lin^T x + b

    This aligns with your Stage II description (ICNN potential, T(x)=∇Phi(x)). fileciteturn0file0
    """
    def __init__(
        self,
        d: int,
        hidden: List[int] = [128, 128, 128],
        act: str = "relu",
        add_strong_convexity: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.add_strong_convexity = float(add_strong_convexity)

        if act == "relu":
            self.act = F.relu
        elif act == "softplus":
            self.act = F.softplus
        else:
            raise ValueError("act must be 'relu' or 'softplus' for ICNN convexity")

        # W_x layers: unconstrained
        self.Wxs = nn.ModuleList()
        # U_z layers: constrained nonnegative via softplus param
        self.Uzs_raw = nn.ParameterList()
        self.bs = nn.ParameterList()

        # First layer depends only on x (no z input)
        h0 = hidden[0]
        self.Wxs.append(nn.Linear(d, h0))
        self.bs.append(nn.Parameter(torch.zeros(h0)))

        # Subsequent layers depend on x and previous z
        for idx in range(1, len(hidden)):
            h_in = hidden[idx - 1]
            h_out = hidden[idx]
            self.Wxs.append(nn.Linear(d, h_out))
            self.Uzs_raw.append(nn.Parameter(torch.randn(h_out, h_in) * 0.01))
            self.bs.append(nn.Parameter(torch.zeros(h_out)))

        # Output layer: linear in z_L plus linear in x
        self.w_out = nn.Linear(hidden[-1], 1, bias=True)
        self.w_lin = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d)
        returns Phi(x): (B,)
        """
        # First hidden
        z = self.act(self.Wxs[0](x) + self.bs[0])

        # Remaining hiddens
        for l in range(1, len(self.hidden)):
            Wx = self.Wxs[l](x)
            Uz = F.linear(z, F.softplus(self.Uzs_raw[l - 1]))  # ensures nonneg weights
            z = self.act(Wx + Uz + self.bs[l])

        out = self.w_out(z) + self.w_lin(x)  # (B,1)
        out = out.squeeze(-1)                # (B,)

        if self.add_strong_convexity > 0:
            out = out + 0.5 * self.add_strong_convexity * (x * x).sum(dim=1)
        return out

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        """
        T(x) = ∇_x Phi(x). This is the OT map for quadratic cost (Brenier). fileciteturn0file0
        """
        x_req = x.detach().requires_grad_(True)
        phi = self.forward(x_req).sum()
        grad = torch.autograd.grad(phi, x_req, create_graph=False)[0]
        return grad

def approx_conjugate(
    phi: ICNN,
    y: torch.Tensor,
    n_steps: int = 20,
    lr: float = 0.1,
    clamp: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate Phi*(y) = sup_x <x,y> - Phi(x) via inner gradient ascent on x.

    Returns:
      phi_star: (B,) approximate conjugate values
      x_star:   (B,d) approximate argmax points (detached)
    """
    x = y.detach().clone()
    for _ in range(n_steps):
        x = x.detach().requires_grad_(True)
        obj = (x * y).sum(dim=1) - phi(x)         # (B,)
        loss = -obj.mean()                        # maximize obj
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
        with torch.no_grad():
            x = x - lr * grad                     # gradient descent on -obj = ascent on obj
            if clamp is not None:
                x = torch.clamp(x, -clamp, clamp)

    with torch.no_grad():
        phi_star = (x * y).sum(dim=1) - phi(x)
    return phi_star, x.detach()

def ot_dual_loss(
    phi: ICNN,
    x: torch.Tensor,
    y: torch.Tensor,
    conj_steps: int = 20,
    conj_lr: float = 0.1,
    conj_clamp: Optional[float] = None,
) -> torch.Tensor:
    """
    Loss to MINIMIZE = -J(theta), where
      J(theta) = E_x[Phi_theta(x)] + E_y[Phi_theta^*(y)]
    matches Eq. (ot-objective) in your draft. fileciteturn0file0
    """
    phi_x = phi(x)                               # (B,)
    phi_star_y, _ = approx_conjugate(phi, y, n_steps=conj_steps, lr=conj_lr, clamp=conj_clamp)
    J = phi_x.mean() + phi_star_y.mean()
    return -J

def train_ot_stage2(
    phi: ICNN,
    # real_loader provides x_real batches (private) if option in {"A","C"}
    real_loader: Optional[DataLoader],
    # target_loader provides y batches (public reference ν)
    target_loader: DataLoader,
    option: str = "B",
    # synthetic sampler provides x_synth batches if option in {"B","C"}
    synth_sampler: Optional[Callable[[int], torch.Tensor]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    conj_steps: int = 20,
    conj_lr: float = 0.1,
    conj_clamp: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Client-side Stage II.

    Options (as in your paper):
      - "A": real data only, DP-SGD on theta
      - "B": synthetic only, non-private SGD (post-processing of DP flow)
      - "C": mixed. Here we implement a safe simplified variant:
            concatenate real+synthetic and (optionally) run DP-SGD on the whole loss.
            (You can later refine to DP only on real gradients.)

    Returns dict with loss and epsilon (if DP used).
    """
    option = option.upper()
    if option not in {"A", "B", "C"}:
        raise ValueError("option must be one of 'A','B','C'")

    if option in {"A", "C"} and real_loader is None:
        raise ValueError("real_loader required for option A/C")
    if option in {"B", "C"} and synth_sampler is None:
        raise ValueError("synth_sampler required for option B/C")

    phi.to(device)
    phi.train()
    opt = torch.optim.Adam(phi.parameters(), lr=lr)

    privacy_engine = None
    # DP only makes sense when real data is used.
    if dp is not None and dp.enabled and option in {"A", "C"}:
        try:
            from opacus import PrivacyEngine
        except Exception as e:
            raise RuntimeError(
                "Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP."
            ) from e
        privacy_engine = PrivacyEngine()
        # IMPORTANT: make_private expects the *private* data loader.
        phi, opt, real_loader = privacy_engine.make_private(
            module=phi,
            optimizer=opt,
            data_loader=real_loader,
            noise_multiplier=dp.noise_multiplier,
            max_grad_norm=dp.max_grad_norm,
        )

    y_iter = cycle(target_loader)
    if real_loader is not None:
        x_iter = cycle(real_loader)

    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        # define how many steps per epoch
        steps = len(real_loader) if (option in {"A", "C"} and real_loader is not None) else len(target_loader)
        for _ in range(steps):
            yb = next(y_iter)
            # target loader could yield (y,) or (y, labels)
            if isinstance(yb, (list, tuple)):
                yb = yb[0]
            yb = yb.to(device).float()

            if option == "A":
                xb = next(x_iter)
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(device).float()

            elif option == "B":
                # synthetic-only batch size = y batch size
                xb = synth_sampler(yb.shape[0]).to(device).float()

            else:  # "C"
                xr = next(x_iter)
                if isinstance(xr, (list, tuple)):
                    xr = xr[0]
                xr = xr.to(device).float()
                xs = synth_sampler(xr.shape[0]).to(device).float()
                xb = torch.cat([xr, xs], dim=0)
                # also expand y to match xb size for stability (use two y batches)
                y2 = next(y_iter)
                if isinstance(y2, (list, tuple)):
                    y2 = y2[0]
                y2 = y2.to(device).float()
                yb = torch.cat([yb, y2], dim=0)

            loss = ot_dual_loss(
                phi, xb, yb,
                conj_steps=conj_steps, conj_lr=conj_lr, conj_clamp=conj_clamp
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            last_loss = float(loss.detach().cpu().item())

        if ep % max(1, epochs // 5) == 0:
            print(f"[Stage II/{option}] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}")

    out: Dict[str, float] = {"ot_loss": last_loss}
    if privacy_engine is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon_ot"] = eps
        out["delta_ot"] = float(dp.delta)
        print(f"[Stage II/{option}] DP ε={eps:.3f}, δ={dp.delta:g}")
    return out

# -----------------------------
# Stage III: Server synthesis + classifier
# -----------------------------

class Classifier(nn.Module):
    def __init__(self, d: int, num_classes: int, hidden: List[int] = [256, 256]):
        super().__init__()
        self.net = MLP(d, num_classes, hidden=hidden, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def sample_labels_from_prior(prior: torch.Tensor, n: int) -> torch.Tensor:
    """
    prior: (C,) probabilities on device
    returns labels: (n,) int64 on same device
    """
    return torch.multinomial(prior, num_samples=n, replacement=True).long()

@torch.no_grad()
def server_synthesize(
    clients: List[Dict],
    M_per_client: int,
    num_classes: int,
    flow_steps: int = 50,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements server-side synthesis in Eq. (server-synth). fileciteturn0file0

    Each element in clients is a dict containing:
      - "flow": VelocityField (DP-trained)
      - "ot":   ICNN (DP-trained or post-processed)
      - optional "prior": tensor (C,)
    """
    ys: List[torch.Tensor] = []
    ls: List[torch.Tensor] = []
    for idx, c in enumerate(clients):
        flow: VelocityField = c["flow"].to(device).eval()
        ot: ICNN = c["ot"].to(device).eval()
        prior: Optional[torch.Tensor] = c.get("prior", None)
        if prior is None:
            prior = torch.ones(num_classes, device=device) / float(num_classes)
        else:
            prior = prior.to(device)

        labels = sample_labels_from_prior(prior, M_per_client).to(device)
        x_tilde = sample_flow_euler(flow, labels, n_steps=flow_steps)       # (M,d)
        y_tilde = ot.transport(x_tilde)                                     # (M,d)
        ys.append(y_tilde.cpu())
        ls.append(labels.cpu())
        print(f"[Server] client {idx} synthesized {M_per_client} samples")

    Y = torch.cat(ys, dim=0)
    L = torch.cat(ls, dim=0)
    return Y, L

def train_classifier(
    clf: Classifier,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, float]:
    clf.to(device)
    clf.train()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

        if ep % max(1, epochs // 5) == 0:
            msg = f"[Classifier] epoch {ep:04d}/{epochs} loss={last_loss:.4f}"
            if test_loader is not None:
                acc = eval_classifier(clf, test_loader, device=device)["acc"]
                msg += f"  test_acc={acc:.3f}"
            print(msg)

    out: Dict[str, float] = {"clf_loss": last_loss}
    if test_loader is not None:
        out.update(eval_classifier(clf, test_loader, device=device))
    return out

@torch.no_grad()
def eval_classifier(
    clf: Classifier,
    loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    clf.eval()
    n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        pred = clf(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        n += int(yb.numel())
    acc = correct / max(1, n)
    return {"acc": float(acc)}

# -----------------------------
# Synthetic federated toy data (Tier A starter)
# -----------------------------

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
      target_ref:      TensorDataset(y)    (unlabeled reference ν)
      target_test:     TensorDataset(y, label)
    """
    set_seed(seed)
    # Base means per class
    base_means = torch.randn(num_classes, d) * 3.0
    base_cov = 0.5

    def sample_base(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, num_classes, (n,))
        x = base_means[labels] + torch.randn(n, d) * base_cov
        return x, labels

    def random_affine() -> Tuple[torch.Tensor, torch.Tensor]:
        # Random rotation + scaling
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        s = torch.diag(torch.exp(torch.randn(d) * 0.2))
        A = Q @ s
        b = torch.randn(d) * 0.5
        return A, b

    client_datasets: List[TensorDataset] = []
    for i in range(K):
        A, b = random_affine()
        x, labels = sample_base(n_per_client)
        x = x @ A.T + b
        client_datasets.append(TensorDataset(x, labels))

    # Target domain
    A_t, b_t = random_affine()
    x_ref, _ = sample_base(n_target_ref)
    y_ref = x_ref @ A_t.T + b_t
    x_test, l_test = sample_base(n_target_test)
    y_test = x_test @ A_t.T + b_t

    target_ref = TensorDataset(y_ref)                # unlabeled
    target_test = TensorDataset(y_test, l_test)      # labeled for eval only
    return client_datasets, target_ref, target_test

# -----------------------------
# End-to-end demo runner (toy)
# -----------------------------

def run_toy_demo(
    device: str = "cpu",
    option_stage2: str = "B",
    dp_stage1: bool = False,
    dp_stage2: bool = False,
) -> None:
    """
    End-to-end demo for the toy 2D Gaussian setup.

    Recommended first run:
      option_stage2="B", dp_stage1=False, dp_stage2=False

    Then turn on dp_stage1 (and keep option B) to match your cleanest DP story. fileciteturn0file0
    """
    K = 3
    num_classes = 3
    d = 2

    client_datasets, target_ref, target_test = make_toy_federated_gaussians(
        K=K, n_per_client=1500, n_target_ref=2000, n_target_test=1000,
        d=d, num_classes=num_classes, seed=0
    )

    # Public target reference loader ν
    target_loader = DataLoader(target_ref, batch_size=256, shuffle=True, drop_last=True)
    target_test_loader = DataLoader(target_test, batch_size=512, shuffle=False)

    clients_out: List[Dict] = []
    for i in range(K):
        ds = client_datasets[i]
        loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

        # --- Stage I
        flow = VelocityField(d=d, num_classes=num_classes, hidden=[128, 128], time_emb_dim=32, label_emb_dim=32)
        dp1 = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5) if dp_stage1 else None
        train_flow_stage1(flow, loader, epochs=20, lr=1e-3, dp=dp1, device=device)

        # Optional label prior (non-accounted sketch)
        all_labels = torch.cat([b[1] for b in loader], dim=0)
        prior = dp_label_prior_from_counts(all_labels, num_classes=num_classes, sigma=1.0, device="cpu")

        # synthetic sampler for Stage II option B/C
        def synth_sampler(batch_size: int, flow=flow) -> torch.Tensor:
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            return sample_flow_euler(flow.to(device).eval(), labels, n_steps=50).cpu()

        # --- Stage II
        ot = ICNN(d=d, hidden=[128, 128], act="relu", add_strong_convexity=0.1)
        dp2 = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5) if dp_stage2 else None

        # real_loader supplies x only (drop labels)
        real_x_loader = DataLoader(TensorDataset(ds.tensors[0]), batch_size=256, shuffle=True, drop_last=True)

        train_ot_stage2(
            ot,
            real_loader=real_x_loader if option_stage2.upper() in {"A", "C"} else None,
            target_loader=target_loader,
            option=option_stage2,
            synth_sampler=(lambda bs: synth_sampler(bs)) if option_stage2.upper() in {"B", "C"} else None,
            epochs=30,
            lr=1e-3,
            dp=dp2,
            conj_steps=20,
            conj_lr=0.2,
            conj_clamp=10.0,
            device=device,
        )

        clients_out.append({"flow": flow.cpu(), "ot": ot.cpu(), "prior": prior})

    # --- Stage III (server)
    Ysyn, Lsyn = server_synthesize(
        clients_out, M_per_client=5000, num_classes=num_classes, flow_steps=50, device=device
    )
    syn_loader = DataLoader(TensorDataset(Ysyn, Lsyn), batch_size=512, shuffle=True, drop_last=True)

    clf = Classifier(d=d, num_classes=num_classes, hidden=[128, 128])
    stats = train_classifier(clf, syn_loader, test_loader=target_test_loader, epochs=30, lr=1e-3, device=device)
    print("Final stats:", stats)

if __name__ == "__main__":
    run_toy_demo(device="cpu", option_stage2="B", dp_stage1=False, dp_stage2=False)
