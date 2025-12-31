from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from noisyflow.nn import MLP
from noisyflow.stage1.networks import SinusoidalTimeEmbedding


class ICNN(nn.Module):
    """
    Simple fully-connected ICNN: convex in x by constraining U_l >= 0.

    z_{l+1} = act(W_l x + U_l z_l + b_l), with U_l elementwise >= 0.
    Output: scalar Phi(x) = w_out^T z_L + w_lin^T x + b
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

        self.Wxs = nn.ModuleList()
        self.Uzs_raw = nn.ParameterList()
        self.bs = nn.ParameterList()

        h0 = hidden[0]
        self.Wxs.append(nn.Linear(d, h0))
        self.bs.append(nn.Parameter(torch.zeros(h0)))

        for idx in range(1, len(hidden)):
            h_in = hidden[idx - 1]
            h_out = hidden[idx]
            self.Wxs.append(nn.Linear(d, h_out))
            self.Uzs_raw.append(nn.Parameter(torch.randn(h_out, h_in) * 0.01))
            self.bs.append(nn.Parameter(torch.zeros(h_out)))

        self.w_out = nn.Linear(hidden[-1], 1, bias=True)
        self.w_lin = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d)
        returns Phi(x): (B,)
        """
        z = self.act(self.Wxs[0](x) + self.bs[0])

        for l in range(1, len(self.hidden)):
            Wx = self.Wxs[l](x)
            Uz = F.linear(z, F.softplus(self.Uzs_raw[l - 1]))
            z = self.act(Wx + Uz + self.bs[l])

        out = self.w_out(z) + self.w_lin(x)
        out = out.squeeze(-1)

        if self.add_strong_convexity > 0:
            out = out + 0.5 * self.add_strong_convexity * (x * x).sum(dim=1)
        return out

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        """
        T(x) = grad_x Phi(x). This is the OT map for quadratic cost (Brenier).
        """
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            phi = self.forward(x_req).sum()
            grad = torch.autograd.grad(phi, x_req, create_graph=False)[0]
        return grad


class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.kernel(), self.bias)

    def kernel(self) -> torch.Tensor:
        return F.softplus(self.weight, beta=self.beta)


class CellOTICNN(nn.Module):
    """
    ICNN architecture matching the CellOT implementation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        activation: str = "LeakyReLU",
        softplus_W_kernels: bool = False,
        softplus_beta: float = 1.0,
        fnorm_penalty: float = 0.0,
        kernel_init_fxn: Optional[callable] = None,
    ):
        super().__init__()
        self.fnorm_penalty = float(fnorm_penalty)
        self.softplus_W_kernels = bool(softplus_W_kernels)

        act_name = activation.lower().replace("_", "")
        if act_name == "relu":
            self.sigma = nn.ReLU
        elif act_name == "leakyrelu":
            self.sigma = nn.LeakyReLU
        else:
            raise ValueError("activation must be 'ReLU' or 'LeakyReLU'")

        units = list(hidden_units) + [1]

        if self.softplus_W_kernels:

            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)

        else:
            WLinear = nn.Linear

        self.W = nn.ModuleList(
            [WLinear(idim, odim, bias=False) for idim, odim in zip(units[:-1], units[1:])]
        )
        self.A = nn.ModuleList([nn.Linear(input_dim, odim, bias=True) for odim in units])

        if kernel_init_fxn is not None:
            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.W:
                kernel_init_fxn(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.sigma(0.2)(self.A[0](x))
        z = z * z

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)
        return y

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        assert x.requires_grad
        grad_outputs = torch.ones((x.size(0), 1), device=x.device).float()
        (output,) = torch.autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=grad_outputs,
        )
        return output

    def clamp_w(self) -> None:
        if self.softplus_W_kernels:
            return
        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)

    def penalize_w(self) -> torch.Tensor:
        return self.fnorm_penalty * sum(F.relu(-w.weight).norm() for w in self.W)


class RectifiedFlowOT(nn.Module):
    """
    Time-dependent vector field OT map trained via a rectified-flow / flow-matching objective.

    The learned map is `transport(x0)`: integrate x' = v(x,t) from t=0..1 starting at x0.
    """

    def __init__(
        self,
        d: int,
        hidden: List[int] = [256, 256],
        time_emb_dim: int = 64,
        act: str = "silu",
        transport_steps: int = 50,
    ):
        super().__init__()
        self.d = int(d)
        self.transport_steps = int(transport_steps)
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = MLP(self.d + time_emb_dim, self.d, hidden=hidden, act=act)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d)
        t: (B, 1) or (B,)
        returns: v(x,t) in R^d
        """
        te = self.time_emb(t)
        h = torch.cat([x, te], dim=-1)
        return self.mlp(h)

    @torch.no_grad()
    def transport(self, x: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Deterministic Euler integration of x' = v(x,t) from t=0..1.
        """
        steps = int(self.transport_steps if n_steps is None else n_steps)
        dt = 1.0 / float(max(1, steps))
        z = x.detach()
        for k in range(max(1, steps)):
            t = torch.full((z.shape[0], 1), float(k) / float(max(1, steps)), device=z.device, dtype=z.dtype)
            z = z + dt * self.forward(z, t)
        return z
