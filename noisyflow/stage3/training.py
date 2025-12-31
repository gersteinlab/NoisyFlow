from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler
from noisyflow.stage2.networks import ICNN
from noisyflow.stage3.networks import Classifier


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
    Server-side synthesis in Eq. (server-synth).

    Each element in clients is a dict containing:
      - "flow": VelocityField (DP-trained)
      - "ot":   ICNN or CellOTICNN (DP-trained or post-processed)
      - optional "prior": tensor (C,)
    """
    ys: List[torch.Tensor] = []
    ls: List[torch.Tensor] = []
    for idx, c in enumerate(clients):
        flow: VelocityField = c["flow"].to(device).eval()
        ot: torch.nn.Module = c["ot"].to(device).eval()
        prior: Optional[torch.Tensor] = c.get("prior", None)
        if prior is None:
            prior = torch.ones(num_classes, device=device) / float(num_classes)
        else:
            prior = prior.to(device)

        labels = sample_labels_from_prior(prior, M_per_client).to(device)
        x_tilde = sample_flow_euler(flow, labels, n_steps=flow_steps)
        with torch.enable_grad():
            x_req = x_tilde.detach().requires_grad_(True)
            y_tilde = ot.transport(x_req)
        ys.append(y_tilde.detach().cpu())
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
