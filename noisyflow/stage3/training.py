from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
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
                clf.train()
            print(msg)

    out: Dict[str, float] = {"clf_loss": last_loss}
    if test_loader is not None:
        out.update(eval_classifier(clf, test_loader, device=device))
    return out


def _collect_numpy_xy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            n_ds = len(dataset)
        except TypeError:
            n_ds = None
        if n_ds is not None and n_ds > 0:
            # For RF we want the full dataset (DataLoader.drop_last can otherwise drop samples).
            batch_size = getattr(loader, "batch_size", None)
            if batch_size is None:
                batch_size = min(1024, n_ds)
            batch_size = max(1, min(int(batch_size), int(n_ds)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for xb, yb in loader:
        xs.append(xb.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64, copy=False)
    return X, y


def train_random_forest_classifier(
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    *,
    seed: int = 0,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    name: str = "Classifier/RF",
) -> Dict[str, float]:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for RandomForestClassifier (pip install scikit-learn).") from exc

    X_train, y_train = _collect_numpy_xy(train_loader)
    if X_train.size == 0:
        raise ValueError("Empty training set for random forest classifier")

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        random_state=int(seed),
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    out: Dict[str, float] = {
        "clf_loss": float("nan"),
        "train_n": float(len(y_train)),
    }
    if test_loader is not None:
        X_test, y_test = _collect_numpy_xy(test_loader)
        if X_test.size == 0:
            out["acc"] = float("nan")
        else:
            pred = clf.predict(X_test)
            out["acc"] = float((pred == y_test).mean())
        out["test_n"] = float(len(y_test))
        print(
            f"[{name}] train_n={int(out['train_n'])}  test_n={int(out['test_n'])}  n_estimators={n_estimators}  test_acc={out['acc']:.3f}"
        )
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
