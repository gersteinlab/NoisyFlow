from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot, train_ot_stage2_rectified_flow
from noisyflow.stage3.networks import Classifier


def collect_losses(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> torch.Tensor:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            batch_losses = F.cross_entropy(logits, yb, reduction="none")
            losses.append(batch_losses.detach().cpu())
    if not losses:
        return torch.empty(0)
    return torch.cat(losses, dim=0)


def _balanced_sample(
    train_losses: np.ndarray,
    test_losses: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(len(train_losses), len(test_losses))
    if max_samples is not None:
        n = min(n, int(max_samples))

    if len(train_losses) > n:
        idx = rng.choice(len(train_losses), size=n, replace=False)
        train_losses = train_losses[idx]
    if len(test_losses) > n:
        idx = rng.choice(len(test_losses), size=n, replace=False)
        test_losses = test_losses[idx]

    return train_losses, test_losses


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def loss_threshold_attack(
    train_losses: torch.Tensor,
    test_losses: torch.Tensor,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    train_np = train_losses.detach().cpu().numpy().astype(np.float64)
    test_np = test_losses.detach().cpu().numpy().astype(np.float64)

    labels = np.concatenate([np.ones_like(train_np), np.zeros_like(test_np)])
    losses = np.concatenate([train_np, test_np])

    if threshold is None:
        order = np.argsort(losses)
        losses_sorted = losses[order]
        labels_sorted = labels[order]
        prefix_pos = np.cumsum(labels_sorted)
        total_pos = int(prefix_pos[-1]) if prefix_pos.size > 0 else 0
        total = int(labels_sorted.size)
        total_neg = total - total_pos

        pos_in_first_k = np.concatenate(([0], prefix_pos))
        k = np.arange(0, total + 1)
        neg_in_first_k = k - pos_in_first_k
        neg_in_rest = total_neg - neg_in_first_k
        acc = (pos_in_first_k + neg_in_rest) / float(total) if total > 0 else np.array([0.0])
        best_k = int(np.argmax(acc))

        if total == 0:
            threshold = 0.0
        elif best_k == 0:
            threshold = float(losses_sorted[0] - 1e-6)
        elif best_k >= total:
            threshold = float(losses_sorted[-1])
        else:
            threshold = float(0.5 * (losses_sorted[best_k - 1] + losses_sorted[best_k]))

    preds = (losses <= threshold).astype(np.int64)
    acc = float((preds == labels).mean()) if labels.size > 0 else float("nan")
    auc = _roc_auc(-losses, labels)
    advantage = acc - 0.5 if np.isfinite(acc) else float("nan")

    return {
        "attack_acc": acc,
        "attack_auc": auc,
        "attack_threshold": float(threshold),
        "attack_advantage": float(advantage),
    }


def run_loss_attack(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    train_losses = collect_losses(model, train_loader, device=device)
    test_losses = collect_losses(model, test_loader, device=device)

    if max_samples is not None:
        train_np, test_np = _balanced_sample(
            train_losses.numpy(),
            test_losses.numpy(),
            max_samples=max_samples,
            seed=seed,
        )
        train_losses = torch.from_numpy(train_np)
        test_losses = torch.from_numpy(test_np)

    return loss_threshold_attack(train_losses, test_losses)


def _standardize_features(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    return (x - mean) / std, mean, std


def _apply_standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _kernel_init_from_config(cfg: Dict[str, Any]) -> Optional[Callable[[torch.Tensor], None]]:
    if not cfg:
        return None
    name = str(cfg.get("name", "uniform")).lower()
    if name == "uniform":
        a = float(cfg.get("a", 0.0))
        b = float(cfg.get("b", 0.1))

        def init(tensor: torch.Tensor) -> None:
            torch.nn.init.uniform_(tensor, a=a, b=b)

        return init
    if name == "normal":
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 0.1))

        def init(tensor: torch.Tensor) -> None:
            torch.nn.init.normal_(tensor, mean=mean, std=std)

        return init
    raise ValueError(f"Unknown kernel_init name '{name}'")


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_set: str = "stats",
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("MIA feature extraction requires (x, y) batches.")
            xb, yb = batch[0], batch[1]
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, yb, reduction="none").unsqueeze(1)

            if feature_set == "loss":
                feat = loss
            elif feature_set == "probs":
                feat = probs
            elif feature_set == "logits":
                feat = logits
            elif feature_set == "stats":
                max_prob = probs.max(dim=1, keepdim=True).values
                entropy = -(probs * (probs + 1e-12).log()).sum(dim=1, keepdim=True)
                if probs.shape[1] >= 2:
                    top2 = torch.topk(probs, k=2, dim=1).values
                    margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                else:
                    margin = probs[:, :1]
                correct = (logits.argmax(dim=1) == yb).float().unsqueeze(1)
                feat = torch.cat([loss, max_prob, entropy, correct, margin], dim=1)
            else:
                raise ValueError("feature_set must be one of 'loss', 'stats', 'probs', 'logits'")

            feats.append(feat.detach().cpu())

    if not feats:
        return torch.empty(0)
    return torch.cat(feats, dim=0)


class AttackMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_binary_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def _train_classifier(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def _attack_metrics(scores: np.ndarray, labels: np.ndarray, prefix: str) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (scores >= 0.0).astype(np.int64)
    acc = float((preds == labels).mean()) if labels.size > 0 else float("nan")
    auc = _roc_auc(scores, labels)
    advantage = acc - 0.5 if np.isfinite(acc) else float("nan")
    return {
        f"{prefix}_acc": acc,
        f"{prefix}_auc": auc,
        f"{prefix}_advantage": float(advantage),
    }


def _balanced_feature_sample(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n = min(len(train_x), len(test_x))
    if max_samples is not None:
        n = min(n, int(max_samples))
    if len(train_x) > n:
        idx = rng.choice(len(train_x), size=n, replace=False)
        train_x = train_x[idx]
    if len(test_x) > n:
        idx = rng.choice(len(test_x), size=n, replace=False)
        test_x = test_x[idx]
    return train_x, test_x


def run_shadow_attack(
    data_builder: Callable[..., Tuple[List[TensorDataset], TensorDataset, TensorDataset]],
    data_params: Dict[str, Any],
    d: int,
    num_classes: int,
    target_model: torch.nn.Module,
    target_member_loader: DataLoader,
    target_nonmember_loader: DataLoader,
    num_shadow_models: int = 2,
    shadow_train_size: int = 2000,
    shadow_test_size: int = 2000,
    shadow_epochs: int = 5,
    shadow_lr: float = 1e-3,
    shadow_hidden: Optional[List[int]] = None,
    shadow_batch_size: int = 256,
    attack_epochs: int = 20,
    attack_lr: float = 1e-3,
    attack_hidden: Optional[List[int]] = None,
    attack_batch_size: int = 256,
    feature_set: str = "stats",
    max_samples_per_shadow: Optional[int] = 2000,
    seed: int = 0,
    data_overrides: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    if shadow_hidden is None:
        shadow_hidden = [128, 128]
    if attack_hidden is None:
        attack_hidden = [64, 32]

    attack_feats: List[torch.Tensor] = []
    attack_labels: List[torch.Tensor] = []

    base_seed = int(data_params.get("seed", 0))
    for i in range(num_shadow_models):
        params = dict(data_params)
        if data_overrides:
            params.update(data_overrides)
        params["seed"] = base_seed + seed + 1000 + i
        params["n_target_test"] = shadow_train_size + shadow_test_size
        params.setdefault("n_target_ref", 1)

        _, _, target_test = data_builder(**params)
        x_all, y_all = target_test.tensors
        if x_all.shape[0] < shadow_train_size + shadow_test_size:
            raise ValueError("shadow dataset too small for requested train/test sizes")

        rng = np.random.default_rng(seed + i)
        perm = rng.permutation(x_all.shape[0])
        train_idx = perm[:shadow_train_size]
        test_idx = perm[shadow_train_size : shadow_train_size + shadow_test_size]

        shadow_train = TensorDataset(x_all[train_idx], y_all[train_idx])
        shadow_test = TensorDataset(x_all[test_idx], y_all[test_idx])

        shadow_train_loader = DataLoader(
            shadow_train, batch_size=shadow_batch_size, shuffle=True, drop_last=False
        )
        shadow_eval_train_loader = DataLoader(
            shadow_train, batch_size=shadow_batch_size, shuffle=False, drop_last=False
        )
        shadow_eval_test_loader = DataLoader(
            shadow_test, batch_size=shadow_batch_size, shuffle=False, drop_last=False
        )

        shadow_model = Classifier(d=d, num_classes=num_classes, hidden=shadow_hidden)
        _train_classifier(shadow_model, shadow_train_loader, epochs=shadow_epochs, lr=shadow_lr, device=device)

        train_feat = extract_features(shadow_model, shadow_eval_train_loader, feature_set, device=device)
        test_feat = extract_features(shadow_model, shadow_eval_test_loader, feature_set, device=device)

        train_feat, test_feat = _balanced_feature_sample(
            train_feat, test_feat, max_samples_per_shadow, seed=seed + i
        )

        attack_feats.append(train_feat)
        attack_labels.append(torch.ones(train_feat.shape[0], dtype=torch.float32))
        attack_feats.append(test_feat)
        attack_labels.append(torch.zeros(test_feat.shape[0], dtype=torch.float32))

    if not attack_feats:
        raise RuntimeError("No shadow features collected for attack model training.")

    attack_x = torch.cat(attack_feats, dim=0)
    attack_y = torch.cat(attack_labels, dim=0)

    perm = torch.randperm(attack_x.shape[0])
    attack_x = attack_x[perm]
    attack_y = attack_y[perm]

    attack_x, mean, std = _standardize_features(attack_x)

    attack_model = AttackMLP(in_dim=attack_x.shape[1], hidden=attack_hidden)
    _train_binary_classifier(
        attack_model,
        attack_x,
        attack_y,
        epochs=attack_epochs,
        lr=attack_lr,
        batch_size=attack_batch_size,
        device=device,
    )

    member_feat = extract_features(target_model, target_member_loader, feature_set, device=device)
    nonmember_feat = extract_features(target_model, target_nonmember_loader, feature_set, device=device)

    member_feat = _apply_standardize(member_feat, mean, std)
    nonmember_feat = _apply_standardize(nonmember_feat, mean, std)

    attack_model.eval()
    with torch.no_grad():
        member_scores = attack_model(member_feat.to(device)).detach().cpu().numpy()
        nonmember_scores = attack_model(nonmember_feat.to(device)).detach().cpu().numpy()

    labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    scores = np.concatenate([member_scores, nonmember_scores])
    metrics = _attack_metrics(scores, labels, prefix="shadow_attack")
    metrics["shadow_feature_set"] = feature_set
    return metrics


def flow_matching_loss_per_example(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_samples: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    total = torch.zeros(x.shape[0], device=x.device)
    for _ in range(max(1, num_samples)):
        z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        t = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype, generator=gen)
        x_t = (1.0 - t) * z + t * x
        v_star = x - z
        v = model(x_t, t, y)
        total = total + ((v - v_star) ** 2).sum(dim=1)
    return total / float(max(1, num_samples))


def collect_stage_features(
    flow: torch.nn.Module,
    ot: Optional[torch.nn.Module],
    loader: DataLoader,
    use_ot: bool,
    num_flow_samples: int = 1,
    include_ot_transport_norm: bool = True,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    flow.eval()
    if ot is not None:
        ot.eval()
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            flow_loss = flow_matching_loss_per_example(
                flow, xb, yb, num_samples=num_flow_samples, seed=seed + batch_idx
            ).unsqueeze(1)
            batch_feats = [flow_loss]
            if use_ot and ot is not None:
                phi = None
                try:
                    phi = ot(xb)
                except TypeError:
                    phi = None
                if isinstance(phi, torch.Tensor):
                    if phi.dim() == 2 and phi.shape[1] == 1:
                        phi = phi.squeeze(1)
                    if phi.dim() == 1:
                        batch_feats.append(phi.unsqueeze(1))
                if include_ot_transport_norm:
                    with torch.enable_grad():
                        xb_req = xb.detach().requires_grad_(True)
                        transport = ot.transport(xb_req)
                    tnorm = transport.norm(dim=1, keepdim=True).detach()
                    batch_feats.append(tnorm)
            features.append(torch.cat(batch_feats, dim=1).detach().cpu())
    if not features:
        return torch.empty(0)
    return torch.cat(features, dim=0)


def run_stage_mia_attack(
    member_features: torch.Tensor,
    nonmember_features: torch.Tensor,
    attack_hidden: List[int],
    attack_epochs: int,
    attack_lr: float,
    attack_batch_size: int,
    attack_train_frac: float = 0.5,
    max_samples: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, float]:
    member_features, nonmember_features = _balanced_feature_sample(
        member_features, nonmember_features, max_samples=max_samples, seed=seed
    )
    if member_features.shape[0] == 0 or nonmember_features.shape[0] == 0:
        raise RuntimeError("Insufficient samples for stage MIA attack.")

    rng = np.random.default_rng(seed)
    n = min(member_features.shape[0], nonmember_features.shape[0])
    n_train = max(1, int(n * attack_train_frac))
    n_train = min(n_train, n - 1) if n > 1 else n_train

    mem_idx = rng.permutation(member_features.shape[0])
    non_idx = rng.permutation(nonmember_features.shape[0])
    mem_train = member_features[mem_idx[:n_train]]
    mem_eval = member_features[mem_idx[n_train:n]]
    non_train = nonmember_features[non_idx[:n_train]]
    non_eval = nonmember_features[non_idx[n_train:n]]

    attack_x_train = torch.cat([mem_train, non_train], dim=0)
    attack_y_train = torch.cat(
        [
            torch.ones(mem_train.shape[0], dtype=torch.float32),
            torch.zeros(non_train.shape[0], dtype=torch.float32),
        ],
        dim=0,
    )

    attack_x_train, mean, std = _standardize_features(attack_x_train)

    attack_model = AttackMLP(in_dim=attack_x_train.shape[1], hidden=attack_hidden)
    _train_binary_classifier(
        attack_model,
        attack_x_train,
        attack_y_train,
        epochs=attack_epochs,
        lr=attack_lr,
        batch_size=attack_batch_size,
        device=device,
    )

    attack_x_eval = torch.cat([mem_eval, non_eval], dim=0)
    attack_y_eval = np.concatenate(
        [
            np.ones(mem_eval.shape[0], dtype=np.int64),
            np.zeros(non_eval.shape[0], dtype=np.int64),
        ]
    )
    attack_x_eval = _apply_standardize(attack_x_eval, mean, std)

    attack_model.eval()
    with torch.no_grad():
        scores = attack_model(attack_x_eval.to(device)).detach().cpu().numpy()

    metrics = _attack_metrics(scores, attack_y_eval, prefix="stage_mia_attack")
    metrics["stage_mia_train_frac"] = float(attack_train_frac)
    return metrics


def _split_dataset(ds: TensorDataset, holdout_fraction: float, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    if holdout_fraction <= 0.0:
        raise ValueError("holdout_fraction must be > 0 for stage shadow MIA")
    n = ds.tensors[0].shape[0]
    n_holdout = max(1, int(n * holdout_fraction))
    n_holdout = min(n_holdout, n - 1) if n > 1 else n_holdout
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    hold_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_tensors = [t[train_idx] for t in ds.tensors]
    hold_tensors = [t[hold_idx] for t in ds.tensors]
    return TensorDataset(*train_tensors), TensorDataset(*hold_tensors)


def run_stage_shadow_attack(
    data_builder: Callable[..., Tuple[List[TensorDataset], TensorDataset, TensorDataset]],
    data_params: Dict[str, Any],
    target_clients: List[Dict[str, Any]],
    flow_kwargs: Dict[str, Any],
    ot_kwargs: Dict[str, Any],
    stage2_option: str,
    stage1_train_kwargs: Dict[str, Any],
    stage2_train_kwargs: Dict[str, Any],
    batch_size: int,
    target_batch_size: int,
    drop_last: bool,
    num_shadow_models: int,
    holdout_fraction: float,
    num_flow_samples: int,
    include_ot_transport_norm: bool,
    attack_hidden: List[int],
    attack_epochs: int,
    attack_lr: float,
    attack_batch_size: int,
    attack_train_frac: float,
    max_samples_per_shadow: Optional[int],
    seed: int,
    data_overrides: Optional[Dict[str, Any]],
    cellot_enabled: bool = False,
    cellot_hidden_units: Optional[List[int]] = None,
    cellot_activation: str = "LeakyReLU",
    cellot_softplus_W_kernels: bool = False,
    cellot_softplus_beta: float = 1.0,
    cellot_kernel_init: Optional[Dict[str, Any]] = None,
    cellot_f_fnorm_penalty: float = 0.0,
    cellot_g_fnorm_penalty: float = 0.0,
    cellot_n_inner_iters: int = 10,
    cellot_optim: Optional[Dict[str, Any]] = None,
    cellot_n_iters: Optional[int] = None,
    rectified_flow_enabled: bool = False,
    rectified_flow_hidden: Optional[List[int]] = None,
    rectified_flow_time_emb_dim: int = 64,
    rectified_flow_act: str = "silu",
    rectified_flow_transport_steps: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    use_ot = stage2_option.upper() in {"A", "C"}
    if cellot_enabled and rectified_flow_enabled:
        raise ValueError("Shadow stage OT can only use one of CellOT or RectifiedFlow.")
    attack_feats: List[torch.Tensor] = []
    attack_labels: List[torch.Tensor] = []

    base_seed = int(data_params.get("seed", 0))
    for i in range(num_shadow_models):
        params = dict(data_params)
        if data_overrides:
            params.update(data_overrides)
        params["seed"] = base_seed + seed + 2000 + i
        client_datasets, target_ref, _ = data_builder(**params)

        target_loader = None
        if use_ot:
            target_loader = DataLoader(
                target_ref,
                batch_size=target_batch_size,
                shuffle=True,
                drop_last=drop_last,
            )

        for c_idx, ds in enumerate(client_datasets):
            train_ds, holdout_ds = _split_dataset(ds, holdout_fraction, seed=seed + i + c_idx)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
            )

            flow = VelocityField(**flow_kwargs)
            train_flow_stage1(
                flow,
                train_loader,
                epochs=int(stage1_train_kwargs.get("epochs", 1)),
                lr=float(stage1_train_kwargs.get("lr", 1e-3)),
                dp=None,
                device=device,
            )

            ot = None
            if use_ot:
                real_x_loader = DataLoader(
                    TensorDataset(train_ds.tensors[0]),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                )

                if cellot_enabled:
                    if stage2_option.upper() != "A":
                        raise ValueError("CellOT shadow training supports stage2.option A only.")
                    kernel_init = _kernel_init_from_config(cellot_kernel_init or {})
                    f = CellOTICNN(
                        input_dim=ot_kwargs["d"],
                        hidden_units=cellot_hidden_units or [64, 64, 64, 64],
                        activation=cellot_activation,
                        softplus_W_kernels=cellot_softplus_W_kernels,
                        softplus_beta=cellot_softplus_beta,
                        fnorm_penalty=cellot_f_fnorm_penalty,
                        kernel_init_fxn=kernel_init,
                    )
                    ot = CellOTICNN(
                        input_dim=ot_kwargs["d"],
                        hidden_units=cellot_hidden_units or [64, 64, 64, 64],
                        activation=cellot_activation,
                        softplus_W_kernels=cellot_softplus_W_kernels,
                        softplus_beta=cellot_softplus_beta,
                        fnorm_penalty=cellot_g_fnorm_penalty,
                        kernel_init_fxn=kernel_init,
                    )
                    train_ot_stage2_cellot(
                        f,
                        ot,
                        source_loader=real_x_loader,
                        target_loader=target_loader,
                        epochs=int(stage2_train_kwargs.get("epochs", 1)),
                        n_inner_iters=int(stage2_train_kwargs.get("n_inner_iters", cellot_n_inner_iters)),
                        lr_f=float(stage2_train_kwargs.get("lr", 1e-3)),
                        lr_g=float(stage2_train_kwargs.get("lr", 1e-3)),
                        optim_cfg=cellot_optim,
                        n_iters=cellot_n_iters,
                        dp=None,
                        device=device,
                    )
                elif rectified_flow_enabled:
                    if stage2_option.upper() != "A":
                        raise ValueError("RectifiedFlow shadow training supports stage2.option A only.")
                    ot = RectifiedFlowOT(
                        d=ot_kwargs["d"],
                        hidden=rectified_flow_hidden or [256, 256],
                        time_emb_dim=int(rectified_flow_time_emb_dim),
                        act=str(rectified_flow_act),
                        transport_steps=int(rectified_flow_transport_steps),
                    )
                    train_ot_stage2_rectified_flow(
                        ot,
                        source_loader=real_x_loader,
                        target_loader=target_loader,
                        option="A",
                        synth_sampler=None,
                        epochs=int(stage2_train_kwargs.get("epochs", 1)),
                        lr=float(stage2_train_kwargs.get("lr", 1e-3)),
                        dp=None,
                        device=device,
                    )
                else:
                    ot = ICNN(**ot_kwargs)

                    def synth_sampler(bs: int, flow_model=flow) -> torch.Tensor:
                        labels = torch.randint(0, flow_kwargs["num_classes"], (bs,), device=device)
                        return sample_flow_euler(
                            flow_model.to(device).eval(),
                            labels,
                            n_steps=int(stage2_train_kwargs.get("flow_steps", 50)),
                        ).cpu()

                    train_ot_stage2(
                        ot,
                        real_loader=real_x_loader if stage2_option.upper() in {"A", "C"} else None,
                        target_loader=target_loader,
                        option=stage2_option,
                        synth_sampler=(lambda bs: synth_sampler(bs))
                        if stage2_option.upper() in {"B", "C"}
                        else None,
                        epochs=int(stage2_train_kwargs.get("epochs", 1)),
                        lr=float(stage2_train_kwargs.get("lr", 1e-3)),
                        dp=None,
                        conj_steps=int(stage2_train_kwargs.get("conj_steps", 5)),
                        conj_lr=float(stage2_train_kwargs.get("conj_lr", 0.1)),
                        conj_clamp=stage2_train_kwargs.get("conj_clamp", None),
                        device=device,
                    )

            member_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            nonmember_loader = DataLoader(
                holdout_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            mem_feat = collect_stage_features(
                flow,
                ot,
                member_loader,
                use_ot=use_ot,
                num_flow_samples=num_flow_samples,
                include_ot_transport_norm=include_ot_transport_norm,
                seed=seed + i + c_idx,
                device=device,
            )
            non_feat = collect_stage_features(
                flow,
                ot,
                nonmember_loader,
                use_ot=use_ot,
                num_flow_samples=num_flow_samples,
                include_ot_transport_norm=include_ot_transport_norm,
                seed=seed + i + c_idx + 100,
                device=device,
            )

            mem_feat, non_feat = _balanced_feature_sample(
                mem_feat, non_feat, max_samples=max_samples_per_shadow, seed=seed + i + c_idx
            )

            attack_feats.append(mem_feat)
            attack_labels.append(torch.ones(mem_feat.shape[0], dtype=torch.float32))
            attack_feats.append(non_feat)
            attack_labels.append(torch.zeros(non_feat.shape[0], dtype=torch.float32))

    if not attack_feats:
        raise RuntimeError("No shadow stage features collected for attack training.")

    attack_x = torch.cat(attack_feats, dim=0)
    attack_y = torch.cat(attack_labels, dim=0)

    perm = torch.randperm(attack_x.shape[0])
    attack_x = attack_x[perm]
    attack_y = attack_y[perm]

    attack_x, mean, std = _standardize_features(attack_x)
    attack_model = AttackMLP(in_dim=attack_x.shape[1], hidden=attack_hidden)
    _train_binary_classifier(
        attack_model,
        attack_x,
        attack_y,
        epochs=attack_epochs,
        lr=attack_lr,
        batch_size=attack_batch_size,
        device=device,
    )

    target_member_feats: List[torch.Tensor] = []
    target_nonmember_feats: List[torch.Tensor] = []
    for entry in target_clients:
        flow = entry["flow"].to(device)
        ot = entry["ot"].to(device) if use_ot else None
        member_loader = DataLoader(
            entry["members"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        nonmember_loader = DataLoader(
            entry["nonmembers"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        target_member_feats.append(
            collect_stage_features(
                flow,
                ot,
                member_loader,
                use_ot=use_ot,
                num_flow_samples=num_flow_samples,
                include_ot_transport_norm=include_ot_transport_norm,
                seed=seed,
                device=device,
            )
        )
        target_nonmember_feats.append(
            collect_stage_features(
                flow,
                ot,
                nonmember_loader,
                use_ot=use_ot,
                num_flow_samples=num_flow_samples,
                include_ot_transport_norm=include_ot_transport_norm,
                seed=seed + 123,
                device=device,
            )
        )

    member_feat = torch.cat(target_member_feats, dim=0) if target_member_feats else torch.empty(0)
    nonmember_feat = torch.cat(target_nonmember_feats, dim=0) if target_nonmember_feats else torch.empty(0)

    member_feat = _apply_standardize(member_feat, mean, std)
    nonmember_feat = _apply_standardize(nonmember_feat, mean, std)

    attack_model.eval()
    with torch.no_grad():
        member_scores = attack_model(member_feat.to(device)).detach().cpu().numpy()
        nonmember_scores = attack_model(nonmember_feat.to(device)).detach().cpu().numpy()

    labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    scores = np.concatenate([member_scores, nonmember_scores])
    metrics = _attack_metrics(scores, labels, prefix="stage_shadow_mia")
    return metrics
