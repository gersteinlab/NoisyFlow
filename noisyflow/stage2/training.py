from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.utils import DPConfig, cycle, unwrap_model


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
        obj = (x * y).sum(dim=1) - phi(x)
        loss = -obj.mean()
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
        with torch.no_grad():
            x = x - lr * grad
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
    """
    phi_x = phi(x)
    phi_star_y, _ = approx_conjugate(phi, y, n_steps=conj_steps, lr=conj_lr, clamp=conj_clamp)
    J = phi_x.mean() + phi_star_y.mean()
    return -J


def compute_loss_g(
    f: CellOTICNN,
    g: CellOTICNN,
    source: torch.Tensor,
    transport: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if transport is None:
        transport = g.transport(source)
    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


def compute_loss_f(
    f: CellOTICNN,
    g: CellOTICNN,
    source: torch.Tensor,
    target: torch.Tensor,
    transport: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if transport is None:
        transport = g.transport(source)
    return -f(transport) + f(target)


def rectified_flow_ot_loss(
    v: RectifiedFlowOT,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Rectified-flow objective between source and target batches.

    Sample t ~ Unif[0,1], set x_t = (1-t) source + t target, and regress v(x_t,t) to (target - source).
    """
    if source.shape != target.shape:
        raise ValueError(f"source and target must have the same shape, got {source.shape} vs {target.shape}")
    t = torch.rand(source.shape[0], 1, device=source.device, dtype=source.dtype)
    x_t = (1.0 - t) * source + t * target
    v_star = target - source
    v_pred = v(x_t, t)
    return ((v_pred - v_star) ** 2).sum(dim=1).mean()


def train_ot_stage2_rectified_flow(
    v: RectifiedFlowOT,
    source_loader: Optional[DataLoader],
    target_loader: DataLoader,
    option: str = "A",
    pair_by_label: bool = False,
    pair_by_ot: bool = False,
    synth_sampler: Optional[Callable[..., torch.Tensor]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Stage II alternative OT model: learn a time-dependent vector field via rectified flow / flow matching.

    Options:
      - "A": source from private real data (supports DP-SGD on v)
      - "B": source from synthetic sampler (post-processing; ignores dp)
      - "C": mix private real data + synthetic source (supports DP-SGD on v)
    """
    option = option.upper()
    if option not in {"A", "B", "C"}:
        raise ValueError("RectifiedFlow OT supports option 'A', 'B', or 'C'.")
    if option in {"A", "C"} and source_loader is None:
        raise ValueError(f"source_loader required for RectifiedFlow option {option}")
    if option in {"B", "C"} and synth_sampler is None:
        raise ValueError(f"synth_sampler required for RectifiedFlow option {option}")

    def _as_x_and_label(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            label = batch[1] if len(batch) >= 2 else None
            return x, label
        return batch, None

    def _sample_target(target_iter, batch_size: int) -> torch.Tensor:
        chunks = []
        n = 0
        while n < batch_size:
            tb, _ = _as_x_and_label(next(target_iter))
            chunks.append(tb)
            n += int(tb.shape[0])
        out = torch.cat(chunks, dim=0)
        return out[:batch_size]

    def _hungarian_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape != target.shape:
            raise ValueError(f"OT matching requires same shapes, got {source.shape} vs {target.shape}")
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pair_by_ot requires SciPy (pip install scipy).") from exc

        xs = source.detach().cpu().numpy()
        ys = target.detach().cpu().numpy()
        cost = ((xs[:, None, :] - ys[None, :, :]) ** 2).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.empty_like(col_ind)
        perm[row_ind] = col_ind
        perm_t = torch.from_numpy(perm).to(device=target.device, dtype=torch.long)
        return target.index_select(0, perm_t)

    def _match_target(
        source: torch.Tensor,
        target: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not pair_by_ot:
            return target
        if labels is None:
            return _hungarian_match(source, target)
        labels = labels.to(source.device).long().view(-1)
        out = target.clone()
        for c in labels.unique().tolist():
            c_int = int(c)
            mask = labels == c_int
            idx = mask.nonzero(as_tuple=False).view(-1)
            if int(idx.numel()) <= 1:
                continue
            out[idx] = _hungarian_match(source[idx], target[idx])
        return out

    target_pool_by_label: Optional[Dict[int, torch.Tensor]] = None
    if pair_by_label and option in {"A", "C"}:
        if isinstance(getattr(target_loader, "dataset", None), TensorDataset) and len(target_loader.dataset.tensors) >= 2:
            ys_all = target_loader.dataset.tensors[0].to(device).float()
            ls_all = target_loader.dataset.tensors[1].to(device).long().view(-1)
            if ls_all.numel() == 0:
                print("[Stage II/RectifiedFlow] WARNING: target labels empty; disabling pair_by_label.")
                pair_by_label = False
            else:
                num_classes = int(ls_all.max().item() + 1)
                target_pool_by_label = {
                    c: ys_all[ls_all == c] for c in range(num_classes)
                }
                empty = [c for c, pool in target_pool_by_label.items() if int(pool.shape[0]) == 0]
                if empty:
                    print(
                        f"[Stage II/RectifiedFlow] WARNING: no target samples for labels {empty}; disabling pair_by_label."
                    )
                    target_pool_by_label = None
                    pair_by_label = False
        else:
            print(
                "[Stage II/RectifiedFlow] WARNING: pair_by_label requires target_loader.dataset to be a labeled TensorDataset; disabling."
            )
            pair_by_label = False

    def _sample_target_matched(labels: torch.Tensor) -> torch.Tensor:
        assert target_pool_by_label is not None
        # Assume all target pools share the same feature shape (N, d).
        some_pool = next(iter(target_pool_by_label.values()))
        if some_pool.dim() != 2:
            raise ValueError(f"Expected target features to have shape (N,d), got {tuple(some_pool.shape)}")
        d = int(some_pool.shape[1])
        out = torch.empty((labels.shape[0], d), device=device, dtype=some_pool.dtype)
        for c in labels.unique().tolist():
            c_int = int(c)
            mask = labels == c_int
            pool = target_pool_by_label.get(c_int, None)
            if pool is None or int(pool.shape[0]) == 0:
                raise ValueError(f"No target pool available for label {c_int}")
            idx = torch.randint(0, int(pool.shape[0]), (int(mask.sum().item()),), device=device)
            out[mask] = pool[idx]
        return out

    v.to(device)
    v.train()
    opt = torch.optim.Adam(v.parameters(), lr=lr)

    privacy_engine = None
    if dp is not None and dp.enabled and option in {"A", "C"}:
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
        v, opt, source_loader = _make_private_with_mode(
            privacy_engine,
            module=v,
            optimizer=opt,
            data_loader=source_loader,
            dp=dp,
            grad_sample_mode=getattr(dp, "grad_sample_mode", None),
        )
        if pair_by_ot:
            print(
                "[Stage II/RectifiedFlow] WARNING: pair_by_ot is ignored when using DP-SGD (stage2.option in {'A','C'}) because OT matching couples samples within a batch."
            )
            pair_by_ot = False

    target_iter = cycle(target_loader)

    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        if option == "A":
            assert source_loader is not None
            for xb in source_loader:
                xb, x_labels = _as_x_and_label(xb)
                xb = xb.to(device).float()
                if pair_by_label and x_labels is not None and target_pool_by_label is not None:
                    x_labels = x_labels.to(device).long().view(-1)
                    yb = _sample_target_matched(x_labels)
                else:
                    yb = _sample_target(target_iter, xb.shape[0]).to(device).float()
                yb = _match_target(xb, yb, labels=x_labels if pair_by_label else None)

                loss = rectified_flow_ot_loss(v, xb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu().item())
        elif option == "C":
            assert source_loader is not None
            assert synth_sampler is not None
            for xb in source_loader:
                xb, x_labels = _as_x_and_label(xb)
                xb = xb.to(device).float()
                if x_labels is not None:
                    x_labels = x_labels.to(device).long().view(-1)

                if pair_by_label and x_labels is not None and target_pool_by_label is not None:
                    yb_real = _sample_target_matched(x_labels)
                    yb_syn = _sample_target_matched(x_labels)
                    try:
                        xb_syn = synth_sampler(xb.shape[0], labels=x_labels).to(device).float()  # type: ignore[misc]
                    except TypeError:
                        xb_syn = synth_sampler(xb.shape[0]).to(device).float()  # type: ignore[misc]
                    labels_syn = x_labels
                else:
                    yb_real = _sample_target(target_iter, xb.shape[0]).to(device).float()
                    yb_syn = _sample_target(target_iter, xb.shape[0]).to(device).float()
                    xb_syn = synth_sampler(xb.shape[0]).to(device).float()  # type: ignore[misc]
                    labels_syn = None

                xb_cat = torch.cat([xb, xb_syn], dim=0)
                yb_cat = torch.cat([yb_real, yb_syn], dim=0)
                if pair_by_label and x_labels is not None:
                    labels_cat = torch.cat([x_labels, labels_syn if labels_syn is not None else x_labels], dim=0)
                else:
                    labels_cat = None
                yb_cat = _match_target(xb_cat, yb_cat, labels=labels_cat if pair_by_label else None)

                loss = rectified_flow_ot_loss(v, xb_cat, yb_cat)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu().item())
        else:
            for yb in target_loader:
                yb, y_labels = _as_x_and_label(yb)
                yb = yb.to(device).float()
                if pair_by_label and y_labels is not None:
                    y_labels = y_labels.to(device).long().view(-1)
                    try:
                        xb = synth_sampler(yb.shape[0], labels=y_labels).to(device).float()  # type: ignore[misc]
                    except TypeError:
                        xb = synth_sampler(yb.shape[0]).to(device).float()  # type: ignore[misc]
                else:
                    xb = synth_sampler(yb.shape[0]).to(device).float()  # type: ignore[misc]
                yb = _match_target(xb, yb, labels=y_labels if pair_by_label else None)

                loss = rectified_flow_ot_loss(v, xb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu().item())

        if ep % max(1, epochs // 5) == 0:
            print(f"[Stage II/RectifiedFlow-{option}] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}")

    out: Dict[str, float] = {"ot_loss": last_loss}
    if privacy_engine is not None and dp is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon_ot"] = eps
        out["delta_ot"] = float(dp.delta)
        print(f"[Stage II/RectifiedFlow-{option}] DP eps={eps:.3f}, delta={dp.delta:g}")
    return out


def _make_private_with_mode(
    privacy_engine,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp: DPConfig,
    grad_sample_mode: Optional[str] = None,
    require_mode: bool = False,
):
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
        except TypeError as exc:
            if require_mode:
                raise RuntimeError(
                    "Opacus does not support grad_sample_mode on this version. "
                    "Upgrade opacus to use CellOT DP training."
                ) from exc
    return privacy_engine.make_private(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
    )


def _build_cellot_optimizer(
    params,
    optim_cfg: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> torch.optim.Optimizer:
    cfg = dict(optim_cfg or {})
    cfg.update(overrides or {})
    optimizer_name = str(cfg.pop("optimizer", "Adam"))
    if optimizer_name != "Adam":
        raise ValueError("CellOT optimizer must be Adam to match reference implementation.")
    lr = float(cfg.pop("lr", 1e-4))
    beta1 = float(cfg.pop("beta1", 0.5))
    beta2 = float(cfg.pop("beta2", 0.9))
    weight_decay = float(cfg.pop("weight_decay", 0.0))
    return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)


def train_ot_stage2(
    phi: ICNN,
    real_loader: Optional[DataLoader],
    target_loader: DataLoader,
    option: str = "B",
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

    Options:
      - "A": real data only, DP-SGD on theta
      - "B": synthetic only, non-private SGD (post-processing of DP flow)
      - "C": mixed. Here we implement a safe simplified variant:
            concatenate real+synthetic and (optionally) run DP-SGD on the whole loss.
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
    if dp is not None and dp.enabled and option in {"A", "C"}:
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
        phi, opt, real_loader = _make_private_with_mode(
            privacy_engine,
            module=phi,
            optimizer=opt,
            data_loader=real_loader,
            dp=dp,
            grad_sample_mode=getattr(dp, "grad_sample_mode", None),
        )

    y_iter = cycle(target_loader)
    if real_loader is not None:
        x_iter = cycle(real_loader)

    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        steps = len(real_loader) if (option in {"A", "C"} and real_loader is not None) else len(target_loader)
        for _ in range(steps):
            yb = next(y_iter)
            if isinstance(yb, (list, tuple)):
                yb = yb[0]
            yb = yb.to(device).float()

            if option == "A":
                xb = next(x_iter)
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(device).float()
            elif option == "B":
                xb = synth_sampler(yb.shape[0]).to(device).float()
            else:
                xr = next(x_iter)
                if isinstance(xr, (list, tuple)):
                    xr = xr[0]
                xr = xr.to(device).float()
                xs = synth_sampler(xr.shape[0]).to(device).float()
                xb = torch.cat([xr, xs], dim=0)
                y2 = next(y_iter)
                if isinstance(y2, (list, tuple)):
                    y2 = y2[0]
                y2 = y2.to(device).float()
                yb = torch.cat([yb, y2], dim=0)

            loss = ot_dual_loss(
                phi,
                xb,
                yb,
                conj_steps=conj_steps,
                conj_lr=conj_lr,
                conj_clamp=conj_clamp,
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
        print(f"[Stage II/{option}] DP eps={eps:.3f}, delta={dp.delta:g}")
    return out


def train_ot_stage2_cellot(
    f: CellOTICNN,
    g: CellOTICNN,
    source_loader: DataLoader,
    target_loader: DataLoader,
    epochs: int = 10,
    n_inner_iters: int = 10,
    lr_f: Optional[float] = 1e-4,
    lr_g: Optional[float] = 1e-4,
    optim_cfg: Optional[Dict[str, Any]] = None,
    n_iters: Optional[int] = None,
    dp: Optional[DPConfig] = None,
    device: str = "cpu",
    synth_sampler: Optional[Callable[[int], torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    CellOT-style Stage II training with two ICNNs:
      - g: transport potential (trained on source, DP if enabled)
      - f: critic potential (trained on public target + g(source or synthetic source))
    If n_iters is provided, uses it as the total number of update steps.
    optim_cfg mirrors CellOT's Adam settings with optional f/g overrides.
    If DP is enabled, f updates can use a synthetic source sampler to avoid
    touching private source batches outside the DP loop.
    """
    f.to(device)
    g.to(device)
    f.train()
    g.train()

    base_optim = dict(optim_cfg or {})
    f_overrides = dict(base_optim.pop("f", {}) or {})
    g_overrides = dict(base_optim.pop("g", {}) or {})
    if lr_f is not None and "lr" not in f_overrides and "lr" not in base_optim:
        f_overrides["lr"] = lr_f
    if lr_g is not None and "lr" not in g_overrides and "lr" not in base_optim:
        g_overrides["lr"] = lr_g
    opt_f = _build_cellot_optimizer(f.parameters(), base_optim, f_overrides)
    opt_g = _build_cellot_optimizer(g.parameters(), base_optim, g_overrides)

    privacy_engine = None
    if dp is not None and dp.enabled:
        try:
            from opacus import PrivacyEngine
        except Exception as e:
            raise RuntimeError(
                "Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP."
            ) from e
        requested_mode = getattr(dp, "grad_sample_mode", None)
        mode = str(requested_mode).lower() if requested_mode is not None else None
        # CellOT training uses higher-order gradients (via autograd.grad on the transport map).
        # Opacus hook-based modes ("hooks"/"functorch") can miss per-sample gradients for some
        # parameters in this setting. ExpandedWeights ("ew") is robust here.
        if mode in {None, "ew", "expanded_weights", "expandedweights"}:
            mode = "ew"
        elif mode in {"hooks", "functorch"}:
            print(
                f"[Stage II/CellOT] WARNING: dp.grad_sample_mode='{requested_mode}' is not compatible with CellOT OT gradients; using 'ew' instead."
            )
            mode = "ew"
        else:
            raise ValueError(
                f"Unsupported dp.grad_sample_mode='{requested_mode}' for CellOT DP. Use 'ew'."
            )
        try:
            privacy_engine = PrivacyEngine(secure_mode=getattr(dp, "secure_mode", False))
        except TypeError:
            privacy_engine = PrivacyEngine()
        grad_sample_mode = mode
        g, opt_g, source_loader = _make_private_with_mode(
            privacy_engine,
            module=g,
            optimizer=opt_g,
            data_loader=source_loader,
            dp=dp,
            grad_sample_mode=grad_sample_mode,
            require_mode=True,
        )

    use_synth_for_f = dp is not None and dp.enabled
    if use_synth_for_f and synth_sampler is None:
        raise ValueError("synth_sampler required when DP is enabled to avoid using private source for f updates.")

    def _get_hook_controller(module: torch.nn.Module) -> Optional[torch.nn.Module]:
        if hasattr(module, "disable_hooks") and hasattr(module, "enable_hooks"):
            return module
        inner = getattr(module, "module", None)
        if inner is not None and hasattr(inner, "disable_hooks") and hasattr(inner, "enable_hooks"):
            return inner
        return None

    g_inner = unwrap_model(g)
    f_inner = unwrap_model(f)

    def transport_fn(model: torch.nn.Module, x: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
        grad_outputs = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        hook_controller = _get_hook_controller(model)
        hooks_were_enabled = False
        if hook_controller is not None:
            hooks_were_enabled = bool(getattr(hook_controller, "hooks_enabled", False))
        output = model(x)
        # Avoid consuming Opacus activations during autograd.grad; keep them for the main backward.
        if hook_controller is not None and hooks_were_enabled:
            hook_controller.disable_hooks()
        try:
            (output,) = torch.autograd.grad(
                output,
                x,
                create_graph=create_graph,
                only_inputs=True,
                grad_outputs=grad_outputs,
            )
        finally:
            if hook_controller is not None and hooks_were_enabled:
                hook_controller.enable_hooks()
        return output

    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    total_steps = int(n_iters) if n_iters is not None else epochs * len(source_loader)
    log_every = max(1, total_steps // 5)

    last_gl = float("nan")
    last_fl = float("nan")
    for step in range(1, total_steps + 1):
        target = next(target_iter)
        if isinstance(target, (list, tuple)):
            target = target[0]
        target = target.to(device).float()

        for _ in range(n_inner_iters):
            source = next(source_iter)
            if isinstance(source, (list, tuple)):
                source = source[0]
            source = source.to(device).float().requires_grad_(True)

            opt_g.zero_grad(set_to_none=True)
            transport = transport_fn(g, source, create_graph=True)
            gl = compute_loss_g(f, g, source, transport=transport).mean()
            if (
                hasattr(g_inner, "softplus_W_kernels")
                and not g_inner.softplus_W_kernels
                and g_inner.fnorm_penalty > 0
            ):
                gl = gl + g_inner.penalize_w()
            gl.backward()
            opt_g.step()
            last_gl = float(gl.detach().cpu().item())

        if use_synth_for_f:
            source = synth_sampler(target.size(0))
        else:
            source = next(source_iter)
            if isinstance(source, (list, tuple)):
                source = source[0]
        source = source.to(device).float().requires_grad_(True)

        opt_f.zero_grad(set_to_none=True)
        transport = transport_fn(g_inner, source, create_graph=False).detach()
        fl = compute_loss_f(f, g, source, target, transport=transport).mean()
        fl.backward()
        opt_f.step()
        if hasattr(f_inner, "clamp_w"):
            f_inner.clamp_w()
        last_fl = float(fl.detach().cpu().item())

        if step % log_every == 0:
            print(
                f"[Stage II/CellOT] iter {step:04d}/{total_steps}  g_loss={last_gl:.4f}  f_loss={last_fl:.4f}"
            )

    out: Dict[str, float] = {"ot_loss": last_fl, "g_loss": last_gl, "f_loss": last_fl}
    if privacy_engine is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon_ot"] = eps
        out["delta_ot"] = float(dp.delta)
        print(f"[Stage II/CellOT] DP eps={eps:.3f}, delta={dp.delta:g}")
    return out
