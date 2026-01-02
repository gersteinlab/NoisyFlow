from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from noisyflow.attacks.membership_inference import (
    collect_stage_features,
    run_loss_attack,
    run_shadow_attack,
    run_stage_mia_attack,
    run_stage_shadow_attack,
)
from noisyflow.config import ExperimentConfig, PrivacyCurveConfig, load_config
from noisyflow.data import (
    make_cellot_lupuspatients_kang_hvg,
    make_federated_cell_dataset,
    make_federated_mixture_gaussians,
    make_toy_federated_gaussians,
)
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot, train_ot_stage2_rectified_flow
from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import server_synthesize_with_raw, train_classifier, train_random_forest_classifier
from noisyflow.utils import DPConfig, dp_label_prior_from_counts, set_seed, unwrap_model
from noisyflow.metrics import rbf_mmd2_multi_gamma, sliced_w2_distance


data_builders = {
    "toy_federated_gaussians": make_toy_federated_gaussians,
    "federated_mixture_gaussians": make_federated_mixture_gaussians,
    "mixture_gaussians": make_federated_mixture_gaussians,
    "federated_cell_dataset": make_federated_cell_dataset,
    "cellot_lupuspatients_kang_hvg": make_cellot_lupuspatients_kang_hvg,
}


def _build_datasets(cfg: ExperimentConfig):
    if cfg.data.type not in data_builders:
        raise ValueError(f"Unknown data.type '{cfg.data.type}'")
    return data_builders[cfg.data.type](**cfg.data.params)


def _infer_dims(cfg: ExperimentConfig, client_datasets: List[TensorDataset]) -> Tuple[int, int]:
    d = cfg.data.params.get("d")
    if d is None:
        d = int(client_datasets[0].tensors[0].shape[1])
    num_classes = cfg.data.params.get("num_classes")
    if num_classes is None:
        num_classes = int(client_datasets[0].tensors[1].max().item() + 1)
    return int(d), int(num_classes)


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


def _split_dataset(ds: TensorDataset, holdout_fraction: float, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    if holdout_fraction <= 0.0:
        raise ValueError("holdout_fraction must be > 0 when stage_mia is enabled")
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


def _subsample_labeled_dataset(
    ds: TensorDataset,
    n: Optional[int],
    num_classes: int,
    seed: int,
) -> TensorDataset:
    if n is None:
        return ds
    n = int(n)
    if n <= 0:
        raise ValueError("ref_train_size must be > 0")
    if n >= len(ds):
        return ds
    labels = ds.tensors[1].long().cpu().numpy()
    rng = np.random.default_rng(seed)

    per_class = max(1, n // max(1, num_classes))
    indices: List[int] = []
    for c in range(num_classes):
        idx_c = np.flatnonzero(labels == c)
        if idx_c.size == 0:
            continue
        rng.shuffle(idx_c)
        indices.extend(idx_c[: min(per_class, idx_c.size)].tolist())

    if len(indices) < n:
        all_idx = np.arange(labels.shape[0])
        mask = np.ones(labels.shape[0], dtype=bool)
        mask[np.array(indices, dtype=np.int64)] = False
        remaining = all_idx[mask]
        rng.shuffle(remaining)
        indices.extend(remaining[: (n - len(indices))].tolist())

    idx = np.array(indices[:n], dtype=np.int64)
    rng.shuffle(idx)
    tensors = [t[idx] for t in ds.tensors]
    return TensorDataset(*tensors)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = cfg.device

    data_builder = data_builders.get(cfg.data.type)
    if data_builder is None:
        raise ValueError(f"Unknown data.type '{cfg.data.type}'")
    client_datasets, target_ref, target_test = data_builder(**cfg.data.params)
    d, num_classes = _infer_dims(cfg, client_datasets)

    target_loader = DataLoader(
        target_ref,
        batch_size=cfg.loaders.target_batch_size,
        shuffle=True,
        drop_last=cfg.loaders.drop_last,
    )
    target_test_loader = DataLoader(
        target_test,
        batch_size=cfg.loaders.test_batch_size,
        shuffle=False,
    )

    clients_out: List[Dict] = []
    mia_clients: List[Dict] = []
    stage1_eps: List[float] = []
    stage2_eps: List[float] = []
    needs_holdout = cfg.stage_mia.enabled or cfg.stage_shadow_mia.enabled
    for idx, ds in enumerate(client_datasets):
        if needs_holdout:
            train_ds, holdout_ds = _split_dataset(
                ds,
                holdout_fraction=cfg.stage_shadow_mia.holdout_fraction
                if cfg.stage_shadow_mia.enabled
                else cfg.stage_mia.holdout_fraction,
                seed=(cfg.stage_shadow_mia.seed if cfg.stage_shadow_mia.enabled else cfg.stage_mia.seed) + idx,
            )
        else:
            train_ds, holdout_ds = ds, None
        loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        flow = VelocityField(
            d=d,
            num_classes=num_classes,
            hidden=cfg.stage1.hidden,
            time_emb_dim=cfg.stage1.time_emb_dim,
            label_emb_dim=cfg.stage1.label_emb_dim,
        )
        flow_stats = train_flow_stage1(
            flow,
            loader,
            epochs=cfg.stage1.epochs,
            lr=cfg.stage1.lr,
            dp=cfg.stage1.dp,
            device=device,
        )
        if "epsilon_flow" in flow_stats:
            stage1_eps.append(float(flow_stats["epsilon_flow"]))

        prior = None
        if cfg.stage1.label_prior.enabled:
            labels = train_ds.tensors[1]
            prior = dp_label_prior_from_counts(
                labels,
                num_classes=num_classes,
                mechanism=cfg.stage1.label_prior.mechanism,
                sigma=cfg.stage1.label_prior.sigma,
                device="cpu",
            )

        def synth_sampler(
            batch_size: int,
            labels: Optional[torch.Tensor] = None,
            flow=flow,
        ) -> torch.Tensor:
            if labels is None:
                labels = torch.randint(0, num_classes, (batch_size,), device=device)
            else:
                labels = labels.to(device).long().view(-1)
                if int(labels.numel()) != int(batch_size):
                    raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
            return sample_flow_euler(flow.to(device).eval(), labels, n_steps=cfg.stage2.flow_steps).cpu()

        use_cellot = cfg.stage2.cellot.enabled
        use_rectified_flow = cfg.stage2.rectified_flow.enabled
        if use_cellot and use_rectified_flow:
            raise ValueError("Choose only one Stage2 model: stage2.cellot.enabled or stage2.rectified_flow.enabled.")

        real_x_loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        if use_cellot:
            if cfg.stage2.option.upper() != "A":
                raise ValueError("CellOT mode currently supports stage2.option A only.")
            kernel_init = _kernel_init_from_config(cfg.stage2.cellot.kernel_init)
            f = CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.f_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            ot = CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.g_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            ot_stats = train_ot_stage2_cellot(
                f,
                ot,
                source_loader=real_x_loader,
                target_loader=target_loader,
                epochs=cfg.stage2.epochs,
                n_inner_iters=cfg.stage2.cellot.n_inner_iters,
                lr_f=cfg.stage2.lr,
                lr_g=cfg.stage2.lr,
                optim_cfg=cfg.stage2.cellot.optim,
                n_iters=cfg.stage2.cellot.n_iters,
                dp=cfg.stage2.dp,
                synth_sampler=synth_sampler,
                device=device,
            )
        elif use_rectified_flow:
            ot = RectifiedFlowOT(
                d=d,
                hidden=cfg.stage2.rectified_flow.hidden,
                time_emb_dim=cfg.stage2.rectified_flow.time_emb_dim,
                act=cfg.stage2.rectified_flow.act,
                transport_steps=cfg.stage2.rectified_flow.transport_steps,
            )
            ot_stats = train_ot_stage2_rectified_flow(
                ot,
                source_loader=real_x_loader if cfg.stage2.option.upper() in {"A", "C"} else None,
                target_loader=target_loader,
                option=cfg.stage2.option,
                pair_by_label=cfg.stage2.pair_by_label,
                pair_by_ot=cfg.stage2.pair_by_ot,
                synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                dp=cfg.stage2.dp,
                device=device,
            )
        else:
            ot = ICNN(
                d=d,
                hidden=cfg.stage2.hidden,
                act=cfg.stage2.act,
                add_strong_convexity=cfg.stage2.add_strong_convexity,
            )
            ot_stats = train_ot_stage2(
                ot,
                real_loader=real_x_loader if cfg.stage2.option.upper() in {"A", "C"} else None,
                target_loader=target_loader,
                option=cfg.stage2.option,
                synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                dp=cfg.stage2.dp,
                conj_steps=cfg.stage2.conj_steps,
                conj_lr=cfg.stage2.conj_lr,
                conj_clamp=cfg.stage2.conj_clamp,
                device=device,
            )
        if "epsilon_ot" in ot_stats:
            stage2_eps.append(float(ot_stats["epsilon_ot"]))

        flow_cpu = unwrap_model(flow).cpu()
        ot_cpu = unwrap_model(ot).cpu()
        clients_out.append({"flow": flow_cpu, "ot": ot_cpu, "prior": prior})
        if needs_holdout and holdout_ds is not None:
            mia_clients.append(
                {
                    "flow": flow_cpu,
                    "ot": ot_cpu,
                    "members": train_ds,
                    "nonmembers": holdout_ds,
                }
            )

    stats_sw2: Dict[str, float] = {}
    if isinstance(target_ref, TensorDataset) and len(target_ref.tensors) >= 1:
        try:
            x_ref = target_ref.tensors[0]
            x_private = torch.cat([ds.tensors[0] for ds in client_datasets], dim=0)
            sw2_private_ref = sliced_w2_distance(x_private, x_ref, num_projections=128, max_samples=2000, seed=cfg.seed)
            stats_sw2["sw2_private_ref"] = float(sw2_private_ref)
        except Exception as exc:
            print(f"[Metrics/SW2] WARNING: failed to compute sw2_private_ref ({exc})")
            stats_sw2 = {}

    y_syn, l_syn, x_syn_raw = server_synthesize_with_raw(
        clients_out,
        M_per_client=cfg.stage3.M_per_client,
        num_classes=num_classes,
        flow_steps=cfg.stage3.flow_steps,
        device=device,
    )
    if isinstance(target_ref, TensorDataset) and len(target_ref.tensors) >= 1:
        try:
            x_ref = target_ref.tensors[0]
            stats_sw2["sw2_synth_ref"] = float(
                sliced_w2_distance(x_syn_raw, x_ref, num_projections=128, max_samples=2000, seed=cfg.seed)
            )
            stats_sw2["sw2_synth_transported_ref"] = float(
                sliced_w2_distance(y_syn, x_ref, num_projections=128, max_samples=2000, seed=cfg.seed)
            )
            print(
                "[Metrics/SW2] private~ref={:.4f}  synth~ref={:.4f}  transported~ref={:.4f}".format(
                    stats_sw2.get("sw2_private_ref", float("nan")),
                    stats_sw2.get("sw2_synth_ref", float("nan")),
                    stats_sw2.get("sw2_synth_transported_ref", float("nan")),
                )
            )
        except Exception as exc:
            print(f"[Metrics/SW2] WARNING: failed to compute SW2 metrics ({exc})")
            stats_sw2 = {}
    stats_mmd: Dict[str, float] = {}
    if cfg.data.type in {"federated_cell_dataset", "cellot_lupuspatients_kang_hvg"}:
        try:
            gammas = [2.0, 1.0, 0.5, 0.1, 0.01, 0.005]
            mmds = rbf_mmd2_multi_gamma(
                y_syn,
                target_test.tensors[0],
                gammas=gammas,
                max_samples=2000,
                seed=cfg.seed,
            )
            stats_mmd = {
                "mmd_rbf_min": float(min(mmds)),
                "mmd_rbf_mean": float(sum(mmds) / float(len(mmds))),
            }
            print(f"[Metrics/MMD] min={stats_mmd['mmd_rbf_min']:.6f} mean={stats_mmd['mmd_rbf_mean']:.6f}")
        except Exception as exc:
            print(f"[Metrics/MMD] WARNING: failed to compute MMD ({exc})")
            stats_mmd = {}
    syn_loader = DataLoader(
        TensorDataset(y_syn, l_syn),
        batch_size=cfg.loaders.synth_batch_size,
        shuffle=True,
        drop_last=cfg.loaders.drop_last,
    )
    syn_eval_loader = DataLoader(
        TensorDataset(y_syn, l_syn),
        batch_size=cfg.loaders.synth_batch_size,
        shuffle=False,
        drop_last=False,
    )

    clf = None
    try:
        stats = train_random_forest_classifier(
            syn_loader,
            test_loader=target_test_loader,
            seed=cfg.seed,
            name="Classifier/RF-synth",
        )
    except RuntimeError as exc:
        print(f"[Classifier/RF] WARNING: {exc} Falling back to MLP classifier.")
        clf = Classifier(d=d, num_classes=num_classes, hidden=cfg.stage3.hidden)
        stats = train_classifier(
            clf,
            syn_loader,
            test_loader=target_test_loader,
            epochs=cfg.stage3.epochs,
            lr=cfg.stage3.lr,
            device=device,
        )
    out: Dict[str, float] = dict(stats)
    out.update(stats_mmd)
    out.update(stats_sw2)

    if (cfg.membership_inference.enabled or cfg.shadow_mia.enabled) and clf is None:
        clf = Classifier(d=d, num_classes=num_classes, hidden=cfg.stage3.hidden)
        train_classifier(
            clf,
            syn_loader,
            test_loader=None,
            epochs=cfg.stage3.epochs,
            lr=cfg.stage3.lr,
            device=device,
        )
    if cfg.stage_mia.enabled:
        use_ot = cfg.stage2.option.upper() in {"A", "C"}
        member_feats: List[torch.Tensor] = []
        nonmember_feats: List[torch.Tensor] = []
        for entry in mia_clients:
            flow = entry["flow"].to(device)
            ot = entry["ot"].to(device) if use_ot else None
            member_loader = DataLoader(
                entry["members"],
                batch_size=cfg.loaders.batch_size,
                shuffle=False,
                drop_last=False,
            )
            nonmember_loader = DataLoader(
                entry["nonmembers"],
                batch_size=cfg.loaders.batch_size,
                shuffle=False,
                drop_last=False,
            )
            member_feats.append(
                collect_stage_features(
                    flow,
                    ot,
                    member_loader,
                    use_ot=use_ot,
                    num_flow_samples=cfg.stage_mia.num_flow_samples,
                    include_ot_transport_norm=cfg.stage_mia.include_ot_transport_norm,
                    seed=cfg.stage_mia.seed,
                    device=device,
                )
            )
            nonmember_feats.append(
                collect_stage_features(
                    flow,
                    ot,
                    nonmember_loader,
                    use_ot=use_ot,
                    num_flow_samples=cfg.stage_mia.num_flow_samples,
                    include_ot_transport_norm=cfg.stage_mia.include_ot_transport_norm,
                    seed=cfg.stage_mia.seed,
                    device=device,
                )
            )

        all_member = torch.cat(member_feats, dim=0) if member_feats else torch.empty(0)
        all_nonmember = torch.cat(nonmember_feats, dim=0) if nonmember_feats else torch.empty(0)
        stage_mia_stats = run_stage_mia_attack(
            all_member,
            all_nonmember,
            attack_hidden=cfg.stage_mia.attack_hidden,
            attack_epochs=cfg.stage_mia.attack_epochs,
            attack_lr=cfg.stage_mia.attack_lr,
            attack_batch_size=cfg.stage_mia.attack_batch_size,
            attack_train_frac=cfg.stage_mia.attack_train_frac,
            max_samples=cfg.stage_mia.max_samples,
            seed=cfg.stage_mia.seed,
            device=device,
        )
        out.update(stage_mia_stats)
    if cfg.stage_shadow_mia.enabled:
        use_ot = cfg.stage2.option.upper() in {"A", "C"}
        flow_kwargs = {
            "d": d,
            "num_classes": num_classes,
            "hidden": cfg.stage1.hidden,
            "time_emb_dim": cfg.stage1.time_emb_dim,
            "label_emb_dim": cfg.stage1.label_emb_dim,
        }
        ot_kwargs = {
            "d": d,
            "hidden": cfg.stage2.hidden,
            "act": cfg.stage2.act,
            "add_strong_convexity": cfg.stage2.add_strong_convexity,
        }
        stage_shadow_stats = run_stage_shadow_attack(
            data_builder=data_builder,
            data_params=cfg.data.params,
            target_clients=mia_clients,
            flow_kwargs=flow_kwargs,
            ot_kwargs=ot_kwargs,
            stage2_option=cfg.stage2.option,
            stage1_train_kwargs={"epochs": cfg.stage1.epochs, "lr": cfg.stage1.lr},
            stage2_train_kwargs={
                "epochs": cfg.stage2.epochs,
                "lr": cfg.stage2.lr,
                "conj_steps": cfg.stage2.conj_steps,
                "conj_lr": cfg.stage2.conj_lr,
                "conj_clamp": cfg.stage2.conj_clamp,
                "flow_steps": cfg.stage2.flow_steps,
                "n_inner_iters": cfg.stage2.cellot.n_inner_iters,
            },
            batch_size=cfg.loaders.batch_size,
            target_batch_size=cfg.loaders.target_batch_size,
            drop_last=cfg.loaders.drop_last,
            num_shadow_models=cfg.stage_shadow_mia.num_shadow_models,
            holdout_fraction=cfg.stage_shadow_mia.holdout_fraction,
            num_flow_samples=cfg.stage_shadow_mia.num_flow_samples,
            include_ot_transport_norm=cfg.stage_shadow_mia.include_ot_transport_norm,
            attack_hidden=cfg.stage_shadow_mia.attack_hidden,
            attack_epochs=cfg.stage_shadow_mia.attack_epochs,
            attack_lr=cfg.stage_shadow_mia.attack_lr,
            attack_batch_size=cfg.stage_shadow_mia.attack_batch_size,
            attack_train_frac=cfg.stage_shadow_mia.attack_train_frac,
            max_samples_per_shadow=cfg.stage_shadow_mia.max_samples_per_shadow,
            seed=cfg.stage_shadow_mia.seed,
            data_overrides=cfg.stage_shadow_mia.data_overrides,
            cellot_enabled=cfg.stage2.cellot.enabled,
            cellot_hidden_units=cfg.stage2.cellot.hidden_units,
            cellot_activation=cfg.stage2.cellot.activation,
            cellot_softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
            cellot_softplus_beta=cfg.stage2.cellot.softplus_beta,
            cellot_kernel_init=cfg.stage2.cellot.kernel_init,
            cellot_f_fnorm_penalty=cfg.stage2.cellot.f_fnorm_penalty,
            cellot_g_fnorm_penalty=cfg.stage2.cellot.g_fnorm_penalty,
            cellot_n_inner_iters=cfg.stage2.cellot.n_inner_iters,
            cellot_optim=cfg.stage2.cellot.optim,
            cellot_n_iters=cfg.stage2.cellot.n_iters,
            rectified_flow_enabled=cfg.stage2.rectified_flow.enabled,
            rectified_flow_hidden=cfg.stage2.rectified_flow.hidden,
            rectified_flow_time_emb_dim=cfg.stage2.rectified_flow.time_emb_dim,
            rectified_flow_act=cfg.stage2.rectified_flow.act,
            rectified_flow_transport_steps=cfg.stage2.rectified_flow.transport_steps,
            device=device,
        )
        out.update(stage_shadow_stats)
    if cfg.membership_inference.enabled:
        if clf is None:
            raise RuntimeError("Internal error: classifier not initialized for membership inference.")
        mia_stats = run_loss_attack(
            clf,
            syn_eval_loader,
            target_test_loader,
            device=device,
            max_samples=cfg.membership_inference.max_samples,
            seed=cfg.membership_inference.seed,
        )
        out.update(mia_stats)
    if cfg.shadow_mia.enabled:
        if clf is None:
            raise RuntimeError("Internal error: classifier not initialized for shadow MIA.")
        shadow_stats = run_shadow_attack(
            data_builder=data_builder,
            data_params=cfg.data.params,
            d=d,
            num_classes=num_classes,
            target_model=clf,
            target_member_loader=syn_eval_loader,
            target_nonmember_loader=target_test_loader,
            num_shadow_models=cfg.shadow_mia.num_shadow_models,
            shadow_train_size=cfg.shadow_mia.shadow_train_size,
            shadow_test_size=cfg.shadow_mia.shadow_test_size,
            shadow_epochs=cfg.shadow_mia.shadow_epochs,
            shadow_lr=cfg.shadow_mia.shadow_lr,
            shadow_hidden=cfg.shadow_mia.shadow_hidden,
            shadow_batch_size=cfg.shadow_mia.shadow_batch_size,
            attack_epochs=cfg.shadow_mia.attack_epochs,
            attack_lr=cfg.shadow_mia.attack_lr,
            attack_hidden=cfg.shadow_mia.attack_hidden,
            attack_batch_size=cfg.shadow_mia.attack_batch_size,
            feature_set=cfg.shadow_mia.feature_set,
            max_samples_per_shadow=cfg.shadow_mia.max_samples_per_shadow,
            seed=cfg.shadow_mia.seed,
            data_overrides=cfg.shadow_mia.data_overrides,
            device=device,
        )
        out.update(shadow_stats)
    if stage1_eps:
        out["epsilon_flow_max"] = float(max(stage1_eps))
    if stage2_eps:
        out["epsilon_ot_max"] = float(max(stage2_eps))
    if stage1_eps or stage2_eps:
        out["epsilon_total_max"] = float(max(stage1_eps or [0.0]) + max(stage2_eps or [0.0]))

    nan = float("nan")
    out.setdefault("clf_loss_ref_only", nan)
    out.setdefault("acc_ref_only", nan)
    out.setdefault("clf_loss_ref_plus_synth", nan)
    out.setdefault("acc_ref_plus_synth", nan)
    if isinstance(target_ref, TensorDataset) and len(target_ref.tensors) >= 2:
        ref_supervised_ds = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
        ref_supervised_ds = _subsample_labeled_dataset(
            ref_supervised_ds,
            n=cfg.stage3.ref_train_size,
            num_classes=num_classes,
            seed=cfg.seed,
        )
        ref_train_loader = DataLoader(
            ref_supervised_ds,
            batch_size=cfg.loaders.target_batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )
        try:
            ref_stats = train_random_forest_classifier(
                ref_train_loader,
                test_loader=target_test_loader,
                seed=cfg.seed,
                name="Classifier/RF-ref_only",
            )
        except RuntimeError as exc:
            print(f"[Classifier/RF] WARNING: {exc} Falling back to MLP classifier.")
            ref_clf = Classifier(d=d, num_classes=num_classes, hidden=cfg.stage3.hidden)
            ref_stats = train_classifier(
                ref_clf,
                ref_train_loader,
                test_loader=target_test_loader,
                epochs=cfg.stage3.epochs,
                lr=cfg.stage3.lr,
                device=device,
            )
        out["clf_loss_ref_only"] = float(ref_stats.get("clf_loss", nan))
        out["acc_ref_only"] = float(ref_stats.get("acc", nan))

        syn_supervised_ds = TensorDataset(y_syn, l_syn)
        syn_supervised_ds = _subsample_labeled_dataset(
            syn_supervised_ds,
            n=cfg.stage3.combined_synth_train_size,
            num_classes=num_classes,
            seed=cfg.seed,
        )
        combined_ds = ConcatDataset([ref_supervised_ds, syn_supervised_ds])
        combined_loader = DataLoader(
            combined_ds,
            batch_size=cfg.loaders.synth_batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )
        try:
            combined_stats = train_random_forest_classifier(
                combined_loader,
                test_loader=target_test_loader,
                seed=cfg.seed,
                name="Classifier/RF-ref+syn",
            )
        except RuntimeError as exc:
            print(f"[Classifier/RF] WARNING: {exc} Falling back to MLP classifier.")
            combined_clf = Classifier(d=d, num_classes=num_classes, hidden=cfg.stage3.hidden)
            combined_stats = train_classifier(
                combined_clf,
                combined_loader,
                test_loader=target_test_loader,
                epochs=cfg.stage3.epochs,
                lr=cfg.stage3.lr,
                device=device,
            )
        out["clf_loss_ref_plus_synth"] = float(combined_stats.get("clf_loss", nan))
        out["acc_ref_plus_synth"] = float(combined_stats.get("acc", nan))
    return out


def _set_dp_config(dp_cfg: Optional[DPConfig], noise_multiplier: float) -> DPConfig:
    if dp_cfg is None:
        dp_cfg = DPConfig()
    dp_cfg.enabled = True
    dp_cfg.noise_multiplier = float(noise_multiplier)
    return dp_cfg


def _select_epsilon(stats: Dict[str, float], stage: str) -> Optional[float]:
    if stage == "stage1":
        return stats.get("epsilon_flow_max")
    if stage == "stage2":
        return stats.get("epsilon_ot_max")
    if stage == "both":
        return stats.get("epsilon_total_max")
    raise ValueError(f"Unknown privacy curve stage '{stage}'")


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "acc":
        return "accuracy (synth)"
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    if metric == "acc_ref_plus_synth":
        return "accuracy (ref+synth)"
    return metric


def _plot_privacy_curve(results: List[Dict[str, Optional[float]]], output_path: str, metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install matplotlib.") from e

    points = [
        (r["epsilon"], r["utility"])
        for r in results
        if r.get("epsilon") is not None and r.get("utility") is not None
    ]
    if not points:
        raise RuntimeError("No valid (epsilon, utility) points available for plotting.")

    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("epsilon (approx)")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved privacy-utility curve to {output_path}")


def run_privacy_curve(cfg: ExperimentConfig, curve_cfg: PrivacyCurveConfig) -> List[Dict[str, Optional[float]]]:
    stage = curve_cfg.stage.strip().lower()
    if stage not in {"stage1", "stage2", "both"}:
        raise ValueError("privacy_curve.stage must be one of 'stage1', 'stage2', 'both'")

    if stage in {"stage2", "both"} and cfg.stage2.option.upper() not in {"A", "C"}:
        raise ValueError("privacy_curve.stage includes stage2 but stage2.option is not A or C")

    metric = curve_cfg.metric.strip()
    if not metric:
        raise ValueError("privacy_curve.metric must be a non-empty stats key (e.g., 'acc_ref_plus_synth').")

    results: List[Dict[str, Optional[float]]] = []
    for nm in curve_cfg.noise_multipliers:
        sweep_cfg = copy.deepcopy(cfg)

        if stage in {"stage1", "both"}:
            sweep_cfg.stage1.dp = _set_dp_config(sweep_cfg.stage1.dp, nm)
        if stage in {"stage2", "both"}:
            sweep_cfg.stage2.dp = _set_dp_config(sweep_cfg.stage2.dp, nm)

        stats = run_experiment(sweep_cfg)
        results.append(
            {
                "noise_multiplier": float(nm),
                "epsilon": _select_epsilon(stats, stage),
                "utility": stats.get(metric),
            }
        )

    out_path = Path(curve_cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_privacy_curve(results, str(out_path), metric=metric)
    try:
        json_path = out_path.with_suffix(".json")
        payload = {"stage": stage, "metric": metric, "results": results}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved privacy-utility sweep results to {json_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[PrivacyCurve] WARNING: failed to write JSON results ({exc})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NoisyFlow experiments from a YAML config.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.privacy_curve.enabled:
        results = run_privacy_curve(cfg, cfg.privacy_curve)
        print("Privacy-utility sweep:", results)
    else:
        stats = run_experiment(cfg)
        print("Final stats:", stats)


if __name__ == "__main__":
    main()
