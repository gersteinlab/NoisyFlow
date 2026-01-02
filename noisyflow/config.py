from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from noisyflow.utils import DPConfig


@dataclass
class DataConfig:
    type: str = "federated_mixture_gaussians"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoaderConfig:
    batch_size: int = 256
    target_batch_size: int = 256
    test_batch_size: int = 512
    synth_batch_size: int = 512
    drop_last: bool = True


@dataclass
class LabelPriorConfig:
    enabled: bool = True
    mechanism: str = "gaussian"
    sigma: float = 1.0


@dataclass
class Stage1Config:
    epochs: int = 20
    lr: float = 1e-3
    hidden: List[int] = field(default_factory=lambda: [128, 128])
    time_emb_dim: int = 32
    label_emb_dim: int = 32
    label_prior: LabelPriorConfig = field(default_factory=LabelPriorConfig)
    dp: Optional[DPConfig] = None


@dataclass
class CellOTConfig:
    enabled: bool = False
    hidden_units: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    activation: str = "LeakyReLU"
    softplus_W_kernels: bool = False
    softplus_beta: float = 1.0
    kernel_init: Dict[str, Any] = field(default_factory=lambda: {"name": "uniform", "b": 0.1})
    optim: Dict[str, Any] = field(
        default_factory=lambda: {
            "optimizer": "Adam",
            "lr": 1e-4,
            "beta1": 0.5,
            "beta2": 0.9,
            "weight_decay": 0.0,
        }
    )
    f_fnorm_penalty: float = 0.0
    g_fnorm_penalty: float = 0.0
    n_inner_iters: int = 10
    n_iters: Optional[int] = None


@dataclass
class RectifiedFlowOTConfig:
    enabled: bool = False
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    time_emb_dim: int = 64
    act: str = "silu"
    transport_steps: int = 50


@dataclass
class Stage2Config:
    option: str = "B"
    pair_by_label: bool = False
    pair_by_ot: bool = False
    epochs: int = 30
    lr: float = 1e-3
    hidden: List[int] = field(default_factory=lambda: [128, 128])
    act: str = "relu"
    add_strong_convexity: float = 0.1
    flow_steps: int = 50
    conj_steps: int = 20
    conj_lr: float = 0.2
    conj_clamp: Optional[float] = 10.0
    dp: Optional[DPConfig] = None
    cellot: CellOTConfig = field(default_factory=lambda: CellOTConfig())
    rectified_flow: RectifiedFlowOTConfig = field(default_factory=lambda: RectifiedFlowOTConfig())


@dataclass
class Stage3Config:
    epochs: int = 30
    lr: float = 1e-3
    hidden: List[int] = field(default_factory=lambda: [128, 128])
    flow_steps: int = 50
    M_per_client: int = 5000
    ref_train_size: Optional[int] = None
    combined_synth_train_size: Optional[int] = None


@dataclass
class PrivacyCurveConfig:
    enabled: bool = False
    stage: str = "stage1"
    noise_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    output_path: str = "privacy_utility.png"
    metric: str = "acc"


@dataclass
class MembershipInferenceConfig:
    enabled: bool = False
    max_samples: Optional[int] = 2000
    seed: int = 0


@dataclass
class ShadowMIAConfig:
    enabled: bool = False
    num_shadow_models: int = 2
    shadow_train_size: int = 2000
    shadow_test_size: int = 2000
    shadow_epochs: int = 5
    shadow_lr: float = 1e-3
    shadow_hidden: List[int] = field(default_factory=lambda: [128, 128])
    shadow_batch_size: int = 256
    attack_epochs: int = 20
    attack_lr: float = 1e-3
    attack_hidden: List[int] = field(default_factory=lambda: [64, 32])
    attack_batch_size: int = 256
    feature_set: str = "stats"
    max_samples_per_shadow: Optional[int] = 2000
    seed: int = 0
    data_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMIAConfig:
    enabled: bool = False
    holdout_fraction: float = 0.2
    num_flow_samples: int = 1
    include_ot_transport_norm: bool = True
    attack_train_frac: float = 0.5
    attack_hidden: List[int] = field(default_factory=lambda: [64, 32])
    attack_epochs: int = 20
    attack_lr: float = 1e-3
    attack_batch_size: int = 256
    max_samples: Optional[int] = 2000
    seed: int = 0


@dataclass
class StageShadowMIAConfig:
    enabled: bool = False
    num_shadow_models: int = 2
    holdout_fraction: float = 0.2
    num_flow_samples: int = 1
    include_ot_transport_norm: bool = True
    attack_train_frac: float = 0.5
    attack_hidden: List[int] = field(default_factory=lambda: [64, 32])
    attack_epochs: int = 20
    attack_lr: float = 1e-3
    attack_batch_size: int = 256
    max_samples_per_shadow: Optional[int] = 2000
    seed: int = 0
    data_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    loaders: LoaderConfig = field(default_factory=LoaderConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    privacy_curve: PrivacyCurveConfig = field(default_factory=PrivacyCurveConfig)
    membership_inference: MembershipInferenceConfig = field(default_factory=MembershipInferenceConfig)
    shadow_mia: ShadowMIAConfig = field(default_factory=ShadowMIAConfig)
    stage_mia: StageMIAConfig = field(default_factory=StageMIAConfig)
    stage_shadow_mia: StageShadowMIAConfig = field(default_factory=StageShadowMIAConfig)


def _dp_from_dict(data: Optional[Dict[str, Any]]) -> Optional[DPConfig]:
    if not data:
        return None
    return DPConfig(
        enabled=bool(data.get("enabled", True)),
        max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        noise_multiplier=float(data.get("noise_multiplier", 1.0)),
        delta=float(data.get("delta", 1e-5)),
        grad_sample_mode=data.get("grad_sample_mode", None),
        secure_mode=bool(data.get("secure_mode", False)),
    )


def _label_prior_from_dict(data: Optional[Dict[str, Any]]) -> LabelPriorConfig:
    if not data:
        return LabelPriorConfig()
    return LabelPriorConfig(
        enabled=bool(data.get("enabled", True)),
        mechanism=str(data.get("mechanism", "gaussian")),
        sigma=float(data.get("sigma", 1.0)),
    )


def load_config(path: str) -> ExperimentConfig:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML is required for YAML config loading. Install pyyaml.") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data_raw = data.get("data", {}) or {}
    data_params = dict(data_raw.get("params", {}) or {})
    for key, value in data_raw.items():
        if key not in {"type", "params"}:
            data_params[key] = value
    data_cfg = DataConfig(type=str(data_raw.get("type", "federated_mixture_gaussians")), params=data_params)

    loaders_cfg = LoaderConfig(**(data.get("loaders", {}) or {}))

    stage1_raw = data.get("stage1", {}) or {}
    stage1_cfg = Stage1Config(
        epochs=int(stage1_raw.get("epochs", 20)),
        lr=float(stage1_raw.get("lr", 1e-3)),
        hidden=list(stage1_raw.get("hidden", [128, 128])),
        time_emb_dim=int(stage1_raw.get("time_emb_dim", 32)),
        label_emb_dim=int(stage1_raw.get("label_emb_dim", 32)),
        label_prior=_label_prior_from_dict(stage1_raw.get("label_prior", {}) or {}),
        dp=_dp_from_dict(stage1_raw.get("dp", {}) or {}),
    )

    stage2_raw = data.get("stage2", {}) or {}
    conj_clamp = stage2_raw.get("conj_clamp", 10.0)
    if conj_clamp is not None:
        conj_clamp = float(conj_clamp)
    cellot_raw = stage2_raw.get("cellot", {}) or {}
    cellot_optim = dict(cellot_raw.get("optim", {}) or {})
    if not cellot_optim:
        cellot_optim = {
            "optimizer": "Adam",
            "lr": 1e-4,
            "beta1": 0.5,
            "beta2": 0.9,
            "weight_decay": 0.0,
        }
    cellot_n_iters = cellot_raw.get("n_iters", None)
    if cellot_n_iters is not None:
        cellot_n_iters = int(cellot_n_iters)
    cellot_cfg = CellOTConfig(
        enabled=bool(cellot_raw.get("enabled", False)),
        hidden_units=list(cellot_raw.get("hidden_units", [64, 64, 64, 64])),
        activation=str(cellot_raw.get("activation", "LeakyReLU")),
        softplus_W_kernels=bool(cellot_raw.get("softplus_W_kernels", False)),
        softplus_beta=float(cellot_raw.get("softplus_beta", 1.0)),
        kernel_init=dict(cellot_raw.get("kernel_init", {"name": "uniform", "b": 0.1}) or {}),
        optim=cellot_optim,
        f_fnorm_penalty=float(cellot_raw.get("f_fnorm_penalty", 0.0)),
        g_fnorm_penalty=float(cellot_raw.get("g_fnorm_penalty", 0.0)),
        n_inner_iters=int(cellot_raw.get("n_inner_iters", 10)),
        n_iters=cellot_n_iters,
    )
    rf_raw = stage2_raw.get("rectified_flow", {}) or {}
    rf_cfg = RectifiedFlowOTConfig(
        enabled=bool(rf_raw.get("enabled", False)),
        hidden=list(rf_raw.get("hidden", [256, 256])),
        time_emb_dim=int(rf_raw.get("time_emb_dim", 64)),
        act=str(rf_raw.get("act", "silu")),
        transport_steps=int(rf_raw.get("transport_steps", 50)),
    )
    stage2_cfg = Stage2Config(
        option=str(stage2_raw.get("option", "B")),
        pair_by_label=bool(stage2_raw.get("pair_by_label", False)),
        pair_by_ot=bool(stage2_raw.get("pair_by_ot", False)),
        epochs=int(stage2_raw.get("epochs", 30)),
        lr=float(stage2_raw.get("lr", 1e-3)),
        hidden=list(stage2_raw.get("hidden", [128, 128])),
        act=str(stage2_raw.get("act", "relu")),
        add_strong_convexity=float(stage2_raw.get("add_strong_convexity", 0.1)),
        flow_steps=int(stage2_raw.get("flow_steps", 50)),
        conj_steps=int(stage2_raw.get("conj_steps", 20)),
        conj_lr=float(stage2_raw.get("conj_lr", 0.2)),
        conj_clamp=conj_clamp,
        dp=_dp_from_dict(stage2_raw.get("dp", {}) or {}),
        cellot=cellot_cfg,
        rectified_flow=rf_cfg,
    )

    stage3_raw = data.get("stage3", {}) or {}
    ref_train_size = stage3_raw.get("ref_train_size", None)
    if ref_train_size is not None:
        ref_train_size = int(ref_train_size)
    combined_synth_train_size = stage3_raw.get("combined_synth_train_size", None)
    if combined_synth_train_size is not None:
        combined_synth_train_size = int(combined_synth_train_size)
    stage3_cfg = Stage3Config(
        epochs=int(stage3_raw.get("epochs", 30)),
        lr=float(stage3_raw.get("lr", 1e-3)),
        hidden=list(stage3_raw.get("hidden", [128, 128])),
        flow_steps=int(stage3_raw.get("flow_steps", 50)),
        M_per_client=int(stage3_raw.get("M_per_client", 5000)),
        ref_train_size=ref_train_size,
        combined_synth_train_size=combined_synth_train_size,
    )

    privacy_raw = data.get("privacy_curve", {}) or {}
    privacy_cfg = PrivacyCurveConfig(
        enabled=bool(privacy_raw.get("enabled", False)),
        stage=str(privacy_raw.get("stage", "stage1")),
        noise_multipliers=list(privacy_raw.get("noise_multipliers", [0.5, 1.0, 2.0, 4.0])),
        output_path=str(privacy_raw.get("output_path", "privacy_utility.png")),
        metric=str(privacy_raw.get("metric", "acc")),
    )

    mia_raw = data.get("membership_inference", {}) or {}
    max_samples = mia_raw.get("max_samples", 2000)
    if max_samples is not None:
        max_samples = int(max_samples)
    mia_cfg = MembershipInferenceConfig(
        enabled=bool(mia_raw.get("enabled", False)),
        max_samples=max_samples,
        seed=int(mia_raw.get("seed", 0)),
    )

    shadow_raw = data.get("shadow_mia", {}) or {}
    shadow_max_samples = shadow_raw.get("max_samples_per_shadow", 2000)
    if shadow_max_samples is not None:
        shadow_max_samples = int(shadow_max_samples)
    shadow_cfg = ShadowMIAConfig(
        enabled=bool(shadow_raw.get("enabled", False)),
        num_shadow_models=int(shadow_raw.get("num_shadow_models", 2)),
        shadow_train_size=int(shadow_raw.get("shadow_train_size", 2000)),
        shadow_test_size=int(shadow_raw.get("shadow_test_size", 2000)),
        shadow_epochs=int(shadow_raw.get("shadow_epochs", 5)),
        shadow_lr=float(shadow_raw.get("shadow_lr", 1e-3)),
        shadow_hidden=list(shadow_raw.get("shadow_hidden", [128, 128])),
        shadow_batch_size=int(shadow_raw.get("shadow_batch_size", 256)),
        attack_epochs=int(shadow_raw.get("attack_epochs", 20)),
        attack_lr=float(shadow_raw.get("attack_lr", 1e-3)),
        attack_hidden=list(shadow_raw.get("attack_hidden", [64, 32])),
        attack_batch_size=int(shadow_raw.get("attack_batch_size", 256)),
        feature_set=str(shadow_raw.get("feature_set", "stats")),
        max_samples_per_shadow=shadow_max_samples,
        seed=int(shadow_raw.get("seed", 0)),
        data_overrides=dict(shadow_raw.get("data_overrides", {}) or {}),
    )

    stage_mia_raw = data.get("stage_mia", {}) or {}
    stage_mia_max_samples = stage_mia_raw.get("max_samples", 2000)
    if stage_mia_max_samples is not None:
        stage_mia_max_samples = int(stage_mia_max_samples)
    stage_mia_cfg = StageMIAConfig(
        enabled=bool(stage_mia_raw.get("enabled", False)),
        holdout_fraction=float(stage_mia_raw.get("holdout_fraction", 0.2)),
        num_flow_samples=int(stage_mia_raw.get("num_flow_samples", 1)),
        include_ot_transport_norm=bool(stage_mia_raw.get("include_ot_transport_norm", True)),
        attack_train_frac=float(stage_mia_raw.get("attack_train_frac", 0.5)),
        attack_hidden=list(stage_mia_raw.get("attack_hidden", [64, 32])),
        attack_epochs=int(stage_mia_raw.get("attack_epochs", 20)),
        attack_lr=float(stage_mia_raw.get("attack_lr", 1e-3)),
        attack_batch_size=int(stage_mia_raw.get("attack_batch_size", 256)),
        max_samples=stage_mia_max_samples,
        seed=int(stage_mia_raw.get("seed", 0)),
    )

    stage_shadow_raw = data.get("stage_shadow_mia", {}) or {}
    stage_shadow_max_samples = stage_shadow_raw.get("max_samples_per_shadow", 2000)
    if stage_shadow_max_samples is not None:
        stage_shadow_max_samples = int(stage_shadow_max_samples)
    stage_shadow_cfg = StageShadowMIAConfig(
        enabled=bool(stage_shadow_raw.get("enabled", False)),
        num_shadow_models=int(stage_shadow_raw.get("num_shadow_models", 2)),
        holdout_fraction=float(stage_shadow_raw.get("holdout_fraction", 0.2)),
        num_flow_samples=int(stage_shadow_raw.get("num_flow_samples", 1)),
        include_ot_transport_norm=bool(stage_shadow_raw.get("include_ot_transport_norm", True)),
        attack_train_frac=float(stage_shadow_raw.get("attack_train_frac", 0.5)),
        attack_hidden=list(stage_shadow_raw.get("attack_hidden", [64, 32])),
        attack_epochs=int(stage_shadow_raw.get("attack_epochs", 20)),
        attack_lr=float(stage_shadow_raw.get("attack_lr", 1e-3)),
        attack_batch_size=int(stage_shadow_raw.get("attack_batch_size", 256)),
        max_samples_per_shadow=stage_shadow_max_samples,
        seed=int(stage_shadow_raw.get("seed", 0)),
        data_overrides=dict(stage_shadow_raw.get("data_overrides", {}) or {}),
    )

    return ExperimentConfig(
        seed=int(data.get("seed", 0)),
        device=str(data.get("device", "cpu")),
        data=data_cfg,
        loaders=loaders_cfg,
        stage1=stage1_cfg,
        stage2=stage2_cfg,
        stage3=stage3_cfg,
        privacy_curve=privacy_cfg,
        membership_inference=mia_cfg,
        shadow_mia=shadow_cfg,
        stage_mia=stage_mia_cfg,
        stage_shadow_mia=stage_shadow_cfg,
    )
