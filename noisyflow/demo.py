from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.data.toy import make_toy_federated_gaussians
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
from noisyflow.stage2.networks import CellOTICNN, ICNN
from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot
from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import server_synthesize, train_classifier
from noisyflow.utils import DPConfig, dp_label_prior_from_counts, unwrap_model


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
    """
    K = 3
    num_classes = 3
    d = 2

    client_datasets, target_ref, target_test = make_toy_federated_gaussians(
        K=K,
        n_per_client=1500,
        n_target_ref=2000,
        n_target_test=1000,
        d=d,
        num_classes=num_classes,
        seed=0,
    )

    target_loader = DataLoader(target_ref, batch_size=256, shuffle=True, drop_last=True)
    target_test_loader = DataLoader(target_test, batch_size=512, shuffle=False)

    clients_out: List[Dict] = []
    for i in range(K):
        ds = client_datasets[i]
        loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

        flow = VelocityField(
            d=d,
            num_classes=num_classes,
            hidden=[128, 128],
            time_emb_dim=32,
            label_emb_dim=32,
        )
        dp1 = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5) if dp_stage1 else None
        train_flow_stage1(flow, loader, epochs=20, lr=1e-3, dp=dp1, device=device)

        all_labels = torch.cat([b[1] for b in loader], dim=0)
        prior = dp_label_prior_from_counts(all_labels, num_classes=num_classes, sigma=1.0, device="cpu")

        def synth_sampler(batch_size: int, flow=flow) -> torch.Tensor:
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            return sample_flow_euler(flow.to(device).eval(), labels, n_steps=50).cpu()

        dp2 = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5) if dp_stage2 else None
        use_cellot = dp2 is not None and option_stage2.upper() == "A"

        real_x_loader = DataLoader(
            TensorDataset(ds.tensors[0]),
            batch_size=256,
            shuffle=True,
            drop_last=True,
        )

        if use_cellot:
            f = CellOTICNN(
                input_dim=d,
                hidden_units=[64, 64, 64, 64],
                activation="LeakyReLU",
                softplus_W_kernels=False,
                softplus_beta=1.0,
                fnorm_penalty=0.0,
                kernel_init_fxn=None,
            )
            ot = CellOTICNN(
                input_dim=d,
                hidden_units=[64, 64, 64, 64],
                activation="LeakyReLU",
                softplus_W_kernels=False,
                softplus_beta=1.0,
                fnorm_penalty=1.0,
                kernel_init_fxn=None,
            )
            train_ot_stage2_cellot(
                f,
                ot,
                source_loader=real_x_loader,
                target_loader=target_loader,
                epochs=30,
                n_inner_iters=10,
                lr_f=None,
                lr_g=None,
                optim_cfg={
                    "optimizer": "Adam",
                    "lr": 1e-4,
                    "beta1": 0.5,
                    "beta2": 0.9,
                    "weight_decay": 0.0,
                },
                dp=dp2,
                synth_sampler=synth_sampler,
                device=device,
            )
        else:
            ot = ICNN(d=d, hidden=[128, 128], act="relu", add_strong_convexity=0.1)
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

        clients_out.append({"flow": unwrap_model(flow).cpu(), "ot": unwrap_model(ot).cpu(), "prior": prior})

    Ysyn, Lsyn = server_synthesize(
        clients_out, M_per_client=5000, num_classes=num_classes, flow_steps=50, device=device
    )
    syn_loader = DataLoader(TensorDataset(Ysyn, Lsyn), batch_size=512, shuffle=True, drop_last=True)

    clf = Classifier(d=d, num_classes=num_classes, hidden=[128, 128])
    stats = train_classifier(clf, syn_loader, test_loader=target_test_loader, epochs=30, lr=1e-3, device=device)
    print("Final stats:", stats)
