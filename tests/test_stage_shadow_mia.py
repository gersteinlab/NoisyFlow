import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.attacks.membership_inference import run_stage_shadow_attack
from noisyflow.data.synthetic import make_federated_mixture_gaussians
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage2.networks import ICNN
from noisyflow.stage1.training import train_flow_stage1
from noisyflow.stage2.training import train_ot_stage2


class StageShadowMIATests(unittest.TestCase):
    def test_run_stage_shadow_attack_smoke(self):
        torch.manual_seed(0)
        d = 2
        num_classes = 2

        client_datasets, target_ref, _ = make_federated_mixture_gaussians(
            K=1,
            n_per_client=20,
            n_target_ref=20,
            n_target_test=20,
            d=d,
            num_classes=num_classes,
            component_scale=1.0,
            component_cov=0.1,
            seed=0,
        )
        ds = client_datasets[0]
        train_ds = TensorDataset(ds.tensors[0][:10], ds.tensors[1][:10])
        holdout_ds = TensorDataset(ds.tensors[0][10:], ds.tensors[1][10:])

        flow = VelocityField(d=d, num_classes=num_classes, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        flow_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
        train_flow_stage1(flow, flow_loader, epochs=1, lr=1e-2, dp=None, device="cpu")

        ot = ICNN(d=d, hidden=[8, 8], act="relu", add_strong_convexity=0.0)
        real_x_loader = DataLoader(TensorDataset(train_ds.tensors[0]), batch_size=5, shuffle=True)
        target_loader = DataLoader(target_ref, batch_size=5, shuffle=True, drop_last=True)
        train_ot_stage2(
            ot,
            real_loader=real_x_loader,
            target_loader=target_loader,
            option="A",
            synth_sampler=None,
            epochs=1,
            lr=1e-2,
            dp=None,
            conj_steps=2,
            conj_lr=0.1,
            conj_clamp=5.0,
            device="cpu",
        )

        target_clients = [
            {
                "flow": flow,
                "ot": ot,
                "members": train_ds,
                "nonmembers": holdout_ds,
            }
        ]

        stats = run_stage_shadow_attack(
            data_builder=make_federated_mixture_gaussians,
            data_params={
                "K": 1,
                "n_per_client": 20,
                "n_target_ref": 20,
                "n_target_test": 20,
                "d": d,
                "num_classes": num_classes,
                "component_scale": 1.0,
                "component_cov": 0.1,
                "seed": 1,
            },
            target_clients=target_clients,
            flow_kwargs={
                "d": d,
                "num_classes": num_classes,
                "hidden": [8],
                "time_emb_dim": 8,
                "label_emb_dim": 8,
            },
            ot_kwargs={"d": d, "hidden": [8, 8], "act": "relu", "add_strong_convexity": 0.0},
            stage2_option="A",
            stage1_train_kwargs={"epochs": 1, "lr": 1e-2},
            stage2_train_kwargs={"epochs": 1, "lr": 1e-2, "conj_steps": 2, "conj_lr": 0.1, "conj_clamp": 5.0},
            batch_size=5,
            target_batch_size=5,
            drop_last=True,
            num_shadow_models=1,
            holdout_fraction=0.5,
            num_flow_samples=1,
            include_ot_transport_norm=True,
            attack_hidden=[8],
            attack_epochs=1,
            attack_lr=1e-2,
            attack_batch_size=5,
            attack_train_frac=0.5,
            max_samples_per_shadow=10,
            seed=0,
            data_overrides={"K": 1, "n_per_client": 10},
            device="cpu",
        )

        self.assertIn("stage_shadow_mia_acc", stats)
        self.assertIn("stage_shadow_mia_auc", stats)


if __name__ == "__main__":
    unittest.main()
