import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.attacks.membership_inference import collect_stage_features, run_stage_mia_attack
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage2.networks import ICNN


class StageMIATests(unittest.TestCase):
    def test_collect_stage_features_and_attack(self):
        torch.manual_seed(0)
        flow = VelocityField(d=2, num_classes=2, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        ot = ICNN(d=2, hidden=[8, 8], act="relu", add_strong_convexity=0.0)

        x_mem = torch.randn(10, 2)
        y_mem = torch.randint(0, 2, (10,))
        x_non = torch.randn(10, 2)
        y_non = torch.randint(0, 2, (10,))

        mem_loader = DataLoader(TensorDataset(x_mem, y_mem), batch_size=5, shuffle=False)
        non_loader = DataLoader(TensorDataset(x_non, y_non), batch_size=5, shuffle=False)

        mem_feats = collect_stage_features(
            flow,
            ot,
            mem_loader,
            use_ot=True,
            num_flow_samples=1,
            include_ot_transport_norm=True,
            seed=0,
            device="cpu",
        )
        non_feats = collect_stage_features(
            flow,
            ot,
            non_loader,
            use_ot=True,
            num_flow_samples=1,
            include_ot_transport_norm=True,
            seed=0,
            device="cpu",
        )

        self.assertEqual(mem_feats.shape[1], 3)
        self.assertEqual(non_feats.shape[1], 3)

        stats = run_stage_mia_attack(
            mem_feats,
            non_feats,
            attack_hidden=[8],
            attack_epochs=1,
            attack_lr=1e-2,
            attack_batch_size=4,
            attack_train_frac=0.5,
            max_samples=10,
            seed=0,
            device="cpu",
        )

        self.assertIn("stage_mia_attack_acc", stats)
        self.assertIn("stage_mia_attack_auc", stats)


if __name__ == "__main__":
    unittest.main()
