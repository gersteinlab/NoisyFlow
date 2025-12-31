import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.attacks.membership_inference import run_shadow_attack
from noisyflow.data.synthetic import make_federated_mixture_gaussians
from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import train_classifier


class ShadowMIATests(unittest.TestCase):
    def test_run_shadow_attack_smoke(self):
        torch.manual_seed(0)
        x = torch.randn(40, 2)
        y = torch.randint(0, 2, (40,))
        train_ds = TensorDataset(x[:20], y[:20])
        test_ds = TensorDataset(x[20:], y[20:])
        train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=5, shuffle=False)

        clf = Classifier(d=2, num_classes=2, hidden=[8])
        train_classifier(clf, train_loader, test_loader=None, epochs=1, lr=1e-2, device="cpu")

        stats = run_shadow_attack(
            data_builder=make_federated_mixture_gaussians,
            data_params={
                "K": 1,
                "n_per_client": 10,
                "n_target_ref": 1,
                "n_target_test": 20,
                "d": 2,
                "num_classes": 2,
                "component_scale": 1.0,
                "component_cov": 0.1,
                "seed": 1,
            },
            d=2,
            num_classes=2,
            target_model=clf,
            target_member_loader=train_loader,
            target_nonmember_loader=test_loader,
            num_shadow_models=1,
            shadow_train_size=10,
            shadow_test_size=10,
            shadow_epochs=1,
            shadow_lr=1e-2,
            shadow_hidden=[8],
            shadow_batch_size=5,
            attack_epochs=1,
            attack_lr=1e-2,
            attack_hidden=[8],
            attack_batch_size=5,
            feature_set="stats",
            max_samples_per_shadow=10,
            seed=0,
            data_overrides={"K": 1, "n_per_client": 5, "n_target_ref": 1},
            device="cpu",
        )

        self.assertIn("shadow_attack_acc", stats)
        self.assertIn("shadow_attack_auc", stats)
        self.assertTrue(0.0 <= stats["shadow_attack_acc"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
