import unittest

import torch

from noisyflow.attacks.membership_inference import loss_threshold_attack


class MembershipInferenceTests(unittest.TestCase):
    def test_loss_threshold_attack_separates(self):
        train_losses = torch.tensor([0.1, 0.2, 0.15, 0.3])
        test_losses = torch.tensor([0.8, 0.7, 0.9, 1.0])
        stats = loss_threshold_attack(train_losses, test_losses)
        self.assertGreaterEqual(stats["attack_acc"], 0.9)
        self.assertGreaterEqual(stats["attack_auc"], 0.9)
        self.assertIn("attack_threshold", stats)


if __name__ == "__main__":
    unittest.main()
