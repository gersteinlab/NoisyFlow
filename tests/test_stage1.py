import unittest

import torch

from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import flow_matching_loss, sample_flow_euler


class Stage1Tests(unittest.TestCase):
    def test_flow_matching_loss_scalar(self):
        torch.manual_seed(0)
        model = VelocityField(d=4, num_classes=3, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        x = torch.randn(5, 4)
        y = torch.tensor([0, 1, 2, 1, 0])
        loss = flow_matching_loss(model, x, y)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sample_flow_euler_shape(self):
        torch.manual_seed(0)
        model = VelocityField(d=3, num_classes=2, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        labels = torch.tensor([0, 1, 1, 0])
        samples = sample_flow_euler(model, labels, n_steps=5)
        self.assertEqual(samples.shape, (4, 3))
        self.assertFalse(samples.requires_grad)


if __name__ == "__main__":
    unittest.main()
