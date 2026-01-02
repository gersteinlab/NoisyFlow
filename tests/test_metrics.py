import unittest

import torch

from noisyflow.metrics import rbf_mmd2, rbf_mmd2_multi_gamma, sliced_w2_distance


class MetricsTests(unittest.TestCase):
    def test_rbf_mmd2_scalar_finite(self):
        torch.manual_seed(0)
        x = torch.randn(64, 5)
        y = torch.randn(48, 5)
        mmd2 = rbf_mmd2(x, y, gamma=1.0)
        self.assertEqual(mmd2.dim(), 0)
        self.assertTrue(torch.isfinite(mmd2).item())
        self.assertGreaterEqual(float(mmd2.item()), -1e-6)

    def test_rbf_mmd2_multi_gamma_len(self):
        torch.manual_seed(0)
        x = torch.randn(300, 3)
        y = torch.randn(200, 3)
        out = rbf_mmd2_multi_gamma(x, y, gammas=[0.5, 1.0, 2.0], max_samples=128, seed=0)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(out)).tolist()))

    def test_sliced_w2_distance_nonnegative(self):
        torch.manual_seed(0)
        x = torch.randn(256, 5)
        y = torch.randn(256, 5)
        d = sliced_w2_distance(x, y, num_projections=32, max_samples=128, seed=0)
        self.assertTrue(torch.isfinite(torch.tensor(d)).item())
        self.assertGreaterEqual(float(d), 0.0)

    def test_sliced_w2_distance_zero_on_identical(self):
        torch.manual_seed(0)
        x = torch.randn(128, 4)
        d = sliced_w2_distance(x, x.clone(), num_projections=64, max_samples=None, seed=0)
        self.assertLessEqual(abs(float(d)), 1e-6)


if __name__ == "__main__":
    unittest.main()
