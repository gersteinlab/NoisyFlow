import unittest

import numpy as np
import torch

from noisyflow.utils import dp_label_prior_from_counts, set_seed


class UtilsTests(unittest.TestCase):
    def test_set_seed_reproducible(self):
        set_seed(123)
        py_val1 = np.random.randint(0, 100000)
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()

        set_seed(123)
        py_val2 = np.random.randint(0, 100000)
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        self.assertEqual(py_val1, py_val2)
        self.assertEqual(np_val1, np_val2)
        self.assertEqual(torch_val1, torch_val2)

    def test_dp_label_prior_from_counts_sigma_zero(self):
        labels = torch.tensor([0, 0, 1, 2, 2, 2])
        prior = dp_label_prior_from_counts(labels, num_classes=3, sigma=0.0)
        expected = torch.tensor([2 / 6, 1 / 6, 3 / 6], dtype=prior.dtype)
        self.assertTrue(torch.allclose(prior, expected, atol=1e-6))
        self.assertAlmostEqual(float(prior.sum().item()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
