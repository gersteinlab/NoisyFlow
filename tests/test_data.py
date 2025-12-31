import unittest

import torch

from noisyflow.data.synthetic import make_federated_mixture_gaussians
from noisyflow.data.toy import make_toy_federated_gaussians


class DataTests(unittest.TestCase):
    def test_make_federated_mixture_gaussians_shapes(self):
        client_datasets, target_ref, target_test = make_federated_mixture_gaussians(
            K=2,
            n_per_client=10,
            n_target_ref=8,
            n_target_test=6,
            d=3,
            num_classes=4,
            component_scale=1.0,
            component_cov=0.1,
            seed=0,
        )
        self.assertEqual(len(client_datasets), 2)
        for ds in client_datasets:
            x, y = ds.tensors
            self.assertEqual(x.shape, (10, 3))
            self.assertEqual(y.shape, (10,))
            self.assertTrue(torch.is_tensor(x))
            self.assertTrue(torch.is_tensor(y))
        y_ref = target_ref.tensors[0]
        self.assertEqual(y_ref.shape, (8, 3))
        y_test, l_test = target_test.tensors
        self.assertEqual(y_test.shape, (6, 3))
        self.assertEqual(l_test.shape, (6,))

    def test_make_toy_federated_gaussians_shapes(self):
        client_datasets, target_ref, target_test = make_toy_federated_gaussians(
            K=2,
            n_per_client=10,
            n_target_ref=8,
            n_target_test=6,
            d=2,
            num_classes=3,
            seed=1,
        )
        self.assertEqual(len(client_datasets), 2)
        for ds in client_datasets:
            x, y = ds.tensors
            self.assertEqual(x.shape, (10, 2))
            self.assertEqual(y.shape, (10,))
        y_ref = target_ref.tensors[0]
        self.assertEqual(y_ref.shape, (8, 2))
        y_test, l_test = target_test.tensors
        self.assertEqual(y_test.shape, (6, 2))
        self.assertEqual(l_test.shape, (6,))


if __name__ == "__main__":
    unittest.main()
