import tempfile
import unittest

import numpy as np
import torch

from noisyflow.data.cell import make_federated_cell_dataset


class CellDataTests(unittest.TestCase):
    def _write_npz(self, path: str) -> None:
        rng = np.random.default_rng(0)
        # 3 clients, 2 conditions, 2 labels.
        clients = np.array([0] * 20 + [1] * 20 + [2] * 20)
        conditions = np.array(["ctrl"] * 10 + ["stim"] * 10)  # length 20 pattern
        conditions = np.tile(conditions, 3)
        labels = np.array(["A"] * 30 + ["B"] * 30)
        x = rng.normal(size=(60, 5)).astype(np.float32)

        np.savez(path, X=x, label=labels, client=clients, condition=conditions)

    def test_make_federated_cell_dataset_iid_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/toy_cells.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_cell_dataset(
                path=npz_path,
                split_mode="iid",
                source_condition="ctrl",
                target_condition="stim",
                target_test_size=0.25,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 3)
            for ds in client_datasets:
                x, y = ds.tensors
                self.assertEqual(x.shape[1], 5)
                self.assertEqual(x.shape[0], 10)
                self.assertEqual(y.shape, (10,))
                self.assertTrue(torch.isfinite(x).all())

            x_ref, y_ref = target_ref.tensors
            x_test, y_test = target_test.tensors
            self.assertEqual(x_ref.shape[1], 5)
            self.assertEqual(x_test.shape[1], 5)
            self.assertEqual(y_ref.dim(), 1)
            self.assertEqual(y_test.dim(), 1)
            self.assertGreater(int(x_ref.shape[0]), 0)
            self.assertGreater(int(x_test.shape[0]), 0)
            self.assertLessEqual(int(y_ref.max().item()), 1)
            self.assertLessEqual(int(y_test.max().item()), 1)

    def test_make_federated_cell_dataset_ood_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/toy_cells.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_cell_dataset(
                path=npz_path,
                split_mode="ood",
                holdout_client=2,
                source_condition="ctrl",
                target_condition="stim",
                seed=0,
            )

            # Only clients 0 and 1 remain for source.
            self.assertEqual(len(client_datasets), 2)
            self.assertEqual(int(target_test.tensors[0].shape[0]), 10)  # holdout has 10 stim rows
            self.assertEqual(int(target_ref.tensors[0].shape[0]), 20)  # other clients have 20 stim rows


if __name__ == "__main__":
    unittest.main()

