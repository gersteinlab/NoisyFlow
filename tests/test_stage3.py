import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import eval_classifier, sample_labels_from_prior, train_classifier


class Stage3Tests(unittest.TestCase):
    def test_sample_labels_from_prior(self):
        prior = torch.tensor([0.2, 0.3, 0.5])
        labels = sample_labels_from_prior(prior, 10)
        self.assertEqual(labels.shape, (10,))
        self.assertTrue(labels.dtype == torch.int64)

    def test_train_classifier_smoke(self):
        torch.manual_seed(0)
        x = torch.randn(20, 3)
        y = torch.randint(0, 2, (20,))
        train_loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=True)
        test_loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
        clf = Classifier(d=3, num_classes=2, hidden=[8])
        stats = train_classifier(clf, train_loader, test_loader=test_loader, epochs=1, lr=1e-2, device="cpu")
        self.assertIn("clf_loss", stats)
        self.assertIn("acc", stats)
        self.assertGreaterEqual(stats["acc"], 0.0)
        self.assertLessEqual(stats["acc"], 1.0)

    def test_eval_classifier(self):
        torch.manual_seed(0)
        x = torch.randn(10, 2)
        y = torch.zeros(10, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
        clf = Classifier(d=2, num_classes=1, hidden=[4])
        stats = eval_classifier(clf, loader, device="cpu")
        self.assertIn("acc", stats)
        self.assertGreaterEqual(stats["acc"], 0.0)
        self.assertLessEqual(stats["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
