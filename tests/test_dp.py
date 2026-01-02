import importlib.util
import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import train_flow_stage1
from noisyflow.stage2.networks import CellOTICNN, RectifiedFlowOT
from noisyflow.stage2.training import train_ot_stage2_cellot, train_ot_stage2_rectified_flow
from noisyflow.utils import DPConfig


opacus_spec = importlib.util.find_spec("opacus")


@unittest.skipIf(opacus_spec is None, "opacus not installed")
class DPTests(unittest.TestCase):
    def test_stage1_dp_training_reports_epsilon(self):
        torch.manual_seed(0)
        x = torch.randn(20, 3)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=True, drop_last=True)

        model = VelocityField(d=3, num_classes=2, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        dp = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5)
        stats = train_flow_stage1(model, loader, epochs=1, lr=1e-3, dp=dp, device="cpu")

        self.assertIn("epsilon_flow", stats)
        self.assertGreater(stats["epsilon_flow"], 0.0)

    def test_stage2_dp_with_opacus_step_by_step(self):
        """Step-by-step: build loaders, init models, run DP training, assert DP stats."""
        torch.manual_seed(0)

        x = torch.randn(20, 2)
        y = torch.randn(20, 2)
        real_loader = DataLoader(TensorDataset(x), batch_size=5, shuffle=True, drop_last=True)
        target_loader = DataLoader(TensorDataset(y), batch_size=5, shuffle=True, drop_last=True)

        f = CellOTICNN(
            input_dim=2,
            hidden_units=[8, 8],
            activation="LeakyReLU",
            softplus_W_kernels=False,
            softplus_beta=1.0,
            fnorm_penalty=0.0,
        )
        g = CellOTICNN(
            input_dim=2,
            hidden_units=[8, 8],
            activation="LeakyReLU",
            softplus_W_kernels=False,
            softplus_beta=1.0,
            fnorm_penalty=0.0,
        )

        def synth_sampler(bs: int) -> torch.Tensor:
            return torch.randn(bs, 2)

        dp = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5)
        stats = train_ot_stage2_cellot(
            f,
            g,
            source_loader=real_loader,
            target_loader=target_loader,
            epochs=1,
            n_inner_iters=2,
            lr_f=1e-3,
            lr_g=1e-3,
            dp=dp,
            synth_sampler=synth_sampler,
            device="cpu",
        )

        self.assertIn("epsilon_ot", stats)
        self.assertIn("delta_ot", stats)
        self.assertIn("ot_loss", stats)
        self.assertGreater(stats["epsilon_ot"], 0.0)
        self.assertGreater(stats["delta_ot"], 0.0)
        self.assertTrue(math.isfinite(stats["epsilon_ot"]))
        self.assertTrue(math.isfinite(stats["ot_loss"]))
        self.assertEqual(stats["delta_ot"], dp.delta)

    def test_stage2_rectified_flow_dp_reports_epsilon(self):
        torch.manual_seed(0)
        x = torch.randn(20, 2)
        y = torch.randn(20, 2)
        source_loader = DataLoader(TensorDataset(x), batch_size=5, shuffle=True, drop_last=True)
        target_loader = DataLoader(TensorDataset(y), batch_size=7, shuffle=True, drop_last=True)

        model = RectifiedFlowOT(d=2, hidden=[8, 8], time_emb_dim=8, act="silu", transport_steps=5)
        dp = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5)
        stats = train_ot_stage2_rectified_flow(
            model,
            source_loader=source_loader,
            target_loader=target_loader,
            option="A",
            epochs=1,
            lr=1e-3,
            dp=dp,
            device="cpu",
        )

        self.assertIn("epsilon_ot", stats)
        self.assertIn("delta_ot", stats)
        self.assertIn("ot_loss", stats)
        self.assertGreater(stats["epsilon_ot"], 0.0)
        self.assertGreater(stats["delta_ot"], 0.0)
        self.assertTrue(math.isfinite(stats["epsilon_ot"]))
        self.assertTrue(math.isfinite(stats["ot_loss"]))
        self.assertEqual(stats["delta_ot"], dp.delta)

    def test_stage2_rectified_flow_mixed_dp_reports_epsilon(self):
        torch.manual_seed(0)
        x = torch.randn(20, 2)
        y = torch.randn(20, 2)
        source_loader = DataLoader(TensorDataset(x), batch_size=5, shuffle=True, drop_last=True)
        target_loader = DataLoader(TensorDataset(y), batch_size=7, shuffle=True, drop_last=True)

        model = RectifiedFlowOT(d=2, hidden=[8, 8], time_emb_dim=8, act="silu", transport_steps=5)

        def synth_sampler(bs: int, **_kwargs) -> torch.Tensor:
            return torch.randn(bs, 2)

        dp = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5)
        stats = train_ot_stage2_rectified_flow(
            model,
            source_loader=source_loader,
            target_loader=target_loader,
            option="C",
            synth_sampler=synth_sampler,
            epochs=1,
            lr=1e-3,
            dp=dp,
            device="cpu",
        )

        self.assertIn("epsilon_ot", stats)
        self.assertIn("delta_ot", stats)
        self.assertIn("ot_loss", stats)
        self.assertGreater(stats["epsilon_ot"], 0.0)
        self.assertGreater(stats["delta_ot"], 0.0)
        self.assertTrue(math.isfinite(stats["epsilon_ot"]))
        self.assertTrue(math.isfinite(stats["ot_loss"]))
        self.assertEqual(stats["delta_ot"], dp.delta)


if __name__ == "__main__":
    unittest.main()
