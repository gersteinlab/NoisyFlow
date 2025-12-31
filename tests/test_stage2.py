import unittest

import torch

from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.stage2.training import (
    approx_conjugate,
    compute_loss_f,
    compute_loss_g,
    ot_dual_loss,
    rectified_flow_ot_loss,
)


class Stage2Tests(unittest.TestCase):
    def test_icnn_transport_shape(self):
        torch.manual_seed(0)
        model = ICNN(d=3, hidden=[4, 4], act="relu", add_strong_convexity=0.0)
        x = torch.randn(5, 3)
        y = model.transport(x)
        self.assertEqual(y.shape, (5, 3))
        self.assertTrue(torch.isfinite(y).all())

    def test_ot_dual_loss_scalar(self):
        torch.manual_seed(0)
        model = ICNN(d=2, hidden=[4, 4], act="relu", add_strong_convexity=0.0)
        x = torch.randn(6, 2)
        y = torch.randn(6, 2)
        loss = ot_dual_loss(model, x, y, conj_steps=3, conj_lr=0.1, conj_clamp=5.0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_approx_conjugate_shapes(self):
        torch.manual_seed(0)
        model = ICNN(d=2, hidden=[4, 4], act="relu", add_strong_convexity=0.0)
        y = torch.randn(4, 2)
        phi_star, x_star = approx_conjugate(model, y, n_steps=3, lr=0.1, clamp=5.0)
        self.assertEqual(phi_star.shape, (4,))
        self.assertEqual(x_star.shape, (4, 2))

    def test_cellot_transport_shape(self):
        torch.manual_seed(0)
        model = CellOTICNN(
            input_dim=3,
            hidden_units=[4, 4],
            activation="LeakyReLU",
            softplus_W_kernels=False,
            softplus_beta=1.0,
            fnorm_penalty=0.0,
        )
        x = torch.randn(5, 3).requires_grad_(True)
        y = model.transport(x)
        self.assertEqual(y.shape, (5, 3))
        self.assertTrue(torch.isfinite(y).all())

    def test_cellot_losses_scalar(self):
        torch.manual_seed(0)
        f = CellOTICNN(
            input_dim=2,
            hidden_units=[4, 4],
            activation="LeakyReLU",
            softplus_W_kernels=False,
            softplus_beta=1.0,
            fnorm_penalty=0.0,
        )
        g = CellOTICNN(
            input_dim=2,
            hidden_units=[4, 4],
            activation="LeakyReLU",
            softplus_W_kernels=False,
            softplus_beta=1.0,
            fnorm_penalty=0.0,
        )
        source = torch.randn(6, 2).requires_grad_(True)
        target = torch.randn(6, 2)
        gl = compute_loss_g(f, g, source).mean()
        fl = compute_loss_f(f, g, source, target).mean()
        self.assertEqual(gl.dim(), 0)
        self.assertEqual(fl.dim(), 0)
        self.assertTrue(torch.isfinite(gl).item())
        self.assertTrue(torch.isfinite(fl).item())

    def test_rectified_flow_transport_shape(self):
        torch.manual_seed(0)
        model = RectifiedFlowOT(d=3, hidden=[8, 8], time_emb_dim=8, act="silu", transport_steps=5)
        x = torch.randn(5, 3).requires_grad_(True)
        y = model.transport(x)
        self.assertEqual(y.shape, (5, 3))
        self.assertTrue(torch.isfinite(y).all())

    def test_rectified_flow_loss_scalar(self):
        torch.manual_seed(0)
        model = RectifiedFlowOT(d=2, hidden=[8], time_emb_dim=8, act="silu", transport_steps=3)
        source = torch.randn(6, 2)
        target = torch.randn(6, 2)
        loss = rectified_flow_ot_loss(model, source, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
