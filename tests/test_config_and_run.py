import importlib.util
import os
import tempfile
import unittest

from noisyflow.config import (
    DataConfig,
    ExperimentConfig,
    LabelPriorConfig,
    LoaderConfig,
    PrivacyCurveConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    load_config,
)
from run import run_experiment, run_privacy_curve


yaml_spec = importlib.util.find_spec("yaml")
if yaml_spec is not None:
    import yaml
else:
    yaml = None
opacus_spec = importlib.util.find_spec("opacus")
mpl_spec = importlib.util.find_spec("matplotlib")


class ConfigAndRunTests(unittest.TestCase):
    @unittest.skipIf(yaml is None, "pyyaml not installed")
    def test_load_config_yaml(self):
        data = {
            "seed": 7,
            "device": "cpu",
            "data": {
                "type": "federated_mixture_gaussians",
                "K": 2,
                "params": {"n_per_client": 10, "d": 2, "num_classes": 3},
            },
            "stage2": {"flow_steps": 7},
            "privacy_curve": {"enabled": True, "stage": "stage1", "noise_multipliers": [1.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/cfg.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
            cfg = load_config(path)

        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.data.type, "federated_mixture_gaussians")
        self.assertEqual(cfg.data.params["K"], 2)
        self.assertEqual(cfg.data.params["n_per_client"], 10)
        self.assertEqual(cfg.stage2.flow_steps, 7)
        self.assertTrue(cfg.privacy_curve.enabled)
        self.assertEqual(cfg.privacy_curve.noise_multipliers, [1.0])

    def test_run_experiment_smoke(self):
        cfg = ExperimentConfig(
            seed=0,
            device="cpu",
            data=DataConfig(
                type="federated_mixture_gaussians",
                params={
                    "K": 1,
                    "n_per_client": 20,
                    "n_target_ref": 20,
                    "n_target_test": 20,
                    "d": 2,
                    "num_classes": 3,
                    "component_scale": 1.0,
                    "component_cov": 0.1,
                    "seed": 0,
                },
            ),
            loaders=LoaderConfig(
                batch_size=10,
                target_batch_size=10,
                test_batch_size=10,
                synth_batch_size=10,
                drop_last=True,
            ),
            stage1=Stage1Config(
                epochs=1,
                lr=1e-3,
                hidden=[16],
                time_emb_dim=8,
                label_emb_dim=8,
                label_prior=LabelPriorConfig(enabled=False),
                dp=None,
            ),
            stage2=Stage2Config(
                option="B",
                epochs=1,
                lr=1e-3,
                hidden=[16],
                act="relu",
                add_strong_convexity=0.0,
                flow_steps=5,
                conj_steps=2,
                conj_lr=0.1,
                conj_clamp=5.0,
                dp=None,
            ),
            stage3=Stage3Config(
                epochs=1,
                lr=1e-3,
                hidden=[16],
                flow_steps=5,
                M_per_client=20,
            ),
            privacy_curve=PrivacyCurveConfig(enabled=False),
        )

        stats = run_experiment(cfg)
        self.assertIn("clf_loss", stats)
        self.assertIn("acc", stats)
        self.assertGreaterEqual(stats["acc"], 0.0)
        self.assertLessEqual(stats["acc"], 1.0)

    @unittest.skipIf(opacus_spec is None or mpl_spec is None, "opacus or matplotlib not installed")
    def test_run_privacy_curve_smoke(self):
        cfg = ExperimentConfig(
            seed=0,
            device="cpu",
            data=DataConfig(
                type="federated_mixture_gaussians",
                params={
                    "K": 1,
                    "n_per_client": 20,
                    "n_target_ref": 20,
                    "n_target_test": 20,
                    "d": 2,
                    "num_classes": 3,
                    "component_scale": 1.0,
                    "component_cov": 0.1,
                    "seed": 0,
                },
            ),
            loaders=LoaderConfig(
                batch_size=10,
                target_batch_size=10,
                test_batch_size=10,
                synth_batch_size=10,
                drop_last=True,
            ),
            stage1=Stage1Config(
                epochs=1,
                lr=1e-3,
                hidden=[16],
                time_emb_dim=8,
                label_emb_dim=8,
                label_prior=LabelPriorConfig(enabled=False),
                dp=None,
            ),
            stage2=Stage2Config(
                option="B",
                epochs=1,
                lr=1e-3,
                hidden=[16],
                act="relu",
                add_strong_convexity=0.0,
                flow_steps=5,
                conj_steps=2,
                conj_lr=0.1,
                conj_clamp=5.0,
                dp=None,
            ),
            stage3=Stage3Config(
                epochs=1,
                lr=1e-3,
                hidden=[16],
                flow_steps=5,
                M_per_client=20,
            ),
            privacy_curve=PrivacyCurveConfig(enabled=False),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "curve.png")
            curve = PrivacyCurveConfig(
                enabled=True,
                stage="stage1",
                noise_multipliers=[1.0],
                output_path=out_path,
            )
            results = run_privacy_curve(cfg, curve)
            self.assertEqual(len(results), 1)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
