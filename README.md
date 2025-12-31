# NoisyFlow

NoisyFlow is a three-stage pipeline for federated synthetic data generation with optional differential privacy.
It trains a flow-matching generator per client, fits an optimal transport map to a target domain, and
then synthesizes data for downstream classification.

## Features
- Stage 1: flow matching generator with optional DP-SGD (Opacus).
- Stage 2: ICNN or CellOT transport (options A/B/C).
- Stage 3: server-side synthesis and classifier training.
- Optional privacy-utility sweeps and membership inference evaluations.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
pip install pyyaml  # required for YAML configs
pip install opacus  # optional, DP-SGD in stage1/stage2
pip install matplotlib  # optional, privacy curve plots
```

## Quickstart
```bash
python run.py --config configs/default.yaml
```

For a smaller smoke test:
```bash
python run.py --config configs/quick_smoke.yaml
```
Note: `configs/quick_smoke.yaml` sets `device: cuda`; switch to `cpu` if you do not have a GPU.

To run the toy demo script:
```bash
python noisyflow_sketch.py
```

## Configuration
- Configs live under `configs/` and are loaded by `run.py`.
- `data.type` supports `federated_mixture_gaussians`, `mixture_gaussians`, and `toy_federated_gaussians`.
- Enabling DP (`stage1.dp` / `stage2.dp`) requires Opacus; stage2 DP requires
  `stage2.cellot.enabled: true` and `stage2.option: A`.
- `privacy_curve.enabled: true` runs a sweep and writes `privacy_utility.png` (requires matplotlib).

## Documentation
Start here: `docs/README.md`.
- `docs/overview.md`: Pipeline overview and stage summary.
- `docs/configuration.md`: Full config reference.
- `docs/data.md`: Synthetic data builders.
- `docs/experiments.md`: CLI usage and experiment configs.
- `docs/attacks.md`: Membership inference attack details.
- `docs/architecture.md`: Code map.

## Tests
```bash
python -m unittest
```
Tests that depend on optional packages (pyyaml, opacus, matplotlib) are skipped if those packages
are not installed.
