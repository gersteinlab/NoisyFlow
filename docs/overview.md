# NoisyFlow Overview

## Pipeline summary
NoisyFlow is a three-stage pipeline for federated synthetic data generation with optional differential privacy (DP). Each client trains a flow-matching generator, learns an optimal transport (OT) map into a target domain, and then a server synthesizes labeled samples for downstream classification.

## Stage 1: Client flow matching
- Model: `VelocityField` in `noisyflow/stage1/networks.py`.
- Loss: flow matching over random time and Gaussian noise in `noisyflow/stage1/training.py`.
- Optional DP-SGD via Opacus when `stage1.dp.enabled: true`.
- Optional label prior with noisy counts via `stage1.label_prior`.
- Output per client: trained flow model and (optional) label prior.

## Stage 2: Client OT map
- Model options: ICNN (`noisyflow/stage2/networks.py`) or CellOT ICNN pair.
- Options in `stage2.option`:
  - A: real client data to target reference data.
  - B: synthetic data only (post-processing of stage 1).
  - C: mixed real + synthetic (concatenated batches).
- ICNN training lives in `noisyflow/stage2/training.py`.
- CellOT training is enabled by `stage2.cellot.enabled: true` and supports option A.
- Stage 2 DP (from `stage2.dp`) requires CellOT with option A in the CLI entrypoint.

## Stage 3: Server synthesis and classifier
- Server samples labels from each client prior (or uniform), draws flow samples, then transports them with the OT map.
- Synthesis and classifier training live in `noisyflow/stage3/training.py`.
- Final classifier metrics are reported in `run.py`.

## Privacy curve and attacks
- `privacy_curve` runs multiple experiments across noise multipliers and writes a plot if Matplotlib is installed.
- Membership inference attacks live in `noisyflow/attacks/membership_inference.py`.
- See `docs/attacks.md` for the attack configuration and outputs.
