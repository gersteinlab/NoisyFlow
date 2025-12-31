# Code Map

## Entry points
- `run.py`: Main CLI entrypoint. Loads configs, runs experiments, and optionally sweeps DP noise multipliers.
- `noisyflow/demo.py`: Minimal end-to-end demo for the toy dataset.
- `noisyflow_sketch.py`: Convenience script that re-exports key functions and runs the toy demo.

## Configuration
- `noisyflow/config.py`: Dataclasses for config schemas and the YAML loader.

## Stage 1
- `noisyflow/stage1/networks.py`: `VelocityField` and `SinusoidalTimeEmbedding`.
- `noisyflow/stage1/training.py`: Flow matching loss, training loop, and Euler sampler.

## Stage 2
- `noisyflow/stage2/networks.py`: ICNN and CellOT ICNN architectures plus OT transport via gradients.
- `noisyflow/stage2/training.py`: ICNN OT training, CellOT training, and DP integration.

## Stage 3
- `noisyflow/stage3/networks.py`: MLP classifier for downstream evaluation.
- `noisyflow/stage3/training.py`: Server synthesis and classifier training/eval.

## Data
- `noisyflow/data/synthetic.py`: Main synthetic federated mixture builder.
- `noisyflow/data/toy.py`: Simpler toy builder.

## Attacks
- `noisyflow/attacks/membership_inference.py`: Loss-threshold MIA, shadow MIA, stage MIA, and stage shadow MIA.

## Utilities
- `noisyflow/nn.py`: Simple configurable MLP.
- `noisyflow/utils.py`: DP config, RNG seeding, and helper utilities.
