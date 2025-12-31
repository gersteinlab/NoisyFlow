# Repository Guidelines

## Project Structure & Module Organization
- `noisyflow/` is the main package, with core logic and utilities.
- `noisyflow/stage1/`, `noisyflow/stage2/`, `noisyflow/stage3/` each contain `networks.py` and `training.py` for the three pipeline stages.
- `noisyflow/data/` hosts synthetic data builders; `noisyflow/attacks/` holds membership inference utilities.
- `configs/` contains YAML experiment configs used by `run.py`.
- `tests/` contains the unit test suite; `run.py` is the primary CLI entrypoint.

## Build, Test, and Development Commands
- `python run.py --config configs/default.yaml` runs the default experiment.
- `python run.py --config configs/quick_smoke.yaml` runs a small smoke test (set `device: cpu` if you do not have CUDA).
- `python -m unittest discover -s tests` runs the full test suite.
- `python -m unittest tests.test_stage1` runs a single test module.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 style; type hints are common across modules.
- Use `snake_case` for functions, variables, and modules; use `CapWords` for classes.
- Keep stage-specific changes inside the matching `noisyflow/stageX/` directory and pair models with their training helpers.

## Testing Guidelines
- conda activate edm to activate conda envs
- Tests are written with `unittest` in `tests/test_*.py` files and `unittest.TestCase` classes.
- Some tests are skipped if optional dependencies (PyYAML, Opacus, Matplotlib) are missing; mention skips in PR notes.
- Favor fast, deterministic tests when modifying training or data paths.

## Commit & Pull Request Guidelines
- The git history only shows `first-commit`, so no strict convention exists; use short, imperative subjects (e.g., "Add stage2 smoke config").
- PRs should include a brief summary, configs used, tests run, and any generated plots (e.g., `privacy_utility.png`) when relevant.

## Configuration & Reproducibility Tips
- Experiment settings live in `configs/*.yaml`. Keep changes localized and update `seed`, `device`, and stage blocks as needed.
- `run.py` defaults to `configs/default.yaml`; enable sweeps with `privacy_curve.enabled: true`.
