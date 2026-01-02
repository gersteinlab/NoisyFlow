# Repository Guidelines

## Overview

NoisyFlow is a three-stage federated synthetic data generation pipeline with optional differential privacy. See `README.md` for full feature description and installation instructions.

---

## Project Structure & Module Organization

```
noisyflow/               # Main package
├── stage1/              # Flow-matching generator (networks.py, training.py)
├── stage2/              # Optimal transport (ICNN, CellOT, RectifiedFlowOT)
├── stage3/              # Server synthesis & classifier training
├── data/                # Synthetic data builders
├── attacks/             # Membership inference attack utilities
├── config.py            # ExperimentConfig dataclass & YAML loading
├── metrics.py           # SW2, MMD, evaluation metrics
├── utils.py             # Helpers (DPConfig, seeding, model unwrapping)
└── nn.py                # Shared neural network building blocks

configs/                 # YAML experiment configs used by run.py
tests/                   # Unit test suite (test_*.py)
scripts/                 # Standalone utility scripts (plotting, benchmarking, data fetching)
docs/                    # Detailed documentation (start with docs/README.md)
tex/                     # LaTeX source files for paper figures/tables
plots/                   # Generated figures and visualizations (.pdf)
run.py                   # Primary CLI entrypoint
```

### Key Conventions

- Stage-specific logic belongs in `noisyflow/stageX/` with paired `networks.py` (models) and `training.py` (training loops).
- Shared utilities go in `noisyflow/utils.py`; evaluation metrics in `noisyflow/metrics.py`.
- Data loaders and builders are defined in `noisyflow/data/`.

---

## Environment Setup

```bash
# Preferred: conda environment
conda activate edm

# Alternative: venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package      | Required | Notes                                        |
|--------------|----------|----------------------------------------------|
| numpy        | Yes      | Core numerical operations                    |
| torch        | Yes      | Neural network framework                     |
| pyyaml       | Yes      | Config loading                               |
| opacus       | No       | DP-SGD in stage1/stage2 (optional)           |
| matplotlib   | No       | Privacy curve plots (optional)               |
| scikit-learn | No       | RandomForest baseline classifier (optional)  |

---

## Build, Test, and Development Commands

### Running Experiments

```bash
# Default experiment
python run.py --config configs/default.yaml

# Small smoke test (GPU)
python run.py --config configs/quick_smoke.yaml

# Smoke test (CPU fallback)
python run.py --config configs/quick_smoke.yaml  # edit device: cpu in YAML

# Toy demo script
python noisyflow_sketch.py
```

### Running Tests

```bash
# Full test suite
python -m unittest discover -s tests

# Single test module
python -m unittest tests.test_stage1

# Specific test class
python -m unittest tests.test_stage1.TestFlowTraining
```

**Note:** Prefer CUDA/GPU for tests when available; some tests skip on CPU for speed.

---

## Coding Style & Naming Conventions

- **Indentation:** 4 spaces (PEP 8 compliant).
- **Type hints:** Use them consistently across functions and method signatures.
- **Naming:**
  - `snake_case` for functions, variables, modules
  - `CapWords` (PascalCase) for classes
  - `UPPER_SNAKE_CASE` for constants
- **Imports:** Group stdlib → third-party → local, separated by blank lines.
- **Docstrings:** Use Google-style docstrings for public functions and classes.

### File Organization

```python
# At top of file:
from __future__ import annotations  # If using forward refs

import stdlib_modules
import third_party_modules
from noisyflow.module import local_imports

# Constants
DEFAULT_EPOCHS = 100

# Classes
class MyNetwork(nn.Module):
    ...

# Functions
def train_model(...):
    ...
```

---

## Testing Guidelines

- Tests use `unittest` framework with `unittest.TestCase` classes in `tests/test_*.py`.
- Tests that require optional dependencies (PyYAML, Opacus, Matplotlib) are skipped gracefully if packages are missing.
- **Favor fast, deterministic tests:** Use small data sizes, few epochs, and fixed seeds when modifying training or data paths.
- Run the conda environment: `conda activate edm` before testing.

### Test Naming

```python
class TestStage1Training(unittest.TestCase):
    def test_flow_converges_on_toy_data(self):
        ...
    def test_dp_sgd_enabled_computes_epsilon(self):
        ...
```

---

## Configuration & Reproducibility

- Experiment settings live in `configs/*.yaml`; modify existing configs or create new ones.
- `run.py` defaults to `configs/default.yaml`.
- Enable privacy-utility sweeps with `privacy_curve.enabled: true`.

### Key Config Blocks

```yaml
seed: 42                    # For reproducibility
device: cuda                # or cpu
data:
  type: federated_mixture_gaussians
  params: {...}
stage1:
  epochs: 100
  dp:                       # Optional DP-SGD config
    enabled: true
    noise_multiplier: 1.0
stage2:
  option: A                 # A, B, or C
  cellot:
    enabled: false
  rectified_flow:
    enabled: true
stage3:
  M_per_client: 1000
privacy_curve:
  enabled: false
```

### Reproducibility Checklist

1. Set `seed` in config for deterministic results.
2. Record the full config YAML with experiment outputs.
3. Note CUDA/cuDNN versions if results are device-sensitive.

---

## Commit & Pull Request Guidelines

- Use short, imperative commit subjects: `"Add stage2 smoke config"`, `"Fix DP epsilon computation"`.
- PRs should include:
  - Brief summary of changes
  - Config files used (if applicable)
  - Tests run and their outcomes
  - Generated plots (e.g., `plots/privacy_utility.pdf`) when relevant

---

## Plotting & Figures

- **Output location:** Place all plots in `plots/` folder.
- **Format:** Use `.pdf` (vector graphics) for publication quality.
- **Style requirements:**
  - Distinct line types and clear markers
  - Readable axis labels and legends
  - Colorblind-friendly palettes (e.g., viridis, ColorBrewer)
- **Scripts:** Use `scripts/plot_*.py` for reusable plotting utilities.

### Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epsilons, accuracies, marker="o", linewidth=2)
ax.set_xlabel(r"Privacy budget $\varepsilon$")
ax.set_ylabel("Accuracy")
ax.grid(True, alpha=0.3)
fig.savefig("plots/privacy_utility.pdf", bbox_inches="tight")
```

---

## Scripts & Utilities

Located in `scripts/`:

| Script                                 | Purpose                                 |
|----------------------------------------|-----------------------------------------|
| `fetch_cellot_datasets.py`             | Download CellOT/Lupus datasets          |
| `plot_privacy_curve_stage2_schemes.py` | Compare Stage2 transport schemes        |
| `plot_sw2_compare.py`                  | Sliced Wasserstein distance comparisons |
| `benchmark_inference_ot.py`            | OT inference benchmarking               |

---

## Documentation

Start with `docs/README.md`. Key documents:

- `docs/overview.md` — Pipeline overview and stage summary
- `docs/configuration.md` — Full config reference
- `docs/data.md` — Synthetic data builders
- `docs/experiments.md` — CLI usage and experiment recipes
- `docs/attacks.md` — Membership inference attack details
- `docs/architecture.md` — Code map and module relationships

---

## Privacy & Differential Privacy Notes

- DP-SGD is supported in Stage1 and Stage2 via [Opacus](https://opacus.ai/).
- Enable with `stage1.dp.enabled: true` or `stage2.dp.enabled: true`.
- Stage2 DP requires `stage2.cellot.enabled: true` and `stage2.option: A`.
- Privacy budget (ε) is computed automatically and reported in experiment stats.
- Use `privacy_curve.enabled: true` for ε vs. utility sweeps.

---

## LaTeX & Paper Writing (COLT Style)

LaTeX source files live in `tex/`. Follow a **COLT conference tone**: precise, mathematical, and direct.

### Writing Principles

- **Be concise.** Every sentence should convey information; cut filler words.
- **Be precise.** Use exact mathematical language; avoid vague qualifiers.
- **Be direct.** State results and methods plainly; avoid hedging.

### Avoid Fluffy Language

| ❌ Avoid | ✅ Prefer |
|----------|----------|
| "It is interesting to note that..." | (delete, state the fact directly) |
| "We can clearly see that..." | (delete, state the observation) |
| "In order to" | "To" |
| "Due to the fact that" | "Because" / "Since" |
| "A large number of" | "Many" / give the exact count |
| "very", "really", "quite", "extremely" | (delete or quantify) |
| "It should be noted that" | (delete) |
| "We would like to point out" | (delete, just state it) |
| "plays an important role" | "affects" / "determines" |
| "state-of-the-art" | cite the specific method |
| "novel", "innovative" | describe what is new concretely |

### Sentence Structure

- **Active voice:** "We train the model" not "The model is trained."
- **Subject-verb-object:** Front-load the main claim; defer qualifications.
- **One idea per sentence:** Split run-on sentences.

### Mathematical Writing

- Define notation before use: "Let $X \in \mathbb{R}^d$ denote the input."
- Use `\coloneqq` (or `:=`) for definitions vs. `=` for equalities.
- Number only referenced equations; use `\eqref{}` for references.
- Keep inline math short; display complex expressions.

### Example Transformations

```latex
% ❌ Fluffy
It is worth noting that our method achieves a very significant 
improvement over the baseline in terms of accuracy.

% ✅ Direct
Our method improves accuracy over the baseline by 12\%.
```

```latex
% ❌ Vague
The privacy guarantee is quite strong under reasonable assumptions.

% ✅ Precise  
Under Assumption~\ref{ass:bounded}, the mechanism satisfies 
$(\varepsilon, \delta)$-DP with $\varepsilon = 1$ and $\delta = 10^{-5}$.
```

### COLT-Specific Conventions

- Theorem environments: `\begin{theorem}`, `\begin{lemma}`, `\begin{proposition}`
- Proofs: Use `\begin{proof}...\end{proof}` with `\qedhere` if needed.
- References: `Theorem~\ref{thm:main}`, `Section~\ref{sec:method}` (capitalized, non-breaking space).
- Citations: `\citet{Author2024}` for "Author (2024) showed..." and `\citep{Author2024}` for "(Author, 2024)".
