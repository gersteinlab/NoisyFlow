# Configuration Reference

All experiments are configured via YAML files in `configs/` and loaded by `run.py`.

## Config loading behavior
- `data.params` is merged with any extra keys under `data` that are not `type` or `params`.
  This lets you write either `data.params.K` or `data.K` in YAML.

Example:
```yaml
seed: 0
device: cpu

data:
  type: federated_mixture_gaussians
  K: 3
  n_per_client: 1500
  params:
    seed: 0
```

## Top-level keys
- `seed`: Random seed for NumPy and PyTorch.
- `device`: `cpu` or `cuda`.
- `data`: Dataset builder settings.
- `loaders`: Batch sizes for all stages.
- `stage1`, `stage2`, `stage3`: Per-stage model and training settings.
- `privacy_curve`: Sweep settings for DP noise multipliers.
- `membership_inference`: Loss-threshold attack on the final classifier.
- `shadow_mia`: Shadow-model attack on the final classifier.
- `stage_mia`: Attack on stage 1/2 models using per-example features.
- `stage_shadow_mia`: Shadow-model attack on stage 1/2 models.

## `data`
- `type`: `federated_mixture_gaussians`, `mixture_gaussians` (alias), `toy_federated_gaussians`, `federated_cell_dataset`, or `cellot_lupuspatients_kang_hvg`.
- `params`: Passed through to the chosen data builder. See `docs/data.md`.

## `loaders`
- `batch_size`: Client training batches.
- `target_batch_size`: Target reference batches.
- `test_batch_size`: Target test batches.
- `synth_batch_size`: Synthetic batches for classifier training.
- `drop_last`: Whether to drop last partial batch.

## `stage1`
- `epochs`, `lr`: Flow training schedule.
- `hidden`: Hidden layer widths for the flow MLP.
- `time_emb_dim`, `label_emb_dim`: Embedding dimensions.
- `label_prior`: Optional DP label prior from noisy counts.
  - `enabled`: Boolean.
  - `mechanism`: `gaussian` or `laplace`.
  - `sigma`: Noise scale.
- `dp`: Optional DP-SGD config (Opacus required).
  - `enabled`, `max_grad_norm`, `noise_multiplier`, `delta`.

## `stage2`
- `option`: `A`, `B`, or `C`.
- `epochs`, `lr`: OT training schedule.
- `hidden`, `act`, `add_strong_convexity`: ICNN settings.
- `flow_steps`: Flow sampling steps for synthetic batches.
- `conj_steps`, `conj_lr`, `conj_clamp`: Conjugate approximation settings.
- `dp`: Optional DP-SGD config.
  - In `run.py`, DP for stage 2 requires `stage2.cellot.enabled: true` and `stage2.option: A`.
- `cellot`: CellOT settings (option A only).
  - `enabled`: Boolean.
  - `hidden_units`, `activation`, `softplus_W_kernels`, `softplus_beta`.
  - `kernel_init`: `name: uniform|normal` with `a`/`b` or `mean`/`std`.
  - `optim`: Adam settings. Optional `f` or `g` sub-keys override per-network optimizer args.
  - `f_fnorm_penalty`, `g_fnorm_penalty`.
  - `n_inner_iters`: Inner steps per outer update.
  - `n_iters`: Optional total update steps (overrides epochs * steps per epoch).

## `stage3`
- `epochs`, `lr`: Classifier training schedule.
- `hidden`: Classifier MLP sizes.
- `flow_steps`: Euler steps for server-side sampling.
- `M_per_client`: Number of synthetic samples per client.
- `ref_train_size`: Labeled target points used for `acc_ref_only` / `acc_ref_plus_synth` baselines.
- `combined_synth_train_size`: Optional cap on synthetic samples used in the `ref+synth` classifier training.

## `privacy_curve`
- `enabled`: Boolean. If true, `run.py` runs a sweep and writes a plot.
- `stage`: `stage1`, `stage2`, or `both`.
  - `stage2` and `both` require `stage2.option` to be `A` or `C`.
- `metric`: Which `run_experiment` stats key to plot as utility (default: `acc`).
  - Common choices: `acc` (synth-only), `acc_ref_plus_synth`, `acc_ref_only`.
- `noise_multipliers`: List of DP noise multipliers.
- `output_path`: Plot file path (requires Matplotlib).

## `membership_inference`
- `enabled`: Boolean.
- `max_samples`: Max samples per class when balancing train/test losses.
- `seed`: Sampling seed.

## `shadow_mia`
- `enabled`: Boolean.
- `num_shadow_models`: Number of shadow models.
- `shadow_train_size`, `shadow_test_size`: Sizes per shadow model.
- `shadow_epochs`, `shadow_lr`, `shadow_hidden`, `shadow_batch_size`.
- `attack_epochs`, `attack_lr`, `attack_hidden`, `attack_batch_size`.
- `feature_set`: `loss`, `stats`, `probs`, or `logits`.
- `max_samples_per_shadow`: Optional cap per shadow model.
- `seed`: Sampling seed.
- `data_overrides`: Optional overrides applied to the data builder for shadow models.

## `stage_mia`
- `enabled`: Boolean.
- `holdout_fraction`: Fraction of each client dataset held out for attack data.
- `num_flow_samples`: Monte Carlo samples for per-example flow loss.
- `include_ot_transport_norm`: Adds OT transport norm as a feature when OT is used.
- `attack_train_frac`: Fraction of member/nonmember features used to train the attack.
- `attack_hidden`, `attack_epochs`, `attack_lr`, `attack_batch_size`.
- `max_samples`: Optional cap for balanced member/nonmember samples.
- `seed`: Sampling seed.

## `stage_shadow_mia`
- `enabled`: Boolean.
- `num_shadow_models`: Shadow models per client.
- `holdout_fraction`, `num_flow_samples`, `include_ot_transport_norm`.
- `attack_train_frac`, `attack_hidden`, `attack_epochs`, `attack_lr`, `attack_batch_size`.
- `max_samples_per_shadow`: Optional cap per shadow model.
- `seed`: Sampling seed.
- `data_overrides`: Optional overrides applied to the data builder for shadow models.
