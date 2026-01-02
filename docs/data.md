# Data Builders

All data builders return:
- `client_datasets`: list of `TensorDataset(x, label)` for each client.
- `target_ref`: `TensorDataset(y, label)` with public labeled target reference data.
- `target_test`: `TensorDataset(y, label)` for evaluation.

`run.py` supports the following `data.type` values:
- `federated_mixture_gaussians`
- `mixture_gaussians` (alias of `federated_mixture_gaussians`)
- `toy_federated_gaussians`
- `federated_cell_dataset` (generic `.h5ad` / `.npz` single-cell loader)
- `cellot_lupuspatients_kang_hvg` (CellOT Kang lupuspatients convenience wrapper)

## `make_federated_mixture_gaussians`
Defined in `noisyflow/data/synthetic.py`.

Parameters:
- `K`: Number of clients.
- `n_per_client`: Samples per client.
- `n_target_ref`: Target reference samples.
- `n_target_test`: Target test samples.
- `d`: Feature dimension.
- `num_classes`: Number of classes.
- `component_scale`: Scale of class means.
- `component_cov`: Std of each Gaussian component.
- `class_probs`: Optional list of class probabilities.
- `scale_logstd`, `shift_scale`: Control random affine transforms.
- `seed`: RNG seed.

Notes:
- Each client applies a random affine transformation to the base mixture.
- The target domain uses a different affine transformation.

## `make_toy_federated_gaussians`
Defined in `noisyflow/data/toy.py`.

Parameters:
- `K`, `n_per_client`, `n_target_ref`, `n_target_test`, `d`, `num_classes`, `seed`.

Notes:
- Similar to `make_federated_mixture_gaussians` but with a simpler setup and fewer knobs.

## Example config
```yaml
data:
  type: federated_mixture_gaussians
  params:
    K: 3
    n_per_client: 1500
    n_target_ref: 2000
    n_target_test: 1000
    d: 2
    num_classes: 3
    component_scale: 3.0
    component_cov: 0.5
    seed: 0
```

## `make_federated_cell_dataset`
Defined in `noisyflow/data/cell.py`.

This loader supports CellOT-style data with a source and target condition (e.g., `ctrl`→`stim`)
and a client partition key (e.g., donor/patient id).

Input formats:
- `.h5ad`: reads `adata.X` and uses `adata.obs[label_key]`, `adata.obs[client_key]`, `adata.obs[condition_key]`
  (requires `anndata` + `h5py`)
- `.npz`: expects arrays: `X` (N,d), `label` (N,), `client` (N,), `condition` (N,)

Key parameters:
- `path`: dataset path
- `label_key`, `client_key`, `condition_key`: metadata keys (for `.h5ad`)
- `source_condition`, `target_condition`: condition values defining source/target
- `split_mode`: `ood` (use `holdout_client`) or `iid`
- `holdout_client`: client id reserved for target test split in `ood` mode
- `target_ref_size`, `target_test_size`: optionally subsample target splits (int or fraction)
- `max_clients`, `min_cells_per_client`: control the federated client list
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)
