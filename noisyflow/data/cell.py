from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class _CellTable:
    x: np.ndarray
    labels: np.ndarray
    clients: np.ndarray
    conditions: np.ndarray


def _load_cell_table_from_npz(path: Path) -> _CellTable:
    data = np.load(path, allow_pickle=True)
    try:
        x = np.asarray(data["X"])
        labels = np.asarray(data["label"])
        clients = np.asarray(data["client"])
        conditions = np.asarray(data["condition"])
    except KeyError as exc:
        raise KeyError(
            f"Missing key {exc!s} in {path}. Expected keys: 'X', 'label', 'client', 'condition'."
        ) from exc
    if x.ndim != 2:
        raise ValueError(f"Expected X to have shape (N,d), got {x.shape}")
    n = int(x.shape[0])
    for name, arr in {"label": labels, "client": clients, "condition": conditions}.items():
        if arr.shape[0] != n:
            raise ValueError(f"Expected {name} to have shape (N,), got {arr.shape} with N={n}")
    return _CellTable(x=x, labels=labels, clients=clients, conditions=conditions)


def _load_cell_table_from_h5ad(path: Path, *, label_key: str, client_key: str, condition_key: str) -> _CellTable:
    try:
        import anndata
    except Exception as exc:
        raise RuntimeError(
            "Loading .h5ad requires anndata (and h5py). Install with `pip install anndata h5py` "
            "or convert the dataset to .npz and use that instead."
        ) from exc

    adata = anndata.read_h5ad(path)
    x = np.asarray(adata.X)
    if x.ndim != 2:
        raise ValueError(f"Expected adata.X to have shape (N,d), got {x.shape}")
    obs = adata.obs
    for key in (label_key, client_key, condition_key):
        if key not in obs.columns:
            raise KeyError(f"Missing obs['{key}'] in {path}. Available: {list(obs.columns)}")
    labels = np.asarray(obs[label_key])
    clients = np.asarray(obs[client_key])
    conditions = np.asarray(obs[condition_key])
    return _CellTable(x=x, labels=labels, clients=clients, conditions=conditions)


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    uniq = np.unique(labels)
    mapping: Dict[Any, int] = {v: i for i, v in enumerate(uniq.tolist())}
    encoded = np.vectorize(mapping.__getitem__, otypes=[np.int64])(labels)
    return encoded.astype(np.int64, copy=False), uniq.tolist()


def _as_size(value: Optional[Union[int, float]], *, total: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("size parameters must be int, float, or None")
    if isinstance(value, float):
        if not (0.0 < value < 1.0):
            raise ValueError("Fractional sizes must be in (0,1)")
        return max(1, int(round(total * float(value))))
    value = int(value)
    if value <= 0:
        raise ValueError("Size parameters must be > 0 when provided")
    return value


def _subsample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    n: Optional[int],
    *,
    stratify: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n is None or n >= int(indices.shape[0]):
        out = indices.copy()
        rng.shuffle(out)
        return out
    if stratify is None:
        out = indices.copy()
        rng.shuffle(out)
        return out[:n]

    stratify = np.asarray(stratify)
    if stratify.shape[0] != indices.shape[0]:
        raise ValueError("stratify must align with indices")

    chosen: List[int] = []
    classes = np.unique(stratify)
    per_class = max(1, int(n // max(1, classes.size)))
    for c in classes.tolist():
        idx_c = indices[stratify == c]
        if idx_c.size == 0:
            continue
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)
        chosen.extend(idx_c[: min(per_class, int(idx_c.size))].tolist())

    if len(chosen) < n:
        remaining = np.setdiff1d(indices, np.array(chosen, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, n - len(chosen))].tolist())

    out = np.array(chosen[:n], dtype=np.int64)
    rng.shuffle(out)
    return out


def _maybe_preprocess(
    x_train: np.ndarray,
    x_all: Sequence[np.ndarray],
    *,
    standardize: bool,
    pca_dim: Optional[int],
    seed: int,
) -> List[np.ndarray]:
    if not standardize and pca_dim is None:
        return [np.asarray(x, dtype=np.float32) for x in x_all]

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for standardize=True and/or pca_dim. "
            "Install scikit-learn or disable these options."
        ) from exc

    x_fit = np.asarray(x_train, dtype=np.float32)
    transforms: List[Any] = []

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(x_fit)
        transforms.append(scaler)
        x_fit = scaler.transform(x_fit).astype(np.float32, copy=False)

    if pca_dim is not None:
        pca_dim = int(pca_dim)
        if pca_dim <= 0:
            raise ValueError("pca_dim must be > 0")
        pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=int(seed))
        pca.fit(x_fit)
        transforms.append(pca)

    out: List[np.ndarray] = []
    for x in x_all:
        z = np.asarray(x, dtype=np.float32)
        for t in transforms:
            z = t.transform(z).astype(np.float32, copy=False)
        out.append(z)
    return out


def make_federated_cell_dataset(
    *,
    path: str,
    label_key: str = "cell_type",
    client_key: str = "sample_id",
    condition_key: str = "condition",
    source_condition: str = "ctrl",
    target_condition: str = "stim",
    split_mode: str = "ood",
    holdout_client: Optional[Union[str, int]] = None,
    target_ref_size: Optional[Union[int, float]] = None,
    target_test_size: Optional[Union[int, float]] = None,
    max_clients: Optional[int] = None,
    min_cells_per_client: int = 1,
    standardize: bool = False,
    pca_dim: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from a single-cell table.

    Supported input formats:
      - `.h5ad`: uses `anndata.read_h5ad` and pulls columns from `adata.obs`
      - `.npz`: expects arrays: `X` (N,d), `label` (N,), `client` (N,), `condition` (N,)

    Splits:
      - `split_mode='ood'` with `holdout_client`: target_test is target-condition cells from the holdout client.
      - otherwise: target cells are split iid into target_ref / target_test.
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path_obj.suffix.lower()
    if suffix == ".npz":
        table = _load_cell_table_from_npz(path_obj)
    elif suffix in {".h5ad", ".hdf5"}:
        table = _load_cell_table_from_h5ad(
            path_obj, label_key=label_key, client_key=client_key, condition_key=condition_key
        )
    else:
        raise ValueError(f"Unsupported dataset extension '{suffix}'. Use .h5ad or .npz.")

    x = np.asarray(table.x, dtype=np.float32)
    labels_raw = np.asarray(table.labels)
    clients_raw = np.asarray(table.clients)
    conditions_raw = np.asarray(table.conditions)

    encoded_labels, _ = _encode_labels(labels_raw)

    mask_source = conditions_raw == source_condition
    mask_target = conditions_raw == target_condition
    if not np.any(mask_source):
        raise ValueError(f"No rows found for source_condition='{source_condition}'")
    if not np.any(mask_target):
        raise ValueError(f"No rows found for target_condition='{target_condition}'")

    source_idx_all = np.flatnonzero(mask_source)
    target_idx_all = np.flatnonzero(mask_target)

    split_mode = str(split_mode).strip().lower()
    if split_mode not in {"ood", "iid"}:
        raise ValueError("split_mode must be 'ood' or 'iid'")

    ood = split_mode == "ood" and holdout_client is not None
    if ood:
        holdout_mask = clients_raw == holdout_client
        if not np.any(holdout_mask):
            raise ValueError(f"holdout_client={holdout_client!r} not found in client ids")
        source_idx = source_idx_all[~holdout_mask[source_idx_all]]
        target_ref_idx = target_idx_all[~holdout_mask[target_idx_all]]
        target_test_idx = target_idx_all[holdout_mask[target_idx_all]]
    else:
        source_idx = source_idx_all
        n_test = _as_size(target_test_size, total=int(target_idx_all.shape[0]))
        if n_test is None:
            n_test = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
        target_perm = target_idx_all.copy()
        rng.shuffle(target_perm)
        target_test_idx = target_perm[:n_test]
        target_ref_idx = target_perm[n_test:]

    target_ref_n = _as_size(target_ref_size, total=int(target_ref_idx.shape[0]))
    if target_ref_n is not None:
        target_ref_idx = _subsample_indices(
            rng,
            target_ref_idx,
            target_ref_n,
            stratify=encoded_labels[target_ref_idx],
        )
    target_test_n = _as_size(target_test_size, total=int(target_test_idx.shape[0]))
    if target_test_n is not None and (ood or split_mode == "iid"):
        target_test_idx = _subsample_indices(
            rng,
            target_test_idx,
            target_test_n,
            stratify=encoded_labels[target_test_idx],
        )

    if int(target_ref_idx.shape[0]) == 0 or int(target_test_idx.shape[0]) == 0:
        raise ValueError("Empty target_ref or target_test split; adjust split parameters.")

    client_ids = np.unique(clients_raw[source_idx])
    if max_clients is not None:
        max_clients = int(max_clients)
        if max_clients <= 0:
            raise ValueError("max_clients must be > 0")
        client_ids = client_ids[:max_clients]

    client_datasets: List[TensorDataset] = []
    client_x_blocks: List[np.ndarray] = []
    for cid in client_ids.tolist():
        idx = source_idx[clients_raw[source_idx] == cid]
        if int(idx.shape[0]) < int(min_cells_per_client):
            continue
        x_c = x[idx]
        y_c = encoded_labels[idx]
        client_x_blocks.append(x_c)
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; check client_key/min_cells_per_client/holdout_client.")

    x_target_ref = x[target_ref_idx]
    y_target_ref = encoded_labels[target_ref_idx]
    x_target_test = x[target_test_idx]
    y_target_test = encoded_labels[target_test_idx]

    # Optional global preprocessing (fit on source+target_ref, apply to all splits).
    if standardize or pca_dim is not None:
        x_fit = np.concatenate([np.concatenate(client_x_blocks, axis=0), x_target_ref], axis=0)
        client_x_blocks_proc, x_target_ref_proc, x_target_test_proc = _maybe_preprocess(
            x_fit,
            [np.concatenate(client_x_blocks, axis=0), x_target_ref, x_target_test],
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )
        x_all_clients_proc = client_x_blocks_proc

        # Re-split client block back into per-client arrays.
        offsets = np.cumsum([0] + [int(ds.tensors[0].shape[0]) for ds in client_datasets])
        new_client_datasets: List[TensorDataset] = []
        for i, ds in enumerate(client_datasets):
            a = int(offsets[i])
            b = int(offsets[i + 1])
            x_c = x_all_clients_proc[a:b]
            y_c = ds.tensors[1].numpy().astype(np.int64, copy=False)
            new_client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))
        client_datasets = new_client_datasets
        x_target_ref = x_target_ref_proc
        x_target_test = x_target_test_proc

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test


def make_cellot_lupuspatients_kang_hvg(
    *,
    path: str,
    holdout_client: Optional[Union[str, int]] = 101,
    seed: int = 0,
    **kwargs: Any,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Convenience wrapper matching the CellOT lupuspatients (Kang) config defaults.
    """
    return make_federated_cell_dataset(
        path=path,
        label_key=str(kwargs.pop("label_key", "cell_type")),
        client_key=str(kwargs.pop("client_key", "sample_id")),
        condition_key=str(kwargs.pop("condition_key", "condition")),
        source_condition=str(kwargs.pop("source_condition", "ctrl")),
        target_condition=str(kwargs.pop("target_condition", "stim")),
        split_mode=str(kwargs.pop("split_mode", "ood")),
        holdout_client=holdout_client,
        seed=seed,
        **kwargs,
    )

