"""Data loading, adjacency processing and metrics used by DASTGCN.

Expected dataset directory:
    data_path/train.npz
    data_path/val.npz
    data_path/test.npz

Each npz file should contain arrays named ``x`` and ``y`` with shape
[B, T, N, C].
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import torch


class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        if len(xs) != len(ys):
            raise ValueError("x and y must contain the same number of samples")
        if len(xs) == 0:
            raise ValueError("The dataset is empty")

        self.batch_size = int(batch_size)
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (self.batch_size - (len(xs) % self.batch_size)) % self.batch_size
            if num_padding > 0:
                x_padding = np.repeat(xs[-1:], num_padding, axis=0)
                y_padding = np.repeat(ys[-1:], num_padding, axis=0)
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = self.size // self.batch_size
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                yield self.xs[start_ind:end_ind], self.ys[start_ind:end_ind]
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_pickle(pickle_file: str) -> Any:
    try:
        with open(pickle_file, "rb") as file:
            return pickle.load(file)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as file:
            return pickle.load(file, encoding="latin1")


def sym_adj(adj):
    adj = np.asarray(adj, dtype=np.float32)
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    d_inv_sqrt[nonzero] = rowsum[nonzero] ** -0.5
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def asym_adj(adj):
    adj = np.asarray(adj, dtype=np.float32)
    rowsum = adj.sum(axis=1)
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    d_inv[nonzero] = rowsum[nonzero] ** -1.0
    return (d_inv[:, None] * adj).astype(np.float32)


def calculate_normalized_laplacian(adj):
    adj = np.asarray(adj, dtype=np.float32)
    return np.eye(adj.shape[0], dtype=np.float32) - sym_adj(adj)


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    adj_mx = np.asarray(adj_mx, dtype=np.float32)
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max = float(np.max(np.real(np.linalg.eigvals(lap))))
    lambda_max = max(float(lambda_max), 1e-8)
    identity = np.eye(lap.shape[0], dtype=np.float32)
    return ((2.0 / lambda_max) * lap - identity).astype(np.float32)


def _extract_adjacency(pickle_object: Any):
    """Accept plain adjacency or the common (ids, id_to_ind, adj) tuple."""
    sensor_ids = None
    sensor_id_to_ind = None
    adjacency = pickle_object

    if isinstance(pickle_object, (tuple, list)) and len(pickle_object) == 3:
        first, second, third = pickle_object
        third_array = np.asarray(third)
        if third_array.ndim == 2:
            sensor_ids, sensor_id_to_ind, adjacency = first, second, third

    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"Adjacency must be square [N,N], received {adjacency.shape}"
        )
    return sensor_ids, sensor_id_to_ind, adjacency


def load_adj(pkl_filename: str, adjtype: str):
    """Load adjacency and return a list of support matrices.

    Supported adjtype values follow the common Graph WaveNet convention.
    """
    sensor_ids, sensor_id_to_ind, adj_mx = _extract_adjacency(load_pickle(pkl_filename))

    if adjtype == "scalap":
        supports = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        supports = [calculate_normalized_laplacian(adj_mx)]
    elif adjtype == "symnadj":
        supports = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        supports = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        supports = [asym_adj(adj_mx), asym_adj(adj_mx.T)]
    elif adjtype == "identity":
        supports = [np.eye(adj_mx.shape[0], dtype=np.float32)]
    elif adjtype == "raw":
        supports = [adj_mx]
    else:
        raise ValueError(f"Unknown adjacency type: {adjtype}")

    supports = [np.asarray(support, dtype=np.float32) for support in supports]

    # Keep compatibility with train.py by returning only supports. The internal
    # loader still accepts the common three-object pickle format.
    return supports


def load_dataset(
    dataset_dir: str,
    batch_size: int,
    valid_batch_size: int | None = None,
    test_batch_size: int | None = None,
) -> Dict[str, Any]:
    valid_batch_size = valid_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size

    data: Dict[str, Any] = {}
    for category in ("train", "val", "test"):
        path = os.path.join(dataset_dir, f"{category}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        loaded = np.load(path)
        if "x" not in loaded or "y" not in loaded:
            raise KeyError(f"{path} must contain arrays named 'x' and 'y'")
        data[f"x_{category}"] = loaded["x"].astype(np.float32)
        data[f"y_{category}"] = loaded["y"].astype(np.float32)

    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(),
        std=data["x_train"][..., 0].std(),
    )

    # Normalize only the first traffic feature, matching common traffic datasets.
    for category in ("train", "val", "test"):
        data[f"x_{category}"][..., 0] = scaler.transform(
            data[f"x_{category}"][..., 0]
        )

    data["train_loader"] = DataLoader(
        data["x_train"], data["y_train"], batch_size
    )
    data["val_loader"] = DataLoader(
        data["x_val"], data["y_val"], valid_batch_size
    )
    data["test_loader"] = DataLoader(
        data["x_test"], data["y_test"], test_batch_size
    )
    data["scaler"] = scaler
    return data


def _build_mask(labels: torch.Tensor, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.to(labels.dtype)
    mask_mean = mask.mean()
    if mask_mean > 0:
        mask = mask / mask_mean
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    return mask


def masked_mse(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    loss = (preds - labels) ** 2
    loss = torch.nan_to_num(loss * mask, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    loss = torch.abs(preds - labels)
    loss = torch.nan_to_num(loss * mask, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


def masked_mape(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    denominator = torch.abs(labels).clamp_min(1e-5)
    loss = torch.abs(preds - labels) / denominator
    loss = torch.nan_to_num(loss * mask, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


def svar(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted variance of absolute residuals."""
    error = torch.abs(preds - labels).reshape(-1)
    if error.numel() == 0:
        return error.new_tensor(0.0)
    scale = error.max().clamp_min(1e-8)
    weights = torch.softmax(error / scale, dim=0)
    centered = error - error.mean()
    return torch.mean(weights * centered.square())


def metric(pred, real) -> Tuple[float, float, float, float]:
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    svar_value = svar(pred, real).item()
    return mae, mape, rmse, svar_value
