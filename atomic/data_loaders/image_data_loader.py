from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from torchvision import datasets, transforms


@dataclass
class ImageSequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    d_in: int
    num_classes: int


def _flatten_to_sequence(x: np.ndarray, T: int) -> np.ndarray:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float32)
    d_flat = flat.shape[1]
    d_in = int(np.ceil(d_flat / float(T)))
    total = T * d_in
    if total != d_flat:
        pad = np.zeros((n, total - d_flat), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=1)
    return flat.reshape(n, T, d_in)


def _subsample(x: np.ndarray, y: np.ndarray, n: int, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError(f"Requested sample count must be positive, got {n}.")
    if n > x.shape[0]:
        raise ValueError(f"Requested {n} samples but only {x.shape[0]} are available.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx], y[idx]


def _mnist_arrays(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.ToTensor()
    tr = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    te = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    x_train = tr.data.numpy().astype(np.float32) / 255.0
    y_train = tr.targets.numpy().astype(np.int64)
    x_test = te.data.numpy().astype(np.float32) / 255.0
    y_test = te.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def _cifar_arrays(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.ToTensor()
    tr = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    te = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    x_train = tr.data.astype(np.float32) / 255.0
    y_train = np.asarray(tr.targets, dtype=np.int64)
    x_test = te.data.astype(np.float32) / 255.0
    y_test = np.asarray(te.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


def load_mnist_seq_dataset(
    *,
    task: str,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> ImageSequenceDataset:
    if task not in ("mnist_seq", "mnist_perm_seq"):
        raise ValueError(f"Unsupported MNIST task={task}. Expected mnist_seq|mnist_perm_seq.")
    x_train_all, y_train_all, x_test_all, y_test_all = _mnist_arrays(seed=seed)
    x_train, y_train = _subsample(x_train_all, y_train_all, n_train, seed)
    x_val, y_val = _subsample(x_train_all, y_train_all, n_val, seed + 1)
    x_test, y_test = _subsample(x_test_all, y_test_all, n_test, seed + 2)

    x_train_seq = _flatten_to_sequence(x_train, T=T)
    x_val_seq = _flatten_to_sequence(x_val, T=T)
    x_test_seq = _flatten_to_sequence(x_test, T=T)
    if task == "mnist_perm_seq":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(x_train_seq.shape[2])
        x_train_seq = x_train_seq[:, :, perm]
        x_val_seq = x_val_seq[:, :, perm]
        x_test_seq = x_test_seq[:, :, perm]
    return ImageSequenceDataset(
        X_train=x_train_seq,
        y_train=y_train,
        X_val=x_val_seq,
        y_val=y_val,
        X_test=x_test_seq,
        y_test=y_test,
        d_in=x_train_seq.shape[2],
        num_classes=10,
    )


def load_cifar_seq_dataset(
    *,
    task: str,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> ImageSequenceDataset:
    if task != "cifar_seq":
        raise ValueError(f"Unsupported CIFAR task={task}. Expected cifar_seq.")
    x_train_all, y_train_all, x_test_all, y_test_all = _cifar_arrays(seed=seed)
    x_train, y_train = _subsample(x_train_all, y_train_all, n_train, seed)
    x_val, y_val = _subsample(x_train_all, y_train_all, n_val, seed + 1)
    x_test, y_test = _subsample(x_test_all, y_test_all, n_test, seed + 2)
    x_train_seq = _flatten_to_sequence(x_train, T=T)
    x_val_seq = _flatten_to_sequence(x_val, T=T)
    x_test_seq = _flatten_to_sequence(x_test, T=T)
    return ImageSequenceDataset(
        X_train=x_train_seq,
        y_train=y_train,
        X_val=x_val_seq,
        y_val=y_val,
        X_test=x_test_seq,
        y_test=y_test,
        d_in=x_train_seq.shape[2],
        num_classes=10,
    )

__all__ = ["ImageSequenceDataset", "load_mnist_seq_dataset", "load_cifar_seq_dataset"]
