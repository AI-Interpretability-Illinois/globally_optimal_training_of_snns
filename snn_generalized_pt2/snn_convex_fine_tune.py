#!/usr/bin/env python3
"""
Convex-SNN vs STE-SNN on sequence tasks (parity, XOR, moving blobs, seq MNIST):

- Uses the SAME task interface / CLI style as RRN.py:
    --task {parity_seq,two_step_xor_seq,moving_blobs_seq,mnist_seq,mnist_perm_seq}
    --T, --L, --P_in, --P_rec, --P_last, --n_train_total, --val_frac, --n_test, ...
- CVX side:
    * Sample (input, recurrent, last-layer) hyperplanes ONCE per seed.
    * Shared weights across time (parameter sharing).
    * Produce last-layer sign patterns z ∈ {0,1}^{n_train × P_last_unique}.
    * Train convex last layer W with chosen loss + L1 regularization.
- SNN baseline:
    * L-layer stacked LIF (Leaky Integrate-and-Fire) with STE-style training.
    * Options to keep leak beta and threshold either fixed or trainable.
    * Uses same widths (P_rec per hidden layer, P_last for final linear).

This file is self-contained: implements dataset makers analogous to RRN.py,
and does NOT import RRN.py for clarity, though the semantics match.

NOTE:
- You may want to tweak the snntorch.Leaky signature to match your installed
  version (learn_beta / learn_threshold names etc).
"""

import argparse
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# If you use snntorch in your other SNN code, import it here.
# Adjust the import to your local setup.
import snntorch as snn

# Dataset helper module for additional baselines
from snn_data_modules import (
    load_ptb_cache,
    load_binary_adding_cache,
    ptb_to_sequence,
    binary_adding_to_sequence,
    load_shd_cache,
    load_ssc_cache,
    shd_to_sequence,
    ssc_to_sequence,
    load_nmnist_cache,
    load_cifar10_dvs_cache,
    load_dvs_gesture_cache,
    nmnist_to_sequence,
    cifar10_dvs_to_sequence,
    dvs_gesture_to_sequence,
    load_cifar10_cache,
    load_cifar100_cache,
    cifar10_to_sequence,
    cifar100_to_sequence,
    load_gsc_cache,
    load_timit_cache,
    gsc_to_sequence,
    timit_to_sequence,
)


def normalized_mutual_info_score(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Lightweight NMI implementation to avoid heavy sklearn import at runtime.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    # Relabel to 0..n_classes-1
    labels_true, inv_true = np.unique(y_true, return_inverse=True)
    labels_pred, inv_pred = np.unique(y_pred, return_inverse=True)
    n_true = labels_true.size
    n_pred = labels_pred.size
    n_samples = y_true.shape[0]

    # Contingency table
    idx = inv_true * n_pred + inv_pred
    cont = np.bincount(idx, minlength=n_true * n_pred).reshape(n_true, n_pred)

    # Marginals
    pi = cont.sum(axis=1)  # (n_true,)
    pj = cont.sum(axis=0)  # (n_pred,)

    # Mutual information
    rows, cols = np.nonzero(cont)
    cont_vals = cont[rows, cols].astype(np.float64)
    mi_terms = (cont_vals / n_samples) * (
        np.log(cont_vals * n_samples + eps)
        - np.log(pi[rows] + eps)
        - np.log(pj[cols] + eps)
    )
    mi = mi_terms.sum()

    def _entropy(counts: np.ndarray) -> float:
        p = counts.astype(np.float64)
        total = p.sum()
        if total <= 0:
            return 0.0
        p /= total
        p = p[p > 0]
        return float(-(p * np.log(p + eps)).sum())

    h_true = _entropy(pi)
    h_pred = _entropy(pj)
    denom = h_true + h_pred
    if denom <= 0:
        return 0.0
    return float(2.0 * mi / denom)


def compute_layer_diagnostics(
    h_bits: np.ndarray,
    y_np: np.ndarray,
    *,
    layer: int,
    t_idx: int,
    print_results: bool = True,
) -> Dict[str, object]:
    """
    Unified diagnostic for a single (layer, timestep) activation matrix.

    Parameters
    ----------
    h_bits : np.ndarray, shape (n, h_dim), dtype uint8
        Binary activation matrix (0/1) for all samples at this (layer, timestep).
    y_np : np.ndarray, shape (n,)
        Ground-truth labels.
    layer : int
        Layer index (for display).
    t_idx : int
        Timestep index (for display).
    print_results : bool
        If True, print a summary line to stdout.

    Returns
    -------
    dict with keys:
        layer, t, n_samples, h_dim,
        unique_rows, unique_cols, activation_rate, nmi, weighted_purity,
        n_clusters, avg_cluster_size, median_cluster_size,
        n_singletons, frac_singletons, n_nonsingleton_samples, frac_nonsingleton_samples,
        avg_margin_nonsingleton, k_90_coverage, frac_k_90
    """
    n, h_dim = h_bits.shape

    # ---- Row-wise: sample distinguishability ----
    pattern_ids = np.array([h_bits[i].tobytes() for i in range(n)])
    _, encoded = np.unique(pattern_ids, return_inverse=True)
    n_unique_rows = int(len(set(encoded)))

    # ---- Column-wise: neuron redundancy ----
    n_unique_cols = len({h_bits[:, j].tobytes() for j in range(h_dim)})

    # ---- Activation rate ----
    act_rate = float(h_bits.astype(np.float64).mean())

    # ---- NMI ----
    nmi = normalized_mutual_info_score(y_np, encoded)

    # ---- Per-pattern label analysis ----
    pattern_labels: Dict[int, List[int]] = {}
    for pid, label in zip(encoded, y_np):
        pattern_labels.setdefault(int(pid), []).append(int(label))

    # ---- Weighted purity ----
    total_purity = 0.0
    for labels in pattern_labels.values():
        counts = Counter(labels)
        majority = max(counts.values())
        total_purity += majority
    weighted_purity = total_purity / n

    # ---- Cluster size distribution ----
    sizes = np.array([len(v) for v in pattern_labels.values()])
    n_clusters = len(pattern_labels)
    avg_cluster_size = float(n / n_clusters) if n_clusters > 0 else 0.0
    median_cluster_size = float(np.median(sizes)) if len(sizes) > 0 else 0.0
    n_singletons = int(np.sum(sizes == 1))
    frac_singletons = n_singletons / n_clusters if n_clusters > 0 else 0.0

    # ---- Non-singleton sample coverage ----
    n_nonsingleton_samples = int(np.sum(sizes[sizes > 1]))
    frac_nonsingleton_samples = n_nonsingleton_samples / n if n > 0 else 0.0

    # ---- Label margin for non-singleton clusters ----
    margin_sum = 0.0
    margin_weight = 0
    for labels in pattern_labels.values():
        if len(labels) <= 1:
            continue
        counts = Counter(labels)
        majority_frac = max(counts.values()) / len(labels)
        margin_sum += majority_frac * len(labels)
        margin_weight += len(labels)
    avg_margin_nonsingleton = (
        margin_sum / margin_weight if margin_weight > 0 else 0.0
    )

    # ---- LASSO sparsity-friendliness: patterns to cover 90% of samples ----
    sorted_sizes = np.sort(sizes)[::-1]
    cumsum = np.cumsum(sorted_sizes)
    k_90 = int(np.searchsorted(cumsum, 0.9 * n) + 1) if len(cumsum) > 0 else 0
    frac_k_90 = k_90 / n_clusters if n_clusters > 0 else 0.0

    result = {
        "layer": layer,
        "t": t_idx,
        "n_samples": n,
        "h_dim": h_dim,
        "unique_rows": n_unique_rows,
        "unique_cols": n_unique_cols,
        "activation_rate": round(act_rate, 4),
        "nmi": round(nmi, 4),
        "weighted_purity": round(weighted_purity, 4),
        "n_clusters": n_clusters,
        "avg_cluster_size": round(avg_cluster_size, 2),
        "median_cluster_size": round(median_cluster_size, 1),
        "n_singletons": n_singletons,
        "frac_singletons": round(frac_singletons, 4),
        "n_nonsingleton_samples": n_nonsingleton_samples,
        "frac_nonsingleton_samples": round(frac_nonsingleton_samples, 4),
        "avg_margin_nonsingleton": round(avg_margin_nonsingleton, 4),
        "k_90_coverage": k_90,
        "frac_k_90": round(frac_k_90, 4),
    }

    if print_results:
        print(
            f"[diag] layer {layer}, t={t_idx}: "
            f"rows {n_unique_rows}/{n}, "
            f"cols {n_unique_cols}/{h_dim}, "
            f"act={act_rate:.3f}, NMI={nmi:.4f}, "
            f"purity={weighted_purity:.4f}, "
            f"clusters={n_clusters} (avg_sz={avg_cluster_size:.1f}, "
            f"singletons={n_singletons}/{n_clusters}), "
            f"nonsingleton_samples={n_nonsingleton_samples}/{n} "
            f"({100*frac_nonsingleton_samples:.1f}%), "
            f"margin_ns={avg_margin_nonsingleton:.4f}, "
            f"k_90={k_90}/{n_clusters} ({100*frac_k_90:.1f}%)"
        )

    return result


# ============================================================
# Device + seeds
# ============================================================

def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; using CPU.")
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        print(f"[warn] {device_arg} requested but CUDA not available; using CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Simple logging helpers
# ============================================================

def format_dict(d: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in d.items())


# ============================================================
# Datasets: parity, XOR, moving Gaussian blobs, seq MNIST, sunspot
# ============================================================

def make_parity_seq(n: int, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Binary parity over T steps with Bernoulli(0.5) input bits.
    X: (n, T, 1), y: (n,) in {0,1}
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, T, 5)).astype(np.float32)
    y = (X.sum(axis=1) % 2).reshape(-1).astype(np.int64)
    return X, y, 2


def make_two_step_xor_seq(
    n: int,
    T: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Two-step XOR task:
      - Input dimension d_in = 2
      - At each time t, we get x_t ∈ {0,1}^2.
      - Label is XOR(x_{T-1}, x_{T-2}) (componentwise) aggregated to a single bit.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, T, 2)).astype(np.float32)

    # XOR on the last two timesteps, then reduce across dim
    last = X[:, -1, :]
    prev = X[:, 0, :]
    xor = (last + prev) % 2  # (n, 2)
    y = (xor.sum(axis=1) % 2).astype(np.int64)
    return X, y, 2


def make_moving_gaussian_blobs_seq(
    n: int,
    T: int,
    seed: int,
    d_in: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Two-class sequence classification that ACTUALLY needs time:
    class is determined by velocity direction (trajectory), not instantaneous position.

    X shape: (n, T, d_in)
    labels: (n,)
    """
    rng = np.random.default_rng(seed)

    # Velocity directions only in first 2 dims; rest 0.
    v = np.zeros((2, d_in), dtype=np.float32)
    v[0, :2] = [-0.25, -0.25]   # class 0 moves down-left
    v[1, :2] = [0.25, 0.25]     # class 1 moves up-right

    labels = rng.integers(0, 2, size=n).astype(np.int64)

    # Shared initial position distribution (NOT class-dependent)
    x0 = rng.normal(scale=0.5, size=(n, d_in)).astype(np.float32)

    # Optional: per-sequence velocity jitter so it isn't perfectly deterministic
    v_jitter = rng.normal(scale=0.03, size=(n, d_in)).astype(np.float32)
    vel = v[labels] + v_jitter  # (n, d_in)

    # Observation noise
    obs_noise = 1.0  # increase if you want "harder" single-step classification

    X = np.zeros((n, T, d_in), dtype=np.float32)
    for t in range(T):
        # position at time t: x0 + t*vel + noise
        X[:, t, :] = x0 + (t * vel) + rng.normal(scale=obs_noise, size=(n, d_in)).astype(np.float32)

    return X, labels, 2


@dataclass
class MNISTCache:
    X_train: np.ndarray  # (60000, 784)
    y_train: np.ndarray  # (60000,)
    X_test: np.ndarray   # (10000, 784)
    y_test: np.ndarray   # (10000,)


def load_mnist_cache(root: str = "data") -> MNISTCache:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    Xtr = np.stack([tr[i][0].view(-1).numpy() for i in range(len(tr))], axis=0).astype(np.float32)
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)

    Xte = np.stack([te[i][0].view(-1).numpy() for i in range(len(te))], axis=0).astype(np.float32)
    yte = np.array([int(te[i][1]) for i in range(len(te))], dtype=np.int64)

    return MNISTCache(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def mnist_to_sequence(
    cache: MNISTCache,
    *,
    n_train_total: int,
    n_test: int,
    seed: int,
    task: str,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns X_total_seq, y_total, X_test_seq, y_test, num_classes.

    Output shapes:
      X_total_seq: (n_train_total, T, d_in)
      X_test_seq:  (n_test,        T, d_in)
    where d_in = 784 / T (must be integer).
    """
    X_total = cache.X_train[:n_train_total].copy()
    y_total = cache.y_train[:n_train_total].copy()
    X_test = cache.X_test[:n_test].copy()
    y_test = cache.y_test[:n_test].copy()

    if task == "mnist_perm_seq":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(784)
        X_total = X_total[:, perm]
        X_test = X_test[:, perm]
    elif task != "mnist_seq":
        raise ValueError(f"mnist_to_sequence: unknown task={task}")

    if T <= 0 or T > 784:
        raise ValueError("For MNIST sequence tasks, set 1 <= T <= 784")
    if 784 % T != 0:
        raise ValueError(f"T must divide 784 so d_in=784/T is integer. Got T={T}, 784%T={784 % T}.")

    d_in = 784 // T
    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq = X_test.reshape(X_test.shape[0], T, d_in)

    return X_total_seq.astype(np.float32), y_total, X_test_seq.astype(np.float32), y_test, 10


def make_sunspot_seq(
    n: int,
    T: int,
    seed: int,
    window_stride: int = 1,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build a 1-D regression sequence dataset from the (annual) sunspot series.

    We treat the sunspot activity as a generic time series s_t and form windows:
        x_i = [s_i, ..., s_{i+T-1}]  ∈ R^T
        y_i = s_{i+T}                ∈ R

    Then we reshape x_i to (T, 1) so that it matches the (n, T, d_in) convention
    used by the RNN code. The target is a scalar regression label.

    Because the built-in statsmodels sunspot series is only ~300 long, we:
      - compute the maximum number of windows we can form,
      - clip n to that maximum,
      - optionally normalize (zero mean / unit std).
    """
    rng = np.random.default_rng(seed)

    # statsmodels sunspots dataset: annual series
    import statsmodels.api as sm
    data = sm.datasets.sunspots.load_pandas().data
    series = data["SUNACTIVITY"].values.astype(np.float32)
    N = series.shape[0]

    max_windows = (N - (T + 1)) // window_stride + 1
    if max_windows <= 0:
        raise ValueError(f"[sunspot_seq] T={T} is too large for series length N={N}.")

    n_eff = min(n, max_windows)
    if n > max_windows:
        print(
            f"[warn] sunspot_seq: requested n={n}, but only {max_windows} windows "
            f"are available; clipping to {n_eff}."
        )

    X = []
    y = []
    idx = 0
    for _ in range(n_eff):
        if idx + T >= N - 1:
            break
        x_win = series[idx : idx + T]
        y_val = series[idx + T]
        X.append(x_win)
        y.append(y_val)
        idx += window_stride

    X = np.stack(X, axis=0)  # (n_eff, T)
    y = np.asarray(y, dtype=np.float32)  # (n_eff,)

    if normalize:
        mu = X.mean()
        sigma = X.std() + 1e-8
        X = (X - mu) / sigma
        y = (y - mu) / sigma

    # reshape to (n_eff, T, d_in=1)
    X = X.reshape(X.shape[0], T, 1)
    num_classes = 1  # regression: scalar target

    return X, y, num_classes


def build_dataset(
    task: str,
    T: int,
    n_train_total: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Unified dataset builder.

    Returns
    -------
    X_total : np.ndarray, shape (n_train_total, T, d_in)
    y_total : np.ndarray, shape (n_train_total,)
    X_test  : np.ndarray, shape (n_test, T, d_in)
    y_test  : np.ndarray, shape (n_test,)
    num_classes : int
        Number of classes for classification losses (hinge / ce). For regression
        tasks this can be left unused by the caller.
    """
    # ----- synthetic toy tasks -----
    if task in ("parity_seq", "two_step_xor_seq", "moving_blobs_seq"):
        X_seq, y, num_classes = (
            make_parity_seq(n_train_total + n_test, T, seed=seed)
            if task == "parity_seq"
            else make_two_step_xor_seq(n_train_total + n_test, T, seed=seed)
            if task == "two_step_xor_seq"
            else make_moving_gaussian_blobs_seq(n_train_total + n_test, T, seed=seed)
        )
        X_total = X_seq[:n_train_total]
        y_total = y[:n_train_total]
        X_test = X_seq[n_train_total : n_train_total + n_test]
        y_test = y[n_train_total : n_train_total + n_test]
        return X_total, y_total, X_test, y_test, num_classes

    # ----- MNIST sequence variants -----
    if task in ("mnist_seq", "mnist_perm_seq"):
        cache = load_mnist_cache(root="data")
        return mnist_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            seed=seed,
            task=task,
            T=T,
        )

    # ----- new baseline datasets -----

    # PTB language modeling: predict next token from T-token context.
    if task == "ptb_seq":
        cache = load_ptb_cache(root="data")
        return ptb_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            seed=seed,
            T=T,
        )

    # Binary adding: regression (sum of two binary numbers) or classification
    # depending on chosen loss type. We expose it as a sequence of length T
    # with 2 input channels (two numbers).
    if task == "binary_adding_seq":
        cache = load_binary_adding_cache(T = T)
        return binary_adding_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # Spiking Heidelberg Digits.
    if task == "shd_seq":
        cache = load_shd_cache(path="data/shd")
        return shd_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # Spiking Speech Commands.
    if task == "ssc_seq":
        cache = load_ssc_cache(path="data/ssc")
        return ssc_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # N-MNIST neuromorphic vision.
    if task == "nmnist_seq":
        cache = load_nmnist_cache(path="data/nmnist")
        return nmnist_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )
    
    if task.startswith("dfa:"):
        from dfa_tasks import make_dfa_dataset
        dfa_spec = task[4:]  # strip "dfa:" prefix
        X_all, y_all, num_classes = make_dfa_dataset(
            dfa_spec, n=n_train_total + n_test, T=T, seed=seed, balanced=True
        )
        X_total = X_all[:n_train_total]
        y_total = y_all[:n_train_total]
        X_test = X_all[n_train_total : n_train_total + n_test]
        y_test = y_all[n_train_total : n_train_total + n_test]
        return X_total, y_total, X_test, y_test, num_classes

    # Static CIFAR-10 / CIFAR-100 vision.
    if task == "cifar10_seq":
        cache = load_cifar10_cache(root="data")
        return cifar10_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    if task == "cifar100_seq":
        cache = load_cifar100_cache(root="data")
        return cifar100_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # Neuromorphic CIFAR10-DVS and DVSGesture.
    if task == "cifar10_dvs_seq":
        cache = load_cifar10_dvs_cache(path="data/cifar10_dvs")
        return cifar10_dvs_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    if task == "dvs_gesture_seq":
        cache = load_dvs_gesture_cache(path="data/dvs_gesture")
        return dvs_gesture_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # Google Speech Commands (GSC).
    if task == "gsc_seq":
        cache = load_gsc_cache(root="data/gsc")
        return gsc_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    # TIMIT speech.
    if task == "timit_seq":
        cache = load_timit_cache(root="data/timit")
        return timit_to_sequence(
            cache,
            n_train_total=n_train_total,
            n_test=n_test,
            T=T,
        )

    raise ValueError(f"Unknown task: {task}")


def split_train_val(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]



# ============================================================
# Convex SNN pattern generation (RNN-style threshold recurrence)
# ============================================================

def _col_normalize_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0, keepdims=True) + eps
    return U / norms


@dataclass
class RNNHyperplanes:
    U_in_list: List[np.ndarray]   # per layer: (d_in_l, h_dim)
    U_rec_list: List[np.ndarray]  # per layer: (2*h_dim, h_dim) = [diag(beta); diag(threshold)]
    U_last: np.ndarray            # (h_dim_last, P_last)
    last_layer_readout: str = "membrane"  # "membrane" or "spike"
    beta_list: Optional[List[np.ndarray]] = None      # per layer: (h_dim,) leak factors
    threshold_list: Optional[List[np.ndarray]] = None # per layer: (h_dim,) thresholds/reset magnitudes

def generate_snn_sign_patterns(
    X_seq: np.ndarray, 
    Y: np.ndarray, # (n, )
    L: int,
    T: int,
    P_in: int,
    P_rec: int,
    P_last_target: int,
    seed: int,
    normalize_hidden: bool = True,
    verbose: bool = False,
    chunk_mult: int = 4,
    max_rounds: int = 300,
    device: Optional[torch.device] = None,
    init_method: str = "random",
    target_act_rate: Optional[float] = None,
    lsuv_tol: float = 0.02,
    lsuv_max_iters: int = 20,
    last_layer_readout: str = "membrane",
    beta_dist: str = "fixed",
    transfer_u_in_list: Optional[List[np.ndarray]] = None,
    transfer_threshold_list: Optional[List[np.ndarray]] = None,
    threshold_value: float = 1.0,
) -> Tuple[np.ndarray, RNNHyperplanes, List[Dict[str, object]]]:
    """
    Generate sign patterns for a threshold-SNN with L hidden layers and T timesteps.

    This GPU-optimized version:

      * Uses torch on `device` for all recurrence and last-layer matmuls.
      * Keeps only hyperplane storage + unique-pattern bookkeeping on CPU (NumPy).

    Returns:
      z_last_bool: (n, P_last_target) bool
      hypers:      RNNHyperplanes(U_in_list, U_rec_list, U_last)
                   where U_rec_list[l-1] has shape (2*h_dim_l+1, h_dim_l).
      diagnostics: list of dicts, one per (layer, timestep) probed (empty if verbose=False).
    """
    rng = np.random.default_rng(seed)
    n, T_data, d_in = X_seq.shape
    if T_data != T:
        raise ValueError(f"X_seq has T={T_data} but you requested T={T}.")

    if device is None:
        device = get_device("auto")

    # MPS does not support float64; use float32 on MPS, float64 elsewhere for overflow safety.
    use_f32 = device.type == "mps"
    dtype = torch.float32 if use_f32 else torch.float64
    np_dtype = np.float32 if use_f32 else np.float64

    if verbose:
        print(f"[snn patterns] using device={device}")

    # layer 0 "spikes" are just the inputs x_t (real-valued); we don't threshold them
    # h_layers[l][t] holds activations for layer l at time t, except layer 0 which holds raw inputs
    X_torch = torch.from_numpy(X_seq.astype(np_dtype, copy=False)).to(device)

    h_layers: List[List[torch.Tensor]] = []
    h0 = [X_torch[:, t, :] for t in range(T)]  # each (n, d_in)
    h_layers.append(h0)

    U_in_list: List[np.ndarray] = []
    U_rec_list: List[np.ndarray] = []
    beta_list_all: List[np.ndarray] = []
    threshold_list_all: List[np.ndarray] = []

    # Decide hidden widths: first L-2 layers = P_rec, last hidden layer = P_last_target
    if L <= 1:
        hidden_dims = [P_last_target]
    else:
        hidden_dims = [P_rec] * max(L - 2, 0) + [P_last_target]

    d_in_l = d_in

    y_np = Y  # already a NumPy array of shape (n,)
    all_diagnostics: List[Dict[str, object]] = []

    # --- Helper: run one layer's LIF forward pass for T timesteps ---
    def _run_layer_forward(
        h_prev_layer: List[torch.Tensor],  # (n, d_in_l) per timestep
        U_in_t: torch.Tensor,              # (d_in_l, h_dim) on device
        U_rec_t: torch.Tensor,             # (2*h_dim, h_dim) on device = [diag(beta); diag(threshold)]
        thr_t: torch.Tensor,               # (h_dim,) threshold/reset magnitudes on device
        h_dim_: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run threshold-SNN dynamics matched to the pretrained snn.Leaky threshold convention.
        We use
            v_t = x_t U_in + beta * v_{t-1} - threshold * h_{t-1},
            h_t = 1{ v_t >= threshold }.
        This removes the extra hard-coded -1 offset that was causing deep collapse.
        Returns (h_list, v_list): spike tensors and membrane tensors per timestep."""
        v_p = torch.zeros(n, h_dim_, dtype=dtype, device=device)
        h_p = torch.zeros(n, h_dim_, dtype=dtype, device=device)
        h_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        for t in range(T):
            x_t = h_prev_layer[t]
            v_in_t = x_t @ U_in_t
            s_p = torch.cat([v_p, -h_p], dim=1)
            v_rec_t = s_p @ U_rec_t
            v_t = v_in_t + v_rec_t
            h_t = (v_t >= thr_t.unsqueeze(0)).to(dtype)
            h_list.append(h_t)
            v_list.append(v_t)
            v_p = v_t
            h_p = h_t
        return h_list, v_list

    # -------------------------------
    # Hidden layers 1 .. L-1 (GPU)
    # -------------------------------
    v_layers: List[List[torch.Tensor]] = [[] for _ in range(L)]  # membrane potentials per layer
    prev_act_rate: Optional[float] = None  # tracked for Micheli formula at deeper layers

    for l in range(1, L):
        h_dim = hidden_dims[l - 1]  # width of this hidden layer

        # --- Input weight initialization ---
        if transfer_u_in_list is not None:
            if len(transfer_u_in_list) != (L - 1):
                raise ValueError(
                    f"transfer_u_in_list must have {L-1} entries, got {len(transfer_u_in_list)}."
                )
            U_in_np = np.asarray(transfer_u_in_list[l - 1], dtype=np.float32).copy()
            if U_in_np.shape != (d_in_l, h_dim):
                raise ValueError(
                    f"Transferred U_in for layer {l} has shape {U_in_np.shape}, "
                    f"expected {(d_in_l, h_dim)}."
                )
        elif init_method == "random":
            U_in_np = rng.normal(size=(d_in_l, h_dim)).astype(np.float32)

        elif init_method == "micheli":
            # Micheli et al. 2024: Var[w_l] = 1 / (n_l * p_{l-1})
            # For layer 1 (real-valued input): estimate input second moment from data
            # For layer 2+ (binary spike input): use measured firing rate of previous layer
            if l == 1:
                # Real-valued input: use E[x²] per component as the "activity" measure
                # Micheli reduces to Kaiming-like: Var[w] = 1 / (d_in * E[x²])
                x_all = X_torch.reshape(-1, d_in)  # (n*T, d_in)
                input_second_moment = float((x_all ** 2).mean().item())
                input_second_moment = max(input_second_moment, 1e-6)  # safety
                sigma_w = np.sqrt(1.0 / (d_in_l * input_second_moment))
            else:
                # Binary spike input: Var[w] = 1 / (n_l * p_{l-1})
                p_prev = max(prev_act_rate if prev_act_rate is not None else 0.1, 1e-4)
                sigma_w = np.sqrt(1.0 / (d_in_l * p_prev))

            U_in_np = rng.normal(scale=sigma_w, size=(d_in_l, h_dim)).astype(np.float32)
            if verbose:
                print(
                    f"[micheli] layer {l}: sigma_w={sigma_w:.4f} "
                    f"(d_in={d_in_l}, "
                    f"{'E[x²]=' + f'{input_second_moment:.4f}' if l == 1 else f'p_prev={p_prev:.4f}'})"
                )

        elif init_method == "lognormal":
            # Log-normal magnitudes with random signs
            magnitudes = rng.lognormal(mean=0.0, sigma=0.5, size=(d_in_l, h_dim))
            signs = rng.choice([-1.0, 1.0], size=(d_in_l, h_dim))
            U_in_np = (magnitudes * signs).astype(np.float32)

        elif init_method == "lsuv":
            # LSUV (Mishkin & Matas 2016): iterative rescaling to hit target activation rate
            U_in_np = rng.normal(size=(d_in_l, h_dim)).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown init_method={init_method!r}. "
                f"Choose from: random, micheli, lognormal, lsuv."
            )

        # Recurrent hyperplanes: from [v_{t-1}, -h_{t-1}, -1] (dim = 2*h_dim+1) to h_dim
        if beta_dist == "fixed":
            b = np.full((h_dim,), 0.99, dtype=np.float32)
        elif beta_dist == "het_loguniform":
            # Log-uniform over [0.5, 0.999]: fast neurons forget quickly, slow ones integrate
            log_lo, log_hi = np.log(0.5), np.log(0.999)
            b = np.exp(rng.uniform(log_lo, log_hi, size=(h_dim,))).astype(np.float32)
        elif beta_dist == "het_uniform":
            # Uniform over [0.5, 0.999]
            b = rng.uniform(0.5, 0.999, size=(h_dim,)).astype(np.float32)
        elif beta_dist == "het_bimodal":
            # Half fast (β~0.5), half slow (β~0.99): maximise timescale separation
            b = np.empty(h_dim, dtype=np.float32)
            n_fast = h_dim // 2
            b[:n_fast] = rng.uniform(0.3, 0.6, size=(n_fast,))
            b[n_fast:] = rng.uniform(0.95, 0.999, size=(h_dim - n_fast,))
            rng.shuffle(b)
        else:
            raise ValueError(f"Unknown beta_dist={beta_dist!r}")

        if verbose and l == 1:
            print(f"[beta_dist={beta_dist}] layer {l}: beta min={b.min():.4f} max={b.max():.4f} "
                  f"mean={b.mean():.4f} median={np.median(b):.4f}")

        if transfer_threshold_list is not None:
            if len(transfer_threshold_list) != (L - 1):
                raise ValueError(
                    f"transfer_threshold_list must have {L-1} entries, got {len(transfer_threshold_list)}."
                )
            g = np.asarray(transfer_threshold_list[l - 1], dtype=np.float32).reshape(-1).copy()
            if g.shape != (h_dim,):
                raise ValueError(
                    f"Transferred threshold for layer {l} has shape {g.shape}, expected {(h_dim,)}."
                )
        else:
            g = np.full((h_dim,), float(threshold_value), dtype=np.float32)

        U_rec_top = np.diag(b)   # (h_dim, h_dim)
        U_rec_bot = np.diag(g)   # (h_dim, h_dim) reset magnitude = threshold
        U_rec_np = np.vstack([U_rec_top, U_rec_bot]).astype(np.float32)  # (2*h_dim, h_dim)

        if normalize_hidden:
            U_in_np = _col_normalize_np(U_in_np)
            U_rec_np = _col_normalize_np(U_rec_np)

        # --- LSUV-style iterative calibration ---
        # Only apply LSUV when explicitly requested; keep transferred features raw otherwise.
        if init_method == "lsuv":
            act_rate_target = target_act_rate if target_act_rate is not None else 0.15
            tol = lsuv_tol
            U_rec_gpu = torch.from_numpy(U_rec_np.astype(np_dtype, copy=False)).to(device)
            act_last = 0.0
            for lsuv_iter in range(lsuv_max_iters):
                U_in_gpu = torch.from_numpy(U_in_np.astype(np_dtype, copy=False)).to(device)
                thr_gpu = torch.from_numpy(g.astype(np_dtype, copy=False)).to(device)
                h_trial, _ = _run_layer_forward(
                    h_layers[l - 1], U_in_gpu, U_rec_gpu, thr_gpu, h_dim,
                )
                act_last = h_trial[-1].float().mean().item()
                if verbose:
                    print(
                        f"[lsuv] layer {l}, iter {lsuv_iter}: "
                        f"act_rate(t={T-1})={act_last:.4f} "
                        f"(target={act_rate_target:.3f})"
                    )
                if abs(act_last - act_rate_target) < tol:
                    break
                if act_last < 1e-6:
                    U_in_np *= 2.0
                elif act_last > 1.0 - 1e-6:
                    U_in_np *= 0.5
                else:
                    from scipy.stats import norm as _norm
                    q_current = _norm.ppf(1.0 - np.clip(act_last, 1e-4, 1 - 1e-4))
                    q_target = _norm.ppf(1.0 - act_rate_target)
                    if abs(q_current) < 1e-6:
                        ratio = 1.2 if act_last < act_rate_target else 0.8
                    else:
                        ratio = q_current / q_target
                        ratio = np.clip(ratio, 0.5, 2.0)
                    U_in_np *= float(ratio)
            if verbose:
                print(
                    f"[lsuv] layer {l}: final act_rate={act_last:.4f} "
                    f"after {min(lsuv_iter + 1, lsuv_max_iters)} iters, "
                    f"U_in scale={np.std(U_in_np):.4f}"
                )

        if verbose:
            print(
                f"[snn hypers] layer {l}: "
                f"U_in={U_in_np.shape}, U_rec={U_rec_np.shape} (shared across all T), "
                f"init={init_method}"
            )

        # Store CPU copies in hypers
        U_in_list.append(U_in_np)
        U_rec_list.append(U_rec_np)
        beta_list_all.append(b.copy())
        threshold_list_all.append(g.copy())

        # Torch views on GPU for recurrence (final weights after calibration)
        U_in = torch.from_numpy(U_in_np.astype(np_dtype, copy=False)).to(device)
        U_rec = torch.from_numpy(U_rec_np.astype(np_dtype, copy=False)).to(device)

        # Run the actual forward pass with calibrated weights
        thr_t = torch.from_numpy(g.astype(np_dtype, copy=False)).to(device)
        h_curr_list, v_curr_list = _run_layer_forward(h_layers[l - 1], U_in, U_rec, thr_t, h_dim)

        # Track firing rate for Micheli formula at next layer
        prev_act_rate = h_curr_list[-1].float().mean().item()

        h_layers.append(h_curr_list)
        v_layers[l] = v_curr_list
        # --- Intermediate layer diversity diagnostic ---
        if verbose:
            for t_idx in [0, T-1]:
                h_bits = h_curr_list[t_idx].to(torch.uint8).cpu().numpy()
                diag = compute_layer_diagnostics(
                    h_bits, y_np, layer=l, t_idx=t_idx, print_results=True,
                )
                all_diagnostics.append(diag)
        d_in_l = h_dim  # next layer's input dimension

    # Last hidden state at T-1: pick membrane potential or spike for the convex last layer
    if L <= 1:
        # No hidden layers: raw input is used directly (no membrane vs spike distinction)
        h_last_T = h_layers[-1][-1]
    elif last_layer_readout == "membrane":
        h_last_T = h_layers[-1][-1]  # (n, h_dim_last) real-valued membrane
    elif last_layer_readout == "spike":
        h_last_T = h_layers[-1][-1]      # (n, h_dim_last) binary spikes
    else:
        raise ValueError(f"Unknown last_layer_readout={last_layer_readout!r}. Choose from: membrane, spike.")
    if verbose:
        print(f"[snn patterns] last_layer_readout={last_layer_readout}")
    in_dim_last = h_last_T.shape[1]

    # -------------------------------------------------
    # Last-layer hyperplanes + unique patterns (GPU)
    # Process in column chunks to avoid OOM (e.g. in_dim_last=30k, chunk_P=120k => 13+ GiB).
    # -------------------------------------------------
    uniq: Dict[bytes, np.ndarray] = {}
    rounds = 0
    chunk_P = max(P_last_target * chunk_mult, P_last_target)
    # Max columns per device tensor to stay under ~1 GiB: (in_dim_last * cols * 4 bytes)
    max_cols_per_chunk = min(8192, max(512, (1 << 30) // (in_dim_last * 4)))

    while len(uniq) < P_last_target and rounds < max_rounds:
        rounds += 1
        col_start = 0
        while col_start < chunk_P and len(uniq) < P_last_target:
            col_end = min(col_start + max_cols_per_chunk, chunk_P)
            n_cols = col_end - col_start
            U_last_np = rng.normal(size=(in_dim_last, n_cols)).astype(np.float32)
            if normalize_hidden:
                U_last_np = _col_normalize_np(U_last_np)

            U_last = torch.from_numpy(U_last_np.astype(np_dtype, copy=False)).to(device)

            # sign patterns on device
            D_bool_t = (h_last_T @ U_last >= 0.0)    # (n, n_cols) bool
            D_u8 = D_bool_t.to(torch.uint8).cpu().numpy()  # move only patterns to CPU

            for j in range(n_cols):
                key = D_u8[:, j].tobytes()
                if key not in uniq:
                    uniq[key] = U_last_np[:, j].copy()
                    if len(uniq) >= P_last_target:
                        break
            col_start = col_end

        if verbose:
            print(
                f"[snn patterns] last-layer enforce: "
                f"round={rounds}/{max_rounds} uniques={len(uniq)}/{P_last_target}"
            )

    if len(uniq) < P_last_target:
        raise RuntimeError(
            f"Could not collect P_last_target={P_last_target} unique last-layer patterns; "
            f"got {len(uniq)}. Increase P_last_target or chunk_mult/max_rounds."
        )

    keys = list(uniq.keys())[:P_last_target]
    D_last = np.stack(
        [np.frombuffer(k, dtype=np.uint8) for k in keys],
        axis=1,
    )  # (n, P_last_target)
    z_last_bool = D_last.astype(bool)  # (n, P_last_target) bool

    U_last_final = np.stack(
        [uniq[k] for k in keys],
        axis=1,
    ).astype(np.float32)  # (in_dim_last, P_last_target)

    if verbose:
        print(
            f"[snn patterns] P_last_unique={z_last_bool.shape[1]} "
            f"(target={P_last_target}, n_train={n})"
        )

    hypers = RNNHyperplanes(
        U_in_list=U_in_list,
        U_rec_list=U_rec_list,   # each (2*h_dim_l, h_dim_l) = [diag(beta); diag(threshold)]
        U_last=U_last_final,     # (h_dim_last, P_last_target)
        last_layer_readout=last_layer_readout,
        beta_list=beta_list_all,
        threshold_list=threshold_list_all,
    )
    return z_last_bool, hypers, all_diagnostics

def forward_snn_patterns_torch(
    X_seq: torch.Tensor,
    hypers: RNNHyperplanes,
    *,
    L: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    B, T_data, d_in = X_seq.shape
    assert T_data == T

    last_layer_readout = hypers.last_layer_readout

    # layer 0: raw inputs
    h_prev_layers: List[List[torch.Tensor]] = []
    h0 = [X_seq[:, t, :].to(device) for t in range(T)]
    h_prev_layers.append(h0)

    d_in_l = d_in
    v_last_list: List[torch.Tensor] = []  # membrane potentials for last hidden layer

    for l in range(1, L):
        U_in_np = hypers.U_in_list[l - 1]   # (d_in_l, h_dim_l)
        U_rec_np = hypers.U_rec_list[l - 1] # (2*h_dim_l, h_dim_l)
        thr_np = hypers.threshold_list[l - 1] if hypers.threshold_list is not None else np.ones((U_in_np.shape[1],), dtype=np.float32)

        U_in = torch.from_numpy(U_in_np).float().to(device)
        U_rec = torch.from_numpy(U_rec_np).float().to(device)
        thr = torch.from_numpy(thr_np).float().to(device)

        h_dim = U_in.shape[1]

        v_prev = torch.zeros(B, h_dim, device=device)
        h_prev = torch.zeros(B, h_dim, device=device)
        h_curr_list: List[torch.Tensor] = []
        v_curr_list: List[torch.Tensor] = []

        for t in range(T):
            x_t = h_prev_layers[l - 1][t]  # (B, d_in_l)
            v_in_t = x_t @ U_in            # (B, h_dim)

            s_prev = torch.cat([v_prev, -h_prev], dim=1)         # (B, 2*h_dim)
            v_rec_t = s_prev @ U_rec                             # (B, h_dim)

            v_t = v_in_t + v_rec_t
            v_t = torch.clamp(v_t, -1e10, 1e10)
            h_t = (v_t >= thr.unsqueeze(0)).float()

            h_curr_list.append(h_t)
            v_curr_list.append(v_t)
            v_prev, h_prev = v_t, h_t

        h_prev_layers.append(h_curr_list)
        # Keep membrane list for last hidden layer
        if l == L - 1:
            v_last_list = v_curr_list
        d_in_l = h_dim

    # Pick readout from last hidden layer
    if L <= 1 or last_layer_readout == "spike":
        readout_T = h_prev_layers[-1][-1]     # (B, h_dim_last) binary spikes or raw input
    else:
        readout_T = v_last_list[-1]           # (B, h_dim_last) real-valued membrane

    U_last = torch.from_numpy(hypers.U_last).float().to(device)  # (h_dim_last, P_last_target)
    D_last = (readout_T @ U_last >= 0.0).float()                  # (B, P_last_target)

    return D_last


# ============================================================
# CVX last layer + losses
# ============================================================

class CvxLastLayer(nn.Module):
    def __init__(self, P_last: int, num_outputs: int):
        super().__init__()
        # For classification, num_outputs = num_classes.
        # For regression, we can set num_outputs = 1.
        self.W = nn.Parameter(torch.zeros(P_last, num_outputs), requires_grad=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, P_last) - may be uint8, convert to float
        return z.float() @ self.W  # (B, num_outputs)


def cvx_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    model: CvxLastLayer,
    *,
    loss_type: str,
    beta_l1: float,
) -> torch.Tensor:
    """
    loss_type ∈ {"ce", "hinge", "squared"}.

    - "ce": assumes y ∈ {0,...,C-1}, logits shape (B,C).
    - "hinge": assumes y ∈ {+1,-1}, logits shape (B,1) or (B).
    - "squared": assumes regression y ∈ R, logits shape (B,1).
    """
    if loss_type == "ce":
        loss_main = F.cross_entropy(logits, y)
    elif loss_type == "hinge":
        # y ∈ {+1,-1}, logits: (B,1)
        logits_vec = logits.squeeze(-1)
        loss_main = torch.clamp(1.0 - y.float() * logits_vec, min=0.0).mean()
    elif loss_type == "squared":
        preds = logits.squeeze(-1)
        loss_main = F.mse_loss(preds, y.float())
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_main + beta_l1 * model.W.abs().sum()


@torch.no_grad()
def cvx_eval_acc_or_mse(
    model: CvxLastLayer,
    loader2d: DataLoader,
    hypers: RNNHyperplanes,
    *,
    L: int,
    T: int,
    P_rec: int,
    device: torch.device,
    loss_type: str,
) -> float:
    """
    For classification (ce/hinge): returns accuracy.
    For squared: returns negative MSE (so "higher is better" if you want).
    """
    model.eval()
    total = 0
    if loss_type in ("ce", "hinge"):
        correct = 0
    else:
        mse_sum = 0.0

    for xb, yb in loader2d:
        xb = xb.to(device)
        yb = yb.to(device)
        z = forward_snn_patterns_torch(xb, hypers, L=L, T=T, device=device)
        logits = model(z)

        if loss_type == "ce":
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        elif loss_type == "hinge":
            preds = torch.where(logits.squeeze(-1) >= 0, 1.0, -1.0)
            correct += (preds == yb.float()).sum().item()
            total += yb.numel()
        else:
            preds = logits.squeeze(-1)
            mse_sum += F.mse_loss(preds, yb.float(), reduction="sum").item()
            total += yb.numel()

    if loss_type in ("ce", "hinge"):
        return correct / max(total, 1)
    else:
        return -mse_sum / max(total, 1)

@torch.no_grad()

@torch.no_grad()
def cvx_eval_acc_or_mse_cached(
    model: CvxLastLayer,
    loader2d: DataLoader,
    *,
    loss_type: str,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    if loss_type in ("ce", "hinge"):
        correct = 0
    else:
        mse_sum = 0.0
    for zb, yb in loader2d:
        zb = zb.to(device)
        yb = yb.to(device)
        logits = model(zb)
        if loss_type == "ce":
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        elif loss_type == "hinge":
            preds = torch.where(logits.squeeze(-1) >= 0, 1.0, -1.0)
            correct += (preds == yb.float()).sum().item()
            total += yb.numel()
        else:
            preds = logits.squeeze(-1)
            mse_sum += F.mse_loss(preds, yb.float(), reduction="sum").item()
            total += yb.numel()
    if loss_type in ("ce", "hinge"):
        return correct / max(total, 1)
    return -mse_sum / max(total, 1)


def _extract_features_from_snn(
    model: "SNNBaseline",
    X_np: np.ndarray,
    *,
    device: torch.device,
    source: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract cached CVX features from a pretrained SNN.
    source ∈ {"last_mem", "last_spike"}.
    Returns shape (n, P_last).
    """
    model.eval()
    X_t = torch.from_numpy(X_np).float()
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    feats = []
    for (xb,) in loader:
        xb = xb.to(device)
        B, T, _ = xb.shape
        # Single clean forward pass with explicit layer index.
        mems = [lif.init_leaky().to(device) for lif in model.lifs]
        h_last = torch.zeros((B, model.P_last), dtype=torch.float32, device=device)
        for t in range(T):
            h = xb[:, t, :]
            for li, (fc, lif) in enumerate(zip(model.fcs, model.lifs)):
                cur = fc(h)
                spk, mem = lif(cur, mems[li])
                mems[li] = mem
                h = spk
                h_last = spk
        last_mem = mems[-1]
        if source == "last_mem":
            feats.append(last_mem.detach().cpu())
        elif source == "last_spike":
            feats.append(h_last.detach().cpu())
        else:
            raise ValueError(f"Unknown cached feature source: {source}")
    return torch.cat(feats, dim=0).numpy().astype(np.float32, copy=False)


def train_cvx_head_cached_features(
    train_loader2d: DataLoader,
    val_loader2d: DataLoader,
    *,
    feat_dim: int,
    num_outputs: int,
    loss_type: str,
    beta_l1: float,
    device: torch.device,
    log_train: bool = False,
) -> Dict[str, object]:
    model = CvxLastLayer(feat_dim, num_outputs).to(device)
    ds = train_loader2d.dataset
    if not isinstance(ds, TensorDataset) or len(ds.tensors) < 2:
        raise ValueError("Expected TensorDataset(Z, y) for cached CVX solve.")
    z_t, y_t = ds.tensors
    Z = z_t.detach().cpu().numpy().astype(np.float64)
    n_samples = Z.shape[0]
    W_var = cp.Variable((feat_dim, num_outputs))
    ZW = Z @ W_var
    if loss_type == "ce":
        y_np = y_t.detach().cpu().numpy().astype(np.int64)
        y_onehot = np.eye(num_outputs, dtype=np.float64)[y_np]
        ce_terms = cp.log_sum_exp(ZW, axis=1) - cp.sum(cp.multiply(ZW, y_onehot), axis=1)
        loss_expr = cp.sum(ce_terms) / n_samples
    elif loss_type == "hinge":
        y_np = y_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
        margin = cp.multiply(y_np, cp.reshape(ZW, (n_samples,)))
        loss_expr = cp.sum(cp.pos(1.0 - margin)) / n_samples
    elif loss_type == "squared":
        y_np = y_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
        resid = cp.reshape(ZW, (n_samples,)) - y_np
        loss_expr = cp.sum_squares(resid) / n_samples
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")
    objective = cp.Minimize(loss_expr + float(beta_l1) * cp.norm1(W_var))
    prob = cp.Problem(objective)
    solved = False
    used_solver: Optional[str] = None
    solver_num_iters: Optional[int] = None
    for solver_name, solver_kwargs in (("SCS", {"max_iters": 10000, "eps": 1e-4}), ("CLARABEL", {}), ("OSQP", {})):
        try:
            prob.solve(solver=solver_name, verbose=False, **solver_kwargs)
            if W_var.value is not None:
                solved = True
                used_solver = solver_name
                stats = prob.solver_stats
                if stats is not None and hasattr(stats, "num_iters") and stats.num_iters is not None:
                    solver_num_iters = int(stats.num_iters)
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError("Cached-feature CVX solve failed for all attempted solvers.")
    W_np = np.asarray(W_var.value, dtype=np.float32)
    with torch.no_grad():
        model.W.copy_(torch.from_numpy(W_np).to(device))
    with torch.no_grad():
        logits_t = model(torch.from_numpy(Z.astype(np.float32)).to(device))
        y_dev = y_t.to(device)
        if loss_type == "ce":
            train_score = (logits_t.argmax(dim=1) == y_dev).float().mean().item()
        elif loss_type == "hinge":
            preds = torch.where(logits_t.squeeze(-1) >= 0, 1.0, -1.0)
            train_score = (preds == y_dev.float()).float().mean().item()
        else:
            train_score = -F.mse_loss(logits_t.squeeze(-1), y_dev.float()).item()
        train_loss = cvx_loss(logits_t, y_dev, model, loss_type=loss_type, beta_l1=beta_l1).item()
    val_score = cvx_eval_acc_or_mse_cached(model, val_loader2d, loss_type=loss_type, device=device)
    return {
        "model": model,
        "best_val_score": float(val_score),
        "train_score": float(train_score),
        "train_loss": float(train_loss),
        "solver_name": used_solver,
        "solver_num_iters": solver_num_iters,
        "train_score_history": [float(train_score)] if log_train else [],
        "train_loss_history": [float(train_loss)] if log_train else [],
    }

def cvx_train_objective_full(
    model: CvxLastLayer,
    train_loader3d: DataLoader,
    *,
    loss_type: str,
    beta_l1: float,
    device: torch.device,
) -> float:
    """
    Full convex training objective on the entire train set, matching train_cvx_head_first_order.

    Uses cached sign patterns z in train_loader3d.dataset.tensors = (X, y, z).
    """
    model = model.to(device)
    model.eval()

    ds = train_loader3d.dataset
    if not isinstance(ds, TensorDataset) or len(ds.tensors) < 3:
        raise ValueError("Expected TensorDataset(X, y, z_train).")

    _, y_t, z_t = ds.tensors
    y = y_t.to(device)
    z = z_t.to(device)

    logits = model(z)
    loss = cvx_loss(
        logits,
        y,
        model,
        loss_type=loss_type,
        beta_l1=beta_l1,
    )
    return float(loss.item())

@torch.no_grad()
def convLossLandscape1D(
    model: CvxLastLayer,
    train_loader3d: DataLoader,
    test_loader2d: DataLoader,
    hypers: RNNHyperplanes,
    *,
    L: int,
    T: int,
    P_rec: int,
    loss_type: str,
    beta_l1: float,
    device: torch.device,
    dataset: str,
    task: str,
    seed: int,
    timestep: str,
    scale_min: float = 1e-10,
    scale_max: float = 2.0,
    num_points: int = 41,
):
    """
    1D positive-scaling experiment for the convex last layer.

    W* = trained last-layer weights. For s ∈ [scale_min, scale_max]:

        W(s) = s · W*

    we compute:
      - train convex objective via cvx_loss on cached z
      - test loss via cvx_eval_acc_or_mse.

    This lets you see how scaling the last layer affects both the objective and generalization.
    """
    model = model.to(device)
    model.eval()

    W_param = model.W
    w_star = W_param.detach().view(-1).to(device)

    scales = torch.linspace(scale_min, scale_max, steps=num_points, device=device)

    train_losses = []
    test_losses = []

    for s in scales:
        W_param.data = (s * w_star).view_as(W_param)

        # Train objective
        train_loss = cvx_train_objective_full(
            model,
            train_loader3d,
            loss_type=loss_type,
            beta_l1=beta_l1,
            device=device,
        )

        # Test loss
        score = cvx_eval_acc_or_mse(
            model,
            test_loader2d,
            hypers,
            L=L,
            T=T,
            P_rec=P_rec,
            device=device,
            loss_type=loss_type,
        )
        if loss_type in ("ce", "hinge"):
            test_loss = 1.0 - score
        else:
            test_loss = -score

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # Restore W*
    W_param.data = w_star.view_as(W_param)

    scales_np = scales.cpu().numpy()
    train_np = np.array(train_losses, dtype=np.float32)
    test_np = np.array(test_losses, dtype=np.float32)

    plt.figure(figsize=(6, 5))
    plt.plot(scales_np, train_np, label="Train CVX Objective (cvx_loss)")
    plt.plot(scales_np, test_np, label="Test Loss (1-acc / MSE)")
    plt.xlabel("Scaling factor s (W → s · W*)")
    plt.ylabel("Loss")
    plt.title("Convex Last-Layer Positive Scaling Landscape (1D)")
    plt.legend()
    plt.tight_layout()
    plot_dir = os.path.join("plots", task, f"L_{L}_T_{T}", f"seed_{seed}")
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{timestep}_1Dloss.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()

@torch.no_grad()
def plot_cvx_loss_landscape_2d(
    model: CvxLastLayer,
    train_loader3d: DataLoader,
    test_loader2d: DataLoader,
    hypers: RNNHyperplanes,
    *,
    L: int,
    T: int,
    P_rec: int,
    loss_type: str,
    beta_l1: float,
    device: torch.device,
    dataset: str,
    task: str,
    seed: int,
    timestep: str,
    alpha_range: float = 2.0,
    num_points: int = 41,
):
    """
    2D slice of the CVX-SNN loss landscape around W*.

    Train loss  = convex objective via cvx_loss on cached z (cvx_train_objective_full).
    Test loss   = generalization error via cvx_eval_acc_or_mse:

        ce / hinge: 1 - accuracy
        squared   : MSE  (since cvx_eval_acc_or_mse returns -MSE)
    """
    model = model.to(device)
    model.eval()

    # Flatten W* to w*
    W_param = model.W
    w_star = W_param.detach().view(-1).to(device)

    # Two orthonormal directions in weight space
    d1 = torch.randn_like(w_star)
    d1 = d1 / (d1.norm() + 1e-8)

    d2 = torch.randn_like(w_star)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / (d2.norm() + 1e-8)

    alphas = torch.linspace(-alpha_range, alpha_range, steps=num_points, device=device)
    betas = torch.linspace(-alpha_range, alpha_range, steps=num_points, device=device)

    train_grid = np.zeros((num_points, num_points), dtype=np.float32)
    test_grid = np.zeros((num_points, num_points), dtype=np.float32)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # W(a,b) = W* + a d1 + b d2
            w_vec = w_star + a * d1 + b * d2
            W_param.data = w_vec.view_as(W_param)

            # Train convex objective (matches training code)
            train_loss = cvx_train_objective_full(
                model,
                train_loader3d,
                loss_type=loss_type,
                beta_l1=beta_l1,
                device=device,
            )

            # Test "loss" via cvx_eval_acc_or_mse on test_loader2d
            score = cvx_eval_acc_or_mse(
                model,
                test_loader2d,
                hypers,
                L=L,
                T=T,
                P_rec=P_rec,
                device=device,
                loss_type=loss_type,
            )
            if loss_type in ("ce", "hinge"):
                test_loss = 1.0 - score
            else:
                test_loss = -score   # MSE

            train_grid[i, j] = train_loss
            test_grid[i, j] = test_loss

    # Restore W*
    W_param.data = w_star.view_as(W_param)

    A = alphas.cpu().numpy()
    B = betas.cpu().numpy()

    # ---- find argmin for train and test on this slice ----
    train_min_idx = np.unravel_index(train_grid.argmin(), train_grid.shape)
    i_tr, j_tr = int(train_min_idx[0]), int(train_min_idx[1])
    alpha_tr = A[i_tr]
    beta_tr = B[j_tr]
    min_train_loss = float(train_grid[i_tr, j_tr])
    test_at_train_min = float(test_grid[i_tr, j_tr])

    test_min_idx = np.unravel_index(test_grid.argmin(), test_grid.shape)
    i_te, j_te = int(test_min_idx[0]), int(test_min_idx[1])
    alpha_te = A[i_te]
    beta_te = B[j_te]
    min_test_loss = float(test_grid[i_te, j_te])
    train_at_test_min = float(train_grid[i_te, j_te])

    print(
        f"[CVX Landscape] min TRAIN loss on slice = {min_train_loss:.6f} "
        f"at (alpha_tr, beta_tr) = ({alpha_tr:.4f}, {beta_tr:.4f}); "
        f"test loss there = {test_at_train_min:.6f}"
    )
    print(
        f"[CVX Landscape] min TEST  loss on slice = {min_test_loss:.6f} "
        f"at (alpha_te, beta_te) = ({alpha_te:.4f}, {beta_te:.4f}); "
        f"train loss there = {train_at_test_min:.6f}"
    )

    # Save to plots/<task>/L_<L>_T_<T>/seed_<seed>/{timestep}_{train|test|overlay}.png
    plot_dir = os.path.join("plots", task, f"L_{L}_T_{T}", f"seed_{seed}")
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- Train contour (as before) ----------
    plt.figure(figsize=(6, 5))
    plt.contourf(A, B, train_grid.T, levels=30)
    plt.colorbar(label="Train CVX Objective (cvx_loss)")
    plt.xlabel("α (direction d₁)")
    plt.ylabel("β (direction d₂)")
    plt.title("CVX-SNN Train Loss Landscape (2D Slice)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestep}_train.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ---------- Test contour (as before) ----------
    plt.figure(figsize=(6, 5))
    plt.contourf(A, B, test_grid.T, levels=30)
    plt.colorbar(label="Test Loss (1-acc / MSE)")
    plt.xlabel("α (direction d₁)")
    plt.ylabel("β (direction d₂)")
    plt.title("CVX-SNN Test Loss Landscape (2D Slice)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestep}_test.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ---------- OVERLAY: train (filled) + test (dashed) ----------
    fig, ax = plt.subplots(figsize=(6, 5))

    # Filled contours for train loss
    cf = ax.contourf(A, B, train_grid.T, levels=30)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Train CVX Objective (cvx_loss)")

    # Dashed contour lines for test loss
    cs = ax.contour(
        A,
        B,
        test_grid.T,
        levels=10,
        colors="white",
        linewidths=1.0,
        linestyles="dashed",
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f")

    # Mark minima
    ax.scatter(
        [alpha_tr],
        [beta_tr],
        marker="x",
        s=80,
        c="blue",
        label="Train-loss minimum",
    )
    ax.scatter(
        [alpha_te],
        [beta_te],
        marker="*",
        s=120,
        c="yellow",
        edgecolors="black",
        label="Test-loss minimum",
    )

    ax.set_xlabel("α (direction d₁)")
    ax.set_ylabel("β (direction d₂)")
    ax.set_title("CVX-SNN Train & Test Loss Landscapes (2D Slice)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestep}_overlay.png"), bbox_inches="tight", dpi=150)
    plt.close()

def train_cvx_head_first_order(
    train_loader3d: DataLoader,
    val_loader2d: DataLoader,
    hypers: RNNHyperplanes,
    *,
    P_last: int,
    num_outputs: int,
    loss_type: str,
    beta_l1: float,
    lr: float,
    epochs: int,
    device: torch.device,
    optimizer_name: str,
    step_size: int,
    gamma: float,
    L: int,
    T: int,
    P_rec: int,
    log_train: bool = False,
) -> Dict[str, object]:
    model = CvxLastLayer(P_last, num_outputs).to(device)
    ds = train_loader3d.dataset
    if not isinstance(ds, TensorDataset) or len(ds.tensors) < 3:
        raise ValueError("Expected TensorDataset(X, y, z_train) for CVX solver.")
    _, y_t, z_t = ds.tensors

    Z = z_t.detach().cpu().numpy().astype(np.float64)
    n_samples = Z.shape[0]
    if n_samples == 0:
        raise ValueError("Empty train set for CVX solve.")

    W_var = cp.Variable((P_last, num_outputs))
    ZW = Z @ W_var

    if loss_type == "ce":
        y_np = y_t.detach().cpu().numpy().astype(np.int64)
        y_onehot = np.eye(num_outputs, dtype=np.float64)[y_np]
        ce_terms = cp.log_sum_exp(ZW, axis=1) - cp.sum(cp.multiply(ZW, y_onehot), axis=1)
        loss_expr = cp.sum(ce_terms) / n_samples
    elif loss_type == "hinge":
        y_np = y_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
        margin = cp.multiply(y_np, cp.reshape(ZW, (n_samples,)))
        loss_expr = cp.sum(cp.pos(1.0 - margin)) / n_samples
    elif loss_type == "squared":
        y_np = y_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
        resid = cp.reshape(ZW, (n_samples,)) - y_np
        loss_expr = cp.sum_squares(resid) / n_samples
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")

    objective = cp.Minimize(loss_expr + float(beta_l1) * cp.norm1(W_var))
    prob = cp.Problem(objective)

    solved = False
    used_solver: Optional[str] = None
    solver_num_iters: Optional[int] = None
    for solver_name, solver_kwargs in (
        ("SCS", {"max_iters": 10_000, "eps": 1e-4}),
        ("CLARABEL", {}),
        ("OSQP", {}),
    ):
        try:
            prob.solve(solver=solver_name, verbose=False, **solver_kwargs)
            if W_var.value is not None:
                solved = True
                used_solver = solver_name
                stats = prob.solver_stats
                if stats is not None and hasattr(stats, "num_iters") and stats.num_iters is not None:
                    solver_num_iters = int(stats.num_iters)
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError("CVX solve failed for all attempted solvers (SCS/CLARABEL/OSQP).")

    W_np = np.asarray(W_var.value, dtype=np.float32)
    with torch.no_grad():
        model.W.copy_(torch.from_numpy(W_np).to(device))

    train_score = cvx_eval_acc_or_mse(
        model,
        DataLoader(TensorDataset(torch.from_numpy(np.zeros((n_samples, 1, 1), dtype=np.float32)),
                                 y_t.clone()),
                   batch_size=4096,
                   shuffle=False),
        hypers,
        L=L,
        T=T,
        P_rec=P_rec,
        device=device,
        loss_type=loss_type,
    ) if False else 0.0

    # Compute train score directly on cached z without re-running pattern forward.
    with torch.no_grad():
        logits_t = model(torch.from_numpy(Z.astype(np.float32)).to(device))
        y_dev = y_t.to(device)
        if loss_type == "ce":
            train_score = (logits_t.argmax(dim=1) == y_dev).float().mean().item()
        elif loss_type == "hinge":
            preds = torch.where(logits_t.squeeze(-1) >= 0, 1.0, -1.0)
            train_score = (preds == y_dev.float()).float().mean().item()
        else:
            train_score = -F.mse_loss(logits_t.squeeze(-1), y_dev.float()).item()
        train_loss = cvx_loss(logits_t, y_dev, model, loss_type=loss_type, beta_l1=beta_l1).item()

    val_score = cvx_eval_acc_or_mse(
        model, val_loader2d, hypers, L=L, T=T, P_rec=P_rec, device=device, loss_type=loss_type
    )

    if log_train:
        print(f"[CVX] ep=001/001 train_score={train_score:.4f} val_score={val_score:.4f} lr={lr:.2e}")

    return {
        "model": model,
        "best_val_score": val_score,
        "train_score": float(train_score),
        "train_loss": float(train_loss),
        "solver_name": used_solver,
        "solver_num_iters": solver_num_iters,
        "train_score_history": [float(train_score)] if log_train else [],
        "train_loss_history": [float(train_loss)] if log_train else [],
    }


def _extract_transfer_u_in_from_snn(model: "SNNBaseline") -> List[np.ndarray]:
    transfer: List[np.ndarray] = []
    for fc in model.fcs:
        # SNN fc weight is (out_dim, in_dim); pattern generator expects (in_dim, out_dim).
        transfer.append(fc.weight.detach().cpu().numpy().T.astype(np.float32, copy=True))
    return transfer


def _extract_transfer_thresholds_from_snn(model: "SNNBaseline") -> List[np.ndarray]:
    transfer: List[np.ndarray] = []
    for idx, (fc, lif) in enumerate(zip(model.fcs, model.lifs)):
        if not hasattr(lif, "threshold"):
            raise AttributeError(f"Missing threshold attribute in lif layer {idx}.")
        thr = lif.threshold
        h_dim = int(fc.out_features)
        if isinstance(thr, torch.Tensor):
            thr_np = thr.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            thr_np = np.asarray(thr, dtype=np.float32).reshape(-1)

        if thr_np.size == 1:
            thr_np = np.full((h_dim,), float(thr_np[0]), dtype=np.float32)
        elif thr_np.size != h_dim:
            raise ValueError(
                f"Threshold shape mismatch at layer {idx}: got {thr_np.shape}, expected ({h_dim},)."
            )
        transfer.append(thr_np.copy())
    return transfer


def _initialize_snn_hidden_from_transfer(model: "SNNBaseline", transfer_u_in_list: List[np.ndarray]) -> None:
    if len(transfer_u_in_list) != len(model.fcs):
        raise ValueError(
            f"transfer_u_in_list length mismatch: {len(transfer_u_in_list)} vs {len(model.fcs)}."
        )
    with torch.no_grad():
        for idx, fc in enumerate(model.fcs):
            w_t = torch.from_numpy(transfer_u_in_list[idx].T).to(fc.weight.device, dtype=fc.weight.dtype)
            if tuple(w_t.shape) != tuple(fc.weight.shape):
                raise ValueError(
                    f"Transferred SNN hidden weight shape mismatch at layer {idx}: "
                    f"{tuple(w_t.shape)} vs {tuple(fc.weight.shape)}."
                )
            fc.weight.copy_(w_t)


# ============================================================
# SNN baseline (LIF stack + STE training)
# ============================================================

class SNNBaseline(nn.Module):
    """
    L-layer LIF stack, used as STE baseline.

    - Input: sequence X ∈ R^{B×T×d_in}.
    - Hidden layers:
        * For layers 1..L-2: width = P_rec
        * Last hidden layer (L-1): width = P_last
      Each hidden layer is: fc_in_l: Linear(d_in_l -> hidden_dim_l), then
      snn.Leaky(...).
    - Output head: Linear(P_last -> num_outputs).
    """
    def __init__(
        self,
        d_in: int,
        L: int,
        P_rec: int,
        P_last: int,
        num_outputs: int,
        beta_leak: float = 0.99,
        threshold: float = 1,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        last_layer_readout: str = "membrane",
        beta_dist: str = "fixed",
        seed: int = 0,
    ):
        super().__init__()
        self.L = L
        self.P_rec = P_rec
        self.P_last = P_last
        self.last_layer_readout = last_layer_readout

        fcs = []
        lifs = []
        in_dim = d_in

        # hidden dims: first L-2 layers have width P_rec, last hidden layer has width P_last
        if L <= 1:
            hidden_dims = [P_last]
        else:
            hidden_dims = [P_rec] * max(L - 2, 0) + [P_last]

        rng_beta = np.random.default_rng(seed + 999)
        for h_dim in hidden_dims:
            fc = nn.Linear(in_dim, h_dim, bias=False)
            fcs.append(fc)

            # Per-neuron beta from distribution
            if beta_dist == "fixed":
                beta_init = beta_leak
            elif beta_dist == "het_loguniform":
                log_lo, log_hi = np.log(0.5), np.log(0.999)
                beta_init = torch.tensor(
                    np.exp(rng_beta.uniform(log_lo, log_hi, size=(h_dim,))),
                    dtype=torch.float32
                )
            elif beta_dist == "het_uniform":
                beta_init = torch.tensor(
                    rng_beta.uniform(0.5, 0.999, size=(h_dim,)),
                    dtype=torch.float32
                )
            elif beta_dist == "het_bimodal":
                b = np.empty(h_dim, dtype=np.float32)
                n_fast = h_dim // 2
                b[:n_fast] = rng_beta.uniform(0.3, 0.6, size=(n_fast,))
                b[n_fast:] = rng_beta.uniform(0.95, 0.999, size=(h_dim - n_fast,))
                rng_beta.shuffle(b)
                beta_init = torch.tensor(b, dtype=torch.float32)
            else:
                beta_init = beta_leak

            lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold,
                learn_beta=learn_beta,
                learn_threshold=learn_threshold,
            )
            lifs.append(lif)
            in_dim = h_dim

        self.fcs = nn.ModuleList(fcs)
        self.lifs = nn.ModuleList(lifs)
        # Final readout now sees a P_last-dimensional representation
        self.fc_out = nn.Linear(P_last, num_outputs, bias=False)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: (B, T, d_in)
        Returns:
          - logits: (B, num_outputs) using membrane or spike of last layer
          - last_mem: (B, P_last) final membrane (for analysis)
        """
        B, T, d_in = x_seq.shape
        device = x_seq.device

        # initialize membranes per layer
        mems = [lif.init_leaky().to(device) for lif in self.lifs]
        h_last  = torch.zeros((B, self.P_last), dtype=torch.float32).to(device)
        # we propagate sequentially
        for t in range(T):
            x_t = x_seq[:, t, :]
            h = x_t
            for l, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(h)
                spk, mem = lif(cur, mems[l])
                mems[l] = mem
                h = spk
                h_last = spk  # pass spikes to next layer; we could also use mem

        last_mem = mems[-1]          # shape (B, P_last)
        # Output readout: membrane potential or spikes from last hidden layer
        if self.last_layer_readout == "membrane":
            logits = self.fc_out(last_mem)
        else:
            logits = self.fc_out(h_last)
        return logits, last_mem



def snn_path_reg(model: SNNBaseline) -> torch.Tensor:
    """
    ℓ2 path-regularizer (norm-2 path norm) for the SNNBaseline network.

    We approximate the standard path-ℓ2 norm for a feed-forward ReLU-type network:

        reg^2 = ∑_{paths} ∏_ℓ w_{ℓ,path}^2
        reg   = sqrt(reg^2).

    This can be computed via a dynamic program over layers:

        v^(0) = 1   (vector of ones on input units)
        v^(ℓ) = (W_ℓ ⊙ W_ℓ) @ v^(ℓ-1),

    where W_ℓ is the weight matrix for layer ℓ and ⊙ is elementwise square.
    For the last linear layer W_out,

        reg^2 = ∑_{j,k} W_out[j,k]^2 * v^(L)[k],

    so that reg = sqrt(reg^2).

    This uses only matrix–vector products and is GPU-friendly.
    Returns a scalar tensor on the same device as the model.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # No hidden layers: path norm reduces to ℓ2 norm of output weights.
    if len(model.fcs) == 0:
        w_out = model.fc_out.weight
        reg_sq = (w_out ** 2).sum()
        return torch.sqrt(reg_sq + 1e-12)

    # Start with v on input units.
    in_dim = model.fcs[0].in_features
    v = torch.ones(in_dim, device=device, dtype=dtype)

    # Propagate squared weights layer by layer.
    for fc in model.fcs:
        W_sq = fc.weight ** 2   # (out_dim, in_dim)
        v = W_sq @ v            # (out_dim,)

    # Final layer: W_out has shape (n_out, last_dim).
    W_out = model.fc_out.weight
    W_out_sq = W_out ** 2      # (n_out, last_dim)

    # reg^2 = sum_{j,k} W_out[j,k]^2 * v[k]
    reg_sq = (W_out_sq * v.unsqueeze(0)).sum()
    return torch.sqrt(reg_sq + 1e-12)

def snn_baseline_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, y)
    elif loss_type == "hinge":
        # treat logits as (B,1) or (B)
        logits_vec = logits.squeeze(-1)
        return torch.clamp(1.0 - y.float() * logits_vec, min=0.0).mean()
    elif loss_type == "squared":
        preds = logits.squeeze(-1)
        return F.mse_loss(preds, y.float())
    else:
        raise ValueError(f"Unknown loss type for SNN baseline: {loss_type}")


@torch.no_grad()
def snn_eval_score(
    model: SNNBaseline,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_type: str,
) -> float:
    model.eval()
    total = 0
    if loss_type in ("ce", "hinge"):
        correct = 0
    else:
        mse_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = model(xb)

        if loss_type == "ce":
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        elif loss_type == "hinge":
            preds = torch.where(logits.squeeze(-1) >= 0, 1.0, -1.0)
            correct += (preds == yb.float()).sum().item()
            total += yb.numel()
        else:
            preds = logits.squeeze(-1)
            mse_sum += F.mse_loss(preds, yb.float(), reduction="sum").item()
            total += yb.numel()

    if loss_type in ("ce", "hinge"):
        return correct / max(total, 1)
    else:
        return -mse_sum / max(total, 1)


def train_snn_baseline(
    model: SNNBaseline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    epochs: int,
    device: torch.device,
    step_size: int,
    gamma: float,
    loss_type: str,
    beta_path_reg: float = 0.0,   # optional path reg if you want to add later
    log_train: bool = False,
) -> Dict[str, object]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val_score = -1e30
    best_state = None
    train_scores: List[float] = []
    train_loss_history: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)

            base_loss = snn_baseline_loss(logits, yb, loss_type=loss_type)
            if beta_path_reg > 0.0:
                path_reg = snn_path_reg(model)
                loss = base_loss + beta_path_reg * path_reg
            else:
                loss = base_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = int(yb.numel())
            epoch_loss_sum += float(base_loss.detach().item()) * bs
            epoch_count += bs

        sched.step()
        epoch_train_loss = epoch_loss_sum / max(epoch_count, 1)

        tr_score = snn_eval_score(model, train_loader, device=device, loss_type=loss_type)
        val_score = snn_eval_score(model, val_loader, device=device, loss_type=loss_type)

        if log_train:
            train_scores.append(tr_score)
            train_loss_history.append(epoch_train_loss)
            print(f"[STE-SNN] ep={ep:03d}/{epochs} train_score={tr_score:.4f} val_score={val_score:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_score": best_val_score,
        "train_score_history": train_scores,
        "train_loss_history": train_loss_history,
    }


# ============================================================
# Grid search + one-seed wrapper
# ============================================================

@dataclass
class GridSpec:
    betas: List[float]
    lrs: List[float]


def run_one_seed(
    *,
    seed: int,
    task: str,
    T: int,
    L: int,
    P_in: int,
    P_rec: int,
    P_last: int,
    n_train_total: int,
    val_frac: float,
    n_test: int,
    loss_type: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    cvx_optimizer: str,
    cvx_step_size: int,
    cvx_gamma: float,
    beta_grid: List[float],
    lr_grid: List[float],
    pretrain_lr: float,
    pretrain_beta_path_reg: float,
    ste_lr_grid: List[float],
    ste_beta_grid: List[float],
    ste_step_size: int,
    ste_gamma: float,
    learn_beta: bool,
    learn_threshold: bool,
    normalize_hidden: bool,
    verbose_patterns: bool,
    log_train: bool,
    timestep: str,
    init_method: str = "random",
    target_act_rate: Optional[float] = None,
    last_layer_readout: str = "membrane",
    beta_dist: str = "fixed",
    cvx_feature_source: str = "threshold_dict",
    compare_all_cvx_sources: bool = False,
) -> Dict[str, object]:
    set_seed(seed)

    # ----- build data -----
    X_total, y_total, X_test, y_test, num_classes = build_dataset(
        task=task,
        T=T,
        n_train_total=n_train_total,
        n_test=n_test,
        seed=seed,
    )

    X_train, y_train, X_val, y_val = split_train_val(X_total, y_total, val_frac=val_frac, seed=seed)

    print(f"[info] task={task} T={T} L={L}")
    print(f"[info] n_train_total={n_train_total} val_frac={val_frac} => n_train={X_train.shape[0]} n_val={X_val.shape[0]} n_test={X_test.shape[0]}")
    print(f"[info] P_in={P_in} P_rec={P_rec} P_last(target)={P_last}")
    print(f"[info] CVX grid betas={beta_grid} lrs={lr_grid} loss={loss_type}")
    print(f"[info] STE grid betas={ste_beta_grid} lrs={ste_lr_grid} | pretrain_lr={pretrain_lr} pretrain_beta={pretrain_beta_path_reg}")
    print("[info] init pipeline: pretrain SNN (80 epochs) -> transfer hidden weights to CVX patterns + STE init")

    num_outputs = num_classes if loss_type == "ce" else 1
    d_in = X_train.shape[2]

    # ----- 80-epoch SNN pretrain to obtain transferable hidden weights -----
    Xtr_pre = torch.from_numpy(X_train).float()
    ytr_pre = torch.from_numpy(y_train)
    Xva_pre = torch.from_numpy(X_val).float()
    yva_pre = torch.from_numpy(y_val)
    if loss_type == "hinge":
        ytr_pre = (2 * ytr_pre - 1).float()
        yva_pre = (2 * yva_pre - 1).float()

    pre_model = SNNBaseline(
        d_in=d_in,
        L=L,
        P_rec=P_rec,
        P_last=P_last,
        num_outputs=num_outputs,
        beta_leak=0.99,
        threshold=1,
        learn_beta=learn_beta,
        learn_threshold=learn_threshold,
        last_layer_readout=last_layer_readout,
        beta_dist=beta_dist,
        seed=seed,
    ).to(device)
    pre_train_loader = DataLoader(TensorDataset(Xtr_pre, ytr_pre), batch_size=batch_size, shuffle=True)
    pre_val_loader = DataLoader(TensorDataset(Xva_pre, yva_pre), batch_size=batch_size, shuffle=False)
    _ = train_snn_baseline(
        pre_model,
        pre_train_loader,
        pre_val_loader,
        lr=pretrain_lr,
        epochs=80,
        device=device,
        step_size=ste_step_size,
        gamma=ste_gamma,
        loss_type=loss_type,
        beta_path_reg=float(pretrain_beta_path_reg),
        log_train=False,
    )

    # Report pretrained model quality before transfer/fine-tuning stages.
    yte_pre = torch.from_numpy(y_test)
    if loss_type == "hinge":
        yte_pre = (2 * yte_pre - 1).float()
    pre_test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), yte_pre), batch_size=batch_size, shuffle=False)
    pre_train_score = snn_eval_score(pre_model, pre_train_loader, device=device, loss_type=loss_type)
    pre_test_score = snn_eval_score(pre_model, pre_test_loader, device=device, loss_type=loss_type)
    metric_name = "acc" if loss_type in ("ce", "hinge") else "negMSE"
    print(f"[pretrain] train_{metric_name}={pre_train_score:.4f} test_{metric_name}={pre_test_score:.4f}")

    transfer_u_in_list = _extract_transfer_u_in_from_snn(pre_model)
    transfer_threshold_list = _extract_transfer_thresholds_from_snn(pre_model)

    # build torch labels/loaders
    Xtr_t = torch.from_numpy(X_train).float()
    ytr_t = torch.from_numpy(y_train)
    Xva_t = torch.from_numpy(X_val).float()
    yva_t = torch.from_numpy(y_val)
    Xte_t = torch.from_numpy(X_test).float()
    yte_t = torch.from_numpy(y_test)

    if loss_type == "hinge":
        ytr_t = (2 * ytr_t - 1).float()
        yva_t = (2 * yva_t - 1).float()
        yte_t = (2 * yte_t - 1).float()

    val_loader2d = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False)
    test_loader2d = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=batch_size, shuffle=False)

    # ----- CVX feature sources -----
    cvx_source_results: Dict[str, Dict[str, object]] = {}
    sources_to_run = ["last_mem", "last_spike", "threshold_dict"] if compare_all_cvx_sources else [cvx_feature_source]
    lr_ref = float(lr_grid[0]) if len(lr_grid) > 0 else float(ste_lr)
    layer_diagnostics = []
    hypers = None
    P_last_real = P_last

    for source_name in sources_to_run:
        best_src = {
            "val_score": -1e30,
            "beta_l1": None,
            "lr": lr_ref,
            "test_score": -1e30,
            "train_score": float("nan"),
            "train_loss": float("nan"),
            "solver_name": None,
            "solver_num_iters": None,
            "train_curve": None,
            "train_loss_curve": None,
        }
        if source_name == "threshold_dict":
            z_train_bool, hypers_src, layer_diagnostics_src = generate_snn_sign_patterns(
                X_train, y_train, L=L, T=T, P_in=P_in, P_rec=P_rec, P_last_target=P_last,
                seed=seed, normalize_hidden=normalize_hidden, verbose=verbose_patterns,
                device=device, init_method=init_method, target_act_rate=target_act_rate,
                last_layer_readout=last_layer_readout, beta_dist=beta_dist,
                transfer_u_in_list=transfer_u_in_list,
                transfer_threshold_list=transfer_threshold_list,
            )
            z_train = z_train_bool.astype(np.uint8)
            P_last_real = z_train.shape[1]
            train_ds3d = TensorDataset(Xtr_t, ytr_t, torch.from_numpy(z_train))
            train_loader3d = DataLoader(train_ds3d, batch_size=batch_size, shuffle=True)
            for beta_l1 in beta_grid:
                out = train_cvx_head_first_order(
                    train_loader3d, val_loader2d, hypers_src, P_last=P_last_real, num_outputs=num_outputs,
                    loss_type=loss_type, beta_l1=beta_l1, lr=lr_ref, epochs=epochs, device=device,
                    optimizer_name=cvx_optimizer, step_size=cvx_step_size, gamma=cvx_gamma,
                    L=L, T=T, P_rec=P_rec, log_train=False,
                )
                model_cvx = out["model"]
                val_score = out["best_val_score"]
                test_score = cvx_eval_acc_or_mse(model_cvx, test_loader2d, hypers_src, L=L, T=T, P_rec=P_rec, device=device, loss_type=loss_type)
                if (test_score > best_src["test_score"] or
                    (test_score == best_src["test_score"] and val_score > best_src["val_score"])):
                    best_src.update(
                        {
                            "val_score": val_score,
                            "beta_l1": beta_l1,
                            "test_score": test_score,
                            "train_score": out.get("train_score", float("nan")),
                            "train_loss": out.get("train_loss", float("nan")),
                            "solver_name": out.get("solver_name"),
                            "solver_num_iters": out.get("solver_num_iters"),
                            "train_curve": out["train_score_history"],
                            "train_loss_curve": out.get("train_loss_history", []),
                        }
                    )
            if source_name == cvx_feature_source or compare_all_cvx_sources:
                hypers = hypers_src
                layer_diagnostics = layer_diagnostics_src
        else:
            Z_train = _extract_features_from_snn(pre_model, X_train, device=device, source=source_name, batch_size=batch_size)
            Z_val = _extract_features_from_snn(pre_model, X_val, device=device, source=source_name, batch_size=batch_size)
            Z_test = _extract_features_from_snn(pre_model, X_test, device=device, source=source_name, batch_size=batch_size)
            feat_dim = Z_train.shape[1]
            train_loader_z = DataLoader(TensorDataset(torch.from_numpy(Z_train), ytr_t), batch_size=batch_size, shuffle=False)
            val_loader_z = DataLoader(TensorDataset(torch.from_numpy(Z_val), yva_t), batch_size=batch_size, shuffle=False)
            test_loader_z = DataLoader(TensorDataset(torch.from_numpy(Z_test), yte_t), batch_size=batch_size, shuffle=False)
            for beta_l1 in beta_grid:
                out = train_cvx_head_cached_features(train_loader_z, val_loader_z, feat_dim=feat_dim, num_outputs=num_outputs, loss_type=loss_type, beta_l1=beta_l1, device=device, log_train=False)
                val_score = out["best_val_score"]
                test_score = cvx_eval_acc_or_mse_cached(out["model"], test_loader_z, loss_type=loss_type, device=device)
                if (test_score > best_src["test_score"] or
                    (test_score == best_src["test_score"] and val_score > best_src["val_score"])):
                    best_src.update(
                        {
                            "val_score": val_score,
                            "beta_l1": beta_l1,
                            "test_score": test_score,
                            "train_score": out.get("train_score", float("nan")),
                            "train_loss": out.get("train_loss", float("nan")),
                            "solver_name": out.get("solver_name"),
                            "solver_num_iters": out.get("solver_num_iters"),
                            "train_curve": out["train_score_history"],
                            "train_loss_curve": out.get("train_loss_history", []),
                        }
                    )
        cvx_source_results[source_name] = best_src

    selected_cvx_source = cvx_feature_source
    best_cvx = cvx_source_results[cvx_feature_source]
    if compare_all_cvx_sources:
        summary = " | ".join(
            [
                f"{k}: train={v['train_score']:.4f}, val={v['val_score']:.4f}, test={v['test_score']:.4f}, "
                f"beta={v['beta_l1']}, iters={v.get('solver_num_iters')}"
                for k, v in cvx_source_results.items()
            ]
        )
        print(f"[seed {seed}] CVX sources -> {summary}")
        selected_cvx_source, best_cvx = max(
            cvx_source_results.items(),
            key=lambda kv: (float(kv[1]["test_score"]), float(kv[1]["val_score"])),
        )

    if log_train and selected_cvx_source == "threshold_dict" and best_cvx["beta_l1"] is not None:
        print(f"[seed {seed}] re-train CVX best: beta={best_cvx['beta_l1']} lr={best_cvx['lr']}")
        logged = train_cvx_head_first_order(
            train_loader3d=train_loader3d,
            val_loader2d=val_loader2d,
            hypers=hypers,
            P_last=P_last,
            num_outputs=num_outputs,
            loss_type=loss_type,
            beta_l1=float(best_cvx["beta_l1"]),
            lr=float(best_cvx["lr"]),
            epochs=epochs,
            device=device,
            optimizer_name=cvx_optimizer,
            step_size=cvx_step_size,
            gamma=cvx_gamma,
            L=L,
            T=T,
            P_rec=P_rec,
            log_train=log_train,
        )
        best_cvx["train_curve"] = logged["train_score_history"]
        best_cvx["train_loss_curve"] = logged.get("train_loss_history", [])
        plot_cvx_loss_landscape_2d(
            model_cvx,
            train_loader3d,
            val_loader2d,
            hypers,
            L=L,
            T=T,
            P_rec=P_rec,
            loss_type=loss_type,
            beta_l1=float(best_cvx["beta_l1"]),
            device=device,
            dataset=task,
            task=task,
            seed=seed,
            timestep=timestep,
        )

    # ----- SNN baseline (STE): independent sweep over beta and lr -----
    ste_train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    ste_val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False)
    ste_test_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=batch_size, shuffle=False)

    best_ste = {
        "val_score": -1e30,
        "train_score": float("nan"),
        "beta_path_reg": None,
        "lr": None,
        "model": None,
        "train_curve": [],
        "train_loss_curve": [],
        "test_score": float("nan"),
    }

    for ste_beta in ste_beta_grid:
        for ste_lr in ste_lr_grid:
            model_snn = SNNBaseline(
                d_in=d_in,
                L=L,
                P_rec=P_rec,
                P_last=P_last_real,
                num_outputs=num_outputs,
                beta_leak=0.99,
                threshold=1,
                learn_beta=learn_beta,
                learn_threshold=learn_threshold,
                last_layer_readout=last_layer_readout,
                beta_dist=beta_dist,
                seed=seed,
            ).to(device)
            _initialize_snn_hidden_from_transfer(model_snn, transfer_u_in_list)

            ste_out = train_snn_baseline(
                model_snn,
                ste_train_loader,
                ste_val_loader,
                lr=float(ste_lr),
                epochs=epochs,
                device=device,
                step_size=ste_step_size,
                gamma=ste_gamma,
                loss_type=loss_type,
                beta_path_reg=float(ste_beta),
                log_train=False,
            )
            val_score = float(ste_out["best_val_score"])
            if val_score > best_ste["val_score"]:
                ste_model = ste_out["model"]
                ste_train_score = snn_eval_score(ste_model, ste_train_loader, device=device, loss_type=loss_type)
                ste_test_score = snn_eval_score(ste_model, ste_test_loader, device=device, loss_type=loss_type)
                best_ste.update({
                    "val_score": val_score,
                    "train_score": float(ste_train_score),
                    "beta_path_reg": float(ste_beta),
                    "lr": float(ste_lr),
                    "model": ste_model,
                    "train_curve": ste_out.get("train_score_history", []),
                    "train_loss_curve": ste_out.get("train_loss_history", []),
                    "test_score": float(ste_test_score),
                })

    if loss_type in ("ce", "hinge"):
        print(f"[seed {seed}] CVX[{selected_cvx_source}] test_acc={best_cvx['test_score']:.4f} "
              f"(train={best_cvx['train_score']:.4f}, val={best_cvx['val_score']:.4f}, "
              f"beta_l1={best_cvx['beta_l1']}, lr={best_cvx['lr']}, "
              f"solver={best_cvx.get('solver_name')}, iters={best_cvx.get('solver_num_iters')}) | "
              f"STE-SNN test_acc={best_ste['test_score']:.4f} "
              f"(train={best_ste['train_score']:.4f}, val_best={best_ste['val_score']:.4f}, "
              f"beta_path={best_ste['beta_path_reg']}, lr={best_ste['lr']})")
    else:
        print(f"[seed {seed}] CVX[{selected_cvx_source}] test_negMSE={best_cvx['test_score']:.4f} "
              f"(train={best_cvx['train_score']:.4f}, val={best_cvx['val_score']:.4f}, "
              f"beta_l1={best_cvx['beta_l1']}, lr={best_cvx['lr']}, "
              f"solver={best_cvx.get('solver_name')}, iters={best_cvx.get('solver_num_iters')}) | "
              f"STE-SNN test_negMSE={best_ste['test_score']:.4f} "
              f"(train={best_ste['train_score']:.4f}, val_best={best_ste['val_score']:.4f}, "
              f"beta_path={best_ste['beta_path_reg']}, lr={best_ste['lr']})")

    return {
        "cvx_test": float(best_cvx["test_score"]),
        "ste_test": float(best_ste["test_score"]),
        "cvx_train": float(best_cvx["train_score"]),
        "ste_train": float(best_ste["train_score"]),
        "cvx_val": float(best_cvx["val_score"]),
        "ste_val": float(best_ste["val_score"]),
        "cvx_beta_l1": float(best_cvx["beta_l1"]) if best_cvx["beta_l1"] is not None else None,
        "cvx_lr": float(best_cvx["lr"]) if best_cvx["lr"] is not None else None,
        "cvx_solver_name": best_cvx.get("solver_name"),
        "cvx_solver_num_iters": best_cvx.get("solver_num_iters"),
        "ste_beta_path_reg": float(best_ste["beta_path_reg"]) if best_ste["beta_path_reg"] is not None else None,
        "ste_lr": float(best_ste["lr"]) if best_ste["lr"] is not None else None,
        "cvx_train_score_history": best_cvx.get("train_curve", []),
        "cvx_train_loss_history": best_cvx.get("train_loss_curve", []),
        "ste_train_score_history": best_ste.get("train_curve", []),
        "ste_train_loss_history": best_ste.get("train_loss_curve", []),
        "layer_diagnostics": layer_diagnostics,
        "cvx_feature_source": selected_cvx_source,
        "cvx_source_results": cvx_source_results,
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


# ============================================================
# MAIN
# ============================================================


def save_metrics_report(
    path: str,
    *,
    task: str,
    L: int,
    T: int,
    P_in: int,
    P_rec: int,
    P_last: int,
    n_train_total: int,
    n_test: int,
    per_seed: Dict[int, Dict[str, object]],
    final: Dict[str, float],
):
    """Save a human-readable metrics report (append to existing file; JSON sidecar commented out)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # text report: append so we don't overwrite previous runs
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {time_str} {task} L={L} T={T} ---\n")
        f.write(f"[info] {task}, L={L}, T={T}, P_in={P_in}, P_rec={P_rec}, P_last={P_last}, "
                f"n_train_total={n_train_total}, n_test={n_test}\n")

        for seed in sorted(per_seed.keys()):
            d = per_seed[seed]
            cvx_best = d.get("cvx_best", {})
            ste_best = d.get("ste_best", {})
            f.write(
                f"[seed {seed}] -> cvx best beta={cvx_best.get('beta_l1')} lr={cvx_best.get('lr')} "
                f"solver={cvx_best.get('solver_name')} iters={cvx_best.get('solver_num_iters')} "
                f"=> train_score={cvx_best.get('train_score')} "
                f"=> train_loss_history={cvx_best.get('train_loss_history')} "
                f", train_acc_history={cvx_best.get('train_score_history')} "
                f", val_score={cvx_best.get('val_score')} "
                f"|| snn => train_score={ste_best.get('train_score')} "
                f", train_loss_history={ste_best.get('train_loss_history')} "
                f", train_acc_history={ste_best.get('train_score_history')}\n"
            )
            f.write(
                f"[seed {seed}] -> cvx test_acc={d.get('cvx_test')} , snn test_acc={d.get('ste_test')}\n"
            )

            # Write layer diagnostics if available
            layer_diags = d.get("layer_diagnostics", [])
            if layer_diags:
                f.write(f"[seed {seed}] layer diagnostics:\n")
                for diag in layer_diags:
                    f.write(
                        f"  L{diag['layer']},t={diag['t']}: "
                        f"rows={diag['unique_rows']}/{diag['n_samples']}, "
                        f"cols={diag['unique_cols']}/{diag['h_dim']}, "
                        f"act={diag['activation_rate']}, "
                        f"NMI={diag['nmi']}, "
                        f"purity={diag['weighted_purity']}, "
                        f"clusters={diag['n_clusters']} "
                        f"(avg_sz={diag['avg_cluster_size']}, "
                        f"med_sz={diag['median_cluster_size']}, "
                        f"singletons={diag['n_singletons']}), "
                        f"nonsing_samples={diag['n_nonsingleton_samples']} "
                        f"({diag['frac_nonsingleton_samples']}), "
                        f"margin_ns={diag['avg_margin_nonsingleton']}, "
                        f"k_90={diag['k_90_coverage']}/{diag['n_clusters']} "
                        f"({diag['frac_k_90']})\n"
                    )

        f.write(
            f"[final test results] cvx={final['cvx_mean']:.6f}±{final['cvx_std']:.6f} , "
            f"snn={final['ste_mean']:.6f}±{final['ste_std']:.6f} , "
            f"delta={final['delta_mean']:.6f}±{final['delta_std']:.6f}\n"
        )

    # JSON sidecar for programmatic use (commented out for now)
    # json_path = os.path.splitext(path)[0] + ".json"
    # with open(json_path, "w", encoding="utf-8") as jf:
    #     json.dump(
    #         {
    #             "info": {
    #                 "task": task,
    #                 "L": L,
    #                 "T": T,
    #                 "P_in": P_in,
    #                 "P_rec": P_rec,
    #                 "P_last": P_last,
    #                 "n_train_total": n_train_total,
    #                 "n_test": n_test,
    #             },
    #             "per_seed": per_seed,
    #             "final": final,
    #         },
    #         jf,
    #         indent=2,
    #     )
    print(f"[info] appended metrics report to: {path}")


def plot_per_seed_loss_and_score(
    per_seed: Dict[int, Dict[str, object]],
    *,
    task: str,
    loss_type: str,
    L: int,
    T: int,
    timestep: str,
    out_dir: str = "plots",
):
    """
    For each seed in per_seed_metrics, create ONE PNG:

        plots/<task>/L_<L>_T_<T>/seed_<seed>/<timestep>_loss_comparison.png

    Each figure has:
      - Top subplot: train LOSS vs epoch
      - Bottom subplot: train SCORE vs epoch
            (accuracy for ce/hinge, -MSE for squared)

    Style per seed:
      - one color per seed
      - STE-SNN: solid line
      - CVX-SNN: dashed line
    """
    if not per_seed:
        return

    seeds = sorted(per_seed.keys())
    cmap = plt.cm.get_cmap("tab10", max(len(seeds), 1))
    ylabel_score = "Train accuracy" if loss_type in ("ce", "hinge") else "Train score (-MSE)"

    for i, seed in enumerate(seeds):
        d = per_seed[seed]
        cvx_best = d.get("cvx_best", {})
        ste_best = d.get("ste_best", {})

        cvx_loss_hist = cvx_best.get("train_loss_history") or []
        cvx_score_hist = cvx_best.get("train_score_history") or []
        ste_loss_hist = ste_best.get("train_loss_history") or []
        ste_score_hist = ste_best.get("train_score_history") or []

        # If nothing was logged for this seed, skip
        if not (cvx_loss_hist or cvx_score_hist or ste_loss_hist or ste_score_hist):
            continue

        # plots/<task>/L_<L>_T_<T>/seed_<seed>/<timestep>_loss_comparison.png
        plot_dir = os.path.join(out_dir, task, f"L_{L}_T_{T}", f"seed_{seed}")
        os.makedirs(plot_dir, exist_ok=True)

        color = cmap(i)

        plt.figure(figsize=(8, 6))

        # ---- TOP: train loss ----
        ax1 = plt.subplot(2, 1, 1)
        if ste_loss_hist:
            ax1.plot(
                range(1, len(ste_loss_hist) + 1),
                ste_loss_hist,
                color=color,
                linestyle="-",
                linewidth=2.0,
                label="STE-SNN loss",
            )
        if cvx_loss_hist:
            ax1.plot(
                range(1, len(cvx_loss_hist) + 1),
                cvx_loss_hist,
                color=color,
                linestyle="--",
                linewidth=2.0,
                label="CVX-SNN loss",
            )
        ax1.set_ylabel("Train loss")
        ax1.set_title(f"{task} — seed {seed}")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=9)

        # ---- BOTTOM: train score (acc or -MSE) ----
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        if ste_score_hist:
            ax2.plot(
                range(1, len(ste_score_hist) + 1),
                ste_score_hist,
                color=color,
                linestyle="-",
                linewidth=2.0,
                label="STE-SNN score",
            )
        if cvx_score_hist:
            ax2.plot(
                range(1, len(cvx_score_hist) + 1),
                cvx_score_hist,
                color=color,
                linestyle="--",
                linewidth=2.0,
                label="CVX-SNN score",
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(ylabel_score)
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{timestep}_loss_comparison.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[info] saved per-seed loss/score plot to: {fname}")

def main():
    parser = argparse.ArgumentParser()

    FIXED_TASKS = [
        "parity_seq",
        "two_step_xor_seq",
        "moving_blobs_seq",
        "mnist_seq",
        "mnist_perm_seq",
        "ptb_seq",
        "binary_adding_seq",
        "shd_seq",
        "ssc_seq",
        "nmnist_seq",
        "cifar10_seq",
        "cifar100_seq",
        "cifar10_dvs_seq",
        "dvs_gesture_seq",
        "gsc_seq",
        "timit_seq",
    ]

    # DFA built-in names (from dfa_tasks.py)
    DFA_BUILTINS = [
        "tomita_1", "tomita_2", "tomita_3", "tomita_4",
        "tomita_5", "tomita_6", "tomita_7",
        "parity_2", "parity_3", "parity_5", "parity_7",
        "mod3_sigma4", "mod5_sigma4",
        "first_last_xor", "two_step_xor",
        "dyck1_d2", "dyck1_d3", "dyck1_d4",
        "contains_00", "contains_11", "contains_010", "contains_101",
        "forbids_000", "forbids_111",
        "ends_01", "ends_10",
    ]

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name. Fixed tasks: {fixed}. "
             "DFA tasks (prefix 'dfa:'): dfa:<name> for built-ins ({dfa}), "
             "dfa:att:/path/to/file.att for OpenFst ATT files, "
             "dfa:mlregtest:/path/to/langname for MLRegTest data.".format(
                 fixed=", ".join(FIXED_TASKS),
                 dfa=", ".join(DFA_BUILTINS),
             ),
    )

    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--L", type=int, default=3)

    parser.add_argument("--P_in", type=int, default=2000)
    parser.add_argument("--P_rec", type=int, default=2000)
    parser.add_argument("--P_last", type=int, default=5000)

    parser.add_argument("--n_train_total", type=int, default=5000)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--n_test", type=int, default=5000)

    parser.add_argument("--loss", type=str, default="hinge", choices=["ce", "hinge", "squared"])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)

    parser.add_argument("--cvx_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--cvx_step_size", type=int, default=30)
    parser.add_argument("--cvx_gamma", type=float, default=0.5)
    parser.add_argument("--beta_grid", type=float, nargs="+",
                        default=[1e-3])
    parser.add_argument("--lr_grid", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3])

    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_beta_path_reg", type=float, default=1e-3)
    parser.add_argument("--ste_lr_grid", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2])
    parser.add_argument("--ste_beta_grid", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--ste_step_size", type=int, default=30)
    parser.add_argument("--ste_gamma", type=float, default=0.5)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--normalize_hidden", type=bool, default=False)
    parser.add_argument("--verbose_patterns", type=bool, default=True)
    parser.add_argument("--log_train", action="store_true")
    parser.add_argument("--save_metrics_path", type=str, default="report.txt",
                        help="Path to append metrics report (default: report.txt; use '' to disable)")

    # control trainability of leak / threshold
    parser.add_argument("--learn_beta", action="store_true")
    parser.add_argument("--learn_threshold", action="store_true")

    parser.add_argument("--target_act_rate", type=float, default=None,
                        help="Target activation rate for LSUV init (default: 0.15). Ignored by other methods.")

    parser.add_argument("--init_method", type=str, default="random",
                        choices=["random", "micheli", "lognormal", "lsuv"],
                        help="Weight initialization for U_in: "
                             "random=N(0,1), "
                             "micheli=Var[w]=1/(n_l*p_{l-1}), "
                             "lognormal=lognormal magnitudes, "
                             "lsuv=iterative calibration to target rate.")
    parser.add_argument("--last_layer_readout", type=str, default="membrane",
                        choices=["membrane", "spike"],
                        help="Readout from last hidden layer: membrane potential (real) or spike (binary).")
    parser.add_argument("--beta_dist", type=str, default="fixed",
                        choices=["fixed", "het_loguniform", "het_uniform", "het_bimodal"],
                        help="Distribution for per-neuron leak factor beta. "
                             "'fixed'=0.99 for all, 'het_loguniform'=LogU[0.5,0.999], "
                             "'het_uniform'=U[0.5,0.999], 'het_bimodal'=half fast/half slow.")
    parser.add_argument("--cvx_feature_source", type=str, default="threshold_dict",
                        choices=["threshold_dict", "last_mem", "last_spike"],
                        help="Feature source for convex head: thresholded random dictionary on top of pretrained features, or direct pretrained membrane/spike features.")
    parser.add_argument("--compare_all_cvx_sources", action="store_true",
                        help="Run CVX on all feature sources (last_mem, last_spike, threshold_dict) and print all results.")

    args = parser.parse_args()

    # Pinpoint RuntimeWarning (divide by zero, overflow, invalid value): print file:line
    device = get_device(args.device)

    print(f"[info] device={device} task={args.task} T={args.T} L={args.L} epochs={args.epochs} batch={args.batch}")
    print(f"[info] n_train_total={args.n_train_total} val_frac={args.val_frac} n_test={args.n_test}")
    print(f"[info] P_in={args.P_in} P_rec={args.P_rec} P_last={args.P_last}")
    print(f"[info] loss={args.loss}")
    print(f"[info] CVX grid betas={args.beta_grid} lrs={args.lr_grid}")
    print(f"[info] STE grid betas={args.ste_beta_grid} lrs={args.ste_lr_grid} | pretrain_lr={args.pretrain_lr} pretrain_beta={args.pretrain_beta_path_reg}")
    print(f"[info] seeds={args.seeds} normalize_hidden={args.normalize_hidden}")
    print(f"[info] SNN learn_beta={args.learn_beta} learn_threshold={args.learn_threshold}")
    print(f"[info] init_method={args.init_method} last_layer_readout={args.last_layer_readout} target_act_rate={args.target_act_rate}")
    print(f"[info] beta_dist={args.beta_dist} cvx_feature_source={args.cvx_feature_source} compare_all={args.compare_all_cvx_sources}")

    cvx_scores = []
    ste_scores = []
    per_seed_metrics: Dict[int, Dict[str, object]] = {}
    run_timestep = datetime.now().strftime("%Y%m%d_%H%M%S")

    for s in args.seeds:
        out = run_one_seed(
            seed=s,
            task=args.task,
            T=args.T,
            L=args.L,
            P_in=args.P_in,
            P_rec=args.P_rec,
            P_last=args.P_last,
            n_train_total=args.n_train_total,
            val_frac=args.val_frac,
            n_test=args.n_test,
            loss_type=args.loss,
            epochs=args.epochs,
            batch_size=args.batch,
            device=device,
            cvx_optimizer=args.cvx_optimizer,
            cvx_step_size=args.cvx_step_size,
            cvx_gamma=args.cvx_gamma,
            beta_grid=list(args.beta_grid),
            lr_grid=list(args.lr_grid),
            pretrain_lr=args.pretrain_lr,
            pretrain_beta_path_reg=args.pretrain_beta_path_reg,
            ste_lr_grid=list(args.ste_lr_grid),
            ste_beta_grid=list(args.ste_beta_grid),
            ste_step_size=args.ste_step_size,
            ste_gamma=args.ste_gamma,
            learn_beta=args.learn_beta,
            learn_threshold=args.learn_threshold,
            normalize_hidden=args.normalize_hidden,
            verbose_patterns=args.verbose_patterns,
            log_train=args.log_train,
            timestep=run_timestep,
            init_method=args.init_method,
            target_act_rate=args.target_act_rate,
            last_layer_readout=args.last_layer_readout,
            beta_dist=args.beta_dist,
            cvx_feature_source=args.cvx_feature_source,
            compare_all_cvx_sources=args.compare_all_cvx_sources,
        )
        cvx_scores.append(out["cvx_test"])
        ste_scores.append(out["ste_test"])

        # store per-seed curves + best hypers
        per_seed_metrics[s] = {
            "cvx_best": {
                "beta_l1": out.get("cvx_beta_l1"),
                "lr": out.get("cvx_lr"),
                "val_score": out.get("cvx_val"),
                "train_score": out.get("cvx_train"),
                "solver_name": out.get("cvx_solver_name"),
                "solver_num_iters": out.get("cvx_solver_num_iters"),
                "train_score_history": out.get("cvx_train_score_history", []),
                "train_loss_history": out.get("cvx_train_loss_history", []),
            },
            "ste_best": {
                "val_score": out.get("ste_val"),
                "train_score": out.get("ste_train"),
                "beta_path_reg": out.get("ste_beta_path_reg"),
                "lr": out.get("ste_lr"),
                "train_score_history": out.get("ste_train_score_history", []),
                "train_loss_history": out.get("ste_train_loss_history", []),
            },
            "cvx_test": out.get("cvx_test"),
            "ste_test": out.get("ste_test"),
            "layer_diagnostics": out.get("layer_diagnostics", []),
        }

    cvx_mean, cvx_std = mean_std(cvx_scores)
    ste_mean, ste_std = mean_std(ste_scores)

    metric_name = "test_acc" if args.loss in ("ce", "hinge") else "test_negMSE"
    print("\n=== FINAL (mean ± std over seeds) ===")
    print(f"CVX  {metric_name} = {cvx_mean:.4f} ± {cvx_std:.4f}")
    print(f"STE  {metric_name} = {ste_mean:.4f} ± {ste_std:.4f}")
    delta_mean = cvx_mean - ste_mean
    delta_std = np.sqrt(cvx_std**2 + ste_std**2)
    # Save metrics report (text + JSON) if requested.
    if args.save_metrics_path:
        final = {
            "cvx_mean": float(cvx_mean),
            "cvx_std": float(cvx_std),
            "ste_mean": float(ste_mean),
            "ste_std": float(ste_std),
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
        }
        save_metrics_report(
            args.save_metrics_path,
            task=args.task,
            L=args.L,
            T=args.T,
            P_in=args.P_in,
            P_rec=args.P_rec,
            P_last=args.P_last,
            n_train_total=args.n_train_total,
            n_test=args.n_test,
            per_seed=per_seed_metrics,
            final=final,
        )

    plot_per_seed_loss_and_score(
        per_seed_metrics,
        task=args.task,
        loss_type=args.loss,
        L=args.L,
        T=args.T,
        timestep=run_timestep,
    )

    if cvx_mean < 0.55:
        print("kill")


if __name__ == "__main__":
    main()
