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
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    X = rng.integers(0, 2, size=(n, T, 1)).astype(np.float32)
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
    U_in_list: List[np.ndarray]   # per layer: (d_in_l, P_in)
    U_rec_list: List[np.ndarray]  # per layer: (P_rec_l-1, P_rec)
    U_last: np.ndarray            # (P_rec_last, P_last)

def generate_snn_sign_patterns(
    X_seq: np.ndarray,   # (n, T, d_in)
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
) -> Tuple[np.ndarray, RNNHyperplanes]:
    """
    Generate sign patterns for a threshold-SNN with L hidden layers and T timesteps.

    This GPU-optimized version:

      * Uses torch on `device` for all recurrence and last-layer matmuls.
      * Keeps only hyperplane storage + unique-pattern bookkeeping on CPU (NumPy).

    Returns:
      z_last_bool: (n, P_last_target) bool
      hypers:      RNNHyperplanes(U_in_list, U_rec_list, U_last)
                   where U_rec_list[l-1] has shape (2*h_dim_l+1, h_dim_l).
    """
    rng = np.random.default_rng(seed)
    n, T_data, d_in = X_seq.shape
    if T_data != T:
        raise ValueError(f"X_seq has T={T_data} but you requested T={T}.")

    if device is None:
        device = torch.device("cpu")

    if verbose:
        print(f"[snn patterns] using device={device}")

    # layer 0 "spikes" are just the inputs x_t (real-valued); we don't threshold them
    # h_layers[l][t] holds activations for layer l at time t, except layer 0 which holds raw inputs
    # Use float64 for recurrence to reduce overflow risk for large P_rec / P_last_target.
    X_torch = torch.from_numpy(X_seq.astype(np.float64, copy=False)).to(device)

    h_layers: List[List[torch.Tensor]] = []
    h0 = [X_torch[:, t, :] for t in range(T)]  # each (n, d_in)
    h_layers.append(h0)

    U_in_list: List[np.ndarray] = []
    U_rec_list: List[np.ndarray] = []

    # Decide hidden widths: first L-2 layers = P_rec, last hidden layer = P_last_target
    if L <= 1:
        hidden_dims = [P_last_target]
    else:
        hidden_dims = [P_rec] * max(L - 2, 0) + [P_last_target]

    d_in_l = d_in

    # -------------------------------
    # Hidden layers 1 .. L-1 (GPU)
    # -------------------------------
    for l in range(1, L):
        h_dim = hidden_dims[l - 1]  # width of this hidden layer

        # Input hyperplanes: from previous layer's h^{l-1,t} (dim = d_in_l) to h_dim neurons
        U_in_np = rng.normal(size=(d_in_l, h_dim)).astype(np.float32)  # (d_in_l, h_dim)

        # Recurrent hyperplanes: from [v_{t-1}, -h_{t-1}, -1] (dim = 2*h_dim+1) to h_dim
        b = np.full((h_dim,), 0.99, dtype=np.float32)  # Decay rate
        g = np.full((h_dim,), 1.0, dtype=np.float32)   # Threshold

        U_rec_top = np.diag(b)   # (h_dim, h_dim)
        U_rec_bot = np.diag(g)   # (h_dim, h_dim)
        U_rec_np = np.vstack([U_rec_top, U_rec_bot, g]).astype(np.float32)  # (2*h_dim+1, h_dim)

        if normalize_hidden:
            U_in_np = _col_normalize_np(U_in_np)
            U_rec_np = _col_normalize_np(U_rec_np)

        if verbose:
            print(
                f"[snn hypers] layer {l}: "
                f"U_in={U_in_np.shape}, U_rec={U_rec_np.shape} (shared across all T)"
            )

        # Store CPU copies in hypers
        U_in_list.append(U_in_np)
        U_rec_list.append(U_rec_np)

        # Torch views on GPU for recurrence
        U_in = torch.from_numpy(U_in_np.astype(np.float64, copy=False)).to(device)
        U_rec = torch.from_numpy(U_rec_np.astype(np.float64, copy=False)).to(device)

        v_prev = torch.zeros(n, h_dim, dtype=torch.float64, device=device)
        h_prev = torch.zeros(n, h_dim, dtype=torch.float64, device=device)
        h_curr_list: List[torch.Tensor] = []

        for t in range(T):
            # input to this layer at time t
            x_t = h_layers[l - 1][t]  # (n, d_in_l)

            # input contribution
            v_in_t = x_t @ U_in  # (n, h_dim)

            # recurrent contribution from previous membrane + spikes: [v_prev, -h_prev, -1]
            s_prev = torch.cat(
                [
                    v_prev,
                    -h_prev,
                    -torch.ones((n, 1), dtype=torch.float64, device=device),
                ],
                dim=1,
            )  # (n, 2*h_dim + 1)
            v_rec_t = s_prev @ U_rec  # (n, h_dim)

            # new membrane and spikes
            v_t = v_in_t + v_rec_t
            h_t = (v_t >= 0.0).to(torch.float64)

            h_curr_list.append(h_t)
            v_prev = v_t
            h_prev = h_t

        h_layers.append(h_curr_list)
        d_in_l = h_dim  # next layer's input dimension

    # Last hidden state h^{L-1, T-1} is used for the convex last layer
    h_last_T = h_layers[-1][-1]  # (n, h_dim_last) where h_dim_last = hidden_dims[-1]
    in_dim_last = h_last_T.shape[1]

    # -------------------------------------------------
    # Last-layer hyperplanes + unique patterns (GPU)
    # -------------------------------------------------
    uniq: Dict[bytes, np.ndarray] = {}
    rounds = 0
    chunk_P = max(P_last_target * chunk_mult, P_last_target)

    while len(uniq) < P_last_target and rounds < max_rounds:
        rounds += 1
        U_last_np = rng.normal(size=(in_dim_last, chunk_P)).astype(np.float32)
        if normalize_hidden:
            U_last_np = _col_normalize_np(U_last_np)

        U_last = torch.from_numpy(U_last_np.astype(np.float64, copy=False)).to(device)

        # sign patterns on device
        D_bool_t = (h_last_T @ U_last >= 0.0)    # (n, chunk_P) bool
        D_u8 = D_bool_t.to(torch.uint8).cpu().numpy()  # move only patterns to CPU

        for j in range(chunk_P):
            key = D_u8[:, j].tobytes()
            if key not in uniq:
                uniq[key] = U_last_np[:, j].copy()
                if len(uniq) >= P_last_target:
                    break

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
        U_rec_list=U_rec_list,   # each (2*h_dim_l+1, h_dim_l)
        U_last=U_last_final,     # (h_dim_last, P_last_target)
    )
    return z_last_bool, hypers

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

    # layer 0: raw inputs
    h_prev_layers: List[List[torch.Tensor]] = []
    h0 = [X_seq[:, t, :].to(device) for t in range(T)]
    h_prev_layers.append(h0)

    d_in_l = d_in

    for l in range(1, L):
        U_in_np = hypers.U_in_list[l - 1]   # (d_in_l, h_dim_l)
        U_rec_np = hypers.U_rec_list[l - 1] # (2*h_dim_l+1, h_dim_l)

        U_in = torch.from_numpy(U_in_np).float().to(device)
        U_rec = torch.from_numpy(U_rec_np).float().to(device)

        h_dim = U_in.shape[1]

        v_prev = torch.zeros(B, h_dim, device=device)
        h_prev = torch.zeros(B, h_dim, device=device)
        h_curr_list: List[torch.Tensor] = []

        for t in range(T):
            x_t = h_prev_layers[l - 1][t]  # (B, d_in_l)
            v_in_t = x_t @ U_in            # (B, h_dim)

            ones = torch.ones(B, 1, device=device)
            s_prev = torch.cat([v_prev, -h_prev, -ones], dim=1)  # (B, 2*h_dim+1)
            v_rec_t = s_prev @ U_rec                             # (B, h_dim)

            v_t = v_in_t + v_rec_t
            v_t = torch.clamp(v_t, -1e10, 1e10)
            h_t = (v_t >= 0.0).float()

            h_curr_list.append(h_t)
            v_prev, h_prev = v_t, h_t

        h_prev_layers.append(h_curr_list)
        d_in_l = h_dim

    # last hidden state has width = h_dim_last (which is P_last after your change)
    h_last_T = h_prev_layers[-1][-1]  # (B, h_dim_last)

    U_last = torch.from_numpy(hypers.U_last).float().to(device)  # (h_dim_last, P_last_target)
    D_last = (h_last_T @ U_last >= 0.0).float()                  # (B, P_last_target)

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
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("plots", exist_ok=True)
    fname = f"plots/losslandscape1D_L{L}_T{T}_{dataset}_{time_str}.png"
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
    alpha_range: float = 5.0,
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

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("plots", exist_ok=True)
    base = f"plots/losslandscape2D_L{L}_T{T}_{dataset}_{time_str}"

    # ---------- Train contour (as before) ----------
    plt.figure(figsize=(6, 5))
    plt.contourf(A, B, train_grid.T, levels=30)
    plt.colorbar(label="Train CVX Objective (cvx_loss)")
    plt.xlabel("α (direction d₁)")
    plt.ylabel("β (direction d₂)")
    plt.title("CVX-SNN Train Loss Landscape (2D Slice)")
    plt.tight_layout()
    plt.savefig(f"{base}_train.png", bbox_inches="tight", dpi=150)
    plt.close()

    # ---------- Test contour (as before) ----------
    plt.figure(figsize=(6, 5))
    plt.contourf(A, B, test_grid.T, levels=30)
    plt.colorbar(label="Test Loss (1-acc / MSE)")
    plt.xlabel("α (direction d₁)")
    plt.ylabel("β (direction d₂)")
    plt.title("CVX-SNN Test Loss Landscape (2D Slice)")
    plt.tight_layout()
    plt.savefig(f"{base}_test.png", bbox_inches="tight", dpi=150)
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
    plt.savefig(f"{base}_overlay.png", bbox_inches="tight", dpi=150)
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

    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val_score = -1e30
    best_state = None
    train_acc_history: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, zb in train_loader3d:
            xb = xb.to(device)
            yb = yb.to(device)
            zb = zb.to(device)
            logits = model(zb)
            loss = cvx_loss(logits, yb, model, loss_type=loss_type, beta_l1=beta_l1)
            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()

        train_score = cvx_eval_acc_or_mse(
            model, train_loader2d=None,  # we'll re-use patterns via train_loader3d
            hypers=hypers, L=L, T=T, P_rec=P_rec, device=device, loss_type=loss_type
        ) if False else None  # optional; we log using cached z below

        # More efficient: accuracy over cached z in train_loader3d
        if loss_type in ("ce", "hinge"):
            correct, total = 0, 0
            model.eval()
            with torch.no_grad():
                for _, yb, zb in train_loader3d:
                    yb = yb.to(device)
                    zb = zb.to(device)
                    logits = model(zb)
                    if loss_type == "ce":
                        preds = logits.argmax(dim=1)
                        correct += (preds == yb).sum().item()
                    else:
                        preds = torch.where(logits.squeeze(-1) >= 0, 1.0, -1.0)
                        correct += (preds == yb.float()).sum().item()
                    total += yb.numel()
            train_score = correct / max(total, 1)
        else:
            # squared regression: track negative MSE on train cached z
            mse_sum, tot = 0.0, 0
            model.eval()
            with torch.no_grad():
                for _, yb, zb in train_loader3d:
                    yb = yb.to(device)
                    zb = zb.to(device)
                    logits = model(zb).squeeze(-1)
                    mse_sum += F.mse_loss(logits, yb.float(), reduction="sum").item()
                    tot += yb.numel()
            train_score = -mse_sum / max(tot, 1)

        if log_train:
            train_acc_history.append(train_score)

        val_score = cvx_eval_acc_or_mse(
            model, val_loader2d, hypers, L=L, T=T, P_rec=P_rec, device=device, loss_type=loss_type
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if log_train:
            print(f"[CVX] ep={ep:03d}/{epochs} train_score={train_score:.4f} val_score={val_score:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_score": best_val_score,
        "train_score_history": train_acc_history,
    }


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
    ):
        super().__init__()
        self.L = L
        self.P_rec = P_rec
        self.P_last = P_last

        fcs = []
        lifs = []
        in_dim = d_in

        # hidden dims: first L-2 layers have width P_rec, last hidden layer has width P_last
        if L <= 1:
            hidden_dims = [P_last]
        else:
            hidden_dims = [P_rec] * max(L - 2, 0) + [P_last]

        for h_dim in hidden_dims:
            fc = nn.Linear(in_dim, h_dim, bias=False)
            fcs.append(fc)
            # NOTE: adjust keyword args if your snntorch version differs
            lif = snn.Leaky(
                beta=beta_leak,
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
          - logits: (B, num_outputs) using last membrane of last layer
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
        logits = self.fc_out(h_last)
        return logits, last_mem


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

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = snn_baseline_loss(logits, yb, loss_type=loss_type)
            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()

        tr_score = snn_eval_score(model, train_loader, device=device, loss_type=loss_type)
        val_score = snn_eval_score(model, val_loader, device=device, loss_type=loss_type)

        if log_train:
            train_scores.append(tr_score)
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
    ste_lr: float,
    ste_step_size: int,
    ste_gamma: float,
    learn_beta: bool,
    learn_threshold: bool,
    normalize_hidden: bool,
    verbose_patterns: bool,
    log_train: bool,
) -> Dict[str, float]:
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

    # ----- CVX patterns (generated once) -----
    z_train_bool, hypers = generate_snn_sign_patterns(
        X_train,
        L=L,
        T=T,
        P_in=P_in,
        P_rec=P_rec,
        P_last_target=P_last,
        seed=seed,
        normalize_hidden=normalize_hidden,
        verbose=verbose_patterns,
    )
    z_train = z_train_bool.astype(np.uint8)
    P_last_real = z_train.shape[1]

    # build torch loaders
    Xtr_t = torch.from_numpy(X_train).float()
    ytr_t = torch.from_numpy(y_train)
    Xva_t = torch.from_numpy(X_val).float()
    yva_t = torch.from_numpy(y_val)
    Xte_t = torch.from_numpy(X_test).float()
    yte_t = torch.from_numpy(y_test)

    # For hinge, y should be ±1
    if loss_type == "hinge":
        ytr_t = (2 * ytr_t - 1).float()
        yva_t = (2 * yva_t - 1).float()
        yte_t = (2 * yte_t - 1).float()

    train_ds3d = TensorDataset(Xtr_t, ytr_t, torch.from_numpy(z_train))
    val_ds2d = TensorDataset(Xva_t, yva_t)
    test_ds2d = TensorDataset(Xte_t, yte_t)

    train_loader3d = DataLoader(train_ds3d, batch_size=batch_size, shuffle=True)
    val_loader2d = DataLoader(val_ds2d, batch_size=batch_size, shuffle=False)
    test_loader2d = DataLoader(test_ds2d, batch_size=batch_size, shuffle=False)

    # ----- CVX grid search over (beta_l1, lr) -----
    num_outputs = num_classes if loss_type == "ce" else 1
    best_cvx = {
        "val_score": -1e30,
        "beta_l1": None,
        "lr": None,
        "test_score": None,
        "train_curve": None,
    }

    for beta_l1 in beta_grid:
        for lr in lr_grid:
            out = train_cvx_head_first_order(
                train_loader3d,
                val_loader2d,
                hypers,
                P_last=P_last_real,
                num_outputs=num_outputs,
                loss_type=loss_type,
                beta_l1=beta_l1,
                lr=lr,
                epochs=epochs,
                device=device,
                optimizer_name=cvx_optimizer,
                step_size=cvx_step_size,
                gamma=cvx_gamma,
                L=L,
                T=T,
                P_rec=P_rec,
                log_train=False,
            )
            model_cvx = out["model"]
            val_score = out["best_val_score"]
            if val_score > best_cvx["val_score"]:
                test_score = cvx_eval_acc_or_mse(
                    model_cvx,
                    test_loader2d,
                    hypers,
                    L=L,
                    T=T,
                    P_rec=P_rec,
                    device=device,
                    loss_type=loss_type,
                )
                best_cvx.update({
                    "val_score": val_score,
                    "beta_l1": beta_l1,
                    "lr": lr,
                    "test_score": test_score,
                    "train_curve": out["train_score_history"],
                })

    if log_train:
        print(f"[seed {seed}] re-train CVX best for logging: beta={best_cvx['beta_l1']} lr={best_cvx['lr']}")
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
        best_cvx["model"] = logged["model"]
        best_cvx["val_score"] = logged["best_val_score"]
        best_cvx["train_curve"] = logged["train_score_history"]
        plot_cvx_loss_landscape_2d(
            model=best_cvx["model"],
            train_loader3d=train_loader3d,
            test_loader2d=test_loader2d,
            hypers=hypers,
            L=L,
            T=T,
            P_rec=P_rec,
            loss_type=loss_type,
            beta_l1=float(best_cvx["beta_l1"]),
            device=device,
            dataset=task
        )

        convLossLandscape1D(
            model=best_cvx["model"],
            train_loader3d=train_loader3d,
            test_loader2d=test_loader2d,
            hypers=hypers,
            L=L,
            T=T,
            P_rec=P_rec,
            loss_type=loss_type,
            beta_l1=float(best_cvx["beta_l1"]),
            device=device,
            dataset=task
        )
    # ----- SNN baseline (STE) -----
    d_in = X_train.shape[2]
    model_snn = SNNBaseline(
        d_in=d_in,
        L=L,
        P_rec=P_rec,
        num_outputs=num_outputs,
        beta_leak=0.99,
        threshold=1,
        learn_beta=learn_beta,
        learn_threshold=learn_threshold,
    ).to(device)

    ste_train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    ste_val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False)
    ste_test_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=batch_size, shuffle=False)

    ste_out = train_snn_baseline(
        model_snn,
        ste_train_loader,
        ste_val_loader,
        lr=ste_lr,
        epochs=epochs,
        device=device,
        step_size=ste_step_size,
        gamma=ste_gamma,
        loss_type=loss_type,
        log_train=True,
    )
    ste_model = ste_out["model"]
    ste_test_score = snn_eval_score(ste_model, ste_test_loader, device=device, loss_type=loss_type)

    if loss_type in ("ce", "hinge"):
        print(f"[seed {seed}] CVX test_acc={best_cvx['test_score']:.4f} "
              f"(val={best_cvx['val_score']:.4f}, beta_l1={best_cvx['beta_l1']}, lr={best_cvx['lr']}) | "
              f"STE-SNN test_acc={ste_test_score:.4f} (val_best={ste_out['best_val_score']:.4f})")
    else:
        print(f"[seed {seed}] CVX test_negMSE={best_cvx['test_score']:.4f} "
              f"(val={best_cvx['val_score']:.4f}, beta_l1={best_cvx['beta_l1']}, lr={best_cvx['lr']}) | "
              f"STE-SNN test_negMSE={ste_test_score:.4f} (val_best={ste_out['best_val_score']:.4f})")

    return {
        "cvx_test": float(best_cvx["test_score"]),
        "ste_test": float(ste_test_score),
        "cvx_val": float(best_cvx["val_score"]),
        "ste_val": float(ste_out["best_val_score"]),
        "cvx_beta_l1": float(best_cvx["beta_l1"]),
        "cvx_lr": float(best_cvx["lr"]),
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
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
        ],
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
                        default=[1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0])
    parser.add_argument("--lr_grid", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3])

    parser.add_argument("--ste_lr", type=float, default=1e-3)
    parser.add_argument("--ste_step_size", type=int, default=30)
    parser.add_argument("--ste_gamma", type=float, default=0.5)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])

    parser.add_argument("--normalize_hidden", type=bool, default=False)
    parser.add_argument("--verbose_patterns", type=bool, default=True)
    parser.add_argument("--log_train", action="store_true")

    # control trainability of leak / threshold
    parser.add_argument("--learn_beta", action="store_true")
    parser.add_argument("--learn_threshold", action="store_true")

    args = parser.parse_args()

    # Pinpoint RuntimeWarning (divide by zero, overflow, invalid value): print file:line
    device = get_device(args.device)

    print(f"[info] device={device} task={args.task} T={args.T} L={args.L} epochs={args.epochs} batch={args.batch}")
    print(f"[info] n_train_total={args.n_train_total} val_frac={args.val_frac} n_test={args.n_test}")
    print(f"[info] P_in={args.P_in} P_rec={args.P_rec} P_last={args.P_last}")
    print(f"[info] loss={args.loss}")
    print(f"[info] CVX grid betas={args.beta_grid} lrs={args.lr_grid}")
    print(f"[info] seeds={args.seeds} normalize_hidden={args.normalize_hidden}")
    print(f"[info] SNN learn_beta={args.learn_beta} learn_threshold={args.learn_threshold}")

    cvx_scores = []
    ste_scores = []

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
            ste_lr=args.ste_lr,
            ste_step_size=args.ste_step_size,
            ste_gamma=args.ste_gamma,
            learn_beta=args.learn_beta,
            learn_threshold=args.learn_threshold,
            normalize_hidden=args.normalize_hidden,
            verbose_patterns=args.verbose_patterns,
            log_train=args.log_train,
        )
        cvx_scores.append(out["cvx_test"])
        ste_scores.append(out["ste_test"])

    cvx_mean, cvx_std = mean_std(cvx_scores)
    ste_mean, ste_std = mean_std(ste_scores)

    metric_name = "test_acc" if args.loss in ("ce", "hinge") else "test_negMSE"
    print("\n=== FINAL (mean ± std over seeds) ===")
    print(f"CVX  {metric_name} = {cvx_mean:.4f} ± {cvx_std:.4f}")
    print(f"STE  {metric_name} = {ste_mean:.4f} ± {ste_std:.4f}")


if __name__ == "__main__":
    main()
