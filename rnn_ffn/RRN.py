#!/usr/bin/env python3
"""convex_rnn_threshold_Llayers_T_logged.py

Convex-Lasso-style (fixed random hyperplanes + convex last layer) vs STE training
for **threshold-only** RNN/FCN style models.

This file is the RNN generalization of the FCN script you shared.

Key properties (kept consistent with your requirements):
  - Arbitrary depth L (number of threshold layers) and arbitrary timesteps T.
  - For **each** (time, layer), we avoid joint reconstruction by sampling **separate**
    Gaussian matrices for input vs recurrent parts (P_in, P_rec) and stacking them
    vertically: [P_in ; P_rec].
  - Hidden weights are **fixed** after sampling; only the last convex head W is trained.
  - CVX head training is first-order (SGD/Adam) with **L1** regularization.
  - Supports loss in the convex head: cross-entropy / hinge (binary) / squared (one-hot).
  - Enforces last-layer unique patterns P_last_unique >= n_train by sampling until enough.
  - Logs **train + val** (and optionally test) metrics over epochs.
  - MNIST is loaded once (cached arrays), then subsetted for each seed.

Notes on beta scaling (per your comment):
  - If you set --beta_grid values, the effective regularization used in the convex head is
      beta_hat = beta / sqrt(m_{L-1})
    where m_{L-1} is the width feeding the convex head (the last pattern dimension).

Run examples (binary synthetic):
  python RRN.py --task parity_seq --T 50 --d_in 1 \
    --n_train_total 2000 --val_frac 0.2 --n_test 2000 \
    --L 2 --P_in 2048 --P_rec 1024 --P_last 2000 \
    --loss ce --epochs 100 --batch 128 --device mps --log_train

Run examples (sequential MNIST):
  python RRN.py --task mnist_perm_seq --T 784 --d_in 1 \
    --n_train_total 12665 --val_frac 0.2 --n_test 10000 \
    --L 2 --P_in 20000 --P_rec 20000 --P_last 12665 \
    --loss ce --epochs 100 --batch 128 --device mps --log_train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms



# ============================================================
# Device + seed
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Dataset wrappers (reference-style)
# ============================================================

class PrepareData2D(Dataset):
    """(X_seq, y) where X_seq is (N,T,d_in)."""

    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class PrepareData3D(Dataset):
    """(X_seq, y, z_last) where z_last is cached last-layer patterns (uint8)."""

    def __init__(self, X_seq: np.ndarray, y: np.ndarray, z_last: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()
        self.z = torch.from_numpy(z_last.astype(np.uint8))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.z[idx]


# ============================================================
# Utility: column normalization + uniqueness of sign patterns
# ============================================================

def _col_normalize_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0, keepdims=True) + eps
    return U / norms


def _dedupe_bool_cols_keep_first(D_bool: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Keep first occurrence of each unique boolean column."""
    seen: set[bytes] = set()
    keep: List[int] = []
    D_u8 = D_bool.astype(np.uint8)
    for j in range(D_u8.shape[1]):
        key = D_u8[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    keep_idx = np.asarray(keep, dtype=np.int64)
    return D_bool[:, keep_idx], U[:, keep_idx]


# ============================================================
# Synthetic sequence datasets (T-timesteps)
# ============================================================

def make_parity_seq(n: int, T: int, d_in: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Binary parity over T*d_in Bernoulli bits; y in {0,1}."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, T, d_in)).astype(np.float32)
    parity = (X.sum(axis=(1, 2)) % 2).astype(np.int64)
    return X, parity, 2


def make_two_step_xor_seq(n: int, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Binary XOR of predicates at t=0 and t=T-1; d_in=2; y in {0,1}."""
    if T < 2:
        raise ValueError("two_step_xor_seq requires T>=2")
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, T, 2)).astype(np.float32)
    a = np.array([1.0, -1.0], dtype=np.float32)
    p0 = (X[:, 0, :] @ a >= 0).astype(np.int64)
    p1 = (X[:, -1, :] @ a >= 0).astype(np.int64)
    y = (p0 ^ p1).astype(np.int64)
    return X, y, 2


def make_moving_gaussian_blobs_seq(
    n: int, T: int, seed: int, d_in: int = 50
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



# ============================================================
# MNIST cache + sequence views
# ============================================================

@dataclass
class MNISTCache:
    X_train: np.ndarray  # (60000, 784)
    y_train: np.ndarray  # (60000,)
    X_test: np.ndarray   # (10000, 784)
    y_test: np.ndarray   # (10000,)


def load_mnist_cache(root: str = "data") -> MNISTCache:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    # One-time extraction (yes it's a bit slow; but it's only once per run)
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
    where d_in = 784 // T (requires 784 % T == 0).
    """
    X_total = cache.X_train[:n_train_total].copy()
    y_total = cache.y_train[:n_train_total].copy()
    X_test  = cache.X_test[:n_test].copy()
    y_test  = cache.y_test[:n_test].copy()

    if task == "mnist_perm_seq":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(784)
        X_total = X_total[:, perm]
        X_test  = X_test[:, perm]
    elif task != "mnist_seq":
        raise ValueError(f"Unknown MNIST task: {task}")

    if T <= 0 or T > 784:
        raise ValueError("For MNIST sequence tasks, set 1 <= T <= 784")
    if 784 % T != 0:
        raise ValueError(f"T must divide 784 so d_in=784/T is integer. Got T={T} (784%T={784%T}).")

    d_in = 784 // T  # <-- KEY FIX

    # chunk into timesteps
    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq  = X_test.reshape(X_test.shape[0],  T, d_in)

    return X_total_seq.astype(np.float32), y_total, X_test_seq.astype(np.float32), y_test, 10


# ============================================================
# Sunspot time-series → regression sequences
# ============================================================
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
        start = idx
        end = start + T
        target_idx = end
        if target_idx >= N:
            break
        X.append(series[start:end])
        y.append(series[target_idx])
        idx += window_stride

    X = np.asarray(X, dtype=np.float32)  # (n_eff, T)
    y = np.asarray(y, dtype=np.float32)  # (n_eff,)

    if normalize:
        mu = X.mean()
        sigma = X.std() + 1e-6
        X = (X - mu) / sigma
        y = (y - mu) / sigma

    # reshape to (n_eff, T, d_in=1)
    X = X.reshape(X.shape[0], T, 1)
    num_classes = 1  # regression: scalar target

    return X, y, num_classes


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
# Convex RNN pattern sampling (general L, T)
# ============================================================


@dataclass
class RNNHyperplanes:
    """Shared (tied) random hyperplanes per layer across all timesteps.

    For each hidden layer l=1..L-1 we sample:
      U_in[l]  : (d_in, P_in)
      U_rec[l] : (m_l, P_rec) where m_l = P_rec (after stacking)

    And for the final convex head feature map we sample:
      U_last   : (m_{L-1}, P_last_unique_target)

    Note: This matches the Draft-2 idea of sampling *one* set of hyperplanes per layer and
    reusing them across all timesteps (parameter sharing).
    """
    U_in: List[np.ndarray]      # length L-1, each (d_in, P_in)
    U_rec: List[np.ndarray]     # length L-1, each (m_l, P_rec) with m_l = P_in + P_rec
    U_last: np.ndarray          # (m_{L-1}, P_last)
    hidden_dims: List[int]      # length L-1, each m_l = P_in + P_rec


def _col_normalize_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0, keepdims=True) + eps
    return U / norms


def _unique_cols_from_bool(D_bool: np.ndarray) -> np.ndarray:
    """Return indices of first occurrences of unique boolean columns in (N,P)."""
    seen = set()
    keep = []
    D_u8 = D_bool.astype(np.uint8)
    for j in range(D_u8.shape[1]):
        key = D_u8[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    return np.asarray(keep, dtype=np.int64)


def _rnn_forward_hidden_last_np(
    X: np.ndarray,
    U_in: List[np.ndarray],
    U_rec: List[np.ndarray],
) -> np.ndarray:
    """Forward pass of the threshold RNN (numpy) returning the last hidden layer at final timestep.

    X: (N,T,d_in)
    Returns: H_last_T: (N, m_{L-1})
    """
    N, T, d_in = X.shape
    Lm1 = len(U_in)
    assert Lm1 == len(U_rec)

    # hidden states per layer
    h_prev = [np.zeros((N, U_in[l].shape[1]), dtype=np.float32) for l in range(Lm1)]

    for t in range(T):
        x_t = X[:, t, :].astype(np.float32)
        for l in range(Lm1):
            pre_in = x_t @ U_in[l]                      # (N, P_in)
            pre_rec = h_prev[l] @ U_rec[l]              # (N, P_rec)
            pre = pre_in + pre_rec   # (N,P_in) 
            h = (pre >= 0).astype(np.float32)
            h_prev[l] = h
            x_t = h  # feed to next layer

    return h_prev[-1]  # (N, m_{L-1})


def generate_rnn_sign_patterns_unique(
    X_train: np.ndarray,
    *,
    L: int,
    T: int,
    P_in: int,
    P_rec: int,
    P_last_target: int,
    n_train: int,
    seed: int,
    normalize: bool = True,
    verbose: bool = True,
    chunk_mult: int = 4,
    max_rounds: int = 300,
) -> Tuple[np.ndarray, RNNHyperplanes]:
    """Option A (paper-faithful + stable): tie hyperplanes across timesteps, enforce uniqueness only
    for the feature matrix that feeds the convex head (last timestep, last hidden layer).

    Returns:
      z_train_bool: (n_train, P_last_unique) bool (P_last_unique == P_last_target)
      hypers: RNNHyperplanes with shared U_in/U_rec and chosen U_last
    """
    if X_train.ndim != 3:
        raise ValueError("X_train must be (N,T,d_in).")
    N, T_infer, d_in = X_train.shape
    if T_infer != T:
        raise ValueError(f"X_train has T={T_infer}, but T={T} was requested.")
    if L < 2:
        raise ValueError("RNN depth L must be >= 2 (at least one hidden layer + convex head).")
    if P_last_target < n_train:
        raise ValueError("P_last_target must be >= n_train (D_last >= n).")

    rng = np.random.default_rng(seed)

    # --------- Sample SHARED hyperplanes per hidden layer (tied across all timesteps) ----------
    U_in_list: List[np.ndarray] = []
    U_rec_list: List[np.ndarray] = []
    hidden_dims: List[int] = []

    for l in range(L - 1):
        # by construction, each layer outputs m_l = P_in + P_rec after stacking [in; rec]
        m_l = P_in
        hidden_dims.append(m_l)

        U_in = rng.normal(size=(d_in, P_in)).astype(np.float32)
        if normalize:
            U_in = _col_normalize_np(U_in)

        # recurrent input dimension is m_l (previous hidden of same layer)
        U_rec = rng.normal(size=(m_l, P_rec)).astype(np.float32)
        if normalize:
            U_rec = _col_normalize_np(U_rec)

        U_in_list.append(U_in)
        U_rec_list.append(U_rec)

        # next layer input dimension becomes m_l
        d_in = m_l

        if verbose:
            print(f"[rnn hypers] layer {l+1}: U_in={U_in.shape} U_rec={U_rec.shape} (shared across all T)")

    # --------- Compute last hidden layer at final timestep (TRAIN ONLY) ----------
    H_last_T = _rnn_forward_hidden_last_np(X_train, U_in_list, U_rec_list)  # (N, m_{L-1})
    m_last = H_last_T.shape[1]

    # --------- Sample U_last until we get enough UNIQUE patterns at final timestep ----------
    uniq: Dict[bytes, np.ndarray] = {}
    rounds = 0
    chunk_P = max(P_last_target, chunk_mult * P_last_target)

    while len(uniq) < P_last_target and rounds < max_rounds:
        rounds += 1
        U = rng.normal(size=(m_last, chunk_P)).astype(np.float32)
        if normalize:
            U = _col_normalize_np(U)

        D_bool = (H_last_T @ U >= 0)  # (N,chunk_P)
        D_u8 = D_bool.astype(np.uint8)

        for j in range(D_u8.shape[1]):
            key = D_u8[:, j].tobytes()
            if key not in uniq:
                uniq[key] = U[:, j].copy()
                if len(uniq) >= P_last_target:
                    break

        if verbose:
            print(f"[rnn patterns] last-layer enforce: round={rounds}/{max_rounds} uniques={len(uniq)}/{P_last_target}")

    if len(uniq) < P_last_target:
        raise RuntimeError(
            f"Failed to generate P_last_target={P_last_target} unique last-step patterns after "
            f"{max_rounds} rounds. Got {len(uniq)}. Increase P_in/P_rec, chunk_mult, or max_rounds."
        )

    keys = list(uniq.keys())[:P_last_target]
    D_last = np.stack([np.frombuffer(k, dtype=np.uint8) for k in keys], axis=1)  # (N,P_last_target)
    z_train_bool = (D_last.astype(np.float32) >= 0.5)  # bool
    U_last = np.stack([uniq[k] for k in keys], axis=1).astype(np.float32)

    if verbose:
        print(f"[rnn patterns] P_last_unique={P_last_target} (target={P_last_target}, n_train={n_train})")

    hypers = RNNHyperplanes(U_in=U_in_list, U_rec=U_rec_list, U_last=U_last, hidden_dims=hidden_dims)
    return z_train_bool, hypers


def forward_rnn_patterns_torch(
    x: torch.Tensor,
    hypers: RNNHyperplanes,
    *,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """Recompute last-step patterns for val/test using shared hyperplanes (torch).

    x: (B,T,d_in) float
    returns: z_last_bool (B, P_last) bool
    """
    if x.ndim != 3:
        raise ValueError("x must be (B,T,d_in).")
    B, T_infer, d_in = x.shape
    if T_infer != T:
        raise ValueError(f"x has T={T_infer}, but T={T} was requested.")

    # hidden states per layer
    h_prev: List[torch.Tensor] = []
    for l in range(len(hypers.U_in)):
        m_l = hypers.U_rec[l].shape[0]
        h_prev.append(torch.zeros(B, m_l, device=device, dtype=torch.float32))

    for t in range(T):
        x_t = x[:, t, :].float()
        for l in range(len(hypers.U_in)):
            U_in = torch.from_numpy(hypers.U_in[l]).float().to(device)     # (d_in, P_in)
            U_rec = torch.from_numpy(hypers.U_rec[l]).float().to(device)   # (m_l, P_rec)
            pre_in = x_t @ U_in
            pre_rec = h_prev[l] @ U_rec
            pre = pre_in + pre_rec
            h = (pre >= 0).float()
            h_prev[l] = h
            x_t = h

    U_last = torch.from_numpy(hypers.U_last).float().to(device)  # (m_last, P_last)
    z_last = (h_prev[-1] @ U_last >= 0)
    return z_last


class CvxHead(nn.Module):
    def __init__(self, P_last: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(P_last, num_classes), requires_grad=True)

    def forward(self, z_last: torch.Tensor) -> torch.Tensor:
        return z_last.float() @ self.W


def cvx_loss(logits: torch.Tensor, y: torch.Tensor, head: CvxHead, *, loss_name: str, beta: float, beta_scale_dim: int) -> torch.Tensor:
    """Convex objective in W: loss + (beta/sqrt(m_{L-1})) * ||W||_1."""
    beta_hat = beta / float(np.sqrt(max(1, beta_scale_dim)))
    reg = beta_hat * head.W.abs().sum()

    if loss_name == "ce":
        return F.cross_entropy(logits, y) + reg

    if loss_name == "squared":
        C = logits.shape[1]
        y_oh = F.one_hot(y, num_classes=C).float()
        return 0.5 * F.mse_loss(logits, y_oh) + reg

    if loss_name in ("squared", "mse"):
        # regression: scalar target
        loss_data = F.mse_loss(logits.squeeze(-1), y.float())
        return loss_data + reg
    if loss_name == "mae":
        # regression: absolute error
        loss_data = F.l1_loss(logits.squeeze(-1), y.float())
        return loss_data + reg

    if loss_name == "hinge":
        if logits.shape[1] != 2:
            raise ValueError("hinge loss implemented for binary classification only (C=2)")
        y_pm = y.float() * 2.0 - 1.0
        score = logits[:, 1] - logits[:, 0]
        return torch.relu(1.0 - y_pm * score).mean() + reg

    raise ValueError(f"Unknown loss_name: {loss_name}")


@torch.no_grad()
def eval_acc_cached_z(head: CvxHead, loader3d: DataLoader, device: torch.device) -> float:
    head.eval()
    correct, total = 0, 0
    for _x, _y, _z in loader3d:
        _y = _y.to(device)
        _z = _z.to(device)
        pred = torch.argmax(head(_z), dim=1)
        correct += (pred == _y).sum().item()
        total += _y.numel()
    return correct / total


@torch.no_grad()
def eval_acc_recompute_z(head: CvxHead, loader2d: DataLoader, hypers: RNNHyperplanes, T: int, device: torch.device) -> float:
    head.eval()
    correct, total = 0, 0
    for Xb, yb in loader2d:
        Xb = Xb.to(device)
        yb = yb.to(device)
        zb = forward_rnn_patterns_torch(Xb, hypers, T=T, device=device)
        pred = torch.argmax(head(zb), dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def train_cvx_head_first_order(
    *,
    train_loader3d: DataLoader,
    val_loader2d: DataLoader,
    hypers: RNNHyperplanes,
    T: int,
    P_last: int,
    num_classes: int,
    loss_name: str,
    beta: float,
    lr: float,
    epochs: int,
    device: torch.device,
    optimizer_name: str,
    step_size: int,
    gamma: float,
    log_train: bool,
) -> Dict[str, object]:
    head = CvxHead(P_last, num_classes).to(device)
    if optimizer_name == "adam":
        opt = torch.optim.Adam(head.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val = -1.0
    best_state = None

    if log_train:
        tr0 = eval_acc_cached_z(head, train_loader3d, device)
        va0 = eval_acc_recompute_z(head, val_loader2d, hypers, T=T, device=device)
        print(f"[CVX] ep=000/{epochs} train_acc={tr0:.4f} val_acc={va0:.4f} lr={lr:.2e}")

    for ep in range(1, epochs + 1):
        head.train()
        last_loss = None
        for _x, _y, _z in train_loader3d:
            _y = _y.to(device)
            _z = _z.to(device)
            logits = head(_z)
            loss = cvx_loss(logits, _y, head, loss_name=loss_name, beta=beta, beta_scale_dim=hypers.U_last.shape[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        sched.step()

        val_acc = eval_acc_recompute_z(head, val_loader2d, hypers, T=T, device=device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if log_train and (ep <= 5 or ep % 10 == 0 or ep == epochs):
            tr = eval_acc_cached_z(head, train_loader3d, device)
            lr_now = sched.get_last_lr()[0]
            print(f"[CVX] ep={ep:03d}/{epochs} last_loss={last_loss:.4f} train_acc={tr:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")

    if best_state is not None:
        head.load_state_dict(best_state)
    return {"head": head, "best_val": float(best_val)}


# ============================================================
# STE RNN baseline
# ============================================================

class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STE_RNN_Threshold(nn.Module):
    """Simple stacked threshold RNN (shared weights across time)."""

    def __init__(self, d_in: int, widths: List[int], num_classes: int):
        super().__init__()
        self.L = len(widths)
        self.widths = widths
        self.in_fcs = nn.ModuleList()
        self.rec_fcs = nn.ModuleList()
        prev_in = d_in
        for w in widths:
            self.in_fcs.append(nn.Linear(prev_in, w, bias=False))
            self.rec_fcs.append(nn.Linear(w, w, bias=False))
            prev_in = w
        self.out = nn.Linear(widths[-1], num_classes, bias=True)

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = X_seq.shape
        h = [torch.zeros(B, w, device=X_seq.device, dtype=X_seq.dtype) for w in self.widths]
        for t in range(T):
            x_l = X_seq[:, t, :]
            for l in range(self.L):
                pre = self.in_fcs[l](x_l) + self.rec_fcs[l](h[l])
                h[l] = ThresholdSTE.apply(pre)
                x_l = h[l]
        return self.out(h[-1])


def path_reg_squared_rnn(model: STE_RNN_Threshold) -> torch.Tensor:
    """Tractable path-like surrogate for RNN."""
    out2 = model.out.weight.pow(2).sum()
    s = 0.0
    for l in range(model.L):
        s = s + model.in_fcs[l].weight.pow(2).sum() + model.rec_fcs[l].weight.pow(2).sum()
    return out2 * s


@torch.no_grad()
def ste_eval_acc(model: STE_RNN_Threshold, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = torch.argmax(model(Xb), dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def train_ste_rnn(
    *,
    d_in: int,
    widths: List[int],
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    epochs: int,
    beta_path: float,
    device: torch.device,
    step_size: int,
    gamma: float,
    log_train: bool,
    loss_name: str,
) -> Dict[str, object]:
    model = STE_RNN_Threshold(d_in=d_in, widths=widths, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val = -1.0
    best_state = None

    if log_train:
        tr0 = ste_eval_acc(model, train_loader, device)
        va0 = ste_eval_acc(model, val_loader, device)
        print(f"[STE] ep=000/{epochs} train_acc={tr0:.4f} val_acc={va0:.4f} lr={lr:.2e}")

    loss_name = loss_name.lower()

    for ep in range(1, epochs + 1):
        model.train()
        last_loss = None
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            if loss_name in ("ce", "cross_entropy", "hinge"):
                # classification – keep using CE in the STE baseline for hinge tasks too
                loss_data = F.cross_entropy(logits, yb.long())
            elif loss_name in ("squared", "mse"):
                loss_data = F.mse_loss(logits.squeeze(-1), yb.float())
            elif loss_name == "mae":
                loss_data = F.l1_loss(logits.squeeze(-1), yb.float())
            else:
                raise ValueError(f"Unknown loss_name for STE: {loss_name}")

            loss = loss_data + beta_path * path_reg_squared_rnn(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        sched.step()

        val_acc = ste_eval_acc(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if log_train and (ep <= 5 or ep % 10 == 0 or ep == epochs):
            tr = ste_eval_acc(model, train_loader, device)
            lr_now = sched.get_last_lr()[0]
            print(f"[STE] ep={ep:03d}/{epochs} last_loss={last_loss:.4f} train_acc={tr:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "best_val": float(best_val)}


# ============================================================
# Task factory
# ============================================================

def make_task_data(
    *,
    task: str,
    n_train_total: int,
    n_test: int,
    T: int,
    d_in: int,
    seed: int,
    mnist_cache: Optional[MNISTCache],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return X_total_seq, y_total, X_test_seq, y_test, num_classes."""

    if task in {"mnist_seq", "mnist_perm_seq"}:
        if mnist_cache is None:
            raise ValueError("MNIST cache must be provided for MNIST tasks")
        return mnist_to_sequence(
            mnist_cache,
            n_train_total=n_train_total,
            n_test=n_test,
            seed=seed,
            task=task,
            T=T,
        )

    if task == "parity_seq":
        X, y, C = make_parity_seq(n_train_total + n_test, T, d_in, seed)
        return X[:n_train_total], y[:n_train_total], X[n_train_total:], y[n_train_total:], C

    if task == "two_step_xor_seq":
        X, y, C = make_two_step_xor_seq(n_train_total + n_test, T, seed)
        return X[:n_train_total], y[:n_train_total], X[n_train_total:], y[n_train_total:], C

    if task == "moving_blobs_seq":
        X, y, C = make_moving_gaussian_blobs_seq(n_train_total + n_test, T, seed)
        return X[:n_train_total], y[:n_train_total], X[n_train_total:], y[n_train_total:], C

    if task == "sunspot_seq":
        # We ignore d_in for sunspots (d_in=1 always).
        # Build a single big dataset and then split it into train_total + test.
        X_all, y_all, num_classes = make_sunspot_seq(
            n=n_train_total + n_test,
            T=T,
            seed=seed,
        )
        n_total = X_all.shape[0]
        if n_total < n_train_total + n_test:
            print(
                f"[warn] sunspot_seq: only {n_total} sequences available; "
                f"requested n_train_total={n_train_total}, n_test={n_test}."
            )
        n_train_eff = min(n_train_total, n_total)
        n_test_eff = min(n_test, n_total - n_train_eff)

        X_train = X_all[:n_train_eff]
        y_train = y_all[:n_train_eff]
        X_test = X_all[n_train_eff:n_train_eff + n_test_eff]
        y_test = y_all[n_train_eff:n_train_eff + n_test_eff]

        return X_train, y_train, X_test, y_test, num_classes
    
    raise ValueError(f"Unknown task: {task}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, required=True,
                    choices=["parity_seq", "two_step_xor_seq", "moving_blobs_seq", "mnist_seq", "mnist_perm_seq"])
    ap.add_argument("--T", type=int, default=2, help="Timesteps")
    ap.add_argument("--d_in", type=int, default=1, help="Input dim for synthetic tasks")
    ap.add_argument("--n_train_total", type=int, default=2000)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_test", type=int, default=2000)

    ap.add_argument("--L", type=int, default=2, help="Number of threshold layers")
    ap.add_argument("--P_in", type=int, default=2048)
    ap.add_argument("--P_rec", type=int, default=2048)
    ap.add_argument("--P_last", type=int, default=0, help="Target last-layer uniques (0 => n_train)")

    ap.add_argument("--loss", type=str, default="ce", choices=["hinge", "ce", "squared", "mse", "mae"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)

    ap.add_argument("--cvx_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--cvx_step_size", type=int, default=30)
    ap.add_argument("--cvx_gamma", type=float, default=0.5)
    ap.add_argument("--beta_grid", type=float, nargs="+", default=[1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0])
    ap.add_argument("--lr_grid", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3])

    ap.add_argument("--ste_lr", type=float, default=1e-3)
    ap.add_argument("--ste_beta_path", type=float, default=1e-4)
    ap.add_argument("--ste_step_size", type=int, default=30)
    ap.add_argument("--ste_gamma", type=float, default=0.5)

    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--verbose_patterns", action="store_true")
    ap.add_argument("--log_train",type=bool, default=True)

    args = ap.parse_args()
    device = get_device(args.device)

    mnist_cache = None
    if args.task in {"mnist_seq", "mnist_perm_seq"}:
        print("[info] loading MNIST once (no double-download)...")
        mnist_cache = load_mnist_cache()

    print(f"[info] device={device} task={args.task} T={args.T} L={args.L} epochs={args.epochs} batch={args.batch}")
    print(f"[info] n_train_total={args.n_train_total} val_frac={args.val_frac} n_test={args.n_test}")
    print(f"[info] P_in={args.P_in} P_rec={args.P_rec} P_last(target)={args.P_last if args.P_last>0 else 'n_train'}")
    print(f"[info] CVX grid betas={args.beta_grid} lrs={args.lr_grid} loss={args.loss}")

    cvx_tests: List[float] = []
    ste_tests: List[float] = []

    for s in args.seeds:
        set_seed(s)
        X_total, y_total, X_test, y_test, C = make_task_data(
            task=args.task,
            n_train_total=args.n_train_total,
            n_test=args.n_test,
            T=args.T,
            d_in=args.d_in,
            seed=s,
            mnist_cache=mnist_cache,
        )

        if args.loss == "hinge" and C != 2:
            raise ValueError("--loss hinge requires binary task (num_classes=2)")

        X_train, y_train, X_val, y_val = split_train_val(X_total, y_total, args.val_frac, seed=s)
        n_train = X_train.shape[0]
        P_last_target = max(n_train, int(args.P_last)) if args.P_last > 0 else n_train

        # Patterns once per seed (train only)
        z_train_bool, hypers = generate_rnn_sign_patterns_unique(
            X_train,
            L=args.L,
            T=args.T,
            P_in=args.P_in,
            P_rec=args.P_rec,
            P_last_target=P_last_target,
            n_train=n_train,
            seed=s,
            normalize=True,
            verbose=True,
        )
        z_train = z_train_bool.astype(np.uint8)
        P_last = z_train.shape[1]
        print(f"[seed {s}] patterns done: P_last_unique={P_last} (target={P_last_target}, n_train={n_train})")

        train_ds3d = PrepareData3D(X_train, y_train, z_train)
        val_ds2d = PrepareData2D(X_val, y_val)
        test_ds2d = PrepareData2D(X_test, y_test)

        train_loader3d = DataLoader(train_ds3d, batch_size=args.batch, shuffle=True)
        val_loader2d = DataLoader(val_ds2d, batch_size=512, shuffle=False)
        test_loader2d = DataLoader(test_ds2d, batch_size=512, shuffle=False)

        # Grid search (reuse same patterns)
        best = {"val": -1.0, "beta": None, "lr": None, "head": None}
        for beta in args.beta_grid:
            for lr in args.lr_grid:
                out = train_cvx_head_first_order(
                    train_loader3d=train_loader3d,
                    val_loader2d=val_loader2d,
                    hypers=hypers,
                    T=args.T,
                    P_last=P_last,
                    num_classes=C,
                    loss_name=args.loss,
                    beta=beta,
                    lr=lr,
                    epochs=args.epochs,
                    device=device,
                    optimizer_name=args.cvx_optimizer,
                    step_size=args.cvx_step_size,
                    gamma=args.cvx_gamma,
                    log_train=args.log_train,
                )
                if out["best_val"] > best["val"]:
                    best = {"val": out["best_val"], "beta": beta, "lr": lr, "head": out["head"]}

        # optional logging for best
        if args.log_train:
            print(f"[seed {s}] re-train CVX best for logging: beta={best['beta']} lr={best['lr']}")
            logged = train_cvx_head_first_order(
                train_loader3d=train_loader3d,
                val_loader2d=val_loader2d,
                hypers=hypers,
                T=args.T,
                P_last=P_last,
                num_classes=C,
                loss_name=args.loss,
                beta=float(best["beta"]),
                lr=float(best["lr"]),
                epochs=args.epochs,
                device=device,
                optimizer_name=args.cvx_optimizer,
                step_size=args.cvx_step_size,
                gamma=args.cvx_gamma,
                log_train=args.log_train,
            )
            best["head"] = logged["head"]
            best["val"] = logged["best_val"]

        cvx_test = eval_acc_recompute_z(best["head"], test_loader2d, hypers, T=args.T, device=device)

        # STE baseline: simple comparable width
        # STE baseline should mirror the *hidden-state dimension* of the tied-hyperplane RNN.
        # In our construction, each hidden layer has dimension m_l = P_in + P_rec (due to [in; rec] stacking).
        ste_width = args.P_in
        widths = [ste_width for _ in range(max(1, args.L - 1))]
        ste_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                                     batch_size=args.batch, shuffle=True)
        ste_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                                   batch_size=512, shuffle=False)
        ste_test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                                    batch_size=512, shuffle=False)

        ste_out = train_ste_rnn(
            d_in=X_train.shape[2],
            widths=widths,
            num_classes=C,
            train_loader=ste_train_loader,
            val_loader=ste_val_loader,
            lr=args.ste_lr,
            epochs=args.epochs,
            beta_path=args.ste_beta_path,
            device=device,
            step_size=args.ste_step_size,
            gamma=args.ste_gamma,
            loss_name = args.loss,
            log_train=args.log_train,
        )
        ste_test = ste_eval_acc(ste_out["model"], ste_test_loader, device)

        cvx_tests.append(float(cvx_test))
        ste_tests.append(float(ste_test))
        print(f"[seed {s}] CVX test={cvx_test:.4f} (val={best['val']:.4f}, beta={best['beta']}, lr={best['lr']}, P_last={P_last}) "
              f"| STE test={ste_test:.4f} (val={ste_out['best_val']:.4f}, width={ste_width})")

    cvx_mean, cvx_std = float(np.mean(cvx_tests)), float(np.std(cvx_tests))
    ste_mean, ste_std = float(np.mean(ste_tests)), float(np.std(ste_tests))
    print("\n=== FINAL (mean ± std over seeds) ===")
    print(f"CVX test_acc = {cvx_mean:.4f} ± {cvx_std:.4f}")
    print(f"STE test_acc = {ste_mean:.4f} ± {ste_std:.4f}")


if __name__ == "__main__":
    main()
