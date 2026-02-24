#!/usr/bin/env python3
"""
Data modules / cache structures + sequence conversion helpers for SNN experiments.

This file mirrors the MNISTCache + mnist_to_sequence pattern you use in snn.py
and extends it with:

    Caches:
        - MNIST
        - PTB (Penn Treebank language modeling)
        - Binary-Adding synthetic
        - SHD (Spiking Heidelberg Digits)
        - SSC (Spiking Speech Commands)
        - N-MNIST
        - CIFAR-10
        - CIFAR-100
        - CIFAR10-DVS
        - DVSGesture
        - GSC (Google Speech Commands)
        - TIMIT

    Sequence helpers (mnist-style):
        - mnist_to_sequence
        - ptb_to_sequence
        - binary_adding_to_sequence
        - cifar10_to_sequence
        - cifar100_to_sequence
        - nmnist_to_sequence      (*event-based; simple binning*)
        - cifar10_dvs_to_sequence (*event-based; simple binning*)
        - dvs_gesture_to_sequence (*event-based; simple binning*)
        - shd_to_sequence         (*event-based; simple binning*)
        - ssc_to_sequence         (*event-based; simple binning*)
        - gsc_to_sequence         (*MFCC-like frames placeholder*)
        - timit_to_sequence       (*MFCC-like frames placeholder*)

All sequence helpers follow the convention:

    def X_to_sequence(..., T: int, ...) -> (X_total_seq, y_total, X_test_seq, y_test, num_classes)

with X_*_seq ∈ ℝ^{n × T × d_in} and integer labels y_*, like your existing mnist_to_sequence.

For neuromorphic/audio datasets, the precise event / waveform structure can vary depending
on the exact source; the implementations here use reasonable defaults but you may need to
tweak them to match your local datasets.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import torch
from torchvision import datasets, transforms


# =============================================================================
#  MNIST (classic vision baseline) + sequence helper
# =============================================================================

@dataclass
class MNISTCache:
    X_train: np.ndarray  # (60000, 784)
    y_train: np.ndarray  # (60000,)
    X_test: np.ndarray   # (10000, 784)
    y_test: np.ndarray   # (10000,)


def load_mnist_cache(root: str = "data") -> MNISTCache:
    """
    One-time extraction of MNIST into flat NumPy arrays, normalized exactly
    like your original MNISTCache implementation.
    """
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    # Explicit flattening
    Xtr = np.stack(
        [tr[i][0].view(-1).numpy() for i in range(len(tr))],
        axis=0,
    ).astype(np.float32)
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)

    Xte = np.stack(
        [te[i][0].view(-1).numpy() for i in range(len(te))],
        axis=0,
    ).astype(np.float32)
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

    task ∈ {"mnist_seq", "mnist_perm_seq"}.
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
        raise ValueError(f"Unknown MNIST task: {task}")

    if T <= 0 or T > 784:
        raise ValueError("For MNIST sequence tasks, set 1 <= T <= 784")
    if 784 % T != 0:
        raise ValueError(
            f"T must divide 784 so d_in=784/T is integer. Got T={T} (784%T={784 % T})."
        )

    d_in = 784 // T

    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq = X_test.reshape(X_test.shape[0], T, d_in)

    return (
        X_total_seq.astype(np.float32),
        y_total,
        X_test_seq.astype(np.float32),
        y_test,
        10,
    )


# =============================================================================
#  PTB: Penn Treebank (LM) + sequence helper
# =============================================================================

@dataclass
class PTBCache:
    """
    Minimal PTB cache.

    We store lists of raw text lines. The sequence helper will:
        - tokenize on whitespace
        - build a vocabulary from train_texts
        - map tokens → integer ids
        - sample (T-step) contexts to predict the next token (LM style)
    """
    train_texts: List[str]
    valid_texts: List[str]
    test_texts: List[str]


def load_ptb_cache(root: str = "data") -> PTBCache:
    try:
        from torchtext.datasets import PennTreebank
    except ImportError as e:
        raise ImportError(
            "torchtext is required for PTB. Install with `pip install torchtext`."
        ) from e

    train_iter = PennTreebank(root=root, split="train")
    valid_iter = PennTreebank(root=root, split="valid")
    test_iter = PennTreebank(root=root, split="test")

    train_texts = list(train_iter)
    valid_texts = list(valid_iter)
    test_texts = list(test_iter)

    return PTBCache(
        train_texts=train_texts,
        valid_texts=valid_texts,
        test_texts=test_texts,
    )


def _tokenize_corpus(texts: List[str]) -> List[str]:
    tokens: List[str] = []
    for line in texts:
        tokens.extend(line.strip().split())
    return tokens


def ptb_to_sequence(
    cache: PTBCache,
    *,
    n_train_total: int,
    n_test: int,
    seed: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build LM-style sequences from PTB:

      Input:  sequence of T token ids (context)
      Target: next token id (classification over vocab)

    We treat each token id as a scalar feature (d_in=1) at each timestep.
    Shapes:
        X_total_seq: (n_train_total, T, 1)
        X_test_seq:  (n_test,        T, 1)
    """
    # 1) Build vocab from training texts
    train_tokens = _tokenize_corpus(cache.train_texts)
    test_tokens = _tokenize_corpus(cache.test_texts)

    vocab = sorted(set(train_tokens))
    word2id = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # 2) Convert tokens → ids
    train_ids = np.array([word2id[w] for w in train_tokens if w in word2id], dtype=np.int64)
    test_ids = np.array(
        [word2id[w] for w in test_tokens if w in word2id],
        dtype=np.int64,
    )

    if train_ids.shape[0] <= T or test_ids.shape[0] <= T:
        raise ValueError("PTB corpus too short for given T.")

    rng = np.random.default_rng(seed)

    def sample_sequences(ids: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        max_start = ids.shape[0] - (T + 1)
        if max_start <= 0:
            raise ValueError("Not enough tokens to form sequences.")
        X_list = []
        y_list = []
        for _ in range(N):
            i0 = int(rng.integers(0, max_start))
            seq = ids[i0 : i0 + T]
            target = ids[i0 + T]
            X_list.append(seq)
            y_list.append(target)
        X_arr = np.stack(X_list, axis=0).astype(np.float32)
        y_arr = np.array(y_list, dtype=np.int64)
        # make it (N, T, 1)
        X_arr = X_arr.reshape(X_arr.shape[0], T, 1)
        return X_arr, y_arr

    X_total_seq, y_total = sample_sequences(train_ids, n_train_total)
    X_test_seq, y_test = sample_sequences(test_ids, n_test)

    return X_total_seq, y_total, X_test_seq, y_test, vocab_size


# =============================================================================
#  Binary-Adding synthetic + sequence helper
# =============================================================================

@dataclass
class BinaryAddingCache:
    """
    Simple synthetic dataset for the binary-adding task.

    Shapes:
        X_train: (n_train, T, 2)  float32 in {0.0, 1.0}
        y_train: (n_train,)       float32 (sum of bits)
        X_test:  (n_test,  T, 2)
        y_test:  (n_test,)
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    T: int


def _make_binary_adding_split(
    n: int,
    T: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, 2), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        a = rng.integers(0, 2, size=T).astype(np.float32)
        b = rng.integers(0, 2, size=T).astype(np.float32)
        X[i, :, 0] = a
        X[i, :, 1] = b
        y[i] = a.sum() + b.sum()
    return X, y


def load_binary_adding_cache(
    n_train: int = 60000,
    n_test: int = 10000,
    T: int = 200,
    seed: int = 0,
) -> BinaryAddingCache:
    X_tr, y_tr = _make_binary_adding_split(n_train, T, seed=seed)
    X_te, y_te = _make_binary_adding_split(n_test, T, seed=seed + 1)
    return BinaryAddingCache(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        T=T,
    )


def binary_adding_to_sequence(
    cache: BinaryAddingCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    For binary-adding we already have (n, T, 2).

    We:
        - take first n_train_total from X_train
        - take first n_test from X_test
    Targets remain scalar sums, so this is a regression task with num_classes=None.
    """
    if T != cache.T:
        raise ValueError(f"Binary-adding cache was built with T={cache.T}, got T={T}.")

    X_total = cache.X_train[:n_train_total].copy()
    y_total = cache.y_train[:n_train_total].copy()
    X_test = cache.X_test[:n_test].copy()
    y_test = cache.y_test[:n_test].copy()

    # keep as float32
    return (
        X_total.astype(np.float32),
        y_total.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
        -1,  # use -1 to denote regression in your main loop
    )


# =============================================================================
#  SHD / SSC (HDF5, event-based) + sequence helpers
# =============================================================================

@dataclass
class SHDCache:
    """
    Cache for Spiking Heidelberg Digits (SHD).

    These are event-based sequences stored in HDF5. The exact structure can
    differ by source; we assume:

        spikes_train[i] = (times_i, units_i)
        spikes_test[i]  = (times_j, units_j)

    as two arrays; or a single structured array that we unpack.

    You might need to tweak `_extract_events_from_sample` to match your files.
    """
    spikes_train: Any
    labels_train: np.ndarray
    spikes_test: Any
    labels_test: np.ndarray


def load_shd_cache(path: str = "data/shd") -> SHDCache:
    import h5py
    import os

    train_path = os.path.join(path, "shd_train.h5")
    test_path = os.path.join(path, "shd_test.h5")

    with h5py.File(train_path, "r") as f_tr:
        spikes_train = f_tr["spikes"][:]
        labels_train = np.array(f_tr["labels"][:], dtype=np.int64)

    with h5py.File(test_path, "r") as f_te:
        spikes_test = f_te["spikes"][:]
        labels_test = np.array(f_te["labels"][:], dtype=np.int64)

    return SHDCache(
        spikes_train=spikes_train,
        labels_train=labels_train,
        spikes_test=spikes_test,
        labels_test=labels_test,
    )


@dataclass
class SSCCache:
    """
    Cache for Spiking Speech Commands (SSC).

    Same comments as SHD regarding event structure.
    """
    spikes_train: Any
    labels_train: np.ndarray
    spikes_test: Any
    labels_test: np.ndarray


def load_ssc_cache(path: str = "data/ssc") -> SSCCache:
    import h5py
    import os

    train_path = os.path.join(path, "ssc_train.h5")
    test_path = os.path.join(path, "ssc_test.h5")

    with h5py.File(train_path, "r") as f_tr:
        spikes_train = f_tr["spikes"][:]
        labels_train = np.array(f_tr["labels"][:], dtype=np.int64)

    with h5py.File(test_path, "r") as f_te:
        spikes_test = f_te["spikes"][:]
        labels_test = np.array(f_te["labels"][:], dtype=np.int64)

    return SSCCache(
        spikes_train=spikes_train,
        labels_train=labels_train,
        spikes_test=spikes_test,
        labels_test=labels_test,
    )


def _bin_events_generic(
    times: np.ndarray,
    units: np.ndarray,
    T: int,
    num_units: int,
) -> np.ndarray:
    """
    Generic event-binner:

        - Normalizes times to [0, 1]
        - Splits into T bins
        - For each bin, counts spikes per unit
        - Returns array of shape (T, num_units)
    """
    if times.size == 0:
        return np.zeros((T, num_units), dtype=np.float32)

    t_min = times.min()
    t_max = times.max()
    if t_max == t_min:
        t_norm = np.zeros_like(times, dtype=np.float32)
    else:
        t_norm = (times - t_min) / (t_max - t_min)

    bin_idx = np.clip(
        (t_norm * T).astype(np.int64),
        0,
        T - 1,
    )

    X = np.zeros((T, num_units), dtype=np.float32)
    for t, u in zip(bin_idx, units):
        if 0 <= u < num_units:
            X[t, int(u)] += 1.0
    return X


def shd_to_sequence(
    cache: SHDCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convert SHD events to spike-count sequences (T, num_units).

    This assumes that each element of `spikes_train` / `spikes_test` can be
    unpacked into (times, units) arrays. If your HDF5 layout differs, adjust
    `_extract_events_from_sample` accordingly.
    """

    def _extract_events_from_sample(sample: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Try to handle a few common representations:

            - sample is a structured array with fields 't' and 'u'
            - sample is a 2D array [2, n_events] or [n_events, 2]
        """
        arr = np.array(sample)
        if arr.dtype.names is not None and "t" in arr.dtype.names:
            times = arr["t"].astype(np.float32)
            if "u" in arr.dtype.names:
                units = arr["u"].astype(np.int64)
            else:
                # fallback: treat index as unit id
                units = np.arange(times.shape[0], dtype=np.int64)
        elif arr.ndim == 2 and arr.shape[0] == 2:
            times = arr[0].astype(np.float32)
            units = arr[1].astype(np.int64)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            times = arr[:, 0].astype(np.float32)
            units = arr[:, 1].astype(np.int64)
        else:
            raise ValueError("Unknown SHD event format; please adapt _extract_events_from_sample.")
        return times, units

    # Infer number of units from all train/test events
    all_units: List[int] = []
    for s in cache.spikes_train:
        _, u = _extract_events_from_sample(s)
        all_units.append(int(u.max()) if u.size > 0 else 0)
    for s in cache.spikes_test:
        _, u = _extract_events_from_sample(s)
        all_units.append(int(u.max()) if u.size > 0 else 0)
    num_units = max(all_units) + 1 if all_units else 1

    X_total_list: List[np.ndarray] = []
    for i in range(min(n_train_total, len(cache.spikes_train))):
        times, units = _extract_events_from_sample(cache.spikes_train[i])
        X_total_list.append(_bin_events_generic(times, units, T=T, num_units=num_units))
    X_total = np.stack(X_total_list, axis=0)
    y_total = cache.labels_train[: X_total.shape[0]]

    X_test_list: List[np.ndarray] = []
    for i in range(min(n_test, len(cache.spikes_test))):
        times, units = _extract_events_from_sample(cache.spikes_test[i])
        X_test_list.append(_bin_events_generic(times, units, T=T, num_units=num_units))
    X_test = np.stack(X_test_list, axis=0)
    y_test = cache.labels_test[: X_test.shape[0]]

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if (y_total.size > 0) else 1
    return X_total, y_total, X_test, y_test, num_classes


def ssc_to_sequence(
    cache: SSCCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Same logic as shd_to_sequence, but for Spiking Speech Commands.

    Uses the same generic event representation assumptions.
    """

    def _extract_events_from_sample(sample: Any) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.array(sample)
        if arr.dtype.names is not None and "t" in arr.dtype.names:
            times = arr["t"].astype(np.float32)
            if "u" in arr.dtype.names:
                units = arr["u"].astype(np.int64)
            else:
                units = np.arange(times.shape[0], dtype=np.int64)
        elif arr.ndim == 2 and arr.shape[0] == 2:
            times = arr[0].astype(np.float32)
            units = arr[1].astype(np.int64)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            times = arr[:, 0].astype(np.float32)
            units = arr[:, 1].astype(np.int64)
        else:
            raise ValueError("Unknown SSC event format; please adapt _extract_events_from_sample.")
        return times, units

    all_units: List[int] = []
    for s in cache.spikes_train:
        _, u = _extract_events_from_sample(s)
        all_units.append(int(u.max()) if u.size > 0 else 0)
    for s in cache.spikes_test:
        _, u = _extract_events_from_sample(s)
        all_units.append(int(u.max()) if u.size > 0 else 0)
    num_units = max(all_units) + 1 if all_units else 1

    X_total_list: List[np.ndarray] = []
    for i in range(min(n_train_total, len(cache.spikes_train))):
        times, units = _extract_events_from_sample(cache.spikes_train[i])
        X_total_list.append(_bin_events_generic(times, units, T=T, num_units=num_units))
    X_total = np.stack(X_total_list, axis=0)
    y_total = cache.labels_train[: X_total.shape[0]]

    X_test_list: List[np.ndarray] = []
    for i in range(min(n_test, len(cache.spikes_test))):
        times, units = _extract_events_from_sample(cache.spikes_test[i])
        X_test_list.append(_bin_events_generic(times, units, T=T, num_units=num_units))
    X_test = np.stack(X_test_list, axis=0)
    y_test = cache.labels_test[: X_test.shape[0]]

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if (y_total.size > 0) else 1
    return X_total, y_total, X_test, y_test, num_classes


# =============================================================================
#  N-MNIST (tonic) + sequence helper
# =============================================================================

@dataclass
class NMNISTCache:
    """
    Cache for N-MNIST using tonic dataset objects.

    We keep train/test dataset objects as-is and build sequences by binning
    events into T time bins, flattening spatial dimensions.
    """
    train_ds: Any
    test_ds: Any


def load_nmnist_cache(path: str = "data/nmnist") -> NMNISTCache:
    try:
        import tonic
    except ImportError as e:
        raise ImportError(
            "The `tonic` package is required for N-MNIST. Install with `pip install tonic`."
        ) from e

    train_ds = tonic.datasets.NMNIST(save_to=path, train=True)
    test_ds = tonic.datasets.NMNIST(save_to=path, train=False)
    return NMNISTCache(train_ds=train_ds, test_ds=test_ds)


def _bin_tonic_events(
    events: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Binning for tonic-style events, where `events` is a structured array with
    at least 't', 'x', 'y' fields, and often 'p' (polarity).

    We:
        - normalize times to [0, 1]
        - split into T bins
        - build spike-count images for each bin
        - if 'p' is present, we use 2 channels (on/off); else 1 channel
        - flatten spatial+channel dims to d_in
    """
    if events.size == 0:
        # can't infer shape; default to 1-dim zero vector
        return np.zeros((T, 1), dtype=np.float32)

    # infer spatial extent
    xs = events["x"]
    ys = events["y"]
    H = int(ys.max()) + 1
    W = int(xs.max()) + 1

    times = events["t"].astype(np.float32)
    t_min = times.min()
    t_max = times.max()
    if t_max == t_min:
        t_norm = np.zeros_like(times, dtype=np.float32)
    else:
        t_norm = (times - t_min) / (t_max - t_min)
    bin_idx = np.clip((t_norm * T).astype(np.int64), 0, T - 1)

    has_p = "p" in events.dtype.names
    C = 2 if has_p else 1

    X = np.zeros((T, C, H, W), dtype=np.float32)
    if has_p:
        p = events["p"].astype(np.int64)
        for t, x, y, pol in zip(bin_idx, xs, ys, p):
            if 0 <= x < W and 0 <= y < H and 0 <= t < T:
                X[t, int(pol), int(y), int(x)] += 1.0
    else:
        for t, x, y in zip(bin_idx, xs, ys):
            if 0 <= x < W and 0 <= y < H and 0 <= t < T:
                X[t, 0, int(y), int(x)] += 1.0

    # flatten to (T, d_in)
    return X.reshape(T, -1)


def nmnist_to_sequence(
    cache: NMNISTCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    train_ds, test_ds = cache.train_ds, cache.test_ds

    N_tr = min(n_train_total, len(train_ds))
    N_te = min(n_test, len(test_ds))

    X_total_list: List[np.ndarray] = []
    y_total_list: List[int] = []
    for i in range(N_tr):
        events, label = train_ds[i]
        events = np.array(events)
        X_total_list.append(_bin_tonic_events(events, T=T))
        y_total_list.append(int(label))

    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for i in range(N_te):
        events, label = test_ds[i]
        events = np.array(events)
        X_test_list.append(_bin_tonic_events(events, T=T))
        y_test_list.append(int(label))

    X_total = np.stack(X_total_list, axis=0)
    y_total = np.array(y_total_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if y_total.size > 0 else 1
    return X_total, y_total, X_test, y_test, num_classes


# =============================================================================
#  CIFAR-10 / CIFAR-100 (static vision) + sequence helpers
# =============================================================================

@dataclass
class CIFAR10Cache:
    """
    CIFAR-10 flattened cache.

    Shapes:
        X_train: (50000, 3*32*32)
        y_train: (50000,)
        X_test:  (10000, 3*32*32)
        y_test:  (10000,)
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_cifar10_cache(
    root: str = "data",
    normalize: bool = True,
) -> CIFAR10Cache:
    if normalize:
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )
    else:
        tfm = transforms.ToTensor()

    tr = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    te = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    Xtr = np.stack([tr[i][0].numpy().reshape(-1) for i in range(len(tr))], axis=0).astype(
        np.float32
    )
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)

    Xte = np.stack([te[i][0].numpy().reshape(-1) for i in range(len(te))], axis=0).astype(
        np.float32
    )
    yte = np.array([int(te[i][1]) for i in range(len(te))], dtype=np.int64)

    return CIFAR10Cache(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


@dataclass
class CIFAR100Cache:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_cifar100_cache(
    root: str = "data",
    normalize: bool = True,
) -> CIFAR100Cache:
    if normalize:
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )
    else:
        tfm = transforms.ToTensor()

    tr = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
    te = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)

    Xtr = np.stack([tr[i][0].numpy().reshape(-1) for i in range(len(tr))], axis=0).astype(
        np.float32
    )
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)

    Xte = np.stack([te[i][0].numpy().reshape(-1) for i in range(len(te))], axis=0).astype(
        np.float32
    )
    yte = np.array([int(te[i][1]) for i in range(len(te))], dtype=np.int64)

    return CIFAR100Cache(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def cifar10_to_sequence(
    cache: CIFAR10Cache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Same idea as mnist_to_sequence: flatten 3x32x32=3072 to a 1D vector, then
    chunk into T timesteps of size d_in=3072//T.
    """
    D = cache.X_train.shape[1]
    if D % T != 0:
        raise ValueError(f"For CIFAR-10, require T|{D}. Got T={T}, {D}%T={D % T}.")

    d_in = D // T

    X_total = cache.X_train[:n_train_total].copy()
    y_total = cache.y_train[:n_train_total].copy()
    X_test = cache.X_test[:n_test].copy()
    y_test = cache.y_test[:n_test].copy()

    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq = X_test.reshape(X_test.shape[0], T, d_in)

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if y_total.size > 0 else 10
    return X_total_seq, y_total, X_test_seq, y_test, num_classes


def cifar100_to_sequence(
    cache: CIFAR100Cache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    D = cache.X_train.shape[1]
    if D % T != 0:
        raise ValueError(f"For CIFAR-100, require T|{D}. Got T={T}, {D}%T={D % T}.")

    d_in = D // T

    X_total = cache.X_train[:n_train_total].copy()
    y_total = cache.y_train[:n_train_total].copy()
    X_test = cache.X_test[:n_test].copy()
    y_test = cache.y_test[:n_test].copy()

    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq = X_test.reshape(X_test.shape[0], T, d_in)

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if y_total.size > 0 else 100
    return X_total_seq, y_total, X_test_seq, y_test, num_classes


# =============================================================================
#  CIFAR10-DVS / DVSGesture (tonic) + sequence helpers
# =============================================================================

@dataclass
class CIFAR10DVSCache:
    train_ds: Any
    test_ds: Any


def load_cifar10_dvs_cache(path: str = "data/cifar10_dvs") -> CIFAR10DVSCache:
    try:
        import tonic
    except ImportError as e:
        raise ImportError(
            "The `tonic` package is required for CIFAR10-DVS. Install with `pip install tonic`."
        ) from e

    train_ds = tonic.datasets.CIFAR10DVS(save_to=path, train=True)
    test_ds = tonic.datasets.CIFAR10DVS(save_to=path, train=False)
    return CIFAR10DVSCache(train_ds=train_ds, test_ds=test_ds)


@dataclass
class DVSGestureCache:
    train_ds: Any
    test_ds: Any


def load_dvs_gesture_cache(path: str = "data/dvs_gesture") -> DVSGestureCache:
    try:
        import tonic
    except ImportError as e:
        raise ImportError(
            "The `tonic` package is required for DVSGesture. Install with `pip install tonic`."
        ) from e

    train_ds = tonic.datasets.DVSGesture(save_to=path, train=True)
    test_ds = tonic.datasets.DVSGesture(save_to=path, train=False)
    return DVSGestureCache(train_ds=train_ds, test_ds=test_ds)


def cifar10_dvs_to_sequence(
    cache: CIFAR10DVSCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    train_ds, test_ds = cache.train_ds, cache.test_ds

    N_tr = min(n_train_total, len(train_ds))
    N_te = min(n_test, len(test_ds))

    X_total_list: List[np.ndarray] = []
    y_total_list: List[int] = []
    for i in range(N_tr):
        events, label = train_ds[i]
        events = np.array(events)
        X_total_list.append(_bin_tonic_events(events, T=T))
        y_total_list.append(int(label))

    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for i in range(N_te):
        events, label = test_ds[i]
        events = np.array(events)
        X_test_list.append(_bin_tonic_events(events, T=T))
        y_test_list.append(int(label))

    X_total = np.stack(X_total_list, axis=0)
    y_total = np.array(y_total_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if y_total.size > 0 else 10
    return X_total, y_total, X_test, y_test, num_classes


def dvs_gesture_to_sequence(
    cache: DVSGestureCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    train_ds, test_ds = cache.train_ds, cache.test_ds

    N_tr = min(n_train_total, len(train_ds))
    N_te = min(n_test, len(test_ds))

    X_total_list: List[np.ndarray] = []
    y_total_list: List[int] = []
    for i in range(N_tr):
        events, label = train_ds[i]
        events = np.array(events)
        X_total_list.append(_bin_tonic_events(events, T=T))
        y_total_list.append(int(label))

    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for i in range(N_te):
        events, label = test_ds[i]
        events = np.array(events)
        X_test_list.append(_bin_tonic_events(events, T=T))
        y_test_list.append(int(label))

    X_total = np.stack(X_total_list, axis=0)
    y_total = np.array(y_total_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    num_classes = int(max(y_total.max(), y_test.max()) + 1) if y_total.size > 0 else 11
    return X_total, y_total, X_test, y_test, num_classes


# =============================================================================
#  GSC / TIMIT (torchaudio) + sequence helpers
# =============================================================================

@dataclass
class GSCCache:
    """
    Cache for Google Speech Commands.

    We hold the torchaudio dataset and an index split. The sequence helper
    will:
        - load waveforms
        - compute very simple log-magnitude spectrograms
        - chunk spectrogram time axis into T frames (or pad / crop)
        - flatten frequency axis
    """
    dataset: Any
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_gsc_cache(
    root: str = "data/gsc",
    train_fraction: float = 0.9,
    seed: int = 0,
) -> GSCCache:
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "torchaudio is required for GSC. Install with `pip install torchaudio`."
        ) from e

    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(train_fraction * n)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return GSCCache(dataset=dataset, train_indices=train_indices, test_indices=test_indices)


@dataclass
class TIMITCache:
    """
    Cache for TIMIT (torchaudio).
    """
    train_ds: Any
    test_ds: Any


def load_timit_cache(root: str = "data/timit") -> TIMITCache:
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "torchaudio is required for TIMIT. Install with `pip install torchaudio`."
        ) from e

    train_ds = torchaudio.datasets.TIMIT(root=root, download=True, train=True)
    test_ds = torchaudio.datasets.TIMIT(root=root, download=True, train=False)
    return TIMITCache(train_ds=train_ds, test_ds=test_ds)


def _waveform_to_frames(
    waveform: torch.Tensor,
    T: int,
    n_fft: int = 256,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Convert waveform (1, L) or (L,) to (T, d_in) via a log-magnitude spectrogram.

    We:
        - ensure mono
        - compute STFT magnitude
        - if time frames < T: pad with zeros
        - if time frames > T: center-crop or downsample
    """
    import torch.fft

    if waveform.ndim == 2:
        # (channels, L) → mono
        waveform = waveform.mean(dim=0)
    waveform = waveform.float()

    # simple STFT via torch.stft (avoid depending on torchaudio.transforms here)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=True,
    )
    mag = spec.abs()  # (freq, time)

    # log-magnitude for numerical stability
    mag = torch.log1p(mag)
    F, T_frames = mag.shape

    # match T along time axis
    if T_frames < T:
        pad = T - T_frames
        mag = torch.cat(
            [mag, torch.zeros(F, pad, dtype=mag.dtype, device=mag.device)],
            dim=1,
        )
    elif T_frames > T:
        # simple center-crop
        start = (T_frames - T) // 2
        mag = mag[:, start : start + T]

    # Now shape is (F, T). Transpose to (T, F) then flatten freq if desired.
    mag_TF = mag.T  # (T, F)
    return mag_TF.numpy().astype(np.float32)  # (T, d_in=F)


def gsc_to_sequence(
    cache: GSCCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build simple spectrogram sequences for GSC.

    For now we treat each keyword as its own class. If you want the official
    train/validation/test split, adjust indices accordingly.
    """
    dataset = cache.dataset

    N_tr = min(n_train_total, cache.train_indices.shape[0])
    N_te = min(n_test, cache.test_indices.shape[0])

    X_total_list: List[np.ndarray] = []
    y_total_list: List[int] = []

    # We'll build a label mapping keyword → id
    labels: List[str] = []

    def _get_label(idx: int) -> str:
        _, _, label, *_ = dataset[idx]
        return str(label)

    # first get all labels and build mapping
    for idx in cache.train_indices[:N_tr]:
        labels.append(_get_label(int(idx)))
    for idx in cache.test_indices[:N_te]:
        labels.append(_get_label(int(idx)))
    unique_labels = sorted(set(labels))
    lab2id = {lab: i for i, lab in enumerate(unique_labels)}

    for idx in cache.train_indices[:N_tr]:
        waveform, sample_rate, label, *_ = dataset[int(idx)]
        X = _waveform_to_frames(waveform, T=T)
        X_total_list.append(X)
        y_total_list.append(lab2id[str(label)])

    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for idx in cache.test_indices[:N_te]:
        waveform, sample_rate, label, *_ = dataset[int(idx)]
        X = _waveform_to_frames(waveform, T=T)
        X_test_list.append(X)
        y_test_list.append(lab2id[str(label)])

    X_total = np.stack(X_total_list, axis=0)
    y_total = np.array(y_total_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)
    num_classes = len(unique_labels)
    return X_total, y_total, X_test, y_test, num_classes


def timit_to_sequence(
    cache: TIMITCache,
    *,
    n_train_total: int,
    n_test: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build spectrogram sequences for TIMIT.

    We treat each utterance as belonging to its speaker (or you can remap
    to phone/word labels). For now, we map speaker IDs to class ids.
    """
    train_ds, test_ds = cache.train_ds, cache.test_ds

    def _speaker_label_train(sample_idx: int) -> str:
        # TIMIT: (waveform, sample_rate, utterance, phonetic, word, speaker_id)
        _, _, _, _, _, speaker_id = train_ds[sample_idx]
        return str(speaker_id)

    def _speaker_label_test(sample_idx: int) -> str:
        _, _, _, _, _, speaker_id = test_ds[sample_idx]
        return str(speaker_id)

    N_tr = min(n_train_total, len(train_ds))
    N_te = min(n_test, len(test_ds))

    labels: List[str] = []
    for i in range(N_tr):
        labels.append(_speaker_label_train(i))
    for i in range(N_te):
        labels.append(_speaker_label_test(i))
    unique_labels = sorted(set(labels))
    lab2id = {lab: i for i, lab in enumerate(unique_labels)}

    X_total_list: List[np.ndarray] = []
    y_total_list: List[int] = []
    for i in range(N_tr):
        waveform, sr, *_ = train_ds[i]
        X = _waveform_to_frames(waveform, T=T)
        X_total_list.append(X)
        y_total_list.append(lab2id[_speaker_label_train(i)])

    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for i in range(N_te):
        waveform, sr, *_ = test_ds[i]
        X = _waveform_to_frames(waveform, T=T)
        X_test_list.append(X)
        y_test_list.append(lab2id[_speaker_label_test(i)])

    X_total = np.stack(X_total_list, axis=0)
    y_total = np.array(y_total_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)
    num_classes = len(unique_labels)
    return X_total, y_total, X_test, y_test, num_classes


# =============================================================================
#  Registry of cache loaders + sequence helpers
# =============================================================================

DATASET_CACHE_LOADERS: Dict[str, Any] = {
    # classic vision
    "mnist": load_mnist_cache,
    "cifar10": load_cifar10_cache,
    "cifar100": load_cifar100_cache,

    # language / algorithmic
    "ptb": load_ptb_cache,
    "binary_adding": load_binary_adding_cache,

    # neuromorphic audio
    "shd": load_shd_cache,
    "ssc": load_ssc_cache,

    # neuromorphic vision
    "n_mnist": load_nmnist_cache,
    "cifar10_dvs": load_cifar10_dvs_cache,
    "dvs_gesture": load_dvs_gesture_cache,

    # speech
    "gsc": load_gsc_cache,
    "timit": load_timit_cache,
}


def load_dataset_cache(name: str, **kwargs) -> Any:
    """
    Generic cache loader:

        cache = load_dataset_cache("cifar10", root="data")
    """
    if name not in DATASET_CACHE_LOADERS:
        raise ValueError(f"Unknown dataset cache name: {name!r}")
    return DATASET_CACHE_LOADERS[name](**kwargs)


# For convenience, a mapping from "task" string (CLI-style) to a sequence helper.
# You can plug this directly into snn.py around the place where you currently
# handle parity_seq / two_step_xor_seq / moving_blobs_seq / mnist_seq / mnist_perm_seq.
SEQUENCE_HELPERS: Dict[str, Any] = {
    # MNIST
    "mnist_seq": lambda cache, **kw: mnist_to_sequence(cache, task="mnist_seq", **kw),
    "mnist_perm_seq": lambda cache, **kw: mnist_to_sequence(cache, task="mnist_perm_seq", **kw),

    # PTB
    "ptb_seq": ptb_to_sequence,

    # Binary-adding
    "binary_adding_seq": binary_adding_to_sequence,

    # CIFAR (static)
    "cifar10_seq": cifar10_to_sequence,
    "cifar100_seq": cifar100_to_sequence,

    # Neuromorphic vision
    "nmnist_seq": nmnist_to_sequence,
    "cifar10_dvs_seq": cifar10_dvs_to_sequence,
    "dvs_gesture_seq": dvs_gesture_to_sequence,

    # Neuromorphic audio
    "shd_seq": shd_to_sequence,
    "ssc_seq": ssc_to_sequence,

    # Speech
    "gsc_seq": gsc_to_sequence,
    "timit_seq": timit_to_sequence,
}