#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from torchvision import datasets, transforms


# ============================================================
# Utilities
# ============================================================

def choose_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_arith_module_path() -> Optional[str]:
    candidates = [
        Path(__file__).with_name("arithmetic_test_bench_seq.py"),
        Path(__file__).with_name("arithmetic_test_bench.py"),
        Path("/mnt/data/arithmetic_test_bench_seq.py"),
        Path("/mnt/data/arithmetic_test_bench.py"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _row_to_dict(row) -> dict:
    if hasattr(row, "__dict__"):
        return asdict(row)
    raise TypeError("Expected dataclass row")


# ============================================================
# Dataset loaders
# ============================================================

@dataclass
class SequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    T: int
    d_in: int
    dataset_name: str


@dataclass
class MNISTCache:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_mnist_cache(root: str = "data") -> MNISTCache:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    Xtr = np.stack([tr[i][0].view(-1).numpy() for i in range(len(tr))], axis=0).astype(np.float32)
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)
    Xte = np.stack([te[i][0].view(-1).numpy() for i in range(len(te))], axis=0).astype(np.float32)
    yte = np.array([int(te[i][1]) for i in range(len(te))], dtype=np.int64)
    return MNISTCache(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)


def build_mnist_seq_dataset(*, task: str, T: int, n_train_total: int, n_val: int, n_test: int, seed: int) -> SequenceDataset:
    cache = load_mnist_cache(root="data")
    X_total = cache.X_train[: n_train_total + n_val].copy()
    y_total = cache.y_train[: n_train_total + n_val].copy()
    X_test = cache.X_test[:n_test].copy()
    y_test = cache.y_test[:n_test].copy()
    if task == "mnist_perm_seq":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(784)
        X_total = X_total[:, perm]
        X_test = X_test[:, perm]
    elif task != "mnist_seq":
        raise ValueError(f"Unknown MNIST task: {task}")
    if 784 % T != 0:
        raise ValueError(f"T must divide 784. Got T={T}.")
    d_in = 784 // T
    X_total_seq = X_total.reshape(X_total.shape[0], T, d_in)
    X_test_seq = X_test.reshape(X_test.shape[0], T, d_in)
    X_train = X_total_seq[:n_train_total]
    y_train = y_total[:n_train_total]
    X_val = X_total_seq[n_train_total:n_train_total + n_val]
    y_val = y_total[n_train_total:n_train_total + n_val]
    return SequenceDataset(
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.float32), y_val=y_val.astype(np.int64),
        X_test=X_test_seq.astype(np.float32), y_test=y_test.astype(np.int64),
        num_classes=10, T=T, d_in=d_in, dataset_name=task,
    )


def build_arithmetic_seq_dataset(*, op: str, base: int, n_digits: int, n_train: int, n_val: int, n_test: int, seed: int) -> SequenceDataset:
    arith_path = _resolve_arith_module_path()
    if arith_path is None:
        raise FileNotFoundError("Could not locate arithmetic_test_bench_seq.py or arithmetic_test_bench.py")
    arith = _load_module("arith_layerwise", arith_path)
    if not hasattr(arith, "generate_samples_for_op_base_seq"):
        raise RuntimeError("Arithmetic module found, but per-timestep seq helpers are unavailable.")
    train_samples = arith.generate_samples_for_op_base_seq(op, base, n_digits, n_train, seed + 11)
    val_samples = arith.generate_samples_for_op_base_seq(op, base, n_digits, n_val, seed + 29)
    test_samples = arith.generate_samples_for_op_base_seq(op, base, n_digits, n_test, seed + 47)
    verify = getattr(arith, "verify_sample_seq")
    for s in train_samples[: min(5, len(train_samples))]:
        verify(s)
    def stack(samples):
        X = np.stack([s.inputs.astype(np.float32) / max(base - 1, 1) for s in samples], axis=0)
        y = np.stack([s.target_tokens.astype(np.int64) for s in samples], axis=0)
        return X, y
    X_train, y_train = stack(train_samples)
    X_val, y_val = stack(val_samples)
    X_test, y_test = stack(test_samples)
    return SequenceDataset(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        num_classes=int(base), T=int(X_train.shape[1]), d_in=int(X_train.shape[2]),
        dataset_name=f"arith_seq::{op}::base{base}::digits{n_digits}",
    )


# ============================================================
# Model / objectives (aligned with current fine-tune style)
# ============================================================

class ThreeLayerBlock(nn.Module):
    """A 3-layer block: two LIF hidden stages + P_last projection + per-timestep classifier."""

    def __init__(
        self,
        d_in: int,
        hidden_dim: int,
        p_last: int,
        num_classes: int,
        *,
        beta_leak: float = 0.99,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        last_layer_readout: str = "membrane",
    ):
        super().__init__()
        self.last_layer_readout = last_layer_readout
        self.fc1 = nn.Linear(d_in, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, p_last, bias=False)
        self.lif1 = snn.Leaky(beta=beta_leak, threshold=threshold, learn_beta=learn_beta, learn_threshold=learn_threshold)
        self.lif2 = snn.Leaky(beta=beta_leak, threshold=threshold, learn_beta=learn_beta, learn_threshold=learn_threshold)
        self.classifier = nn.Linear(p_last, num_classes, bias=False)

    def forward_features(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        device = x_seq.device
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)
        feat_list = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            spk1, mem1 = self.lif1(self.fc1(x_t), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            readout = mem2 if self.last_layer_readout == "membrane" else spk2
            feat_list.append(self.proj(readout))
        return torch.stack(feat_list, dim=1)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x_seq)
        logits = self.classifier(feats)
        return logits, feats


def sequence_hinge_ovr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiclass one-vs-rest hinge loss.

    - If y is (B,), supervise final timestep only (MNIST-style sequence classification).
    - If y is (B,T), supervise all timesteps (arithmetic-style token prediction).
    """
    if y.ndim == 1:
        scores = logits[:, -1, :]  # (B,C)
        targets = y
    else:
        B, T, C = logits.shape
        scores = logits.reshape(B * T, C)  # (B*T,C)
        targets = y.reshape(B * T)         # (B*T,)

    n_samples, n_classes = scores.shape
    y_pm1 = -torch.ones((n_samples, n_classes), device=scores.device, dtype=scores.dtype)
    y_pm1[torch.arange(n_samples, device=scores.device), targets] = 1.0
    return torch.clamp(1.0 - y_pm1 * scores, min=0.0).mean()


def block_path_reg(model: ThreeLayerBlock) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    v = torch.ones(model.fc1.in_features, device=device, dtype=dtype)
    for layer in (model.fc1, model.fc2, model.proj):
        v = (layer.weight ** 2) @ v
    reg_sq = ((model.classifier.weight ** 2) * v.unsqueeze(0)).sum()
    return torch.sqrt(reg_sq + 1e-12)


@torch.no_grad()
def evaluate_block(model: ThreeLayerBlock, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    X = X.to(device)
    y = y.to(device)
    logits, feats = model(X)
    if y.ndim == 1:
        preds_last = logits[:, -1, :].argmax(dim=1)
        acc_last = float((preds_last == y).float().mean().item())
        tok = acc_last
        seq = acc_last
    else:
        preds = logits.argmax(dim=2)
        tok = float((preds == y).float().mean().item())
        seq = float((preds == y).all(dim=1).float().mean().item())
    return tok, seq, feats.detach().cpu()


@torch.no_grad()
def evaluate_stacked_blocks(
    models: List[ThreeLayerBlock],
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    if len(models) == 0:
        raise ValueError("Expected at least one block for stacked evaluation.")
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    logits = None
    for idx, model in enumerate(models):
        model = model.to(device)
        model.eval()
        logits, feats = model(x_t)
        if idx < len(models) - 1:
            x_t = feats
    assert logits is not None
    if y_t.ndim == 1:
        preds_last = logits[:, -1, :].argmax(dim=1)
        acc_last = float((preds_last == y_t).float().mean().item())
        return acc_last, acc_last
    preds = logits.argmax(dim=2)
    tok = float((preds == y_t).float().mean().item())
    seq = float((preds == y_t).all(dim=1).float().mean().item())
    return tok, seq


# ============================================================
# CVX head on frozen features (same per-timestep OVR hinge-LASSO idea)
# ============================================================

@dataclass
class CVXHeadResult:
    token_train_acc: float
    token_val_acc: float
    token_test_acc: float
    seq_train_acc: float
    seq_val_acc: float
    seq_test_acc: float
    beta: float
    primal_obj: float
    nonzero_count: int


def solve_binary_hinge_lasso(D: np.ndarray, y_pm1: np.ndarray, rho: float) -> Tuple[np.ndarray, float]:
    n, p = D.shape
    Df = D.astype(np.float64)
    y = y_pm1.astype(np.float64)
    w = cp.Variable(p)
    xi = cp.Variable(n, nonneg=True)
    linear_scores = cp.sum(cp.multiply(Df, cp.reshape(w, (1, p))), axis=1)
    margins = cp.multiply(y, linear_scores)
    prob = cp.Problem(cp.Minimize((1.0 / n) * cp.sum(xi) + rho * cp.norm1(w)), [margins >= 1.0 - xi])
    last_err = None
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            if solver is cp.CLARABEL:
                prob.solve(solver=solver, verbose=False)
            elif solver is cp.OSQP:
                prob.solve(solver=solver, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
            else:
                prob.solve(solver=solver, verbose=False, eps=1e-5, max_iters=50000)
            if w.value is not None:
                break
        except Exception as e:
            last_err = e
            continue
    if w.value is None:
        raise RuntimeError(f"CVX binary solve failed. Last error: {last_err}")
    w_star = np.asarray(w.value, dtype=np.float64).reshape(-1)
    if not np.isfinite(w_star).all():
        raise FloatingPointError("Non-finite binary CVX weights.")
    primal_obj = float(prob.value)
    if not np.isfinite(primal_obj):
        raise FloatingPointError("Non-finite binary CVX primal objective.")
    return w_star, primal_obj


def _acc_from_scores(scores: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    preds = scores.argmax(axis=2)
    tok = float((preds == y).mean())
    seq = float(np.all(preds == y, axis=1).mean())
    return tok, seq


def fit_cvx_head_all_timesteps(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int,
    beta_grid: List[float],
) -> CVXHeadResult:
    N, T, P = Z_train.shape
    best: Optional[CVXHeadResult] = None
    target_is_sequence = y_train.ndim == 2
    for beta in beta_grid:
        rho = float(beta / math.sqrt(max(P, 1)))
        W_list: List[np.ndarray] = []
        primal_sum = 0.0
        nnz = 0
        candidate_failed = False
        t_indices = list(range(T)) if target_is_sequence else [T - 1]
        for t in t_indices:
            D_t = Z_train[:, t, :]
            W_t = np.zeros((P, num_classes), dtype=np.float64)
            for c in range(num_classes):
                y_ref = y_train[:, t] if target_is_sequence else y_train
                y_bin = np.where(y_ref == c, 1.0, -1.0).astype(np.float64)
                try:
                    w, primal_obj = solve_binary_hinge_lasso(D_t, y_bin, rho)
                except (RuntimeError, FloatingPointError, ValueError):
                    candidate_failed = True
                    break
                W_t[:, c] = w
                primal_sum += float(primal_obj)
            if candidate_failed:
                break
            nnz += int(np.count_nonzero(np.abs(W_t) > 1e-10))
            if not np.isfinite(W_t).all():
                candidate_failed = True
                break
            W_list.append(W_t)
        if candidate_failed:
            continue
        if target_is_sequence:
            def score_all(Z_all: np.ndarray) -> np.ndarray:
                scores_list = []
                for t in range(T):
                    score_t = np.einsum("np,pc->nc", Z_all[:, t, :], W_list[t], optimize=True)
                    if not np.isfinite(score_t).all():
                        raise FloatingPointError(f"Non-finite class scores at timestep {t}.")
                    scores_list.append(score_t)
                return np.stack(scores_list, axis=1)
            sc_tr = score_all(Z_train)
            sc_va = score_all(Z_val)
            sc_te = score_all(Z_test)
            tok_tr, seq_tr = _acc_from_scores(sc_tr, y_train)
            tok_va, seq_va = _acc_from_scores(sc_va, y_val)
            tok_te, seq_te = _acc_from_scores(sc_te, y_test)
        else:
            W_last = W_list[0]
            sc_tr_last = np.einsum("np,pc->nc", Z_train[:, -1, :], W_last, optimize=True)
            sc_va_last = np.einsum("np,pc->nc", Z_val[:, -1, :], W_last, optimize=True)
            sc_te_last = np.einsum("np,pc->nc", Z_test[:, -1, :], W_last, optimize=True)
            if (not np.isfinite(sc_tr_last).all() or
                not np.isfinite(sc_va_last).all() or
                not np.isfinite(sc_te_last).all()):
                continue
            tok_tr = float((sc_tr_last.argmax(axis=1) == y_train).mean())
            tok_va = float((sc_va_last.argmax(axis=1) == y_val).mean())
            tok_te = float((sc_te_last.argmax(axis=1) == y_test).mean())
            seq_tr, seq_va, seq_te = tok_tr, tok_va, tok_te
        out = CVXHeadResult(tok_tr, tok_va, tok_te, seq_tr, seq_va, seq_te, float(beta), float(primal_sum), int(nnz))
        if best is None or out.seq_val_acc > best.seq_val_acc or (out.seq_val_acc == best.seq_val_acc and out.token_val_acc > best.token_val_acc):
            best = out
    if best is None:
        raise RuntimeError("No numerically valid CVX candidate found for this block.")
    return best


# ============================================================
# Layer-wise stacking with sweeps
# ============================================================

@dataclass
class BlockMetrics:
    dataset: str
    seed: int
    block_idx: int
    input_dim: int
    ste_token_train: float
    ste_token_val: float
    ste_token_test: float
    ste_seq_train: float
    ste_seq_val: float
    ste_seq_test: float
    ste_beta_path: float
    ste_lr: float
    cvx_token_train: float
    cvx_token_val: float
    cvx_token_test: float
    cvx_seq_train: float
    cvx_seq_val: float
    cvx_seq_test: float
    cvx_beta: float
    cvx_primal: float
    cvx_nnz: int


@dataclass
class TrainResult:
    model: ThreeLayerBlock
    lr: float
    beta_path: float
    tr_tok: float
    tr_seq: float
    va_tok: float
    va_seq: float
    te_tok: float
    te_seq: float
    Z_train_cvx: np.ndarray
    Z_val_cvx: np.ndarray
    Z_test_cvx: np.ndarray
    Z_train: np.ndarray
    Z_val: np.ndarray
    Z_test: np.ndarray


def _extract_transfer_hidden_from_block(model: ThreeLayerBlock) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        model.fc1.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        model.fc2.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        model.proj.weight.detach().cpu().numpy().astype(np.float32, copy=True),
    )


def _initialize_hidden_from_transfer(
    model: ThreeLayerBlock,
    transfer_hidden: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    w1, w2, wp = transfer_hidden
    with torch.no_grad():
        t1 = torch.from_numpy(w1).to(model.fc1.weight.device, dtype=model.fc1.weight.dtype)
        t2 = torch.from_numpy(w2).to(model.fc2.weight.device, dtype=model.fc2.weight.dtype)
        tp = torch.from_numpy(wp).to(model.proj.weight.device, dtype=model.proj.weight.dtype)
        if tuple(t1.shape) != tuple(model.fc1.weight.shape):
            raise ValueError(f"fc1 transfer shape mismatch: {tuple(t1.shape)} vs {tuple(model.fc1.weight.shape)}")
        if tuple(t2.shape) != tuple(model.fc2.weight.shape):
            raise ValueError(f"fc2 transfer shape mismatch: {tuple(t2.shape)} vs {tuple(model.fc2.weight.shape)}")
        if tuple(tp.shape) != tuple(model.proj.weight.shape):
            raise ValueError(f"proj transfer shape mismatch: {tuple(tp.shape)} vs {tuple(model.proj.weight.shape)}")
        model.fc1.weight.copy_(t1)
        model.fc2.weight.copy_(t2)
        model.proj.weight.copy_(tp)


def _fit_block_once(
    model: ThreeLayerBlock,
    *,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    device: torch.device,
    lr: float,
    beta_path: float,
    epochs: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_key = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        logits, _ = model(X_train_t.to(device))
        loss = sequence_hinge_ovr_loss(logits, y_train_t.to(device))
        if beta_path > 0.0:
            loss = loss + beta_path * block_path_reg(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        va_tok, va_seq, _ = evaluate_block(model, X_val_t, y_val_t, device)
        val_key = va_seq + 1e-4 * va_tok
        if val_key > best_val_key:
            best_val_key = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def train_block_with_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_dim: int,
    p_last: int,
    num_classes: int,
    epochs: int,
    ste_lr_grid: List[float],
    ste_beta_grid: List[float],
    pretrain_lr: float,
    pretrain_beta_path_reg: float,
    pretrain_epochs: int,
    device: torch.device,
    seed: int,
    beta_leak: float,
    threshold: float,
    last_layer_readout: str,
    init_mode: str,
) -> TrainResult:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    transfer_hidden: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    feat_tr_cvx: Optional[torch.Tensor] = None
    feat_va_cvx: Optional[torch.Tensor] = None
    feat_te_cvx: Optional[torch.Tensor] = None
    if init_mode == "pretrain":
        # Pretrain once, then transfer-initialize STE sweep candidates (fine-tune style).
        set_seed(seed)
        pre_model = ThreeLayerBlock(
            d_in=int(X_train.shape[2]), hidden_dim=hidden_dim, p_last=p_last, num_classes=num_classes,
            beta_leak=beta_leak, threshold=threshold, learn_beta=False, learn_threshold=False,
            last_layer_readout=last_layer_readout,
        ).to(device)
        _fit_block_once(
            pre_model,
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            device=device,
            lr=float(pretrain_lr),
            beta_path=float(pretrain_beta_path_reg),
            epochs=int(pretrain_epochs),
        )
        _, _, feat_tr_cvx = evaluate_block(pre_model, X_train_t, y_train_t, device)
        _, _, feat_va_cvx = evaluate_block(pre_model, X_val_t, y_val_t, device)
        _, _, feat_te_cvx = evaluate_block(pre_model, X_test_t, y_test_t, device)
        transfer_hidden = _extract_transfer_hidden_from_block(pre_model)
    elif init_mode != "gaussian":
        raise ValueError(f"Unknown init_mode={init_mode!r}. Choose from: pretrain, gaussian.")

    best_key = -1.0
    best_result: Optional[TrainResult] = None

    for beta_path in ste_beta_grid:
        for lr in ste_lr_grid:
            set_seed(seed)
            model = ThreeLayerBlock(
                d_in=int(X_train.shape[2]), hidden_dim=hidden_dim, p_last=p_last, num_classes=num_classes,
                beta_leak=beta_leak, threshold=threshold, learn_beta=False, learn_threshold=False,
                last_layer_readout=last_layer_readout,
            ).to(device)
            if transfer_hidden is not None:
                _initialize_hidden_from_transfer(model, transfer_hidden)
            _fit_block_once(
                model,
                X_train_t=X_train_t,
                y_train_t=y_train_t,
                X_val_t=X_val_t,
                y_val_t=y_val_t,
                device=device,
                lr=float(lr),
                beta_path=float(beta_path),
                epochs=int(epochs),
            )

            tr_tok, tr_seq, feat_tr = evaluate_block(model, X_train_t, y_train_t, device)
            va_tok, va_seq, feat_va = evaluate_block(model, X_val_t, y_val_t, device)
            te_tok, te_seq, feat_te = evaluate_block(model, X_test_t, y_test_t, device)
            key = va_seq + 1e-4 * va_tok
            if key > best_key:
                best_key = key
                if init_mode == "gaussian":
                    feat_tr_cvx = feat_tr
                    feat_va_cvx = feat_va
                    feat_te_cvx = feat_te
                assert feat_tr_cvx is not None and feat_va_cvx is not None and feat_te_cvx is not None
                best_result = TrainResult(
                    model=model.cpu(), lr=float(lr), beta_path=float(beta_path),
                    tr_tok=tr_tok, tr_seq=tr_seq, va_tok=va_tok, va_seq=va_seq, te_tok=te_tok, te_seq=te_seq,
                    Z_train_cvx=feat_tr_cvx.numpy().astype(np.float32),
                    Z_val_cvx=feat_va_cvx.numpy().astype(np.float32),
                    Z_test_cvx=feat_te_cvx.numpy().astype(np.float32),
                    Z_train=feat_tr.numpy().astype(np.float32),
                    Z_val=feat_va.numpy().astype(np.float32),
                    Z_test=feat_te.numpy().astype(np.float32),
                )
    assert best_result is not None
    return best_result


def run_layerwise(
    ds: SequenceDataset,
    *,
    num_blocks: int,
    p_rec: int,
    p_last: int,
    epochs: int,
    ste_lr_grid: List[float],
    ste_beta_grid: List[float],
    pretrain_lr: float,
    pretrain_beta_path_reg: float,
    pretrain_epochs: int,
    cvx_beta_grid: List[float],
    device: torch.device,
    seed: int,
    beta_leak: float,
    threshold: float,
    last_layer_readout: str,
    init_mode: str,
) -> List[BlockMetrics]:
    set_seed(seed)
    X_train_raw = ds.X_train.copy()
    X_val_raw = ds.X_val.copy()
    X_test_raw = ds.X_test.copy()
    X_train_np = X_train_raw.copy()
    X_val_np = X_val_raw.copy()
    X_test_np = X_test_raw.copy()
    metrics: List[BlockMetrics] = []
    stacked_models: List[ThreeLayerBlock] = []

    for b in range(num_blocks):
        # Intermediate blocks feed subsequent blocks, so keep transfer width at P_rec.
        # Only the final output block projects to P_last.
        block_p_last = int(p_last if b == num_blocks - 1 else p_rec)
        ste = train_block_with_sweep(
            X_train_np, ds.y_train, X_val_np, ds.y_val, X_test_np, ds.y_test,
            hidden_dim=p_rec, p_last=block_p_last, num_classes=ds.num_classes,
            epochs=epochs, ste_lr_grid=ste_lr_grid, ste_beta_grid=ste_beta_grid,
            pretrain_lr=pretrain_lr, pretrain_beta_path_reg=pretrain_beta_path_reg, pretrain_epochs=pretrain_epochs,
            device=device, seed=seed + b, beta_leak=beta_leak, threshold=threshold,
            last_layer_readout=last_layer_readout, init_mode=init_mode,
        )
        stacked_models.append(ste.model)
        stack_tr_tok, stack_tr_seq = evaluate_stacked_blocks(stacked_models, X_train_raw, ds.y_train, device)
        stack_va_tok, stack_va_seq = evaluate_stacked_blocks(stacked_models, X_val_raw, ds.y_val, device)
        stack_te_tok, stack_te_seq = evaluate_stacked_blocks(stacked_models, X_test_raw, ds.y_test, device)
        cvx = fit_cvx_head_all_timesteps(
            ste.Z_train_cvx.astype(np.float64), ds.y_train,
            ste.Z_val_cvx.astype(np.float64), ds.y_val,
            ste.Z_test_cvx.astype(np.float64), ds.y_test,
            num_classes=ds.num_classes, beta_grid=cvx_beta_grid,
        )
        metrics.append(BlockMetrics(
            dataset=ds.dataset_name, seed=seed, block_idx=b + 1, input_dim=int(X_train_np.shape[2]),
            ste_token_train=stack_tr_tok, ste_token_val=stack_va_tok, ste_token_test=stack_te_tok,
            ste_seq_train=stack_tr_seq, ste_seq_val=stack_va_seq, ste_seq_test=stack_te_seq,
            ste_beta_path=ste.beta_path, ste_lr=ste.lr,
            cvx_token_train=cvx.token_train_acc, cvx_token_val=cvx.token_val_acc, cvx_token_test=cvx.token_test_acc,
            cvx_seq_train=cvx.seq_train_acc, cvx_seq_val=cvx.seq_val_acc, cvx_seq_test=cvx.seq_test_acc,
            cvx_beta=cvx.beta, cvx_primal=cvx.primal_obj, cvx_nnz=cvx.nonzero_count,
        ))
        # Stacking: current block features become next block's input sequence.
        X_train_np = ste.Z_train.astype(np.float32)
        X_val_np = ste.Z_val.astype(np.float32)
        X_test_np = ste.Z_test.astype(np.float32)
    return metrics


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-wise 3-layer stacking benchmark with swept STE blocks and CVX heads.")
    ap.add_argument("--task", type=str, default="mnist_seq", choices=["mnist_seq", "mnist_perm_seq", "arithmetic_seq"])
    ap.add_argument("--T", type=int, default=28)
    ap.add_argument("--n_train_total", type=int, default=1024)
    ap.add_argument("--n_val", type=int, default=205)
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--arith_op", type=str, default="add")
    ap.add_argument("--arith_base", type=int, default=2)
    ap.add_argument("--n_digits", type=int, default=5)
    ap.add_argument("--num_blocks", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=128)
    ap.add_argument("--P_last", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--pretrain_epochs", type=int, default=80)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--pretrain_beta_path_reg", type=float, default=1e-3)
    ap.add_argument("--ste_lr_grid", nargs="+", type=float, default=[1e-3, 5e-3, 1e-2])
    ap.add_argument("--ste_beta_grid", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--cvx_beta_grid", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--last_layer_readout", type=str, default="membrane", choices=["membrane", "spike"])
    ap.add_argument("--init_mode", type=str, default="pretrain", choices=["pretrain", "gaussian"])
    ap.add_argument("--num_runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_csv", type=str, default="cvx_layer_wise_results.csv")
    args = ap.parse_args()

    device = choose_best_device()
    print(f"[info] device={device}")

    if args.task in ("mnist_seq", "mnist_perm_seq"):
        ds = build_mnist_seq_dataset(task=args.task, T=args.T, n_train_total=args.n_train_total, n_val=args.n_val, n_test=args.n_test, seed=args.seed)
    else:
        ds = build_arithmetic_seq_dataset(op=args.arith_op, base=args.arith_base, n_digits=args.n_digits, n_train=args.n_train_total, n_val=args.n_val, n_test=args.n_test, seed=args.seed)

    all_rows: List[BlockMetrics] = []
    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        rows = run_layerwise(
            ds, num_blocks=args.num_blocks, p_rec=args.P_rec, p_last=args.P_last, epochs=args.epochs,
            ste_lr_grid=list(args.ste_lr_grid), ste_beta_grid=list(args.ste_beta_grid),
            pretrain_lr=args.pretrain_lr, pretrain_beta_path_reg=args.pretrain_beta_path_reg, pretrain_epochs=args.pretrain_epochs,
            cvx_beta_grid=list(args.cvx_beta_grid),
            device=device, seed=seed, beta_leak=args.beta_leak, threshold=args.threshold,
            last_layer_readout=args.last_layer_readout, init_mode=args.init_mode,
        )
        for row in rows:
            print(
                f"[seed {row.seed}] block={row.block_idx} in_dim={row.input_dim} | "
                f"STE tok/seq train={row.ste_token_train:.4f}/{row.ste_seq_train:.4f} "
                f"val={row.ste_token_val:.4f}/{row.ste_seq_val:.4f} test={row.ste_token_test:.4f}/{row.ste_seq_test:.4f} "
                f"| ste_beta={row.ste_beta_path:.2e} ste_lr={row.ste_lr:.2e} || "
                f"CVX tok/seq train={row.cvx_token_train:.4f}/{row.cvx_seq_train:.4f} "
                f"val={row.cvx_token_val:.4f}/{row.cvx_seq_val:.4f} test={row.cvx_token_test:.4f}/{row.cvx_seq_test:.4f} "
                f"| cvx_beta={row.cvx_beta:.2e} nnz={row.cvx_nnz}"
            )
        all_rows.extend(rows)

    if all_rows:
        fieldnames = list(_row_to_dict(all_rows[0]).keys())
        out_path = Path(args.results_csv)
        write_header = (not out_path.exists()) or (out_path.stat().st_size == 0)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in all_rows:
                writer.writerow(_row_to_dict(row))
        print(f"[done] appended {len(all_rows)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
