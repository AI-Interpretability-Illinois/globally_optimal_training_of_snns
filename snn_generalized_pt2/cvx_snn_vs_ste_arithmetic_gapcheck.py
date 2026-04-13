#!/usr/bin/env python3
"""
CVX-SNN vs STE-SNN comparator on the per-timestep arithmetic sequence benchmark.

Key design choices:
- Uses the per-timestep arithmetic sequence bench.
- CVX side uses a random recurrent threshold/SNN feature extractor and solves a separate
  convex head at EACH timestep t over D^{L-2,t}; primal and dual are both solved.
- CVX sweeps beta and last-layer bias.
- STE baseline uses the path-regularized objective style from snn_p2.py and predicts a token
  at every timestep with a shared output layer.
- Reports token accuracy, exact-sequence accuracy, and duality-gap diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Needed for Python 3.14 dataclass/type introspection during dynamic import.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_arith_module_path() -> str:
    local_path = Path(__file__).with_name("arithmetic_test_bench.py")
    if local_path.exists():
        return str(local_path)
    legacy_path = Path("/mnt/data/arithmetic_test_bench_seq.py")
    if legacy_path.exists():
        return str(legacy_path)
    raise FileNotFoundError(
        "Could not find arithmetic bench module. Expected either "
        f"{local_path} or {legacy_path}."
    )


ARITH = _load_module("arith_seq_bench", _resolve_arith_module_path())


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

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


def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _col_normalize_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0, keepdims=True) + eps
    return U / norms


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

@dataclass
class SequenceDataset:
    X_train: np.ndarray  # (N,T,d)
    y_out_train: np.ndarray  # (N,T)
    y_carry_train: np.ndarray  # (N,T)
    X_val: np.ndarray
    y_out_val: np.ndarray
    y_carry_val: np.ndarray
    X_test: np.ndarray
    y_out_test: np.ndarray
    y_carry_test: np.ndarray
    dataset_name: str
    num_classes: int
    T: int
    d_in: int


def _samples_to_xy(samples, value_scale: float = 1.0):
    X_full = np.stack([s.inputs.astype(np.float32) / value_scale for s in samples], axis=0)
    # Model input is only operand digits. Carry/remainder channel is excluded from inputs.
    X = X_full[:, :, :2]
    y_out = np.stack([s.target_tokens.astype(np.int64) for s in samples], axis=0)
    carry_in = np.stack([s.inputs[:, 2].astype(np.int64) for s in samples], axis=0)
    # carry_out(t) is carry_in(t+1); final timestep has no next carry_out, set to 0.
    y_carry = np.zeros_like(y_out, dtype=np.int64)
    y_carry[:, :-1] = carry_in[:, 1:]
    y_carry[:, -1] = 0
    return X, y_out, y_carry


def build_arithmetic_dataset(
    op: str,
    base: int,
    n_digits: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> SequenceDataset:
    train_samples = ARITH.generate_samples_for_op_base_seq(op, base, n_digits, n_train, seed + 11)
    val_samples   = ARITH.generate_samples_for_op_base_seq(op, base, n_digits, n_val, seed + 29)
    test_samples  = ARITH.generate_samples_for_op_base_seq(op, base, n_digits, n_test, seed + 47)

    for s in train_samples[: min(5, len(train_samples))]:
        ARITH.verify_sample_seq(s)
    for s in val_samples[: min(5, len(val_samples))]:
        ARITH.verify_sample_seq(s)
    for s in test_samples[: min(5, len(test_samples))]:
        ARITH.verify_sample_seq(s)

    scale = max(base - 1, 1)
    X_train, y_out_train, y_carry_train = _samples_to_xy(train_samples, value_scale=scale)
    X_val, y_out_val, y_carry_val = _samples_to_xy(val_samples, value_scale=scale)
    X_test, y_out_test, y_carry_test = _samples_to_xy(test_samples, value_scale=scale)

    T = int(X_train.shape[1])
    d_in = int(X_train.shape[2])
    return SequenceDataset(
        X_train=X_train,
        y_out_train=y_out_train,
        y_carry_train=y_carry_train,
        X_val=X_val,
        y_out_val=y_out_val,
        y_carry_val=y_carry_val,
        X_test=X_test,
        y_out_test=y_out_test,
        y_carry_test=y_carry_test,
        dataset_name=f"arith_seq::{op}::base{base}::digits{n_digits}",
        num_classes=int(base),
        T=T,
        d_in=d_in,
    )


# ---------------------------------------------------------------------
# Random recurrent threshold/SNN feature extractor
# ---------------------------------------------------------------------

@dataclass
class RandomRNNHyperplanes:
    U_in_list: List[np.ndarray]
    U_rec_list: List[np.ndarray]
    U_last: np.ndarray
    last_layer_readout: str = "membrane"


def sample_random_hypers(
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    seed: int,
    normalize_hidden: bool = True,
) -> RandomRNNHyperplanes:
    rng = np.random.default_rng(seed)
    U_in_list = []
    U_rec_list = []
    d_in_l = d_in
    hidden_dims = [P_rec] * max(L - 2, 0) + [P_last]
    if L <= 1:
        hidden_dims = [P_last]
    for h_dim in hidden_dims:
        U_in = rng.normal(size=(d_in_l, h_dim)).astype(np.float32)
        U_rec = rng.normal(size=(2 * h_dim + 1, h_dim)).astype(np.float32)
        if normalize_hidden:
            U_in = _col_normalize_np(U_in)
            U_rec = _col_normalize_np(U_rec)
        U_in_list.append(U_in)
        U_rec_list.append(U_rec)
        d_in_l = h_dim
    U_last = rng.normal(size=(hidden_dims[-1], P_last)).astype(np.float32)
    U_last = _col_normalize_np(U_last)
    return RandomRNNHyperplanes(U_in_list=U_in_list, U_rec_list=U_rec_list, U_last=U_last)


@torch.no_grad()
def forward_snn_patterns_all_timesteps_with_bias(
    X_seq: torch.Tensor,
    hypers: RandomRNNHyperplanes,
    *,
    L: int,
    device: torch.device,
    last_bias: float,
) -> torch.Tensor:
    """Return binary last-layer patterns at every timestep.

    Output: (B, T, P_last)
    """
    B, T_eff, _ = X_seq.shape
    h_prev_layers: List[List[torch.Tensor]] = []
    h0 = [X_seq[:, t, :].to(device) for t in range(T_eff)]
    h_prev_layers.append(h0)

    final_readouts: List[torch.Tensor] = []
    for l in range(1, L):
        U_in = torch.from_numpy(hypers.U_in_list[l - 1]).float().to(device)
        U_rec = torch.from_numpy(hypers.U_rec_list[l - 1]).float().to(device)
        h_dim = U_in.shape[1]

        v_prev = torch.zeros(B, h_dim, device=device)
        h_prev = torch.zeros(B, h_dim, device=device)
        h_curr_list: List[torch.Tensor] = []
        v_curr_list: List[torch.Tensor] = []
        for t in range(T_eff):
            x_t = h_prev_layers[l - 1][t]
            v_in_t = x_t @ U_in
            ones = torch.ones(B, 1, device=device)
            s_prev = torch.cat([v_prev, -h_prev, -ones], dim=1)
            v_rec_t = s_prev @ U_rec
            v_t = v_in_t + v_rec_t
            h_t = (v_t >= 0.0).float()
            h_curr_list.append(h_t)
            v_curr_list.append(v_t)
            v_prev, h_prev = v_t, h_t
        h_prev_layers.append(h_curr_list)
        if l == L - 1:
            final_readouts = v_curr_list if hypers.last_layer_readout == "membrane" else h_curr_list

    if L <= 1:
        final_readouts = h_prev_layers[-1]
    U_last = torch.from_numpy(hypers.U_last).float().to(device)
    D_list = []
    for t in range(T_eff):
        D_t = (final_readouts[t] @ U_last - float(last_bias) >= 0.0).float()
        D_list.append(D_t)
    return torch.stack(D_list, dim=1)


# ---------------------------------------------------------------------
# CVX primal / dual for one-vs-rest hinge-LASSO per timestep
# ---------------------------------------------------------------------

@dataclass
class BinaryCvxSolution:
    w: np.ndarray
    primal_obj: float
    dual_obj: float
    gap: float


@dataclass
class DualityGapCheck:
    primal_obj: float
    dual_obj: float
    abs_gap: float
    rel_gap: float
    is_close: bool
    atol: float
    rtol: float


def check_duality_gap(
    primal_obj: float,
    dual_obj: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> DualityGapCheck:
    primal = float(primal_obj)
    dual = float(dual_obj)
    abs_gap = abs(primal - dual)
    scale = max(1.0, abs(primal), abs(dual))
    rel_gap = abs_gap / scale
    is_close = abs_gap <= atol + rtol * scale
    return DualityGapCheck(primal, dual, abs_gap, rel_gap, is_close, float(atol), float(rtol))


def solve_binary_hinge_lasso_primal_dual(
    D: np.ndarray,
    y_pm1: np.ndarray,
    rho: float,
    solver_order: Tuple[str, ...] = ("CLARABEL", "OSQP", "SCS"),
) -> BinaryCvxSolution:
    n, p = D.shape
    y = y_pm1.astype(np.float64)
    Df = D.astype(np.float64)

    # Primal: (1/n) sum xi + rho ||w||_1  s.t. y_i <d_i,w> >= 1 - xi_i
    w = cp.Variable(p)
    xi = cp.Variable(n, nonneg=True)
    margins = cp.multiply(y, Df @ w)
    primal_obj_expr = (1.0 / n) * cp.sum(xi) + rho * cp.norm1(w)
    primal_prob = cp.Problem(cp.Minimize(primal_obj_expr), [margins >= 1.0 - xi])

    last_err = None
    for s in solver_order:
        try:
            if s == "CLARABEL":
                primal_prob.solve(solver=cp.CLARABEL, verbose=False)
            elif s == "OSQP":
                primal_prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
            elif s == "SCS":
                primal_prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=50000)
            if w.value is not None:
                break
        except Exception as e:
            last_err = e
            continue
    if w.value is None:
        raise RuntimeError(f"Primal solver failed. Last error: {last_err}")
    w_star = np.asarray(w.value, dtype=np.float64).reshape(-1)
    if not np.isfinite(w_star).all():
        raise FloatingPointError("Non-finite primal weights (w_star).")

    # Use solver-reported primal objective to avoid extra unstable dense matmul.
    primal_val = float(primal_prob.value)
    if not np.isfinite(primal_val):
        raise FloatingPointError("Non-finite primal objective value.")

    # Dual derived from Draft_2 hinge specialization:
    # max_{lambda} - sum_i y_i lambda_i
    # s.t. y_i lambda_i in [-1/n, 0], ||D^T lambda||_inf <= rho
    lam = cp.Variable(n)
    constraints = [
        cp.multiply(y, lam) <= 0.0,
        cp.multiply(y, lam) >= -1.0 / n,
        Df.T @ lam <= rho,
        Df.T @ lam >= -rho,
    ]
    dual_prob = cp.Problem(cp.Maximize(-cp.sum(cp.multiply(y, lam))), constraints)

    last_err = None
    for s in solver_order:
        try:
            if s == "CLARABEL":
                dual_prob.solve(solver=cp.CLARABEL, verbose=False)
            elif s == "OSQP":
                dual_prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
            elif s == "SCS":
                dual_prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=50000)
            if lam.value is not None:
                break
        except Exception as e:
            last_err = e
            continue
    if lam.value is None:
        raise RuntimeError(f"Dual solver failed. Last error: {last_err}")

    dual_val = float(dual_prob.value)
    if not np.isfinite(dual_val):
        raise FloatingPointError("Non-finite dual objective value.")
    return BinaryCvxSolution(w=w_star, primal_obj=primal_val, dual_obj=dual_val, gap=float(primal_val - dual_val))


@dataclass
class MultiTimeCvxResult:
    W_out_list: List[np.ndarray]   # each (P_last, C)
    W_carry_list: List[np.ndarray] # each (P_last, C)
    out_token_train_acc: float
    out_token_val_acc: float
    out_token_test_acc: float
    carry_token_train_acc: float
    carry_token_val_acc: float
    carry_token_test_acc: float
    token_train_acc: float
    token_val_acc: float
    token_test_acc: float
    seq_train_acc: float
    seq_val_acc: float
    seq_test_acc: float
    primal_obj: float
    dual_obj: float
    gap: float
    beta: float
    bias: float
    rho: float
    nonzero_count: int
    class_gaps: List[float]


def _acc_from_scores_pair(
    scores_out: np.ndarray,
    scores_carry: np.ndarray,
    y_out: np.ndarray,
    y_carry: np.ndarray,
) -> Tuple[float, float, float, float]:
    pred_out = scores_out.argmax(axis=2)
    pred_carry = scores_carry.argmax(axis=2)
    out_ok = pred_out == y_out
    carry_ok = pred_carry == y_carry
    both_ok = out_ok & carry_ok
    out_token_acc = float(out_ok.mean())
    carry_token_acc = float(carry_ok.mean())
    token_acc = float(both_ok.mean())
    seq_acc = float(np.all(both_ok, axis=1).mean())
    return out_token_acc, carry_token_acc, token_acc, seq_acc


def solve_multiclass_ovr_hinge_lasso_all_timesteps(
    D_train_all: np.ndarray,  # (N,T,P)
    y_out_train: np.ndarray,      # (N,T)
    y_carry_train: np.ndarray,    # (N,T)
    D_val_all: np.ndarray,
    y_out_val: np.ndarray,
    y_carry_val: np.ndarray,
    D_test_all: np.ndarray,
    y_out_test: np.ndarray,
    y_carry_test: np.ndarray,
    num_classes: int,
    rho: float,
    carry_weight: float,
) -> MultiTimeCvxResult:
    N, T, P = D_train_all.shape
    W_out_list: List[np.ndarray] = []
    W_carry_list: List[np.ndarray] = []
    primal_sum = 0.0
    dual_sum = 0.0
    class_gaps: List[float] = []

    for t in range(T):
        D_train = D_train_all[:, t, :]
        W_out_t = np.zeros((P, num_classes), dtype=np.float64)
        W_carry_t = np.zeros((P, num_classes), dtype=np.float64)
        for c in range(num_classes):
            y_out_bin = np.where(y_out_train[:, t] == c, 1.0, -1.0).astype(np.float64)
            sol_out = solve_binary_hinge_lasso_primal_dual(D_train, y_out_bin, rho=rho)
            W_out_t[:, c] = sol_out.w
            primal_sum += sol_out.primal_obj
            dual_sum += sol_out.dual_obj
            class_gaps.append(sol_out.gap)

            y_carry_bin = np.where(y_carry_train[:, t] == c, 1.0, -1.0).astype(np.float64)
            sol_carry = solve_binary_hinge_lasso_primal_dual(D_train, y_carry_bin, rho=rho)
            W_carry_t[:, c] = sol_carry.w
            primal_sum += carry_weight * sol_carry.primal_obj
            dual_sum += carry_weight * sol_carry.dual_obj
            class_gaps.append(carry_weight * sol_carry.gap)
        if not np.isfinite(W_out_t).all() or not np.isfinite(W_carry_t).all():
            raise FloatingPointError(f"Non-finite classifier weights at timestep t={t}.")
        W_out_list.append(W_out_t)
        W_carry_list.append(W_carry_t)

    def _scores(D_all: np.ndarray, W_all: List[np.ndarray]) -> np.ndarray:
        score_list: List[np.ndarray] = []
        for t in range(T):
            score_t = np.einsum("np,pc->nc", D_all[:, t, :], W_all[t], optimize=True)
            if not np.isfinite(score_t).all():
                raise FloatingPointError(f"Non-finite class scores at timestep t={t}.")
            score_list.append(score_t)
        return np.stack(score_list, axis=1)

    out_train_scores = _scores(D_train_all, W_out_list)
    out_val_scores = _scores(D_val_all, W_out_list)
    out_test_scores = _scores(D_test_all, W_out_list)

    carry_train_scores = _scores(D_train_all, W_carry_list)
    carry_val_scores = _scores(D_val_all, W_carry_list)
    carry_test_scores = _scores(D_test_all, W_carry_list)

    out_token_train_acc, carry_token_train_acc, token_train_acc, seq_train_acc = _acc_from_scores_pair(
        out_train_scores, carry_train_scores, y_out_train, y_carry_train
    )
    out_token_val_acc, carry_token_val_acc, token_val_acc, seq_val_acc = _acc_from_scores_pair(
        out_val_scores, carry_val_scores, y_out_val, y_carry_val
    )
    out_token_test_acc, carry_token_test_acc, token_test_acc, seq_test_acc = _acc_from_scores_pair(
        out_test_scores, carry_test_scores, y_out_test, y_carry_test
    )

    nnz = int(sum(np.count_nonzero(np.abs(Wt) > 1e-10) for Wt in W_out_list))
    nnz += int(sum(np.count_nonzero(np.abs(Wt) > 1e-10) for Wt in W_carry_list))
    return MultiTimeCvxResult(
        W_out_list=W_out_list,
        W_carry_list=W_carry_list,
        out_token_train_acc=out_token_train_acc,
        out_token_val_acc=out_token_val_acc,
        out_token_test_acc=out_token_test_acc,
        carry_token_train_acc=carry_token_train_acc,
        carry_token_val_acc=carry_token_val_acc,
        carry_token_test_acc=carry_token_test_acc,
        token_train_acc=token_train_acc,
        token_val_acc=token_val_acc,
        token_test_acc=token_test_acc,
        seq_train_acc=seq_train_acc,
        seq_val_acc=seq_val_acc,
        seq_test_acc=seq_test_acc,
        primal_obj=float(primal_sum),
        dual_obj=float(dual_sum),
        gap=float(primal_sum - dual_sum),
        beta=float("nan"),
        bias=float("nan"),
        rho=float(rho),
        nonzero_count=nnz,
        class_gaps=class_gaps,
    )


# ---------------------------------------------------------------------
# SNN baseline: shared recurrent extractor + shared output layer at all t
# ---------------------------------------------------------------------

class SNNBaselineSeq(nn.Module):
    def __init__(
        self,
        d_in: int,
        L: int,
        P_rec: int,
        P_last: int,
        num_outputs: int,
        beta_leak: float = 0.99,
        threshold: float = 1.0,
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
        fcs: List[nn.Module] = []
        lifs: List[nn.Module] = []
        in_dim = d_in
        hidden_dims = [P_rec] * max(L - 2, 0) + [P_last]
        if L <= 1:
            hidden_dims = [P_last]
        rng_beta = np.random.default_rng(seed + 999)
        for h_dim in hidden_dims:
            fcs.append(nn.Linear(in_dim, h_dim, bias=False))
            if beta_dist == "fixed":
                beta_init = beta_leak
            else:
                beta_init = beta_leak
            lifs.append(
                snn.Leaky(
                    beta=beta_init,
                    threshold=threshold,
                    learn_beta=learn_beta,
                    learn_threshold=learn_threshold,
                )
            )
            in_dim = h_dim
        self.fcs = nn.ModuleList(fcs)
        self.lifs = nn.ModuleList(lifs)
        self.fc_out_token = nn.Linear(P_last, num_outputs, bias=False)
        self.fc_out_carry = nn.Linear(P_last, num_outputs, bias=False)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_seq.shape
        device = x_seq.device
        mems = [lif.init_leaky().to(device) for lif in self.lifs]
        logits_out_seq: List[torch.Tensor] = []
        logits_carry_seq: List[torch.Tensor] = []
        for t in range(T):
            h = x_seq[:, t, :]
            last_spk = None
            last_mem = None
            for li, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(h)
                spk, mem = lif(cur, mems[li])
                mems[li] = mem
                h = spk
                last_spk = spk
                last_mem = mem
            readout = last_mem if self.last_layer_readout == "membrane" else last_spk
            logits_out_seq.append(self.fc_out_token(readout))
            logits_carry_seq.append(self.fc_out_carry(readout))
        logits_out = torch.stack(logits_out_seq, dim=1)      # (B,T,C)
        logits_carry = torch.stack(logits_carry_seq, dim=1)  # (B,T,C)
        return logits_out, logits_carry, logits_out[:, -1, :]


def snn_path_reg(model: SNNBaselineSeq) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(model.fcs) == 0:
        reg_sq = (model.fc_out_token.weight ** 2).sum() + (model.fc_out_carry.weight ** 2).sum()
        return torch.sqrt(reg_sq + 1e-12)
    in_dim = model.fcs[0].in_features
    v = torch.ones(in_dim, device=device, dtype=dtype)
    for fc in model.fcs:
        v = (fc.weight ** 2) @ v
    reg_sq = ((model.fc_out_token.weight ** 2) * v.unsqueeze(0)).sum()
    reg_sq = reg_sq + ((model.fc_out_carry.weight ** 2) * v.unsqueeze(0)).sum()
    return torch.sqrt(reg_sq + 1e-12)


def snn_baseline_ce_loss(
    logits_out_seq: torch.Tensor,
    logits_carry_seq: torch.Tensor,
    y_out: torch.Tensor,
    y_carry: torch.Tensor,
    carry_weight: float,
) -> torch.Tensor:
    B, T, C = logits_out_seq.shape
    out_ce = F.cross_entropy(logits_out_seq.reshape(B * T, C), y_out.reshape(B * T))
    carry_ce = F.cross_entropy(logits_carry_seq.reshape(B * T, C), y_carry.reshape(B * T))
    return out_ce + carry_weight * carry_ce


def _ovr_hinge(logits_seq: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    B, T, C = logits_seq.shape
    logits_flat = logits_seq.reshape(B * T, C)
    y_flat = y.reshape(B * T)
    targets = -torch.ones_like(logits_flat)
    targets.scatter_(1, y_flat.unsqueeze(1), 1.0)
    return torch.relu(margin - targets * logits_flat).mean()


def snn_baseline_hinge_ovr_loss(
    logits_out_seq: torch.Tensor,
    logits_carry_seq: torch.Tensor,
    y_out: torch.Tensor,
    y_carry: torch.Tensor,
    carry_weight: float,
    margin: float = 1.0,
) -> torch.Tensor:
    out_h = _ovr_hinge(logits_out_seq, y_out, margin=margin)
    carry_h = _ovr_hinge(logits_carry_seq, y_carry, margin=margin)
    return out_h + carry_weight * carry_h


def snn_baseline_loss(
    logits_out_seq: torch.Tensor,
    logits_carry_seq: torch.Tensor,
    y_out: torch.Tensor,
    y_carry: torch.Tensor,
    ste_loss: str = "ce",
    carry_weight: float = 2.0,
) -> torch.Tensor:
    if ste_loss == "ce":
        return snn_baseline_ce_loss(logits_out_seq, logits_carry_seq, y_out, y_carry, carry_weight)
    if ste_loss in ("hinge", "hinge_ovr"):
        return snn_baseline_hinge_ovr_loss(logits_out_seq, logits_carry_seq, y_out, y_carry, carry_weight)
    raise ValueError(f"Unknown ste_loss={ste_loss!r}. Choose from: ce, hinge.")


@torch.no_grad()
def snn_eval_scores(
    model: SNNBaselineSeq,
    X: torch.Tensor,
    y_out: torch.Tensor,
    y_carry: torch.Tensor,
) -> Tuple[float, float, float, float]:
    model.eval()
    logits_out_seq, logits_carry_seq, _ = model(X)
    pred_out = logits_out_seq.argmax(dim=2)
    pred_carry = logits_carry_seq.argmax(dim=2)
    out_ok = pred_out == y_out
    carry_ok = pred_carry == y_carry
    both_ok = out_ok & carry_ok
    out_acc = float(out_ok.float().mean().item())
    carry_acc = float(carry_ok.float().mean().item())
    token_acc = float(both_ok.float().mean().item())
    seq_acc = float(both_ok.all(dim=1).float().mean().item())
    return out_acc, carry_acc, token_acc, seq_acc


@dataclass
class SnnTrainResult:
    out_token_train_acc: float
    out_token_val_acc: float
    out_token_test_acc: float
    carry_token_train_acc: float
    carry_token_val_acc: float
    carry_token_test_acc: float
    token_train_acc: float
    token_val_acc: float
    token_test_acc: float
    seq_train_acc: float
    seq_val_acc: float
    seq_test_acc: float
    beta_path_reg: float
    lr: float
    ste_loss: str


def train_snn_baseline(
    model: SNNBaselineSeq,
    X_train: torch.Tensor,
    y_out_train: torch.Tensor,
    y_carry_train: torch.Tensor,
    X_val: torch.Tensor,
    y_out_val: torch.Tensor,
    y_carry_val: torch.Tensor,
    X_test: torch.Tensor,
    y_out_test: torch.Tensor,
    y_carry_test: torch.Tensor,
    *,
    lr: float,
    epochs: int,
    device: torch.device,
    step_size: int,
    gamma: float,
    beta_path_reg: float = 0.0,
    ste_loss: str = "ce",
    carry_weight: float = 2.0,
) -> SnnTrainResult:
    model = model.to(device)
    X_train = X_train.to(device)
    y_out_train = y_out_train.to(device)
    y_carry_train = y_carry_train.to(device)
    X_val = X_val.to(device)
    y_out_val = y_out_val.to(device)
    y_carry_val = y_carry_val.to(device)
    X_test = X_test.to(device)
    y_out_test = y_out_test.to(device)
    y_carry_test = y_carry_test.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    best_val = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        logits_out_seq, logits_carry_seq, _ = model(X_train)
        loss = snn_baseline_loss(
            logits_out_seq, logits_carry_seq, y_out_train, y_carry_train,
            ste_loss=ste_loss, carry_weight=carry_weight,
        )
        if beta_path_reg > 0.0:
            loss = loss + beta_path_reg * snn_path_reg(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        _, _, _, val_seq = snn_eval_scores(model, X_val, y_out_val, y_carry_val)
        if val_seq > best_val:
            best_val = val_seq
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    tr_out, tr_carry, tr_tok, tr_seq = snn_eval_scores(model, X_train, y_out_train, y_carry_train)
    va_out, va_carry, va_tok, va_seq = snn_eval_scores(model, X_val, y_out_val, y_carry_val)
    te_out, te_carry, te_tok, te_seq = snn_eval_scores(model, X_test, y_out_test, y_carry_test)
    return SnnTrainResult(
        tr_out, va_out, te_out,
        tr_carry, va_carry, te_carry,
        tr_tok, va_tok, te_tok,
        tr_seq, va_seq, te_seq,
        beta_path_reg, lr, ste_loss,
    )


# ---------------------------------------------------------------------
# Experiment config and driver
# ---------------------------------------------------------------------

@dataclass
class ComparatorConfig:
    random_seed: int = 0
    n_train: int = 512
    n_val: int = 128
    n_test: int = 256
    n_digits: int = 4
    ops: Tuple[str, ...] = ("add", "sub", "mul", "div")
    bases: Tuple[int, ...] = (2, 3, 5, 7, 10)
    L: int = 3
    P_rec: int = 64
    P_last: int = 128
    epochs: int = 150
    ste_lr_grid: Tuple[float, ...] = (1e-3, 5e-3, 1e-2)
    ste_beta_grid: Tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-3)
    cvx_beta_grid: Tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
    bias_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    normalize_hidden: bool = True
    num_runs: int = 3
    ste_step_size: int = 50
    ste_gamma: float = 0.5
    last_layer_readout: str = "membrane"
    ste_loss: str = "hinge"
    carry_weight: float = 2.0


def run_one_seed(ds: SequenceDataset, cfg: ComparatorConfig, seed: int, device: torch.device) -> Dict[str, float]:
    set_seed(seed)
    hypers = sample_random_hypers(ds.d_in, cfg.L, cfg.P_rec, cfg.P_last, seed, normalize_hidden=cfg.normalize_hidden)

    Xtr = torch.tensor(ds.X_train, dtype=torch.float32)
    Xva = torch.tensor(ds.X_val, dtype=torch.float32)
    Xte = torch.tensor(ds.X_test, dtype=torch.float32)

    # CVX sweep over beta and bias using exact sequence val acc.
    best_cvx: Optional[MultiTimeCvxResult] = None
    best_beta = None
    best_bias = None
    for bias in cfg.bias_grid:
        with torch.no_grad():
            z_train = forward_snn_patterns_all_timesteps_with_bias(Xtr.to(device), hypers, L=cfg.L, device=device, last_bias=float(bias)).cpu().numpy().astype(np.float64)
            z_val = forward_snn_patterns_all_timesteps_with_bias(Xva.to(device), hypers, L=cfg.L, device=device, last_bias=float(bias)).cpu().numpy().astype(np.float64)
            z_test = forward_snn_patterns_all_timesteps_with_bias(Xte.to(device), hypers, L=cfg.L, device=device, last_bias=float(bias)).cpu().numpy().astype(np.float64)
        for beta in cfg.cvx_beta_grid:
            rho = float(beta / math.sqrt(max(cfg.P_last, 1)))
            try:
                out = solve_multiclass_ovr_hinge_lasso_all_timesteps(
                    z_train, ds.y_out_train, ds.y_carry_train,
                    z_val, ds.y_out_val, ds.y_carry_val,
                    z_test, ds.y_out_test, ds.y_carry_test,
                    ds.num_classes, rho, cfg.carry_weight,
                )
            except (FloatingPointError, ValueError, RuntimeError) as e:
                print(
                    f"[warn] Skipping CVX candidate due to numerical issue: "
                    f"beta={beta:.2e}, bias={bias:.3f}, detail={e}"
                )
                continue
            out.beta = float(beta)
            out.bias = float(bias)
            if (best_cvx is None or out.seq_val_acc > best_cvx.seq_val_acc or
                (out.seq_val_acc == best_cvx.seq_val_acc and out.token_val_acc > best_cvx.token_val_acc)):
                best_cvx = out
                best_beta = beta
                best_bias = bias
    if best_cvx is None:
        raise RuntimeError(
            "No numerically valid CVX candidate found across current beta/bias grid."
        )

    # STE baseline: always sweep lr x beta_path_reg
    Xtr_d = torch.tensor(ds.X_train, dtype=torch.float32, device=device)
    ytr_out_d = torch.tensor(ds.y_out_train, dtype=torch.long, device=device)
    ytr_carry_d = torch.tensor(ds.y_carry_train, dtype=torch.long, device=device)
    Xva_d = torch.tensor(ds.X_val, dtype=torch.float32, device=device)
    yva_out_d = torch.tensor(ds.y_out_val, dtype=torch.long, device=device)
    yva_carry_d = torch.tensor(ds.y_carry_val, dtype=torch.long, device=device)
    Xte_d = torch.tensor(ds.X_test, dtype=torch.float32, device=device)
    yte_out_d = torch.tensor(ds.y_out_test, dtype=torch.long, device=device)
    yte_carry_d = torch.tensor(ds.y_carry_test, dtype=torch.long, device=device)

    best_ste: Optional[SnnTrainResult] = None
    for beta in cfg.ste_beta_grid:
        for lr in cfg.ste_lr_grid:
            model = SNNBaselineSeq(
                d_in=ds.d_in,
                L=cfg.L,
                P_rec=cfg.P_rec,
                P_last=cfg.P_last,
                num_outputs=ds.num_classes,
                beta_leak=0.99,
                threshold=1.0,
                learn_beta=False,
                learn_threshold=False,
                last_layer_readout=cfg.last_layer_readout,
                beta_dist="fixed",
                seed=seed,
            )
            out = train_snn_baseline(
                model,
                Xtr_d, ytr_out_d, ytr_carry_d,
                Xva_d, yva_out_d, yva_carry_d,
                Xte_d, yte_out_d, yte_carry_d,
                lr=lr,
                epochs=cfg.epochs,
                device=device,
                step_size=cfg.ste_step_size,
                gamma=cfg.ste_gamma,
                beta_path_reg=beta,
                ste_loss=cfg.ste_loss,
                carry_weight=cfg.carry_weight,
            )
            if best_ste is None or out.seq_val_acc > best_ste.seq_val_acc or (out.seq_val_acc == best_ste.seq_val_acc and out.token_val_acc > best_ste.token_val_acc):
                best_ste = out
    assert best_ste is not None

    chk = check_duality_gap(best_cvx.primal_obj, best_cvx.dual_obj)
    max_class_gap = float(max(best_cvx.class_gaps)) if len(best_cvx.class_gaps) > 0 else float("nan")
    mean_class_gap = float(np.mean(best_cvx.class_gaps)) if len(best_cvx.class_gaps) > 0 else float("nan")

    return {
        "cvx_out_token_train_acc": best_cvx.out_token_train_acc,
        "cvx_out_token_val_acc": best_cvx.out_token_val_acc,
        "cvx_out_token_test_acc": best_cvx.out_token_test_acc,
        "cvx_carry_token_train_acc": best_cvx.carry_token_train_acc,
        "cvx_carry_token_val_acc": best_cvx.carry_token_val_acc,
        "cvx_carry_token_test_acc": best_cvx.carry_token_test_acc,
        "cvx_token_train_acc": best_cvx.token_train_acc,
        "cvx_token_val_acc": best_cvx.token_val_acc,
        "cvx_token_test_acc": best_cvx.token_test_acc,
        "cvx_seq_train_acc": best_cvx.seq_train_acc,
        "cvx_seq_val_acc": best_cvx.seq_val_acc,
        "cvx_seq_test_acc": best_cvx.seq_test_acc,
        "cvx_primal": chk.primal_obj,
        "cvx_dual": chk.dual_obj,
        "cvx_abs_gap": chk.abs_gap,
        "cvx_rel_gap": chk.rel_gap,
        "cvx_gap_is_close": float(chk.is_close),
        "cvx_max_class_gap": max_class_gap,
        "cvx_mean_class_gap": mean_class_gap,
        "cvx_beta": float(best_beta),
        "cvx_bias": float(best_bias),
        "cvx_nonzero": float(best_cvx.nonzero_count),
        "ste_out_token_train_acc": best_ste.out_token_train_acc,
        "ste_out_token_val_acc": best_ste.out_token_val_acc,
        "ste_out_token_test_acc": best_ste.out_token_test_acc,
        "ste_carry_token_train_acc": best_ste.carry_token_train_acc,
        "ste_carry_token_val_acc": best_ste.carry_token_val_acc,
        "ste_carry_token_test_acc": best_ste.carry_token_test_acc,
        "ste_token_train_acc": best_ste.token_train_acc,
        "ste_token_val_acc": best_ste.token_val_acc,
        "ste_token_test_acc": best_ste.token_test_acc,
        "ste_seq_train_acc": best_ste.seq_train_acc,
        "ste_seq_val_acc": best_ste.seq_val_acc,
        "ste_seq_test_acc": best_ste.seq_test_acc,
        "ste_beta_path_reg": float(best_ste.beta_path_reg),
        "ste_lr": float(best_ste.lr),
        "ste_loss_name": 1.0 if best_ste.ste_loss in ("hinge", "hinge_ovr") else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--n_digits", type=int, default=4)
    ap.add_argument("--ops", nargs="+", default=["add", "sub", "mul", "div"])
    ap.add_argument("--bases", nargs="+", type=int, default=[2, 3, 5, 7, 10])
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=64)
    ap.add_argument("--P_last", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--num_runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ste_lr_grid", nargs="+", type=float, default=[1e-3, 5e-3, 1e-2])
    ap.add_argument("--ste_beta_grid", nargs="+", type=float, default=[0.0, 1e-6, 1e-4, 1e-3])
    ap.add_argument("--cvx_beta_grid", nargs="+", type=float, default=[1e-8, 1e-6, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--bias_grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--results_csv", type=str, default="cvx_snn_vs_ste_arithmetic_seq_results.csv")
    ap.add_argument("--ste_loss", type=str, default="hinge_ovr", choices=["ce", "hinge", "hinge_ovr"])
    ap.add_argument("--carry_weight", type=float, default=2.0)
    args = ap.parse_args()

    cfg = ComparatorConfig(
        random_seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_digits=args.n_digits,
        ops=tuple(args.ops),
        bases=tuple(args.bases),
        L=args.L,
        P_rec=args.P_rec,
        P_last=args.P_last,
        epochs=args.epochs,
        ste_lr_grid=tuple(args.ste_lr_grid),
        ste_beta_grid=tuple(args.ste_beta_grid),
        cvx_beta_grid=tuple(args.cvx_beta_grid),
        bias_grid=tuple(args.bias_grid),
        num_runs=args.num_runs,
        ste_loss=args.ste_loss,
        carry_weight=args.carry_weight,
    )

    device = choose_best_device()
    print(f"[info] device={device}")
    rows: List[Dict[str, float]] = []
    for op in cfg.ops:
        for base in cfg.bases:
            ds = build_arithmetic_dataset(op, base, cfg.n_digits, cfg.n_train, cfg.n_val, cfg.n_test, cfg.random_seed)
            print("\n" + "=" * 100)
            print(f"Dataset: {ds.dataset_name} | classes={ds.num_classes} | T={ds.T} | d_in={ds.d_in}")
            print("=" * 100)
            per_seed = []
            for run_idx in range(cfg.num_runs):
                seed = cfg.random_seed + run_idx
                out = run_one_seed(ds, cfg, seed, device)
                per_seed.append(out)
                print(
                    f"[seed {seed}] CVX tok/seq train={out['cvx_token_train_acc']:.4f}/{out['cvx_seq_train_acc']:.4f} "
                    f"val={out['cvx_token_val_acc']:.4f}/{out['cvx_seq_val_acc']:.4f} "
                    f"test={out['cvx_token_test_acc']:.4f}/{out['cvx_seq_test_acc']:.4f} "
                    f"| primal={out['cvx_primal']:.6f} dual={out['cvx_dual']:.6f} "
                    f"abs_gap={out['cvx_abs_gap']:.3e} rel_gap={out['cvx_rel_gap']:.3e} ok={bool(out['cvx_gap_is_close'])} "
                    f"| beta={out['cvx_beta']:.2e} bias={out['cvx_bias']:.3f} || "
                    f"STE tok/seq train={out['ste_token_train_acc']:.4f}/{out['ste_seq_train_acc']:.4f} "
                    f"val={out['ste_token_val_acc']:.4f}/{out['ste_seq_val_acc']:.4f} "
                    f"test={out['ste_token_test_acc']:.4f}/{out['ste_seq_test_acc']:.4f} "
                    f"| ste_loss={cfg.ste_loss} beta_path={out['ste_beta_path_reg']:.2e} lr={out['ste_lr']:.2e}"
                )
            # aggregate
            agg = {"dataset": ds.dataset_name, "op": op, "base": base, "T": ds.T}
            for key in per_seed[0].keys():
                vals = [float(r[key]) for r in per_seed]
                m, s = mean_std(vals)
                agg[f"{key}_mean"] = m
                agg[f"{key}_std"] = s
            rows.append(agg)

    if rows:
        keys = list(rows[0].keys())
        out_path = Path(args.results_csv)
        write_header = (not out_path.exists()) or (out_path.stat().st_size == 0)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\n[done] appended {len(rows)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
