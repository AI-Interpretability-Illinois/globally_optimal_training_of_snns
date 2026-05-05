#!/usr/bin/env python3
"""
DFA with **raw symbol** inputs (no state): one output head, hinge on the **last** timestep
(binary string accept), matching ``LossFunction.hinge`` for ``y`` of shape (N,).

**Five-stage pipeline** (same order as ``run_arithmetic_add_carry_finetune_bench`` and
``run_dfa_state_label_finetune_bench``), with **last-step** metrics only (no teacher-forcing /
full-sequence eval):

1. STE pretrain (split A)
2. CVX (binary hinge, L1) on LIF readout with init from (1)
3. STE finetune (split B) with full warm-start from (1)
4. CVX (binary hinge) from **Gaussian** LIF init, split A
5. STE finetune (split B) on a fresh SNN with readout ``w`` from (4), branches re-initialized

Defaults for ``T``, sample counts, and hyperparameter grids are aligned with
``run_dfa_state_label_finetune_bench.py``; ``--last_layer_readout`` defaults to **spike** here.
OOD at ``T_ood = {2,5,10} * T_train`` (configurable) includes per-block independent metrics,
``mean_first_wrong_block_among_error_seq`` (0-based block index; end sentinel ``m`` if no errors),
and counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __package__ in (None, ""):
    from data_loaders.dfa_data_loader import DFA, get_dfa, make_dfa_dataset
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers.cvx_solve import InitializationConfig, _build_feature_map, solve_binary_l1_primal_dual
    from solvers import cvx_parallel_Solve as cvx_par
    from solvers.loss_functions import LossFunction, solve_multiclass_softmax_ce_l1_primal_dual
else:
    from .data_loaders.dfa_data_loader import DFA, get_dfa, make_dfa_dataset
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers.cvx_solve import InitializationConfig, _build_feature_map, solve_binary_l1_primal_dual
    from .solvers import cvx_parallel_Solve as cvx_par
    from .solvers.loss_functions import LossFunction, solve_multiclass_softmax_ce_l1_primal_dual


def _load_arith_module() -> Any:
    try:
        import run_arithmetic_add_carry_finetune_bench as m

        return m
    except ImportError:
        pass
    name = "run_arithmetic_add_carry_finetune_bench"
    p = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load run_arithmetic_add_carry_finetune_bench")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ARITH = _load_arith_module()


# ---------------------------------------------------------------------------#
# SNN: one binary head, last-timestep hinge in LossFunction
# ---------------------------------------------------------------------------#


def _parallel_branch_width(total_width: int, k_parallel: int, name: str) -> int:
    if k_parallel <= 0:
        raise ValueError(f"K_parallel must be positive, got {k_parallel}.")
    if total_width % k_parallel != 0:
        raise ValueError(
            f"{name}={total_width} must be divisible by K_parallel={k_parallel} for equal-width parallel subnetworks."
        )
    return total_width // k_parallel


class _SNNParallelBranch(nn.Module):
    def __init__(self, d_in: int, hidden_dims: List[int], beta_leak: float, threshold: float):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        in_dim = d_in
        for h_dim in hidden_dims:
            self.fcs.append(nn.Linear(in_dim, h_dim, bias=False))
            self.lifs.append(snn.Leaky(beta=beta_leak, threshold=threshold))
            in_dim = h_dim


class DFALastStepSNN(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        L: int,
        P_rec: int,
        P_last: int,
        K_parallel: int,
        beta_leak: float,
        threshold: float,
        last_layer_readout: str,
        num_head_classes: int = 1,
    ) -> None:
        super().__init__()
        c = int(num_head_classes)
        if c < 1:
            raise ValueError(f"num_head_classes must be >= 1, got {c}.")
        self.k_parallel = int(K_parallel)
        self.last_layer_readout = str(last_layer_readout)
        self.num_head_classes = c
        self._multiclass = c > 1
        sub_p_rec = _parallel_branch_width(int(P_rec), self.k_parallel, "P_rec")
        sub_p_last = _parallel_branch_width(int(P_last), self.k_parallel, "P_last")
        hidden_dims = [sub_p_rec] * max(int(L) - 2, 0) + [sub_p_last]
        if int(L) <= 1:
            hidden_dims = [sub_p_last]
        self.branches = nn.ModuleList(
            [_SNNParallelBranch(d_in, hidden_dims, beta_leak, threshold) for _ in range(self.k_parallel)]
        )
        out_ch = 1 if c == 1 else c
        self.head = nn.Linear(int(P_last), int(out_ch), bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, steps, _ = x_seq.shape
        branch_mems = [
            [lif.init_leaky().to(x_seq.device) for lif in branch.lifs]
            for branch in self.branches
        ]
        logits: List[torch.Tensor] = []
        for t in range(steps):
            x_t = x_seq[:, t, :]
            readouts: List[torch.Tensor] = []
            for b_idx, branch in enumerate(self.branches):
                h = x_t
                last_spk = None
                last_mem = None
                for i, (fc, lif) in enumerate(zip(branch.fcs, branch.lifs)):
                    spk, mem = lif(fc(h), branch_mems[b_idx][i])
                    branch_mems[b_idx][i] = mem
                    h = spk
                    last_spk = spk
                    last_mem = mem
                if last_spk is None or last_mem is None:
                    raise RuntimeError("Empty SNN branch.")
                readouts.append(last_mem if self.last_layer_readout == "membrane" else last_spk)
            h_cat = torch.cat(readouts, dim=1)
            logits.append(self.head(h_cat))
        return torch.stack(logits, dim=1)


def dfa_last_path_reg(model: DFALastStepSNN) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    v_parts: List[torch.Tensor] = []
    for branch in model.branches:
        in_dim = branch.fcs[0].in_features
        v = torch.ones(in_dim, device=device, dtype=dtype)
        for fc in branch.fcs:
            v = (fc.weight**2) @ v
        v_parts.append(v)
    v_full = torch.cat(v_parts, dim=0)
    head_sq = (model.head.weight**2).sum(dim=0)
    reg_sq = (head_sq * v_full).sum()
    return torch.sqrt(reg_sq + 1e-12)


@dataclass
class DFALastDS:
    X: np.ndarray
    y: np.ndarray
    d_in: int


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def onehot_batch_to_symbol_strings(X: np.ndarray, alphabet: List[str]) -> List[List[str]]:
    d = len(alphabet)
    n, t_dim, dtot = X.shape
    if dtot != d:
        raise ValueError(f"Expected d_in={d} (|alphabet|), got {dtot}.")
    out: List[List[str]] = []
    for i in range(n):
        row: List[str] = []
        for t in range(t_dim):
            j = int(X[i, t].argmax())
            if X[i, t, j] <= 0.0:
                raise ValueError("Invalid one-hot row (no active symbol).")
            row.append(alphabet[j])
        out.append(row)
    return out


def last_step_class_pred(model: DFALastStepSNN, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32, device=device))
    last = z[:, -1, :]
    if int(model.head.out_features) == 1:
        return (last[:, 0] >= 0.0).long().cpu().numpy().astype(np.int64)
    return last.argmax(dim=-1).cpu().numpy().astype(np.int64)


def last_step_binary_pred(model: DFALastStepSNN, x: np.ndarray) -> np.ndarray:
    if int(model.head.out_features) != 1:
        raise ValueError("last_step_binary_pred requires a single binary logit (num_head_classes=1).")
    return last_step_class_pred(model, x)


def last_step_val_objective(model: DFALastStepSNN, x: np.ndarray, y: np.ndarray, beta: float) -> float:
    """Validation loss: binary hinge+reg, or (multiclass) CE+reg for ``num_head_classes>1``."""
    device = next(model.parameters()).device
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32, device=device))
        if int(model.head.out_features) == 1:
            h = LossFunction.hinge(y_t, z)
        else:
            h = LossFunction.ce(y_t, z)
        reg = dfa_last_path_reg(model) if float(beta) > 0.0 else torch.zeros((), device=device, dtype=h.dtype)
        total = h + float(beta) * reg
    return float(total.item())


def last_step_hinge_val(model: DFALastStepSNN, x: np.ndarray, y: np.ndarray, beta: float) -> float:
    if int(model.head.out_features) != 1:
        raise ValueError("last_step_hinge_val is for binary; use last_step_val_objective for multiclass CE.")
    return last_step_val_objective(model, x, y, beta)


def _extract_laststep_branch_weights(model: DFALastStepSNN) -> List[np.ndarray]:
    """Hidden LIF weights only (no readout head), for CVX ``pretraining`` init."""
    out: List[np.ndarray] = []
    for br in model.branches:
        for fc in br.fcs:
            out.append(fc.weight.detach().cpu().numpy().astype(np.float64, copy=True))
    return out


def ste_sweep_laststep(
    tr: DFALastDS,
    va: DFALastDS,
    te: DFALastDS,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    last_layer_readout: str,
    ste_epochs: int,
    batch_size: int,
    optimizer_name: str,
    beta_leak: float,
    threshold: float,
    seed: int,
    ste_lr_grid: Sequence[float],
    ste_beta_grid: Sequence[float],
    *,
    num_head_classes: int = 1,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    head_weight: Optional[np.ndarray] = None,
) -> Tuple[DFALastStepSNN, Dict[str, float], float]:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu"))
    )
    best: Optional[float] = None
    best_p: Optional[Dict[str, float]] = None
    best_st: Optional[Dict[str, torch.Tensor]] = None
    d_in = tr.d_in
    nhc = int(num_head_classes)
    if nhc < 1:
        raise ValueError("num_head_classes must be >= 1.")
    n = tr.X.shape[0]
    bs = n if int(batch_size) < 0 else int(batch_size)
    rng = np.random.default_rng(seed)
    xtr = torch.tensor(tr.X, dtype=torch.float32, device=device)
    ytr = torch.tensor(tr.y, dtype=torch.long, device=device)
    for lr in ste_lr_grid:
        for bbeta in ste_beta_grid:
            _set_seed(seed)
            model = DFALastStepSNN(
                d_in=d_in,
                L=L,
                P_rec=P_rec,
                P_last=P_last,
                K_parallel=K_parallel,
                beta_leak=beta_leak,
                threshold=threshold,
                last_layer_readout=last_layer_readout,
                num_head_classes=nhc,
            ).to(device)
            if init_state_dict is not None:
                model.load_state_dict({k: v.to(device) for k, v in init_state_dict.items()})
            if head_weight is not None:
                hw = torch.as_tensor(
                    head_weight, dtype=model.head.weight.dtype, device=device
                )
                if hw.ndim == 1:
                    hw = hw.reshape(1, -1)
                if tuple(hw.shape) != tuple(model.head.weight.shape):
                    raise ValueError(
                        f"head_weight shape {tuple(hw.shape)} != head {tuple(model.head.weight.shape)}"
                    )
                model.head.weight.data.copy_(hw)
            if str(optimizer_name).lower() == "sgd":
                opt: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
            else:
                opt = torch.optim.Adam(model.parameters(), lr=float(lr))
            sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)
            for _ in range(int(ste_epochs)):
                model.train()
                perm = np.arange(n) if bs >= n else rng.permutation(n)
                for s0 in range(0, n, bs):
                    idx = perm[s0 : s0 + bs]
                    xb = xtr[idx]
                    yb = ytr[idx]
                    z = model(xb)
                    h = (
                        LossFunction.hinge(yb, z)
                        if int(model.head.out_features) == 1
                        else LossFunction.ce(yb, z)
                    )
                    reg = dfa_last_path_reg(model) if float(bbeta) > 0.0 else torch.zeros((), device=device, dtype=h.dtype)
                    loss = h + float(bbeta) * reg
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                va_h = last_step_val_objective(model, va.X, va.y, bbeta)
                sched.step(va_h)
            val = last_step_val_objective(model, va.X, va.y, bbeta)
            if best is None or val < best:
                best = val
                best_p = {"lr": float(lr), "beta": float(bbeta)}
                best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_p is None or best_st is None:
        raise RuntimeError("STE sweep found no last-step model.")

    _set_seed(seed)
    m = DFALastStepSNN(
        d_in=d_in,
        L=L,
        P_rec=P_rec,
        P_last=P_last,
        K_parallel=K_parallel,
        beta_leak=beta_leak,
        threshold=threshold,
        last_layer_readout=last_layer_readout,
        num_head_classes=nhc,
    ).to(device)
    m.load_state_dict(best_st)
    test_h = last_step_val_objective(m, te.X, te.y, best_p["beta"])
    return m, best_p, test_h


def _first_wrong_block_stats(
    pr_block: np.ndarray,
    y_block: np.ndarray,
) -> Dict[str, float]:
    """
    Per-sequence over blocks of length m: index (0..m-1) of the first block where
    ``pr_block`` disagrees with ``y_block``. Used for OOD length generalization
    (same idea as mean-first-wrong *timestep* in carry/arith metrics).

    If a sequence is correct on all blocks, it is excluded from the
    *among_error* mean. If *no* sequence has any error, the mean is ``m`` (end sentinel,
    like ``T`` in carry metrics when the error set is empty).
    """
    if pr_block.shape != y_block.shape:
        raise ValueError(
            f"pr_block {pr_block.shape} and y_block {y_block.shape} must match."
        )
    n, m = int(pr_block.shape[0]), int(pr_block.shape[1])
    if n < 1 or m < 1:
        raise ValueError("Expected at least one sequence and one block.")
    pr_ = pr_block.reshape(n, m).astype(np.int64, copy=False)
    y_ = y_block.reshape(n, m).astype(np.int64, copy=False)
    first_wrong: List[int] = []
    n_all_right = 0
    for i in range(n):
        diff = pr_[i] != y_[i]
        if bool(np.any(diff)):
            first_wrong.append(int(np.argmax(diff)))
        else:
            n_all_right += 1
    n_err = len(first_wrong)
    if n_err == 0:
        return {
            "mean_first_wrong_block_among_error_seq": float(m),
            "std_first_wrong_block_among_error_seq": 0.0,
            "n_seq": float(n),
            "n_seq_with_any_block_error": 0.0,
            "n_seq_all_blocks_correct": float(n_all_right),
        }
    return {
        "mean_first_wrong_block_among_error_seq": float(np.mean(first_wrong)),
        "std_first_wrong_block_among_error_seq": float(np.std(first_wrong, ddof=0)) if n_err > 1 else 0.0,
        "n_seq": float(n),
        "n_seq_with_any_block_error": float(n_err),
        "n_seq_all_blocks_correct": float(n_all_right),
    }


def per_block_independent_ood(
    dfa: DFA,
    X_full: np.ndarray,
    model: DFALastStepSNN,
    T0: int,
) -> Dict[str, Any]:
    n, l_tot, dtot = X_full.shape
    if l_tot % T0 != 0:
        raise ValueError(
            f"OOD length {l_tot} not divisible by block size T0={T0} for per-block metrics."
        )
    m = l_tot // T0
    sy = onehot_batch_to_symbol_strings(X_full, dfa.alphabet)
    y_block = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for b in range(m):
            seg = sy[i][b * T0 : (b + 1) * T0]
            y_block[i, b] = 1 if dfa.accepts(seg) else 0
    pr_block = np.zeros((n, m), dtype=np.int64)
    block_accs: List[float] = []
    for b in range(m):
        xb = X_full[:, b * T0 : (b + 1) * T0, :]
        pr = last_step_binary_pred(model, xb)
        pr_block[:, b] = pr
        acc = float((pr == y_block[:, b]).mean())
        block_accs.append(acc)
    out: Dict[str, Any] = {
        "num_blocks": m,
        "T_block": T0,
        "block_accs": block_accs,
        "block_acc_mean": float(np.mean(block_accs)),
        **{k: float(v) for k, v in _first_wrong_block_stats(pr_block, y_block).items()},
    }
    for b, acc in enumerate(block_accs):
        out[f"block_independent_{b}_last_acc"] = acc
    return out


def _build_lif_last_step_features(
    x_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
    init_cfg: InitializationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(init_cfg.K_parallel) > 1:
        icp = cvx_par.InitializationConfig(**asdict(init_cfg))
        dtr, dva, dte, _ = cvx_par._build_feature_map(
            x_tr, x_va, x_te, icp, all_timesteps=False
        )
    else:
        dtr, dva, dte, _ = _build_feature_map(
            x_tr, x_va, x_te, init_cfg, all_timesteps=False
        )
    return dtr, dva, dte


def _hinge_val_linear(d: np.ndarray, w: np.ndarray, y_01: np.ndarray) -> float:
    y_pm1 = np.where(y_01 == 1, 1.0, -1.0).astype(np.float64)
    s = d @ w.reshape(-1)
    return float(np.mean(np.maximum(0.0, 1.0 - y_pm1 * s)))


def _acc_linear(d: np.ndarray, w: np.ndarray, y_01: np.ndarray) -> float:
    pr = (d @ w.reshape(-1) >= 0.0).astype(np.int64)
    return float((pr == y_01.astype(np.int64)).mean())


def cvx_binary_lif_laststep_sweep(
    *,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    last_layer_readout: str,
    beta_leak: float,
    threshold: float,
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    init_mode: str,
    pretrained_weights: Optional[Sequence[np.ndarray]],
    seed: int,
) -> Dict[str, Any]:
    y_tr = y_tr.reshape(-1).astype(np.int64)
    y_va = y_va.reshape(-1).astype(np.int64)
    y_te = y_te.reshape(-1).astype(np.int64)
    y_pm1_tr = np.where(y_tr == 1, 1.0, -1.0).astype(np.float64)
    best: Optional[Tuple[float, float, int, int, float, float, float, np.ndarray, InitializationConfig]] = None
    for bi, beta in enumerate(beta_grid):
        for bj, bias in enumerate(bias_grid):
            _set_seed(int(seed))
            pre_w = None if pretrained_weights is None else [np.asarray(t, dtype=np.float64) for t in pretrained_weights]
            if str(init_mode) not in ("gaussian", "pretraining"):
                raise ValueError(f"init_mode must be gaussian or pretraining, got {init_mode!r}.")
            if str(init_mode) == "pretraining" and (pre_w is None or len(pre_w) < 1):
                raise ValueError("pretraining requires non-empty pretrained_weights (branch weights).")
            init_cfg = InitializationConfig(
                mode=str(init_mode),
                seed=int(seed),
                L=int(L),
                P_rec=int(P_rec),
                P_last=int(P_last),
                K_parallel=int(K_parallel),
                feature_count=int(P_last),
                last_layer_readout=str(last_layer_readout),
                bias=float(bias),
                beta_leak=float(beta_leak),
                threshold=float(threshold),
                variant="standard",
                pretrained_weights=pre_w,
            )
            d_tr, d_va, d_te = _build_lif_last_step_features(x_tr, x_va, x_te, init_cfg)
            if d_tr.shape[0] != y_tr.shape[0]:
                raise ValueError(f"Feature/label n mismatch: {d_tr.shape[0]} vs {y_tr.shape[0]}.")
            rho = float(beta)
            sol = solve_binary_l1_primal_dual(
                d_tr, y_pm1_tr, rho, "hinge", sample_weight=None
            )
            w_star, p_obj, d_obj, gap = sol.w, sol.primal_obj, sol.dual_obj, sol.gap
            vloss = _hinge_val_linear(d_va, w_star, y_va)
            tloss = _hinge_val_linear(d_te, w_star, y_te)
            key = (float(vloss), float(bi), float(bj))
            if best is None or key < (best[0], best[1], best[2]):
                best = (
                    float(vloss),
                    float(bi),
                    float(bj),
                    float(beta),
                    float(bias),
                    float(p_obj),
                    float(d_obj),
                    float(gap),
                    w_star.astype(np.float64, copy=True),
                    init_cfg,
                )
    if best is None:
        raise RuntimeError("CVX last-step search failed (empty grid?).")
    w_best = best[8]
    init_cfg_b = best[9]
    _dtr_b, d_va_b, d_te_b = _build_lif_last_step_features(x_tr, x_va, x_te, init_cfg_b)
    return {
        "w": w_best,
        "init_cfg": {
            "mode": str(init_cfg_b.mode),
            "variant": str(init_cfg_b.variant),
            "seed": int(init_cfg_b.seed),
            "L": int(init_cfg_b.L),
            "P_rec": int(init_cfg_b.P_rec),
            "P_last": int(init_cfg_b.P_last),
            "K_parallel": int(init_cfg_b.K_parallel),
            "last_layer_readout": str(init_cfg_b.last_layer_readout),
            "bias": float(best[4]),
        },
        "init_cfg_obj": init_cfg_b,
        "selected": {"beta_cvx": float(best[3]), "bias_cvx": float(best[4])},
        "diagnostics": {
            "primal_value": float(best[5]),
            "dual_value": float(best[6]),
            "gap": float(best[7]),
            "val_hinge_linear": float(best[0]),
        },
        "id_val": {
            "last_step_acc": _acc_linear(d_va_b, w_best, y_va),
            "hinge": _hinge_val_linear(d_va_b, w_best, y_va),
        },
        "id_test": {
            "last_step_acc": _acc_linear(d_te_b, w_best, y_te),
            "hinge": _hinge_val_linear(d_te_b, w_best, y_te),
        },
    }


def _mean_ce_numpy(scores: np.ndarray, y: np.ndarray) -> float:
    if scores.ndim != 2:
        raise ValueError(f"scores must be (n, C), got {scores.shape}.")
    y = y.reshape(-1).astype(np.int64, copy=False)
    n, _c = scores.shape
    if y.shape[0] != n:
        raise ValueError("y length mismatch.")
    m = np.max(scores, axis=1, keepdims=True)
    ex = np.exp(scores - m)
    logsum = m + np.log(np.sum(ex, axis=1, keepdims=True))
    logp = scores - logsum
    return float(-np.mean(logp[np.arange(n, dtype=np.int64), y]))


def _acc_multiclass_numpy(scores: np.ndarray, y: np.ndarray) -> float:
    y = y.reshape(-1).astype(np.int64, copy=False)
    pr = np.argmax(scores, axis=1)
    return float((pr == y).mean())


def cvx_multiclass_lif_laststep_sweep(
    *,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    last_layer_readout: str,
    beta_leak: float,
    threshold: float,
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    init_mode: str,
    pretrained_weights: Optional[Sequence[np.ndarray]],
    seed: int,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Multiclass softmax CE + L1 on LIF readout features; ``W`` has shape ``(P_last, num_classes)``.
    """
    C = int(num_classes)
    if C < 2:
        raise ValueError("cvx_multiclass_lif_laststep_sweep requires num_classes >= 2.")
    y_tr = y_tr.reshape(-1).astype(np.int64)
    y_va = y_va.reshape(-1).astype(np.int64)
    y_te = y_te.reshape(-1).astype(np.int64)
    if int(y_tr.min()) < 0 or int(y_tr.max()) >= C:
        raise ValueError(f"y_tr out of range for num_classes={C}.")
    best: Optional[Tuple[float, float, int, int, float, float, float, np.ndarray, InitializationConfig]] = None
    for bi, beta in enumerate(beta_grid):
        for bj, bias in enumerate(bias_grid):
            _set_seed(int(seed))
            pre_w = None if pretrained_weights is None else [np.asarray(t, dtype=np.float64) for t in pretrained_weights]
            if str(init_mode) not in ("gaussian", "pretraining"):
                raise ValueError(f"init_mode must be gaussian or pretraining, got {init_mode!r}.")
            if str(init_mode) == "pretraining" and (pre_w is None or len(pre_w) < 1):
                raise ValueError("pretraining requires non-empty pretrained_weights (branch weights).")
            init_cfg = InitializationConfig(
                mode=str(init_mode),
                seed=int(seed),
                L=int(L),
                P_rec=int(P_rec),
                P_last=int(P_last),
                K_parallel=int(K_parallel),
                feature_count=int(P_last),
                last_layer_readout=str(last_layer_readout),
                bias=float(bias),
                beta_leak=float(beta_leak),
                threshold=float(threshold),
                variant="standard",
                pretrained_weights=pre_w,
            )
            d_tr, d_va, d_te = _build_lif_last_step_features(x_tr, x_va, x_te, init_cfg)
            if d_tr.shape[0] != y_tr.shape[0]:
                raise ValueError(f"Feature/label n mismatch: {d_tr.shape[0]} vs {y_tr.shape[0]}.")
            rho = float(beta)
            w_star, p_obj, d_obj, gap = solve_multiclass_softmax_ce_l1_primal_dual(
                d_tr, y_tr, rho, C, sample_weight=None
            )
            s_va = d_va @ w_star
            s_te = d_te @ w_star
            vloss = _mean_ce_numpy(s_va, y_va)
            key = (float(vloss), float(bi), float(bj))
            if best is None or key < (best[0], best[1], best[2]):
                best = (
                    float(vloss),
                    float(bi),
                    float(bj),
                    float(beta),
                    float(bias),
                    float(p_obj),
                    float(d_obj),
                    float(gap),
                    w_star.astype(np.float64, copy=True),
                    init_cfg,
                )
    if best is None:
        raise RuntimeError("CVX multiclass last-step search failed (empty grid?).")
    w_best = best[8]
    init_cfg_b = best[9]
    _dtr_b, d_va_b, d_te_b = _build_lif_last_step_features(x_tr, x_va, x_te, init_cfg_b)
    s_va_b = d_va_b @ w_best
    s_te_b = d_te_b @ w_best
    return {
        "w": w_best,
        "num_classes": C,
        "init_cfg": {
            "mode": str(init_cfg_b.mode),
            "variant": str(init_cfg_b.variant),
            "seed": int(init_cfg_b.seed),
            "L": int(init_cfg_b.L),
            "P_rec": int(init_cfg_b.P_rec),
            "P_last": int(init_cfg_b.P_last),
            "K_parallel": int(init_cfg_b.K_parallel),
            "last_layer_readout": str(init_cfg_b.last_layer_readout),
            "bias": float(best[4]),
        },
        "init_cfg_obj": init_cfg_b,
        "selected": {"beta_cvx": float(best[3]), "bias_cvx": float(best[4])},
        "diagnostics": {
            "primal_value": float(best[5]),
            "dual_value": float(best[6]),
            "gap": float(best[7]),
            "val_mean_ce": float(best[0]),
        },
        "id_val": {
            "last_step_acc": _acc_multiclass_numpy(s_va_b, y_va),
            "mean_ce": _mean_ce_numpy(s_va_b, y_va),
        },
        "id_test": {
            "last_step_acc": _acc_multiclass_numpy(s_te_b, y_te),
            "mean_ce": _mean_ce_numpy(s_te_b, y_te),
        },
    }


def _ood_cvx(
    w: np.ndarray,
    init_cfg: InitializationConfig,
    dfa_spec: str,
    n_ood: int,
    T0: int,
    mult: int,
    data_seed: int,
    ood_per_block_seed: int,
) -> Dict[str, Any]:
    T1 = int(mult * T0)
    if T1 % T0 != 0:
        raise ValueError(f"T_ood={T1} not divisible by T_train={T0}.")
    Xo, yo, ncls = make_dfa_dataset(str(dfa_spec), int(n_ood), T1, seed=int(data_seed), balanced=True)
    n_tr = int(max(1, n_ood // 2))
    Xa, _, _ = make_dfa_dataset(str(dfa_spec), n_tr, T0, seed=int(data_seed) + 1, balanced=True)
    dfa = get_dfa(str(dfa_spec))
    if int(ncls) != 2 or int(Xo.shape[2]) != int(Xa.shape[2]):
        raise ValueError(f"OOD d_in/numcls mismatch: Xo {Xo.shape} ncls={ncls}.")
    _, _, d_ood = _build_lif_last_step_features(
        Xa, Xa, Xo, init_cfg
    )
    full_block: Dict[str, Any] = {
        "T_ood": T1,
        "T_train": T0,
        "multiplier": int(mult),
        "last_step_acc": _acc_linear(
            d_ood, w, yo.reshape(-1).astype(np.int64)
        ),
    }
    n0, l_tot, _ = Xo.shape
    full_block["per_block_independent"] = per_block_independent_ood_cvx(
        dfa, Xo, w, init_cfg, dfa_spec, n0, T0, l_tot, ood_per_block_seed
    )
    return full_block


def per_block_independent_ood_cvx(
    dfa: DFA,
    X_full: np.ndarray,
    w: np.ndarray,
    init_cfg: InitializationConfig,
    dfa_spec: str,
    n: int,
    T0: int,
    l_tot: int,
    feature_seed: int,
) -> Dict[str, Any]:
    n0, t_dim, _ = X_full.shape
    if t_dim != l_tot or n0 != n:
        raise ValueError(f"X_full shape {X_full.shape} != expected n={n} l_tot={l_tot}.")
    if l_tot % T0 != 0:
        raise ValueError(f"OOD length {l_tot} not divisible by T0={T0} for per-block metrics.")
    m = l_tot // T0
    n_tr_ = int(max(1, n // 2))
    Xa, _, _ = make_dfa_dataset(str(dfa_spec), n_tr_, T0, seed=feature_seed, balanced=True)
    pr_block = np.zeros((n, m), dtype=np.int64)
    y_block = np.zeros((n, m), dtype=np.int64)
    accs: List[float] = []
    for b in range(m):
        xb = X_full[:, b * T0 : (b + 1) * T0, :]
        _, _, d_blk = _build_lif_last_step_features(Xa, Xa, xb, init_cfg)
        syb = onehot_batch_to_symbol_strings(xb, dfa.alphabet)
        y_b = np.array([1 if dfa.accepts(syb[i]) else 0 for i in range(n)], dtype=np.int64)
        pr = (d_blk @ w.reshape(-1) >= 0.0).astype(np.int64)
        pr_block[:, b] = pr
        y_block[:, b] = y_b
        accs.append(float((pr == y_b).mean()))
    out: Dict[str, Any] = {
        "num_blocks": m,
        "T_block": T0,
        "block_accs": accs,
        "block_acc_mean": float(np.mean(accs)),
        **{k: float(v) for k, v in _first_wrong_block_stats(pr_block, y_block).items()},
    }
    for b, a in enumerate(accs):
        out[f"block_independent_{b}_last_acc"] = a
    return out


def _metrics_nn(
    dfa: DFA,
    model: DFALastStepSNN,
    va: DFALastDS,
    te: DFALastDS,
    best_beta: float,
    dfa_spec: str,
    n_ood: int,
    T0: int,
    ood_mults: List[int],
    ood_data_base_seed: int,
) -> Dict[str, Any]:
    y_te = te.y
    pr_te = last_step_class_pred(model, te.X)
    out: Dict[str, Any] = {
        "id_test": {
            "last_step_acc": float((pr_te == y_te).mean()),
            "val_total_objective": last_step_val_objective(model, va.X, va.y, best_beta),
            "test_total_objective": last_step_val_objective(model, te.X, te.y, best_beta),
        },
    }
    out["ood_eval"] = {}
    for mult in ood_mults:
        T1 = int(mult * T0)
        Xo, yo, ncls = make_dfa_dataset(
            str(dfa_spec), n_ood, T1, seed=ood_data_base_seed + 1000 + int(mult), balanced=True
        )
        if int(ncls) != 2 or int(Xo.shape[2]) != te.d_in:
            raise ValueError(
                f"OOD layout mismatch: train d_in={te.d_in} ood X.shape={Xo.shape} num_classes={ncls}."
            )
        pr_ood = last_step_binary_pred(model, Xo)
        bl: Dict[str, Any] = {
            "T_ood": T1,
            "T_train": T0,
            "multiplier": int(mult),
            "last_step_acc": float((pr_ood == yo).mean()),
        }
        if T1 % T0 != 0:
            raise ValueError(
                f"T_ood={T1} not divisible by T_train={T0}; use multipliers in OOD T / T0."
            )
        bl["per_block_independent"] = per_block_independent_ood(dfa, Xo, model, T0)
        out["ood_eval"][f"Ttrain{T0}_Tood{T1}_x{int(mult)}"] = bl
    return out


def _stage_cvx_ood(
    bundle: Dict[str, Any],
    dfa_spec: str,
    n_ood: int,
    T0: int,
    ood_mults: List[int],
    ood_data_base_seed: int,
) -> Dict[str, Any]:
    w = np.asarray(bundle["w"], dtype=np.float64)
    ic = cast(InitializationConfig, bundle["init_cfg_obj"])
    out: Dict[str, Any] = {
        "id_val": dict(bundle["id_val"]),
        "id_test": dict(bundle["id_test"]),
        "ood_eval": {},
    }
    for mult in ood_mults:
        T1 = int(mult * T0)
        dseed = ood_data_base_seed + 1000 + int(mult)
        pseed = ood_data_base_seed + 2000 + int(mult)
        out["ood_eval"][
            f"Ttrain{T0}_Tood{T1}_x{int(mult)}"
        ] = _ood_cvx(w, ic, dfa_spec, n_ood, T0, int(mult), dseed, pseed)
    return out


def _public_cvx_record(bundle: Dict[str, Any], ood: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_params": dict(bundle["selected"]),
        "init_cfg": dict(bundle["init_cfg"]),
        "diagnostics": dict(bundle["diagnostics"]),
        **ood,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DFA last-step: 5-stage pipeline (STE + binary CVX + finetune), last-step acc only."
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--dfa_spec", type=str, default="random_8_2")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--n_train_pre", type=int, default=2304)
    ap.add_argument("--n_val_pre", type=int, default=512)
    ap.add_argument("--n_train_ft", type=int, default=2304)
    ap.add_argument("--n_val_ft", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--n_test_ood", type=int, default=1024)
    ap.add_argument("--ood_T_multipliers", type=int, nargs="*", default=[2, 5, 10])
    ap.add_argument("--finetune_seed_offset", type=int, default=1000)
    ap.add_argument("--eval_seed_offset", type=int, default=2000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="If >0, cap n_train for both pretrain and finetune (after --debug). 0 = no cap.",
    )
    ap.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="If >0, cap n_val for pretrain and finetune. 0 = no cap.",
    )
    ap.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="If >0, cap n_test and n_test_ood. 0 = no cap.",
    )

    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=1024)
    ap.add_argument("--P_last", type=int, default=2048)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--beta_leak", type=float, default=0.80)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--last_layer_readout", choices=["membrane", "spike"], default="membrane")
    ap.add_argument("--optimizer_name", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--ste_pretrain_epochs", type=int, default=100)
    ap.add_argument("--ste_finetune_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=-1)
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=list(LR_GRID_DEFAULT))
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    if str(args.dfa_spec) == "dyck1_unbounded":
        raise ValueError("Use bounded dyck1_d* for this benchmark.")
    t_dim = int(args.T)
    n_tr, n_vap, n_trf, n_vaf, n_te, n_ood = (
        int(args.n_train_pre),
        int(args.n_val_pre),
        int(args.n_train_ft),
        int(args.n_val_ft),
        int(args.n_test),
        int(args.n_test_ood),
    )
    s_pre, s_ft = int(args.ste_pretrain_epochs), int(args.ste_finetune_epochs)
    if bool(args.debug):
        n_tr, n_vap, n_trf, n_vaf, n_te, n_ood = 64, 32, 64, 32, 32, 32
        s_pre, s_ft = 2, 2
        args.ste_lr_grid = [0.01]
        args.ste_beta_grid = [0.0]
        args.cvx_beta_grid = [0.0]

    def _cap(n: int, cap: int) -> int:
        if int(cap) > 0:
            return min(int(n), int(cap))
        return int(n)

    mtr, mva, mte = int(args.max_train_samples), int(args.max_val_samples), int(args.max_test_samples)
    if mtr > 0:
        n_tr = _cap(n_tr, mtr)
        n_trf = _cap(n_trf, mtr)
    if mva > 0:
        n_vap = _cap(n_vap, mva)
        n_vaf = _cap(n_vaf, mva)
    if mte > 0:
        n_te = _cap(n_te, mte)
        n_ood = _cap(n_ood, mte)
    for name, n in (
        ("n_train_pre (after caps)", n_tr),
        ("n_val_pre (after caps)", n_vap),
        ("n_train_ft (after caps)", n_trf),
        ("n_val_ft (after caps)", n_vaf),
        ("n_test (after caps)", n_te),
        ("n_test_ood (after caps)", n_ood),
    ):
        if n < 1:
            raise ValueError(f"Invalid {name}={n}; increase sizes or relax max_*_samples caps.")

    ood_m = [int(m) for m in args.ood_T_multipliers]
    lro = str(args.last_layer_readout)
    dfa = get_dfa(str(args.dfa_spec))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root)).expanduser().resolve()
    else:
        out_root = Path.cwd() / "sweep_results" / f"dfa_laststep_hybrid_{args.dfa_spec}_T{t_dim}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    config_dump: Dict[str, Any] = {
        "seeds": list(int(s) for s in args.seeds),
        "dfa_spec": str(args.dfa_spec),
        "T": t_dim,
        "n_train_pre": n_tr,
        "n_val_pre": n_vap,
        "n_train_ft": n_trf,
        "n_val_ft": n_vaf,
        "n_test": n_te,
        "n_test_ood": n_ood,
        "ood_T_multipliers": ood_m,
        "finetune_seed_offset": int(args.finetune_seed_offset),
        "eval_seed_offset": int(args.eval_seed_offset),
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "last_layer_readout": lro,
        "beta_leak": float(args.beta_leak),
        "threshold": float(args.threshold),
        "optimizer_name": str(args.optimizer_name),
        "ste_pretrain_epochs": s_pre,
        "ste_finetune_epochs": s_ft,
        "ste_lr_grid": list(args.ste_lr_grid),
        "ste_beta_grid": list(args.ste_beta_grid),
        "cvx_beta_grid": list(args.cvx_beta_grid),
        "cvx_bias_grid": list(args.cvx_bias_grid),
        "max_train_samples": mtr,
        "max_val_samples": mva,
        "max_test_samples": mte,
        "stages": [
            "ste_pretrain",
            "cvx_from_ste_pretrain",
            "ste_finetune_from_ste_pretrain_new_train",
            "cvx_pretrain_gaussian",
            "ste_finetune_from_cvx_pretrain_new_train",
        ],
        "debug": bool(args.debug),
    }
    (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2, default=str) + "\n")

    seed_payloads: List[Dict[str, Any]] = []
    for base_seed in [int(s) for s in args.seeds]:
        pre_seed = int(base_seed)
        ft_seed = int(base_seed) + int(args.finetune_seed_offset)
        eval_seed = int(base_seed) + int(args.eval_seed_offset)
        ood_data_base = int(eval_seed)
        print(
            f"[dfa_laststep] start seed={base_seed} (pre={pre_seed} ft={ft_seed} eval={eval_seed})",
            flush=True,
        )

        Xtr, ytr, _ = make_dfa_dataset(
            str(args.dfa_spec), n_tr, t_dim, seed=pre_seed + 11, balanced=True
        )
        Xva, yva, _ = make_dfa_dataset(
            str(args.dfa_spec), n_vap, t_dim, seed=pre_seed + 29, balanced=True
        )
        Xte, yte, _ = make_dfa_dataset(
            str(args.dfa_spec), n_te, t_dim, seed=eval_seed + 47, balanced=True
        )
        d_in = int(Xtr.shape[2])
        if int(Xva.shape[2]) != d_in or int(Xte.shape[2]) != d_in:
            raise ValueError("d_in mismatch across pretrain/val/test splits.")

        Xtrf, ytrf, _ = make_dfa_dataset(
            str(args.dfa_spec), n_trf, t_dim, seed=ft_seed + 11, balanced=True
        )
        Xvaf, yvaf, _ = make_dfa_dataset(
            str(args.dfa_spec), n_vaf, t_dim, seed=ft_seed + 29, balanced=True
        )
        if int(Xtrf.shape[2]) != d_in or int(Xvaf.shape[2]) != d_in:
            raise ValueError("d_in mismatch on finetune splits.")

        tr_p = DFALastDS(Xtr, ytr, d_in)
        va_p = DFALastDS(Xva, yva, d_in)
        te_ds = DFALastDS(Xte, yte, d_in)
        tr_f = DFALastDS(Xtrf, ytrf, d_in)
        va_f = DFALastDS(Xvaf, yvaf, d_in)

        m_ste, ste_pre_sel, _ = ste_sweep_laststep(
            tr_p,
            va_p,
            te_ds,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            ste_epochs=s_pre,
            batch_size=int(args.batch_size),
            optimizer_name=str(args.optimizer_name),
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            seed=pre_seed,
            ste_lr_grid=args.ste_lr_grid,
            ste_beta_grid=args.ste_beta_grid,
        )
        b_ste = float(ste_pre_sel["beta"])
        ev_ste = _metrics_nn(
            dfa, m_ste, va_p, te_ds, b_ste, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
        )
        br_w = _extract_laststep_branch_weights(m_ste)

        cvx_fs = cvx_binary_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="pretraining",
            pretrained_weights=br_w,
            seed=pre_seed,
        )
        ood_cvx_fs = _stage_cvx_ood(
            cvx_fs, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
        )
        m_sf, ste_ft_ste_sel, _ = ste_sweep_laststep(
            tr_f,
            va_f,
            te_ds,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            ste_epochs=s_ft,
            batch_size=int(args.batch_size),
            optimizer_name=str(args.optimizer_name),
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            seed=ft_seed,
            ste_lr_grid=args.ste_lr_grid,
            ste_beta_grid=args.ste_beta_grid,
            init_state_dict={k: v.detach().cpu().clone() for k, v in m_ste.state_dict().items()},
        )
        b_sf = float(ste_ft_ste_sel["beta"])
        ev_sf = _metrics_nn(
            dfa, m_sf, va_f, te_ds, b_sf, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
        )

        cvx_g = cvx_binary_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="gaussian",
            pretrained_weights=None,
            seed=pre_seed,
        )
        ood_cvx_g = _stage_cvx_ood(
            cvx_g, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
        )
        w_cvxg = np.asarray(cvx_g["w"], dtype=np.float64)
        m_cx, ste_ft_cvx_sel, _ = ste_sweep_laststep(
            tr_f,
            va_f,
            te_ds,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            ste_epochs=s_ft,
            batch_size=int(args.batch_size),
            optimizer_name=str(args.optimizer_name),
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            seed=ft_seed,
            ste_lr_grid=args.ste_lr_grid,
            ste_beta_grid=args.ste_beta_grid,
            init_state_dict=None,
            head_weight=w_cvxg,
        )
        b_cx = float(ste_ft_cvx_sel["beta"])
        ev_cx = _metrics_nn(
            dfa, m_cx, va_f, te_ds, b_cx, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
        )

        seed_pl: Dict[str, Any] = {
            "seed": int(base_seed),
            "dfa_spec": str(args.dfa_spec),
            "T": t_dim,
            "d_in": d_in,
            "split_seeds": {
                "pretrain_train": pre_seed + 11,
                "pretrain_val": pre_seed + 29,
                "finetune_train": ft_seed + 11,
                "finetune_val": ft_seed + 29,
                "eval_test": eval_seed + 47,
            },
            "stages": {
                "ste_pretrain": {"selected_params": ste_pre_sel, **ev_ste},
                "cvx_from_ste_pretrain": _public_cvx_record(cvx_fs, ood_cvx_fs),
                "ste_finetune_from_ste_pretrain_new_train": {
                    "selected_params": ste_ft_ste_sel,
                    **ev_sf,
                },
                "cvx_pretrain_gaussian": _public_cvx_record(cvx_g, ood_cvx_g),
                "ste_finetune_from_cvx_pretrain_new_train": {
                    "selected_params": ste_ft_cvx_sel,
                    **ev_cx,
                },
            },
        }
        seed_payloads.append(seed_pl)
        sdir = out_root / f"seed_{int(base_seed)}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "metrics.json").write_text(json.dumps(seed_pl, indent=2, default=str) + "\n")
        print(f"[dfa_laststep] finished seed={base_seed} -> {sdir / 'metrics.json'}", flush=True)

    root_payload: Dict[str, Any] = {
        "run_config": config_dump,
        "out_root": str(out_root),
        "n_seeds": int(len(seed_payloads)),
        "seeds": seed_payloads,
    }
    (out_root / "metrics.json").write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    oj = str(args.output_json).strip()
    if oj:
        Path(oj).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    print(json.dumps({"out_root": str(out_root), "n_seeds": len(seed_payloads)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
