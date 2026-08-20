"""Criticality-tuned LSM + two-head ridge for carry-augmented AR addition.

R: freeze a criticality-tuned reservoir, fit closed-form ridge heads on
teacher-forced (sum, carry) tokens, evaluate with autoregressive carry
rollout. R-CVX reuses the exported hidden stack as CVX ``pretraining`` init.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .LSM import LSMBaselineSeq, LSMModelConfig, choose_device, extract_lsm_hidden_weight_list
from .lsm_criticality import (
    CriticalityGrid,
    build_tuned_model,
    tune_reservoir_criticality,
    _reservoir_features,
)


def _ovr_pm1(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    Y = -torch.ones((y.shape[0], int(num_classes)), device=device)
    Y[torch.arange(y.shape[0], device=device), y.long()] = 1.0
    return Y


def _ridge_multiclass(
    *,
    Phi: torch.Tensor,
    y: torch.Tensor,
    Phi_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    ridge_lambdas: Sequence[float],
    tag: str,
) -> Tuple[torch.Tensor, float, float]:
    device = Phi.device
    nc = int(num_classes)
    Y = _ovr_pm1(y, nc, device)
    G = Phi.t() @ Phi
    RHS = Phi.t() @ Y
    eye = torch.eye(Phi.shape[1], device=device, dtype=G.dtype)
    Yv = _ovr_pm1(y_val, nc, device)
    best: Optional[Tuple[float, float, torch.Tensor]] = None
    for lam in ridge_lambdas:
        W = torch.linalg.solve(G + float(lam) * eye, RHS)
        vm = float(((Phi_val @ W - Yv) ** 2).mean().item())
        print(f"[lsm-ridge {tag}] lambda={float(lam):.4g} val_mse={vm:.6f}", flush=True)
        if best is None or vm < best[0]:
            best = (vm, float(lam), W)
    if best is None:
        raise RuntimeError(f"ridge sweep empty for {tag}.")
    val_mse, lam_sel, W_sel = best
    print(f"[lsm-ridge {tag}] SELECTED lambda={lam_sel:.4g}", flush=True)
    return W_sel, lam_sel, val_mse


def lsm_predict_sum_carry_autoregressive(
    model: LSMBaselineSeq,
    w_sum: np.ndarray,
    w_carry: np.ndarray,
    x_seq: np.ndarray,
    *,
    base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    W_sum = torch.tensor(np.asarray(w_sum, dtype=np.float64), dtype=torch.float32, device=device)
    W_carry = torch.tensor(np.asarray(w_carry, dtype=np.float64), dtype=torch.float32, device=device)
    n, T, _ = x_seq.shape
    scale = max(int(base) - 1, 1)
    x_roll = x_seq.astype(np.float32, copy=True)
    sum_pred = np.zeros((n, T), dtype=np.int64)
    carry_pred = np.zeros((n, T), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for t in range(T):
            xt = torch.tensor(x_roll, dtype=torch.float32, device=device)
            feats = _reservoir_features(model, xt)
            ft = feats[:, t, :]
            sum_scores = ft @ W_sum
            carry_scores = ft @ W_carry
            if W_sum.ndim == 1 or (W_sum.ndim == 2 and W_sum.shape[1] == 1):
                sum_pred[:, t] = (sum_scores.reshape(n) >= 0.0).to(torch.int64).cpu().numpy()
            else:
                sum_pred[:, t] = torch.argmax(sum_scores, dim=1).to(torch.int64).cpu().numpy()
            if W_carry.ndim == 1 or (W_carry.ndim == 2 and W_carry.shape[1] == 1):
                carry_pred[:, t] = (carry_scores.reshape(n) >= 0.0).to(torch.int64).cpu().numpy()
            else:
                carry_pred[:, t] = torch.argmax(carry_scores, dim=1).to(torch.int64).cpu().numpy()
            if t + 1 < T:
                x_roll[:, t + 1, 2] = carry_pred[:, t].astype(np.float32) / float(scale)
    return sum_pred, carry_pred


@dataclass
class LSMCarryARResult:
    model: LSMBaselineSeq
    hidden_weights: List[np.ndarray]
    w_sum: np.ndarray
    w_carry: np.ndarray
    selected: Dict[str, float]
    criticality: Dict[str, Any]


def lsm_fit_two_head_ar(
    *,
    x_train: np.ndarray,
    y_sum_train: np.ndarray,
    y_carry_train: np.ndarray,
    x_val: np.ndarray,
    y_sum_val: np.ndarray,
    y_carry_val: np.ndarray,
    d_in: int,
    num_sum_classes: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    seed: int,
    last_layer_readout: str,
    ridge_lambdas: Sequence[float],
    beta_leak_grid: Sequence[float],
    threshold_grid: Sequence[float],
    input_scale_grid: Sequence[float],
    reservoir_variant: str = "normalized",
    probe_subsample: int = 256,
    debug: bool = False,
) -> LSMCarryARResult:
    if int(num_sum_classes) < 2:
        raise ValueError(f"num_sum_classes must be >= 2, got {num_sum_classes}.")
    run_device = choose_device()
    base_cfg = LSMModelConfig(
        d_in=int(d_in),
        num_classes=int(num_sum_classes),
        L=int(L),
        P_rec=int(P_rec),
        P_last=int(P_last),
        K_parallel=int(K_parallel),
        last_layer_readout=str(last_layer_readout),
        reservoir_seed=int(seed),
        reservoir_variant=str(reservoir_variant),
    )
    if bool(debug):
        grid = CriticalityGrid(
            beta_leak=(float(beta_leak_grid[0]),),
            threshold=(float(threshold_grid[0]),),
            input_scale=(float(input_scale_grid[0]),),
        )
        lambdas = (float(ridge_lambdas[0]),)
    else:
        grid = CriticalityGrid(
            beta_leak=tuple(float(x) for x in beta_leak_grid),
            threshold=tuple(float(x) for x in threshold_grid),
            input_scale=tuple(float(x) for x in input_scale_grid),
        )
        lambdas = tuple(float(x) for x in ridge_lambdas)
    if len(lambdas) == 0:
        raise ValueError("ridge_lambdas is empty.")

    probe = x_train
    n_probe = int(probe_subsample)
    if n_probe <= 0:
        raise ValueError(f"probe_subsample must be positive, got {probe_subsample}.")
    if x_train.shape[0] > n_probe:
        idx = np.random.default_rng(int(seed)).choice(x_train.shape[0], size=n_probe, replace=False)
        probe = x_train[idx]

    crit = tune_reservoir_criticality(base_cfg, probe, grid=grid, device=run_device)
    model = build_tuned_model(crit, device=run_device)

    def feats(x: np.ndarray) -> torch.Tensor:
        return _reservoir_features(model, torch.tensor(x, dtype=torch.float32, device=run_device))

    F_tr = feats(x_train)
    F_va = feats(x_val)
    P = int(F_tr.shape[2])
    Phi_tr = F_tr.reshape(-1, P)
    Phi_va = F_va.reshape(-1, P)
    y_s_tr = torch.tensor(y_sum_train.reshape(-1), dtype=torch.int64, device=run_device)
    y_s_va = torch.tensor(y_sum_val.reshape(-1), dtype=torch.int64, device=run_device)
    y_c_tr = torch.tensor(y_carry_train.reshape(-1), dtype=torch.int64, device=run_device)
    y_c_va = torch.tensor(y_carry_val.reshape(-1), dtype=torch.int64, device=run_device)

    W_sum, lam_sum, mse_sum = _ridge_multiclass(
        Phi=Phi_tr,
        y=y_s_tr,
        Phi_val=Phi_va,
        y_val=y_s_va,
        num_classes=int(num_sum_classes),
        ridge_lambdas=lambdas,
        tag="sum",
    )
    W_carry, lam_carry, mse_carry = _ridge_multiclass(
        Phi=Phi_tr,
        y=y_c_tr,
        Phi_val=Phi_va,
        y_val=y_c_va,
        num_classes=2,
        ridge_lambdas=lambdas,
        tag="carry",
    )
    selected = {
        "ridge_lambda_sum": float(lam_sum),
        "ridge_lambda_carry": float(lam_carry),
        "val_mse_sum": float(mse_sum),
        "val_mse_carry": float(mse_carry),
        "beta_leak": float(crit.tuned_config.beta_leak),
        "threshold": float(crit.tuned_config.threshold),
        "input_scale": float(crit.input_scale),
        "branching_ratio": float(crit.branching_ratio),
    }
    return LSMCarryARResult(
        model=model,
        hidden_weights=extract_lsm_hidden_weight_list(model),
        w_sum=W_sum.detach().cpu().numpy().copy(),
        w_carry=W_carry.detach().cpu().numpy().copy(),
        selected=selected,
        criticality={
            "beta_leak": float(crit.tuned_config.beta_leak),
            "threshold": float(crit.tuned_config.threshold),
            "input_scale": float(crit.input_scale),
            "branching_ratio": float(crit.branching_ratio),
            "table": crit.table,
        },
    )
