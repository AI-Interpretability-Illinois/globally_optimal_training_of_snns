from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from . import cvx_parallel_Solve as cvx_par
from .cvx_solve import (
    InitializationConfig,
    _build_feature_map,
    solve_binary_l1_primal_dual,
)
from .loss_functions import LossFunction, solve_multiclass_softmax_ce_l1_primal_dual
from .ovr_cvx_metrics import multiclass_ovr_cvx_data_loss
from .ste_carry_teacher_solve import carry_teacher_forcing_token_metrics


class _CarryTFData(Protocol):
    X_train: np.ndarray
    y_sum_train: np.ndarray
    y_carry_train: np.ndarray
    X_val: np.ndarray
    y_sum_val: np.ndarray
    y_carry_val: np.ndarray
    y_sum_test: np.ndarray
    y_carry_test: np.ndarray
    X_test: np.ndarray
    num_sum_classes: int


def resolve_cvx_sum_loss_name(cvx_sum_loss: str, num_classes: int) -> str:
    s = str(cvx_sum_loss)
    nc = int(num_classes)
    if s == "auto":
        return "hinge" if nc == 2 else "ce"
    if s == "hinge" and nc > 2:
        return "hinge_ovr"
    if s == "hinge_ovr" and nc == 2:
        raise ValueError("cvx_sum_loss='hinge_ovr' is only for base>2; use 'hinge' or 'ce' for binary sum.")
    return s


def resolve_cvx_carry_loss_name(cvx_carry_loss: str) -> str:
    s = str(cvx_carry_loss)
    if s == "auto":
        return "hinge"
    return s


def _binary_hinge_val_loss(scores: np.ndarray, y_01: np.ndarray) -> float:
    y_pm1 = np.where(y_01 == 1, 1.0, -1.0).astype(np.float64)
    return float(np.mean(np.maximum(0.0, 1.0 - y_pm1 * scores)))


def _binary_logistic_val_loss(scores: np.ndarray, y_01: np.ndarray) -> float:
    y_pm1 = np.where(y_01 == 1, 1.0, -1.0).astype(np.float64)
    return float(np.mean(np.log(1.0 + np.exp(-y_pm1 * scores))))


def _carry_time_ramp_alphas_1d(T: int) -> np.ndarray:
    """α_t = 2t / (T+1), t = 1 … T (1-based), same as :meth:`LossFunction.carry_time_ramp_alphas` on CPU."""
    t = np.arange(1, int(T) + 1, dtype=np.float64)
    return 2.0 * t / float(T + 1)


def _flat_ramp_row_weights(n: int, T: int) -> np.ndarray:
    """
    Non-negative weights, one per flattened row, matching ``(n, T, p).reshape(n * T, p)``:
    time index `t0 = k % T` uses α_{t0+1}. Sum equals ``n * sum_t α_t = n * T`` (so normalized weights sum to 1).
    """
    t0 = (np.arange(int(n) * int(T), dtype=np.int64) % int(T))
    return _carry_time_ramp_alphas_1d(int(T))[t0].astype(np.float64)


def _ramped_binary_val_loss(
    scores_flat: np.ndarray,
    y_01_flat: np.ndarray,
    n: int,
    T: int,
    *,
    kind: str,
) -> float:
    sc = np.asarray(scores_flat, dtype=np.float64).reshape(n, T)
    y = np.asarray(y_01_flat, dtype=np.int64).reshape(n, T)
    y_pm1 = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    if kind == "hinge":
        per = np.maximum(0.0, 1.0 - y_pm1 * sc)
    elif kind == "logistic":
        per = np.log(1.0 + np.exp(-y_pm1 * sc))
    else:
        raise ValueError(f"Invalid kind={kind!r} for _ramped_binary_val_loss.")
    l_t = per.mean(axis=0)
    alpha = _carry_time_ramp_alphas_1d(T)
    z = float(alpha.sum())
    return float((l_t * alpha).sum() / z)


def _ramped_multiclass_ce_val(sum_scores: np.ndarray, y_flat: np.ndarray, n: int, T: int) -> float:
    """Same time aggregation as :meth:`LossFunction.ste_carry_tf_data_losses_scalar` (``ramp``) for sum CE (base>2)."""
    K = int(sum_scores.shape[1])
    log = sum_scores.reshape(n, T, K)
    y2 = y_flat.reshape(n, T).astype(np.int64)
    m = np.max(log, axis=2, keepdims=True)
    ex = np.exp(log - m)
    den = ex.sum(axis=2, keepdims=True)
    p = ex / den
    i = np.arange(n, dtype=np.int64)[:, None]
    j = np.arange(T, dtype=np.int64)[None, :]
    ce_nt = -np.log(p[i, j, y2])
    l_t = ce_nt.mean(axis=0)
    alpha = _carry_time_ramp_alphas_1d(T)
    return float((l_t * alpha).sum() / float(alpha.sum()))


def _ramped_ovr_hinge_val(scores: np.ndarray, y_flat: np.ndarray, a: np.ndarray) -> float:
    """Weighted OVR-style hinge, ``sum_c sum_i a_i l_{i,c}``, matching weighted per-class binary training."""
    n_rows, C = scores.shape
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.shape[0] != n_rows or abs(float(a.sum()) - 1.0) > 1e-6:
        raise ValueError(f"Bad sample weight shape/sum: a.shape={a.shape}, sum={a.sum()}.")
    y_i = y_flat.astype(np.int64)
    total = 0.0
    for c in range(C):
        y_bin = np.where(y_i == c, 1.0, -1.0).astype(np.float64)
        s = scores[:, c]
        marg = y_bin * s
        l = np.maximum(0.0, 1.0 - marg)
        total += float(np.sum(a * l))
    return float(total)


def _build_cvx_features_for_all_timesteps(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(getattr(init_cfg, "K_parallel", 1)) > 1:
        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        d_train, d_val, d_test, _ = cvx_par._build_feature_map(x_train, x_val, x_test, pic, all_timesteps=True)
    else:
        d_train, d_val, d_test, _ = _build_feature_map(x_train, x_val, x_test, init_cfg, all_timesteps=True)
    return d_train, d_val, d_test


def _decode_cvx_preds(sum_scores: np.ndarray, carry_scores: np.ndarray, *, base: int, n: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
    if int(base) == 2:
        sum_pred = (sum_scores.reshape(n * T) >= 0.0).astype(np.int64).reshape(n, T)
    else:
        sum_pred = np.argmax(sum_scores, axis=1).astype(np.int64).reshape(n, T)
    carry_pred = (carry_scores.reshape(n * T) >= 0.0).astype(np.int64).reshape(n, T)
    return sum_pred, carry_pred


def cvx_predict_two_head_ood(
    *,
    w_sum: np.ndarray,
    w_carry: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_ood: np.ndarray,
    init_cfg: InitializationConfig,
    base: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, t, c = x_ood.shape
    if c != x_train.shape[2] or t < 1:
        raise ValueError(
            f"OOD x shape invalid: {x_ood.shape}; expect last dim d_in=={x_train.shape[2]}"
        )
    _, _, d_ood = _build_cvx_features_for_all_timesteps(x_train, x_val, x_ood, init_cfg)
    sum_s = d_ood @ w_sum
    car_s = d_ood @ w_carry
    return _decode_cvx_preds(sum_s, car_s, base=int(base), n=int(n), T=int(t))


def _ovr_hinge_solve(
    d_train: np.ndarray,
    y_tr: np.ndarray,
    rho: float,
    num_classes: int,
    sample_weight: np.ndarray | None = None,
    *,
    compute_dual: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    w = np.zeros((d_train.shape[1], num_classes), dtype=np.float64)
    primal_sum = 0.0
    dual_sum = 0.0
    for c in range(num_classes):
        y_bin = np.where(y_tr == c, 1.0, -1.0).astype(np.float64)
        sol = solve_binary_l1_primal_dual(
            d_train,
            y_bin,
            rho=rho,
            loss_name="hinge",
            sample_weight=sample_weight,
            compute_dual=compute_dual,
        )
        w[:, c] = sol.w
        primal_sum += sol.primal_obj
        dual_sum += sol.dual_obj
    gap = float(primal_sum - dual_sum) if np.isfinite(dual_sum) else float("nan")
    return w, float(primal_sum), float(dual_sum), gap


def _solve_binary_head(
    *,
    D: np.ndarray,
    y_pm1: np.ndarray,
    rho: float,
    loss_name: str,
    sample_weight: np.ndarray | None,
    cvx_method: str,
    compute_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
) -> Tuple[np.ndarray, float, float, float]:
    method = str(cvx_method)
    if method == "cvx_lite":
        from .cvx_lite import solve_binary_l1_primal_lite

        w, primal, _n_it = solve_binary_l1_primal_lite(
            D,
            y_pm1,
            rho,
            loss_name,
            max_iter=int(lite_max_iter),
            tol=float(lite_tol),
            sample_weight=sample_weight,
        )
        return np.asarray(w, dtype=np.float64), float(primal), float("nan"), float("nan")
    if method != "cvx":
        raise ValueError(f"Unknown cvx_method={cvx_method!r}. Expected cvx or cvx_lite.")
    sol = solve_binary_l1_primal_dual(
        D,
        y_pm1,
        rho,
        loss_name,
        sample_weight=sample_weight,
        compute_dual=bool(compute_dual),
    )
    return np.asarray(sol.w, dtype=np.float64), float(sol.primal_obj), float(sol.dual_obj), float(sol.gap)


def _solve_multiclass_sum_head(
    *,
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    loss_name: str,
    num_classes: int,
    sample_weight: np.ndarray | None,
    cvx_method: str,
    compute_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
) -> Tuple[np.ndarray, float, float, float]:
    method = str(cvx_method)
    if loss_name == "ce":
        if method == "cvx_lite":
            from .cvx_lite import solve_multiclass_primal_lite

            W, primal, _n_it = solve_multiclass_primal_lite(
                D,
                y,
                rho,
                "ce",
                int(num_classes),
                max_iter=int(lite_max_iter),
                tol=float(lite_tol),
                sample_weight=sample_weight,
            )
            return np.asarray(W, dtype=np.float64), float(primal), float("nan"), float("nan")
        if method != "cvx":
            raise ValueError(f"Unknown cvx_method={cvx_method!r}. Expected cvx or cvx_lite.")
        W, p, d, g = solve_multiclass_softmax_ce_l1_primal_dual(
            D, y, rho, int(num_classes), sample_weight=sample_weight, compute_dual=bool(compute_dual)
        )
        return np.asarray(W, dtype=np.float64), float(p), float(d), float(g)
    if loss_name == "hinge_ovr":
        if method == "cvx_lite":
            from .cvx_lite import solve_multiclass_primal_lite

            W, primal, _n_it = solve_multiclass_primal_lite(
                D,
                y,
                rho,
                "hinge",
                int(num_classes),
                max_iter=int(lite_max_iter),
                tol=float(lite_tol),
                sample_weight=sample_weight,
            )
            return np.asarray(W, dtype=np.float64), float(primal), float("nan"), float("nan")
        if method != "cvx":
            raise ValueError(f"Unknown cvx_method={cvx_method!r}. Expected cvx or cvx_lite.")
        return _ovr_hinge_solve(
            D, y, rho, int(num_classes), sample_weight=sample_weight, compute_dual=bool(compute_dual)
        )
    raise ValueError(f"Unsupported multiclass sum loss_name={loss_name!r}.")


def cvx_fit_shared_two_head(
    *,
    ds: _CarryTFData,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    cvx_last_layer_readout: str,
    seed: int,
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    lambda_sum: float,
    lambda_carry: float,
    cvx_device: Optional[torch.device],
    cvx_sum_loss: str = "auto",
    cvx_carry_loss: str = "auto",
    cvx_time_loss: str = "ramp",
    init_mode: str = "gaussian",
    pretrained_weights: Optional[Sequence[np.ndarray]] = None,
    cvx_method: str = "cvx",
    compute_dual: bool = False,
    lite_max_iter: int = 5000,
    lite_tol: float = 1e-6,
    beta_leak: float = 0.99,
    threshold: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    _ = cvx_device
    best_score = float("inf")
    best_bundle: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, float]] = None

    y_sum_tr = ds.y_sum_train.reshape(-1).astype(np.int64)
    y_sum_va = ds.y_sum_val.reshape(-1).astype(np.int64)
    y_carry_tr = ds.y_carry_train.reshape(-1).astype(np.int64)
    y_carry_va = ds.y_carry_val.reshape(-1).astype(np.int64)
    y_carry_te = ds.y_carry_test.reshape(-1).astype(np.int64)

    sum_n = resolve_cvx_sum_loss_name(cvx_sum_loss, ds.num_sum_classes)
    carry_n = resolve_cvx_carry_loss_name(cvx_carry_loss)
    if carry_n not in ("hinge", "ce"):
        raise ValueError(f"Invalid cvx_carry_loss (resolved)={carry_n!r}.")
    if sum_n not in ("hinge", "ce", "hinge_ovr"):
        raise ValueError(f"Invalid cvx_sum_loss (resolved)={sum_n!r}.")

    ctl = str(cvx_time_loss)
    if ctl not in ("uniform", "ramp"):
        raise ValueError(f"Invalid cvx_time_loss={ctl!r}. Expected uniform or ramp.")
    n_tr, T = int(ds.y_sum_train.shape[0]), int(ds.y_sum_train.shape[1])
    n_va, T_va = int(ds.y_sum_val.shape[0]), int(ds.y_sum_val.shape[1])
    if T_va != T:
        raise ValueError(f"Train/val T mismatch: T_train={T}, T_val={T_va}.")

    if str(init_mode) not in ("gaussian", "pretraining"):
        raise ValueError(f"init_mode must be gaussian or pretraining, got {init_mode!r}.")
    if str(init_mode) == "pretraining":
        if pretrained_weights is None or len(pretrained_weights) == 0:
            raise ValueError("init_mode='pretraining' requires pretrained_weights.")
    elif pretrained_weights is not None:
        raise ValueError("pretrained_weights is only valid when init_mode='pretraining'.")
    _pretrained_w = (
        None
        if pretrained_weights is None
        else [np.asarray(w, dtype=np.float64) for w in pretrained_weights]
    )
    if str(cvx_method) not in ("cvx", "cvx_lite"):
        raise ValueError(f"cvx_method must be cvx or cvx_lite, got {cvx_method!r}.")

    sw_train: np.ndarray | None
    a_val: np.ndarray | None
    if ctl == "ramp":
        sw_train = _flat_ramp_row_weights(n_tr, T)
        wv = _flat_ramp_row_weights(n_va, T)
        a_val = wv / float(wv.sum())
    else:
        sw_train = None
        a_val = None

    for beta in beta_grid:
        for bias in bias_grid:
            np.random.seed(seed)
            init_cfg = InitializationConfig(
                mode=str(init_mode),
                seed=seed,
                L=L,
                P_rec=P_rec,
                P_last=P_last,
                K_parallel=K_parallel,
                feature_count=P_last,
                last_layer_readout=cvx_last_layer_readout,
                bias=float(bias),
                pretrained_weights=_pretrained_w,
                beta_leak=float(beta_leak),
                threshold=float(threshold),
            )
            d_train, d_val, d_test = _build_cvx_features_for_all_timesteps(ds.X_train, ds.X_val, ds.X_test, init_cfg)

            rho_sum = float(beta) / max(float(lambda_sum), 1e-12)
            rho_carry = float(beta) / max(float(lambda_carry), 1e-12)
            p_sum: float
            d_sum: float
            g_sum: float
            w_sum: np.ndarray
            w_carry: np.ndarray
            p_carry: float
            d_carry: float
            g_carry: float

            if ds.num_sum_classes == 2:
                if sum_n not in ("hinge", "ce"):
                    raise ValueError("For binary sum (base 2) cvx_sum_loss must be auto, hinge, or ce.")
                y_tr_pm1 = np.where(y_sum_tr == 1, 1.0, -1.0).astype(np.float64)
                w_sum, p_sum, d_sum, g_sum = _solve_binary_head(
                    D=d_train,
                    y_pm1=y_tr_pm1,
                    rho=rho_sum,
                    loss_name="hinge" if sum_n == "hinge" else "ce",
                    sample_weight=sw_train,
                    cvx_method=str(cvx_method),
                    compute_dual=bool(compute_dual),
                    lite_max_iter=int(lite_max_iter),
                    lite_tol=float(lite_tol),
                )
                sum_scores_val = d_val @ w_sum
                if sum_n == "hinge":
                    if a_val is None:
                        sum_val_loss = _binary_hinge_val_loss(sum_scores_val, y_sum_va)
                    else:
                        sum_val_loss = _ramped_binary_val_loss(
                            sum_scores_val, y_sum_va, n_va, T, kind="hinge"
                        )
                else:
                    if a_val is None:
                        sum_val_loss = _binary_logistic_val_loss(sum_scores_val, y_sum_va)
                    else:
                        sum_val_loss = _ramped_binary_val_loss(
                            sum_scores_val, y_sum_va, n_va, T, kind="logistic"
                        )
            else:
                if sum_n not in ("ce", "hinge_ovr"):
                    raise ValueError("For base>2, cvx_sum_loss must be auto, ce, or hinge_ovr.")
                w_sum, p_sum, d_sum, g_sum = _solve_multiclass_sum_head(
                    D=d_train,
                    y=y_sum_tr,
                    rho=rho_sum,
                    loss_name=sum_n,
                    num_classes=int(ds.num_sum_classes),
                    sample_weight=sw_train,
                    cvx_method=str(cvx_method),
                    compute_dual=bool(compute_dual),
                    lite_max_iter=int(lite_max_iter),
                    lite_tol=float(lite_tol),
                )
                sum_scores_val = d_val @ w_sum
                if sum_n == "ce":
                    if a_val is None:
                        logits = torch.tensor(sum_scores_val, dtype=torch.float32)
                        labels = torch.tensor(y_sum_va, dtype=torch.long)
                        sum_val_loss = float(LossFunction.ce(labels, logits).item())
                    else:
                        sum_val_loss = _ramped_multiclass_ce_val(sum_scores_val, y_sum_va, n_va, T)
                else:
                    if a_val is None:
                        sum_val_loss = multiclass_ovr_cvx_data_loss("hinge_ovr", sum_scores_val, y_sum_va)
                    else:
                        sum_val_loss = _ramped_ovr_hinge_val(sum_scores_val, y_sum_va, a_val)

            c_loss_name = "hinge" if carry_n == "hinge" else "ce"
            w_carry, p_carry, d_carry, g_carry = _solve_binary_head(
                D=d_train,
                y_pm1=np.where(y_carry_tr == 1, 1.0, -1.0).astype(np.float64),
                rho=rho_carry,
                loss_name=c_loss_name,
                sample_weight=sw_train,
                cvx_method=str(cvx_method),
                compute_dual=bool(compute_dual),
                lite_max_iter=int(lite_max_iter),
                lite_tol=float(lite_tol),
            )
            carry_scores_val = d_val @ w_carry
            if carry_n == "hinge":
                if a_val is None:
                    carry_val_loss = _binary_hinge_val_loss(carry_scores_val, y_carry_va)
                else:
                    carry_val_loss = _ramped_binary_val_loss(
                        carry_scores_val, y_carry_va, n_va, T, kind="hinge"
                    )
            else:
                if a_val is None:
                    carry_val_loss = _binary_logistic_val_loss(carry_scores_val, y_carry_va)
                else:
                    carry_val_loss = _ramped_binary_val_loss(
                        carry_scores_val, y_carry_va, n_va, T, kind="logistic"
                    )

            score = float(lambda_sum) * float(sum_val_loss) + float(lambda_carry) * float(carry_val_loss)

            if score < best_score:
                best_score = score
                best_params = {"beta": float(beta), "bias": float(bias)}
                best_bundle = {
                    "init_cfg": init_cfg,
                    "sum_weights": w_sum,
                    "carry_weights": w_carry,
                    "primal_value": float(lambda_sum) * float(p_sum) + float(lambda_carry) * float(p_carry),
                    "dual_value": float(lambda_sum) * float(d_sum) + float(lambda_carry) * float(d_carry),
                    "gap": float(lambda_sum) * float(g_sum) + float(lambda_carry) * float(g_carry),
                    "cvx_sum_loss": sum_n,
                    "cvx_carry_loss": carry_n,
                    "cvx_time_loss": ctl,
                    "cvx_method": str(cvx_method),
                    "init_mode": str(init_mode),
                }
    if best_bundle is None or best_params is None:
        raise RuntimeError("No CVX candidate found.")

    init_cfg = best_bundle["init_cfg"]
    _, _, d_test = _build_cvx_features_for_all_timesteps(ds.X_train, ds.X_val, ds.X_test, init_cfg)
    w_sum = best_bundle["sum_weights"]
    w_carry = best_bundle["carry_weights"]
    sum_scores_test = d_test @ w_sum
    carry_scores_test = d_test @ w_carry
    n, t_dim = ds.y_sum_test.shape
    sum_pred, carry_pred = _decode_cvx_preds(sum_scores_test, carry_scores_test, base=ds.num_sum_classes, n=n, T=t_dim)
    test_metrics = carry_teacher_forcing_token_metrics(sum_pred, carry_pred, ds.y_sum_test, ds.y_carry_test)
    return best_bundle, best_params, test_metrics
