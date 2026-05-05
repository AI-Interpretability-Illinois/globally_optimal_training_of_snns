from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss_functions import LossFunction, solve_multiclass_softmax_ce_l1_primal_dual
from .ovr_cvx_metrics import multiclass_ovr_cvx_data_loss


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class InitializationConfig:
    mode: str = "gaussian"  # gaussian | pretraining
    variant: str = "standard"  # standard | normalized | orthogonal
    seed: int = 0
    feature_count: int = 512
    bias: float = 0.0
    pretrained_weights: Optional[List[np.ndarray]] = None
    L: int = 3
    P_rec: int = 128
    P_last: int = 64
    K_parallel: int = 1
    beta_leak: float = 0.99
    threshold: float = 1.0
    last_layer_readout: str = "membrane"


@dataclass
class SolveConfig:
    loss_name: str = "hinge_ovr"
    method: str = "cvx"  # cvx | sgd
    beta: float = 1e-2
    lr: float = 1e-3
    optimizer_name: str = "adam"
    epochs: int = 100
    batch_size: Optional[int] = None
    log_every: int = 10
    # When method=cvx and loss_name=ce, solve dual conic problem for gap/dual_obj (default off for speed).
    compute_ce_dual: bool = False
    cvx_ovr_workers: int = 1


@dataclass
class CvxDiagnostics:
    primal_value: float
    dual_value: float
    gap: float
    train_loss: float
    val_loss: float
    test_loss: float


@dataclass
class CvxSolveResult:
    trained_model: object
    loss_history: List[float]
    final_losses: Dict[str, float]
    diagnostics: CvxDiagnostics


@dataclass
class BinaryCvxSolution:
    w: np.ndarray
    primal_obj: float
    dual_obj: float
    gap: float


class LinearFeatureClassifier(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _prepare_sequence_targets(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        return y[:, -1].astype(np.int64)
    raise ValueError(f"Expected labels to be 1D or 2D, got shape={y.shape}.")


def _extract_flat_last_step(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N,T,d), got {x.shape}.")
    return x[:, -1, :].astype(np.float64)


def _hidden_dims_like_snn_p2(L: int, P_rec: int, P_last: int) -> List[int]:
    if L <= 1:
        return [P_last]
    return [P_rec] * max(L - 2, 0) + [P_last]


def _col_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=0, keepdims=True) + eps
    return mat / norms


def _sample_weight_matrix(rng: np.random.Generator, in_dim: int, out_dim: int, variant: str) -> np.ndarray:
    w = rng.standard_normal((in_dim, out_dim)).astype(np.float64)
    if variant == "normalized":
        return _col_normalize(w)
    if variant == "orthogonal":
        q, _ = np.linalg.qr(w)
        if q.shape[1] >= out_dim:
            return q[:, :out_dim]
        # Rare rectangular case where QR reduced columns below target.
        extra = rng.standard_normal((in_dim, out_dim - q.shape[1])).astype(np.float64)
        return np.concatenate([q, _col_normalize(extra)], axis=1)
    if variant == "standard":
        return w
    raise ValueError(f"Unknown gaussian variant={variant}.")


def _coerce_hidden_weight(weight: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
    if tuple(weight.shape) == (out_dim, in_dim):
        return weight.T.astype(np.float64)
    if tuple(weight.shape) == (in_dim, out_dim):
        return weight.astype(np.float64)
    raise ValueError(
        f"Hidden weight shape mismatch: got {tuple(weight.shape)}, expected ({out_dim}, {in_dim}) or ({in_dim}, {out_dim})."
    )


def _run_lif_stack_readout(
    x_seq: np.ndarray,
    U_in_list: List[np.ndarray],
    beta_list: List[np.ndarray],
    threshold_list: List[np.ndarray],
    *,
    last_layer_readout: str,
) -> np.ndarray:
    if x_seq.ndim != 3:
        raise ValueError(f"Expected x_seq shape (N,T,d_in), got {x_seq.shape}.")
    n, T, _ = x_seq.shape
    h_prev = [x_seq[:, t, :].astype(np.float64) for t in range(T)]
    final_spk = None
    final_mem = None
    for l, U_in in enumerate(U_in_list):
        h_dim = U_in.shape[1]
        beta = beta_list[l].reshape(1, h_dim)
        thr = threshold_list[l].reshape(1, h_dim)
        mem = np.zeros((n, h_dim), dtype=np.float64)
        h_curr: List[np.ndarray] = []
        mem_curr: List[np.ndarray] = []
        for t in range(T):
            cur = h_prev[t] @ U_in
            mem = beta * mem + cur
            spk = (mem - thr >= 0.0).astype(np.float64)
            mem = mem - thr * spk
            h_curr.append(spk)
            mem_curr.append(mem.copy())
        h_prev = h_curr
        if l == len(U_in_list) - 1:
            final_spk = h_curr[-1]
            final_mem = mem_curr[-1]
    if final_spk is None or final_mem is None:
        raise RuntimeError("No hidden layers were executed for LIF stack readout.")
    if last_layer_readout == "membrane":
        return final_mem
    if last_layer_readout == "spike":
        return final_spk
    raise ValueError(f"Unsupported last_layer_readout={last_layer_readout}. Expected membrane|spike.")


def _run_lif_stack_readout_all_timesteps(
    x_seq: np.ndarray,
    u_in_list: Sequence[np.ndarray],
    beta_list: Sequence[np.ndarray],
    threshold_list: Sequence[np.ndarray],
    *,
    last_layer_readout: str,
) -> np.ndarray:
    """
    Return the last hidden layer readout at every timestep, shape (N, T, H_last).

    Same LIF dynamics as _run_lif_stack_readout per layer: for each layer, membrane
    ``mem`` is carried across sequence time; input at (t, l) is spikes from below at
    time t (or x[:, t] for l == 0). Update matches _run_lif_stack_readout:
    mem = beta * mem + (h_below @ U_in); spk = 1[mem - thr >= 0]; mem -= thr * spk.
    """
    if x_seq.ndim != 3:
        raise ValueError(f"Expected x_seq shape (N,T,d_in), got {x_seq.shape}.")
    n, steps, _ = x_seq.shape
    if len(u_in_list) == 0:
        raise ValueError("u_in_list must be non-empty.")
    n_layers = len(u_in_list)
    mem_prev: List[np.ndarray] = [np.zeros((n, u_in_list[l].shape[1]), dtype=np.float64) for l in range(n_layers)]
    h_prev: List[np.ndarray] = [np.zeros((n, u_in_list[l].shape[1]), dtype=np.float64) for l in range(n_layers)]
    readouts: List[np.ndarray] = []
    for t in range(steps):
        h_below = x_seq[:, t, :].astype(np.float64, copy=False)
        final_mem_t: Optional[np.ndarray] = None
        final_spk_t: Optional[np.ndarray] = None
        for l, u_in in enumerate(u_in_list):
            h_dim = u_in.shape[1]
            beta = beta_list[l].reshape(1, h_dim)
            thr = threshold_list[l].reshape(1, h_dim)
            cur = h_below @ u_in
            mem = beta * mem_prev[l] + cur
            spk = (mem - thr >= 0.0).astype(np.float64)
            mem = mem - thr * spk
            mem_prev[l] = mem
            h_prev[l] = spk
            h_below = spk
            if l == n_layers - 1:
                final_mem_t = mem
                final_spk_t = spk
        if final_mem_t is None or final_spk_t is None:
            raise RuntimeError("No hidden layers were executed for LIF sequence readout.")
        if last_layer_readout == "membrane":
            readouts.append(final_mem_t)
        elif last_layer_readout == "spike":
            readouts.append(final_spk_t)
        else:
            raise ValueError(f"Unsupported last_layer_readout={last_layer_readout}. Expected membrane|spike.")
    return np.stack(readouts, axis=1)


def _readouts_to_thresholded_features(
    readout_tr: np.ndarray,
    readout_va: np.ndarray,
    readout_te: np.ndarray,
    bias: float,
    *,
    all_timesteps: bool,
    p_last: int,
    feature_count: int,
    last_layer_readout: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map last LIF readout to convex inputs D. No extra linear U_last.

    - membrane: D = 𝟙(h - bias >= 0) (half-space indicators; ``bias`` is the CVX threshold).
    - spike: last hidden output is already {0,1} from the LIF; use it as D with **no** second
      thresholding on ``bias`` (avoids degenerate double-quantization / neuron collapse).
    """
    if int(feature_count) != int(p_last):
        raise ValueError(
            "CVX feature dim must match last readout width (removed U_last). "
            f"Require init_cfg.feature_count == P_last ({p_last}), got feature_count={feature_count}."
        )
    for r in (readout_tr, readout_va, readout_te):
        if int(r.shape[-1]) != int(p_last):
            raise ValueError(f"Readout trailing dim {r.shape[-1]} != P_last {p_last}.")
    if last_layer_readout == "spike":
        if all_timesteps:
            d_train = readout_tr.astype(np.float64, copy=False).reshape(
                readout_tr.shape[0] * readout_tr.shape[1], readout_tr.shape[2]
            )
            d_val = readout_va.astype(np.float64, copy=False).reshape(
                readout_va.shape[0] * readout_va.shape[1], readout_va.shape[2]
            )
            d_test = readout_te.astype(np.float64, copy=False).reshape(
                readout_te.shape[0] * readout_te.shape[1], readout_te.shape[2]
            )
        else:
            d_train = readout_tr.astype(np.float64, copy=False)
            d_val = readout_va.astype(np.float64, copy=False)
            d_test = readout_te.astype(np.float64, copy=False)
        return d_train, d_val, d_test
    if last_layer_readout != "membrane":
        raise ValueError(f"Unsupported last_layer_readout={last_layer_readout}. Expected membrane|spike.")
    if all_timesteps:
        d_train_seq = (readout_tr - bias >= 0.0).astype(np.float64)
        d_val_seq = (readout_va - bias >= 0.0).astype(np.float64)
        d_test_seq = (readout_te - bias >= 0.0).astype(np.float64)
        d_train = d_train_seq.reshape(d_train_seq.shape[0] * d_train_seq.shape[1], d_train_seq.shape[2])
        d_val = d_val_seq.reshape(d_val_seq.shape[0] * d_val_seq.shape[1], d_val_seq.shape[2])
        d_test = d_test_seq.reshape(d_test_seq.shape[0] * d_test_seq.shape[1], d_test_seq.shape[2])
    else:
        d_train = (readout_tr - bias >= 0.0).astype(np.float64)
        d_val = (readout_va - bias >= 0.0).astype(np.float64)
        d_test = (readout_te - bias >= 0.0).astype(np.float64)
    return d_train, d_val, d_test


def _build_feature_map(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
    *,
    all_timesteps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(init_cfg.seed)
    if x_train.ndim != 3 or x_val.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Expected sequence inputs with shape (N,T,d_in) for CVX initialization.")
    d_in = x_train.shape[2]
    hidden_dims = _hidden_dims_like_snn_p2(L=init_cfg.L, P_rec=init_cfg.P_rec, P_last=init_cfg.P_last)
    beta_list = [np.full((h,), float(init_cfg.beta_leak), dtype=np.float64) for h in hidden_dims]
    threshold_list = [np.full((h,), float(init_cfg.threshold), dtype=np.float64) for h in hidden_dims]

    if init_cfg.mode == "gaussian":
        U_in_list: List[np.ndarray] = []
        in_dim_l = d_in
        for h in hidden_dims:
            U_in_list.append(_sample_weight_matrix(rng, in_dim=in_dim_l, out_dim=h, variant=init_cfg.variant))
            in_dim_l = h
        if all_timesteps:
            readout_tr = _run_lif_stack_readout_all_timesteps(
                x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_va = _run_lif_stack_readout_all_timesteps(
                x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_te = _run_lif_stack_readout_all_timesteps(
                x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
        else:
            readout_tr = _run_lif_stack_readout(
                x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_va = _run_lif_stack_readout(
                x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_te = _run_lif_stack_readout(
                x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
        d_train, d_val, d_test = _readouts_to_thresholded_features(
            readout_tr,
            readout_va,
            readout_te,
            float(init_cfg.bias),
            all_timesteps=all_timesteps,
            p_last=int(hidden_dims[-1]),
            feature_count=int(init_cfg.feature_count),
            last_layer_readout=str(init_cfg.last_layer_readout),
        )
        return d_train, d_val, d_test, {"U_in_list": np.array([], dtype=np.float64), "U_last": np.zeros((0, 0), dtype=np.float64)}

    if init_cfg.mode == "pretraining":
        if init_cfg.pretrained_weights is None or len(init_cfg.pretrained_weights) == 0:
            raise ValueError("pretraining mode requires non-empty pretrained_weights.")
        weights = [w.astype(np.float64, copy=False) for w in init_cfg.pretrained_weights]
        n_hidden = len(hidden_dims)
        if len(weights) < n_hidden:
            raise ValueError(
                f"pretraining mode requires at least {n_hidden} pretrained tensors (one per LIF stage), got {len(weights)}."
            )
        U_in_list: List[np.ndarray] = []
        in_dim_l = d_in
        for l, h in enumerate(hidden_dims):
            U_in_list.append(_coerce_hidden_weight(weights[l], in_dim=in_dim_l, out_dim=h))
            in_dim_l = h
        if all_timesteps:
            readout_tr = _run_lif_stack_readout_all_timesteps(
                x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_va = _run_lif_stack_readout_all_timesteps(
                x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_te = _run_lif_stack_readout_all_timesteps(
                x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
        else:
            readout_tr = _run_lif_stack_readout(
                x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_va = _run_lif_stack_readout(
                x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
            readout_te = _run_lif_stack_readout(
                x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
            )
        d_train, d_val, d_test = _readouts_to_thresholded_features(
            readout_tr,
            readout_va,
            readout_te,
            float(init_cfg.bias),
            all_timesteps=all_timesteps,
            p_last=int(hidden_dims[-1]),
            feature_count=int(init_cfg.feature_count),
            last_layer_readout=str(init_cfg.last_layer_readout),
        )
        return d_train, d_val, d_test, {"pretrained_weights": np.array([], dtype=np.float64), "U_last": np.zeros((0, 0), dtype=np.float64)}

    raise ValueError(f"Unknown initialization mode={init_cfg.mode}.")


def _solve_with_fallback(
    problem: cp.Problem,
    has_solution,
    solver_order: Tuple[str, ...],
) -> None:
    for s in solver_order:
        if s == "CLARABEL":
            problem.solve(solver=cp.CLARABEL, verbose=False)
        elif s == "OSQP":
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        elif s == "SCS":
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        else:
            raise ValueError(f"Unsupported solver token={s}")
        if has_solution():
            return


def solve_binary_l1_primal_dual(
    D: np.ndarray,
    y_pm1: np.ndarray,
    rho: float,
    loss_name: str,
    solver_order: Tuple[str, ...] = ("CLARABEL", "OSQP", "SCS"),
    *,
    sample_weight: np.ndarray | None = None,
) -> BinaryCvxSolution:
    """
    Solve binary convex objective with L1 output regularization on ``D @ w``:

    * **Unweighted** (``sample_weight is None``):  ``min_w (1/n) * sum_i l(y_i, d_i^T w) + rho * ||w||_1``,
      which is the same as  ``min_w  sum_i a_i l_i + rho||w||_1`` with  ``a_i = 1/n`` (and ``sum_i a_i = 1``).

    * **Weighted** (``sample_weight`` shape ``(n,)``, all positive):
      Let  ``a = sample_weight / sum(sample_weight)``  (so  ``sum_i a_i = 1``). Then the primal is
        ``min_w  sum_i a_i * l(y_i, d_i^T w) + rho * ||w||_1``,
      with the same L1 on ``w`` as the unweighted case.

    **Dual (Fenchel / Lagrange)**, same conic structure ``||D^T u||_inf <= rho``:

    * **Hinge**  ``l_i = max(0, 1 - y_i z_i)``  with slacks: ``(F(z))_i = xi_i/ (n a_i)`` in the unweighted form, but with weights the slack penalty is  ``a_i * xi_i``; equivalently  ``(1/n) sum xi``  becomes  ``sum a_i xi``  with  ``1 - y z <= xi``:
      **Dual (weighted):**  maximize  ``sum_i  u_i y_i``  (same as ``sum yu``)  subject to  ``D^T u in [-rho, rho]``  and, with  ``v_i = y_i u_i``,  **``0 <= v_i <= a_i``**  (unweighted: ``a_i = 1/n``  gives  ``0 <= y_i u_i <= 1/n``).

    * **Logistic (``ce``)**, **unweighted**  ``a_i=1/n``: dual uses binary entropy  ``(1/n) sum( entr(n y_i u_i) + entr(1 - n y_i u_i) )``  (CVX `entr`).

    * **Logistic, weighted nontrivial** ``a``: we still solve the **correct weighted primal**; a closed-form dual in the same `entr` shape is not wired here, so  ``dual_obj``  and  ``gap``  are  ``NaN``  (primal  ``w``  is valid).

    * **Squared:** weighted primal uses  ``a_i (d_i'w - y_i)^2``; the dual rewrites similarly (omitted in short form).
    """
    n, p = D.shape
    y = y_pm1.astype(np.float64)
    Df = D.astype(np.float64)

    if loss_name not in ("hinge", "hinge_ovr", "squared", "ce"):
        raise ValueError(f"Unsupported loss_name={loss_name}.")

    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight must have shape (n,) with n={n}, got {sw.shape}.")
        if np.any(sw <= 0.0) or not np.isfinite(sw).all():
            raise ValueError("sample_weight must be finite and strictly positive on every row.")
        a = (sw / float(np.sum(sw))).astype(np.float64)
    else:
        a = None

    # ---------- Primal ----------
    w = cp.Variable(p)
    scores = Df @ w
    if loss_name in ("hinge", "hinge_ovr"):
        xi = cp.Variable(n, nonneg=True)
        margins = cp.multiply(y, scores)
        if a is not None:
            primal_obj_expr = cp.sum(cp.multiply(a, xi)) + rho * cp.norm1(w)
        else:
            primal_obj_expr = (1.0 / n) * cp.sum(xi) + rho * cp.norm1(w)
        primal_prob = cp.Problem(cp.Minimize(primal_obj_expr), [margins >= 1.0 - xi])
    elif loss_name == "squared":
        resid = scores - y
        if a is not None:
            primal_obj_expr = cp.sum(cp.multiply(a, cp.square(resid))) + rho * cp.norm1(w)
        else:
            primal_obj_expr = (1.0 / n) * cp.sum_squares(resid) + rho * cp.norm1(w)
        primal_prob = cp.Problem(cp.Minimize(primal_obj_expr))
    else:  # ce -> binary logistic
        margins = cp.multiply(y, scores)
        if a is not None:
            primal_obj_expr = cp.sum(cp.multiply(a, cp.logistic(-margins))) + rho * cp.norm1(w)
        else:
            primal_obj_expr = (1.0 / n) * cp.sum(cp.logistic(-margins)) + rho * cp.norm1(w)
        primal_prob = cp.Problem(cp.Minimize(primal_obj_expr))

    _solve_with_fallback(primal_prob, has_solution=lambda: w.value is not None, solver_order=solver_order)
    if w.value is None:
        raise RuntimeError("Primal solver failed.")
    w_star = np.asarray(w.value, dtype=np.float64).reshape(-1)
    primal_val = float(primal_prob.value)
    if not np.isfinite(w_star).all() or not np.isfinite(primal_val):
        raise FloatingPointError("Non-finite primal solution.")

    # ---------- Dual ----------
    u = cp.Variable(n)
    yu = cp.multiply(y, u)
    constraints: List[cp.Constraint] = [Df.T @ u <= rho, Df.T @ u >= -rho]
    dual_ce_weighted_nontrivial = a is not None and loss_name == "ce" and not np.allclose(a, 1.0 / n, rtol=1e-6, atol=0.0)
    if dual_ce_weighted_nontrivial:
        return BinaryCvxSolution(
            w=w_star,
            primal_obj=primal_val,
            dual_obj=float("nan"),
            gap=float("nan"),
        )

    if loss_name in ("hinge", "hinge_ovr"):
        if a is not None:
            constraints.extend([yu >= 0.0, yu <= a])
        else:
            constraints.extend([yu >= 0.0, yu <= 1.0 / n])
        dual_obj = cp.Maximize(cp.sum(yu))
    elif loss_name == "squared":
        if a is not None:
            raise NotImplementedError("Dual for weighted squared loss with L1 is not implemented; use unweighted (sample_weight=None).")
        dual_obj = cp.Maximize(cp.sum(yu) - (n / 4.0) * cp.sum_squares(u))
    else:  # ce -> binary logistic, unweighted (or a ~ uniform)
        constraints.extend([yu >= 0.0, yu <= 1.0 / n])
        nyu = n * yu
        dual_obj = cp.Maximize((1.0 / n) * cp.sum(cp.entr(nyu) + cp.entr(1.0 - nyu)))

    dual_prob = cp.Problem(dual_obj, constraints)
    _solve_with_fallback(dual_prob, has_solution=lambda: u.value is not None, solver_order=solver_order)
    if u.value is None:
        raise RuntimeError("Dual solver failed.")
    dual_val = float(dual_prob.value)
    if not np.isfinite(dual_val):
        raise FloatingPointError("Non-finite dual objective.")
    gap_v = float(primal_val - dual_val)
    if not np.isfinite(gap_v):
        gap_v = float("nan")
    return BinaryCvxSolution(w=w_star, primal_obj=primal_val, dual_obj=dual_val, gap=gap_v)


def _hinge_ovr_numpy(scores: np.ndarray, y: np.ndarray, margin: float = 1.0) -> float:
    n, c = scores.shape
    targets = -np.ones((n, c), dtype=np.float64)
    targets[np.arange(n), y] = 1.0
    losses = np.maximum(0.0, margin - targets * scores)
    return float(losses.mean())


def _compute_split_loss(loss_name: str, scores: np.ndarray, y: np.ndarray) -> float:
    """
    STE / reporting helper: **mean** OVR hinge, one-hot squared, or softmax CE consistent with
    ``LossFunction`` on 2D scores. Not the same summation as the multiclass CVX OVR stack; for
    CVX method=cvx with hinge/hinge_ovr/squared, use ``multiclass_ovr_cvx_data_loss`` instead.
    """
    if loss_name in ("hinge_ovr", "hinge"):
        return _hinge_ovr_numpy(scores=scores, y=y)
    if loss_name == "squared":
        y_oh = np.eye(scores.shape[1], dtype=np.float64)[y]
        return float(np.mean((scores - y_oh) ** 2))
    if loss_name == "ce":
        logits = torch.tensor(scores, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        return float(LossFunction.ce(labels, logits).item())
    raise ValueError(f"Unsupported loss_name={loss_name}.")


def _multiclass_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.ndim != 2:
        raise ValueError(f"Expected rank-2 logits (N,C), got shape={tuple(logits.shape)}.")
    if y.ndim != 1:
        raise ValueError(f"Expected rank-1 labels (N,), got shape={tuple(y.shape)}.")
    if logits.shape[1] == 1:
        preds = (logits[:, 0] >= 0.0).long()
    else:
        preds = torch.argmax(logits, dim=1)
    return float((preds == y.long()).float().mean().item())


def _token_seq_stats_from_flat_predictions(
    preds_flat: np.ndarray,
    y_flat: np.ndarray,
    n_samples: int,
    steps: int,
) -> Dict[str, float]:
    preds_2d = preds_flat.reshape(n_samples, steps)
    y_2d = y_flat.reshape(n_samples, steps)
    match = preds_2d == y_2d
    token_acc = float(np.mean(match))
    seq_acc = float(np.mean(np.all(match, axis=1)))
    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "token_loss": float(1.0 - token_acc),
        "seq_loss": float(1.0 - seq_acc),
    }


def _solve_one_ovr_class(
    *,
    class_idx: int,
    d_train: np.ndarray,
    y_train: np.ndarray,
    rho: float,
    loss_name: str,
) -> Tuple[int, np.ndarray, float, float]:
    y_bin = np.where(y_train == int(class_idx), 1.0, -1.0).astype(np.float64)
    sol = solve_binary_l1_primal_dual(
        d_train,
        y_bin,
        rho=rho,
        loss_name=loss_name,
    )
    return int(class_idx), np.asarray(sol.w, dtype=np.float64), float(sol.primal_obj), float(sol.dual_obj)


def _run_cvx_method(
    d_train: np.ndarray,
    y_train: np.ndarray,
    d_val: np.ndarray,
    y_val: np.ndarray,
    d_test: np.ndarray,
    y_test: np.ndarray,
    solve_cfg: SolveConfig,
    init_bias: float,
) -> CvxSolveResult:
    num_classes = int(np.max(y_train) + 1)
    rho = float(solve_cfg.beta / math.sqrt(max(d_train.shape[1], 1)))
    ovr_workers = int(solve_cfg.cvx_ovr_workers)
    if ovr_workers <= 0:
        raise ValueError(f"cvx_ovr_workers must be >= 1, got {ovr_workers}.")
    print(
        (
            f"[cvx-run] method=cvx beta={float(solve_cfg.beta):.6g} lr={float(solve_cfg.lr):.6g} "
            f"bias={float(init_bias):.6g} "
            f"batch_size=full ovr_workers={ovr_workers}"
        ),
        flush=True,
    )
    if solve_cfg.loss_name == "ce":
        w, primal_sum, dual_sum, _ = solve_multiclass_softmax_ce_l1_primal_dual(
            d_train,
            y_train,
            rho=rho,
            num_classes=num_classes,
            solver_order=("CLARABEL", "SCS"),
            compute_dual=solve_cfg.compute_ce_dual,
        )
    else:
        w = np.zeros((d_train.shape[1], num_classes), dtype=np.float64)
        primal_sum = 0.0
        dual_sum = 0.0
        if ovr_workers == 1 or num_classes == 1:
            for c in range(num_classes):
                _, w_c, p_c, d_c = _solve_one_ovr_class(
                    class_idx=c,
                    d_train=d_train,
                    y_train=y_train,
                    rho=rho,
                    loss_name=solve_cfg.loss_name,
                )
                w[:, c] = w_c
                primal_sum += p_c
                dual_sum += d_c
        else:
            max_workers = min(int(ovr_workers), int(num_classes))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _solve_one_ovr_class,
                        class_idx=c,
                        d_train=d_train,
                        y_train=y_train,
                        rho=rho,
                        loss_name=solve_cfg.loss_name,
                    )
                    for c in range(num_classes)
                ]
                for fut in as_completed(futures):
                    c_idx, w_c, p_c, d_c = fut.result()
                    w[:, c_idx] = w_c
                    primal_sum += p_c
                    dual_sum += d_c
    train_scores = d_train @ w
    val_scores = d_val @ w
    test_scores = d_test @ w
    l1_penalty = float(np.abs(w).sum())
    if solve_cfg.loss_name == "ce":
        # Joint softmax+L1 primal value matches mean CE + rho||W||_1 from LossFunction.ce on 2D logits.
        train_loss = float(primal_sum - rho * l1_penalty)
        val_loss = _compute_split_loss(solve_cfg.loss_name, val_scores, y_val)
        test_loss = _compute_split_loss(solve_cfg.loss_name, test_scores, y_test)
        train_objective = float(primal_sum)
    else:
        # OVR: independent binary problems per class; ''train_loss'' is the summed (1/n) binary
        # data terms — matches primal_sum - rho||W||_1 (val/test use the same binary reduction).
        train_loss = float(primal_sum - rho * l1_penalty)
        val_loss = multiclass_ovr_cvx_data_loss(solve_cfg.loss_name, val_scores, y_val)
        test_loss = multiclass_ovr_cvx_data_loss(solve_cfg.loss_name, test_scores, y_test)
        train_objective = float(primal_sum)
    val_objective = float(val_loss + rho * l1_penalty)
    test_objective = float(test_loss + rho * l1_penalty)
    gap_val = float(primal_sum - dual_sum) if np.isfinite(dual_sum) else float("nan")
    diag = CvxDiagnostics(
        primal_value=float(primal_sum),
        dual_value=float(dual_sum),
        gap=gap_val,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
    )
    return CvxSolveResult(
        trained_model={"weights": w, "weights_blocks": None, "method": "cvx"},
        loss_history=[train_loss],
        final_losses={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_objective": train_objective,
            "val_objective": val_objective,
            "test_objective": test_objective,
        },
        diagnostics=diag,
    )


def _run_sgd_method(
    d_train: np.ndarray,
    y_train: np.ndarray,
    d_val: np.ndarray,
    y_val: np.ndarray,
    d_test: np.ndarray,
    y_test: np.ndarray,
    solve_cfg: SolveConfig,
    device: torch.device,
    init_bias: float,
) -> CvxSolveResult:
    num_classes = int(np.max(y_train) + 1)
    model = LinearFeatureClassifier(d_in=d_train.shape[1], num_classes=num_classes).to(device)
    rho = float(solve_cfg.beta / math.sqrt(max(d_train.shape[1], 1)))
    if solve_cfg.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=solve_cfg.lr)
    elif solve_cfg.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=solve_cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer={solve_cfg.optimizer_name}.")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    xtr = torch.tensor(d_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    xva = torch.tensor(d_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    xte = torch.tensor(d_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    n_train = int(xtr.shape[0])
    if solve_cfg.batch_size is not None and int(solve_cfg.batch_size) != n_train:
        raise ValueError(
            f"CVX-SGD enforces full-batch training: expected batch_size={n_train}, got {int(solve_cfg.batch_size)}."
        )
    batch_size = n_train
    print(
        (
            f"[cvx-run] method=sgd beta={float(solve_cfg.beta):.6g} lr={float(solve_cfg.lr):.6g} "
            f"bias={float(init_bias):.6g} "
            f"batch_size=full({batch_size}) n_train={n_train}"
        ),
        flush=True,
    )
    loss_history: List[float] = []
    best_val_objective = float("inf")
    best_state = None
    for epoch in range(1, solve_cfg.epochs + 1):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=device)
        for start in range(0, xtr.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = model(xtr[idx])
            loss = LossFunction.compute(name=solve_cfg.loss_name, y=ytr[idx], f_x=logits).value
            reg = rho * model.linear.weight.abs().sum()
            objective = loss + reg
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(xva)
            val_loss = float(LossFunction.compute(name=solve_cfg.loss_name, y=yva, f_x=val_logits).value.item())
            l1_penalty = float(model.linear.weight.abs().sum().item())
            val_objective = val_loss + rho * l1_penalty
        scheduler.step(val_objective)
        if val_objective < best_val_objective:
            best_val_objective = val_objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if solve_cfg.log_every > 0 and epoch % solve_cfg.log_every == 0:
            with torch.no_grad():
                train_logits = model(xtr)
                train_loss = float(LossFunction.compute(name=solve_cfg.loss_name, y=ytr, f_x=train_logits).value.item())
                train_acc = _multiclass_accuracy(train_logits, ytr)
                val_acc = _multiclass_accuracy(val_logits, yva)
                train_objective = train_loss + rho * l1_penalty
            loss_history.append(train_loss)
            print(
                (
                    f"[cvx-sgd] epoch={epoch}/{solve_cfg.epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"train_obj={train_objective:.6f} val_obj={val_objective:.6f} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                ),
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("SGD did not capture a best state.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_loss = float(LossFunction.compute(name=solve_cfg.loss_name, y=ytr, f_x=model(xtr)).value.item())
        val_loss = float(LossFunction.compute(name=solve_cfg.loss_name, y=yva, f_x=model(xva)).value.item())
        test_loss = float(LossFunction.compute(name=solve_cfg.loss_name, y=yte, f_x=model(xte)).value.item())
        l1_penalty = float(model.linear.weight.abs().sum().item())
        train_objective = train_loss + rho * l1_penalty
        val_objective = val_loss + rho * l1_penalty
        test_objective = test_loss + rho * l1_penalty
    diag = CvxDiagnostics(
        primal_value=float(train_objective),
        dual_value=float("nan"),
        gap=float("nan"),
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
    )
    w_np = model.linear.weight.detach().cpu().numpy().astype(np.float64).T
    return CvxSolveResult(
        trained_model={"weights": w_np, "weights_blocks": None, "method": "sgd"},
        loss_history=loss_history,
        final_losses={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_objective": train_objective,
            "val_objective": val_objective,
            "test_objective": test_objective,
        },
        diagnostics=diag,
    )


def cvx_solve(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    init_cfg: InitializationConfig,
    solve_cfg: SolveConfig,
    device: Optional[torch.device] = None,
    precomputed_features: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    precomputed_branch_slices: Optional[Sequence[Tuple[int, int]]] = None,
) -> CvxSolveResult:
    if int(getattr(init_cfg, "K_parallel", 1)) > 1:
        from . import cvx_parallel_Solve as p

        return p.cvx_solve(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            init_cfg=p.InitializationConfig(
                mode=init_cfg.mode,
                variant=init_cfg.variant,
                seed=init_cfg.seed,
                feature_count=init_cfg.feature_count,
                bias=init_cfg.bias,
                pretrained_weights=init_cfg.pretrained_weights,
                L=init_cfg.L,
                P_rec=init_cfg.P_rec,
                P_last=init_cfg.P_last,
                K_parallel=init_cfg.K_parallel,
                beta_leak=init_cfg.beta_leak,
                threshold=init_cfg.threshold,
                last_layer_readout=init_cfg.last_layer_readout,
            ),
            solve_cfg=p.SolveConfig(
                loss_name=solve_cfg.loss_name,
                method=solve_cfg.method,
                beta=solve_cfg.beta,
                lr=solve_cfg.lr,
                optimizer_name=solve_cfg.optimizer_name,
                epochs=solve_cfg.epochs,
                batch_size=solve_cfg.batch_size,
                log_every=solve_cfg.log_every,
                compute_ce_dual=solve_cfg.compute_ce_dual,
                cvx_ovr_workers=solve_cfg.cvx_ovr_workers,
            ),
            device=device,
            precomputed_features=precomputed_features,
            precomputed_branch_slices=precomputed_branch_slices,
        )

    if precomputed_branch_slices is not None:
        raise ValueError(
            "precomputed_branch_slices is only valid when K_parallel > 1; got K_parallel=1."
        )
    supervise_all_timesteps = y_train.ndim == 2
    if supervise_all_timesteps and (y_val.ndim != 2 or y_test.ndim != 2):
        raise ValueError(
            "Arithmetic-style all-timestep supervision requires y_train/y_val/y_test to all be rank-2."
        )
    if supervise_all_timesteps:
        ytr = y_train.reshape(-1).astype(np.int64)
        yva = y_val.reshape(-1).astype(np.int64)
        yte = y_test.reshape(-1).astype(np.int64)
    else:
        ytr = _prepare_sequence_targets(y_train)
        yva = _prepare_sequence_targets(y_val)
        yte = _prepare_sequence_targets(y_test)
    if precomputed_features is not None:
        d_train, d_val, d_test = precomputed_features
        if d_train.ndim != 2 or d_val.ndim != 2 or d_test.ndim != 2:
            raise ValueError(
                "precomputed_features arrays must be 2-D (N*T, P) or (N, P)."
            )
        if d_train.shape[1] != d_val.shape[1] or d_train.shape[1] != d_test.shape[1]:
            raise ValueError(
                "precomputed_features train/val/test feature dim mismatch: "
                f"train={d_train.shape[1]}, val={d_val.shape[1]}, test={d_test.shape[1]}."
            )
        if d_train.shape[0] != ytr.shape[0]:
            raise ValueError(
                f"precomputed d_train rows={d_train.shape[0]} disagree with y_train flat rows={ytr.shape[0]}."
            )
    else:
        d_train, d_val, d_test, _ = _build_feature_map(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            init_cfg=init_cfg,
            all_timesteps=supervise_all_timesteps,
        )
    if solve_cfg.method == "cvx":
        out = _run_cvx_method(
            d_train=d_train,
            y_train=ytr,
            d_val=d_val,
            y_val=yva,
            d_test=d_test,
            y_test=yte,
            solve_cfg=solve_cfg,
            init_bias=float(init_cfg.bias),
        )
    elif solve_cfg.method == "sgd":
        run_device = choose_device() if device is None else device
        out = _run_sgd_method(
            d_train=d_train,
            y_train=ytr,
            d_val=d_val,
            y_val=yva,
            d_test=d_test,
            y_test=yte,
            solve_cfg=solve_cfg,
            device=run_device,
            init_bias=float(init_cfg.bias),
        )
    else:
        raise ValueError(f"Unknown solve method={solve_cfg.method}.")

    if supervise_all_timesteps:
        n_train, steps = y_train.shape
        n_val = y_val.shape[0]
        n_test = y_test.shape[0]
        if isinstance(out.trained_model, dict) and "weights" in out.trained_model:
            w = out.trained_model["weights"]
            pred_train = np.argmax(d_train @ w, axis=1)
            pred_val = np.argmax(d_val @ w, axis=1)
            pred_test = np.argmax(d_test @ w, axis=1)
        else:
            model = out.trained_model
            model.eval()
            with torch.no_grad():
                pred_train = torch.argmax(model(torch.tensor(d_train, dtype=torch.float32)), dim=1).cpu().numpy()
                pred_val = torch.argmax(model(torch.tensor(d_val, dtype=torch.float32)), dim=1).cpu().numpy()
                pred_test = torch.argmax(model(torch.tensor(d_test, dtype=torch.float32)), dim=1).cpu().numpy()
        tr_stats = _token_seq_stats_from_flat_predictions(pred_train, ytr, n_train, steps)
        va_stats = _token_seq_stats_from_flat_predictions(pred_val, yva, n_val, steps)
        te_stats = _token_seq_stats_from_flat_predictions(pred_test, yte, n_test, steps)
        out.final_losses.update(
            {
                "train_token_acc": tr_stats["token_acc"],
                "val_token_acc": va_stats["token_acc"],
                "test_token_acc": te_stats["token_acc"],
                "train_seq_acc": tr_stats["seq_acc"],
                "val_seq_acc": va_stats["seq_acc"],
                "test_seq_acc": te_stats["seq_acc"],
                "train_token_loss": tr_stats["token_loss"],
                "val_token_loss": va_stats["token_loss"],
                "test_token_loss": te_stats["token_loss"],
                "train_seq_loss": tr_stats["seq_loss"],
                "val_seq_loss": va_stats["seq_loss"],
                "test_seq_loss": te_stats["seq_loss"],
            }
        )
        print(
            (
                "[cvx-arithmetic] "
                f"token_acc train={tr_stats['token_acc']:.4f} val={va_stats['token_acc']:.4f} test={te_stats['token_acc']:.4f} "
                f"seq_acc train={tr_stats['seq_acc']:.4f} val={va_stats['seq_acc']:.4f} test={te_stats['seq_acc']:.4f} "
                f"token_loss train={tr_stats['token_loss']:.4f} val={va_stats['token_loss']:.4f} test={te_stats['token_loss']:.4f} "
                f"seq_loss train={tr_stats['seq_loss']:.4f} val={va_stats['seq_loss']:.4f} test={te_stats['seq_loss']:.4f}"
            ),
            flush=True,
        )
    return out
