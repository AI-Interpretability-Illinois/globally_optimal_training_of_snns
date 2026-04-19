from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    name: str
    value: torch.Tensor


class LossFunction:
    """Centralized loss definitions used across STE/CVX pipelines."""

    @staticmethod
    def ce(y: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        if f_x.ndim == 3:
            bsz, steps, num_classes = f_x.shape
            if y.ndim == 1:
                # Non-arithmetic sequence tasks: supervise last timestep only.
                logits = f_x[:, -1, :]
                return F.cross_entropy(logits, y)
            if y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
                return F.cross_entropy(f_x.reshape(bsz * steps, num_classes), y.reshape(bsz * steps))
            raise ValueError(f"CE expects y to have ndim 1 or 2 when f_x is 3D, got y.ndim={y.ndim}.")
        if f_x.ndim == 2:
            return F.cross_entropy(f_x, y)
        raise ValueError(f"CE expects f_x with ndim 2 or 3, got ndim={f_x.ndim}.")

    @staticmethod
    def hinge(y: torch.Tensor, f_x: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        # snn_p2-style binary hinge:
        # - single-label sequence classification (y.ndim==1): use last timestep logits
        # - sequence labels (y.ndim==2): supervise all timesteps
        def _binary_margin_from_logits(z: torch.Tensor) -> torch.Tensor:
            # Accept either a single binary logit channel or 2-class logits.
            if z.ndim >= 1 and z.shape[-1] == 1:
                return z.squeeze(-1)
            if z.ndim >= 1 and z.shape[-1] == 2:
                return z[..., 1] - z[..., 0]
            raise ValueError(f"Binary hinge expects final channel size 1 or 2, got shape={z.shape}.")

        if f_x.ndim == 3:
            margins = _binary_margin_from_logits(f_x)
            if y.ndim == 1:
                # Non-arithmetic sequence tasks: supervise last timestep only.
                f_x = margins[:, -1]
            elif y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
                bsz, steps = margins.shape
                f_x = margins.reshape(bsz * steps)
                y = y.reshape(bsz * steps)
            else:
                raise ValueError(f"Hinge expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
        elif f_x.ndim == 2:
            f_x = _binary_margin_from_logits(f_x)
        if f_x.ndim > 1 and f_x.shape[-1] == 1:
            f_x = f_x.squeeze(-1)
        if y.ndim != f_x.ndim:
            raise ValueError(f"Hinge expects y and f_x with matching ndim, got {y.ndim} vs {f_x.ndim}.")
        if torch.equal(torch.unique(y), torch.tensor([0, 1], device=y.device, dtype=y.dtype)):
            y_pm1 = y * 2 - 1
        else:
            y_pm1 = y
        y_pm1 = y_pm1.to(dtype=f_x.dtype)
        return torch.relu(margin - y_pm1 * f_x).mean()

    @staticmethod
    def hinge_ovr(y: torch.Tensor, f_x: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        if f_x.ndim == 2:
            num_samples, num_classes = f_x.shape
            y_flat = y.reshape(num_samples)
            targets = -torch.ones_like(f_x)
            targets.scatter_(1, y_flat.unsqueeze(1), 1.0)
            return torch.relu(margin - targets * f_x).mean()
        if f_x.ndim == 3:
            bsz, steps, num_classes = f_x.shape
            if y.ndim == 1:
                # Non-arithmetic sequence tasks: supervise last timestep only.
                scores = f_x[:, -1, :]
                y_flat = y
            elif y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
                scores = f_x.reshape(bsz * steps, num_classes)
                y_flat = y.reshape(bsz * steps)
            else:
                raise ValueError(f"hinge_ovr expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
            targets = -torch.ones_like(scores)
            targets.scatter_(1, y_flat.unsqueeze(1), 1.0)
            return torch.relu(margin - targets * scores).mean()
        raise ValueError(f"hinge_ovr expects f_x with ndim 2 or 3, got {f_x.ndim}.")

    @staticmethod
    def squared(y: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        # Keep the same sequence supervision convention as other losses:
        # y.ndim==1 -> last timestep only, y.ndim==2 -> all timesteps.
        if f_x.ndim == 3:
            bsz, steps, num_outputs = f_x.shape
            if y.ndim == 1:
                preds = f_x[:, -1, :]
                if num_outputs == 1:
                    target = y.to(dtype=preds.dtype).reshape(-1, 1)
                    return torch.mean((preds - target) ** 2)
                if num_outputs > 1:
                    target = F.one_hot(y.to(torch.long), num_classes=num_outputs).to(dtype=preds.dtype)
                    return torch.mean((preds - target) ** 2)
                raise ValueError(f"Invalid num_outputs={num_outputs} for squared loss.")
            if y.ndim == 2:
                preds = f_x.reshape(bsz * steps, num_outputs)
                y_flat = y.reshape(bsz * steps)
                if num_outputs == 1:
                    target = y_flat.to(dtype=preds.dtype).reshape(-1, 1)
                    return torch.mean((preds - target) ** 2)
                target = F.one_hot(y_flat.to(torch.long), num_classes=num_outputs).to(dtype=preds.dtype)
                return torch.mean((preds - target) ** 2)
            raise ValueError(f"Squared expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
        y_cast = y.to(dtype=f_x.dtype)
        if y_cast.shape != f_x.shape:
            raise ValueError(f"Squared loss expects y and f_x shapes to match exactly, got {y_cast.shape} vs {f_x.shape}.")
        return torch.mean((f_x - y_cast) ** 2)

    @classmethod
    def compute(cls, name: str, y: torch.Tensor, f_x: torch.Tensor) -> LossOutput:
        if name == "ce":
            return LossOutput(name=name, value=cls.ce(y=y, f_x=f_x))
        if name == "hinge":
            return LossOutput(name=name, value=cls.hinge(y=y, f_x=f_x))
        if name == "hinge_ovr":
            return LossOutput(name=name, value=cls.hinge_ovr(y=y, f_x=f_x))
        if name == "squared":
            return LossOutput(name=name, value=cls.squared(y=y, f_x=f_x))
        raise ValueError(f"Unknown loss function: {name}. Expected one of: ce, hinge, hinge_ovr, squared.")


# ---------------------------------------------------------------------------
# Convex multiclass softmax CE + elementwise L1 (CVXPY): primal, dual, gap
# ---------------------------------------------------------------------------
#
# Primal (W in R^{p×K}, D in R^{n×p}, rows D_i):
#   min_W  (1/n) * sum_i ( logsumexp(D_i W) - (D_i W)_{y_i} ) + rho * sum_{j,k} |W_{jk}|
# Same objective as LossFunction.ce(D @ W, y) (mean) + rho * ||W||_1 in the STE/CVX-SGD head.
#
# Dual (Λ in R^{n×K}), Fenchel conjugate of phi_i(z)=lse(z)-z_{y_i} at -n Λ_i:
#   phi_i^*(-n λ) = sum_k v_{ik} log v_{ik},  v_i = e_{y_i} - n λ_i  on the simplex.
#   Since v_k log v_k = -entr(v_k) in CVXPY (entr(x) = -x log x),
#   -phi_i^*(-n λ_i) = sum_k entr(v_{ik}).
#
#   max_{Λ}  (1/n) * sum_{i,k} entr( E_{ik} - n Λ_{ik} )
#   s.t.  E - n Λ >= 0,  sum_k Λ_{ik} = 0  ∀ i,  | (D^T Λ)_{jk} | <= rho  ∀ j,k
#
# Reference: composition F(DW)+rho||W||_1; dual -F^*(-Λ) with ||D^T Λ||_inf <= rho.


def _solve_cvxpy_with_fallback(
    problem: cp.Problem,
    has_solution,
    solver_order: Tuple[str, ...],
) -> None:
    for s in solver_order:
        if s == "CLARABEL":
            problem.solve(solver=cp.CLARABEL, verbose=False)
        elif s == "OSQP":
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
        elif s == "SCS":
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=50000)
        else:
            raise ValueError(f"Unsupported solver token={s}")
        if has_solution():
            return


def build_softmax_ce_l1_primal_problem(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
) -> Tuple[cp.Problem, cp.Variable]:
    """Primal: mean softmax CE + rho * sum |W|."""
    n, p = D.shape
    Df = D.astype(np.float64)
    y_i = y.astype(np.int64)
    if y_i.ndim != 1 or y_i.shape[0] != n:
        raise ValueError(f"Expected y shape (n,) with n={n}, got {y_i.shape}.")
    if num_classes < 2 or int(y_i.min()) < 0 or int(y_i.max()) >= num_classes:
        raise ValueError(f"Invalid labels or num_classes={num_classes}.")

    E = np.zeros((n, num_classes), dtype=np.float64)
    E[np.arange(n), y_i] = 1.0
    W = cp.Variable((p, num_classes))
    scores = Df @ W
    row_lse = cp.log_sum_exp(scores, axis=1, keepdims=False)
    logits_y = cp.sum(cp.multiply(scores, E), axis=1)
    primal_obj = (1.0 / n) * cp.sum(row_lse - logits_y) + rho * cp.sum(cp.abs(W))
    prob = cp.Problem(cp.Minimize(primal_obj))
    return prob, W


def build_softmax_ce_l1_dual_problem(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
) -> Tuple[cp.Problem, cp.Variable]:
    """Dual of softmax CE + elementwise L1; objective to be maximized."""
    n, p = D.shape
    Df = D.astype(np.float64)
    y_i = y.astype(np.int64)
    if y_i.ndim != 1 or y_i.shape[0] != n:
        raise ValueError(f"Expected y shape (n,) with n={n}, got {y_i.shape}.")
    if num_classes < 2 or int(y_i.min()) < 0 or int(y_i.max()) >= num_classes:
        raise ValueError(f"Invalid labels or num_classes={num_classes}.")

    E = np.zeros((n, num_classes), dtype=np.float64)
    E[np.arange(n), y_i] = 1.0
    Lam = cp.Variable((n, num_classes))
    v = E - n * Lam
    DtLam = Df.T @ Lam
    constraints = [
        v >= 0.0,
        cp.sum(Lam, axis=1) == 0.0,
        DtLam <= rho,
        DtLam >= -rho,
    ]
    dual_obj = (1.0 / n) * cp.sum(cp.entr(v))
    prob = cp.Problem(cp.Maximize(dual_obj), constraints)
    return prob, Lam


def solve_multiclass_softmax_ce_l1_primal_dual(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
    *,
    solver_order: Tuple[str, ...] = ("CLARABEL", "SCS"),
) -> Tuple[np.ndarray, float, float, float]:
    """
    Solve primal and dual for softmax CE + L1; return (W, primal_val, dual_val, gap).

    ``gap = primal_val - dual_val`` should be small when strong duality holds.
    """
    primal_prob, W = build_softmax_ce_l1_primal_problem(D, y, rho, num_classes)
    _solve_cvxpy_with_fallback(
        primal_prob,
        has_solution=lambda: W.value is not None,
        solver_order=solver_order,
    )
    if W.value is None:
        raise RuntimeError("Softmax CE + L1 primal solver failed.")
    w_star = np.asarray(W.value, dtype=np.float64)
    primal_val = float(primal_prob.value)
    if not np.isfinite(w_star).all() or not np.isfinite(primal_val):
        raise FloatingPointError("Non-finite primal solution for softmax CE + L1.")

    dual_prob, Lam = build_softmax_ce_l1_dual_problem(D, y, rho, num_classes)
    _solve_cvxpy_with_fallback(
        dual_prob,
        has_solution=lambda: Lam.value is not None,
        solver_order=solver_order,
    )
    if Lam.value is None:
        raise RuntimeError("Softmax CE + L1 dual solver failed.")
    dual_val = float(dual_prob.value)
    if not np.isfinite(dual_val):
        raise FloatingPointError("Non-finite dual objective for softmax CE + L1.")

    gap = float(primal_val - dual_val)
    return w_star, primal_val, dual_val, gap
