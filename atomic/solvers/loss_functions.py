from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F

from .ovr_cvx_metrics import multiclass_ovr_cvx_data_loss


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

    @staticmethod
    def bce_token_sequence_binary(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Token-level binary cross-entropy with logits for shapes (N,T) and (N,T,1).
        Uses the same all-timestep supervision convention as ``hinge`` / ``ce`` for 3D logits.
        """
        if logits.ndim != 3 or logits.shape[-1] != 1:
            raise ValueError(f"Expected logits (N,T,1), got shape={tuple(logits.shape)}.")
        if y.ndim != 2:
            raise ValueError(f"Expected y (N,T), got shape={tuple(y.shape)}.")
        yf = y.reshape(-1).to(dtype=logits.dtype)
        z = logits.reshape(-1)
        return F.binary_cross_entropy_with_logits(z, yf, reduction="mean")

    @staticmethod
    def ste_carry_tf_sum_loss(
        sum_logits: torch.Tensor,
        y_sum: torch.Tensor,
        *,
        base: int,
        loss_name: str,
    ) -> torch.Tensor:
        """
        Sum digit head for carry teacher-forcing: ``loss_name`` is already resolved (no ``auto``).
        For ``base==2`` the head is a single logit; use ``bce_token_sequence_binary`` for CE.
        For ``base>2``, use ``ce`` or ``hinge_ovr`` (multiclass margin).
        """
        b = int(base)
        name = str(loss_name)
        if name == "ce" and b == 2:
            return LossFunction.bce_token_sequence_binary(y_sum, sum_logits)
        if name == "hinge" and b == 2:
            return LossFunction.hinge(y=y_sum, f_x=sum_logits)
        if name == "ce" and b > 2:
            return LossFunction.ce(y=y_sum, f_x=sum_logits)
        if name == "hinge_ovr" and b > 2:
            return LossFunction.hinge_ovr(y=y_sum, f_x=sum_logits)
        if name == "hinge" and b > 2:
            raise ValueError(
                "Multiclass sum head with loss_name='hinge' is ambiguous; use 'hinge_ovr' for OVR hinge on (N,T,C) logits."
            )
        raise ValueError(f"Invalid ste_carry_tf sum loss: loss_name={name!r}, base={b}.")

    @staticmethod
    def ste_carry_tf_carry_loss(
        carry_logits: torch.Tensor,
        y_carry: torch.Tensor,
        *,
        loss_name: str,
    ) -> torch.Tensor:
        name = str(loss_name)
        if name == "hinge":
            return LossFunction.hinge(y=y_carry, f_x=carry_logits)
        if name == "ce":
            return LossFunction.bce_token_sequence_binary(y_carry, carry_logits)
        raise ValueError(f"Invalid carry head loss: {name!r}. Expected 'hinge' or 'ce'.")

    @staticmethod
    def carry_teacher_forcing_total(
        l_sum: torch.Tensor,
        l_carry: torch.Tensor,
        path_reg: torch.Tensor,
        *,
        lambda_sum: float,
        lambda_carry: float,
        beta: float,
        tf_objective: str,
    ) -> torch.Tensor:
        """
        Combined training objective for carry teacher-forcing.

        * ``joint`` (alias ``lambda_weighted``):  λ_s L_s + λ_c L_c + β R — default historic behavior.
        * ``mean_pair``: (L_s + L_c) / 2 + β R
        * ``lambda_normalized``: (λ_s L_s + λ_c L_c) / (|λ_s| + |λ_c|) + β R
        """
        mode = str(tf_objective)
        if mode in ("joint", "lambda_weighted"):
            return float(lambda_sum) * l_sum + float(lambda_carry) * l_carry + float(beta) * path_reg
        if mode == "mean_pair":
            return 0.5 * l_sum + 0.5 * l_carry + float(beta) * path_reg
        if mode == "lambda_normalized":
            denom = abs(float(lambda_sum)) + abs(float(lambda_carry))
            if denom < 1e-18:
                raise ValueError("lambda_sum and lambda_carry cannot both be zero for lambda_normalized.")
            return (float(lambda_sum) * l_sum + float(lambda_carry) * l_carry) / denom + float(beta) * path_reg
        raise ValueError(
            f"Unknown tf_objective={mode!r}. Expected joint|lambda_weighted|mean_pair|lambda_normalized."
        )

    @staticmethod
    def carry_time_ramp_alphas(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        α_t = 2t / (T+1) for timesteps t = 1, …, T (1-based column index in the add sequence).
        """
        t = torch.arange(1, int(T) + 1, device=device, dtype=dtype)
        return 2.0 * t / float(T + 1)

    @staticmethod
    def _y_pm1_01(y: torch.Tensor) -> torch.Tensor:
        if y.dtype not in (torch.int32, torch.int64, torch.float32, torch.float64):
            y = y.long()
        if torch.equal(torch.unique(y), torch.tensor([0, 1], device=y.device, dtype=y.dtype)):
            return y * 2 - 1
        return y.to(dtype=torch.float32)

    @staticmethod
    def ste_carry_tf_per_timestep_sum(
        sum_logits: torch.Tensor,
        y_sum: torch.Tensor,
        *,
        base: int,
        loss_name: str,
    ) -> torch.Tensor:
        """
        Returns ``(T,)`` vector: at each timestep, mean over batch of the token loss
        (same family as :meth:`ste_carry_tf_sum_loss`, but not reduced over T).
        """
        b = int(base)
        name = str(loss_name)
        if sum_logits.ndim != 3 or y_sum.ndim != 2:
            raise ValueError("Expected sum_logits (B,T,·) and y_sum (B,T).")
        B, T, c_last = sum_logits.shape
        if c_last < 1:
            raise ValueError(f"Invalid sum_logits shape: {sum_logits.shape}.")

        if name == "ce" and b == 2:
            z = sum_logits.reshape(-1)
            yf = y_sum.reshape(-1).to(dtype=sum_logits.dtype)
            return F.binary_cross_entropy_with_logits(z, yf, reduction="none").view(B, T).mean(dim=0)

        if name == "hinge" and b == 2:

            def _margin(z: torch.Tensor) -> torch.Tensor:
                if z.shape[-1] == 1:
                    return z.squeeze(-1)
                if z.shape[-1] == 2:
                    return z[..., 1] - z[..., 0]
                raise ValueError(f"Binary hinge: bad shape {z.shape}.")

            m = _margin(sum_logits)
            yp = LossFunction._y_pm1_01(y_sum)
            h = torch.relu(1.0 - yp * m)
            return h.mean(dim=0)

        if name == "ce" and b > 2:
            ce_t = F.cross_entropy(
                sum_logits.reshape(B * T, c_last),
                y_sum.reshape(B * T),
                reduction="none",
            )
            return ce_t.view(B, T).mean(dim=0)

        if name == "hinge_ovr" and b > 2:
            s = sum_logits.reshape(B * T, c_last)
            yf = y_sum.reshape(B * T)
            targets = -torch.ones_like(s)
            targets.scatter_(1, yf.unsqueeze(1), 1.0)
            row = torch.relu(1.0 - targets * s).mean(dim=1)
            return row.view(B, T).mean(dim=0)

        if name == "hinge" and b > 2:
            raise ValueError("Use loss_name='hinge_ovr' for base>2 sum head in per-timewise loss.")
        raise ValueError(f"Invalid per-timestep sum loss: {name!r}, base={b}.")

    @staticmethod
    def ste_carry_tf_per_timestep_carry(
        carry_logits: torch.Tensor,
        y_carry: torch.Tensor,
        *,
        loss_name: str,
    ) -> torch.Tensor:
        """``(T,)`` mean-over-batch loss per timestep for the carry head."""
        if carry_logits.ndim != 3 or y_carry.ndim != 2 or carry_logits.shape[-1] != 1:
            raise ValueError("Expected carry_logits (B,T,1) and y_carry (B,T).")
        B, T, _ = carry_logits.shape
        name = str(loss_name)
        if name == "hinge":
            m = carry_logits.squeeze(-1)
            yp = LossFunction._y_pm1_01(y_carry)
            h = torch.relu(1.0 - yp * m)
            return h.mean(dim=0)
        if name == "ce":
            z = carry_logits.reshape(-1)
            yf = y_carry.reshape(-1).to(dtype=carry_logits.dtype)
            return F.binary_cross_entropy_with_logits(z, yf, reduction="none").view(B, T).mean(dim=0)
        raise ValueError(f"Invalid carry head loss: {name!r}.")

    @staticmethod
    def ste_carry_tf_data_losses_scalar(
        sum_logits: torch.Tensor,
        y_sum: torch.Tensor,
        carry_logits: torch.Tensor,
        y_carry: torch.Tensor,
        *,
        base: int,
        sum_loss_name: str,
        carry_loss_name: str,
        ste_time_loss: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns scalar ``(l_sum, l_carry)`` for use with :meth:`carry_teacher_forcing_total`.
        * ``ste_time_loss`` ``uniform`` — mean over all timesteps and batch (unchanged from global reductions).
        * ``ramp`` —  ``α_t = 2t/(T+1)`` (t=1…T) on per-timestep *batch-mean* losses:
          L_s = sum_t α_t l_s(t) / sum_t α_t, same for carry.
        """
        mode = str(ste_time_loss)
        l_s_t = LossFunction.ste_carry_tf_per_timestep_sum(
            sum_logits, y_sum, base=base, loss_name=sum_loss_name
        )
        l_c_t = LossFunction.ste_carry_tf_per_timestep_carry(
            carry_logits, y_carry, loss_name=carry_loss_name
        )
        T = int(l_s_t.shape[0])
        if T < 1:
            raise ValueError("T must be >=1.")
        if l_c_t.shape[0] != T:
            raise ValueError("Sum/carry T mismatch.")

        if mode == "uniform":
            l_sum = l_s_t.mean()
            l_carry = l_c_t.mean()
        elif mode == "ramp":
            dev = l_s_t.device
            dt = l_s_t.dtype
            alpha = LossFunction.carry_time_ramp_alphas(T, dev, dt)
            z = float(alpha.sum())
            l_sum = (l_s_t * alpha).sum() / z
            l_carry = (l_c_t * alpha).sum() / z
        else:
            raise ValueError(f"Unknown ste_time_loss={mode!r}. Expected uniform or ramp.")
        return l_sum, l_carry


# ---------------------------------------------------------------------------
# Convex multiclass softmax CE + elementwise L1 (CVXPY): primal, dual, gap
# ---------------------------------------------------------------------------
#
# Row weights  a_i = sw_i / sum sw  (sum a = 1).  Unweighted  <->  a_i = 1/n  (sample_weight=None).
#
# Primal  (W  in  R^{p×K}):
#   min_W  sum_i  a_i * (  logsumexp( D_i W ) - ( D_i W )_{y_i}  )  +  rho  *  |W|_1
#
# Z-form  dual:  z_i  on  the  (K-1)-simplex  (row  sums  1,  z_{ik}  >=  0),  with  the  Fenchel  couple
#   Λ  =  a  ⊙  ( E  -  Z )   (E  =  one-hot  labels,  1' Λ_i  =  0,  1'Z_i  =  1)  to  the  line  s = D W.
#   max_Z  sum_{i,k}  a_i  *  entr( z_{ik} )      (  CVX  entr( x )  =  -x log x,  0*log(0)  = 0  )
#   s.t.   0  <=  Z,  each  row  sum  1,  and  ( D^T  ( a  ⊙  (E  -  Z) ) )  in  [ -rho,  rho  ]  elementwise
# (when  a_i=1/n  this  is  the  same  v=E-nΛ  dual;  the  n-averaging  and  1/n  in  a  are  the  only  relabeling).


def _solve_cvxpy_with_fallback(
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


def _a_row_from_sample_weight(
    n: int,
    sample_weight: Optional[np.ndarray],
) -> np.ndarray:
    if sample_weight is None:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if int(sw.shape[0]) != int(n):
        raise ValueError(f"sample_weight must have length n={n}, got {sw.shape[0]}.")
    if np.any(sw <= 0.0) or not np.isfinite(sw).all():
        raise ValueError("sample_weight must be finite and strictly positive for every entry.")
    return (sw / float(np.sum(sw))).astype(np.float64)


def build_softmax_ce_l1_primal_problem(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[cp.Problem, cp.Variable]:
    """
    Primal:  sum_i  a_i * CE_i( D W )  +  rho  *  sum |W|,  a_i  =  sw  /  sum( sw  ),
    or  a_i=1/n  if  sample_weight  is  None  (STE  mean  CE  convention  over  n  flat  rows).
    """
    n, p = D.shape
    Df = D.astype(np.float64)
    y_i = y.astype(np.int64)
    if y_i.ndim != 1 or y_i.shape[0] != n:
        raise ValueError(f"Expected y shape (n,) with n={n}, got {y_i.shape}.")
    if num_classes < 2 or int(y_i.min()) < 0 or int(y_i.max()) >= num_classes:
        raise ValueError(f"Invalid labels or num_classes={num_classes}.")

    a = _a_row_from_sample_weight(n, sample_weight)
    if not np.isclose(float(np.sum(a)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Internal: row weight vector must sum to 1.")

    E = np.zeros((n, num_classes), dtype=np.float64)
    E[np.arange(n), y_i] = 1.0
    W = cp.Variable((p, num_classes))
    scores = Df @ W
    row_lse = cp.log_sum_exp(scores, axis=1, keepdims=False)
    logits_y = cp.sum(cp.multiply(scores, E), axis=1)
    row_ce = row_lse - logits_y
    primal_obj = cp.sum(cp.multiply(a, row_ce)) + rho * cp.sum(cp.abs(W))
    prob = cp.Problem(cp.Minimize(primal_obj))
    return prob, W


def build_softmax_ce_l1_dual_problem(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[cp.Problem, cp.Variable]:
    """
    Dual:  max  sum_{i,k}  a_i  entr( Z_{ik} )  with  Z  a  (n×K)  simplex  row,  1'Z_i=1,
    and  D^T( a  ⊙  (E  -  Z) )  in  [ -rho,  rho]  (elementwise,  p×K).
    See  the  file  block  above  the  Fenchel  form.
    """
    n, p = D.shape
    Df = D.astype(np.float64)
    y_i = y.astype(np.int64)
    if y_i.ndim != 1 or y_i.shape[0] != n:
        raise ValueError(f"Expected y shape (n,) with n={n}, got {y_i.shape}.")
    if num_classes < 2 or int(y_i.min()) < 0 or int(y_i.max()) >= num_classes:
        raise ValueError(f"Invalid labels or num_classes={num_classes}.")

    a = _a_row_from_sample_weight(n, sample_weight)
    if not np.isclose(float(np.sum(a)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Internal: row weight vector must sum to 1.")

    E = np.zeros((n, num_classes), dtype=np.float64)
    E[np.arange(n), y_i] = 1.0
    Z = cp.Variable((n, num_classes))
    a_col = a.reshape(n, 1)
    M = cp.multiply(a_col, E - Z)
    DtM = Df.T @ M
    constraints: list = [
        Z >= 0.0,
        cp.sum(Z, axis=1) == 1.0,
        DtM <= float(rho),
        DtM >= -float(rho),
    ]
    dual_obj = cp.sum(cp.multiply(a_col, cp.entr(Z)))
    prob = cp.Problem(cp.Maximize(dual_obj), constraints)
    return prob, Z


def solve_multiclass_softmax_ce_l1_primal_dual(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
    *,
    sample_weight: Optional[np.ndarray] = None,
    solver_order: Tuple[str, ...] = ("CLARABEL", "SCS"),
    compute_dual: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Weighted  softmax  CE  +  L1  and  the  z-simplex  dual.  If  ``sample_weight  is  None``,
    uses  a_i=1/n  (mean  over  the  n  flat  training  rows).

    ``gap  =  primal  -  dual``  (numerically  small  at  an  accurate  solution  when  strong  duality  holds).

    If ``compute_dual`` is False, skips the second conic solve; ``dual_val`` and ``gap`` are NaN (faster).
    """
    primal_prob, W = build_softmax_ce_l1_primal_problem(
        D, y, rho, num_classes, sample_weight=sample_weight
    )
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

    if not compute_dual:
        return w_star, primal_val, float("nan"), float("nan")

    dual_prob, Z = build_softmax_ce_l1_dual_problem(
        D, y, rho, num_classes, sample_weight=sample_weight
    )
    _solve_cvxpy_with_fallback(
        dual_prob,
        has_solution=lambda: Z.value is not None,
        solver_order=solver_order,
    )
    if Z.value is None:
        raise RuntimeError("Softmax CE + L1 dual solver failed.")
    dual_val = float(dual_prob.value)
    if not np.isfinite(dual_val):
        raise FloatingPointError("Non-finite dual objective for softmax CE + L1.")

    gap = float(primal_val - dual_val)
    return w_star, primal_val, dual_val, gap
