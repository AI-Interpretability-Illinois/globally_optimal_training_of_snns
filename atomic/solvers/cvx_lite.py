"""Primal-only L1 readout solvers (``cvx_lite``).

Why the full CVX path is slow
-----------------------------
``method=cvx`` hands the readout to CVXPY + CLARABEL/SCS as a *conic* program:

* XOR / hinge: one LP per OVR class, then a **second** dual LP (the OVR helper
  used to ignore ``compute_ce_dual`` and always solve the dual).
* MNIST / softmax CE: one exponential-cone program on ``W ∈ R^{p × K}`` with
  ``n`` log-sum-exp rows. For the MNIST preset that is ``n=6000``, ``p=1024``,
  ``K=10`` — interior-point / SCS first-order cone iterates on a problem of
  that size is how a single ``beta`` sat for ~30 h.

None of that dual / cone machinery is required to *use* the readout: we only
need ``W*`` of the primal

    min_W  (1/n) Σ_i ℓ(y_i, D_i W)  +  ρ ||W||_1

and the primal objective value. Dual / gap diagnostics are left as NaN.

Algorithms (glmnet-style, primal only)
--------------------------------------
* **squared** — cyclic coordinate descent (exact 1-D soft-threshold, the
  classical LASSO CD / glmnet inner loop).
* **hinge / hinge_ovr / ce (binary)** — FISTA with L1 proximal map
  (soft-threshold). Hinge uses the hinge subgradient; logistic uses the
  Lipschitz-smooth logistic gradient. Backtracking on the true primal
  objective, so a step is only accepted when the primal decreases.
* **ce (multiclass)** — FISTA on vec(W) for softmax-CE + elementwise L1,
  same primal as :func:`solvers.loss_functions.solve_multiclass_softmax_ce_l1_primal_dual`.

No dual is formed. No CVXPY is imported.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _record_primal(history: Optional[List[float]], obj: float) -> None:
    if history is None:
        return
    history.append(float(obj))


def _soft_threshold(x: np.ndarray, kappa: float) -> np.ndarray:
    if kappa <= 0.0:
        return x
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def _power_iteration_dtd(D: np.ndarray, *, n_iter: int = 20, seed: int = 0) -> float:
    """Largest eigenvalue of ``D.T @ D`` (= ``||D||_2^2``).

    Uses a thin SVD (``compute_uv=False``) rather than power iteration: the
    latter overflows in float64 on even modest Gaussian design matrices, which
    is the same class of ``matmul`` inf/nan warnings the CVXPY path hit on MNIST.
    ``n_iter`` / ``seed`` are accepted for call-site compatibility and ignored.
    """
    del n_iter, seed
    n, p = D.shape
    if n == 0 or p == 0:
        raise ValueError(f"D must be non-empty, got shape={D.shape}.")
    svals = np.linalg.svd(np.asarray(D, dtype=np.float64), compute_uv=False)
    if svals.size == 0:
        raise RuntimeError(f"SVD of D shape={D.shape} returned no singular values.")
    smax = float(svals[0])
    if not np.isfinite(smax):
        raise FloatingPointError(f"Non-finite spectral norm of D: {smax}.")
    return smax * smax


def _row_weights(n: int, sample_weight: Optional[np.ndarray]) -> np.ndarray:
    """Nonnegative weights ``a`` with ``sum(a)=1``. Unweighted ⇒ ``a_i = 1/n``."""
    if sample_weight is None:
        return np.full(int(n), 1.0 / float(n), dtype=np.float64)
    sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if sw.shape[0] != int(n):
        raise ValueError(f"sample_weight length {sw.shape[0]} != n={n}.")
    if np.any(sw <= 0.0) or (not np.isfinite(sw).all()):
        raise ValueError("sample_weight must be finite and strictly positive on every row.")
    return sw / float(sw.sum())


def _logistic_loss_and_grad(
    D: np.ndarray,
    y_pm1: np.ndarray,
    w: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Weighted logistic ``Σ a_i log(1 + exp(-y ⊙ Dw))`` and its gradient."""
    n = D.shape[0]
    a = _row_weights(n, sample_weight)
    scores = D @ w
    margins = y_pm1 * scores
    # log1p(exp(-m)) = max(-m, 0) + log1p(exp(-|m|))  (stable)
    per = np.maximum(-margins, 0.0) + np.log1p(np.exp(-np.abs(margins)))
    loss = float(np.dot(a, per))
    # d/ds log(1+exp(-y s)) = -y * sigmoid(-y s) = -y / (1 + exp(y s))
    sig_neg = 1.0 / (1.0 + np.exp(np.clip(margins, -60.0, 60.0)))
    resid = a * (-y_pm1 * sig_neg)
    grad = D.T @ resid
    return loss, grad


def _hinge_loss_and_subgrad(
    D: np.ndarray,
    y_pm1: np.ndarray,
    w: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Weighted hinge ``Σ a_i max(0, 1 - y ⊙ Dw)`` and a subgradient."""
    n = D.shape[0]
    a = _row_weights(n, sample_weight)
    scores = D @ w
    margins = y_pm1 * scores
    viol = 1.0 - margins
    loss = float(np.dot(a, np.maximum(viol, 0.0)))
    active = (viol > 0.0).astype(np.float64)
    resid = a * (-y_pm1 * active)
    grad = D.T @ resid
    return loss, grad


def _squared_loss_and_grad(D: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared ``(1/n) ||Dw - y||^2`` and its gradient ``(2/n) D.T (Dw-y)``."""
    n = D.shape[0]
    resid = D @ w - y
    loss = float(np.dot(resid, resid) / float(n))
    grad = (2.0 / float(n)) * (D.T @ resid)
    return loss, grad


def _softmax_ce_loss_and_grad(
    D: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Weighted softmax CE ``Σ a_i (lse(D_i W) - (D_i W)_{y_i})`` and d/dW."""
    n = D.shape[0]
    a = _row_weights(n, sample_weight)
    scores = D @ W
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    Z = exp_s.sum(axis=1, keepdims=True)
    log_z = np.log(Z) + scores.max(axis=1, keepdims=True)
    row_ce = log_z.reshape(-1) - scores[np.arange(n), y]
    loss = float(np.dot(a, row_ce))
    P = exp_s / Z
    P[np.arange(n), y] -= 1.0
    grad = D.T @ (P * a[:, None])
    return loss, grad


def lasso_cd_squared(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    w0: Optional[np.ndarray] = None,
    history: Optional[List[float]] = None,
) -> Tuple[np.ndarray, float, int]:
    """Cyclic coordinate descent for ``(1/n)||Dw - y||^2 + ρ||w||_1``.

    1-D update (column ``j``, residual ``r = Dw - y``):

        w_j ← S( w_j - (D_j·r) / ||D_j||^2 ,  (ρ n) / (2 ||D_j||^2) )
    """
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = D.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != n={n}.")
    if rho < 0.0:
        raise ValueError(f"rho must be >= 0, got {rho}.")
    w = np.zeros(p, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).reshape(-1).copy()
    if w.shape[0] != p:
        raise ValueError(f"w0 length {w.shape[0]} != p={p}.")
    col_sq = np.einsum("ij,ij->j", D, D)
    r = D @ w - y
    prev_obj = float("inf")
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        max_delta = 0.0
        for j in range(p):
            aj = float(col_sq[j])
            if aj == 0.0:
                if w[j] != 0.0:
                    r -= D[:, j] * w[j]
                    w[j] = 0.0
                continue
            xj_dot_r = float(D[:, j] @ r)
            unthresh = w[j] - xj_dot_r / aj
            kappa = (float(rho) * float(n)) / (2.0 * aj)
            w_new = float(np.sign(unthresh) * max(abs(unthresh) - kappa, 0.0))
            delta = w_new - w[j]
            if delta != 0.0:
                r += D[:, j] * delta
                w[j] = w_new
                max_delta = max(max_delta, abs(delta))
        obj = float(np.dot(r, r) / float(n) + float(rho) * float(np.abs(w).sum()))
        if (not np.isfinite(obj)) or (not np.isfinite(w).all()):
            raise FloatingPointError(f"Non-finite CD iterate at iter={n_iter}: obj={obj}.")
        _record_primal(history, obj)
        rel = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
        if rel <= float(tol) and max_delta <= float(tol):
            return w, obj, n_iter
        prev_obj = obj
    obj = float(np.dot(r, r) / float(n) + float(rho) * float(np.abs(w).sum()))
    return w, obj, n_iter


def _fista_binary(
    D: np.ndarray,
    y_pm1: np.ndarray,
    rho: float,
    *,
    loss_name: str,
    max_iter: int = 5000,
    tol: float = 1e-6,
    w0: Optional[np.ndarray] = None,
    lipschitz_seed: int = 0,
    history: Optional[List[float]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """Proximal-gradient + L1 prox for binary hinge / logistic / squared (primal only).

    Logistic / squared are Lipschitz-smooth → FISTA. Hinge is not; we use ISTA
    (no Nesterov momentum) so the extrapolated point cannot blow up.
    """
    D = np.asarray(D, dtype=np.float64)
    y_pm1 = np.asarray(y_pm1, dtype=np.float64).reshape(-1)
    n, p = D.shape
    if y_pm1.shape[0] != n:
        raise ValueError(f"y length {y_pm1.shape[0]} != n={n}.")
    if rho < 0.0:
        raise ValueError(f"rho must be >= 0, got {rho}.")
    a = _row_weights(n, sample_weight)
    a_max = float(a.max())
    if loss_name == "squared":
        if sample_weight is not None:
            raise ValueError("cvx_lite squared does not accept sample_weight.")
        loss_grad = lambda w: _squared_loss_and_grad(D, y_pm1, w)
        lip_factor = 2.0 / float(n)
        nesterov = True
    elif loss_name == "ce":
        loss_grad = lambda w: _logistic_loss_and_grad(D, y_pm1, w, sample_weight=sample_weight)
        lip_factor = 0.25 * a_max
        nesterov = True
    elif loss_name in ("hinge", "hinge_ovr"):
        loss_grad = lambda w: _hinge_loss_and_subgrad(D, y_pm1, w, sample_weight=sample_weight)
        lip_factor = a_max
        nesterov = False
    else:
        raise ValueError(f"Unsupported binary lite loss_name={loss_name}.")

    spec = _power_iteration_dtd(D, seed=int(lipschitz_seed))
    L = lip_factor * spec
    step = 1.0 / L if L > 0.0 else 1.0
    w = np.zeros(p, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).reshape(-1).copy()
    if w.shape[0] != p:
        raise ValueError(f"w0 length {w.shape[0]} != p={p}.")
    z = w.copy()
    t = 1.0
    prev_obj = float("inf")
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        data_loss, grad = loss_grad(z)
        if (not np.isfinite(data_loss)) or (not np.isfinite(grad).all()):
            raise FloatingPointError(
                f"Non-finite loss/grad at iter={n_iter}: loss={data_loss}."
            )
        accepted = False
        step_try = step
        w_next = w
        obj_next = prev_obj
        for _bt in range(30):
            w_next = _soft_threshold(z - step_try * grad, step_try * float(rho))
            if not np.isfinite(w_next).all():
                step_try *= 0.5
                continue
            data_next, _ = loss_grad(w_next)
            obj_next = float(data_next + float(rho) * float(np.abs(w_next).sum()))
            if not np.isfinite(obj_next):
                step_try *= 0.5
                continue
            obj_z = float(data_loss + float(rho) * float(np.abs(z).sum()))
            if obj_next <= obj_z + 1e-12 or obj_next <= prev_obj + 1e-12:
                accepted = True
                step = step_try
                break
            step_try *= 0.5
        if not accepted:
            step = step_try
        if nesterov:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = w_next + ((t - 1.0) / t_next) * (w_next - w)
            if not np.isfinite(z).all():
                z = w_next.copy()
                t_next = 1.0
            t = t_next
        else:
            z = w_next
        w = w_next
        _record_primal(history, obj_next)
        rel = abs(prev_obj - obj_next) / max(1.0, abs(prev_obj))
        if rel <= float(tol):
            return w, obj_next, n_iter
        prev_obj = obj_next
    data_loss, _ = loss_grad(w)
    obj = float(data_loss + float(rho) * float(np.abs(w).sum()))
    return w, obj, n_iter


def fista_softmax_ce_l1(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    num_classes: int,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    W0: Optional[np.ndarray] = None,
    lipschitz_seed: int = 0,
    history: Optional[List[float]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """FISTA + elementwise L1 prox for softmax-CE + ρ||W||_1 (primal only)."""
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n, p = D.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != n={n}.")
    if num_classes < 2 or int(y.min()) < 0 or int(y.max()) >= num_classes:
        raise ValueError(f"Invalid labels or num_classes={num_classes}.")
    if rho < 0.0:
        raise ValueError(f"rho must be >= 0, got {rho}.")
    a = _row_weights(n, sample_weight)
    spec = _power_iteration_dtd(D, seed=int(lipschitz_seed))
    L = float(a.max()) * spec
    step = 1.0 / L if L > 0.0 else 1.0
    W = (
        np.zeros((p, num_classes), dtype=np.float64)
        if W0 is None
        else np.asarray(W0, dtype=np.float64).copy()
    )
    if W.shape != (p, num_classes):
        raise ValueError(f"W0 shape {W.shape} != {(p, num_classes)}.")
    Z = W.copy()
    t = 1.0
    prev_obj = float("inf")
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        data_loss, grad = _softmax_ce_loss_and_grad(D, y, Z, sample_weight=sample_weight)
        if (not np.isfinite(data_loss)) or (not np.isfinite(grad).all()):
            raise FloatingPointError(
                f"Non-finite softmax loss/grad at iter={n_iter}: loss={data_loss}."
            )
        accepted = False
        step_try = step
        W_next = W
        obj_next = prev_obj
        for _bt in range(30):
            W_next = _soft_threshold(Z - step_try * grad, step_try * float(rho))
            data_next, _ = _softmax_ce_loss_and_grad(D, y, W_next, sample_weight=sample_weight)
            obj_next = float(data_next + float(rho) * float(np.abs(W_next).sum()))
            if (not np.isfinite(obj_next)) or (not np.isfinite(W_next).all()):
                raise FloatingPointError(
                    f"Non-finite softmax-FISTA iterate at iter={n_iter}: obj={obj_next}."
                )
            obj_z = float(data_loss + float(rho) * float(np.abs(Z).sum()))
            if obj_next <= obj_z + 1e-12 or obj_next <= prev_obj + 1e-12:
                accepted = True
                step = step_try
                break
            step_try *= 0.5
        if not accepted:
            step = step_try
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_next + ((t - 1.0) / t_next) * (W_next - W)
        W = W_next
        t = t_next
        _record_primal(history, obj_next)
        rel = abs(prev_obj - obj_next) / max(1.0, abs(prev_obj))
        if rel <= float(tol):
            return W, obj_next, n_iter
        prev_obj = obj_next
    data_loss, _ = _softmax_ce_loss_and_grad(D, y, W, sample_weight=sample_weight)
    obj = float(data_loss + float(rho) * float(np.abs(W).sum()))
    return W, obj, n_iter


def solve_binary_l1_primal_lite(
    D: np.ndarray,
    y_pm1: np.ndarray,
    rho: float,
    loss_name: str,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    history: Optional[List[float]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """Primal-only binary L1 readout. Returns ``(w, primal_obj, n_iter)``."""
    if loss_name == "squared":
        if sample_weight is not None:
            raise ValueError("cvx_lite squared does not accept sample_weight.")
        return lasso_cd_squared(D, y_pm1, rho, max_iter=max_iter, tol=tol, history=history)
    return _fista_binary(
        D,
        y_pm1,
        rho,
        loss_name=loss_name,
        max_iter=max_iter,
        tol=tol,
        history=history,
        sample_weight=sample_weight,
    )


def solve_multiclass_primal_lite(
    D: np.ndarray,
    y: np.ndarray,
    rho: float,
    loss_name: str,
    num_classes: int,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    history: Optional[List[float]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """Primal-only multiclass L1 readout. Returns ``(W, primal_obj, n_iter)``.

    Softmax CE is a *joint* problem on ``W``. Hinge / hinge_ovr / squared /
    binary-CE are independent OVR columns whose primal values are summed
    (same reduction as ``method=cvx``). When ``history`` is given, CE appends
    the joint primal each FISTA step; OVR concatenates per-class traces in
    class-index order (not a joint-primal trajectory).
    """
    if loss_name == "ce":
        if num_classes < 2:
            raise ValueError(f"Softmax CE requires num_classes>=2, got {num_classes}.")
        return fista_softmax_ce_l1(
            D,
            y,
            rho,
            num_classes,
            max_iter=max_iter,
            tol=tol,
            history=history,
            sample_weight=sample_weight,
        )
    W = np.zeros((D.shape[1], int(num_classes)), dtype=np.float64)
    primal_sum = 0.0
    n_iter_max = 0
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    for c in range(int(num_classes)):
        y_bin = np.where(y == int(c), 1.0, -1.0).astype(np.float64)
        w_c, p_c, n_it = solve_binary_l1_primal_lite(
            D,
            y_bin,
            rho,
            loss_name,
            max_iter=max_iter,
            tol=tol,
            history=history,
            sample_weight=sample_weight,
        )
        W[:, c] = w_c
        primal_sum += float(p_c)
        n_iter_max = max(n_iter_max, int(n_it))
    return W, float(primal_sum), n_iter_max


if __name__ == "__main__":
    # Limited-sample smoke: squared CD vs a known sparse LASSO instance.
    rng = np.random.default_rng(0)
    n_debug, p_debug = 32, 8
    D = rng.standard_normal((n_debug, p_debug))
    w_true = np.zeros(p_debug)
    w_true[:2] = [1.5, -0.8]
    y = D @ w_true + 0.01 * rng.standard_normal(n_debug)
    rho = 0.05
    w_hat, obj, n_it = lasso_cd_squared(D, y, rho, max_iter=2000, tol=1e-10)
    data_term = float(np.dot(D @ w_hat - y, D @ w_hat - y) / n_debug)
    recon = data_term + rho * float(np.abs(w_hat).sum())
    print(
        f"[cvx_lite debug] squared CD n={n_debug} p={p_debug} n_iter={n_it} "
        f"obj={obj:.8g} recon={recon:.8g} nnz={int(np.count_nonzero(w_hat))}"
    )
    if abs(obj - recon) > 1e-10:
        raise AssertionError(f"CD objective mismatch: {obj} vs {recon}")

    y_pm1 = np.sign(y)
    y_pm1[y_pm1 == 0.0] = 1.0
    w_h, obj_h, n_h = solve_binary_l1_primal_lite(D, y_pm1, rho, "hinge", max_iter=2000, tol=1e-8)
    print(f"[cvx_lite debug] hinge FISTA n_iter={n_h} primal={obj_h:.8g} ||w||1={float(np.abs(w_h).sum()):.6g}")

    y_cls = (y_pm1 > 0).astype(np.int64)
    W_ce, obj_ce, n_ce = solve_multiclass_primal_lite(
        D, y_cls, rho, "ce", num_classes=2, max_iter=2000, tol=1e-8
    )
    print(f"[cvx_lite debug] softmax-CE FISTA n_iter={n_ce} primal={obj_ce:.8g} W.shape={W_ce.shape}")

