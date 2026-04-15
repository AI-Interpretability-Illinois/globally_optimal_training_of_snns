from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss_functions import LossFunction


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
    epochs: int = 150
    batch_size: Optional[int] = None
    log_every: int = 10


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


def _build_feature_map(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
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
        readout_tr = _run_lif_stack_readout(
            x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        readout_va = _run_lif_stack_readout(
            x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        readout_te = _run_lif_stack_readout(
            x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        U_last = _sample_weight_matrix(
            rng, in_dim=hidden_dims[-1], out_dim=int(init_cfg.feature_count), variant=init_cfg.variant
        )
        d_train = (readout_tr @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        d_val = (readout_va @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        d_test = (readout_te @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        return d_train, d_val, d_test, {"U_in_list": np.array([], dtype=np.float64), "U_last": U_last}

    if init_cfg.mode == "pretraining":
        if init_cfg.pretrained_weights is None or len(init_cfg.pretrained_weights) == 0:
            raise ValueError("pretraining mode requires non-empty pretrained_weights.")
        weights = [w.astype(np.float64, copy=False) for w in init_cfg.pretrained_weights]
        n_hidden = len(hidden_dims)
        if len(weights) < n_hidden + 1:
            raise ValueError(
                f"Expected at least {n_hidden + 1} pretrained tensors (hidden layers + classifier), got {len(weights)}."
            )
        U_in_list: List[np.ndarray] = []
        in_dim_l = d_in
        for l, h in enumerate(hidden_dims):
            U_in_list.append(_coerce_hidden_weight(weights[l], in_dim=in_dim_l, out_dim=h))
            in_dim_l = h
        readout_tr = _run_lif_stack_readout(
            x_train, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        readout_va = _run_lif_stack_readout(
            x_val, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        readout_te = _run_lif_stack_readout(
            x_test, U_in_list, beta_list, threshold_list, last_layer_readout=init_cfg.last_layer_readout
        )
        cls_w = weights[n_hidden]
        if cls_w.ndim != 2:
            raise ValueError(f"Classifier weight must be rank-2, got shape {cls_w.shape}.")
        if cls_w.shape[1] == hidden_dims[-1]:
            U_last_base = cls_w.T
        elif cls_w.shape[0] == hidden_dims[-1]:
            U_last_base = cls_w
        else:
            raise ValueError(
                f"Classifier weight shape {cls_w.shape} incompatible with hidden dim {hidden_dims[-1]}."
            )
        target_p = int(init_cfg.feature_count)
        if U_last_base.shape[1] < target_p:
            extra = _sample_weight_matrix(
                rng, in_dim=hidden_dims[-1], out_dim=target_p - U_last_base.shape[1], variant=init_cfg.variant
            )
            U_last = np.concatenate([U_last_base, extra], axis=1)
        else:
            U_last = U_last_base[:, :target_p]
        d_train = (readout_tr @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        d_val = (readout_va @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        d_test = (readout_te @ U_last - init_cfg.bias >= 0.0).astype(np.float64)
        return d_train, d_val, d_test, {"pretrained_weights": np.array([], dtype=np.float64), "U_last": U_last}

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
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
        elif s == "SCS":
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=50000)
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
) -> BinaryCvxSolution:
    """
    Solve binary convex objective with L1 output regularization:
      min_w (1/n) * sum_i l(y_i, d_i^T w) + rho * ||w||_1
    and its Fenchel dual:
      max_u -F^*(-u) s.t. ||D^T u||_inf <= rho
    where F(z)=(1/n) sum_i l(y_i,z_i).
    """
    n, p = D.shape
    y = y_pm1.astype(np.float64)
    Df = D.astype(np.float64)

    if loss_name not in ("hinge", "hinge_ovr", "squared", "ce"):
        raise ValueError(f"Unsupported loss_name={loss_name}.")

    # ---------- Primal ----------
    w = cp.Variable(p)
    scores = Df @ w
    if loss_name in ("hinge", "hinge_ovr"):
        xi = cp.Variable(n, nonneg=True)
        margins = cp.multiply(y, scores)
        primal_obj_expr = (1.0 / n) * cp.sum(xi) + rho * cp.norm1(w)
        primal_prob = cp.Problem(cp.Minimize(primal_obj_expr), [margins >= 1.0 - xi])
    elif loss_name == "squared":
        resid = scores - y
        primal_obj_expr = (1.0 / n) * cp.sum_squares(resid) + rho * cp.norm1(w)
        primal_prob = cp.Problem(cp.Minimize(primal_obj_expr))
    else:  # ce -> binary logistic
        margins = cp.multiply(y, scores)
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
    constraints = [Df.T @ u <= rho, Df.T @ u >= -rho]

    if loss_name in ("hinge", "hinge_ovr"):
        # y_i u_i in [0, 1/n], objective = sum_i y_i u_i
        constraints.extend([yu >= 0.0, yu <= 1.0 / n])
        dual_obj = cp.Maximize(cp.sum(yu))
    elif loss_name == "squared":
        # -F*(-u) = sum_i (y_i u_i) - (n/4) * ||u||_2^2
        dual_obj = cp.Maximize(cp.sum(yu) - (n / 4.0) * cp.sum_squares(u))
    else:  # ce -> binary logistic
        # l(z)=log(1+exp(-y z)); domain: y_i u_i in [0,1/n]
        constraints.extend([yu >= 0.0, yu <= 1.0 / n])
        nyu = n * yu
        # -F*(-u) = (1/n) * sum_i [ entr(n y_i u_i) + entr(1 - n y_i u_i) ]
        dual_obj = cp.Maximize((1.0 / n) * cp.sum(cp.entr(nyu) + cp.entr(1.0 - nyu)))

    dual_prob = cp.Problem(dual_obj, constraints)
    _solve_with_fallback(dual_prob, has_solution=lambda: u.value is not None, solver_order=solver_order)
    if u.value is None:
        raise RuntimeError("Dual solver failed.")
    dual_val = float(dual_prob.value)
    if not np.isfinite(dual_val):
        raise FloatingPointError("Non-finite dual objective.")
    return BinaryCvxSolution(w=w_star, primal_obj=primal_val, dual_obj=dual_val, gap=float(primal_val - dual_val))


def _hinge_ovr_numpy(scores: np.ndarray, y: np.ndarray, margin: float = 1.0) -> float:
    n, c = scores.shape
    targets = -np.ones((n, c), dtype=np.float64)
    targets[np.arange(n), y] = 1.0
    losses = np.maximum(0.0, margin - targets * scores)
    return float(losses.mean())


def _compute_split_loss(loss_name: str, scores: np.ndarray, y: np.ndarray) -> float:
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
    print(
        (
            f"[cvx-run] method=cvx beta={float(solve_cfg.beta):.6g} lr={float(solve_cfg.lr):.6g} "
            f"bias={float(init_bias):.6g} "
            "batch_size=full"
        ),
        flush=True,
    )
    w = np.zeros((d_train.shape[1], num_classes), dtype=np.float64)
    primal_sum = 0.0
    dual_sum = 0.0
    for c in range(num_classes):
        y_bin = np.where(y_train == c, 1.0, -1.0).astype(np.float64)
        sol = solve_binary_l1_primal_dual(
            d_train,
            y_bin,
            rho=rho,
            loss_name=solve_cfg.loss_name,
        )
        w[:, c] = sol.w
        primal_sum += sol.primal_obj
        dual_sum += sol.dual_obj
    train_scores = d_train @ w
    val_scores = d_val @ w
    test_scores = d_test @ w
    train_loss = _compute_split_loss(solve_cfg.loss_name, train_scores, y_train)
    val_loss = _compute_split_loss(solve_cfg.loss_name, val_scores, y_val)
    test_loss = _compute_split_loss(solve_cfg.loss_name, test_scores, y_test)
    l1_penalty = float(np.abs(w).sum())
    train_objective = train_loss + rho * l1_penalty
    val_objective = val_loss + rho * l1_penalty
    test_objective = test_loss + rho * l1_penalty
    diag = CvxDiagnostics(
        primal_value=float(primal_sum),
        dual_value=float(dual_sum),
        gap=float(primal_sum - dual_sum),
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
    )
    return CvxSolveResult(
        trained_model={"weights": w},
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
    return CvxSolveResult(
        trained_model=model.cpu(),
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
) -> CvxSolveResult:
    ytr = _prepare_sequence_targets(y_train)
    yva = _prepare_sequence_targets(y_val)
    yte = _prepare_sequence_targets(y_test)
    d_train, d_val, d_test, _ = _build_feature_map(x_train=x_train, x_val=x_val, x_test=x_test, init_cfg=init_cfg)
    if solve_cfg.method == "cvx":
        return _run_cvx_method(
            d_train=d_train,
            y_train=ytr,
            d_val=d_val,
            y_val=yva,
            d_test=d_test,
            y_test=yte,
            solve_cfg=solve_cfg,
            init_bias=float(init_cfg.bias),
        )
    if solve_cfg.method == "sgd":
        run_device = choose_device() if device is None else device
        return _run_sgd_method(
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
    raise ValueError(f"Unknown solve method={solve_cfg.method}.")
