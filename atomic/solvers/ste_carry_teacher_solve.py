from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss_functions import LossFunction

# Dataset protocol (implemented by run script dataclass).
class _CarryTFData(Protocol):
    X_train: np.ndarray
    y_sum_train: np.ndarray
    y_carry_train: np.ndarray
    X_val: np.ndarray
    y_sum_val: np.ndarray
    y_carry_val: np.ndarray
    X_test: np.ndarray
    y_sum_test: np.ndarray
    y_carry_test: np.ndarray
    num_sum_classes: int
    d_in: int
    T: int


def resolve_ste_sum_loss_name(ste_sum_loss: str, base: int) -> str:
    s = str(ste_sum_loss)
    if s == "auto":
        return "hinge" if int(base) == 2 else "ce"
    return s


def resolve_ste_carry_loss_name(ste_carry_loss: str) -> str:
    s = str(ste_carry_loss)
    if s == "auto":
        return "hinge"
    return s


# ------------------------------------------------------------
# Model (CarryAugmentedSNN: shared trunk, two heads)
# ------------------------------------------------------------


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


class CarryAugmentedSNN(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        base: int,
        L: int,
        P_rec: int,
        P_last: int,
        K_parallel: int,
        beta_leak: float,
        threshold: float,
        last_layer_readout: str,
    ):
        super().__init__()
        self.base = int(base)
        self.k_parallel = int(K_parallel)
        self.last_layer_readout = str(last_layer_readout)
        sub_p_rec = _parallel_branch_width(int(P_rec), self.k_parallel, "P_rec")
        sub_p_last = _parallel_branch_width(int(P_last), self.k_parallel, "P_last")
        hidden_dims = [sub_p_rec] * max(int(L) - 2, 0) + [sub_p_last]
        if int(L) <= 1:
            hidden_dims = [sub_p_last]
        self.branches = nn.ModuleList(
            [_SNNParallelBranch(d_in, hidden_dims, beta_leak, threshold) for _ in range(self.k_parallel)]
        )
        sum_out_dim = 1 if self.base == 2 else self.base
        self.sum_head = nn.Linear(int(P_last), int(sum_out_dim), bias=False)
        self.carry_head = nn.Linear(int(P_last), 1, bias=False)

    def hidden_weight_list(self) -> List[np.ndarray]:
        """LIF-stack weights in CVX pretraining order (branch-major, layer-major)."""
        weights: List[np.ndarray] = []
        for br in self.branches:
            for fc in br.fcs:
                weights.append(fc.weight.detach().cpu().numpy().copy())
        if len(weights) == 0:
            raise RuntimeError("CarryAugmentedSNN has no hidden Linear layers.")
        return weights

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, steps, _ = x_seq.shape
        branch_mems = [
            [lif.init_leaky().to(x_seq.device) for lif in branch.lifs]
            for branch in self.branches
        ]
        sum_logits_seq: List[torch.Tensor] = []
        carry_logits_seq: List[torch.Tensor] = []
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
            sum_logits_seq.append(self.sum_head(h_cat))
            carry_logits_seq.append(self.carry_head(h_cat))
        return torch.stack(sum_logits_seq, dim=1), torch.stack(carry_logits_seq, dim=1)


def carry_snn_path_reg(model: CarryAugmentedSNN) -> torch.Tensor:
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
    head_sq = (model.sum_head.weight**2).sum(dim=0) + (model.carry_head.weight**2).sum(dim=0)
    reg_sq = (head_sq * v_full).sum()
    return torch.sqrt(reg_sq + 1e-12)


def _binary_preds_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3 and logits.shape[-1] == 1:
        return (logits[:, :, 0] >= 0.0).long()
    raise ValueError(f"Expected binary logits shape (N,T,1), got {tuple(logits.shape)}.")


def _mean_first_wrong_time(pred: np.ndarray, y: np.ndarray) -> float:
    """Among sequences with at least one error, mean 0-based index of first error; nan if none wrong."""
    n, t = pred.shape
    if t < 1:
        return float("nan")
    first: list[int] = []
    for i in range(n):
        row_ok = pred[i] == y[i]
        if row_ok.all():
            continue
        first.append(int(np.argmax(~row_ok)))
    return float(np.mean(first)) if first else float("nan")


def carry_teacher_forcing_token_metrics(
    sum_pred: np.ndarray,
    carry_pred: np.ndarray,
    y_sum: np.ndarray,
    y_carry: np.ndarray,
) -> Dict[str, float]:
    sum_ok = sum_pred == y_sum
    carry_ok = carry_pred == y_carry
    both_ok = sum_ok & carry_ok
    sum_seq_ok = sum_ok.all(axis=1)
    carry_seq_ok = carry_ok.all(axis=1)
    out: Dict[str, float] = {
        "sum_token_acc": float(sum_ok.mean()),
        "carry_token_acc": float(carry_ok.mean()),
        "joint_token_acc": float(both_ok.mean()),
        "joint_seq_acc": float(both_ok.all(axis=1).mean()),
        "sum_seq_acc": float(sum_seq_ok.mean()),
        "carry_seq_acc": float(carry_seq_ok.mean()),
        "mean_first_wrong_sum_among_error_seq": _mean_first_wrong_time(sum_pred, y_sum),
        "mean_first_wrong_carry_among_error_seq": _mean_first_wrong_time(carry_pred, y_carry),
    }
    return out


def ste_predict_sum_carry(model: CarryAugmentedSNN, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        sum_logits, carry_logits = model(xt)
    b = int(model.base)
    if b == 2:
        sum_pred = (sum_logits[:, :, 0] >= 0.0).long().cpu().numpy().astype(np.int64)
    else:
        sum_pred = torch.argmax(sum_logits, dim=2).cpu().numpy().astype(np.int64)
    carry_pred = (carry_logits[:, :, 0] >= 0.0).long().cpu().numpy().astype(np.int64)
    return sum_pred, carry_pred


def _eval_ste(
    model: CarryAugmentedSNN,
    x: np.ndarray,
    y_sum: np.ndarray,
    y_carry: np.ndarray,
    *,
    lambda_sum: float,
    lambda_carry: float,
    beta_path_reg: float,
    sum_loss_name: str,
    carry_loss_name: str,
    tf_objective: str,
    ste_time_loss: str,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    ys = torch.tensor(y_sum, dtype=torch.long, device=device)
    yc = torch.tensor(y_carry, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        sum_logits, carry_logits = model(xt)
        loss_sum, loss_carry = LossFunction.ste_carry_tf_data_losses_scalar(
            sum_logits,
            ys,
            carry_logits,
            yc,
            base=int(model.base),
            sum_loss_name=sum_loss_name,
            carry_loss_name=carry_loss_name,
            ste_time_loss=str(ste_time_loss),
        )
        loss_sum_u, loss_carry_u = LossFunction.ste_carry_tf_data_losses_scalar(
            sum_logits,
            ys,
            carry_logits,
            yc,
            base=int(model.base),
            sum_loss_name=sum_loss_name,
            carry_loss_name=carry_loss_name,
            ste_time_loss="uniform",
        )
        reg = carry_snn_path_reg(model) if float(beta_path_reg) > 0.0 else torch.zeros((), device=device, dtype=loss_sum.dtype)
        total = LossFunction.carry_teacher_forcing_total(
            loss_sum,
            loss_carry,
            reg,
            lambda_sum=lambda_sum,
            lambda_carry=lambda_carry,
            beta=float(beta_path_reg),
            tf_objective=tf_objective,
        )
        data_loss = LossFunction.carry_teacher_forcing_total(
            loss_sum,
            loss_carry,
            torch.zeros((), device=device, dtype=loss_sum.dtype),
            lambda_sum=lambda_sum,
            lambda_carry=lambda_carry,
            beta=0.0,
            tf_objective=tf_objective,
        )
        if model.base == 2:
            sum_pred = _binary_preds_from_logits(sum_logits).cpu().numpy().astype(np.int64)
        else:
            sum_pred = torch.argmax(sum_logits, dim=2).cpu().numpy().astype(np.int64)
        carry_pred = _binary_preds_from_logits(carry_logits).cpu().numpy().astype(np.int64)
    metrics = carry_teacher_forcing_token_metrics(sum_pred, carry_pred, y_sum, y_carry)
    metrics.update({
        "loss_sum": float(loss_sum.item()),
        "loss_carry": float(loss_carry.item()),
        "loss_sum_mean_t": float(loss_sum_u.item()),
        "loss_carry_mean_t": float(loss_carry_u.item()),
        "loss_total": float(total.item()),
        "loss_data": float(data_loss.item()),
        "ste_time_loss": str(ste_time_loss),
    })
    return metrics


def ste_sweep_and_train(
    *,
    ds: _CarryTFData,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    ste_last_layer_readout: str,
    ste_epochs: int,
    batch_size: int,
    optimizer_name: str,
    beta_leak: float,
    threshold: float,
    seed: int,
    ste_lr_grid: Sequence[float],
    ste_beta_grid: Sequence[float],
    lambda_sum: float,
    lambda_carry: float,
    ste_sum_loss: str = "auto",
    ste_carry_loss: str = "auto",
    tf_objective: str = "joint",
    ste_time_loss: str = "ramp",
) -> Tuple[CarryAugmentedSNN, Dict[str, float], Dict[str, float]]:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu"))
    )
    sum_n = resolve_ste_sum_loss_name(ste_sum_loss, ds.num_sum_classes)
    carry_n = resolve_ste_carry_loss_name(ste_carry_loss)

    best_score = float("inf")
    best_params: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, Any]] = None

    xtr = torch.tensor(ds.X_train, dtype=torch.float32, device=device)
    ys_tr = torch.tensor(ds.y_sum_train, dtype=torch.long, device=device)
    yc_tr = torch.tensor(ds.y_carry_train, dtype=torch.long, device=device)

    for lr in ste_lr_grid:
        for beta in ste_beta_grid:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = CarryAugmentedSNN(
                d_in=ds.d_in,
                base=ds.num_sum_classes,
                L=L,
                P_rec=P_rec,
                P_last=P_last,
                K_parallel=K_parallel,
                beta_leak=beta_leak,
                threshold=threshold,
                last_layer_readout=ste_last_layer_readout,
            ).to(device)
            params = list(model.parameters())
            if optimizer_name.lower() == "sgd":
                opt = torch.optim.SGD(params, lr=float(lr))
            else:
                opt = torch.optim.Adam(params, lr=float(lr))
            sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
            n = ds.X_train.shape[0]
            if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= n:
                bs = n
            else:
                bs = int(batch_size)
            rng = np.random.default_rng(seed)
            run_best_score = float("inf")
            run_best_state: Optional[Dict[str, Any]] = None
            for _epoch in range(int(ste_epochs)):
                model.train()
                perm = np.arange(n) if bs >= n else rng.permutation(n)
                for start in range(0, n, bs):
                    idx = perm[start: start + bs]
                    xb = xtr[idx]
                    ysb = ys_tr[idx]
                    ycb = yc_tr[idx]
                    sum_logits, carry_logits = model(xb)
                    l_s, l_c = LossFunction.ste_carry_tf_data_losses_scalar(
                        sum_logits,
                        ysb,
                        carry_logits,
                        ycb,
                        base=ds.num_sum_classes,
                        sum_loss_name=sum_n,
                        carry_loss_name=carry_n,
                        ste_time_loss=str(ste_time_loss),
                    )
                    reg = carry_snn_path_reg(model) if float(beta) > 0.0 else torch.zeros((), device=device, dtype=l_s.dtype)
                    loss = LossFunction.carry_teacher_forcing_total(
                        l_s,
                        l_c,
                        reg,
                        lambda_sum=lambda_sum,
                        lambda_carry=lambda_carry,
                        beta=float(beta),
                        tf_objective=tf_objective,
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                va = _eval_ste(
                    model,
                    ds.X_val,
                    ds.y_sum_val,
                    ds.y_carry_val,
                    lambda_sum=lambda_sum,
                    lambda_carry=lambda_carry,
                    beta_path_reg=float(beta),
                    sum_loss_name=sum_n,
                    carry_loss_name=carry_n,
                    tf_objective=tf_objective,
                    ste_time_loss=str(ste_time_loss),
                )
                data_score = float(va["loss_data"])
                sched.step(data_score)
                if data_score < run_best_score:
                    run_best_score = data_score
                    run_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if run_best_state is None:
                raise RuntimeError(f"STE run produced no checkpoint lr={lr} beta={beta}.")
            print(
                f"[ste-ar] lr={float(lr):.4g} beta={float(beta):.4g} bs={bs} "
                f"best_val_data={run_best_score:.6f} last_val_data={float(va['loss_data']):.6f}",
                flush=True,
            )
            if run_best_score < best_score:
                best_score = run_best_score
                best_params = {"lr": float(lr), "beta": float(beta)}
                best_state = run_best_state
    if best_params is None or best_state is None:
        raise RuntimeError("No STE candidate found.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    best_model = CarryAugmentedSNN(
        d_in=ds.d_in,
        base=ds.num_sum_classes,
        L=L,
        P_rec=P_rec,
        P_last=P_last,
        K_parallel=K_parallel,
        beta_leak=beta_leak,
        threshold=threshold,
        last_layer_readout=ste_last_layer_readout,
    ).to(device)
    best_model.load_state_dict(best_state)
    test_metrics = _eval_ste(
        best_model,
        ds.X_test,
        ds.y_sum_test,
        ds.y_carry_test,
        lambda_sum=lambda_sum,
        lambda_carry=lambda_carry,
        beta_path_reg=float(best_params["beta"]),
        sum_loss_name=sum_n,
        carry_loss_name=carry_n,
        tf_objective=tf_objective,
        ste_time_loss=str(ste_time_loss),
    )
    return best_model, best_params, test_metrics
