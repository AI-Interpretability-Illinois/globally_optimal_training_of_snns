from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import snntorch as snn
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
class SteModelConfig:
    d_in: int
    num_classes: int
    L: int
    P_rec: int
    P_last: int
    beta_leak: float = 0.99
    threshold: float = 1.0
    last_layer_readout: str = "membrane"


@dataclass
class SteSolveConfig:
    loss_name: str = "hinge_ovr"
    optimizer_name: str = "adam"
    lr: float = 1e-3
    epochs: int = 100
    batch_size: Optional[int] = None
    log_every: int = 10
    weight_decay: float = 0.0
    beta_path_reg: float = 0.0


@dataclass
class SteSolveResult:
    model: nn.Module
    loss_history: List[float]
    best_losses: Dict[str, float]
    final_train_objective: float


class SNNBaselineSeq(nn.Module):
    def __init__(self, cfg: SteModelConfig):
        super().__init__()
        hidden_dims = [cfg.P_rec] * max(cfg.L - 2, 0) + [cfg.P_last]
        if cfg.L <= 1:
            hidden_dims = [cfg.P_last]
        self.last_layer_readout = cfg.last_layer_readout
        fcs: List[nn.Module] = []
        lifs: List[nn.Module] = []
        in_dim = cfg.d_in
        for h_dim in hidden_dims:
            fcs.append(nn.Linear(in_dim, h_dim, bias=False))
            lifs.append(snn.Leaky(beta=cfg.beta_leak, threshold=cfg.threshold))
            in_dim = h_dim
        self.fcs = nn.ModuleList(fcs)
        self.lifs = nn.ModuleList(lifs)
        self.classifier = nn.Linear(cfg.P_last, cfg.num_classes, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, steps, _ = x_seq.shape
        mems = [lif.init_leaky().to(x_seq.device) for lif in self.lifs]
        logits_seq = []
        for t in range(steps):
            h = x_seq[:, t, :]
            last_spk = None
            last_mem = None
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                spk, mem = lif(fc(h), mems[i])
                mems[i] = mem
                h = spk
                last_spk = spk
                last_mem = mem
            readout = last_mem if self.last_layer_readout == "membrane" else last_spk
            logits_seq.append(self.classifier(readout))
        return torch.stack(logits_seq, dim=1)


def snn_path_reg(model: SNNBaselineSeq) -> torch.Tensor:
    """snn_p2-style path regularizer for the STE baseline."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(model.fcs) == 0:
        reg_sq = (model.classifier.weight ** 2).sum()
        return torch.sqrt(reg_sq + 1e-12)
    in_dim = model.fcs[0].in_features
    v = torch.ones(in_dim, device=device, dtype=dtype)
    for fc in model.fcs:
        v = (fc.weight ** 2) @ v
    reg_sq = ((model.classifier.weight ** 2) * v.unsqueeze(0)).sum()
    return torch.sqrt(reg_sq + 1e-12)


def _compute_eval_loss(
    model: SNNBaselineSeq,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_name: str,
    beta_path_reg: float = 0.0,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = LossFunction.compute(name=loss_name, y=y, f_x=logits).value
        if beta_path_reg > 0.0:
            loss = loss + float(beta_path_reg) * snn_path_reg(model)
        return float(loss.item())


def _sequence_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.ndim != 3:
        raise ValueError(f"Expected sequence logits with shape (N,T,C), got {tuple(logits.shape)}.")
    if y.ndim == 1:
        last_logits = logits[:, -1, :]
        if last_logits.shape[1] == 1:
            preds = (last_logits[:, 0] >= 0.0).long()
        else:
            preds = torch.argmax(last_logits, dim=1)
        return float((preds == y.long()).float().mean().item())
    if y.ndim == 2:
        if tuple(logits.shape[:2]) != tuple(y.shape):
            raise ValueError(f"Sequence label shape mismatch: logits={tuple(logits.shape)}, y={tuple(y.shape)}.")
        if logits.shape[2] == 1:
            preds = (logits[:, :, 0] >= 0.0).long()
        else:
            preds = torch.argmax(logits, dim=2)
        return float((preds == y.long()).float().mean().item())
    raise ValueError(f"Expected labels with rank 1 or 2, got rank {y.ndim}.")


def _compute_eval_accuracy(model: SNNBaselineSeq, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return _sequence_accuracy_from_logits(logits, y)


def _token_seq_stats_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if logits.ndim != 3 or y.ndim != 2:
        raise ValueError(f"Expected logits rank-3 and y rank-2, got {tuple(logits.shape)} and {tuple(y.shape)}.")
    if tuple(logits.shape[:2]) != tuple(y.shape):
        raise ValueError(f"Token/seq stat shape mismatch: logits={tuple(logits.shape)}, y={tuple(y.shape)}.")
    if logits.shape[2] == 1:
        preds = (logits[:, :, 0] >= 0.0).long()
    else:
        preds = torch.argmax(logits, dim=2)
    match = preds == y.long()
    token_acc = float(match.float().mean().item())
    seq_acc = float(match.all(dim=1).float().mean().item())
    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "token_loss": float(1.0 - token_acc),
        "seq_loss": float(1.0 - seq_acc),
    }


def _compute_eval_token_seq_stats(model: SNNBaselineSeq, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return _token_seq_stats_from_logits(logits, y)


def ste_solve(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_cfg: SteModelConfig,
    solve_cfg: SteSolveConfig,
    pretrained_weights: Optional[List[np.ndarray]] = None,
    device: Optional[torch.device] = None,
) -> SteSolveResult:
    run_device = choose_device() if device is None else device
    model = SNNBaselineSeq(model_cfg).to(run_device)
    if pretrained_weights is not None:
        expected = len(model.fcs) + 1
        if len(pretrained_weights) != expected:
            raise ValueError(f"Expected {expected} tensors in pretrained_weights, got {len(pretrained_weights)}.")
        with torch.no_grad():
            for i, fc in enumerate(model.fcs):
                src = torch.from_numpy(pretrained_weights[i]).to(fc.weight.device, dtype=fc.weight.dtype)
                if tuple(src.shape) != tuple(fc.weight.shape):
                    raise ValueError(f"Layer-{i} weight shape mismatch: {tuple(src.shape)} != {tuple(fc.weight.shape)}")
                fc.weight.copy_(src)
            src_cls = torch.from_numpy(pretrained_weights[-1]).to(model.classifier.weight.device, dtype=model.classifier.weight.dtype)
            if tuple(src_cls.shape) != tuple(model.classifier.weight.shape):
                raise ValueError("Classifier weight shape mismatch in pretrained initialization.")
            model.classifier.weight.copy_(src_cls)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=run_device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=run_device)
    test_x = torch.tensor(x_test, dtype=torch.float32, device=run_device)
    train_y = torch.tensor(y_train, dtype=torch.long, device=run_device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=run_device)
    test_y = torch.tensor(y_test, dtype=torch.long, device=run_device)

    if solve_cfg.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=solve_cfg.lr, weight_decay=solve_cfg.weight_decay)
    elif solve_cfg.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=solve_cfg.lr, weight_decay=solve_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_name={solve_cfg.optimizer_name}.")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    n_train = int(train_x.shape[0])
    if solve_cfg.batch_size is not None and int(solve_cfg.batch_size) != n_train:
        raise ValueError(
            f"STE enforces full-batch training: expected batch_size={n_train}, got {int(solve_cfg.batch_size)}."
        )
    batch_size = n_train
    print(
        (
            f"[ste-run] beta={float(solve_cfg.beta_path_reg):.6g} lr={float(solve_cfg.lr):.6g} "
            f"weight_decay={float(solve_cfg.weight_decay):.6g} "
            f"batch_size=full({batch_size}) n_train={n_train}"
        ),
        flush=True,
    )

    loss_history: List[float] = []
    best_val = float("inf")
    best_state = None
    n_samples = n_train
    for epoch in range(1, solve_cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples, device=run_device)
        epoch_loss_accum = 0.0
        for start in range(0, n_samples, batch_size):
            idx = permutation[start : start + batch_size]
            logits = model(train_x[idx])
            loss = LossFunction.compute(name=solve_cfg.loss_name, y=train_y[idx], f_x=logits).value
            if solve_cfg.beta_path_reg > 0.0:
                loss = loss + float(solve_cfg.beta_path_reg) * snn_path_reg(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_accum += float(loss.item()) * int(idx.numel())
        epoch_loss = epoch_loss_accum / float(n_samples)
        val_loss = _compute_eval_loss(
            model,
            val_x,
            val_y,
            solve_cfg.loss_name,
            beta_path_reg=solve_cfg.beta_path_reg,
        )
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if solve_cfg.log_every > 0 and epoch % solve_cfg.log_every == 0:
            train_acc = _compute_eval_accuracy(model, train_x, train_y)
            val_acc = _compute_eval_accuracy(model, val_x, val_y)
            loss_history.append(epoch_loss)
            print(
                (
                    f"[ste] epoch={epoch}/{solve_cfg.epochs} "
                    f"train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                ),
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce any best_state.")
    model.load_state_dict(best_state)

    final_train_objective = _compute_eval_loss(
        model, train_x, train_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
    )
    best_losses = {
        "train_loss": final_train_objective,
        "train_objective": final_train_objective,
        "val_loss": _compute_eval_loss(
            model, val_x, val_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
        ),
        "test_loss": _compute_eval_loss(
            model, test_x, test_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
        ),
    }
    if train_y.ndim == 2 and val_y.ndim == 2 and test_y.ndim == 2:
        tr_stats = _compute_eval_token_seq_stats(model, train_x, train_y)
        va_stats = _compute_eval_token_seq_stats(model, val_x, val_y)
        te_stats = _compute_eval_token_seq_stats(model, test_x, test_y)
        best_losses.update(
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
                "[ste-arithmetic] "
                f"token_acc train={tr_stats['token_acc']:.4f} val={va_stats['token_acc']:.4f} test={te_stats['token_acc']:.4f} "
                f"seq_acc train={tr_stats['seq_acc']:.4f} val={va_stats['seq_acc']:.4f} test={te_stats['seq_acc']:.4f} "
                f"token_loss train={tr_stats['token_loss']:.4f} val={va_stats['token_loss']:.4f} test={te_stats['token_loss']:.4f} "
                f"seq_loss train={tr_stats['seq_loss']:.4f} val={va_stats['seq_loss']:.4f} test={te_stats['seq_loss']:.4f}"
            ),
            flush=True,
        )
    return SteSolveResult(
        model=model,
        loss_history=loss_history,
        best_losses=best_losses,
        final_train_objective=final_train_objective,
    )
