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
    batch_size = train_x.shape[0] if solve_cfg.batch_size is None else solve_cfg.batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    loss_history: List[float] = []
    best_val = float("inf")
    best_state = None
    n_samples = train_x.shape[0]
    for epoch in range(1, solve_cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples, device=run_device)
        epoch_loss_accum = 0.0
        for start in range(0, n_samples, batch_size):
            idx = permutation[start : start + batch_size]
            logits = model(train_x[idx])
            loss = LossFunction.compute(name=solve_cfg.loss_name, y=train_y[idx], f_x=logits).value
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
        if epoch % solve_cfg.log_every == 0:
            loss_history.append(epoch_loss)

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
    return SteSolveResult(
        model=model,
        loss_history=loss_history,
        best_losses=best_losses,
        final_train_objective=final_train_objective,
    )
