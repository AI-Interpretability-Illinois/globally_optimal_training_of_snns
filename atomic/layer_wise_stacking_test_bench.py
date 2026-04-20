from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from solvers import ste_parallel_Solve as ste_par
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from .solvers import ste_parallel_Solve as ste_par
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


@dataclass
class LayerWiseConfig:
    num_blocks: int = 2
    L: int = 5
    P_rec: int = 128
    P_last: int = 64
    K_parallel: int = 1
    ste_pretrain_epochs: int = 80
    ste_finetune_epochs: int = 100
    loss_name: str = "hinge_ovr"
    optimizer_name: str = "adam"
    cvx_method: str = "cvx"  # cvx | sgd
    cvx_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    cvx_lr_grid: Sequence[float] = LR_GRID_DEFAULT
    cvx_bias_grid: Sequence[float] = BIAS_GRID_DEFAULT
    ste_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    ste_lr_grid: Sequence[float] = LR_GRID_DEFAULT


def _extract_weight_list(model: nn.Module) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    if hasattr(model, "fcs") and hasattr(model, "classifier"):
        for fc in model.fcs:
            weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    if hasattr(model, "branches") and hasattr(model, "classifier"):
        for br in model.branches:
            for fc in br.fcs:
                weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    raise TypeError(f"Unsupported SNN layout for weight export: {type(model)}")


def _extract_block_hidden_sequence(model: nn.Module, x_seq: np.ndarray) -> np.ndarray:
    """Run a trained block as frozen feature extractor (classifier dropped)."""
    if x_seq.ndim != 3:
        raise ValueError(f"Expected rank-3 sequence input, got shape={x_seq.shape}.")
    device = next(model.parameters()).device
    x = torch.tensor(x_seq, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        batch, steps, _ = x.shape
        if hasattr(model, "branches"):
            branch_mems = [[lif.init_leaky().to(device) for lif in b.lifs] for b in model.branches]
            hidden_seq: List[torch.Tensor] = []
            for t in range(steps):
                x_t = x[:, t, :]
                readouts: List[torch.Tensor] = []
                for bi, br in enumerate(model.branches):
                    mems = branch_mems[bi]
                    h = x_t
                    last_spk = last_mem = None
                    for i, (fc, lif) in enumerate(zip(br.fcs, br.lifs)):
                        spk, mem = lif(fc(h), mems[i])
                        mems[i] = mem
                        h = spk
                        last_spk, last_mem = spk, mem
                    if last_spk is None or last_mem is None:
                        raise RuntimeError("Empty branch while extracting block features.")
                    readouts.append(last_mem if model.last_layer_readout == "membrane" else last_spk)
                hidden_seq.append(torch.cat(readouts, dim=1))
            out = torch.stack(hidden_seq, dim=1)
            return out.detach().cpu().numpy().astype(np.float32, copy=False)
        mems = [lif.init_leaky().to(device) for lif in model.lifs]
        hidden_seq = []
        for t in range(steps):
            h = x[:, t, :]
            last_spk = None
            last_mem = None
            for i, (fc, lif) in enumerate(zip(model.fcs, model.lifs)):
                spk, mem = lif(fc(h), mems[i])
                mems[i] = mem
                h = spk
                last_spk = spk
                last_mem = mem
            if last_spk is None or last_mem is None:
                raise RuntimeError("Empty hidden stack while extracting block features.")
            readout = last_mem if model.last_layer_readout == "membrane" else last_spk
            hidden_seq.append(readout)
        out = torch.stack(hidden_seq, dim=1)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)


def run_layer_wise_stacking_test_bench(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: LayerWiseConfig,
) -> List[Dict[str, object]]:
    if int(cfg.K_parallel) > 1 and int(cfg.P_last) % int(cfg.K_parallel) != 0:
        raise ValueError(f"P_last={cfg.P_last} must be divisible by K_parallel={cfg.K_parallel}.")
    if int(cfg.L) >= 3 and int(cfg.K_parallel) > 1 and int(cfg.P_rec) % int(cfg.K_parallel) != 0:
        raise ValueError(f"P_rec={cfg.P_rec} must be divisible by K_parallel={cfg.K_parallel} when L>=3.")

    block_rows: List[Dict[str, object]] = []
    current_x_train = x_train.astype(np.float32, copy=False)
    current_x_val = x_val.astype(np.float32, copy=False)
    current_x_test = x_test.astype(np.float32, copy=False)

    for block_idx in range(cfg.num_blocks):
        best_ste_pre = None
        best_ste_pre_val = float("inf")
        best_ste_pre_lr = None
        best_ste_pre_beta = None
        for ste_lr in cfg.ste_lr_grid:
            for ste_beta in cfg.ste_beta_grid:
                out = ste_solve(
                    x_train=current_x_train,
                    y_train=y_train,
                    x_val=current_x_val,
                    y_val=y_val,
                    x_test=current_x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=current_x_train.shape[2],
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                        K_parallel=cfg.K_parallel,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name=cfg.loss_name,
                        optimizer_name=cfg.optimizer_name,
                        lr=float(ste_lr),
                        epochs=cfg.ste_pretrain_epochs,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                )
                val_loss = out.best_losses["val_loss"] + float(ste_beta)
                if val_loss < best_ste_pre_val:
                    best_ste_pre_val = val_loss
                    best_ste_pre = out
                    best_ste_pre_lr = float(ste_lr)
                    best_ste_pre_beta = float(ste_beta)
        if best_ste_pre is None or best_ste_pre_lr is None or best_ste_pre_beta is None:
            raise RuntimeError("Layer-wise STE pretrain sweep failed.")
        if not isinstance(best_ste_pre.model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
            raise TypeError("Expected SNNBaselineSeq model from ste_solve.")
        transferred_weights = _extract_weight_list(best_ste_pre.model)

        best_cvx = None
        best_cvx_val = float("inf")
        best_cvx_beta = None
        best_cvx_lr = None
        best_cvx_bias = None
        cvx_lr_eff = cvx_lr_sweep_values(cfg.cvx_method, cfg.cvx_lr_grid)
        for cvx_beta in cfg.cvx_beta_grid:
            for cvx_lr in cvx_lr_eff:
                for cvx_bias in cfg.cvx_bias_grid:
                    out = cvx_solve(
                        x_train=current_x_train,
                        y_train=y_train,
                        x_val=current_x_val,
                        y_val=y_val,
                        x_test=current_x_test,
                        y_test=y_test,
                        init_cfg=InitializationConfig(
                            mode="pretraining",
                            pretrained_weights=transferred_weights,
                            L=cfg.L,
                            P_rec=cfg.P_rec,
                            P_last=cfg.P_last,
                            K_parallel=cfg.K_parallel,
                            feature_count=cfg.P_last,
                            bias=float(cvx_bias),
                        ),
                        solve_cfg=SolveConfig(
                            method=cfg.cvx_method,
                            loss_name=cfg.loss_name,
                            beta=float(cvx_beta),
                            lr=float(cvx_lr),
                            optimizer_name=cfg.optimizer_name,
                        ),
                    )
                    val_obj = out.final_losses.get("val_objective", out.final_losses["val_loss"])
                    if float(val_obj) < best_cvx_val:
                        best_cvx_val = float(val_obj)
                        best_cvx = out
                        best_cvx_beta = float(cvx_beta)
                        best_cvx_lr = None if cfg.cvx_method == "cvx" else float(cvx_lr)
                        best_cvx_bias = float(cvx_bias)
        if best_cvx is None or best_cvx_beta is None or best_cvx_bias is None:
            raise RuntimeError("Layer-wise CVX sweep failed.")
        if cfg.cvx_method == "sgd" and best_cvx_lr is None:
            raise RuntimeError("Layer-wise CVX-SGD sweep failed to record lr.")

        best_ste_ft = None
        best_ste_ft_val = float("inf")
        best_ste_ft_lr = None
        best_ste_ft_beta = None
        for ste_lr in cfg.ste_lr_grid:
            for ste_beta in cfg.ste_beta_grid:
                out = ste_solve(
                    x_train=current_x_train,
                    y_train=y_train,
                    x_val=current_x_val,
                    y_val=y_val,
                    x_test=current_x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=current_x_train.shape[2],
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                        K_parallel=cfg.K_parallel,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name=cfg.loss_name,
                        optimizer_name=cfg.optimizer_name,
                        lr=float(ste_lr),
                        epochs=cfg.ste_finetune_epochs,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                    pretrained_weights=transferred_weights,
                )
                val_loss = out.best_losses["val_loss"] + float(ste_beta)
                if val_loss < best_ste_ft_val:
                    best_ste_ft_val = val_loss
                    best_ste_ft = out
                    best_ste_ft_lr = float(ste_lr)
                    best_ste_ft_beta = float(ste_beta)
        if best_ste_ft is None or best_ste_ft_lr is None or best_ste_ft_beta is None:
            raise RuntimeError("Layer-wise STE finetune sweep failed.")
        if not isinstance(best_ste_ft.model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
            raise TypeError("Expected SNNBaselineSeq model from ste_solve.")

        block_rows.append(
            {
                "block_idx": block_idx,
                "ste_pre": best_ste_pre,
                "cvx": best_cvx,
                "ste_finetune": best_ste_ft,
                "cvx_init_source": "pretraining",
                "ste_pre_selected_params": {"lr": best_ste_pre_lr, "beta": best_ste_pre_beta},
                "cvx_selected_params": {
                    "lr": best_cvx_lr,
                    "beta": best_cvx_beta,
                    "bias": best_cvx_bias,
                },
                "ste_finetune_selected_params": {"lr": best_ste_ft_lr, "beta": best_ste_ft_beta},
            }
        )

        # Stacking step: drop classifier, freeze trained block, and feed its hidden sequence to next block.
        current_x_train = _extract_block_hidden_sequence(best_ste_ft.model, current_x_train)
        current_x_val = _extract_block_hidden_sequence(best_ste_ft.model, current_x_val)
        current_x_test = _extract_block_hidden_sequence(best_ste_ft.model, current_x_test)

    return block_rows
