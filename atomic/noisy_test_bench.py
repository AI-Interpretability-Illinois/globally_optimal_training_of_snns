from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence

import numpy as np

from .fine_tune import FineTuneConfig, run_fine_tune_pipeline
from .data_loaders.image_data_loader import load_cifar_seq_dataset, load_mnist_seq_dataset
from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT


@dataclass
class NoisySweepConfig:
    dataset_list: tuple[str, ...] = ("mnist_seq",)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    L_list: tuple[int, ...] = (3, 5, 10, 15, 20)
    T_list: tuple[int, ...] = (2, 4, 7, 14, 28)
    n_train_list: tuple[int, ...] = (1024, 8192, 12000, 30000)
    # Width sweep requested: n_train * {1/4, 1/2, 1, 2, 4}
    width_scale_list: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    val_frac: float = 0.2
    n_test: int = 5000
    loss_type: str = "ce"
    pipeline_mode: Literal["fine_tune", "layer_wise_stacking", "both"] = "both"
    layer_wise_num_blocks: int = 2
    optimizer_name: str = "adam"
    cvx_method: str = "cvx"  # cvx | sgd
    cvx_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    cvx_lr_grid: Sequence[float] = LR_GRID_DEFAULT
    ste_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    ste_lr_grid: Sequence[float] = LR_GRID_DEFAULT


def _make_widths_from_train_size(n_train: int, scales: Sequence[float]) -> List[int]:
    widths = sorted({max(1, int(round(n_train * s))) for s in scales})
    if not widths:
        raise ValueError("Width scale list generated an empty width set.")
    return widths


def _scalarize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return repr(list(value))
    return repr(value)


def _write_rows_append(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _scalarize(v) for k, v in row.items()})


def _extract_noisy_result_rows(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    result = base["result"]
    common_base = {k: v for k, v in base.items() if k != "result"}
    if "fine_tune" in result:
        ft = result["fine_tune"]
        selected_readout = ft["selected_readout"]
        selected = ft["by_readout"][selected_readout]
        ste_pre = selected["ste_pretrain"]
        ste_post = selected["ste_post"]
        rows.append(
            {
                **common_base,
                "pipeline": "fine_tune",
                "stage": "ste_pretrain",
                "readout": selected_readout,
                "source": "",
                "block_idx": "",
                "selected_lr": selected.get("ste_pre_selected_params", {}).get("lr", ""),
                "selected_beta": selected.get("ste_pre_selected_params", {}).get("beta", ""),
                "train_loss": ste_pre.best_losses.get("train_loss"),
                "val_loss": ste_pre.best_losses.get("val_loss"),
                "test_loss": ste_pre.best_losses.get("test_loss"),
                "train_objective": ste_pre.best_losses.get("train_objective", ste_pre.final_train_objective),
                "val_objective": ste_pre.best_losses.get("val_loss"),
                "test_objective": ste_pre.best_losses.get("test_loss"),
                "primal_value": "",
                "dual_value": "",
                "duality_gap": "",
            }
        )
        for source, cvx_res in selected["cvx_by_source"].items():
            chosen = selected["cvx_selected_params"][source]
            rows.append(
                {
                    **common_base,
                    "pipeline": "fine_tune",
                    "stage": "cvx",
                    "readout": selected_readout,
                    "source": source,
                    "block_idx": "",
                    "selected_lr": chosen["lr"],
                    "selected_beta": chosen["beta"],
                    "train_loss": cvx_res.final_losses.get("train_loss"),
                    "val_loss": cvx_res.final_losses.get("val_loss"),
                    "test_loss": cvx_res.final_losses.get("test_loss"),
                    "train_objective": cvx_res.final_losses.get("train_objective"),
                    "val_objective": cvx_res.final_losses.get("val_objective"),
                    "test_objective": cvx_res.final_losses.get("test_objective"),
                    "primal_value": cvx_res.diagnostics.primal_value,
                    "dual_value": cvx_res.diagnostics.dual_value,
                    "duality_gap": cvx_res.diagnostics.gap,
                }
            )
        rows.append(
            {
                **common_base,
                "pipeline": "fine_tune",
                "stage": "ste_post",
                "readout": selected_readout,
                "source": "",
                "block_idx": "",
                "selected_lr": selected["ste_post_selected_params"]["lr"],
                "selected_beta": selected["ste_post_selected_params"]["beta"],
                "train_loss": ste_post.best_losses.get("train_loss"),
                "val_loss": ste_post.best_losses.get("val_loss"),
                "test_loss": ste_post.best_losses.get("test_loss"),
                "train_objective": ste_post.best_losses.get("train_objective", ste_post.final_train_objective),
                "val_objective": ste_post.best_losses.get("val_loss"),
                "test_objective": ste_post.best_losses.get("test_loss"),
                "primal_value": "",
                "dual_value": "",
                "duality_gap": "",
            }
        )
    if "layer_wise_stacking" in result:
        for block in result["layer_wise_stacking"]:
            bidx = block["block_idx"]
            ste_pre = block["ste_pre"]
            cvx_out = block["cvx"]
            ste_ft = block["ste_finetune"]
            rows.append(
                {
                    **common_base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "ste_pretrain",
                    "readout": "",
                    "source": "",
                    "block_idx": bidx,
                    "selected_lr": block["ste_pre_selected_params"]["lr"],
                    "selected_beta": block["ste_pre_selected_params"]["beta"],
                    "train_loss": ste_pre.best_losses.get("train_loss"),
                    "val_loss": ste_pre.best_losses.get("val_loss"),
                    "test_loss": ste_pre.best_losses.get("test_loss"),
                    "train_objective": ste_pre.best_losses.get("train_objective", ste_pre.final_train_objective),
                    "val_objective": ste_pre.best_losses.get("val_loss"),
                    "test_objective": ste_pre.best_losses.get("test_loss"),
                    "primal_value": "",
                    "dual_value": "",
                    "duality_gap": "",
                }
            )
            rows.append(
                {
                    **common_base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "cvx",
                    "readout": "",
                    "source": "gaussian",
                    "block_idx": bidx,
                    "selected_lr": block["cvx_selected_params"]["lr"],
                    "selected_beta": block["cvx_selected_params"]["beta"],
                    "train_loss": cvx_out.final_losses.get("train_loss"),
                    "val_loss": cvx_out.final_losses.get("val_loss"),
                    "test_loss": cvx_out.final_losses.get("test_loss"),
                    "train_objective": cvx_out.final_losses.get("train_objective"),
                    "val_objective": cvx_out.final_losses.get("val_objective"),
                    "test_objective": cvx_out.final_losses.get("test_objective"),
                    "primal_value": cvx_out.diagnostics.primal_value,
                    "dual_value": cvx_out.diagnostics.dual_value,
                    "duality_gap": cvx_out.diagnostics.gap,
                }
            )
            rows.append(
                {
                    **common_base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "ste_finetune",
                    "readout": "",
                    "source": "",
                    "block_idx": bidx,
                    "selected_lr": block["ste_finetune_selected_params"]["lr"],
                    "selected_beta": block["ste_finetune_selected_params"]["beta"],
                    "train_loss": ste_ft.best_losses.get("train_loss"),
                    "val_loss": ste_ft.best_losses.get("val_loss"),
                    "test_loss": ste_ft.best_losses.get("test_loss"),
                    "train_objective": ste_ft.best_losses.get("train_objective", ste_ft.final_train_objective),
                    "val_objective": ste_ft.best_losses.get("val_loss"),
                    "test_objective": ste_ft.best_losses.get("test_loss"),
                    "primal_value": "",
                    "dual_value": "",
                    "duality_gap": "",
                }
            )
    return rows


def run_noisy_test_bench(
    sweepCfg: NoisySweepConfig,
    on_result: Callable[[Dict[str, Any]], None] | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not (0.0 < sweepCfg.val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {sweepCfg.val_frac}.")
    for dataset in sweepCfg.dataset_list:
        for L in sweepCfg.L_list:
            for T in sweepCfg.T_list:
                for n_train in sweepCfg.n_train_list:
                    n_val = max(1, int(round(sweepCfg.val_frac * n_train)))
                    width_list = _make_widths_from_train_size(n_train, sweepCfg.width_scale_list)
                    for P_last in width_list:
                        P_rec = max(1, P_last // 2)
                        for seed in sweepCfg.seeds:
                            if dataset in ("mnist_seq", "mnist_perm_seq"):
                                ds = load_mnist_seq_dataset(
                                    task=dataset,
                                    T=T,
                                    n_train=n_train,
                                    n_val=n_val,
                                    n_test=sweepCfg.n_test,
                                    seed=seed,
                                )
                            elif dataset == "cifar_seq":
                                ds = load_cifar_seq_dataset(
                                    task=dataset,
                                    T=T,
                                    n_train=n_train,
                                    n_val=n_val,
                                    n_test=sweepCfg.n_test,
                                )
                            else:
                                raise ValueError(f"Unsupported noisy dataset: {dataset}")
                            run: Dict[str, object] = {}
                            if sweepCfg.pipeline_mode in ("fine_tune", "both"):
                                run["fine_tune"] = run_fine_tune_pipeline(
                                    x_train=ds.X_train,
                                    y_train=ds.y_train,
                                    x_val=ds.X_val,
                                    y_val=ds.y_val,
                                    x_test=ds.X_test,
                                    y_test=ds.y_test,
                                    num_classes=ds.num_classes,
                                    seed=seed,
                                    cfg=FineTuneConfig(
                                        T=T,
                                        P_in=ds.d_in,
                                        P_rec=P_rec,
                                        P_last=P_last,
                                        L=L,
                                        loss_type=sweepCfg.loss_type,
                                        cvx_optimizer=sweepCfg.optimizer_name,
                                        cvx_method=sweepCfg.cvx_method,
                                        beta_grid=sweepCfg.cvx_beta_grid,
                                        lr_grid=sweepCfg.cvx_lr_grid,
                                        ste_beta_grid=sweepCfg.ste_beta_grid,
                                        ste_lr_grid=sweepCfg.ste_lr_grid,
                                    ),
                                )
                            if sweepCfg.pipeline_mode in ("layer_wise_stacking", "both"):
                                run["layer_wise_stacking"] = run_layer_wise_stacking_test_bench(
                                    x_train=ds.X_train,
                                    y_train=ds.y_train,
                                    x_val=ds.X_val,
                                    y_val=ds.y_val,
                                    x_test=ds.X_test,
                                    y_test=ds.y_test,
                                    num_classes=ds.num_classes,
                                    cfg=LayerWiseConfig(
                                        num_blocks=sweepCfg.layer_wise_num_blocks,
                                        L=L,
                                        P_rec=P_rec,
                                        P_last=P_last,
                                        loss_name=sweepCfg.loss_type,
                                        optimizer_name=sweepCfg.optimizer_name,
                                        cvx_method=sweepCfg.cvx_method,
                                        cvx_beta_grid=sweepCfg.cvx_beta_grid,
                                        cvx_lr_grid=sweepCfg.cvx_lr_grid,
                                        ste_beta_grid=sweepCfg.ste_beta_grid,
                                        ste_lr_grid=sweepCfg.ste_lr_grid,
                                    ),
                                )
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "pipeline_mode": sweepCfg.pipeline_mode,
                                    "loss_type": sweepCfg.loss_type,
                                    "optimizer_name": sweepCfg.optimizer_name,
                                    "cvx_method": sweepCfg.cvx_method,
                                    "L": L,
                                    "T": T,
                                    "n_train": n_train,
                                    "n_val": n_val,
                                    "n_test": sweepCfg.n_test,
                                    "P_rec": P_rec,
                                    "P_last": P_last,
                                    "cvx_beta_grid": tuple(float(x) for x in sweepCfg.cvx_beta_grid),
                                    "cvx_lr_grid": tuple(float(x) for x in sweepCfg.cvx_lr_grid),
                                    "ste_beta_grid": tuple(float(x) for x in sweepCfg.ste_beta_grid),
                                    "ste_lr_grid": tuple(float(x) for x in sweepCfg.ste_lr_grid),
                                    "result": run,
                                }
                            )
                            if on_result is not None:
                                on_result(rows[-1])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run noisy benchmark sweep and append per-run CSV rows.")
    parser.add_argument("--csv_path", type=str, default="noisy_data_results.csv")
    parser.add_argument("--dataset", type=str, default="mnist_seq")
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        choices=("fine_tune", "layer_wise_stacking", "both"),
        default="both",
    )
    parser.add_argument(
        "--cvx_method",
        type=str,
        choices=("cvx", "sgd"),
        default="cvx",
        help="Method used by cvx_solve (default: cvx).",
    )
    args = parser.parse_args()

    cfg = NoisySweepConfig(dataset_list=(args.dataset,), pipeline_mode=args.pipeline_mode, cvx_method=args.cvx_method)
    csv_path = Path(args.csv_path)

    def _on_result(row: Dict[str, Any]) -> None:
        flat_rows = _extract_noisy_result_rows(base=row)
        _write_rows_append(csv_path=csv_path, rows=flat_rows)

    run_noisy_test_bench(cfg, on_result=_on_result)


if __name__ == "__main__":
    main()
