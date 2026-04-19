from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import numpy as np

from .data_loaders.arithmetic_data_loader import load_arithmetic_dataset
from .data_loaders.dfa_data_loader import make_dfa_dataset
from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


@dataclass
class DeterministicBenchConfig:
    benchmark: Literal["arithmetic", "dfa"] = "arithmetic"
    init_mode: Literal["gaussian", "fine_tune", "both"] = "gaussian"
    pipeline_mode: Literal["fine_tune", "layer_wise_stacking", "both"] = "fine_tune"
    seed: int = 0
    T: int = 5
    n_train: int = 1024
    n_val: int = 256
    n_test: int = 256
    op: str = "add"
    base: int = 2
    n_digits: int = 4
    dfa_spec: str = "tomita_3"
    P_rec: int = 128
    P_last: int = 64
    L: int = 3
    layer_wise_num_blocks: int = 2
    optimizer_name: str = "adam"
    cvx_method: str = "cvx"  # cvx | sgd
    cvx_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    cvx_lr_grid: Sequence[float] = LR_GRID_DEFAULT
    cvx_bias_grid: Sequence[float] = BIAS_GRID_DEFAULT
    ste_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    ste_lr_grid: Sequence[float] = LR_GRID_DEFAULT


TIMESTEP_SWEEP: tuple[int, ...] = (2, 4, 7, 14, 28)
LAYER_SWEEP: tuple[int, ...] = (3, 5, 10, 15, 20)
TRAIN_SWEEP_SMALL_T: tuple[int, ...] = (512, 2304, 4096, 5888, 7680)
TRAIN_SWEEP_OTHER_T: tuple[int, ...] = (7680, 34560, 61440, 88320, 115200)
WIDTH_SCALE_SWEEP: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
SPECIAL_SMALL_T: tuple[int, ...] = (5, 8)


def _extract_weight_list(model: SNNBaselineSeq) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    for fc in model.fcs:
        weights.append(fc.weight.detach().cpu().numpy().copy())
    weights.append(model.classifier.weight.detach().cpu().numpy().copy())
    return weights


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


def _extract_det_result_rows(base: Dict[str, Any], out: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fine_tune_path = out.get("fine_tune_path")
    if fine_tune_path is not None:
        ste = fine_tune_path["ste"]
        rows.append(
            {
                **base,
                "pipeline": "fine_tune",
                "stage": "ste",
                "source": "",
                "block_idx": "",
                "selected_lr": fine_tune_path["ste_selected_params"]["lr"],
                "selected_beta": fine_tune_path["ste_selected_params"]["beta"],
                "train_loss": ste.best_losses.get("train_loss"),
                "val_loss": ste.best_losses.get("val_loss"),
                "test_loss": ste.best_losses.get("test_loss"),
                "train_objective": ste.best_losses.get("train_objective", ste.final_train_objective),
                "val_objective": ste.best_losses.get("val_loss"),
                "test_objective": ste.best_losses.get("test_loss"),
                "primal_value": "",
                "dual_value": "",
                "duality_gap": "",
            }
        )
        for source, cvx_res in fine_tune_path["cvx_by_init"].items():
            chosen = fine_tune_path["cvx_selected_params"][source]
            rows.append(
                {
                    **base,
                    "pipeline": "fine_tune",
                    "stage": "cvx",
                    "source": source,
                    "block_idx": "",
                    "selected_lr": chosen["lr"],
                    "selected_beta": chosen["beta"],
                    "selected_bias": chosen.get("bias", ""),
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
    layer_wise_path = out.get("layer_wise_stacking_path")
    if layer_wise_path is not None:
        for block in layer_wise_path:
            ste_pre = block["ste_pre"]
            cvx_out = block["cvx"]
            ste_ft = block["ste_finetune"]
            bidx = block["block_idx"]
            rows.append(
                {
                    **base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "ste_pretrain",
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
                    **base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "cvx",
                    "source": "gaussian",
                    "block_idx": bidx,
                    "selected_lr": block["cvx_selected_params"]["lr"],
                    "selected_beta": block["cvx_selected_params"]["beta"],
                    "selected_bias": block["cvx_selected_params"].get("bias", ""),
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
                    **base,
                    "pipeline": "layer_wise_stacking",
                    "stage": "ste_finetune",
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


def _make_base_task_cfg(
    task_name: str,
    seed: int,
    pipeline_mode: str,
    init_mode: str,
) -> DeterministicBenchConfig:
    if task_name == "addition":
        return DeterministicBenchConfig(
            benchmark="arithmetic",
            op="add",
            base=2,
            n_digits=4,
            seed=seed,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            init_mode=init_mode,  # type: ignore[arg-type]
        )
    if task_name == "xor":
        return DeterministicBenchConfig(
            benchmark="dfa",
            dfa_spec="two_step_xor",
            seed=seed,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            init_mode=init_mode,  # type: ignore[arg-type]
        )
    if task_name == "parity":
        return DeterministicBenchConfig(
            benchmark="dfa",
            dfa_spec="parity_2",
            seed=seed,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            init_mode=init_mode,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported task_name={task_name}. Expected addition|xor|parity.")


def _train_sweep_for_t(T: int) -> tuple[int, ...]:
    if T in SPECIAL_SMALL_T:
        return TRAIN_SWEEP_SMALL_T
    return TRAIN_SWEEP_OTHER_T


def _build_sweep_plan_for_task(
    task_name: str,
    seed: int,
    pipeline_mode: str,
    init_mode: str,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    base_cfg = _make_base_task_cfg(
        task_name=task_name,
        seed=seed,
        pipeline_mode=pipeline_mode,
        init_mode=init_mode,
    )

    # Test 1: timestep sweep
    for T in TIMESTEP_SWEEP:
        cfg = DeterministicBenchConfig(**asdict(base_cfg))
        cfg.T = int(T)
        plan.append({"test_bench": "test1_timestep", "task_name": task_name, "cfg": cfg})

    # Test 2: layer sweep
    for L in LAYER_SWEEP:
        cfg = DeterministicBenchConfig(**asdict(base_cfg))
        cfg.L = int(L)
        plan.append({"test_bench": "test2_layers", "task_name": task_name, "cfg": cfg})

    # Test 3: train-size sweep
    for T in (*SPECIAL_SMALL_T, *TIMESTEP_SWEEP):
        for n_train in _train_sweep_for_t(T):
            cfg = DeterministicBenchConfig(**asdict(base_cfg))
            cfg.T = int(T)
            cfg.n_train = int(n_train)
            cfg.n_val = max(1, int(round(0.2 * n_train)))
            cfg.n_test = 5000
            plan.append({"test_bench": "test3_train_size", "task_name": task_name, "cfg": cfg})

    # Test 4: width sweep based on train-size choices
    for T in (*SPECIAL_SMALL_T, *TIMESTEP_SWEEP):
        for n_train in _train_sweep_for_t(T):
            for width_scale in WIDTH_SCALE_SWEEP:
                cfg = DeterministicBenchConfig(**asdict(base_cfg))
                cfg.T = int(T)
                cfg.n_train = int(n_train)
                cfg.n_val = max(1, int(round(0.2 * n_train)))
                cfg.n_test = 5000
                cfg.P_last = max(1, int(round(n_train * float(width_scale))))
                cfg.P_rec = max(1, cfg.P_last // 2)
                plan.append(
                    {
                        "test_bench": "test4_width",
                        "task_name": task_name,
                        "width_scale": float(width_scale),
                        "cfg": cfg,
                    }
                )
    return plan


def run_deterministic_bench(cfg: DeterministicBenchConfig) -> dict:
    if cfg.benchmark == "arithmetic":
        ds = load_arithmetic_dataset(
            op=cfg.op,
            base=cfg.base,
            n_digits=cfg.n_digits,
            n_train=cfg.n_train,
            n_val=cfg.n_val,
            n_test=cfg.n_test,
            seed=cfg.seed,
        )
        x_train, y_train = ds.X_train, ds.y_train
        x_val, y_val = ds.X_val, ds.y_val
        x_test, y_test = ds.X_test, ds.y_test
        num_classes = ds.num_classes
    elif cfg.benchmark == "dfa":
        x_all, y_all, num_classes = make_dfa_dataset(
            dfa_spec=cfg.dfa_spec,
            n=cfg.n_train + cfg.n_val + cfg.n_test,
            T=cfg.T,
            seed=cfg.seed,
            balanced=True,
        )
        x_train = x_all[: cfg.n_train]
        y_train = y_all[: cfg.n_train]
        x_val = x_all[cfg.n_train : cfg.n_train + cfg.n_val]
        y_val = y_all[cfg.n_train : cfg.n_train + cfg.n_val]
        x_test = x_all[cfg.n_train + cfg.n_val :]
        y_test = y_all[cfg.n_train + cfg.n_val :]
    else:
        raise ValueError(f"Unknown deterministic benchmark={cfg.benchmark}.")

    fine_tune_path: Dict[str, object] | None = None
    if cfg.pipeline_mode in ("fine_tune", "both"):
        best_ste = None
        best_ste_val = float("inf")
        best_ste_lr = None
        best_ste_beta = None
        for ste_lr in cfg.ste_lr_grid:
            for ste_beta in cfg.ste_beta_grid:
                out = ste_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=x_train.shape[2],
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name="hinge_ovr",
                        optimizer_name=cfg.optimizer_name,
                        lr=float(ste_lr),
                        epochs=100,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                )
                val_loss = out.best_losses["val_loss"] + float(ste_beta)
                if val_loss < best_ste_val:
                    best_ste_val = val_loss
                    best_ste = out
                    best_ste_lr = float(ste_lr)
                    best_ste_beta = float(ste_beta)
        if best_ste is None or best_ste_lr is None or best_ste_beta is None:
            raise RuntimeError("Deterministic STE sweep failed to produce a candidate.")
        ste_out = best_ste
        if not isinstance(ste_out.model, SNNBaselineSeq):
            raise TypeError("Expected SNNBaselineSeq model from ste_solve.")
        transferred_weights = _extract_weight_list(ste_out.model)

        cvx_by_init: Dict[str, object] = {}
        cvx_selected_params: Dict[str, Dict[str, float | None]] = {}
        if cfg.init_mode in ("gaussian", "both"):
            best_cvx = None
            best_cvx_val = float("inf")
            best_beta = None
            best_lr: float | None = None
            best_bias = None
            cvx_lr_eff = cvx_lr_sweep_values(cfg.cvx_method, cfg.cvx_lr_grid)
            for cvx_beta in cfg.cvx_beta_grid:
                for cvx_lr in cvx_lr_eff:
                    for cvx_bias in cfg.cvx_bias_grid:
                        out = cvx_solve(
                            x_train=x_train,
                            y_train=y_train,
                            x_val=x_val,
                            y_val=y_val,
                            x_test=x_test,
                            y_test=y_test,
                            init_cfg=InitializationConfig(
                                mode="gaussian",
                                L=cfg.L,
                                P_rec=cfg.P_rec,
                                P_last=cfg.P_last,
                                feature_count=cfg.P_last,
                                bias=float(cvx_bias),
                            ),
                            solve_cfg=SolveConfig(
                                method=cfg.cvx_method,
                                loss_name="hinge_ovr",
                                beta=float(cvx_beta),
                                lr=float(cvx_lr),
                                optimizer_name=cfg.optimizer_name,
                            ),
                        )
                        val_obj = out.final_losses.get("val_objective", out.final_losses["val_loss"])
                        if float(val_obj) < best_cvx_val:
                            best_cvx_val = float(val_obj)
                            best_cvx = out
                            best_beta = float(cvx_beta)
                            best_lr = float(cvx_lr) if cfg.cvx_method == "sgd" else None
                            best_bias = float(cvx_bias)
            if best_cvx is None or best_beta is None or best_bias is None:
                raise RuntimeError("Deterministic CVX (gaussian) sweep failed.")
            if cfg.cvx_method == "sgd" and best_lr is None:
                raise RuntimeError("Deterministic CVX (gaussian) sweep missing selected lr for sgd.")
            cvx_by_init["gaussian"] = best_cvx
            cvx_selected_params["gaussian"] = {"beta": best_beta, "lr": best_lr, "bias": best_bias}
        if cfg.init_mode in ("fine_tune", "both"):
            best_cvx = None
            best_cvx_val = float("inf")
            best_beta = None
            best_lr: float | None = None
            best_bias = None
            cvx_lr_eff = cvx_lr_sweep_values(cfg.cvx_method, cfg.cvx_lr_grid)
            for cvx_beta in cfg.cvx_beta_grid:
                for cvx_lr in cvx_lr_eff:
                    for cvx_bias in cfg.cvx_bias_grid:
                        out = cvx_solve(
                            x_train=x_train,
                            y_train=y_train,
                            x_val=x_val,
                            y_val=y_val,
                            x_test=x_test,
                            y_test=y_test,
                            init_cfg=InitializationConfig(
                                mode="pretraining",
                                pretrained_weights=transferred_weights,
                                L=cfg.L,
                                P_rec=cfg.P_rec,
                                P_last=cfg.P_last,
                                feature_count=cfg.P_last,
                                bias=float(cvx_bias),
                            ),
                            solve_cfg=SolveConfig(
                                method=cfg.cvx_method,
                                loss_name="hinge_ovr",
                                beta=float(cvx_beta),
                                lr=float(cvx_lr),
                                optimizer_name=cfg.optimizer_name,
                            ),
                        )
                        val_obj = out.final_losses.get("val_objective", out.final_losses["val_loss"])
                        if float(val_obj) < best_cvx_val:
                            best_cvx_val = float(val_obj)
                            best_cvx = out
                            best_beta = float(cvx_beta)
                            best_lr = float(cvx_lr) if cfg.cvx_method == "sgd" else None
                            best_bias = float(cvx_bias)
            if best_cvx is None or best_beta is None or best_bias is None:
                raise RuntimeError("Deterministic CVX (fine_tune) sweep failed.")
            if cfg.cvx_method == "sgd" and best_lr is None:
                raise RuntimeError("Deterministic CVX (fine_tune) sweep missing selected lr for sgd.")
            cvx_by_init["fine_tune"] = best_cvx
            cvx_selected_params["fine_tune"] = {"beta": best_beta, "lr": best_lr, "bias": best_bias}

        if cfg.init_mode == "gaussian":
            cvx_out = cvx_by_init["gaussian"]
        elif cfg.init_mode == "fine_tune":
            cvx_out = cvx_by_init["fine_tune"]
        else:  # both
            cvx_out = cvx_by_init
        fine_tune_path = {
            "ste": ste_out,
            "cvx": cvx_out,
            "cvx_by_init": cvx_by_init,
            "ste_selected_params": {"lr": best_ste_lr, "beta": best_ste_beta},
            "cvx_selected_params": cvx_selected_params,
        }

    layer_wise_path: List[Dict[str, object]] | None = None
    if cfg.pipeline_mode in ("layer_wise_stacking", "both"):
        layer_wise_path = run_layer_wise_stacking_test_bench(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            cfg=LayerWiseConfig(
                num_blocks=cfg.layer_wise_num_blocks,
                L=cfg.L,
                P_rec=cfg.P_rec,
                P_last=cfg.P_last,
                loss_name="hinge_ovr",
                optimizer_name=cfg.optimizer_name,
                cvx_method=cfg.cvx_method,
                cvx_beta_grid=cfg.cvx_beta_grid,
                cvx_lr_grid=cfg.cvx_lr_grid,
                cvx_bias_grid=cfg.cvx_bias_grid,
                ste_beta_grid=cfg.ste_beta_grid,
                ste_lr_grid=cfg.ste_lr_grid,
            ),
        )

    return {
        # Backward-compatible aliases use fine_tune path when available.
        "ste": fine_tune_path["ste"] if fine_tune_path is not None else None,
        "cvx": fine_tune_path["cvx"] if fine_tune_path is not None else None,
        "cvx_by_init": fine_tune_path["cvx_by_init"] if fine_tune_path is not None else {},
        "fine_tune_path": fine_tune_path,
        "layer_wise_stacking_path": layer_wise_path,
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic benchmark suites and append results to CSV.")
    parser.add_argument("--csv_path", type=str, default="deterministic_data_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        choices=("fine_tune", "layer_wise_stacking", "both"),
        default="both",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        choices=("gaussian", "fine_tune", "both"),
        default="both",
        help="CVX initialization mode used in fine_tune pipeline path.",
    )
    parser.add_argument(
        "--cvx_method",
        type=str,
        choices=("cvx", "sgd"),
        default="cvx",
        help="Method used by cvx_solve (default: cvx).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=("addition", "xor", "parity"),
        help="Subset of tasks to run: addition xor parity",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    for task_name in args.tasks:
        plan = _build_sweep_plan_for_task(
            task_name=task_name,
            seed=args.seed,
            pipeline_mode=args.pipeline_mode,
            init_mode=args.init_mode,
        )
        for item in plan:
            cfg = item["cfg"]
            cfg.cvx_method = args.cvx_method
            out = run_deterministic_bench(cfg)
            base = {
                "task_name": item["task_name"],
                "test_bench": item["test_bench"],
                "benchmark": cfg.benchmark,
                "seed": cfg.seed,
                "pipeline_mode": cfg.pipeline_mode,
                "init_mode": cfg.init_mode,
                "loss_type": "hinge_ovr",
                "optimizer_name": cfg.optimizer_name,
                "cvx_method": cfg.cvx_method,
                "T": cfg.T,
                "L": cfg.L,
                "n_train": cfg.n_train,
                "n_val": cfg.n_val,
                "n_test": cfg.n_test,
                "P_rec": cfg.P_rec,
                "P_last": cfg.P_last,
                "op": cfg.op,
                "base": cfg.base,
                "n_digits": cfg.n_digits,
                "dfa_spec": cfg.dfa_spec,
                "width_scale": item.get("width_scale", ""),
            }
            _write_rows_append(csv_path, _extract_det_result_rows(base=base, out=out))


if __name__ == "__main__":
    main()
