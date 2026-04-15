from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import ArithmeticDataset, load_arithmetic_dataset
    from data_loaders.dfa_data_loader import make_dfa_dataset
    from data_loaders.image_data_loader import ImageSequenceDataset, load_cifar_seq_dataset, load_mnist_seq_dataset
    from data_loaders.uci_data_loader import UciDataset, load_uci_dataset
    from fine_tune import FineTuneConfig, run_fine_tune_pipeline
    from layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers.cvx_solve import (
        InitializationConfig,
        SolveConfig,
        _build_feature_map,
        _prepare_sequence_targets,
        cvx_solve,
    )
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .data_loaders.arithmetic_data_loader import ArithmeticDataset, load_arithmetic_dataset
    from .data_loaders.dfa_data_loader import make_dfa_dataset
    from .data_loaders.image_data_loader import ImageSequenceDataset, load_cifar_seq_dataset, load_mnist_seq_dataset
    from .data_loaders.uci_data_loader import UciDataset, load_uci_dataset
    from .fine_tune import FineTuneConfig, run_fine_tune_pipeline
    from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers.cvx_solve import (
        InitializationConfig,
        SolveConfig,
        _build_feature_map,
        _prepare_sequence_targets,
        cvx_solve,
    )
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_seq_from_uci(ds: UciDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    x_train = ds.X_train[:, None, :].astype(np.float32, copy=False)
    x_val = ds.X_val[:, None, :].astype(np.float32, copy=False)
    x_test = ds.X_test[:, None, :].astype(np.float32, copy=False)
    num_classes = int(np.max(ds.y_train) + 1)
    return x_train, ds.y_train, x_val, ds.y_val, x_test, ds.y_test, num_classes, int(x_train.shape[2])


def _load_dataset_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset in ("mnist_seq", "mnist_perm_seq"):
        ds: ImageSequenceDataset = load_mnist_seq_dataset(
            task=args.dataset,
            T=args.T,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": ds.d_in,
        }
    if args.dataset == "cifar_seq":
        ds = load_cifar_seq_dataset(
            task=args.dataset,
            T=args.T,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": ds.d_in,
        }
    if args.dataset == "arithmetic_seq":
        ds: ArithmeticDataset = load_arithmetic_dataset(
            op=args.arith_op,
            base=args.arith_base,
            n_digits=args.n_digits,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": int(ds.X_train.shape[2]),
        }
    if args.dataset == "dfa":
        n_total = args.n_train + args.n_val + args.n_test
        x_all, y_all, num_classes = make_dfa_dataset(
            dfa_spec=args.dfa_spec,
            n=n_total,
            T=args.T,
            seed=args.seed,
            balanced=True,
        )
        x_train = x_all[: args.n_train]
        y_train = y_all[: args.n_train]
        x_val = x_all[args.n_train : args.n_train + args.n_val]
        y_val = y_all[args.n_train : args.n_train + args.n_val]
        x_test = x_all[args.n_train + args.n_val :]
        y_test = y_all[args.n_train + args.n_val :]
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "num_classes": int(num_classes),
            "d_in": int(x_train.shape[2]),
        }
    if args.dataset == "uci":
        ds = load_uci_dataset(
            name=args.uci_name,
            seed=args.seed,
            test_size=args.uci_test_size,
            val_size=args.uci_val_size,
            standardize=not args.uci_no_standardize,
        )
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes, d_in = _as_seq_from_uci(ds)
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "num_classes": num_classes,
            "d_in": d_in,
        }
    raise ValueError(f"Unsupported dataset={args.dataset}.")


def _ste_last_step_acc(model: SNNBaselineSeq, x_test: np.ndarray, y_test: np.ndarray) -> float:
    device = next(model.parameters()).device
    x = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
        preds = logits[:, -1, :].argmax(dim=1).detach().cpu().numpy()
    if y_test.ndim == 2:
        y_last = y_test[:, -1]
    else:
        y_last = y_test
    return float(np.mean(preds == y_last))


def _cvx_last_step_acc(
    cvx_result: Any,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    init_cfg: InitializationConfig,
) -> float:
    _, _, d_test, _ = _build_feature_map(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        init_cfg=init_cfg,
    )
    y_last = _prepare_sequence_targets(y_test)
    if isinstance(cvx_result.trained_model, dict) and "weights" in cvx_result.trained_model:
        w = cvx_result.trained_model["weights"]
        scores = d_test @ w
        preds = scores.argmax(axis=1)
        return float(np.mean(preds == y_last))
    model = cvx_result.trained_model
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(d_test, dtype=torch.float32))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(preds == y_last))


def _run_simple_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    num_classes = data["num_classes"]
    d_in = data["d_in"]

    best_ste = None
    best_ste_score = float("inf")
    best_ste_params = None
    for ste_lr in LR_GRID_DEFAULT:
        for ste_beta in BETA_GRID_DEFAULT:
            _set_seed(args.seed)
            out = ste_solve(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                model_cfg=SteModelConfig(
                    d_in=d_in,
                    num_classes=num_classes,
                    L=args.L,
                    P_rec=args.P_rec,
                    P_last=args.P_last,
                    last_layer_readout=args.last_layer_readout,
                ),
                solve_cfg=SteSolveConfig(
                    loss_name=args.loss_type,
                    optimizer_name=args.optimizer_name,
                    lr=float(ste_lr),
                    epochs=args.ste_epochs,
                    batch_size=None if args.batch_size == -1 else int(args.batch_size),
                    weight_decay=0.0,
                    beta_path_reg=float(ste_beta),
                ),
            )
            score = float(out.best_losses["val_loss"]) + float(ste_beta)
            if score < best_ste_score:
                best_ste_score = score
                best_ste = out
                best_ste_params = {"lr": float(ste_lr), "beta": float(ste_beta)}
    if best_ste is None or best_ste_params is None:
        raise RuntimeError("Simple mode failed to find STE candidate.")
    if not isinstance(best_ste.model, SNNBaselineSeq):
        raise TypeError("Expected SNNBaselineSeq from ste_solve.")

    best_cvx = None
    best_cvx_score = float("inf")
    best_cvx_params = None
    best_init = None
    for cvx_beta in BETA_GRID_DEFAULT:
        for cvx_lr in LR_GRID_DEFAULT:
            for cvx_bias in BIAS_GRID_DEFAULT:
                init_cfg = InitializationConfig(
                    mode="gaussian",
                    seed=args.seed,
                    L=args.L,
                    P_rec=args.P_rec,
                    P_last=args.P_last,
                    feature_count=args.P_last,
                    last_layer_readout=args.last_layer_readout,
                    bias=float(cvx_bias),
                )
                out = cvx_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    init_cfg=init_cfg,
                    solve_cfg=SolveConfig(
                        method=args.cvx_method,
                        loss_name=args.loss_type,
                        beta=float(cvx_beta),
                        lr=float(cvx_lr),
                        optimizer_name=args.optimizer_name,
                        epochs=args.cvx_epochs,
                        batch_size=None if args.batch_size == -1 else int(args.batch_size),
                    ),
                )
                score = float(out.final_losses.get("val_objective", out.final_losses["val_loss"]))
                if score < best_cvx_score:
                    best_cvx_score = score
                    best_cvx = out
                    best_cvx_params = {"lr": float(cvx_lr), "beta": float(cvx_beta), "bias": float(cvx_bias)}
                    best_init = init_cfg
    if best_cvx is None or best_cvx_params is None or best_init is None:
        raise RuntimeError("Simple mode failed to find CVX candidate.")

    return {
        "mode": "simple",
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(LR_GRID_DEFAULT),
            "bias_grid": list(BIAS_GRID_DEFAULT),
        },
        "ste": {
            "selected_params": best_ste_params,
            "best_losses": best_ste.best_losses,
            "test_last_step_acc": _ste_last_step_acc(best_ste.model, x_test, y_test),
        },
        "cvx_gaussian": {
            "selected_params": best_cvx_params,
            "final_losses": best_cvx.final_losses,
            "diagnostics": asdict(best_cvx.diagnostics),
            "test_last_step_acc": _cvx_last_step_acc(
                best_cvx,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=best_init,
            ),
        },
    }


def _run_fine_tune_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = FineTuneConfig(
        T=args.T,
        P_in=data["d_in"],
        P_rec=args.P_rec,
        P_last=args.P_last,
        L=args.L,
        loss_type=args.loss_type,
        epochs=args.cvx_epochs,
        batch_size=args.batch_size,
        cvx_optimizer=args.optimizer_name,
        cvx_method=args.cvx_method,
        ste_pretrain_epochs=args.ste_pretrain_epochs,
        ste_post_epochs=args.ste_post_epochs,
        last_layer_readout=args.last_layer_readout,
        readout_modes=(args.last_layer_readout,),
        # Fixed internal grids (not exposed as CLI args)
        beta_grid=BETA_GRID_DEFAULT,
        lr_grid=LR_GRID_DEFAULT,
        bias_grid=BIAS_GRID_DEFAULT,
        ste_beta_grid=BETA_GRID_DEFAULT,
        ste_lr_grid=LR_GRID_DEFAULT,
    )
    out = run_fine_tune_pipeline(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_classes=data["num_classes"],
        seed=args.seed,
        cfg=cfg,
    )
    return {
        "mode": "fine_tune",
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(LR_GRID_DEFAULT),
            "bias_grid": list(BIAS_GRID_DEFAULT),
        },
        "result": out,
    }


def _run_layer_wise_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = LayerWiseConfig(
        num_blocks=args.num_blocks,
        L=args.L,
        P_rec=args.P_rec,
        P_last=args.P_last,
        ste_pretrain_epochs=args.ste_pretrain_epochs,
        ste_finetune_epochs=args.ste_post_epochs,
        loss_name=args.loss_type,
        optimizer_name=args.optimizer_name,
        cvx_method=args.cvx_method,
        # Fixed internal grids (not exposed as CLI args)
        cvx_beta_grid=BETA_GRID_DEFAULT,
        cvx_lr_grid=LR_GRID_DEFAULT,
        cvx_bias_grid=BIAS_GRID_DEFAULT,
        ste_beta_grid=BETA_GRID_DEFAULT,
        ste_lr_grid=LR_GRID_DEFAULT,
    )
    out = run_layer_wise_stacking_test_bench(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_classes=data["num_classes"],
        cfg=cfg,
    )
    return {
        "mode": "layer_wise",
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(LR_GRID_DEFAULT),
            "bias_grid": list(BIAS_GRID_DEFAULT),
        },
        "result": out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple atomic test runner: fine_tune, layer_wise, or gaussian-cvx-vs-ste baseline.",
    )
    parser.add_argument("--mode", choices=("fine_tune", "layer_wise", "simple"), default="simple")
    parser.add_argument(
        "--dataset",
        choices=("mnist_seq", "mnist_perm_seq", "cifar_seq", "arithmetic_seq", "dfa", "uci"),
        default="mnist_seq",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--P_rec", type=int, default=128)
    parser.add_argument("--P_last", type=int, default=128)
    parser.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge_ovr")
    parser.add_argument("--optimizer_name", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--cvx_method", choices=("cvx", "sgd"), default="cvx")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--cvx_epochs", type=int, default=150)
    parser.add_argument("--ste_pretrain_epochs", type=int, default=80)
    parser.add_argument("--ste_post_epochs", type=int, default=100)
    parser.add_argument("--ste_epochs", type=int, default=100, help="Used only in --mode simple.")
    parser.add_argument("--num_blocks", type=int, default=2, help="Used only in --mode layer_wise.")
    parser.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="membrane")

    # Arithmetic dataset options
    parser.add_argument("--arith_op", choices=("add", "sub", "mul", "div"), default="add")
    parser.add_argument("--arith_base", type=int, default=2)
    parser.add_argument("--n_digits", type=int, default=5)

    # DFA dataset options
    parser.add_argument("--dfa_spec", type=str, default="tomita_3")

    # UCI dataset options
    parser.add_argument("--uci_name", type=str, default="pima")
    parser.add_argument("--uci_test_size", type=float, default=0.2)
    parser.add_argument("--uci_val_size", type=float, default=0.2)
    parser.add_argument("--uci_no_standardize", action="store_true")

    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    data = _load_dataset_from_args(args)

    if args.mode == "simple":
        result = _run_simple_mode(args, data)
    elif args.mode == "fine_tune":
        result = _run_fine_tune_mode(args, data)
    elif args.mode == "layer_wise":
        result = _run_layer_wise_mode(args, data)
    else:
        raise ValueError(f"Unsupported mode={args.mode}")

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
