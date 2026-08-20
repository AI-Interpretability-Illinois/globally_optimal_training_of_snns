#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from cvx_side_ckpt import ckpt_path, load_weight_list, require_ckpt_task, save_weight_list
    from data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        generate_samples_for_op_base_seq,
        verify_sample_seq,
    )
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers import cvx_parallel_Solve as cvx_par
    from solvers.cvx_solve import InitializationConfig, _build_feature_map
    from solvers.cvx_carry_teacher_solve import (
        cvx_fit_shared_two_head,
        resolve_cvx_carry_loss_name,
        resolve_cvx_sum_loss_name,
    )
    from solvers.lsm_carry_ar import lsm_fit_two_head_ar, lsm_predict_sum_carry_autoregressive
    from solvers.ste_carry_teacher_solve import (
        CarryAugmentedSNN,
        resolve_ste_carry_loss_name,
        resolve_ste_sum_loss_name,
        ste_sweep_and_train,
    )
else:
    from .cvx_side_ckpt import ckpt_path, load_weight_list, require_ckpt_task, save_weight_list
    from .data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        generate_samples_for_op_base_seq,
        verify_sample_seq,
    )
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers import cvx_parallel_Solve as cvx_par
    from .solvers.cvx_solve import InitializationConfig, _build_feature_map
    from .solvers.cvx_carry_teacher_solve import (
        cvx_fit_shared_two_head,
        resolve_cvx_carry_loss_name,
        resolve_cvx_sum_loss_name,
    )
    from .solvers.lsm_carry_ar import lsm_fit_two_head_ar, lsm_predict_sum_carry_autoregressive
    from .solvers.ste_carry_teacher_solve import (
        CarryAugmentedSNN,
        resolve_ste_carry_loss_name,
        resolve_ste_sum_loss_name,
        ste_sweep_and_train,
    )

__all__ = ["main", "CarryAugmentedSNN", "build_carry_augmented_dataset"]


def _add_timesteps(n_digits: int) -> int:
    return int(n_digits) + 1


def _lc_dirname(lambda_carry: float) -> str:
    s = f"{float(lambda_carry):.10g}".replace(".", "p").replace("-", "m")
    return f"lambda_carry_{s}"


def _mean_std_list(vals: List[float]) -> Dict[str, float]:
    a = np.array([float(x) for x in vals], dtype=np.float64)
    if a.size < 1:
        raise ValueError("mean_std: empty list")
    if a.size == 1:
        return {"mean": float(a[0]), "std": 0.0, "n": 1}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": int(a.size)}


def _float_dict_mean_std(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    for d in dicts[1:]:
        if set(d.keys()) != set(keys):
            raise ValueError("float_dict key mismatch")
    out: Dict[str, Any] = {}
    for k in keys:
        vs = [d[k] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs):
            out[k] = _mean_std_list([float(v) for v in vs])
        else:
            out[k] = dicts[0][k]
    return out


def _aggregate_ood(ood_list: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not ood_list or all(x is None for x in ood_list):
        return None
    if not all(x is not None for x in ood_list):
        raise ValueError("OOD: either all None or all present across seeds")
    if len(ood_list) == 1:
        return ood_list[0]
    return {"n_seeds": int(len(ood_list)), "per_seed": True}


def _aggregate_sweep_across_seeds(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
    sweeps = [sp["sweep"] for sp in seed_payloads]
    base_grid = [float(e["lambda_carry"]) for e in sweeps[0]]
    for sw in sweeps[1:]:
        if [float(e["lambda_carry"]) for e in sw] != base_grid:
            raise ValueError("lambda_carry sweep order differs across seeds")
    agg: List[Dict[str, Any]] = []
    for i, lc in enumerate(base_grid):
        ents = [sw[i] for sw in sweeps]
        row: Dict[str, Any] = {"lambda_carry": float(lc)}
        for method in ("ste", "lsm", "cvx", "sg_cvx", "r_cvx"):
            blocks = [e.get(method) for e in ents]
            if all(b is None for b in blocks):
                row[method] = None
                continue
            if any(b is None for b in blocks):
                raise ValueError(f"method={method} is None for some seeds and present for others.")
            payload: Dict[str, Any] = {
                "test_metrics": _float_dict_mean_std([b["test_metrics"] for b in blocks])
            }
            if method not in ("ste", "lsm"):
                payload["diagnostics"] = _float_dict_mean_std([b["diagnostics"] for b in blocks])
            row[method] = payload
        row["ood_eval"] = _aggregate_ood([e.get("ood_eval") for e in ents])
        agg.append(row)
    return agg


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CarryAugmentedDataset:
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
    dataset_name: str


def _samples_to_xy_carry(samples: Sequence[Any], base: int, verify_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vc = min(verify_count, len(samples))
    for i in range(vc):
        verify_sample_seq(samples[i])
    scale = max(base - 1, 1)
    X_full = np.stack([s.inputs.astype(np.float32) / scale for s in samples], axis=0)
    X = X_full[:, :, :3].astype(np.float32, copy=False)
    y_sum = np.stack([s.target_tokens.astype(np.int64) for s in samples], axis=0)
    carry_in = np.stack([s.inputs[:, 2].astype(np.int64) for s in samples], axis=0)
    y_carry = np.zeros_like(y_sum, dtype=np.int64)
    y_carry[:, :-1] = carry_in[:, 1:]
    y_carry[:, -1] = 0
    return X, y_sum, y_carry


def _maybe_subsample_train(ds: CarryAugmentedDataset, max_train: int) -> CarryAugmentedDataset:
    if max_train <= 0 or ds.X_train.shape[0] <= max_train:
        return ds
    return CarryAugmentedDataset(
        X_train=ds.X_train[:max_train],
        y_sum_train=ds.y_sum_train[:max_train],
        y_carry_train=ds.y_carry_train[:max_train],
        X_val=ds.X_val,
        y_sum_val=ds.y_sum_val,
        y_carry_val=ds.y_carry_val,
        X_test=ds.X_test,
        y_sum_test=ds.y_sum_test,
        y_carry_test=ds.y_carry_test,
        num_sum_classes=ds.num_sum_classes,
        d_in=ds.d_in,
        T=ds.T,
        dataset_name=f"{ds.dataset_name}::debug_train{max_train}",
    )


def build_carry_augmented_dataset(
    *,
    base: int,
    n_digits: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    verify_count: int,
    add_initial_carry: str = "random",
) -> CarryAugmentedDataset:
    train_samples = generate_samples_for_op_base_seq("add", base, n_digits, n_train, seed + 11, add_initial_carry=add_initial_carry)
    val_samples = generate_samples_for_op_base_seq("add", base, n_digits, n_val, seed + 29, add_initial_carry=add_initial_carry)
    test_samples = generate_samples_for_op_base_seq("add", base, n_digits, n_test, seed + 47, add_initial_carry=add_initial_carry)
    X_train, y_sum_train, y_carry_train = _samples_to_xy_carry(train_samples, base, verify_count)
    X_val, y_sum_val, y_carry_val = _samples_to_xy_carry(val_samples, base, verify_count)
    X_test, y_sum_test, y_carry_test = _samples_to_xy_carry(test_samples, base, verify_count)
    tag = f"arith_carry_ar_eval::base{base}::digits{n_digits}::ic_{add_initial_carry}"
    return CarryAugmentedDataset(
        X_train=X_train,
        y_sum_train=y_sum_train,
        y_carry_train=y_carry_train,
        X_val=X_val,
        y_sum_val=y_sum_val,
        y_carry_val=y_carry_val,
        X_test=X_test,
        y_sum_test=y_sum_test,
        y_carry_test=y_carry_test,
        num_sum_classes=int(base),
        d_in=int(X_train.shape[2]),
        T=int(X_train.shape[1]),
        dataset_name=tag,
    )


def _make_ood_carry_tensors(*, base: int, n_digits: int, n_test: int, seed: int, verify_count: int, add_initial_carry: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = generate_samples_for_op_base_seq("add", base, n_digits, n_test, seed, add_initial_carry=add_initial_carry)
    return _samples_to_xy_carry(samples, base, verify_count)


def _resolve_cvx_device(device_name: str) -> Optional[torch.device]:
    if device_name == "auto":
        return None
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested cvx_device=cuda but CUDA not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested cvx_device=mps but MPS not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown cvx_device={device_name!r}")


# ------------------------------------------------------------
# Prediction / metrics
# ------------------------------------------------------------

def _decode_sum(sum_logits: np.ndarray, base: int) -> np.ndarray:
    if int(base) == 2:
        return (sum_logits.reshape(-1) >= 0.0).astype(np.int64)
    return np.argmax(sum_logits, axis=1).astype(np.int64)


def _decode_carry(carry_logits: np.ndarray) -> np.ndarray:
    return (carry_logits.reshape(-1) >= 0.0).astype(np.int64)


def carry_autoregressive_metrics(sum_pred: np.ndarray, carry_pred: np.ndarray, y_sum: np.ndarray, y_carry: np.ndarray) -> Dict[str, float]:
    sum_ok = sum_pred == y_sum
    carry_ok = carry_pred == y_carry
    both_ok = sum_ok & carry_ok
    n, T = y_sum.shape
    wrong_sum_rows = ~sum_ok.all(axis=1)
    wrong_carry_rows = ~carry_ok.all(axis=1)
    first_wrong_sum = [int(np.argmax(~sum_ok[i])) for i in range(n) if wrong_sum_rows[i]]
    first_wrong_carry = [int(np.argmax(~carry_ok[i])) for i in range(n) if wrong_carry_rows[i]]
    return {
        "sum_token_acc": float(sum_ok.mean()),
        "carry_token_acc": float(carry_ok.mean()),
        "joint_token_acc": float(both_ok.mean()),
        "joint_seq_acc": float(both_ok.all(axis=1).mean()),
        "sum_seq_acc": float(sum_ok.all(axis=1).mean()),
        "carry_seq_acc": float(carry_ok.all(axis=1).mean()),
        "mean_first_wrong_sum_among_error_seq": float(np.mean(first_wrong_sum)) if first_wrong_sum else float(T),
        "mean_first_wrong_carry_among_error_seq": float(np.mean(first_wrong_carry)) if first_wrong_carry else float(T),
    }


def _build_cvx_features_for_eval(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if int(getattr(init_cfg, "K_parallel", 1)) > 1:
        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        _, _, d_test, _ = cvx_par._build_feature_map(x_train, x_val, x_test, pic, all_timesteps=True)
    else:
        _, _, d_test, _ = _build_feature_map(x_train, x_val, x_test, init_cfg, all_timesteps=True)
    return d_test


def ste_predict_sum_carry_autoregressive(model: CarryAugmentedSNN, x_seq: np.ndarray, *, base: int) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    n, T, _ = x_seq.shape
    scale = max(int(base) - 1, 1)
    x_roll = x_seq.astype(np.float32, copy=True)
    sum_pred = np.zeros((n, T), dtype=np.int64)
    carry_pred = np.zeros((n, T), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for t in range(T):
            xt = torch.tensor(x_roll, dtype=torch.float32, device=device)
            sum_logits, carry_logits = model(xt)
            sum_t = sum_logits[:, t, :].detach().cpu().numpy()
            carry_t = carry_logits[:, t, :].detach().cpu().numpy()
            sum_pred[:, t] = _decode_sum(sum_t, int(base))
            carry_pred[:, t] = _decode_carry(carry_t)
            if t + 1 < T:
                x_roll[:, t + 1, 2] = carry_pred[:, t].astype(np.float32) / float(scale)
    return sum_pred, carry_pred


def cvx_predict_sum_carry_autoregressive(
    *,
    w_sum: np.ndarray,
    w_carry: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_seq: np.ndarray,
    init_cfg: InitializationConfig,
    base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n, T, _ = x_seq.shape
    scale = max(int(base) - 1, 1)
    x_roll = x_seq.astype(np.float32, copy=True)
    sum_pred = np.zeros((n, T), dtype=np.int64)
    carry_pred = np.zeros((n, T), dtype=np.int64)
    for t in range(T):
        d_test = _build_cvx_features_for_eval(x_train=x_train, x_val=x_val, x_test=x_roll, init_cfg=init_cfg)
        sum_scores = d_test @ w_sum
        carry_scores = d_test @ w_carry
        if int(base) == 2:
            sum_scores_t = sum_scores.reshape(n, T)[:, t]
        else:
            sum_scores_t = sum_scores.reshape(n, T, int(base))[:, t, :]
        carry_scores_t = carry_scores.reshape(n, T)[:, t]
        sum_pred[:, t] = _decode_sum(sum_scores_t, int(base))
        carry_pred[:, t] = _decode_carry(carry_scores_t)
        if t + 1 < T:
            x_roll[:, t + 1, 2] = carry_pred[:, t].astype(np.float32) / float(scale)
    return sum_pred, carry_pred


def _ood_eval(
    *,
    ds: CarryAugmentedDataset,
    arith_base: int,
    ste_model: Optional[CarryAugmentedSNN],
    init_cfg: Optional[InitializationConfig],
    w_sum: Optional[np.ndarray],
    w_carry: Optional[np.ndarray],
    ood_digits: List[int],
    n_test_ood: int,
    seed: int,
    verify_count: int,
    add_initial_carry: str,
    cvx_tag: str = "cvx",
    lsm_model=None,
    lsm_w_sum: Optional[np.ndarray] = None,
    lsm_w_carry: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if (
        ste_model is None
        and (w_sum is None or w_carry is None or init_cfg is None)
        and (lsm_model is None or lsm_w_sum is None or lsm_w_carry is None)
    ):
        raise ValueError("OOD eval requires ste_model, CVX weights, or LSM ridge heads.")
    out: Dict[str, Any] = {}
    d_in = int(ds.d_in)
    for nd in ood_digits:
        te_seed = int(seed) + 10_000 + int(nd) * 97
        x_ood, y_s, y_c = _make_ood_carry_tensors(
            base=arith_base,
            n_digits=int(nd),
            n_test=n_test_ood,
            seed=te_seed,
            verify_count=verify_count,
            add_initial_carry=add_initial_carry,
        )
        T_expect = _add_timesteps(int(nd))
        if x_ood.shape[1] != T_expect or x_ood.shape[2] != d_in:
            raise ValueError(
                f"OOD shape mismatch n_digits={nd}: got T={x_ood.shape[1]} d_in={x_ood.shape[2]}, expect T={T_expect} d_in={d_in}."
            )
        block: Dict[str, Any] = {"n_digits": int(nd), "n_test": int(n_test_ood)}
        if ste_model is not None:
            sp, cp = ste_predict_sum_carry_autoregressive(ste_model, x_ood, base=arith_base)
            block["ste"] = carry_autoregressive_metrics(sp, cp, y_s, y_c)
        if lsm_model is not None:
            if lsm_w_sum is None or lsm_w_carry is None:
                raise ValueError("LSM OOD eval requires lsm_w_sum and lsm_w_carry.")
            sp_l, cp_l = lsm_predict_sum_carry_autoregressive(
                lsm_model, lsm_w_sum, lsm_w_carry, x_ood, base=arith_base
            )
            block["lsm"] = carry_autoregressive_metrics(sp_l, cp_l, y_s, y_c)
        if w_sum is not None:
            if init_cfg is None or w_carry is None:
                raise ValueError("CVX OOD eval requires init_cfg and w_carry.")
            s_c, c_c = cvx_predict_sum_carry_autoregressive(
                w_sum=w_sum,
                w_carry=w_carry,
                x_train=ds.X_train,
                x_val=ds.X_val,
                x_seq=x_ood,
                init_cfg=init_cfg,
                base=arith_base,
            )
            block[cvx_tag] = carry_autoregressive_metrics(s_c, c_c, y_s, y_c)
        out[f"n_digits_{nd}"] = block
    return out


def _ar_task_name(base: int) -> str:
    return f"add_ar_b{int(base)}"


def _ar_ste_ckpt(ckpt_dir: Path, *, seed: int, T: int, L: int, K: int, base: int, lambda_carry: float) -> Path:
    return ckpt_path(
        ckpt_dir,
        seed=int(seed),
        T=int(T),
        L=int(L),
        K=int(K),
        tag=f"ste_{_lc_dirname(float(lambda_carry))}",
        task=_ar_task_name(base),
    )


def _ar_lsm_ckpt(ckpt_dir: Path, *, seed: int, T: int, L: int, K: int, base: int) -> Path:
    return ckpt_path(
        ckpt_dir,
        seed=int(seed),
        T=int(T),
        L=int(L),
        K=int(K),
        tag="lsm",
        task=_ar_task_name(base),
    )


def _cvx_ar_fit_and_eval(
    *,
    ds: CarryAugmentedDataset,
    args: argparse.Namespace,
    run_seed: int,
    lambda_carry: float,
    init_mode: str,
    pretrained_weights: Optional[Sequence[np.ndarray]],
    beta_leak: float = 0.99,
    threshold: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    bundle, sel, _tf_metrics = cvx_fit_shared_two_head(
        ds=ds,
        L=int(args.L),
        P_rec=int(args.P_rec),
        P_last=int(args.P_last),
        K_parallel=int(args.K_parallel),
        cvx_last_layer_readout=str(args.cvx_last_layer_readout),
        seed=run_seed,
        beta_grid=args.cvx_beta_grid,
        bias_grid=args.cvx_bias_grid,
        lambda_sum=float(args.lambda_sum),
        lambda_carry=float(lambda_carry),
        cvx_device=_resolve_cvx_device(str(args.cvx_device)),
        cvx_sum_loss=str(args.cvx_sum_loss),
        cvx_carry_loss=str(args.cvx_carry_loss),
        cvx_time_loss=str(args.cvx_time_loss),
        init_mode=str(init_mode),
        pretrained_weights=pretrained_weights,
        cvx_method=str(args.cvx_method),
        compute_dual=False,
        lite_max_iter=int(args.lite_max_iter),
        lite_tol=float(args.lite_tol),
        beta_leak=float(beta_leak),
        threshold=float(threshold),
    )
    s_c, c_c = cvx_predict_sum_carry_autoregressive(
        w_sum=bundle["sum_weights"],
        w_carry=bundle["carry_weights"],
        x_train=ds.X_train,
        x_val=ds.X_val,
        x_seq=ds.X_test,
        init_cfg=bundle["init_cfg"],
        base=int(args.arith_base),
    )
    test = carry_autoregressive_metrics(s_c, c_c, ds.y_sum_test, ds.y_carry_test)
    return bundle, sel, test


def _cvx_entry(sel: Dict[str, float], bundle: Dict[str, Any], test: Dict[str, float]) -> Dict[str, Any]:
    return {
        "selected_params": sel,
        "diagnostics": {
            "primal_value": float(bundle["primal_value"]),
            "dual_value": float(bundle["dual_value"]),
            "gap": float(bundle["gap"]),
        },
        "test_metrics": test,
    }

_TF_OBJECTIVE_CHOICES = ("joint", "lambda_weighted", "mean_pair", "lambda_normalized")
_STE_SUM_LOSS = ("auto", "hinge", "ce", "hinge_ovr")
_STE_CARRY_LOSS = ("auto", "hinge", "ce")
_CVX_SUM_LOSS = ("auto", "hinge", "ce", "hinge_ovr")
_CVX_CARRY_LOSS = ("auto", "hinge", "ce")
_ADD_INITIAL_CARRY = ("zero", "random")


def main() -> None:
    ap = argparse.ArgumentParser(description="Carry-augmented addition with teacher-forced training and autoregressive rollout evaluation.")
    ap.add_argument("--arith_base", type=int, default=2, choices=SUPPORTED_BASES)
    ap.add_argument("--n_digits", type=int, default=5, help="In-distribution train/val/test digit width.")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--n_test_ood", type=int, default=0, help="Test samples per OOD length; 0 = use --n_test.")
    ap.add_argument("--ood_digits", type=int, nargs="*", default=None, help="n_digits for OOD eval; default 10 20 50. Empty list disables.")
    ap.add_argument("--add_initial_carry", type=str, default="random", choices=_ADD_INITIAL_CARRY)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--verify_samples", type=int, default=5)
    ap.add_argument("--debug_max_train", type=int, default=0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=500)
    ap.add_argument("--P_last", type=int, default=1000)
    ap.add_argument("--K_parallel", type=int, default=10)
    ap.add_argument("--ste_last_layer_readout", type=str, default="membrane", choices=["membrane", "spike"])
    ap.add_argument("--cvx_last_layer_readout", type=str, default="spike", choices=["membrane", "spike"])
    ap.add_argument("--ste_epochs", type=int, default=200)
    ap.add_argument("--optimizer_name", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--lambda_sum", type=float, default=1.0)
    ap.add_argument("--lambda_carry", type=float, default=2.0)
    ap.add_argument("--lambda_carry_grid", type=float, nargs="*", default=[2.0, 12.0])
    ap.add_argument(
        "--ste_time_loss",
        type=str,
        default="ramp",
        choices=("uniform", "ramp"),
        help="STE: per-timestep loss mix. 'ramp' = α_t=2t/(T+1) on batch-mean token losses (default). 'uniform' = mean over (batch, time).",
    )
    ap.add_argument(
        "--cvx_time_loss",
        type=str,
        default="ramp",
        choices=("uniform", "ramp"),
        help="CVX: match STE time weighting for train+val. Default ramp (see --ste_time_loss).",
    )
    ap.add_argument("--tf_objective", type=str, default="joint", choices=_TF_OBJECTIVE_CHOICES)
    ap.add_argument("--ste_sum_loss", type=str, default="auto", choices=_STE_SUM_LOSS)
    ap.add_argument("--ste_carry_loss", type=str, default="auto", choices=_STE_CARRY_LOSS)
    ap.add_argument("--cvx_sum_loss", type=str, default="auto", choices=_CVX_SUM_LOSS)
    ap.add_argument("--cvx_carry_loss", type=str, default="auto", choices=_CVX_CARRY_LOSS)
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--cvx_device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--side",
        choices=("all", "cvx", "non_cvx"),
        default="all",
        help=(
            "all: STE + LSM ridge + Gaussian CVX + STE-CVX + R-CVX, AR eval. "
            "non_cvx: STE + LSM ridge; writes *_ste.npz and *_lsm.npz. "
            "cvx: Gaussian CVX + STE-CVX + R-CVX (loads --ckpt_dir); default method is cvx_lite."
        ),
    )
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="STE / LSM hidden-stack checkpoints. Default: <out_root>/ckpts.",
    )
    ap.add_argument(
        "--cvx_method",
        choices=("cvx", "cvx_lite"),
        default="cvx_lite",
        help="cvx = cone program. cvx_lite = primal-only LASSO/FISTA (default).",
    )
    ap.add_argument("--lite_max_iter", type=int, default=5000)
    ap.add_argument("--lite_tol", type=float, default=1e-6)
    ap.add_argument("--ridge_lambdas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--lsm_beta_leak_grid", type=float, nargs="+", default=[0.90, 0.95, 0.99, 0.995])
    ap.add_argument("--lsm_threshold_grid", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--lsm_input_scale_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--lsm_last_layer_readout", type=str, default="membrane", choices=["membrane", "spike"])
    ap.add_argument("--debug", action="store_true", help="Tiny n/grids/epochs; disables OOD.")
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--no_save_metrics", action="store_true")
    args = ap.parse_args()
    if bool(args.debug):
        args.n_train = min(int(args.n_train), 32)
        args.n_val = min(int(args.n_val), 16)
        args.n_test = min(int(args.n_test), 16)
        args.debug_max_train = int(args.n_train) if int(args.debug_max_train) == 0 else min(int(args.debug_max_train), int(args.n_train))
        args.ste_epochs = min(int(args.ste_epochs), 5)
        args.ste_lr_grid = [float(args.ste_lr_grid[0])]
        args.ste_beta_grid = [float(args.ste_beta_grid[0])]
        args.cvx_beta_grid = [float(args.cvx_beta_grid[0])]
        args.lite_max_iter = min(int(args.lite_max_iter), 50)
        if args.ood_digits is None:
            args.ood_digits = []

    n_test_ood = int(args.n_test) if int(args.n_test_ood) == 0 else int(args.n_test_ood)
    ood_digits = [10, 20, 50] if args.ood_digits is None else [int(x) for x in args.ood_digits]
    lambda_carry_sweep = [float(x) for x in args.lambda_carry_grid]
    if len(lambda_carry_sweep) == 0:
        lambda_carry_sweep = [float(args.lambda_carry)]
    seed_list = [int(s) for s in args.seeds]
    aic = str(args.add_initial_carry)
    run_ste = str(args.side) in ("all", "non_cvx")
    run_lsm = str(args.side) in ("all", "non_cvx")
    run_cvx = str(args.side) in ("all", "cvx")

    atomic_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root).strip()).resolve()
    else:
        out_root = atomic_dir / "sweep_results" / f"carry_autoregressive_eval_b{int(args.arith_base)}_L{int(args.L)}_traind{int(args.n_digits)}_K{int(args.K_parallel)}_{stamp}"
    save_metrics = not bool(args.no_save_metrics)
    if save_metrics:
        out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if str(args.ckpt_dir).strip() else (out_root / "ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_dump: Dict[str, Any] = {
        "seeds": seed_list,
        "arith_base": int(args.arith_base),
        "n_digits": int(args.n_digits),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "n_test_ood": n_test_ood,
        "ood_digits": ood_digits,
        "lambda_sum": float(args.lambda_sum),
        "lambda_carry_sweep": lambda_carry_sweep,
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "add_initial_carry": aic,
        "ste_time_loss": str(args.ste_time_loss),
        "cvx_time_loss": str(args.cvx_time_loss),
        "tf_objective": str(args.tf_objective),
        "out_root": str(out_root),
        "ckpt_dir": str(ckpt_dir),
        "side": str(args.side),
        "cvx_method": str(args.cvx_method),
        "training_mode": "teacher_forcing",
        "evaluation_mode": "autoregressive_rollout",
    }
    if save_metrics:
        (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n")

    all_seed_payloads: List[Dict[str, Any]] = []

    for run_seed in seed_list:
        print("\n" + "=" * 80, flush=True)
        print(f"SEED {run_seed}", flush=True)
        print("=" * 80, flush=True)

        _set_seed(run_seed)
        ds = build_carry_augmented_dataset(
            base=int(args.arith_base),
            n_digits=int(args.n_digits),
            n_train=int(args.n_train),
            n_val=int(args.n_val),
            n_test=int(args.n_test),
            seed=run_seed,
            verify_count=int(args.verify_samples),
            add_initial_carry=aic,
        )
        ds = _maybe_subsample_train(ds, int(args.debug_max_train))

        ste_n_sum = resolve_ste_sum_loss_name(str(args.ste_sum_loss), ds.num_sum_classes)
        ste_n_carry = resolve_ste_carry_loss_name(str(args.ste_carry_loss))
        cvx_n_sum = resolve_cvx_sum_loss_name(str(args.cvx_sum_loss), ds.num_sum_classes)
        cvx_n_carry = resolve_cvx_carry_loss_name(str(args.cvx_carry_loss))

        sweep_out: List[Dict[str, Any]] = []
        lsm_result = None
        lsm_block: Optional[Dict[str, Any]] = None
        lsm_hidden: Optional[List[np.ndarray]] = None
        lsm_beta_leak = 0.99
        lsm_threshold = 1.0
        if run_lsm:
            print(f"[lsm-ar] seed={run_seed} criticality + two-head ridge", flush=True)
            lsm_result = lsm_fit_two_head_ar(
                x_train=ds.X_train,
                y_sum_train=ds.y_sum_train,
                y_carry_train=ds.y_carry_train,
                x_val=ds.X_val,
                y_sum_val=ds.y_sum_val,
                y_carry_val=ds.y_carry_val,
                d_in=int(ds.d_in),
                num_sum_classes=int(ds.num_sum_classes),
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                seed=run_seed,
                last_layer_readout=str(args.lsm_last_layer_readout),
                ridge_lambdas=args.ridge_lambdas,
                beta_leak_grid=args.lsm_beta_leak_grid,
                threshold_grid=args.lsm_threshold_grid,
                input_scale_grid=args.lsm_input_scale_grid,
                debug=bool(args.debug),
            )
            lsm_hidden = lsm_result.hidden_weights
            lsm_beta_leak = float(lsm_result.selected["beta_leak"])
            lsm_threshold = float(lsm_result.selected["threshold"])
            lsm_ckpt = _ar_lsm_ckpt(
                ckpt_dir,
                seed=run_seed,
                T=int(ds.T),
                L=int(args.L),
                K=int(args.K_parallel),
                base=int(args.arith_base),
            )
            save_weight_list(
                lsm_ckpt,
                lsm_hidden,
                meta={
                    "tag": "lsm",
                    "task": _ar_task_name(int(args.arith_base)),
                    "arith_base": int(args.arith_base),
                    "n_digits": int(args.n_digits),
                    "T": int(ds.T),
                    "L": int(args.L),
                    "K_parallel": int(args.K_parallel),
                    "P_rec": int(args.P_rec),
                    "P_last": int(args.P_last),
                    "seed": int(run_seed),
                    "beta_leak": lsm_beta_leak,
                    "threshold": lsm_threshold,
                    "input_scale": float(lsm_result.selected["input_scale"]),
                    "branching_ratio": float(lsm_result.selected["branching_ratio"]),
                    "selected_params": lsm_result.selected,
                },
            )
            sp_l, cp_l = lsm_predict_sum_carry_autoregressive(
                lsm_result.model,
                lsm_result.w_sum,
                lsm_result.w_carry,
                ds.X_test,
                base=int(args.arith_base),
            )
            lsm_test = carry_autoregressive_metrics(sp_l, cp_l, ds.y_sum_test, ds.y_carry_test)
            lsm_block = {"selected_params": lsm_result.selected, "test_metrics": lsm_test}

        for lc in lambda_carry_sweep:
            print(f"[sweep] seed={run_seed} lambda_carry={lc} side={args.side}", flush=True)
            ste_model: Optional[CarryAugmentedSNN] = None
            ste_hidden: Optional[List[np.ndarray]] = None
            one_entry: Dict[str, Any] = {
                "lambda_carry": float(lc),
                "ste": None,
                "lsm": lsm_block,
                "cvx": None,
                "sg_cvx": None,
                "r_cvx": None,
                "ood_eval": None,
            }

            if run_ste:
                ste_model, ste_sel, _ = ste_sweep_and_train(
                    ds=ds,
                    L=int(args.L),
                    P_rec=int(args.P_rec),
                    P_last=int(args.P_last),
                    K_parallel=int(args.K_parallel),
                    ste_last_layer_readout=str(args.ste_last_layer_readout),
                    ste_epochs=int(args.ste_epochs),
                    batch_size=int(args.batch_size),
                    optimizer_name=str(args.optimizer_name),
                    beta_leak=float(args.beta_leak),
                    threshold=float(args.threshold),
                    seed=run_seed,
                    ste_lr_grid=args.ste_lr_grid,
                    ste_beta_grid=args.ste_beta_grid,
                    lambda_sum=float(args.lambda_sum),
                    lambda_carry=float(lc),
                    ste_sum_loss=str(args.ste_sum_loss),
                    ste_carry_loss=str(args.ste_carry_loss),
                    tf_objective=str(args.tf_objective),
                    ste_time_loss=str(args.ste_time_loss),
                )
                ste_hidden = ste_model.hidden_weight_list()
                ste_ckpt = _ar_ste_ckpt(
                    ckpt_dir,
                    seed=run_seed,
                    T=int(ds.T),
                    L=int(args.L),
                    K=int(args.K_parallel),
                    base=int(args.arith_base),
                    lambda_carry=float(lc),
                )
                save_weight_list(
                    ste_ckpt,
                    ste_hidden,
                    meta={
                        "tag": f"ste_{_lc_dirname(float(lc))}",
                        "task": _ar_task_name(int(args.arith_base)),
                        "arith_base": int(args.arith_base),
                        "n_digits": int(args.n_digits),
                        "T": int(ds.T),
                        "L": int(args.L),
                        "K_parallel": int(args.K_parallel),
                        "P_rec": int(args.P_rec),
                        "P_last": int(args.P_last),
                        "seed": int(run_seed),
                        "lambda_carry": float(lc),
                        "selected_params": ste_sel,
                    },
                )
                sp, cp = ste_predict_sum_carry_autoregressive(
                    ste_model, ds.X_test, base=int(args.arith_base)
                )
                ste_test = carry_autoregressive_metrics(sp, cp, ds.y_sum_test, ds.y_carry_test)
                one_entry["ste"] = {"selected_params": ste_sel, "test_metrics": ste_test}

            if run_cvx:
                cvx_bundle, cvx_sel, cvx_test = _cvx_ar_fit_and_eval(
                    ds=ds,
                    args=args,
                    run_seed=run_seed,
                    lambda_carry=float(lc),
                    init_mode="gaussian",
                    pretrained_weights=None,
                )
                one_entry["cvx"] = _cvx_entry(cvx_sel, cvx_bundle, cvx_test)
                if ste_hidden is None:
                    ste_ckpt = _ar_ste_ckpt(
                        ckpt_dir,
                        seed=run_seed,
                        T=int(ds.T),
                        L=int(args.L),
                        K=int(args.K_parallel),
                        base=int(args.arith_base),
                        lambda_carry=float(lc),
                    )
                    ste_hidden, ste_meta = load_weight_list(ste_ckpt)
                    require_ckpt_task(ste_meta, expected_task=_ar_task_name(int(args.arith_base)), path=ste_ckpt)
                sg_bundle, sg_sel, sg_test = _cvx_ar_fit_and_eval(
                    ds=ds,
                    args=args,
                    run_seed=run_seed,
                    lambda_carry=float(lc),
                    init_mode="pretraining",
                    pretrained_weights=ste_hidden,
                )
                one_entry["sg_cvx"] = _cvx_entry(sg_sel, sg_bundle, sg_test)
                if lsm_hidden is None:
                    lsm_ckpt = _ar_lsm_ckpt(
                        ckpt_dir,
                        seed=run_seed,
                        T=int(ds.T),
                        L=int(args.L),
                        K=int(args.K_parallel),
                        base=int(args.arith_base),
                    )
                    lsm_hidden, lsm_meta = load_weight_list(lsm_ckpt)
                    require_ckpt_task(lsm_meta, expected_task=_ar_task_name(int(args.arith_base)), path=lsm_ckpt)
                    lsm_beta_leak = float(lsm_meta["beta_leak"])
                    lsm_threshold = float(lsm_meta["threshold"])
                r_bundle, r_sel, r_test = _cvx_ar_fit_and_eval(
                    ds=ds,
                    args=args,
                    run_seed=run_seed,
                    lambda_carry=float(lc),
                    init_mode="pretraining",
                    pretrained_weights=lsm_hidden,
                    beta_leak=float(lsm_beta_leak),
                    threshold=float(lsm_threshold),
                )
                one_entry["r_cvx"] = _cvx_entry(r_sel, r_bundle, r_test)

            if len(ood_digits) > 0:
                ood_report: Dict[str, Any] = {}
                if run_ste:
                    ood_ste = _ood_eval(
                        ds=ds,
                        arith_base=int(args.arith_base),
                        ste_model=ste_model,
                        init_cfg=None,
                        w_sum=None,
                        w_carry=None,
                        ood_digits=ood_digits,
                        n_test_ood=n_test_ood,
                        seed=run_seed,
                        verify_count=int(args.verify_samples),
                        add_initial_carry=aic,
                    )
                    ood_report["ste"] = ood_ste
                if run_lsm:
                    if lsm_result is None:
                        raise RuntimeError("run_lsm is set but lsm_result is missing.")
                    ood_lsm = _ood_eval(
                        ds=ds,
                        arith_base=int(args.arith_base),
                        ste_model=None,
                        init_cfg=None,
                        w_sum=None,
                        w_carry=None,
                        ood_digits=ood_digits,
                        n_test_ood=n_test_ood,
                        seed=run_seed,
                        verify_count=int(args.verify_samples),
                        add_initial_carry=aic,
                        lsm_model=lsm_result.model,
                        lsm_w_sum=lsm_result.w_sum,
                        lsm_w_carry=lsm_result.w_carry,
                    )
                    ood_report["lsm"] = ood_lsm
                if run_cvx:
                    ood_cvx = _ood_eval(
                        ds=ds,
                        arith_base=int(args.arith_base),
                        ste_model=None,
                        init_cfg=cvx_bundle["init_cfg"],
                        w_sum=cvx_bundle["sum_weights"],
                        w_carry=cvx_bundle["carry_weights"],
                        ood_digits=ood_digits,
                        n_test_ood=n_test_ood,
                        seed=run_seed,
                        verify_count=int(args.verify_samples),
                        add_initial_carry=aic,
                        cvx_tag="cvx",
                    )
                    ood_sg = _ood_eval(
                        ds=ds,
                        arith_base=int(args.arith_base),
                        ste_model=None,
                        init_cfg=sg_bundle["init_cfg"],
                        w_sum=sg_bundle["sum_weights"],
                        w_carry=sg_bundle["carry_weights"],
                        ood_digits=ood_digits,
                        n_test_ood=n_test_ood,
                        seed=run_seed,
                        verify_count=int(args.verify_samples),
                        add_initial_carry=aic,
                        cvx_tag="sg_cvx",
                    )
                    ood_r = _ood_eval(
                        ds=ds,
                        arith_base=int(args.arith_base),
                        ste_model=None,
                        init_cfg=r_bundle["init_cfg"],
                        w_sum=r_bundle["sum_weights"],
                        w_carry=r_bundle["carry_weights"],
                        ood_digits=ood_digits,
                        n_test_ood=n_test_ood,
                        seed=run_seed,
                        verify_count=int(args.verify_samples),
                        add_initial_carry=aic,
                        cvx_tag="r_cvx",
                    )
                    ood_report["cvx"] = ood_cvx
                    ood_report["sg_cvx"] = ood_sg
                    ood_report["r_cvx"] = ood_r
                one_entry["ood_eval"] = ood_report
                for split_name, blocks in ood_report.items():
                    for k, block in blocks.items():
                        print(
                            f"  [ood] seed={run_seed} {k} {split_name} "
                            f"joint_token_acc={block[split_name]['joint_token_acc']:.4f}",
                            flush=True,
                        )

            sweep_out.append(one_entry)
            if save_metrics:
                lc_dir = out_root / f"seed_{run_seed}" / _lc_dirname(float(lc))
                lc_dir.mkdir(parents=True, exist_ok=True)
                (lc_dir / "metrics.json").write_text(json.dumps(one_entry, indent=2, default=str) + "\n")

        seed_payload: Dict[str, Any] = {
            "seed": int(run_seed),
            "dataset": ds.dataset_name,
            "sweep": sweep_out,
            "setup": {
                "arith_base": int(args.arith_base),
                "n_digits": int(args.n_digits),
                "n_train": int(args.n_train),
                "n_val": int(args.n_val),
                "n_test": int(args.n_test),
                "n_test_ood": n_test_ood,
                "ood_digits": ood_digits,
                "L": int(args.L),
                "P_rec": int(args.P_rec),
                "P_last": int(args.P_last),
                "K_parallel": int(args.K_parallel),
                "ste_last_layer_readout": str(args.ste_last_layer_readout),
                "cvx_last_layer_readout": str(args.cvx_last_layer_readout),
                "lambda_sum": float(args.lambda_sum),
                "lambda_carry_sweep": lambda_carry_sweep,
                "add_initial_carry": aic,
                "ste_time_loss": str(args.ste_time_loss),
                "cvx_time_loss": str(args.cvx_time_loss),
                "tf_objective": str(args.tf_objective),
                "teacher_forcing_train": True,
                "autoregressive_eval": True,
                "ste_sum_loss": str(args.ste_sum_loss),
                "ste_sum_loss_resolved": ste_n_sum,
                "ste_carry_loss": str(args.ste_carry_loss),
                "ste_carry_loss_resolved": ste_n_carry,
                "cvx_sum_loss": str(args.cvx_sum_loss),
                "cvx_sum_loss_resolved": cvx_n_sum,
                "cvx_carry_loss": str(args.cvx_carry_loss),
                "cvx_carry_loss_resolved": cvx_n_carry,
                "debug_max_train": int(args.debug_max_train),
                "side": str(args.side),
                "cvx_method": str(args.cvx_method),
            },
        }
        if len(sweep_out) == 1:
            seed_payload["ste"] = sweep_out[0]["ste"]
            seed_payload["lsm"] = sweep_out[0].get("lsm")
            seed_payload["cvx"] = sweep_out[0]["cvx"]
            seed_payload["sg_cvx"] = sweep_out[0].get("sg_cvx")
            seed_payload["r_cvx"] = sweep_out[0].get("r_cvx")
            seed_payload["ood_eval"] = sweep_out[0].get("ood_eval")
        if save_metrics:
            sdir = out_root / f"seed_{run_seed}"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "metrics.json").write_text(json.dumps(seed_payload, indent=2, default=str) + "\n")
        all_seed_payloads.append(seed_payload)

    aggregate = {
        "n_seeds": int(len(all_seed_payloads)),
        "seeds": seed_list,
        "description": "For each float metric, mean and std (ddof=1) across seeds. Same lambda_carry order per seed required.",
        "sweep_aggregated": _aggregate_sweep_across_seeds(all_seed_payloads),
    }
    root_metrics = {"run_config": config_dump, "out_root": str(out_root), "seeds": all_seed_payloads, "aggregate": aggregate}
    if save_metrics:
        (out_root / "metrics.json").write_text(json.dumps(root_metrics, indent=2, default=str) + "\n")
        (out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str) + "\n")

    out_print = {"aggregate": aggregate, "last_seed": seed_list[-1] if seed_list else None}
    print(json.dumps(out_print, indent=2, default=str), flush=True)
    if str(args.output_json).strip():
        Path(str(args.output_json).strip()).write_text(json.dumps(root_metrics, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
