#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        generate_samples_for_op_base_seq,
        verify_sample_seq,
    )
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers.cvx_carry_teacher_solve import (
        cvx_fit_shared_two_head,
        cvx_predict_two_head_ood,
        resolve_cvx_carry_loss_name,
        resolve_cvx_sum_loss_name,
    )
    from solvers.ste_carry_teacher_solve import (
        CarryAugmentedSNN,
        carry_teacher_forcing_token_metrics,
        resolve_ste_carry_loss_name,
        resolve_ste_sum_loss_name,
        ste_predict_sum_carry,
        ste_sweep_and_train,
    )
else:
    from .data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        generate_samples_for_op_base_seq,
        verify_sample_seq,
    )
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers.cvx_carry_teacher_solve import (
        cvx_fit_shared_two_head,
        cvx_predict_two_head_ood,
        resolve_cvx_carry_loss_name,
        resolve_cvx_sum_loss_name,
    )
    from .solvers.ste_carry_teacher_solve import (
        CarryAugmentedSNN,
        carry_teacher_forcing_token_metrics,
        resolve_ste_carry_loss_name,
        resolve_ste_sum_loss_name,
        ste_predict_sum_carry,
        ste_sweep_and_train,
    )

__all__ = ["main", "CarryAugmentedSNN", "build_carry_augmented_dataset"]


def _add_timesteps(n_digits: int) -> int:
    return int(n_digits) + 1


def _lc_dirname(lambda_carry: float) -> str:
    s = f"{float(lambda_carry):.10g}"
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return f"lambda_carry_{s}"


def _mean_std_list(vals: List[float]) -> Dict[str, float]:
    a = np.array([float(x) for x in vals], dtype=np.float64)
    if a.size < 1:
        raise ValueError("mean_std: empty list")
    if a.size == 1:
        return {"mean": float(a[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def _float_dict_mean_std(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    for d in dicts[1:]:
        if set(d.keys()) != set(keys):
            raise ValueError(f"float_dict key mismatch: {dicts[0].keys()} vs {d.keys()}")
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
    ood0 = ood_list[0]
    for ox in ood_list[1:]:
        if set(ox.keys()) != set(ood0.keys()):
            raise ValueError("OOD keys mismatch across seeds")
    out: Dict[str, Any] = {}
    for k in ood0:
        b0 = ood0[k]
        blocks = [ox[k] for ox in ood_list]
        out[k] = {
            "n_digits": b0.get("n_digits"),
            "n_test": b0.get("n_test"),
            "ste": _float_dict_mean_std([b["ste"] for b in blocks]),
            "cvx": _float_dict_mean_std([b["cvx"] for b in blocks]),
        }
    return out


def _aggregate_sweep_across_seeds(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
    sweeps = [sp["sweep"] for sp in seed_payloads]
    l0 = [float(e["lambda_carry"]) for e in sweeps[0]]
    for sw in sweeps[1:]:
        if [float(e["lambda_carry"]) for e in sw] != l0:
            raise ValueError("lambda_carry sweep order differs across seeds")
    agg: List[Dict[str, Any]] = []
    for i, lc in enumerate(l0):
        ents = [sw[i] for sw in sweeps]
        ood_ents = [e.get("ood_eval") for e in ents]
        agg.append(
            {
                "lambda_carry": float(lc),
                "ste": {
                    "test_metrics": _float_dict_mean_std([e["ste"]["test_metrics"] for e in ents]),
                },
                "cvx": {
                    "test_metrics": _float_dict_mean_std([e["cvx"]["test_metrics"] for e in ents]),
                    "diagnostics": _float_dict_mean_std([e["cvx"]["diagnostics"] for e in ents]),
                },
                "ood_eval": _aggregate_ood(ood_ents),
            }
        )
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
    X_full = np.stack([s.inputs.astype(np.float32) / max(base - 1, 1) for s in samples], axis=0)
    # Teacher-forced carry-in is already present in channel 2.
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
    train_samples = generate_samples_for_op_base_seq(
        "add", base, n_digits, n_train, seed + 11, add_initial_carry=add_initial_carry
    )
    val_samples = generate_samples_for_op_base_seq(
        "add", base, n_digits, n_val, seed + 29, add_initial_carry=add_initial_carry
    )
    test_samples = generate_samples_for_op_base_seq(
        "add", base, n_digits, n_test, seed + 47, add_initial_carry=add_initial_carry
    )
    X_train, y_sum_train, y_carry_train = _samples_to_xy_carry(train_samples, base, verify_count)
    X_val, y_sum_val, y_carry_val = _samples_to_xy_carry(val_samples, base, verify_count)
    X_test, y_sum_test, y_carry_test = _samples_to_xy_carry(test_samples, base, verify_count)
    tag = f"arith_carry_tf::base{base}::digits{n_digits}::ic_{add_initial_carry}"
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


def _make_ood_carry_tensors(
    *,
    base: int,
    n_digits: int,
    n_test: int,
    seed: int,
    verify_count: int,
    add_initial_carry: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = generate_samples_for_op_base_seq(
        "add",
        base,
        n_digits,
        n_test,
        seed,
        add_initial_carry=add_initial_carry,
    )
    return _samples_to_xy_carry(samples, base, verify_count)


def _ood_eval(
    *,
    ds: CarryAugmentedDataset,
    arith_base: int,
    ste_model: CarryAugmentedSNN,
    init_cfg: object,
    w_sum: np.ndarray,
    w_carry: np.ndarray,
    ood_digits: List[int],
    n_test_ood: int,
    seed: int,
    verify_count: int,
    add_initial_carry: str,
) -> Dict[str, Any]:
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
        sp, cp = ste_predict_sum_carry(ste_model, x_ood)
        s_c, c_c = cvx_predict_two_head_ood(
            w_sum=w_sum,
            w_carry=w_carry,
            x_train=ds.X_train,
            x_val=ds.X_val,
            x_ood=x_ood,
            init_cfg=init_cfg,
            base=arith_base,
        )
        out[f"n_digits_{nd}"] = {
            "n_digits": int(nd),
            "n_test": int(n_test_ood),
            "ste": carry_teacher_forcing_token_metrics(sp, cp, y_s, y_c),
            "cvx": carry_teacher_forcing_token_metrics(s_c, c_c, y_s, y_c),
        }
    return out


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
# CLI
# ------------------------------------------------------------

_TF_OBJECTIVE_CHOICES = ("joint", "lambda_weighted", "mean_pair", "lambda_normalized")
_STE_SUM_LOSS = ("auto", "hinge", "ce", "hinge_ovr")
_STE_CARRY_LOSS = ("auto", "hinge", "ce")
_CVX_SUM_LOSS = ("auto", "hinge", "ce", "hinge_ovr")
_CVX_CARRY_LOSS = ("auto", "hinge", "ce")
_ADD_INITIAL_CARRY = ("zero", "random")


def main() -> None:
    ap = argparse.ArgumentParser(description="Carry-augmented addition with teacher forcing and shared heads.")
    ap.add_argument("--arith_base", type=int, default=2, choices=SUPPORTED_BASES)
    ap.add_argument(
        "--n_digits",
        type=int,
        default=5,
        help="In-distribution train/val/test digit width (same as length-gen n_digits_train).",
    )
    ap.add_argument("--n_train", type=int, default=2304)
    ap.add_argument("--n_val", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument(
        "--n_test_ood",
        type=int,
        default=0,
        help="Test samples per OOD length; 0 = use --n_test.",
    )
    ap.add_argument(
        "--ood_digits",
        type=int,
        nargs="*",
        default=None,
        help="n_digits for OOD length eval; default 10 20 50. Empty list disables. Example: --ood_digits 10 20 30",
    )
    ap.add_argument(
        "--add_initial_carry",
        type=str,
        default="random",
        choices=_ADD_INITIAL_CARRY,
        help="Add op: 'zero' (carry into LSD = 0) or 'random' (Uniform[0, base) per sample) for all arith_base.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds: one full train/sweep/ood per seed, saved under seed_*/; aggregate.json uses mean+-std over seeds.",
    )
    ap.add_argument("--verify_samples", type=int, default=5)
    ap.add_argument("--debug_max_train", type=int, default=0, help="If >0, use only the first N training samples (debug / smoke).")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=256)
    ap.add_argument("--P_last", type=int, default=512)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--ste_last_layer_readout", type=str, default="membrane", choices=["membrane", "spike"])
    ap.add_argument("--cvx_last_layer_readout", type=str, default="spike", choices=["membrane", "spike"])
    ap.add_argument("--ste_epochs", type=int, default=100)
    ap.add_argument("--optimizer_name", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--batch_size", type=int, default=-1)
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--lambda_sum", type=float, default=1.0)
    ap.add_argument(
        "--lambda_carry",
        type=float,
        default=2.0,
        help="Default carry loss weight; ignored when --lambda_carry_grid is non-empty.",
    )
    ap.add_argument(
        "--lambda_carry_grid",
        type=float,
        nargs="*",
        default=None,
        help="If set, sweep over these values (e.g. 0.5 1.0 2.0 4.0). Each run re-trains STE+CVX.",
    )
    ap.add_argument(
        "--tf_objective",
        type=str,
        default="joint",
        choices=_TF_OBJECTIVE_CHOICES,
        help="STE only. joint=λ_s L_s + λ_c L_c + β R (default, historic). mean_pair or lambda_normalized: alternate aggregations of the same two heads and path reg.",
    )
    ap.add_argument(
        "--ste_time_loss",
        type=str,
        default="ramp",
        choices=("uniform", "ramp"),
        help="STE only. uniform: mean over (batch,time) as before. ramp: L_h = sum_t α_t l_h(t) / sum_t α_t with α_t=2t/(T+1), t=1..T, l_h(t)=batch-mean loss at t.",
    )
    ap.add_argument(
        "--cvx_time_loss",
        type=str,
        default="ramp",
        choices=("uniform", "ramp"),
        help="CVX only. Should match --ste_time_loss for fair comparison. uniform: (1/(nT)) sum flat rows. ramp: same α_t and row weights as STE on train + val (multiclass sum train stays unweighted softmax CE; val uses ramped CE).",
    )
    ap.add_argument(
        "--ste_sum_loss",
        type=str,
        default="auto",
        choices=_STE_SUM_LOSS,
        help="auto: base 2 -> hinge, base>2 -> ce.",
    )
    ap.add_argument("--ste_carry_loss", type=str, default="auto", choices=_STE_CARRY_LOSS)
    ap.add_argument(
        "--cvx_sum_loss",
        type=str,
        default="auto",
        choices=_CVX_SUM_LOSS,
        help="auto: base 2 -> hinge, base>2 -> ce. hinge_ovr: multiclass OVR hinge (base>2).",
    )
    ap.add_argument("--cvx_carry_loss", type=str, default="auto", choices=_CVX_CARRY_LOSS)
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=list(LR_GRID_DEFAULT))
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--cvx_device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Run directory. Empty = atomic/sweep_results/carry_teacher_forcing_..._TIMESTAMP. Writes seed_*/metrics.json, run_config.json, aggregate.json.",
    )
    ap.add_argument(
        "--no_save_metrics",
        action="store_true",
        help="Do not create directories or write JSON; still prints aggregate to stdout.",
    )
    args = ap.parse_args()

    if int(args.n_test_ood) == 0:
        n_test_ood = int(args.n_test)
    else:
        n_test_ood = int(args.n_test_ood)


    if args.ood_digits is None:
        ood_digits = [10, 20, 50]
    else:
        ood_digits = [int(x) for x in args.ood_digits]

    if args.lambda_carry_grid is not None and len(args.lambda_carry_grid) > 0:
        lambda_carry_sweep = [float(x) for x in args.lambda_carry_grid]
    else:
        lambda_carry_sweep = [float(args.lambda_carry)]

    seed_list = [int(s) for s in args.seeds]
    aic = str(args.add_initial_carry)

    atomic_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root).strip()).resolve()
    else:
        out_root = (
            atomic_dir
            / "sweep_results"
            / (
                f"carry_teacher_forcing_b{int(args.arith_base)}_L{int(args.L)}_"
                f"traind{int(args.n_digits)}_K{int(args.K_parallel)}_{stamp}"
            )
        )
    save_metrics = not bool(args.no_save_metrics)
    if save_metrics:
        out_root.mkdir(parents=True, exist_ok=True)

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
        "out_root": str(out_root),
    }
    if save_metrics:
        (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n")

    all_seed_payloads: List[Dict[str, Any]] = []

    for run_seed in seed_list:
        print("\n" + "=" * 80, flush=True)
        print(f"SEED {run_seed}", flush=True)
        print("=" * 80, flush=True)

        _set_seed(int(run_seed))
        ds = build_carry_augmented_dataset(
            base=int(args.arith_base),
            n_digits=int(args.n_digits),
            n_train=int(args.n_train),
            n_val=int(args.n_val),
            n_test=int(args.n_test),
            seed=int(run_seed),
            verify_count=int(args.verify_samples),
            add_initial_carry=aic,
        )
        ds = _maybe_subsample_train(ds, int(args.debug_max_train))
        if int(args.debug_max_train) > 0:
            print(
                f"[debug] using debug_max_train={int(args.debug_max_train)}: train_n={ds.X_train.shape[0]}",
                flush=True,
            )

        ste_n_sum = resolve_ste_sum_loss_name(str(args.ste_sum_loss), ds.num_sum_classes)
        ste_n_carry = resolve_ste_carry_loss_name(str(args.ste_carry_loss))
        cvx_n_sum = resolve_cvx_sum_loss_name(str(args.cvx_sum_loss), ds.num_sum_classes)
        cvx_n_carry = resolve_cvx_carry_loss_name(str(args.cvx_carry_loss))

        sweep_out: List[Dict[str, Any]] = []
        for lc in lambda_carry_sweep:
            print(f"[sweep] seed={run_seed} lambda_carry={lc}", flush=True)
            ste_model, ste_sel, ste_test = ste_sweep_and_train(
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
                seed=int(run_seed),
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lc),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
            )
            cvx_bundle, cvx_sel, cvx_test = cvx_fit_shared_two_head(
                ds=ds,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                seed=int(run_seed),
                beta_grid=args.cvx_beta_grid,
                bias_grid=args.cvx_bias_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lc),
                cvx_device=_resolve_cvx_device(str(args.cvx_device)),
                cvx_sum_loss=str(args.cvx_sum_loss),
                cvx_carry_loss=str(args.cvx_carry_loss),
                cvx_time_loss=str(args.cvx_time_loss),
            )
            w_sum = cvx_bundle["sum_weights"]
            w_carry = cvx_bundle["carry_weights"]
            init_cfg = cvx_bundle["init_cfg"]
            ood_report: Optional[Dict[str, Any]] = None
            if len(ood_digits) > 0:
                ood_report = _ood_eval(
                    ds=ds,
                    arith_base=int(args.arith_base),
                    ste_model=ste_model,
                    init_cfg=init_cfg,
                    w_sum=w_sum,
                    w_carry=w_carry,
                    ood_digits=ood_digits,
                    n_test_ood=n_test_ood,
                    seed=int(run_seed),
                    verify_count=int(args.verify_samples),
                    add_initial_carry=aic,
                )
                for k, block in ood_report.items():
                    s = block["ste"]["joint_token_acc"]
                    c = block["cvx"]["joint_token_acc"]
                    print(
                        f"  [ood] seed={run_seed} {k}  STE joint_token_acc={s:.4f}  CVX joint_token_acc={c:.4f}",
                        flush=True,
                    )
            one_entry: Dict[str, Any] = {
                "lambda_carry": float(lc),
                "ste": {"selected_params": ste_sel, "test_metrics": ste_test},
                "cvx": {
                    "selected_params": cvx_sel,
                    "diagnostics": {
                        "primal_value": float(cvx_bundle["primal_value"]),
                        "dual_value": float(cvx_bundle["dual_value"]),
                        "gap": float(cvx_bundle["gap"]),
                    },
                    "test_metrics": cvx_test,
                },
                "ood_eval": ood_report,
            }
            sweep_out.append(one_entry)
            if save_metrics:
                lc_dir = out_root / f"seed_{run_seed}" / _lc_dirname(float(lc))
                lc_dir.mkdir(parents=True, exist_ok=True)
                (lc_dir / "metrics.json").write_text(
                    json.dumps(one_entry, indent=2, default=str) + "\n"
                )
                print(f"[saved] {lc_dir / 'metrics.json'}", flush=True)

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
                "ste_sum_loss": str(args.ste_sum_loss),
                "ste_sum_loss_resolved": ste_n_sum,
                "ste_carry_loss": str(args.ste_carry_loss),
                "ste_carry_loss_resolved": ste_n_carry,
                "cvx_sum_loss": str(args.cvx_sum_loss),
                "cvx_sum_loss_resolved": cvx_n_sum,
                "cvx_carry_loss": str(args.cvx_carry_loss),
                "cvx_carry_loss_resolved": cvx_n_carry,
                "debug_max_train": int(args.debug_max_train),
            },
        }
        if len(sweep_out) == 1:
            seed_payload["ste"] = sweep_out[0]["ste"]
            seed_payload["cvx"] = sweep_out[0]["cvx"]
            seed_payload["ood_eval"] = sweep_out[0].get("ood_eval")
        if save_metrics:
            sdir = out_root / f"seed_{run_seed}"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "metrics.json").write_text(
                json.dumps(seed_payload, indent=2, default=str) + "\n"
            )
            print(f"[saved] {sdir / 'metrics.json'}", flush=True)
        all_seed_payloads.append(seed_payload)

    aggregate = {
        "n_seeds": int(len(all_seed_payloads)),
        "seeds": [int(s) for s in seed_list],
        "description": "For each float metric, mean and std (ddof=1) across seeds. Same lambda_carry order per seed required.",
        "sweep_aggregated": _aggregate_sweep_across_seeds(all_seed_payloads),
    }
    root_metrics = {
        "run_config": config_dump,
        "out_root": str(out_root),
        "seeds": all_seed_payloads,
        "aggregate": aggregate,
    }
    if save_metrics:
        (out_root / "metrics.json").write_text(
            json.dumps(root_metrics, indent=2, default=str) + "\n"
        )
        (out_root / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, default=str) + "\n"
        )
        print(f"[saved] {out_root / 'metrics.json'}", flush=True)
        print(f"[saved] {out_root / 'aggregate.json'}", flush=True)

    out_print = {
        "aggregate": aggregate,
        "last_seed": seed_list[-1] if seed_list else None,
    }
    text = json.dumps(out_print, indent=2, default=str)
    print(text, flush=True)
    if str(args.output_json).strip() != "":
        Path(str(args.output_json).strip()).write_text(json.dumps(root_metrics, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
