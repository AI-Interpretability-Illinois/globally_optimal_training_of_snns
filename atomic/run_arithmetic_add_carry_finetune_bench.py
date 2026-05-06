#!/usr/bin/env python3
from __future__ import annotations

"""
Addition carry hybrid fine-tune benchmark.

Per seed and per lambda_carry, this script runs both hybrid directions on disjoint train splits:
1. STE pretrain on split A.
2. CVX fit on split A from the pretrained STE hidden weights.
3. STE fine-tune on split B from the stage-1 STE weights.
4. CVX pretrain on split A from gaussian CVX initialization.
5. STE fine-tune on split B from the stage-4 CVX hidden+head weights.

For every stage, the script reports and saves by default:
- selected hyperparameters
- teacher-forcing ID / OOD metrics
- autoregressive ID / OOD metrics
- CVX diagnostics when applicable

By default (run_mode=full), no weights are saved.
For run_mode=pretrain_only, the script saves pretrain weights and exits.
For run_mode=finetune_only, the script only runs fine-tuning from saved pretrain weights.

If --out_root is omitted, results go under sweep_results/carry_hybrid_b{B}_d{D}_L{L}_metrics_{timestamp}.
STE and CVX pretrain checkpoints can share one out_root (seed_*/pretrain_weights/ste_lambda_*.npz vs
cvx_lambda_*.npz do not collide). Run configuration is only stored inside metrics JSON (run_config key),
not as a separate file in this directory. STE pretrain_only writes metrics_pretrain_only_ste.json;
CVX pretrain_only writes metrics_pretrain_only.json.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import SUPPORTED_BASES, generate_samples_for_op_base_seq, verify_sample_seq
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers import cvx_parallel_Solve as cvx_par
    from solvers import cvx_carry_teacher_solve as cvx_cts
    from solvers import ste_carry_teacher_solve as ste_cts
    from solvers.cvx_solve import InitializationConfig, _build_feature_map
else:
    from .data_loaders.arithmetic_data_loader import SUPPORTED_BASES, generate_samples_for_op_base_seq, verify_sample_seq
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers import cvx_parallel_Solve as cvx_par
    from .solvers import cvx_carry_teacher_solve as cvx_cts
    from .solvers import ste_carry_teacher_solve as ste_cts
    from .solvers.cvx_solve import InitializationConfig, _build_feature_map


# ------------------------------------------------------------
# small utils
# ------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_timesteps(n_digits: int) -> int:
    return int(n_digits) + 1


def _resolve_cvx_device(name: str) -> Optional[torch.device]:
    if str(name) == 'auto':
        return None
    if str(name) == 'cpu':
        return torch.device('cpu')
    if str(name) == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Requested cvx_device=cuda but CUDA is not available.')
        return torch.device('cuda')
    if str(name) == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise ValueError('Requested cvx_device=mps but MPS is not available.')
        return torch.device('mps')
    raise ValueError(f'Unknown cvx_device={name!r}')


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


# ------------------------------------------------------------
# data
# ------------------------------------------------------------


def _samples_to_xy_carry(samples: Sequence[Any], base: int, verify_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vc = min(int(verify_count), len(samples))
    for i in range(vc):
        verify_sample_seq(samples[i])
    scale = max(int(base) - 1, 1)
    x_full = np.stack([s.inputs.astype(np.float32) / float(scale) for s in samples], axis=0)
    x = x_full[:, :, :3].astype(np.float32, copy=False)
    y_sum = np.stack([s.target_tokens.astype(np.int64) for s in samples], axis=0)
    carry_in = np.stack([s.inputs[:, 2].astype(np.int64) for s in samples], axis=0)
    y_carry = np.zeros_like(y_sum, dtype=np.int64)
    y_carry[:, :-1] = carry_in[:, 1:]
    y_carry[:, -1] = 0
    return x, y_sum, y_carry


def build_carry_augmented_dataset_from_seeds(
    *,
    base: int,
    n_digits: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed_train: int,
    seed_val: int,
    seed_test: int,
    verify_count: int,
    add_initial_carry: str = 'random',
) -> CarryAugmentedDataset:
    tr_samples = generate_samples_for_op_base_seq('add', base, n_digits, n_train, seed_train, add_initial_carry=add_initial_carry)
    va_samples = generate_samples_for_op_base_seq('add', base, n_digits, n_val, seed_val, add_initial_carry=add_initial_carry)
    te_samples = generate_samples_for_op_base_seq('add', base, n_digits, n_test, seed_test, add_initial_carry=add_initial_carry)
    x_tr, y_sum_tr, y_carry_tr = _samples_to_xy_carry(tr_samples, base, verify_count)
    x_va, y_sum_va, y_carry_va = _samples_to_xy_carry(va_samples, base, verify_count)
    x_te, y_sum_te, y_carry_te = _samples_to_xy_carry(te_samples, base, verify_count)
    return CarryAugmentedDataset(
        X_train=x_tr,
        y_sum_train=y_sum_tr,
        y_carry_train=y_carry_tr,
        X_val=x_va,
        y_sum_val=y_sum_va,
        y_carry_val=y_carry_va,
        X_test=x_te,
        y_sum_test=y_sum_te,
        y_carry_test=y_carry_te,
        num_sum_classes=int(base),
        d_in=int(x_tr.shape[2]),
        T=int(x_tr.shape[1]),
        dataset_name=f'arith_carry_hybrid::base{base}::digits{n_digits}::ic_{add_initial_carry}',
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
    samples = generate_samples_for_op_base_seq('add', base, n_digits, n_test, seed, add_initial_carry=add_initial_carry)
    return _samples_to_xy_carry(samples, base, verify_count)


# ------------------------------------------------------------
# carry model weight extraction / init
# ------------------------------------------------------------


def _extract_carry_weight_list(model: ste_cts.CarryAugmentedSNN) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    for br in model.branches:
        for fc in br.fcs:
            weights.append(fc.weight.detach().cpu().numpy().copy())
    weights.append(model.sum_head.weight.detach().cpu().numpy().copy())
    weights.append(model.carry_head.weight.detach().cpu().numpy().copy())
    return weights


def _load_carry_weights_(model: ste_cts.CarryAugmentedSNN, weights: Sequence[np.ndarray]) -> None:
    expected_hidden = sum(len(br.fcs) for br in model.branches)
    if len(weights) < expected_hidden:
        raise ValueError(f'Need at least {expected_hidden} tensors, got {len(weights)}.')
    with torch.no_grad():
        ptr = 0
        for b_idx, br in enumerate(model.branches):
            for l_idx, fc in enumerate(br.fcs):
                src = torch.as_tensor(weights[ptr], dtype=fc.weight.dtype, device=fc.weight.device)
                if tuple(src.shape) != tuple(fc.weight.shape):
                    raise ValueError(
                        f'Hidden weight mismatch at branch={b_idx} layer={l_idx}: {tuple(src.shape)} != {tuple(fc.weight.shape)}'
                    )
                fc.weight.copy_(src)
                ptr += 1
        if len(weights) >= expected_hidden + 2:
            src_sum = torch.as_tensor(weights[expected_hidden], dtype=model.sum_head.weight.dtype, device=model.sum_head.weight.device)
            src_carry = torch.as_tensor(weights[expected_hidden + 1], dtype=model.carry_head.weight.dtype, device=model.carry_head.weight.device)
            if tuple(src_sum.shape) == tuple(model.sum_head.weight.shape):
                model.sum_head.weight.copy_(src_sum)
            if tuple(src_carry.shape) == tuple(model.carry_head.weight.shape):
                model.carry_head.weight.copy_(src_carry)


def _gaussian_cvx_hidden_weights_out_in(
    *,
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    seed: int,
    variant: str = 'standard',
) -> List[np.ndarray]:
    k = int(K_parallel)
    sub_p_rec = cvx_par._parallel_branch_width(int(P_rec), k, 'P_rec')
    sub_p_last = cvx_par._parallel_branch_width(int(P_last), k, 'P_last')
    branch_hidden_dims = cvx_par._hidden_dims_like_snn_p2(int(L), int(sub_p_rec), int(sub_p_last))
    rng = np.random.default_rng(int(seed))
    out: List[np.ndarray] = []
    for _ in range(k):
        in_dim = int(d_in)
        for h in branch_hidden_dims:
            w_in_out = cvx_par._sample_weight_matrix(rng, in_dim, int(h), str(variant))
            out.append(np.asarray(w_in_out.T, dtype=np.float64))
            in_dim = int(h)
    return out


def _cvx_bundle_to_carry_weights(
    *,
    bundle: Dict[str, Any],
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    base: int,
) -> List[np.ndarray]:
    init_cfg = bundle['init_cfg']
    expected_hidden = int(K_parallel) * len(cvx_par._hidden_dims_like_snn_p2(
        int(L),
        int(cvx_par._parallel_branch_width(int(P_rec), int(K_parallel), 'P_rec')),
        int(cvx_par._parallel_branch_width(int(P_last), int(K_parallel), 'P_last')),
    ))
    if str(init_cfg.mode) == 'pretraining':
        if init_cfg.pretrained_weights is None or len(init_cfg.pretrained_weights) < expected_hidden:
            raise ValueError('CVX pretraining bundle missing pretrained hidden weights.')
        hidden = [np.asarray(w, dtype=np.float64).copy() for w in init_cfg.pretrained_weights[:expected_hidden]]
    else:
        hidden = _gaussian_cvx_hidden_weights_out_in(
            d_in=int(d_in),
            L=int(L),
            P_rec=int(P_rec),
            P_last=int(P_last),
            K_parallel=int(K_parallel),
            seed=int(init_cfg.seed),
            variant=str(getattr(init_cfg, 'variant', 'standard')),
        )
    w_sum = np.asarray(bundle['sum_weights'], dtype=np.float64)
    w_carry = np.asarray(bundle['carry_weights'], dtype=np.float64)
    if int(base) == 2:
        sum_head = w_sum.reshape(1, -1)
    else:
        sum_head = w_sum.T
    carry_head = w_carry.reshape(1, -1)
    return hidden + [sum_head, carry_head]


def _eval_cvx_candidate_on_features(
    *,
    beta: float,
    d_train: np.ndarray,
    d_val: np.ndarray,
    y_sum_tr: np.ndarray,
    y_sum_va: np.ndarray,
    y_carry_tr: np.ndarray,
    y_carry_va: np.ndarray,
    sum_n: str,
    carry_n: str,
    num_sum_classes: int,
    n_va: int,
    T: int,
    sw_train: Optional[np.ndarray],
    a_val: Optional[np.ndarray],
    lambda_sum: float,
    lambda_carry: float,
    ovr_workers: int,
) -> Dict[str, Any]:
    rho_sum = float(beta) / max(float(lambda_sum), 1e-12)
    rho_carry = float(beta) / max(float(lambda_carry), 1e-12)

    if int(num_sum_classes) == 2:
        if str(sum_n) not in ('hinge', 'ce'):
            raise ValueError('For binary sum, cvx_sum_loss must resolve to hinge or ce.')
        y_pm1 = np.where(y_sum_tr == 1, 1.0, -1.0).astype(np.float64)
        sol = cvx_cts.solve_binary_l1_primal_dual(
            d_train, y_pm1, rho_sum,
            'hinge' if str(sum_n) == 'hinge' else 'ce',
            sample_weight=sw_train,
        )
        w_sum = np.asarray(sol.w, dtype=np.float64)
        p_sum, d_sum, g_sum = float(sol.primal_obj), float(sol.dual_obj), float(sol.gap)
        sum_scores_val = d_val @ w_sum
        if str(sum_n) == 'hinge':
            sum_val_loss = cvx_cts._binary_hinge_val_loss(sum_scores_val, y_sum_va) if a_val is None else cvx_cts._ramped_binary_val_loss(sum_scores_val, y_sum_va, n_va, T, kind='hinge')
        else:
            sum_val_loss = cvx_cts._binary_logistic_val_loss(sum_scores_val, y_sum_va) if a_val is None else cvx_cts._ramped_binary_val_loss(sum_scores_val, y_sum_va, n_va, T, kind='logistic')
    else:
        if str(sum_n) == 'ce':
            w_sum, p_sum, d_sum, g_sum = cvx_cts.solve_multiclass_softmax_ce_l1_primal_dual(
                d_train, y_sum_tr, rho_sum, int(num_sum_classes), sample_weight=sw_train,
            )
            w_sum = np.asarray(w_sum, dtype=np.float64)
            sum_scores_val = d_val @ w_sum
            sum_val_loss = cvx_cts.multiclass_ovr_cvx_data_loss('ce', sum_scores_val, y_sum_va) if a_val is None else cvx_cts._ramped_multiclass_ce_val(sum_scores_val, y_sum_va, n_va, T)
        elif str(sum_n) == 'hinge_ovr':
            if int(ovr_workers) > 1 and int(num_sum_classes) > 1:
                w_sum_col = np.zeros((d_train.shape[1], int(num_sum_classes)), dtype=np.float64)
                p_sum, d_sum = 0.0, 0.0
                def _solve_class(c_idx: int) -> Tuple[int, np.ndarray, float, float]:
                    y_bin = np.where(y_sum_tr == c_idx, 1.0, -1.0).astype(np.float64)
                    sol = cvx_cts.solve_binary_l1_primal_dual(d_train, y_bin, rho_sum, 'hinge', sample_weight=sw_train)
                    return c_idx, np.asarray(sol.w, dtype=np.float64), float(sol.primal_obj), float(sol.dual_obj)
                with ThreadPoolExecutor(max_workers=min(int(ovr_workers), int(num_sum_classes))) as ex:
                    for c_idx, w_c, p_c, d_c in (f.result() for f in as_completed([ex.submit(_solve_class, c) for c in range(int(num_sum_classes))])):
                        w_sum_col[:, c_idx] = w_c
                        p_sum += p_c
                        d_sum += d_c
                g_sum = float(p_sum - d_sum) if np.isfinite(d_sum) else float('nan')
                w_sum = w_sum_col
            else:
                w_sum, p_sum, d_sum, g_sum = cvx_cts._ovr_hinge_solve(d_train, y_sum_tr, rho_sum, int(num_sum_classes), sample_weight=sw_train)
                w_sum = np.asarray(w_sum, dtype=np.float64)
            sum_scores_val = d_val @ w_sum
            sum_val_loss = cvx_cts.multiclass_ovr_cvx_data_loss('hinge_ovr', sum_scores_val, y_sum_va) if a_val is None else cvx_cts._ramped_ovr_hinge_val(sum_scores_val, y_sum_va, a_val)
        else:
            raise ValueError('For base>2, cvx_sum_loss must resolve to ce or hinge_ovr.')

    sol_carry = cvx_cts.solve_binary_l1_primal_dual(
        d_train,
        np.where(y_carry_tr == 1, 1.0, -1.0).astype(np.float64),
        rho_carry,
        'hinge' if str(carry_n) == 'hinge' else 'ce',
        sample_weight=sw_train,
    )
    w_carry = np.asarray(sol_carry.w, dtype=np.float64)
    p_carry, d_carry, g_carry = float(sol_carry.primal_obj), float(sol_carry.dual_obj), float(sol_carry.gap)
    carry_scores_val = d_val @ w_carry
    if str(carry_n) == 'hinge':
        carry_val_loss = cvx_cts._binary_hinge_val_loss(carry_scores_val, y_carry_va) if a_val is None else cvx_cts._ramped_binary_val_loss(carry_scores_val, y_carry_va, n_va, T, kind='hinge')
    else:
        carry_val_loss = cvx_cts._binary_logistic_val_loss(carry_scores_val, y_carry_va) if a_val is None else cvx_cts._ramped_binary_val_loss(carry_scores_val, y_carry_va, n_va, T, kind='logistic')

    score = float(lambda_sum) * float(sum_val_loss) + float(lambda_carry) * float(carry_val_loss)
    return {
        'beta': float(beta),
        'score': float(score),
        'sum_weights': w_sum,
        'carry_weights': w_carry,
        'primal_value': float(lambda_sum) * float(p_sum) + float(lambda_carry) * float(p_carry),
        'dual_value': float(lambda_sum) * float(d_sum) + float(lambda_carry) * float(d_carry),
        'gap': float(lambda_sum) * float(g_sum) + float(lambda_carry) * float(g_carry),
    }


def _lambda_tag(value: float) -> str:
    raw = np.format_float_positional(float(value), trim='-')
    return raw.replace('.', 'p').replace('-', 'm')


def _pretrain_weights_path(out_root: Path, *, variant: str, seed: int, lambda_carry: float) -> Path:
    if str(variant) not in ('ste', 'cvx'):
        raise ValueError(f'Unknown pretrain variant={variant!r}.')
    return out_root / f'seed_{int(seed)}' / 'pretrain_weights' / f'{variant}_lambda_{_lambda_tag(float(lambda_carry))}.npz'


def _metrics_pretrain_only_filename(pretrain_variant: str) -> str:
    if str(pretrain_variant) == 'ste':
        return 'metrics_pretrain_only_ste.json'
    return 'metrics_pretrain_only.json'


def _save_weight_list_npz(path: Path, *, weights: Sequence[np.ndarray], metadata: Dict[str, Any]) -> None:
    if not weights:
        raise ValueError('Cannot save empty weight list.')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    for i, w in enumerate(weights):
        payload[f'w_{i:03d}'] = np.asarray(w, dtype=np.float64)
    payload['metadata_json'] = np.array(json.dumps(metadata), dtype=np.str_)
    np.savez(path, **payload)


def _load_weight_list_npz(path: Path) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'Pretrained weights not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        keys = sorted([k for k in data.files if k.startswith('w_')])
        if not keys:
            raise ValueError(f'No weight tensors found in: {path}')
        weights = [np.asarray(data[k], dtype=np.float64).copy() for k in keys]
        metadata: Dict[str, Any] = {}
        if 'metadata_json' in data.files:
            metadata_raw = data['metadata_json']
            if metadata_raw.shape != ():
                raise ValueError(f'Expected scalar metadata_json in: {path}')
            metadata = json.loads(str(metadata_raw.item()))
    return weights, metadata


# ------------------------------------------------------------
# training / cvx fit
# ------------------------------------------------------------


def ste_sweep_and_train_init(
    *,
    ds: CarryAugmentedDataset,
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
    ste_sum_loss: str = 'auto',
    ste_carry_loss: str = 'auto',
    tf_objective: str = 'joint',
    ste_time_loss: str = 'ramp',
    pretrained_weights: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[ste_cts.CarryAugmentedSNN, Dict[str, float], Dict[str, float]]:
    device = (
        torch.device('cuda')
        if torch.cuda.is_available()
        else (torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cpu'))
    )
    sum_n = ste_cts.resolve_ste_sum_loss_name(ste_sum_loss, ds.num_sum_classes)
    carry_n = ste_cts.resolve_ste_carry_loss_name(ste_carry_loss)

    best_score = float('inf')
    best_params: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, Any]] = None

    xtr = torch.tensor(ds.X_train, dtype=torch.float32, device=device)
    ys_tr = torch.tensor(ds.y_sum_train, dtype=torch.long, device=device)
    yc_tr = torch.tensor(ds.y_carry_train, dtype=torch.long, device=device)
    n = ds.X_train.shape[0]
    bs = n if batch_size == -1 or batch_size is None else int(batch_size)
    rng = np.random.default_rng(seed)

    for lr in ste_lr_grid:
        for beta in ste_beta_grid:
            _set_seed(seed)
            model = ste_cts.CarryAugmentedSNN(
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
            if pretrained_weights is not None:
                _load_carry_weights_(model, pretrained_weights)
            params = list(model.parameters())
            if optimizer_name.lower() == 'sgd':
                opt = torch.optim.SGD(params, lr=float(lr))
            else:
                opt = torch.optim.Adam(params, lr=float(lr))
            sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
            for _ in range(int(ste_epochs)):
                model.train()
                perm = np.arange(n) if bs >= n else rng.permutation(n)
                for start in range(0, n, bs):
                    idx = perm[start: start + bs]
                    xb = xtr[idx]
                    ysb = ys_tr[idx]
                    ycb = yc_tr[idx]
                    sum_logits, carry_logits = model(xb)
                    l_s, l_c = ste_cts.LossFunction.ste_carry_tf_data_losses_scalar(
                        sum_logits,
                        ysb,
                        carry_logits,
                        ycb,
                        base=ds.num_sum_classes,
                        sum_loss_name=sum_n,
                        carry_loss_name=carry_n,
                        ste_time_loss=str(ste_time_loss),
                    )
                    reg = ste_cts.carry_snn_path_reg(model) if float(beta) > 0.0 else torch.zeros((), device=device, dtype=l_s.dtype)
                    loss = ste_cts.LossFunction.carry_teacher_forcing_total(
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
                va = ste_cts._eval_ste(
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
                sched.step(va['loss_total'])
            val_metrics = ste_cts._eval_ste(
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
            if float(val_metrics['loss_total']) < best_score:
                best_score = float(val_metrics['loss_total'])
                best_params = {'lr': float(lr), 'beta': float(beta)}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None or best_params is None:
        raise RuntimeError('No STE candidate found.')

    _set_seed(seed)
    best_model = ste_cts.CarryAugmentedSNN(
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
    val_metrics = ste_cts._eval_ste(
        best_model,
        ds.X_val,
        ds.y_sum_val,
        ds.y_carry_val,
        lambda_sum=lambda_sum,
        lambda_carry=lambda_carry,
        beta_path_reg=float(best_params['beta']),
        sum_loss_name=sum_n,
        carry_loss_name=carry_n,
        tf_objective=tf_objective,
        ste_time_loss=str(ste_time_loss),
    )
    return best_model, best_params, val_metrics


def cvx_fit_shared_two_head_init(
    *,
    ds: CarryAugmentedDataset,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    cvx_last_layer_readout: str,
    seed: int,
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    lambda_sum: float,
    lambda_carry: float,
    cvx_device: Optional[torch.device],
    cvx_sum_loss: str = 'auto',
    cvx_carry_loss: str = 'auto',
    cvx_time_loss: str = 'ramp',
    init_mode: str = 'gaussian',
    pretrained_weights: Optional[Sequence[np.ndarray]] = None,
    cvx_grid_workers: int = 1,
    cvx_ovr_workers: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    _ = cvx_device
    best_score = float('inf')
    best_bundle: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, float]] = None

    y_sum_tr = ds.y_sum_train.reshape(-1).astype(np.int64)
    y_sum_va = ds.y_sum_val.reshape(-1).astype(np.int64)
    y_carry_tr = ds.y_carry_train.reshape(-1).astype(np.int64)
    y_carry_va = ds.y_carry_val.reshape(-1).astype(np.int64)

    sum_n = cvx_cts.resolve_cvx_sum_loss_name(cvx_sum_loss, ds.num_sum_classes)
    carry_n = cvx_cts.resolve_cvx_carry_loss_name(cvx_carry_loss)
    ctl = str(cvx_time_loss)
    n_tr, T = int(ds.y_sum_train.shape[0]), int(ds.y_sum_train.shape[1])
    n_va, T_va = int(ds.y_sum_val.shape[0]), int(ds.y_sum_val.shape[1])
    if T_va != T:
        raise ValueError(f'Train/val T mismatch: {T} vs {T_va}.')

    sw_train: Optional[np.ndarray]
    a_val: Optional[np.ndarray]
    if ctl == 'ramp':
        sw_train = cvx_cts._flat_ramp_row_weights(n_tr, T)
        wv = cvx_cts._flat_ramp_row_weights(n_va, T)
        a_val = wv / float(wv.sum())
    else:
        sw_train = None
        a_val = None

    grid_workers = int(cvx_grid_workers)
    ovr_workers = int(cvx_ovr_workers)
    if grid_workers <= 0:
        raise ValueError(f'cvx_grid_workers must be >= 1, got {grid_workers}.')
    if ovr_workers <= 0:
        raise ValueError(f'cvx_ovr_workers must be >= 1, got {ovr_workers}.')

    # Build features once per unique bias value — LIF forward pass is deterministic in bias,
    # not in beta, so recomputing it per-beta is pure waste.
    unique_biases = list(dict.fromkeys(float(b) for b in bias_grid))
    _pretrained_w = None if pretrained_weights is None else [np.asarray(w, dtype=np.float64) for w in pretrained_weights]
    bias_to_features: Dict[float, Tuple[np.ndarray, np.ndarray, InitializationConfig]] = {}
    for bias in unique_biases:
        init_cfg = InitializationConfig(
            mode=str(init_mode),
            seed=seed,
            L=L,
            P_rec=P_rec,
            P_last=P_last,
            K_parallel=K_parallel,
            feature_count=P_last,
            last_layer_readout=cvx_last_layer_readout,
            bias=float(bias),
            pretrained_weights=_pretrained_w,
        )
        d_train, d_val, _ = cvx_cts._build_cvx_features_for_all_timesteps(ds.X_train, ds.X_val, ds.X_test, init_cfg)
        bias_to_features[float(bias)] = (d_train, d_val, init_cfg)

    # Now all candidates are pure CVXPY solves — no serial LIF bottleneck inside.
    candidate_pairs = [(float(beta), float(bias)) for beta in beta_grid for bias in bias_grid]

    def _evaluate_candidate(beta: float, bias: float) -> Dict[str, Any]:
        d_train, d_val, init_cfg = bias_to_features[float(bias)]
        eval_payload = _eval_cvx_candidate_on_features(
            beta=float(beta),
            d_train=d_train,
            d_val=d_val,
            y_sum_tr=y_sum_tr,
            y_sum_va=y_sum_va,
            y_carry_tr=y_carry_tr,
            y_carry_va=y_carry_va,
            sum_n=str(sum_n),
            carry_n=str(carry_n),
            num_sum_classes=int(ds.num_sum_classes),
            n_va=int(n_va),
            T=int(T),
            sw_train=sw_train,
            a_val=a_val,
            lambda_sum=float(lambda_sum),
            lambda_carry=float(lambda_carry),
            ovr_workers=int(ovr_workers),
        )
        eval_payload['init_cfg'] = init_cfg
        eval_payload['bias'] = float(bias)
        return eval_payload

    if grid_workers == 1 or len(candidate_pairs) == 1:
        candidate_results = [_evaluate_candidate(beta, bias) for beta, bias in candidate_pairs]
    else:
        candidate_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(int(grid_workers), len(candidate_pairs))) as ex:
            futures = [ex.submit(_evaluate_candidate, beta, bias) for beta, bias in candidate_pairs]
            for fut in as_completed(futures):
                candidate_results.append(fut.result())

    for cand in candidate_results:
        if float(cand['score']) < best_score:
            best_score = float(cand['score'])
            best_params = {'beta': float(cand['beta']), 'bias': float(cand['bias'])}
            best_bundle = {
                'init_cfg': cand['init_cfg'],
                'sum_weights': cand['sum_weights'],
                'carry_weights': cand['carry_weights'],
                'primal_value': float(cand['primal_value']),
                'dual_value': float(cand['dual_value']),
                'gap': float(cand['gap']),
                'cvx_sum_loss': sum_n,
                'cvx_carry_loss': carry_n,
                'cvx_time_loss': ctl,
                'cvx_grid_workers': int(grid_workers),
                'cvx_ovr_workers': int(ovr_workers),
            }

    if best_bundle is None or best_params is None:
        raise RuntimeError('No CVX candidate found.')

    init_cfg = best_bundle['init_cfg']
    _, _, d_test = cvx_cts._build_cvx_features_for_all_timesteps(ds.X_train, ds.X_val, ds.X_test, init_cfg)
    w_sum = best_bundle['sum_weights']
    w_carry = best_bundle['carry_weights']
    sum_scores_test = d_test @ w_sum
    carry_scores_test = d_test @ w_carry
    n, t_dim = ds.y_sum_test.shape
    sum_pred, carry_pred = cvx_cts._decode_cvx_preds(sum_scores_test, carry_scores_test, base=ds.num_sum_classes, n=n, T=t_dim)
    test_metrics = ste_cts.carry_teacher_forcing_token_metrics(sum_pred, carry_pred, ds.y_sum_test, ds.y_carry_test)
    return best_bundle, best_params, test_metrics


# ------------------------------------------------------------
# eval modes
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
        'sum_token_acc': float(sum_ok.mean()),
        'carry_token_acc': float(carry_ok.mean()),
        'joint_token_acc': float(both_ok.mean()),
        'joint_seq_acc': float(both_ok.all(axis=1).mean()),
        'sum_seq_acc': float(sum_ok.all(axis=1).mean()),
        'carry_seq_acc': float(carry_ok.all(axis=1).mean()),
        'mean_first_wrong_sum_among_error_seq': float(np.mean(first_wrong_sum)) if first_wrong_sum else float(T),
        'mean_first_wrong_carry_among_error_seq': float(np.mean(first_wrong_carry)) if first_wrong_carry else float(T),
    }


def _build_cvx_features_for_eval(*, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, init_cfg: InitializationConfig) -> np.ndarray:
    if int(getattr(init_cfg, 'K_parallel', 1)) > 1:
        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        _, _, d_test, _ = cvx_par._build_feature_map(x_train, x_val, x_test, pic, all_timesteps=True)
    else:
        _, _, d_test, _ = _build_feature_map(x_train, x_val, x_test, init_cfg, all_timesteps=True)
    return d_test


def ste_predict_sum_carry_autoregressive(model: ste_cts.CarryAugmentedSNN, x_seq: np.ndarray, *, base: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _eval_stage_single_mode(
    *,
    eval_mode: str,
    ds_fit: CarryAugmentedDataset,
    ds_eval: CarryAugmentedDataset,
    ste_model: Optional[ste_cts.CarryAugmentedSNN] = None,
    cvx_bundle: Optional[Dict[str, Any]] = None,
    ood_digits: Sequence[int],
    n_test_ood: int,
    arith_base: int,
    eval_seed: int,
    verify_count: int,
    add_initial_carry: str,
) -> Dict[str, Any]:
    if (ste_model is None) == (cvx_bundle is None):
        raise ValueError('Provide exactly one of ste_model or cvx_bundle.')

    def _tf_metrics_ste(x: np.ndarray, y_sum: np.ndarray, y_carry: np.ndarray) -> Dict[str, float]:
        sp, cp = ste_cts.ste_predict_sum_carry(ste_model, x)  # type: ignore[arg-type]
        return ste_cts.carry_teacher_forcing_token_metrics(sp, cp, y_sum, y_carry)

    def _tf_metrics_cvx(x: np.ndarray, y_sum: np.ndarray, y_carry: np.ndarray) -> Dict[str, float]:
        sum_pred, carry_pred = cvx_cts.cvx_predict_two_head_ood(
            w_sum=cvx_bundle['sum_weights'],  # type: ignore[index]
            w_carry=cvx_bundle['carry_weights'],  # type: ignore[index]
            x_train=ds_fit.X_train,
            x_val=ds_fit.X_val,
            x_ood=x,
            init_cfg=cvx_bundle['init_cfg'],  # type: ignore[index]
            base=int(arith_base),
        )
        return ste_cts.carry_teacher_forcing_token_metrics(sum_pred, carry_pred, y_sum, y_carry)

    def _ar_metrics_ste(x: np.ndarray, y_sum: np.ndarray, y_carry: np.ndarray) -> Dict[str, float]:
        sp, cp = ste_predict_sum_carry_autoregressive(ste_model, x, base=int(arith_base))  # type: ignore[arg-type]
        return carry_autoregressive_metrics(sp, cp, y_sum, y_carry)

    def _ar_metrics_cvx(x: np.ndarray, y_sum: np.ndarray, y_carry: np.ndarray) -> Dict[str, float]:
        sp, cp = cvx_predict_sum_carry_autoregressive(
            w_sum=cvx_bundle['sum_weights'],  # type: ignore[index]
            w_carry=cvx_bundle['carry_weights'],  # type: ignore[index]
            x_train=ds_fit.X_train,
            x_val=ds_fit.X_val,
            x_seq=x,
            init_cfg=cvx_bundle['init_cfg'],  # type: ignore[index]
            base=int(arith_base),
        )
        return carry_autoregressive_metrics(sp, cp, y_sum, y_carry)

    metric_fn = _tf_metrics_ste if ste_model is not None else _tf_metrics_cvx
    if str(eval_mode) == 'autoregressive':
        metric_fn = _ar_metrics_ste if ste_model is not None else _ar_metrics_cvx

    out: Dict[str, Any] = {
        'id_metrics': metric_fn(ds_eval.X_test, ds_eval.y_sum_test, ds_eval.y_carry_test),
        'ood_eval': {},
    }
    for nd in ood_digits:
        x_ood, y_s, y_c = _make_ood_carry_tensors(
            base=arith_base,
            n_digits=int(nd),
            n_test=int(n_test_ood),
            seed=int(eval_seed) + 10_000 + int(nd) * 97,
            verify_count=verify_count,
            add_initial_carry=add_initial_carry,
        )
        out['ood_eval'][f'n_digits_{int(nd)}'] = {
            'n_digits': int(nd),
            'n_test': int(n_test_ood),
            'metrics': metric_fn(x_ood, y_s, y_c),
        }
    return out


def _eval_stage_all_modes(**kwargs: Any) -> Dict[str, Any]:
    return {
        'teacher_forcing': _eval_stage_single_mode(eval_mode='teacher_forcing', **kwargs),
        'autoregressive': _eval_stage_single_mode(eval_mode='autoregressive', **kwargs),
    }


# ------------------------------------------------------------
# aggregation
# ------------------------------------------------------------


def _mean_std_list(vals: List[float]) -> Dict[str, float]:
    a = np.array([float(x) for x in vals], dtype=np.float64)
    if a.size < 1:
        raise ValueError('mean_std: empty list')
    if a.size == 1:
        return {'mean': float(a[0]), 'std': 0.0, 'n': 1}
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)), 'n': int(a.size)}


def _float_dict_mean_std(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = _mean_std_list([float(v) for v in vals])
        else:
            out[k] = vals[0]
    return out


def _aggregate_eval_mode_payload(mode_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not mode_payloads:
        return {}
    id_metrics = _float_dict_mean_std([sp['id_metrics'] for sp in mode_payloads])
    ood0 = mode_payloads[0]['ood_eval']
    ood_agg: Dict[str, Any] = {}
    for key in ood0:
        blocks = [sp['ood_eval'][key] for sp in mode_payloads]
        ood_agg[key] = {
            'n_digits': blocks[0]['n_digits'],
            'n_test': blocks[0]['n_test'],
            'metrics': _float_dict_mean_std([b['metrics'] for b in blocks]),
        }
    return {'id_metrics': id_metrics, 'ood_eval': ood_agg}


def _aggregate_stage(stage_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stage_payloads:
        return {}
    return {
        'teacher_forcing': _aggregate_eval_mode_payload([sp['teacher_forcing'] for sp in stage_payloads]),
        'autoregressive': _aggregate_eval_mode_payload([sp['autoregressive'] for sp in stage_payloads]),
    }


def _aggregate_lambda_sweep(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
    base_grid = [float(e['lambda_carry']) for e in seed_payloads[0]['lambda_sweep']]
    for sp in seed_payloads[1:]:
        if [float(e['lambda_carry']) for e in sp['lambda_sweep']] != base_grid:
            raise ValueError('lambda_carry sweep order differs across seeds')
    out: List[Dict[str, Any]] = []
    for idx, lc in enumerate(base_grid):
        entries = [sp['lambda_sweep'][idx] for sp in seed_payloads]
        out.append({
            'lambda_carry': float(lc),
            'ste_pretrain': _aggregate_stage([e['ste_pretrain'] for e in entries]),
            'cvx_from_ste_pretrain': _aggregate_stage([e['cvx_from_ste_pretrain'] for e in entries]),
            'ste_finetune_from_ste_pretrain_new_train': _aggregate_stage([e['ste_finetune_from_ste_pretrain_new_train'] for e in entries]),
            'cvx_pretrain': _aggregate_stage([e['cvx_pretrain'] for e in entries]),
            'ste_finetune_from_cvx_pretrain_new_train': _aggregate_stage([e['ste_finetune_from_cvx_pretrain_new_train'] for e in entries]),
        })
    return out


# ------------------------------------------------------------
# run-mode helpers
# ------------------------------------------------------------


def _build_pretrain_dataset(args: argparse.Namespace, *, pre_seed: int, eval_seed: int) -> CarryAugmentedDataset:
    return build_carry_augmented_dataset_from_seeds(
        base=int(args.arith_base),
        n_digits=int(args.n_digits),
        n_train=int(args.n_train_pre),
        n_val=int(args.n_val_pre),
        n_test=int(args.n_test),
        seed_train=int(pre_seed) + 11,
        seed_val=int(pre_seed) + 29,
        seed_test=int(eval_seed) + 47,
        verify_count=int(args.verify_samples),
        add_initial_carry=str(args.add_initial_carry),
    )


def _build_finetune_dataset(args: argparse.Namespace, *, ft_seed: int, eval_seed: int) -> CarryAugmentedDataset:
    return build_carry_augmented_dataset_from_seeds(
        base=int(args.arith_base),
        n_digits=int(args.n_digits),
        n_train=int(args.n_train_ft),
        n_val=int(args.n_val_ft),
        n_test=int(args.n_test),
        seed_train=int(ft_seed) + 11,
        seed_val=int(ft_seed) + 29,
        seed_test=int(eval_seed) + 47,
        verify_count=int(args.verify_samples),
        add_initial_carry=str(args.add_initial_carry),
    )


def _run_pretrain_only(args: argparse.Namespace, *, out_root: Path, cvx_device: Optional[torch.device], config_dump: Dict[str, Any]) -> Dict[str, Any]:
    seed_payloads: List[Dict[str, Any]] = []
    total_saved = 0

    for base_seed in args.seeds:
        pre_seed = int(base_seed)
        eval_seed = int(base_seed) + int(args.eval_seed_offset)
        ds_pre = _build_pretrain_dataset(args, pre_seed=pre_seed, eval_seed=eval_seed)
        lambda_payloads: List[Dict[str, Any]] = []

        for lambda_carry in [float(x) for x in args.lambda_carry_grid]:
            if str(args.pretrain_variant) == 'ste':
                ste_pre_model, ste_pre_sel, _ = ste_sweep_and_train_init(
                    ds=ds_pre,
                    L=int(args.L),
                    P_rec=int(args.P_rec),
                    P_last=int(args.P_last),
                    K_parallel=int(args.K_parallel),
                    ste_last_layer_readout=str(args.ste_last_layer_readout),
                    ste_epochs=int(args.ste_pretrain_epochs),
                    batch_size=int(args.batch_size),
                    optimizer_name=str(args.optimizer_name),
                    beta_leak=float(args.beta_leak),
                    threshold=float(args.threshold),
                    seed=pre_seed,
                    ste_lr_grid=args.ste_lr_grid,
                    ste_beta_grid=args.ste_beta_grid,
                    lambda_sum=float(args.lambda_sum),
                    lambda_carry=float(lambda_carry),
                    ste_sum_loss=str(args.ste_sum_loss),
                    ste_carry_loss=str(args.ste_carry_loss),
                    tf_objective=str(args.tf_objective),
                    ste_time_loss=str(args.ste_time_loss),
                    pretrained_weights=None,
                )
                weights = _extract_carry_weight_list(ste_pre_model)
                selected_params = ste_pre_sel
            elif str(args.pretrain_variant) == 'cvx':
                cvx_pre_bundle, cvx_pre_sel, _ = cvx_fit_shared_two_head_init(
                    ds=ds_pre,
                    L=int(args.L),
                    P_rec=int(args.P_rec),
                    P_last=int(args.P_last),
                    K_parallel=int(args.K_parallel),
                    cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                    seed=pre_seed,
                    beta_grid=args.cvx_beta_grid,
                    bias_grid=args.cvx_bias_grid,
                    lambda_sum=float(args.lambda_sum),
                    lambda_carry=float(lambda_carry),
                    cvx_device=cvx_device,
                    cvx_sum_loss=str(args.cvx_sum_loss),
                    cvx_carry_loss=str(args.cvx_carry_loss),
                    cvx_time_loss=str(args.cvx_time_loss),
                    init_mode='gaussian',
                    pretrained_weights=None,
                    cvx_grid_workers=int(args.cvx_grid_workers),
                    cvx_ovr_workers=int(args.cvx_ovr_workers),
                )
                weights = _cvx_bundle_to_carry_weights(
                    bundle=cvx_pre_bundle,
                    d_in=ds_pre.d_in,
                    L=int(args.L),
                    P_rec=int(args.P_rec),
                    P_last=int(args.P_last),
                    K_parallel=int(args.K_parallel),
                    base=int(args.arith_base),
                )
                selected_params = cvx_pre_sel
            else:
                raise ValueError(f'Unknown pretrain_variant={args.pretrain_variant!r}')

            weights_path = _pretrain_weights_path(
                out_root,
                variant=str(args.pretrain_variant),
                seed=int(base_seed),
                lambda_carry=float(lambda_carry),
            )
            metadata = {
                'seed': int(base_seed),
                'lambda_carry': float(lambda_carry),
                'pretrain_variant': str(args.pretrain_variant),
                'selected_params': selected_params,
                'arith_base': int(args.arith_base),
                'n_digits': int(args.n_digits),
                'L': int(args.L),
                'P_rec': int(args.P_rec),
                'P_last': int(args.P_last),
                'K_parallel': int(args.K_parallel),
            }
            _save_weight_list_npz(weights_path, weights=weights, metadata=metadata)
            total_saved += 1

            lambda_payloads.append({
                'lambda_carry': float(lambda_carry),
                'selected_params': selected_params,
                'weights_path': str(weights_path),
                'weight_tensors_saved': int(len(weights)),
            })

        seed_payload = {
            'seed': int(base_seed),
            'split_seeds': {
                'pretrain_train': pre_seed + 11,
                'pretrain_val': pre_seed + 29,
                'eval_test': eval_seed + 47,
            },
            'pretrain_variant': str(args.pretrain_variant),
            'lambda_sweep': lambda_payloads,
        }
        seed_payloads.append(seed_payload)
        sdir = out_root / f'seed_{int(base_seed)}'
        sdir.mkdir(parents=True, exist_ok=True)
        _mpre = _metrics_pretrain_only_filename(str(args.pretrain_variant))
        (sdir / _mpre).write_text(json.dumps(seed_payload, indent=2, default=str) + '\n')

    root_payload = {
        'run_config': config_dump,
        'out_root': str(out_root),
        'mode': 'pretrain_only',
        'pretrain_variant': str(args.pretrain_variant),
        'n_saved_weight_files': int(total_saved),
        'seeds': seed_payloads,
    }
    _mpre_root = _metrics_pretrain_only_filename(str(args.pretrain_variant))
    (out_root / _mpre_root).write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
    return root_payload


def _aggregate_finetune_only(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
    base_grid = [float(e['lambda_carry']) for e in seed_payloads[0]['lambda_sweep']]
    for sp in seed_payloads[1:]:
        if [float(e['lambda_carry']) for e in sp['lambda_sweep']] != base_grid:
            raise ValueError('lambda_carry sweep order differs across seeds')
    out: List[Dict[str, Any]] = []
    for idx, lc in enumerate(base_grid):
        ft_entries = [sp['lambda_sweep'][idx]['ste_finetune'] for sp in seed_payloads]
        pre_entries = [sp['lambda_sweep'][idx]['pretrain_eval'] for sp in seed_payloads]
        out.append({
            'lambda_carry': float(lc),
            'pretrain_eval': _aggregate_stage(pre_entries),
            'ste_finetune': _aggregate_stage(ft_entries),
        })
    return out


def _run_finetune_only(args: argparse.Namespace, *, out_root: Path, config_dump: Dict[str, Any]) -> Dict[str, Any]:
    if str(args.finetune_variant) == 'ste_from_ste':
        src_variant = 'ste'
    elif str(args.finetune_variant) == 'ste_from_cvx':
        src_variant = 'cvx'
    else:
        raise ValueError(f'Unknown finetune_variant={args.finetune_variant!r}')

    seed_payloads: List[Dict[str, Any]] = []
    for base_seed in args.seeds:
        pre_seed = int(base_seed)
        ft_seed = int(base_seed) + int(args.finetune_seed_offset)
        eval_seed = int(base_seed) + int(args.eval_seed_offset)
        ds_ft = _build_finetune_dataset(args, ft_seed=ft_seed, eval_seed=eval_seed)
        lambda_payloads: List[Dict[str, Any]] = []

        for lambda_carry in [float(x) for x in args.lambda_carry_grid]:
            weights_path = _pretrain_weights_path(
                out_root,
                variant=src_variant,
                seed=int(base_seed),
                lambda_carry=float(lambda_carry),
            )
            pretrained_weights, pretrain_metadata = _load_weight_list_npz(weights_path)

            # Evaluate the loaded pretrain model BEFORE fine-tuning. We rebuild the carry
            # SNN, copy weights in (works for both ste and cvx pretrain because
            # _cvx_bundle_to_carry_weights wrote the npz in carry layout), and run the same
            # teacher-forcing + autoregressive ID/OOD eval used for finetune. The readout
            # matches the pretrain variant so this measures what the pretrain model actually
            # does, not what an STE retraining would do.
            pretrain_readout = (
                str(args.cvx_last_layer_readout)
                if src_variant == 'cvx'
                else str(args.ste_last_layer_readout)
            )
            pretrain_model = ste_cts.CarryAugmentedSNN(
                d_in=ds_ft.d_in,
                base=ds_ft.num_sum_classes,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                last_layer_readout=pretrain_readout,
            )
            _load_carry_weights_(pretrain_model, pretrained_weights)
            pretrain_eval = _eval_stage_all_modes(
                ds_fit=ds_ft,
                ds_eval=ds_ft,
                ste_model=pretrain_model,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )

            ste_ft_model, ste_ft_sel, _ = ste_sweep_and_train_init(
                ds=ds_ft,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=int(args.ste_finetune_epochs),
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=ft_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=pretrained_weights,
            )
            ste_ft_eval = _eval_stage_all_modes(
                ds_fit=ds_ft,
                ds_eval=ds_ft,
                ste_model=ste_ft_model,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )
            lambda_payloads.append({
                'lambda_carry': float(lambda_carry),
                'loaded_pretrain_variant': src_variant,
                'loaded_weights_path': str(weights_path),
                'loaded_weights_metadata': pretrain_metadata,
                'pretrain_eval': {
                    'pretrain_variant': src_variant,
                    'pretrain_last_layer_readout': pretrain_readout,
                    'selected_params': pretrain_metadata.get('selected_params'),
                    **pretrain_eval,
                },
                'ste_finetune': {
                    'selected_params': ste_ft_sel,
                    **ste_ft_eval,
                },
            })

        seed_payload = {
            'seed': int(base_seed),
            'split_seeds': {
                'expected_pretrain_train': pre_seed + 11,
                'expected_pretrain_val': pre_seed + 29,
                'finetune_train': ft_seed + 11,
                'finetune_val': ft_seed + 29,
                'eval_test': eval_seed + 47,
            },
            'finetune_variant': str(args.finetune_variant),
            'lambda_sweep': lambda_payloads,
        }
        seed_payloads.append(seed_payload)
        sdir = out_root / f'seed_{int(base_seed)}'
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / 'metrics_finetune_only.json').write_text(json.dumps(seed_payload, indent=2, default=str) + '\n')

    aggregate = {
        'n_seeds': int(len(seed_payloads)),
        'lambda_sweep': _aggregate_finetune_only(seed_payloads),
    }
    root_payload = {
        'run_config': config_dump,
        'out_root': str(out_root),
        'mode': 'finetune_only',
        'finetune_variant': str(args.finetune_variant),
        'loaded_pretrain_variant': src_variant,
        'seeds': seed_payloads,
        'aggregate': aggregate,
    }
    (out_root / 'metrics_finetune_only.json').write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
    return root_payload


# ------------------------------------------------------------
# main
# ------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description='Addition carry hybrid fine-tune benchmark (both directions, both eval modes, lambda sweep).')
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--arith_base', type=int, default=5)
    ap.add_argument('--n_digits', type=int, default=5)
    ap.add_argument('--n_train_pre', type=int, default=2304)
    ap.add_argument('--n_val_pre', type=int, default=512)
    ap.add_argument('--n_train_ft', type=int, default=2304)
    ap.add_argument('--n_val_ft', type=int, default=512)
    ap.add_argument('--n_test', type=int, default=1024)
    ap.add_argument('--n_test_ood', type=int, default=1024)
    ap.add_argument('--ood_digits', type=int, nargs='*', default=[10, 25, 50])
    ap.add_argument('--verify_samples', type=int, default=5)
    ap.add_argument('--finetune_seed_offset', type=int, default=1000)
    ap.add_argument('--eval_seed_offset', type=int, default=2000)
    ap.add_argument('--add_initial_carry', choices=['zero', 'random'], default='random')

    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--P_rec', type=int, default=256)
    ap.add_argument('--P_last', type=int, default=512)
    ap.add_argument('--K_parallel', type=int, default=2)
    ap.add_argument('--beta_leak', type=float, default=0.99)
    ap.add_argument('--threshold', type=float, default=1.0)
    ap.add_argument('--ste_last_layer_readout', choices=['membrane', 'spike'], default='spike')
    ap.add_argument('--cvx_last_layer_readout', choices=['membrane', 'spike'], default='spike')
    ap.add_argument('--optimizer_name', choices=['adam', 'sgd'], default='adam')

    ap.add_argument('--ste_pretrain_epochs', type=int, default=100)
    ap.add_argument('--ste_finetune_epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=-1)
    ap.add_argument('--lambda_sum', type=float, default=1.0)
    ap.add_argument('--lambda_carry_grid', type=float, nargs='*', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0])
    ap.add_argument('--ste_lr_grid', type=float, nargs='*', default=list(LR_GRID_DEFAULT))
    ap.add_argument('--ste_beta_grid', type=float, nargs='*', default=list(BETA_GRID_DEFAULT))
    ap.add_argument('--cvx_beta_grid', type=float, nargs='*', default=list(BETA_GRID_DEFAULT))
    ap.add_argument('--cvx_bias_grid', type=float, nargs='*', default=[0.0])
    ap.add_argument('--cvx_device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    ap.add_argument('--ste_sum_loss', choices=['auto', 'hinge', 'ce', 'hinge_ovr'], default='auto')
    ap.add_argument('--ste_carry_loss', choices=['auto', 'hinge', 'ce'], default='auto')
    ap.add_argument('--cvx_sum_loss', choices=['auto', 'hinge', 'ce', 'hinge_ovr'], default='auto')
    ap.add_argument('--cvx_carry_loss', choices=['auto', 'hinge', 'ce'], default='auto')
    ap.add_argument('--tf_objective', choices=['joint', 'lambda_weighted', 'mean_pair', 'lambda_normalized'], default='joint')
    ap.add_argument('--ste_time_loss', choices=['uniform', 'ramp'], default='ramp')
    ap.add_argument('--cvx_time_loss', choices=['uniform', 'ramp'], default='ramp')
    ap.add_argument('--cvx_grid_workers', type=int, default=1)
    ap.add_argument('--cvx_ovr_workers', type=int, default=1)
    ap.add_argument('--run_mode', choices=['full', 'pretrain_only', 'finetune_only'], default='full')
    ap.add_argument('--pretrain_variant', choices=['ste', 'cvx'], default='ste')
    ap.add_argument('--finetune_variant', choices=['ste_from_ste', 'ste_from_cvx'], default='ste_from_ste')

    ap.add_argument(
        '--out_root',
        type=str,
        default='',
        help=(
            'Result directory (metrics JSON, checkpoints under seed_*, no standalone run_config file). '
            'If empty, uses sweep_results/carry_hybrid_b{B}_d{D}_L{L}_metrics_{timestamp} under cwd. '
            'Use the same path for STE and CVX pretrain_only to store both weight types in one tree. '
            'Run settings are embedded under run_config in each metrics JSON.'
        ),
    )
    ap.add_argument('--output_json', type=str, default='')
    args = ap.parse_args()

    if int(args.arith_base) not in SUPPORTED_BASES:
        raise ValueError(f'Unsupported base={args.arith_base}. Supported: {SUPPORTED_BASES}.')
    if not args.lambda_carry_grid:
        raise ValueError('lambda_carry_grid must be non-empty.')
    if int(args.cvx_grid_workers) <= 0:
        raise ValueError(f'cvx_grid_workers must be >=1, got {args.cvx_grid_workers}.')
    if int(args.cvx_ovr_workers) <= 0:
        raise ValueError(f'cvx_ovr_workers must be >=1, got {args.cvx_ovr_workers}.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root)).expanduser().resolve()
    else:
        out_root = Path.cwd() / 'sweep_results' / (
            f'carry_hybrid_b{int(args.arith_base)}_d{int(args.n_digits)}'
            f'_L{int(args.L)}_metrics_{stamp}'
        )
    out_root.mkdir(parents=True, exist_ok=True)

    cvx_device = _resolve_cvx_device(str(args.cvx_device))
    if str(args.run_mode) == 'pretrain_only':
        active_stages = [f'{str(args.pretrain_variant)}_pretrain']
    elif str(args.run_mode) == 'finetune_only':
        active_stages = [f'{str(args.finetune_variant)}_finetune']
    else:
        active_stages = [
            'ste_pretrain',
            'cvx_from_ste_pretrain',
            'ste_finetune_from_ste_pretrain_new_train',
            'cvx_pretrain',
            'ste_finetune_from_cvx_pretrain_new_train',
        ]

    config_dump = {
        'seeds': list(args.seeds),
        'arith_base': int(args.arith_base),
        'n_digits': int(args.n_digits),
        'T': _add_timesteps(int(args.n_digits)),
        'n_train_pre': int(args.n_train_pre),
        'n_val_pre': int(args.n_val_pre),
        'n_train_ft': int(args.n_train_ft),
        'n_val_ft': int(args.n_val_ft),
        'n_test': int(args.n_test),
        'n_test_ood': int(args.n_test_ood),
        'ood_digits': list(args.ood_digits),
        'add_initial_carry': str(args.add_initial_carry),
        'finetune_seed_offset': int(args.finetune_seed_offset),
        'eval_seed_offset': int(args.eval_seed_offset),
        'L': int(args.L),
        'P_rec': int(args.P_rec),
        'P_last': int(args.P_last),
        'K_parallel': int(args.K_parallel),
        'ste_last_layer_readout': str(args.ste_last_layer_readout),
        'cvx_last_layer_readout': str(args.cvx_last_layer_readout),
        'optimizer_name': str(args.optimizer_name),
        'ste_pretrain_epochs': int(args.ste_pretrain_epochs),
        'ste_finetune_epochs': int(args.ste_finetune_epochs),
        'lambda_sum': float(args.lambda_sum),
        'lambda_carry_grid': [float(x) for x in args.lambda_carry_grid],
        'ste_lr_grid': list(args.ste_lr_grid),
        'ste_beta_grid': list(args.ste_beta_grid),
        'cvx_beta_grid': list(args.cvx_beta_grid),
        'cvx_bias_grid': list(args.cvx_bias_grid),
        'ste_sum_loss': str(args.ste_sum_loss),
        'ste_carry_loss': str(args.ste_carry_loss),
        'cvx_sum_loss': str(args.cvx_sum_loss),
        'cvx_carry_loss': str(args.cvx_carry_loss),
        'tf_objective': str(args.tf_objective),
        'ste_time_loss': str(args.ste_time_loss),
        'cvx_time_loss': str(args.cvx_time_loss),
        'cvx_grid_workers': int(args.cvx_grid_workers),
        'cvx_ovr_workers': int(args.cvx_ovr_workers),
        'run_mode': str(args.run_mode),
        'pretrain_variant': str(args.pretrain_variant),
        'finetune_variant': str(args.finetune_variant),
        'evaluation_modes_saved_by_default': ['teacher_forcing', 'autoregressive'],
        'stages': active_stages,
    }

    if str(args.run_mode) == 'pretrain_only':
        root_payload = _run_pretrain_only(args, out_root=out_root, cvx_device=cvx_device, config_dump=config_dump)
        if str(args.output_json).strip():
            Path(str(args.output_json)).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
        print(json.dumps({
            'mode': 'pretrain_only',
            'pretrain_variant': str(args.pretrain_variant),
            'n_saved_weight_files': int(root_payload['n_saved_weight_files']),
            'out_root': str(out_root),
        }, indent=2, default=str), flush=True)
        return

    if str(args.run_mode) == 'finetune_only':
        root_payload = _run_finetune_only(args, out_root=out_root, config_dump=config_dump)
        if str(args.output_json).strip():
            Path(str(args.output_json)).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
        print(json.dumps({
            'mode': 'finetune_only',
            'finetune_variant': str(args.finetune_variant),
            'loaded_pretrain_variant': str(root_payload['loaded_pretrain_variant']),
            'aggregate': root_payload['aggregate'],
            'out_root': str(out_root),
        }, indent=2, default=str), flush=True)
        return

    seed_payloads: List[Dict[str, Any]] = []

    for base_seed in args.seeds:
        pre_seed = int(base_seed)
        ft_seed = int(base_seed) + int(args.finetune_seed_offset)
        eval_seed = int(base_seed) + int(args.eval_seed_offset)
        ds_pre = build_carry_augmented_dataset_from_seeds(
            base=int(args.arith_base),
            n_digits=int(args.n_digits),
            n_train=int(args.n_train_pre),
            n_val=int(args.n_val_pre),
            n_test=int(args.n_test),
            seed_train=pre_seed + 11,
            seed_val=pre_seed + 29,
            seed_test=eval_seed + 47,
            verify_count=int(args.verify_samples),
            add_initial_carry=str(args.add_initial_carry),
        )
        ds_ft = build_carry_augmented_dataset_from_seeds(
            base=int(args.arith_base),
            n_digits=int(args.n_digits),
            n_train=int(args.n_train_ft),
            n_val=int(args.n_val_ft),
            n_test=int(args.n_test),
            seed_train=ft_seed + 11,
            seed_val=ft_seed + 29,
            seed_test=eval_seed + 47,
            verify_count=int(args.verify_samples),
            add_initial_carry=str(args.add_initial_carry),
        )

        lambda_sweep_payloads: List[Dict[str, Any]] = []
        for lambda_carry in [float(x) for x in args.lambda_carry_grid]:
            ste_pre_model, ste_pre_sel, _ = ste_sweep_and_train_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=int(args.ste_pretrain_epochs),
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=pre_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=None,
            )
            ste_pre_weights = _extract_carry_weight_list(ste_pre_model)

            cvx_from_ste_bundle, cvx_from_ste_sel, cvx_from_ste_tf_test = cvx_fit_shared_two_head_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                seed=pre_seed,
                beta_grid=args.cvx_beta_grid,
                bias_grid=args.cvx_bias_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                cvx_device=cvx_device,
                cvx_sum_loss=str(args.cvx_sum_loss),
                cvx_carry_loss=str(args.cvx_carry_loss),
                cvx_time_loss=str(args.cvx_time_loss),
                init_mode='pretraining',
                pretrained_weights=ste_pre_weights,
                cvx_grid_workers=int(args.cvx_grid_workers),
                cvx_ovr_workers=int(args.cvx_ovr_workers),
            )

            ste_ft_from_ste_model, ste_ft_from_ste_sel, _ = ste_sweep_and_train_init(
                ds=ds_ft,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=int(args.ste_finetune_epochs),
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=ft_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=ste_pre_weights,
            )

            cvx_pre_bundle, cvx_pre_sel, cvx_pre_tf_test = cvx_fit_shared_two_head_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                seed=pre_seed,
                beta_grid=args.cvx_beta_grid,
                bias_grid=args.cvx_bias_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                cvx_device=cvx_device,
                cvx_sum_loss=str(args.cvx_sum_loss),
                cvx_carry_loss=str(args.cvx_carry_loss),
                cvx_time_loss=str(args.cvx_time_loss),
                init_mode='gaussian',
                pretrained_weights=None,
                cvx_grid_workers=int(args.cvx_grid_workers),
                cvx_ovr_workers=int(args.cvx_ovr_workers),
            )
            cvx_pre_weights = _cvx_bundle_to_carry_weights(
                bundle=cvx_pre_bundle,
                d_in=ds_pre.d_in,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                base=int(args.arith_base),
            )
            ste_ft_from_cvx_model, ste_ft_from_cvx_sel, _ = ste_sweep_and_train_init(
                ds=ds_ft,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=int(args.ste_finetune_epochs),
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=ft_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(args.lambda_sum),
                lambda_carry=float(lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=cvx_pre_weights,
            )

            ste_pre_eval = _eval_stage_all_modes(
                ds_fit=ds_pre,
                ds_eval=ds_pre,
                ste_model=ste_pre_model,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )
            cvx_from_ste_eval = _eval_stage_all_modes(
                ds_fit=ds_pre,
                ds_eval=ds_pre,
                cvx_bundle=cvx_from_ste_bundle,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )
            ste_ft_from_ste_eval = _eval_stage_all_modes(
                ds_fit=ds_ft,
                ds_eval=ds_ft,
                ste_model=ste_ft_from_ste_model,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )
            cvx_pre_eval = _eval_stage_all_modes(
                ds_fit=ds_pre,
                ds_eval=ds_pre,
                cvx_bundle=cvx_pre_bundle,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )
            ste_ft_from_cvx_eval = _eval_stage_all_modes(
                ds_fit=ds_ft,
                ds_eval=ds_ft,
                ste_model=ste_ft_from_cvx_model,
                ood_digits=args.ood_digits,
                n_test_ood=int(args.n_test_ood),
                arith_base=int(args.arith_base),
                eval_seed=eval_seed,
                verify_count=int(args.verify_samples),
                add_initial_carry=str(args.add_initial_carry),
            )

            lambda_sweep_payloads.append({
                'lambda_carry': float(lambda_carry),
                'ste_pretrain': {
                    'selected_params': ste_pre_sel,
                    **ste_pre_eval,
                },
                'cvx_from_ste_pretrain': {
                    'selected_params': cvx_from_ste_sel,
                    'teacher_forcing_pretrain_test_metrics': cvx_from_ste_tf_test,
                    'diagnostics': {
                        'primal_value': float(cvx_from_ste_bundle['primal_value']),
                        'dual_value': float(cvx_from_ste_bundle['dual_value']),
                        'gap': float(cvx_from_ste_bundle['gap']),
                    },
                    **cvx_from_ste_eval,
                },
                'ste_finetune_from_ste_pretrain_new_train': {
                    'selected_params': ste_ft_from_ste_sel,
                    **ste_ft_from_ste_eval,
                },
                'cvx_pretrain': {
                    'selected_params': cvx_pre_sel,
                    'teacher_forcing_pretrain_test_metrics': cvx_pre_tf_test,
                    'diagnostics': {
                        'primal_value': float(cvx_pre_bundle['primal_value']),
                        'dual_value': float(cvx_pre_bundle['dual_value']),
                        'gap': float(cvx_pre_bundle['gap']),
                    },
                    **cvx_pre_eval,
                },
                'ste_finetune_from_cvx_pretrain_new_train': {
                    'selected_params': ste_ft_from_cvx_sel,
                    **ste_ft_from_cvx_eval,
                },
            })

        seed_payload = {
            'seed': int(base_seed),
            'split_seeds': {
                'pretrain_train': pre_seed + 11,
                'pretrain_val': pre_seed + 29,
                'finetune_train': ft_seed + 11,
                'finetune_val': ft_seed + 29,
                'eval_test': eval_seed + 47,
            },
            'lambda_sweep': lambda_sweep_payloads,
        }
        seed_payloads.append(seed_payload)
        sdir = out_root / f'seed_{int(base_seed)}'
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / 'metrics.json').write_text(json.dumps(seed_payload, indent=2, default=str) + '\n')

    aggregate = {
        'n_seeds': int(len(seed_payloads)),
        'lambda_sweep': _aggregate_lambda_sweep(seed_payloads),
    }
    root_payload = {
        'run_config': config_dump,
        'out_root': str(out_root),
        'seeds': seed_payloads,
        'aggregate': aggregate,
    }
    (out_root / 'metrics.json').write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
    if str(args.output_json).strip():
        Path(str(args.output_json)).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + '\n')
    print(json.dumps({'aggregate': aggregate, 'out_root': str(out_root)}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
