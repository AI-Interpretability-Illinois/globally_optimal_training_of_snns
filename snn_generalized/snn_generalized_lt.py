import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF

from snn_2_layer.snn_experiments import a_snn_from_blocks, solve_svm_hyperplane


@dataclass
class LTConfig:
    seed: int
    layers: int
    timesteps: int
    n_train: int
    n_test: int
    train_ratio: float
    d: int
    width: int
    beta: float
    arr_samples: int
    sampled_arrangements: bool
    solver: str
    solver_opts: Dict[str, float]
    lr: float
    epochs: int
    svm_c: float
    weight_decay: float
    debug: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_config(args: argparse.Namespace) -> LTConfig:
    if args.solver.upper() == "SCS":
        solver_opts = {"eps": 1e-9, "max_iters": 200000}
    else:
        solver_opts = {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9, "max_iters": 50000}

    if args.debug:
        return LTConfig(
            seed=args.seed,
            layers=args.layers,
            timesteps=args.timesteps,
            n_train=40,
            n_test=40,
            train_ratio=0.8,
            d=6,
            width=150,
            beta=1e-3,
            arr_samples=5000,
            sampled_arrangements=not args.full_arrangements,
            solver=args.solver,
            solver_opts=solver_opts,
            lr=1e-2,
            epochs=50,
            svm_c=1.0,
            weight_decay=1e-4,
            debug=True,
        )
    return LTConfig(
        seed=args.seed,
        layers=args.layers,
        timesteps=args.timesteps,
        n_train=200,
        n_test=200,
        train_ratio=0.8,
        d=20,
        width=150,
        beta=1e-3,
        arr_samples=5000,
        sampled_arrangements=not args.full_arrangements,
        solver=args.solver,
        solver_opts=solver_opts,
        lr=1e-2,
        epochs=300,
        svm_c=1.0,
        weight_decay=1e-4,
        debug=False,
    )


def generate_moving_gaussian_sequence(
    n: int,
    d: int,
    timesteps: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(2, d)).astype(np.float32)
    labels = rng.integers(0, 2, size=n)
    x_list = []
    for t in range(timesteps):
        shift = rng.normal(scale=0.2, size=means.shape).astype(np.float32) * (t + 1) / timesteps
        centers = means + shift
        x_t = rng.normal(size=(n, d)).astype(np.float32) + centers[labels]
        x_list.append(x_t)
    y = labels * 2 - 1
    return x_list, y.astype(np.float32)


def generate_two_step_xor_sequence(
    n: int,
    d: int,
    timesteps: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    if d < 2:
        raise ValueError("two_step_xor requires d >= 2.")
    x_list = []
    predicates = []
    w = np.zeros((d,), dtype=np.float32)
    w[0] = 1.0
    w[1] = -1.0
    for _ in range(timesteps):
        x_t = rng.integers(0, 2, size=(n, d)).astype(np.float32)
        x_list.append(x_t)
        predicates.append((x_t @ w >= 0).astype(np.int32))
    xor_val = predicates[0]
    for pred in predicates[1:]:
        xor_val = np.bitwise_xor(xor_val, pred)
    y = xor_val * 2 - 1
    return x_list, y.astype(np.float32)


def generate_rotated_mnist_sequence(
    n: int,
    timesteps: int,
    seed: int,
    angle_step: float = 15.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    from torchvision import datasets, transforms

    rng = np.random.default_rng(seed)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    indices = rng.choice(len(mnist), size=n, replace=False)
    images = []
    labels = []
    for idx in indices:
        img, label = mnist[idx]
        images.append(img.squeeze(0).numpy())
        labels.append(label)
    base = np.stack(images, axis=0).astype(np.float32)
    x_list = []
    for t in range(timesteps):
        angle = (t + 1) * angle_step
        rot = transforms.RandomRotation((angle, angle))
        x_t = np.stack(
            [rot(torch.tensor(img).unsqueeze(0)).squeeze(0).numpy() for img in base],
            axis=0,
        ).astype(np.float32)
        x_list.append(x_t.reshape(n, -1))
    y = np.array(labels, dtype=np.int64)
    y = (y > (y.mean())).astype(np.float32) * 2 - 1
    return x_list, y.astype(np.float32)


def build_arrangements_lt(
    x_list: List[np.ndarray],
    layers: int,
    timesteps: int,
    seed: int,
    num_samples: int,
    sampled: bool,
) -> Dict[Tuple[int, int], np.ndarray]:
    n = x_list[0].shape[0]
    h_map: Dict[Tuple[int, int], np.ndarray] = {}
    ones = np.ones((n, 1), dtype=np.float32)
    zero_block = np.zeros((n, 1), dtype=np.float32)

    d_map: Dict[Tuple[int, int], np.ndarray] = {}
    for l in range(1, layers + 1):
        for t in range(1, timesteps + 1):
            if l == 1:
                block1 = x_list[t - 1]
            else:
                block1 = d_map[(l - 1, t)]
            if t == 1:
                block2 = zero_block
                op = "+"
            else:
                block2 = d_map[(l, t - 1)]
                op = "-"
            d_map[(l, t)] = a_snn_from_blocks(
                block1,
                block2,
                op,
                ones,
                sampled,
                num_samples,
                seed + 100 * l + t,
            )
    return d_map


def solve_convex_lasso(
    d_last: np.ndarray,
    y: np.ndarray,
    beta: float,
    width: int,
    solver: str,
    solver_opts: Dict[str, float],
) -> Tuple[np.ndarray, float, float, float]:
    beta_hat = beta / np.sqrt(width)
    w = cp.Variable(d_last.shape[1])
    objective = 0.5 * cp.sum_squares(d_last @ w - y) + beta_hat * cp.norm1(w)
    primal_problem = cp.Problem(cp.Minimize(objective))
    primal_problem.solve(solver=solver, **solver_opts)
    primal_val = float(primal_problem.value)

    u = cp.Variable(d_last.shape[0])
    dual_objective = -0.5 * cp.sum_squares(u) - y @ u
    constraints = [cp.norm_inf(d_last.T @ u) <= beta_hat]
    dual_problem = cp.Problem(cp.Maximize(dual_objective), constraints)
    dual_problem.solve(solver=solver, **solver_opts)
    dual_val = float(dual_problem.value)
    duality_gap = primal_val - dual_val
    return w.value, primal_val, dual_val, duality_gap


def select_patterns_lt(
    d_map: Dict[Tuple[int, int], np.ndarray],
    w_last: np.ndarray,
    layers: int,
    timesteps: int,
    width: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    def closest_patterns(d_curr: np.ndarray, target: np.ndarray) -> np.ndarray:
        h_out = np.zeros((d_curr.shape[0], target.shape[1]), dtype=np.float32)
        for j in range(target.shape[1]):
            diff = np.sum(np.abs(d_curr - target[:, [j]]), axis=0)
            h_out[:, j] = d_curr[:, np.argmin(diff)]
        return h_out

    h_map: Dict[Tuple[int, int], np.ndarray] = {}
    d_last = d_map[(layers, timesteps)]
    top_idx = np.argsort(-np.abs(w_last))[: min(width, d_last.shape[1])]
    if top_idx.size < width:
        pad = np.resize(top_idx, width)
        top_idx = pad
    h_map[(layers, timesteps)] = d_last[:, top_idx]

    for t in range(timesteps - 1, 0, -1):
        d_curr = d_map[(layers, t)]
        h_target = h_map[(layers, t + 1)]
        h_map[(layers, t)] = closest_patterns(d_curr, h_target)

    for l in range(layers - 1, 0, -1):
        for t in range(timesteps, 0, -1):
            d_curr = d_map[(l, t)]
            h_target = h_map[(l + 1, t)]
            h_map[(l, t)] = closest_patterns(d_curr, h_target)
    return h_map


def reconstruct_weights_lt(
    x_list: List[np.ndarray],
    h_map: Dict[Tuple[int, int], np.ndarray],
    layers: int,
    timesteps: int,
    width: int,
    solver: str,
    solver_opts: Dict[str, float],
    C: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    p_in_list: List[np.ndarray] = []
    p_rec_list: List[np.ndarray] = []

    for l in range(1, layers + 1):
        n = x_list[0].shape[0]
        p_in = np.zeros((x_list[0].shape[1] if l == 1 else width, width), dtype=np.float32)
        p_rec = np.zeros((width + 1, width), dtype=np.float32)
        for j in range(width):
            x_blocks = []
            y_blocks = []
            for t in range(1, timesteps + 1):
                if l == 1:
                    x_in = x_list[t - 1]
                else:
                    x_in = h_map[(l - 1, t)]
                if t == 1:
                    h_prev = np.zeros((n, width), dtype=np.float32)
                else:
                    h_prev = h_map[(l, t - 1)]
                rec_feat = np.hstack([-h_prev, np.ones((n, 1), dtype=np.float32)])
                x_blocks.append(np.hstack([x_in, rec_feat]))
                y_blocks.append(2 * h_map[(l, t)][:, j] - 1)
            x_stack = np.vstack(x_blocks)
            y_stack = np.concatenate(y_blocks, axis=0)
            w = solve_svm_hyperplane(x_stack, y_stack, C, solver, solver_opts)
            p_in[:, j] = w[: p_in.shape[0]]
            p_rec[:, j] = w[p_in.shape[0] :]
        p_in_list.append(p_in)
        p_rec_list.append(p_rec)
    return p_in_list, p_rec_list


def forward_reconstructed_lt(
    x_list: List[np.ndarray],
    p_in_list: List[np.ndarray],
    p_rec_list: List[np.ndarray],
    layers: int,
    timesteps: int,
    width: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    n = x_list[0].shape[0]
    h_map: Dict[Tuple[int, int], np.ndarray] = {}

    for l in range(1, layers + 1):
        p_in = p_in_list[l - 1]
        p_rec = p_rec_list[l - 1]
        for t in range(1, timesteps + 1):
            if l == 1:
                x_in = x_list[t - 1]
            else:
                x_in = h_map[(l - 1, t)]
            if t == 1:
                h_prev = np.zeros((n, width), dtype=np.float32)
            else:
                h_prev = h_map[(l, t - 1)]
            rec_feat = np.hstack([-h_prev, np.ones((n, 1), dtype=np.float32)])
            h_map[(l, t)] = (x_in @ p_in + rec_feat @ p_rec >= 0).astype(np.float32)
    return h_map


def train_lif_baseline(
    x_list: List[np.ndarray],
    y: np.ndarray,
    layers: int,
    width: int,
    lr: float,
    epochs: int,
    weight_decay: float,
) -> Tuple[float, List[float], Tuple[List[nn.Linear], List[snn.Leaky], nn.Linear]]:
    x_list_t = [torch.from_numpy(x_t) for x_t in x_list]
    y_t = torch.from_numpy(y)

    if layers < 1:
        raise ValueError("SNN depth must be >= 1.")
    hidden_dims = [width] * layers
    fcs: List[nn.Linear] = []
    lifs: List[snn.Leaky] = []
    in_dim = x_list[0].shape[1]
    for hidden_dim in hidden_dims:
        fcs.append(nn.Linear(in_dim, hidden_dim, bias=True))
        lifs.append(
            snn.Leaky(
                beta=0.9,
                threshold=0.0,
                learn_beta=True,
                learn_threshold=True,
            )
        )
        in_dim = hidden_dim
    fc_out = nn.Linear(in_dim, 1, bias=True)

    params = []
    for fc in fcs:
        params += list(fc.parameters())
    for lif in lifs:
        params += list(lif.parameters())
    params += list(fc_out.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    def forward_last_mem() -> torch.Tensor:
        mems_local = [lif.init_leaky() for lif in lifs]
        last_mem = None
        for x_t in x_list_t:
            h_local = x_t
            for idx, (fc, lif) in enumerate(zip(fcs, lifs)):
                spk, mems_local[idx] = lif(fc(h_local), mems_local[idx])
                h_local = spk
            last_mem = mems_local[-1]
        return last_mem

    def path_norm_penalty() -> torch.Tensor:
        path_mat = fcs[0].weight.pow(2).t()
        for fc in fcs[1:]:
            path_mat = path_mat @ fc.weight.pow(2)
        path_mat = path_mat @ fc_out.weight.pow(2).t()
        return 0.5 * weight_decay * torch.sum(path_mat)

    train_acc_history: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        last_mem = forward_last_mem()
        logits = fc_out(last_mem).squeeze(-1)
        loss = torch.mean((logits - y_t) ** 2) + path_norm_penalty()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits_eval = fc_out(forward_last_mem()).squeeze(-1)
            preds = torch.where(logits_eval >= 0, 1.0, -1.0)
            acc = float((preds == y_t).float().mean().item())
        train_acc_history.append(acc)

    with torch.no_grad():
        last_mem = forward_last_mem()
        preds = torch.where(fc_out(last_mem).squeeze(-1) >= 0, 1.0, -1.0)
    acc = float((preds == y_t).float().mean().item())
    return acc, train_acc_history, (fcs, lifs, fc_out)


def eval_lif_baseline(
    model: Tuple[List[nn.Linear], List[snn.Leaky], nn.Linear],
    x_list: List[np.ndarray],
) -> np.ndarray:
    fcs, lifs, fc_out = model
    x_list_t = [torch.from_numpy(x_t) for x_t in x_list]
    def forward_last_mem() -> torch.Tensor:
        mems_local = [lif.init_leaky() for lif in lifs]
        last_mem = None
        for x_t in x_list_t:
            h_local = x_t
            for idx, (fc, lif) in enumerate(zip(fcs, lifs)):
                spk, mems_local[idx] = lif(fc(h_local), mems_local[idx])
                h_local = spk
            last_mem = mems_local[-1]
        return last_mem

    with torch.no_grad():
        logits = fc_out(forward_last_mem()).squeeze(-1)
        preds = torch.where(logits >= 0, 1.0, -1.0).cpu().numpy()
    return preds


def parse_timesteps_list(t_list: str) -> List[int]:
    values = []
    for item in t_list.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("timesteps list is empty.")
    return values


def parse_datasets_list(d_list: str) -> List[str]:
    values = []
    for item in d_list.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item)
    if not values:
        raise ValueError("datasets list is empty.")
    return values


def split_sequence(
    x_list: List[np.ndarray],
    y: np.ndarray,
    n_train: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    x_train = [x[:n_train] for x in x_list]
    x_test = [x[n_train:] for x in x_list]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return x_train, y_train, x_test, y_test


def describe_layer_params(layers: int, input_dim: int, width: int) -> str:
    dims = []
    in_dim = input_dim
    for layer_idx in range(1, layers + 1):
        dims.append(f"L{layer_idx}:{in_dim}->{width}")
        in_dim = width
    return " | ".join(dims)


def append_lines(path: Path, lines: List[str]) -> None:
    if path.exists():
        existing = path.read_text()
        if existing and not existing.endswith("\n"):
            path.write_text(existing + "\n")
    else:
        path.write_text("")
    with path.open("a") as handle:
        for line in lines:
            handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized L-layer, T-timestep SNN experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--full-arrangements", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timesteps-list", type=str, default="")
    parser.add_argument("--datasets", type=str, default="moving_gaussian")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)


    if args.timesteps_list:
        timesteps_list = parse_timesteps_list(args.timesteps_list)
    else:
        timesteps_list = [config.timesteps]

    datasets = parse_datasets_list(args.datasets)
    for dataset_name in datasets:
        for t_steps in timesteps_list:
            out_path = Path(__file__).with_name(
                f"snn_generalized_lt_L{config.layers}_T{t_steps}_split_{args.train_ratio:.2f}.txt"
            )
            t0 = time.perf_counter()
            print(f"[LT] stage=init dataset={dataset_name} timesteps={t_steps}")
            if not (0.0 < args.train_ratio < 1.0):
                raise ValueError("train_ratio must be in (0,1).")
            n_total = config.n_train + config.n_test
            n_train = max(1, int(round(n_total * args.train_ratio)))
            n_test = n_total - n_train
            if n_test == 0:
                n_test = 1
                n_train = n_total - 1
            if dataset_name == "two_step_xor":
                x_list, y = generate_two_step_xor_sequence(
                    n_total,
                    2,
                    t_steps,
                    config.seed + 1,
                )
            elif dataset_name == "moving_gaussian":
                x_list, y = generate_moving_gaussian_sequence(
                    n_total,
                    config.d,
                    t_steps,
                    config.seed + 1,
                )
            elif dataset_name == "rotated_mnist":
                x_list, y = generate_rotated_mnist_sequence(
                    n_total,
                    t_steps,
                    config.seed + 1,
                )
            else:
                raise ValueError(f"Unsupported dataset '{dataset_name}'.")
            x_train, y_train, x_test, y_test = split_sequence(
                x_list,
                y,
                n_train,
            )
            input_dim = x_list[0].shape[1]
            print(
                "[LT] layer_params "
                f"layers={config.layers} width={config.width} "
                f"input_dim={input_dim} lif_beta=0.9"
            )
            print(f"[LT] layer_dims {describe_layer_params(config.layers, input_dim, config.width)}")
            append_lines(
                out_path,
                [
                    f"RUN snn_generalized_lt.py dataset={dataset_name} layers={config.layers} seed ={config.seed} "
                    f"timesteps={t_steps} train_ratio={args.train_ratio}",
                    f"layer_params layers={config.layers} width={config.width} input_dim={input_dim} lif_beta=0.9",
                    f"layer_dims {describe_layer_params(config.layers, input_dim, config.width)}",
                ],
            )
            print(f"[LT] stage=data_ready elapsed_s={time.perf_counter() - t0:.2f}")
            d_map = build_arrangements_lt(
                x_train,
                config.layers,
                t_steps,
                config.seed,
                config.arr_samples,
                config.sampled_arrangements,
            )
            print(f"[LT] stage=arrangements_done elapsed_s={time.perf_counter() - t0:.2f}")
            d_shape_lines = []
            for l in range(1, config.layers + 1):
                for t in range(1, t_steps + 1):
                    d_curr = d_map[(l, t)]
                    line = f"D({l},{t}) shape = {d_curr.shape}"
                    print(line)
                    d_shape_lines.append(line)
            append_lines(out_path, d_shape_lines)

            d_last = d_map[(config.layers, t_steps)]
            print(f"[LT] stage=convex_solve start elapsed_s={time.perf_counter() - t0:.2f}")
            w_last, primal_val, dual_val, duality_gap = solve_convex_lasso(
                d_last,
                y_train,
                config.beta,
                config.width,
                config.solver,
                config.solver_opts,
            )
            tol = 1e-7
            w_zero_count = int(np.sum(np.abs(w_last) <= tol))
            w_nonzero_count = int(np.sum(np.abs(w_last) > tol))
            w_nonzero_gt_width = w_nonzero_count > config.width
            w_sorted = np.sort(w_last)
            w_last_str = np.array2string(w_sorted, separator=",", threshold=np.inf)
            print(f"[LT] stage=convex_solve done elapsed_s={time.perf_counter() - t0:.2f}")
            gap_display = 0.0 if abs(duality_gap) <= 1e-7 else duality_gap
            print(
                f"[LT] convex_primal={primal_val:.6e} convex_dual={dual_val:.6e} "
                f"duality_gap={gap_display:.6e}"
            )
            print(
                f"[LT] convex_w tol={tol:.1e} zeros={w_zero_count} nonzeros={w_nonzero_count} "
                f"nonzeros_gt_width={w_nonzero_gt_width}"
            )
            print(f"[LT] convex_w values_sorted={w_last_str}")
            convex_train_preds = np.sign(d_last @ w_last)
            convex_train_acc = float(np.mean(convex_train_preds == y_train))
            print(f"[LT] stage=pattern_select start elapsed_s={time.perf_counter() - t0:.2f}")
            h_map = select_patterns_lt(
                d_map,
                w_last,
                config.layers,
                t_steps,
                config.width,
            )
            print(f"[LT] stage=pattern_select done elapsed_s={time.perf_counter() - t0:.2f}")
            print(f"[LT] stage=reconstruction start elapsed_s={time.perf_counter() - t0:.2f}")
            p_in_list, p_rec_list = reconstruct_weights_lt(
                x_train,
                h_map,
                config.layers,
                t_steps,
                config.width,
                config.solver,
                config.solver_opts,
                C=args.svm_c,
            )
            print(f"[LT] stage=reconstruction done elapsed_s={time.perf_counter() - t0:.2f}")
            print(f"[LT] stage=forward_recon start elapsed_s={time.perf_counter() - t0:.2f}")
            h_hat_train = forward_reconstructed_lt(
                x_train,
                p_in_list,
                p_rec_list,
                config.layers,
                t_steps,
                config.width,
            )
            h_hat_test = forward_reconstructed_lt(
                x_test,
                p_in_list,
                p_rec_list,
                config.layers,
                t_steps,
                config.width,
            )
            print(f"[LT] stage=forward_recon done elapsed_s={time.perf_counter() - t0:.2f}")
            v = solve_svm_hyperplane(
                h_hat_train[(config.layers, t_steps)],
                y_train,
                args.svm_c,
                config.solver,
                config.solver_opts,
            )
            recon_train_preds = np.sign(h_hat_train[(config.layers, t_steps)] @ v)
            recon_test_preds = np.sign(h_hat_test[(config.layers, t_steps)] @ v)
            recon_train_acc = float(np.mean(recon_train_preds == y_train))
            recon_test_acc = float(np.mean(recon_test_preds == y_test))

            print(f"[LT] stage=lif_train start elapsed_s={time.perf_counter() - t0:.2f}")
            lif_train_acc, lif_train_acc_history, lif_model = train_lif_baseline(
                x_train,
                y_train,
                config.layers,
                config.width,
                config.lr,
                config.epochs,
                args.weight_decay,
            )
            print(f"[LT] stage=lif_train done elapsed_s={time.perf_counter() - t0:.2f}")
            lif_acc_path = Path(__file__).with_name(
                f"snn_generalized_lt_L{config.layers}_T{t_steps}_split_{args.train_ratio:.2f}_"
                f"{dataset_name}_lif_train_acc.png"
            )
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(lif_train_acc_history) + 1), lif_train_acc_history, marker="o", linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel("Train Accuracy")
            plt.title(f"LIF Baseline Train Accuracy (L={config.layers}, T={t_steps}, {dataset_name})")
            plt.grid(True, linewidth=0.3, alpha=0.6)
            plt.tight_layout()
            plt.savefig(lif_acc_path, dpi=150)
            plt.close()
            lif_train_preds = eval_lif_baseline(lif_model, x_train)
            lif_test_preds = eval_lif_baseline(lif_model, x_test)
            lif_test_acc = float(np.mean(lif_test_preds == y_test))

            print(
                f"[LT] sampled={config.sampled_arrangements} "
                f"dataset={dataset_name} layers={config.layers} timesteps={t_steps} "
                f"convex_train_acc={convex_train_acc:.3f} "
                f"recon_train_acc={recon_train_acc:.3f} recon_test_acc={recon_test_acc:.3f} "
                f"lif_train_acc={lif_train_acc:.3f} lif_test_acc={lif_test_acc:.3f}"
            )
            append_lines(
                out_path,
                [
                    f"convex_primal={primal_val:.6e} convex_dual={dual_val:.6e} "
                    f"duality_gap={gap_display:.6e}",
                    (
                        "convex_w tol="
                        f"{tol:.1e} zeros={w_zero_count} nonzeros={w_nonzero_count} "
                        f"nonzeros_gt_width={w_nonzero_gt_width}"
                    ),
                    f"convex_w values_sorted={w_last_str}",
                    f"convex_train_acc={convex_train_acc:.6f}",
                    f"recon_train_acc={recon_train_acc:.6f} recon_test_acc={recon_test_acc:.6f} "
                    f"lif_train_acc={lif_train_acc:.6f} lif_test_acc={lif_test_acc:.6f}",
                    f"lif_train_acc_plot={lif_acc_path.name}",
                    f"total_elapsed_s={time.perf_counter() - t0:.2f}",
                    "",
                ],
            )
            print(f"[LT] stage=done total_elapsed_s={time.perf_counter() - t0:.2f}")


if __name__ == "__main__":
    main()
