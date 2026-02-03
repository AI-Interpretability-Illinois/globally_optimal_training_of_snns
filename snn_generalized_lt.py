import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from snn_experiments import a_snn_from_blocks, solve_svm_hyperplane


@dataclass
class LTConfig:
    seed: int
    layers: int
    timesteps: int
    n: int
    d: int
    width: int
    beta: float
    arr_samples: int
    sampled_arrangements: bool
    solver: str
    solver_opts: Dict[str, float]
    lr: float
    epochs: int
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
            n=40,
            d=6,
            width=20,
            beta=1e-3,
            arr_samples=120,
            sampled_arrangements=not args.full_arrangements,
            solver=args.solver,
            solver_opts=solver_opts,
            lr=1e-2,
            epochs=50,
            debug=True,
        )
    return LTConfig(
        seed=args.seed,
        layers=args.layers,
        timesteps=args.timesteps,
        n=200,
        d=20,
        width=100,
        beta=1e-3,
        arr_samples=1000,
        sampled_arrangements=not args.full_arrangements,
        solver=args.solver,
        solver_opts=solver_opts,
        lr=1e-2,
        epochs=300,
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


def build_arrangements_lt(
    x_list: List[np.ndarray],
    layers: int,
    timesteps: int,
    seed: int,
    num_samples: int,
    sampled: bool,
) -> Dict[Tuple[int, int], np.ndarray]:
    n = x_list[0].shape[0]
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
) -> np.ndarray:
    beta_hat = beta / np.sqrt(width)
    w = cp.Variable(d_last.shape[1])
    objective = 0.5 * cp.sum_squares(d_last @ w - y) + beta_hat * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver, **solver_opts)
    return w.value


def select_patterns_lt(
    d_map: Dict[Tuple[int, int], np.ndarray],
    w_last: np.ndarray,
    layers: int,
    timesteps: int,
    width: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    h_map: Dict[Tuple[int, int], np.ndarray] = {}
    d_last = d_map[(layers, timesteps)]
    top_idx = np.argsort(-np.abs(w_last))[:width]
    h_map[(layers, timesteps)] = d_last[:, top_idx]

    for t in range(timesteps - 1, 0, -1):
        d_curr = d_map[(layers, t)]
        h_target = h_map[(layers, t + 1)]
        h_t = np.zeros((d_curr.shape[0], width), dtype=np.float32)
        for j in range(width):
            diff = np.sum(np.abs(d_curr - h_target[:, [j]]), axis=0)
            h_t[:, j] = d_curr[:, np.argmin(diff)]
        h_map[(layers, t)] = h_t

    for l in range(1, layers):
        for t in range(1, timesteps + 1):
            d_curr = d_map[(l, t)]
            if d_curr.shape[1] < width:
                reps = int(np.ceil(width / d_curr.shape[1]))
                d_curr = np.tile(d_curr, (1, reps))
            h_map[(l, t)] = d_curr[:, :width]
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
) -> Tuple[float, Tuple[List[nn.Linear], List[snn.Leaky], nn.Linear]]:
    x_list_t = [torch.from_numpy(x_t) for x_t in x_list]
    y_t = torch.from_numpy(y)

    fcs = []
    lifs = []
    in_dim = x_list[0].shape[1]
    for _ in range(layers):
        fcs.append(nn.Linear(in_dim, width, bias=False))
        lifs.append(snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()))
        in_dim = width
    fc_out = nn.Linear(width, 1, bias=False)

    params = []
    for fc in fcs:
        params += list(fc.parameters())
    params += list(fc_out.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        mems = [lif.init_leaky() for lif in lifs]
        for t in range(len(x_list_t)):
            spk = x_list_t[t]
            for idx in range(layers):
                cur = fcs[idx](spk)
                spk, mems[idx] = lifs[idx](cur, mems[idx])
        logits = fc_out(spk).squeeze(-1)
        loss = torch.mean((logits - y_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mems = [lif.init_leaky() for lif in lifs]
        for t in range(len(x_list_t)):
            spk = x_list_t[t]
            for idx in range(layers):
                cur = fcs[idx](spk)
                spk, mems[idx] = lifs[idx](cur, mems[idx])
        preds = fc_out(spk).squeeze(-1).sign()
    acc = float((preds == y_t).float().mean().item())
    return acc, (fcs, lifs, fc_out)


def eval_lif_baseline(
    model: Tuple[List[nn.Linear], List[snn.Leaky], nn.Linear],
    x_list: List[np.ndarray],
) -> np.ndarray:
    fcs, lifs, fc_out = model
    x_list_t = [torch.from_numpy(x_t) for x_t in x_list]
    with torch.no_grad():
        mems = [lif.init_leaky() for lif in lifs]
        for t in range(len(x_list_t)):
            spk = x_list_t[t]
            for idx in range(len(fcs)):
                cur = fcs[idx](spk)
                spk, mems[idx] = lifs[idx](cur, mems[idx])
        preds = fc_out(spk).squeeze(-1).sign().numpy()
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized L-layer, T-timestep SNN experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--full-arrangements", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    t0 = time.perf_counter()
    print("[LT] stage=init")
    x_list, y = generate_moving_gaussian_sequence(
        config.n,
        config.d,
        config.timesteps,
        config.seed + 1,
    )
    print(f"[LT] stage=data_ready elapsed_s={time.perf_counter() - t0:.2f}")
    d_map = build_arrangements_lt(
        x_list,
        config.layers,
        config.timesteps,
        config.seed,
        config.arr_samples,
        config.sampled_arrangements,
    )
    print(f"[LT] stage=arrangements_done elapsed_s={time.perf_counter() - t0:.2f}")
    for l in range(1, config.layers + 1):
        for t in range(1, config.timesteps + 1):
            d_curr = d_map[(l, t)]
            print(f"D({l},{t}) shape = {d_curr.shape}")

    d_last = d_map[(config.layers, config.timesteps)]
    print(f"[LT] stage=convex_solve start elapsed_s={time.perf_counter() - t0:.2f}")
    w_last = solve_convex_lasso(
        d_last,
        y,
        config.beta,
        config.width,
        config.solver,
        config.solver_opts,
    )
    print(f"[LT] stage=convex_solve done elapsed_s={time.perf_counter() - t0:.2f}")
    print(f"[LT] stage=pattern_select start elapsed_s={time.perf_counter() - t0:.2f}")
    h_map = select_patterns_lt(
        d_map,
        w_last,
        config.layers,
        config.timesteps,
        config.width,
    )
    print(f"[LT] stage=pattern_select done elapsed_s={time.perf_counter() - t0:.2f}")
    print(f"[LT] stage=reconstruction start elapsed_s={time.perf_counter() - t0:.2f}")
    p_in_list, p_rec_list = reconstruct_weights_lt(
        x_list,
        h_map,
        config.layers,
        config.timesteps,
        config.width,
        config.solver,
        config.solver_opts,
        C=1000.0,
    )
    print(f"[LT] stage=reconstruction done elapsed_s={time.perf_counter() - t0:.2f}")
    print(f"[LT] stage=forward_recon start elapsed_s={time.perf_counter() - t0:.2f}")
    h_hat = forward_reconstructed_lt(
        x_list,
        p_in_list,
        p_rec_list,
        config.layers,
        config.timesteps,
        config.width,
    )
    print(f"[LT] stage=forward_recon done elapsed_s={time.perf_counter() - t0:.2f}")
    v = solve_svm_hyperplane(
        h_hat[(config.layers, config.timesteps)],
        y,
        1000.0,
        config.solver,
        config.solver_opts,
    )
    recon_preds = np.sign(h_hat[(config.layers, config.timesteps)] @ v)
    recon_acc = float(np.mean(recon_preds == y))

    print(f"[LT] stage=lif_train start elapsed_s={time.perf_counter() - t0:.2f}")
    lif_train_acc, lif_model = train_lif_baseline(
        x_list,
        y,
        config.layers,
        config.width,
        config.lr,
        config.epochs,
    )
    print(f"[LT] stage=lif_train done elapsed_s={time.perf_counter() - t0:.2f}")
    lif_preds = eval_lif_baseline(lif_model, x_list)
    lif_acc = float(np.mean(lif_preds == y))

    print(
        f"[LT] sampled={config.sampled_arrangements} "
        f"layers={config.layers} timesteps={config.timesteps} "
        f"recon_acc={recon_acc:.3f} lif_train_acc={lif_train_acc:.3f} lif_acc={lif_acc:.3f}"
    )
    print(f"[LT] stage=done total_elapsed_s={time.perf_counter() - t0:.2f}")


if __name__ == "__main__":
    main()
