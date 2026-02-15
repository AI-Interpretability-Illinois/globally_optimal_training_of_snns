import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ConeConfig:
    seed: int
    n_train: int
    n_test: int
    d: int
    layers: int
    timesteps: int
    grid_size: int
    grid_span: float
    margin: float
    num_arr_samples: int
    dataset: str
    sampled_arrangements: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def build_config(args: argparse.Namespace) -> ConeConfig:
    if args.debug:
        return ConeConfig(
            seed=args.seed,
            n_train=40,
            n_test=40,
            d=6,
            layers=args.layers,
            timesteps=args.timesteps,
            grid_size=30,
            grid_span=2.0,
            margin=1e-3,
            num_arr_samples=200,
            dataset=args.dataset,
            sampled_arrangements=not args.full_arrangements,
        )
    return ConeConfig(
        seed=args.seed,
        n_train=200,
        n_test=200,
        d=20,
        layers=args.layers,
        timesteps=args.timesteps,
        grid_size=60,
        grid_span=3.0,
        margin=1e-3,
        num_arr_samples=2000,
        dataset=args.dataset,
        sampled_arrangements=not args.full_arrangements,
    )


def split_sequence(
    x_list: list[np.ndarray],
    y: np.ndarray,
    n_train: int,
) -> Tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    x_train = [x[:n_train] for x in x_list]
    x_test = [x[n_train:] for x in x_list]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return x_train, y_train, x_test, y_test


def sample_arrangements(x: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(x.shape[1], num_samples)).astype(np.float32)
    patterns = (x @ u >= 0).astype(np.float32)
    uniq = {}
    for idx in range(patterns.shape[1]):
        key = tuple(patterns[:, idx].tolist())
        if key not in uniq:
            uniq[key] = patterns[:, idx]
    if not uniq:
        raise ValueError("No hyperplane arrangements produced.")
    return np.stack(list(uniq.values()), axis=1)


def find_feasible_w(x: np.ndarray, d: np.ndarray, margin: float) -> np.ndarray:
    w = cp.Variable(x.shape[1])
    s = 2.0 * d - 1.0
    constraints = [cp.multiply(s, x @ w) >= margin]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w)), constraints)
    problem.solve(solver="SCS", eps=1e-6, max_iters=200000)
    if w.value is None:
        raise ValueError("Feasible point not found for reconstruction cone.")
    return w.value.astype(np.float32)


def cone_loss_grid(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    w0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    grid_size: int,
    span: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_vals = np.linspace(-span, span, grid_size)
    b_vals = np.linspace(-span, span, grid_size)
    aa, bb = np.meshgrid(a_vals, b_vals)
    loss = np.full_like(aa, np.nan, dtype=np.float32)
    s = 2.0 * d - 1.0

    for i in range(grid_size):
        for j in range(grid_size):
            w = w0 + aa[i, j] * v1 + bb[i, j] * v2
            if np.any(s * (x @ w) < 0):
                continue
            resid = x @ w - y
            loss[i, j] = 0.5 * float(resid.T @ resid)
    return aa, bb, loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Convex cone loss landscape visualization.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="moving_gaussian")
    parser.add_argument("--full-arrangements", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    if config.dataset == "moving_gaussian":
        from snn_generalized_lt import generate_moving_gaussian_sequence, build_arrangements_lt
        x_list, y = generate_moving_gaussian_sequence(
            config.n_train + config.n_test,
            config.d,
            config.timesteps,
            config.seed + 1,
        )
    elif config.dataset == "rotated_mnist":
        from snn_generalized_lt import generate_rotated_mnist_sequence, build_arrangements_lt
        x_list, y = generate_rotated_mnist_sequence(
            config.n_train + config.n_test,
            config.timesteps,
            config.seed + 1,
        )
    else:
        raise ValueError(f"Unsupported dataset '{config.dataset}'.")

    x_train, y_train, _, _ = split_sequence(x_list, y, config.n_train)
    d_map = build_arrangements_lt(
        x_train,
        config.layers,
        config.timesteps,
        config.seed,
        config.num_arr_samples,
        config.sampled_arrangements,
    )
    d_all = d_map[(config.layers, config.timesteps)]
    rng = np.random.default_rng(config.seed + 3)
    d = d_all[:, rng.integers(0, d_all.shape[1])]
    x = x_train[-1] if config.timesteps == 1 else x_train[0]
    w0 = find_feasible_w(x, d, config.margin)

    v1 = rng.normal(size=(config.d,)).astype(np.float32)
    v2 = rng.normal(size=(config.d,)).astype(np.float32)
    v1 /= max(1e-8, np.linalg.norm(v1))
    v2 /= max(1e-8, np.linalg.norm(v2))

    aa, bb, loss = cone_loss_grid(
        x,
        y_train,
        d,
        w0,
        v1,
        v2,
        config.grid_size,
        config.grid_span,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    mask = np.isfinite(loss)
    ax.plot_surface(aa, bb, np.where(mask, loss, np.nan), cmap="viridis", linewidth=0)
    ax.set_title("Loss landscape within reconstruction cone")
    ax.set_xlabel("alpha (v1)")
    ax.set_ylabel("beta (v2)")
    ax.set_zlabel("Train loss")
    out_path = Path(__file__).with_name(
        f"convex_cone_landscape_{config.dataset}_L{config.layers}_T{config.timesteps}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
