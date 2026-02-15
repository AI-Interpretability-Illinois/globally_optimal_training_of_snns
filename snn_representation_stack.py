import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from snn_experiments import solve_svm_hyperplane


@dataclass
class RepStackConfig:
    seed: int
    timesteps: int
    n: int
    d: int
    m_repr: int
    width: int
    beta: float
    solver: str
    solver_opts: Dict[str, float]
    debug: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def build_config(args: argparse.Namespace) -> RepStackConfig:
    if args.solver.upper() == "SCS":
        solver_opts = {"eps": 1e-9, "max_iters": 200000}
    else:
        solver_opts = {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9, "max_iters": 50000}

    if args.debug:
        return RepStackConfig(
            seed=args.seed,
            timesteps=args.timesteps,
            n=40,
            d=6,
            m_repr=120,
            width=20,
            beta=1e-3,
            solver=args.solver,
            solver_opts=solver_opts,
            debug=True,
        )
    return RepStackConfig(
        seed=args.seed,
        timesteps=args.timesteps,
        n=200,
        d=20,
        m_repr=1000,
        width=100,
        beta=1e-3,
        solver=args.solver,
        solver_opts=solver_opts,
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


def stack_timesteps(x_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(x_list, axis=1)


def create_representation(x_stack: np.ndarray, m_repr: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(x_stack.shape[1], m_repr)).astype(np.float32)
    x_tilde = (x_stack @ h >= 0).astype(np.float32)
    return x_tilde, h


def solve_convex_lasso(
    d_mat: np.ndarray,
    y: np.ndarray,
    beta: float,
    width: int,
    solver: str,
    solver_opts: Dict[str, float],
) -> np.ndarray:
    beta_hat = beta / np.sqrt(width)
    w = cp.Variable(d_mat.shape[1])
    objective = 0.5 * cp.sum_squares(d_mat @ w - y) + beta_hat * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver, **solver_opts)
    return w.value


def select_patterns(d_mat: np.ndarray, w: np.ndarray, width: int) -> np.ndarray:
    top_idx = np.argsort(-np.abs(w))[:width]
    return d_mat[:, top_idx]


def reconstruct_weights_stack(
    x_stack: np.ndarray,
    h_sel: np.ndarray,
    solver: str,
    solver_opts: Dict[str, float],
    C: float,
) -> np.ndarray:
    w_rec = np.zeros((x_stack.shape[1], h_sel.shape[1]), dtype=np.float32)
    for j in range(h_sel.shape[1]):
        y_stack = 2 * h_sel[:, j] - 1
        w = solve_svm_hyperplane(x_stack, y_stack, C, solver, solver_opts)
        w_rec[:, j] = w
    return w_rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Stacked representation SNN reconstruction.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    x_list, y = generate_moving_gaussian_sequence(
        config.n,
        config.d,
        config.timesteps,
        config.seed + 1,
    )
    x_stack = stack_timesteps(x_list)
    x_tilde, h_repr = create_representation(x_stack, config.m_repr, config.seed + 11)
    w = solve_convex_lasso(
        x_tilde,
        y,
        config.beta,
        config.width,
        config.solver,
        config.solver_opts,
    )
    h_sel = select_patterns(x_tilde, w, config.width)
    w_rec = reconstruct_weights_stack(
        x_stack,
        h_sel,
        config.solver,
        config.solver_opts,
        C=1000.0,
    )
    recon_preds = np.sign((x_stack @ w_rec) @ np.sign(np.ones(config.width)))
    recon_acc = float(np.mean(recon_preds == y))

    print(f"[RepStack] X_stack shape={x_stack.shape} X_tilde shape={x_tilde.shape}")
    print(f"[RepStack] selected_patterns={h_sel.shape[1]} recon_acc={recon_acc:.3f}")


if __name__ == "__main__":
    main()
