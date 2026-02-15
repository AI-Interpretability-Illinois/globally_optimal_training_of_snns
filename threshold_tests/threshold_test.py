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


@dataclass
class ExperimentConfig:
    seed: int
    n_train: int
    n_test: int
    d: int
    layers: int
    width: int
    beta: float
    lr: float
    epochs: int
    num_seeds: int
    rep_dim: int
    solver: str
    debug: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_ground_truth(
    n: int,
    d: int,
    seed: int,
    m_star: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w1 = rng.normal(size=(d, m_star)).astype(np.float32)
    w2 = rng.normal(size=(m_star, m_star)).astype(np.float32)
    w3 = rng.normal(size=(m_star,)).astype(np.float32)
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    y = np.sign(h2 @ w3).astype(np.float32)
    return x, y


class SurrogateThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, mode: str) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.mode = mode
        return (input_tensor >= 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        mode = ctx.mode
        if mode == "ste":
            grad = torch.ones_like(input_tensor)
        elif mode == "relu":
            grad = (input_tensor > 0).to(input_tensor.dtype)
        elif mode == "lrelu":
            grad = torch.where(
                input_tensor > 0,
                torch.ones_like(input_tensor),
                0.1 * torch.ones_like(input_tensor),
            )
        elif mode == "crelu":
            grad = ((input_tensor > 0) & (input_tensor < 1)).to(input_tensor.dtype)
        else:
            raise ValueError(f"Unknown surrogate mode: {mode}")
        return grad_output * grad, None


class ThresholdNet(nn.Module):
    def __init__(self, d: int, width: int, layers: int, mode: str, use_bn: bool) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be >= 2.")
        self.mode = mode
        self.use_bn = use_bn
        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        in_dim = d
        for _ in range(layers - 1):
            self.hidden.append(nn.Linear(in_dim, width))
            if use_bn:
                self.bn.append(nn.BatchNorm1d(width))
            in_dim = width
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.hidden):
            x = layer(x)
            if self.use_bn:
                x = self.bn[idx](x)
            x = SurrogateThreshold.apply(x, self.mode)
        return self.out(x).squeeze(-1)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.sign(logits)
    correct = (pred == y).float().mean().item()
    return correct


def l2_penalty(model: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        total = total + (param ** 2).sum()
    return total


def train_nonconvex(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    config: ExperimentConfig,
    mode: str,
) -> Dict[str, List[float]]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    model = ThresholdNet(
        d=config.d,
        width=config.width,
        layers=config.layers,
        mode=mode,
        use_bn=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        mse = 0.5 * torch.mean((logits - y_train) ** 2)
        loss = mse + config.beta * l2_penalty(model)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history["train_loss"].append(loss.item())
            history["train_acc"].append(accuracy_from_logits(logits, y_train))
            history["test_acc"].append(accuracy_from_logits(model(x_test), y_test))
    return history


def solve_convex_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    beta: float,
    solver: str,
) -> Dict[str, float]:
    w = cp.Variable(x_train.shape[1])
    objective = 0.5 * cp.sum_squares(x_train @ w - y_train) + beta * cp.sum_squares(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver, verbose=False)
    w_val = w.value
    train_logits = x_train @ w_val
    test_logits = x_test @ w_val
    train_acc = np.mean(np.sign(train_logits) == y_train)
    test_acc = np.mean(np.sign(test_logits) == y_test)
    return {
        "train_loss": 0.5 * np.mean((train_logits - y_train) ** 2) + beta * np.sum(w_val ** 2),
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def build_representation(x: np.ndarray, rep_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(x.shape[1], rep_dim)).astype(np.float32)
    x_tilde = (x @ h >= 0).astype(np.float32)
    return x_tilde, h


def plot_curves(
    histories: Dict[str, Dict[str, List[float]]],
    convex_summary: Dict[str, float],
    out_dir: Path,
    show: bool,
) -> List[Path]:
    timestamp = int(time.time())
    epochs = len(next(iter(histories.values()))["train_loss"])
    outputs: List[Path] = []

    fig1 = plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(range(epochs), hist["train_loss"], label=name)
    plt.axhline(convex_summary["train_loss"], color="black", linestyle="--", label="Convex-Ridge")
    plt.title("B.6: Training Loss (10-layer threshold net)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fig1_path = out_dir / f"threshold_b6_train_loss_{timestamp}.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    outputs.append(fig1_path)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(range(epochs), hist["train_acc"], label=name)
    plt.axhline(convex_summary["train_acc"], color="black", linestyle="--", label="Convex-Ridge")
    plt.title("B.6: Training Accuracy (10-layer threshold net)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fig2_path = out_dir / f"threshold_b6_train_acc_{timestamp}.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    outputs.append(fig2_path)
    if show:
        plt.show()
    else:
        plt.close(fig2)

    fig3 = plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(range(epochs), hist["test_acc"], label=name)
    plt.axhline(convex_summary["test_acc"], color="black", linestyle="--", label="Convex-Ridge")
    plt.title("B.6: Test Accuracy (10-layer threshold net)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fig3_path = out_dir / f"threshold_b6_test_acc_{timestamp}.png"
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    outputs.append(fig3_path)
    if show:
        plt.show()
    else:
        plt.close(fig3)

    return outputs


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.debug:
        return ExperimentConfig(
            seed=args.seed,
            n_train=40,
            n_test=200,
            d=10,
            layers=10,
            width=64,
            beta=1e-3,
            lr=1e-2,
            epochs=50,
            num_seeds=2,
            rep_dim=128,
            solver=args.solver,
            debug=True,
        )
    return ExperimentConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        layers=args.layers,
        width=args.width,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        num_seeds=args.num_seeds,
        rep_dim=args.rep_dim,
        solver=args.solver,
        debug=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate Appendix B.6 experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--rep-dim", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    x_train, y_train = generate_ground_truth(config.n_train, config.d, config.seed)
    x_test, y_test = generate_ground_truth(config.n_test, config.d, config.seed + 1)

    x_train_torch = torch.tensor(x_train)
    y_train_torch = torch.tensor(y_train)
    x_test_torch = torch.tensor(x_test)
    y_test_torch = torch.tensor(y_test)

    modes = ["ste", "relu", "lrelu", "crelu"]
    histories: Dict[str, Dict[str, List[float]]] = {}

    for mode in modes:
        best_history = None
        best_loss = None
        for seed_offset in range(config.num_seeds):
            set_seed(config.seed + seed_offset + 10)
            history = train_nonconvex(
                x_train_torch,
                y_train_torch,
                x_test_torch,
                y_test_torch,
                config,
                mode,
            )
            final_loss = history["train_loss"][-1]
            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                best_history = history
        histories[f"Nonconvex-{mode.upper()}"] = best_history

    x_train_rep, h = build_representation(x_train, config.rep_dim, config.seed + 100)
    x_test_rep = (x_test @ h >= 0).astype(np.float32)
    convex_summary = solve_convex_ridge(
        x_train_rep,
        y_train,
        x_test_rep,
        y_test,
        config.beta,
        config.solver,
    )

    out_dir = Path(__file__).parent
    plot_curves(histories, convex_summary, out_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
