import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from snn_experiments import (
    apply_representation,
    build_snn_arrangements,
    create_representation_weights,
    eval_leaky_snn,
    forward_reconstructed,
    generate_moving_gaussian_blobs,
    generate_rotated_mnist_pairs,
    generate_two_step_xor,
    reconstruct_weights,
    select_activation_patterns,
    set_seed,
    solve_dual_convex_snn,
    solve_primal_convex_snn,
    solve_svm_hyperplane,
    train_leaky_snn,
)


@dataclass
class SnnEmulationConfig:
    seed: int
    n_train: int
    n_total: int
    m1: int
    m2: int
    m_repr: int
    beta: float
    arr_samples: int
    arr_sample_factor: int
    lr: float
    epochs: int
    svm_C: float
    solver: str
    solver_opts: Dict[str, float]
    sampled_arrangements: bool
    allow_pattern_repeats: bool
    dataset: str
    debug: bool


def build_config(args: argparse.Namespace) -> SnnEmulationConfig:
    if args.solver.upper() == "SCS":
        solver_opts = {"eps": 1e-9, "max_iters": 200000}
    else:
        solver_opts = {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9, "max_iters": 50000}

    if args.debug:
        return SnnEmulationConfig(
            seed=args.seed,
            n_train=40,
            n_total=200,
            m1=20,
            m2=20,
            m_repr=200,
            beta=1e-3,
            arr_samples=200,
            arr_sample_factor=0,
            lr=1e-2,
            epochs=50,
            svm_C=1000.0,
            solver=args.solver,
            solver_opts=solver_opts,
            sampled_arrangements=not args.full_arrangements,
            allow_pattern_repeats=args.allow_pattern_repeats,
            dataset=args.dataset,
            debug=True,
        )

    return SnnEmulationConfig(
        seed=args.seed,
        n_train=100,
        n_total=3000,
        m1=100,
        m2=120,
        m_repr=1000,
        beta=1e-3,
        arr_samples=1000,
        arr_sample_factor=0,
        lr=1e-2,
        epochs=500,
        svm_C=1000.0,
        solver=args.solver,
        solver_opts=solver_opts,
        sampled_arrangements=not args.full_arrangements,
        allow_pattern_repeats=args.allow_pattern_repeats,
        dataset=args.dataset,
        debug=False,
    )


def load_dataset(
    name: str,
    n_total: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "two_step_xor":
        return generate_two_step_xor(n_total, seed)
    if name == "moving_gaussian":
        return generate_moving_gaussian_blobs(n_total, seed)
    if name == "rotated_mnist":
        return generate_rotated_mnist_pairs(n_total, seed)
    raise ValueError(f"Unknown dataset '{name}'.")


def plot_bars(
    labels: Tuple[str, ...],
    values: Tuple[float, ...],
    title: str,
    ylabel: str,
    out_path: Path,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN hyperplane arrangement emulation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full-arrangements", action="store_true")
    parser.add_argument("--allow-pattern-repeats", action="store_true")
    parser.add_argument("--dataset", type=str, default="moving_gaussian")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    x1, x2, y = load_dataset(config.dataset, config.n_train + config.n_total, config.seed + 1)
    x1_train = x1[: config.n_train]
    x2_train = x2[: config.n_train]
    y_train = y[: config.n_train]
    x1_test = x1[config.n_train : config.n_train + config.n_total]
    x2_test = x2[config.n_train : config.n_train + config.n_total]
    y_test = y[config.n_train : config.n_train + config.n_total]

    h_repr = create_representation_weights(x1.shape[1], config.m_repr, config.seed + 11)
    x1_train_tilde = apply_representation(x1_train, h_repr)
    x2_train_tilde = apply_representation(x2_train, h_repr)
    x1_test_tilde = apply_representation(x1_test, h_repr)
    x2_test_tilde = apply_representation(x2_test, h_repr)

    arr_samples = max(config.arr_samples, config.m2 * config.arr_sample_factor)
    d_arr = build_snn_arrangements(
        x1_train_tilde,
        x2_train_tilde,
        config.seed,
        arr_samples,
        config.sampled_arrangements,
    )
    d22 = d_arr["D22"]

    primal = solve_primal_convex_snn(
        d22,
        y_train,
        config.beta,
        config.m2,
        config.solver,
        config.solver_opts,
    )
    dual = solve_dual_convex_snn(
        d22,
        y_train,
        config.beta,
        config.m2,
        config.solver,
        config.solver_opts,
    )
    gap = primal["primal"] - dual["dual"]

    m1_eff = min(config.m1, d_arr["D11"].shape[1], d_arr["D12"].shape[1])
    h1_1, h1_2, h2_1, h2_2 = select_activation_patterns(
        d_arr["D11"],
        d_arr["D12"],
        d_arr["D21"],
        d_arr["D22"],
        primal["w"],
        m1_eff,
        config.m2,
        config.allow_pattern_repeats,
    )
    P1_in, P1_rec, P2_in, P2_rec = reconstruct_weights(
        x1_train_tilde,
        x2_train_tilde,
        h1_1,
        h1_2,
        h2_1,
        h2_2,
        config.svm_C,
        config.solver,
        config.solver_opts,
    )

    h2_train_hat = forward_reconstructed(
        x1_train_tilde,
        x2_train_tilde,
        P1_in,
        P1_rec,
        P2_in,
        P2_rec,
    )
    h2_test_hat = forward_reconstructed(
        x1_test_tilde,
        x2_test_tilde,
        P1_in,
        P1_rec,
        P2_in,
        P2_rec,
    )
    v = solve_svm_hyperplane(
        h2_train_hat,
        y_train,
        config.svm_C,
        config.solver,
        config.solver_opts,
    )
    recon_test_preds = np.sign(h2_test_hat @ v)
    recon_test_acc = float(np.mean(recon_test_preds == y_test))
    convex_train_acc = float(np.mean(np.sign(d22 @ primal["w"]) == y_train))

    lif = train_leaky_snn(
        x1_train_tilde,
        x2_train_tilde,
        y_train,
        m1_eff,
        config.m2,
        config.lr,
        config.epochs,
    )
    lif_test_preds = eval_leaky_snn(lif["layers"], x1_test_tilde, x2_test_tilde)
    lif_test_acc = float(np.mean(lif_test_preds == y_test))

    print(
        f"[SNN:{config.dataset}] D11={d_arr['D11'].shape} "
        f"D12={d_arr['D12'].shape} D21={d_arr['D21'].shape} D22={d_arr['D22'].shape}"
    )
    print(
        f"[SNN:{config.dataset}] convex_train_acc={convex_train_acc:.3f} "
        f"recon_test_acc={recon_test_acc:.3f} "
        f"lif_train_acc={lif['acc']:.3f} lif_test_acc={lif_test_acc:.3f} "
        f"gap={gap:.6f}"
    )

    out_dir = Path(__file__).parent
    timestamp = int(time.time())
    plot_bars(
        ("D11", "D12", "D21", "D22"),
        (
            float(d_arr["D11"].shape[1]),
            float(d_arr["D12"].shape[1]),
            float(d_arr["D21"].shape[1]),
            float(d_arr["D22"].shape[1]),
        ),
        f"SNN Hyperplane Arrangements ({config.dataset})",
        "Number of patterns",
        out_dir / f"snn_arrangements_{timestamp}.png",
        show=not args.no_show,
    )
    plot_bars(
        ("Convex train", "Recon test", "LIF train", "LIF test"),
        (convex_train_acc, recon_test_acc, lif["acc"], lif_test_acc),
        f"SNN Accuracy Summary ({config.dataset})",
        "Accuracy",
        out_dir / f"snn_accuracy_{timestamp}.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
