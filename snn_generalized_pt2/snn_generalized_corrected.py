import time
from pathlib import Path
from typing import List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from snn_generalized_pt2.convex_snn import Convex_SNN

def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; using CPU.")
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def log_lif_baseline_results(
    out_path: Path,
    layers: int,
    timesteps: int,
    width: int,
    input_dim: int,
    train_ratio: float,
    dataset_name: str,
    lif_train_acc_history: List[float],
    lif_train_acc: float,
    lif_test_acc: float,
    total_elapsed_s: float,
) -> None:
    lif_acc_path = out_path.with_name(
        f"snn_generalized_lt_L{layers}_T{timesteps}_split_{train_ratio:.2f}_"
        f"{dataset_name}_lif_train_acc.png"
    )
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(lif_train_acc_history) + 1), lif_train_acc_history, marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title(f"LIF Baseline Train Accuracy (L={layers}, T={timesteps}, {dataset_name})")
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(lif_acc_path, dpi=150)
    plt.close()
    append_lines(
        out_path,
        [
            f"RUN snn_generalized_corrected dataset={dataset_name} layers={layers} "
            f"timesteps={timesteps} train_ratio={train_ratio}",
            f"layer_params layers={layers} width={width} input_dim={input_dim} lif_beta=0.9",
            f"layer_dims {describe_layer_params(layers, input_dim, width)}",
            f"lif_train_acc={lif_train_acc:.6f} lif_test_acc={lif_test_acc:.6f}",
            f"lif_train_acc_plot={lif_acc_path.name}",
            f"total_elapsed_s={total_elapsed_s:.2f}",
            "",
        ],
    )


def run_moving_gaussian_convex_vs_lif(
    seed: int = 0,
    layers: int = 2,
    timesteps: int = 2,
    width: int = 3,
    n_total: int = 20,
    train_ratio: float = 0.8,
    d: int = 6,
    beta: float = 1e-3,
    lr: float = 1e-2,
    epochs: int = 50,
    weight_decay: float = 1e-4,
    sampled: bool = False,
    max_hyperplanes: int = 0,
) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1).")
    n_train = max(1, int(round(n_total * train_ratio)))
    x_list, y = generate_moving_gaussian_sequence(n_total, d, timesteps, seed)
    x_train, y_train, x_test, y_test = split_sequence(x_list, y, n_train)

    widths = np.array([width] * layers)
    convex = Convex_SNN(layers=layers, widths=widths, input_dim=d, beta=beta)
    t0 = time.perf_counter()
    convex.train(x_train, y_train, sampled=sampled, num_samples=max_hyperplanes)
    convex_train_acc = float(convex.training_accuracy)
    convex_test_acc = float(convex.test(x_test, y_test))
    convex_elapsed = time.perf_counter() - t0

    lif_train_acc, lif_hist, lif_model = train_lif_baseline(
        x_train,
        y_train,
        layers,
        width,
        lr,
        epochs,
        weight_decay,
    )
    lif_test_preds = eval_lif_baseline(lif_model, x_test)
    lif_test_acc = float(np.mean(lif_test_preds == y_test))

    print(
        f"[Corrected] moving_gaussian L={layers} T={timesteps} "
        f"convex_train_acc={convex_train_acc:.3f} convex_test_acc={convex_test_acc:.3f} "
        f"lif_train_acc={lif_train_acc:.3f} lif_test_acc={lif_test_acc:.3f} "
        f"elapsed_s={convex_elapsed:.2f}"
    )


if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Convex vs LIF test on moving_gaussian.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--n-total", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--max-hyperplanes", type=int, default=10000)
    args = parser.parse_args()

    try:
        run_moving_gaussian_convex_vs_lif(
            seed=args.seed,
            layers=args.layers,
            timesteps=args.timesteps,
            width=args.width,
            n_total=args.n_total,
            train_ratio=args.train_ratio,
            d=args.d,
            beta=args.beta,
            lr=args.lr,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
        sampled=args.sampled,
        max_hyperplanes=args.max_hyperplanes,
        )
    except Exception:
        traceback.print_exc()
        raise
