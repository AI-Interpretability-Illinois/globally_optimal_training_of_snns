import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import snntorch as snn
from snntorch import datasets, surrogate


@dataclass
class BenchConfig:
    dataset: str
    timesteps: int
    batch_size: int
    hidden: int
    beta: float
    lr: float
    epochs: int
    debug: bool
    limit_train: int
    limit_test: int
    data_root: str


def build_config(args: argparse.Namespace) -> BenchConfig:
    if args.debug:
        return BenchConfig(
            dataset=args.dataset,
            timesteps=args.timesteps,
            batch_size=16,
            hidden=args.hidden,
            beta=args.beta,
            lr=args.lr,
            epochs=1,
            debug=True,
            limit_train=128,
            limit_test=64,
            data_root=args.data_root,
        )
    return BenchConfig(
        dataset=args.dataset,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        hidden=args.hidden,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        debug=False,
        limit_train=0,
        limit_test=0,
        data_root=args.data_root,
    )


def bin_timesteps(spk: torch.Tensor, target_steps: int) -> torch.Tensor:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if spk.dim() < 2:
        raise ValueError("spk tensor must include a time dimension.")
    t = spk.shape[0]
    if target_steps == t:
        return spk
    if target_steps > t:
        reps = int(np.ceil(target_steps / t))
        return spk.repeat_interleave(reps, dim=0)[:target_steps]
    bin_size = int(np.ceil(t / target_steps))
    bins = []
    for idx in range(target_steps):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, t)
        if start >= t:
            break
        chunk = spk[start:end]
        bins.append((chunk.sum(dim=0) > 0).float())
    return torch.stack(bins, dim=0)


def ensure_time_first(spk: torch.Tensor) -> torch.Tensor:
    if spk.dim() < 2:
        raise ValueError("spk tensor must be at least 2D.")
    if spk.shape[0] <= 8 and spk.shape[1] > 8:
        return spk
    return spk.permute(1, 0, *range(2, spk.dim()))


def load_dataset(name: str, root: str, train: bool) -> torch.utils.data.Dataset:
    name = name.lower()
    if name == "nmnist":
        return datasets.NMNIST(root=root, train=train, download=True)
    if name == "shd":
        return datasets.SHD(root=root, train=train, download=True)
    if name == "ssc":
        return datasets.SSC(root=root, train=train, download=True)
    if name == "dvsgesture":
        return datasets.DVSGesture(root=root, train=train, download=True)
    raise ValueError(f"Unsupported dataset '{name}'.")


class LIFNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, beta: float, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1.")
        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(layers):
            self.fcs.append(nn.Linear(in_dim, hidden, bias=False))
            self.lifs.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()))
            in_dim = hidden
        self.fc_out = nn.Linear(in_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mems = [lif.init_leaky() for lif in self.lifs]
        spk_count = 0.0
        for t in range(x.shape[0]):
            h = x[t]
            for idx, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(h)
                h, mems[idx] = lif(cur, mems[idx])
            spk_count = spk_count + self.fc_out(h)
        return spk_count


def flatten_input(spk: torch.Tensor) -> torch.Tensor:
    return spk.reshape(spk.shape[0], spk.shape[1], -1)


def evaluate(model: nn.Module, loader: DataLoader, timesteps: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for spk, y in loader:
            spk = ensure_time_first(spk)
            spk = bin_timesteps(spk, timesteps)
            spk = flatten_input(spk)
            logits = model(spk)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def build_convex_features(loader: DataLoader, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for spk, y in loader:
        spk = ensure_time_first(spk)
        spk = bin_timesteps(spk, timesteps)
        spk = flatten_input(spk)  # (T, B, F)
        spk = spk.permute(1, 0, 2).reshape(spk.shape[1], -1)
        features.append(spk.numpy())
        labels.append(y.numpy())
    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return x, y


def solve_convex_ridge_multiclass(
    x: np.ndarray,
    y: np.ndarray,
    reg: float,
) -> np.ndarray:
    classes = int(np.max(y)) + 1
    y_onehot = np.eye(classes, dtype=np.float32)[y]
    w = cp.Variable((x.shape[1], classes))
    loss = 0.5 * cp.sum_squares(x @ w - y_onehot)
    penalty = 0.5 * reg * cp.sum_squares(w)
    problem = cp.Problem(cp.Minimize(loss + penalty))
    problem.solve(solver="SCS", eps=1e-6, max_iters=200000)
    return w.value


def predict_convex(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    logits = x @ w
    return np.argmax(logits, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="LIF benchmarks with timestep binning.")
    parser.add_argument("--dataset", type=str, default="nmnist")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--convex-reg", type=float, default=1e-2)
    args = parser.parse_args()

    config = build_config(args)
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1).")
    full_set = load_dataset(config.dataset, config.data_root, train=True)
    n_total = len(full_set)
    n_train = max(1, int(round(n_total * args.train_ratio)))
    n_test = n_total - n_train
    if n_test == 0:
        n_test = 1
        n_train = n_total - 1
    train_set = Subset(full_set, list(range(n_train)))
    test_set = Subset(full_set, list(range(n_train, n_train + n_test)))

    if config.limit_train > 0:
        train_set = Subset(train_set, list(range(min(config.limit_train, len(train_set)))))
    if config.limit_test > 0:
        test_set = Subset(test_set, list(range(min(config.limit_test, len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False)

    spk_sample, y_sample = next(iter(train_loader))
    spk_sample = ensure_time_first(spk_sample)
    spk_sample = bin_timesteps(spk_sample, config.timesteps)
    spk_sample = flatten_input(spk_sample)
    input_dim = spk_sample.shape[-1]
    output_dim = int(torch.max(y_sample).item()) + 1

    model = LIFNet(input_dim, config.hidden, output_dim, config.beta, args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for spk, y in train_loader:
            spk = ensure_time_first(spk)
            spk = bin_timesteps(spk, config.timesteps)
            spk = flatten_input(spk)
            optimizer.zero_grad()
            logits = model(spk)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        train_acc = evaluate(model, train_loader, config.timesteps)
        test_acc = evaluate(model, test_loader, config.timesteps)
        print(
            f"[LIF] epoch={epoch} loss={total_loss / max(1, total):.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    final_train_acc = evaluate(model, train_loader, config.timesteps)
    final_test_acc = evaluate(model, test_loader, config.timesteps)
    print(
        f"[LIF] final train_acc={final_train_acc:.4f} "
        f"test_acc={final_test_acc:.4f} timesteps={config.timesteps}"
    )

    t0 = time.perf_counter()
    x_train, y_train = build_convex_features(train_loader, config.timesteps)
    x_test, y_test = build_convex_features(test_loader, config.timesteps)
    w = solve_convex_ridge_multiclass(x_train, y_train, args.convex_reg)
    train_preds = predict_convex(x_train, w)
    test_preds = predict_convex(x_test, w)
    train_acc = float(np.mean(train_preds == y_train))
    test_acc = float(np.mean(test_preds == y_test))
    print(
        f"[CVX] ridge reg={args.convex_reg:.2e} "
        f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
        f"elapsed_s={time.perf_counter() - t0:.2f}"
    )


if __name__ == "__main__":
    main()
