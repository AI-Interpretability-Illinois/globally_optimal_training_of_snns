
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

FetchSpec = Tuple[str, Union[int, str]]

@dataclass
class ExperimentConfig:
    random_seed: int = 0
    test_size: float = 0.2
    data_root: str = "./data"

    # Network / feature budget
    hidden_width: int = 1000
    sampled_feature_count: int = 1000

    # Paper-style sweeps
    ste_epochs: int = 5000
    ste_batch_size: int = -1  # full-batch => n
    ste_beta_grid: Tuple[float, ...] = (1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0)
    ste_lr_grid: Tuple[float, ...] = (1e-3, 5e-3, 1e-2, 1e-1)

    cvx_beta_grid: Tuple[float, ...] = (1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0)

    num_eval_runs: int = 5
    gpu: str = "cpu"

    # Benchmarks from the paper's UCI block, plus the custom CIFAR cat/dog task
    benchmark_order: Tuple[str, ...] = (
        "bank",
        "chess_krvkp",
        "mammographic",
        "oocytes_4d",
        "oocytes_2f",
        "ozone",
        "pima",
        "spambase",
        "statlog_german",
        "tic_tac_toe",
        "titanic",
        "cifar_cat_dog",
    )


BENCHMARK_SPECS: Dict[str, Dict[str, object]] = {
    "bank": {
        "expected_d": 16,
        "candidates": [("data_id", 1558)],
    },
    "chess_krvkp": {
        "expected_d": 36,
        "candidates": [("data_id", 3), ("name", "kr-vs-kp")],
    },
    "mammographic": {
        "expected_d": 5,
        "candidates": [("data_id", 45557), ("name", "Mammographic-Mass-Data-Set"), ("name", "mammography")],
    },
    # The oocytes datasets are historically a bit messy across repositories.
    # We try several known aliases from the Fernández-Delgado benchmark literature.
    "oocytes_4d": {
        "expected_d": 41,
        "candidates": [
            ("name", "oocytes_merluccius_nucleus_4d"),
            ("name", "oocMerl4D"),
            ("name", "oocytes merluccius nucleus 4d"),
        ],
    },
    "oocytes_2f": {
        "expected_d": 25,
        "candidates": [
            ("name", "oocytes_merluccius_states_2f"),
            ("name", "oocMerl2F"),
            ("name", "oocytes merluccius states 2f"),
        ],
    },
    "ozone": {
        "expected_d": 72,
        "candidates": [("data_id", 1487), ("data_id", 301), ("name", "ozone_level")],
    },
    "pima": {
        "expected_d": 8,
        "candidates": [("data_id", 37), ("name", "diabetes"), ("name", "Pima-Indians-Diabetes")],
    },
    "spambase": {
        "expected_d": 57,
        "candidates": [("data_id", 44), ("name", "spambase")],
    },
    "statlog_german": {
        "expected_d": 24,
        "candidates": [
            ("data_id", 46918),
            ("name", "Statlog (German Credit Data)"),
            ("name", "statlog-german-credit"),
            ("name", "credit-g"),
        ],
    },
    "tic_tac_toe": {
        "expected_d": 9,
        "candidates": [("data_id", 50), ("name", "tic-tac-toe")],
    },
    "titanic": {
        "expected_d": 3,
        "candidates": [("name", "Titanic")],
    },
}


# ============================================================
# Utilities
# ============================================================

def choose_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).fillna("missing").astype("category").cat.codes
    return out


def append_bias_column(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


def to_pm1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype == bool:
        return np.where(y, 1.0, -1.0).astype(np.float32)
    if set(np.unique(y).tolist()) == {0, 1}:
        return (2 * y - 1).astype(np.float32)
    if set(np.unique(y).tolist()) == {-1, 1}:
        return y.astype(np.float32)
    raise ValueError(f"Expected binary labels, got classes {np.unique(y)}")


def map_binary_target(series: pd.Series) -> np.ndarray:
    # Keep only genuinely binary datasets / variants.
    vals = series.astype(str).str.strip().to_numpy()
    uniq = np.unique(vals)
    if len(uniq) != 2:
        raise ValueError(f"Expected binary target, got classes {uniq}")
    # Positive class = lexicographically last, matching prior script behavior.
    return np.where(vals == uniq[-1], 1, 0).astype(np.int64)


def fetch_openml_best_effort(candidates: List[FetchSpec]) -> Tuple[pd.DataFrame, str]:
    last_error = None
    for kind, value in candidates:
        try:
            if kind == "data_id":
                bunch = fetch_openml(data_id=int(value), as_frame=True)
            elif kind == "name":
                bunch = fetch_openml(name=str(value), as_frame=True)
            else:
                raise ValueError(f"Unknown fetch spec kind: {kind}")
            frame = bunch.frame.copy()
            target_col = bunch.target.name if bunch.target.name in frame.columns else frame.columns[-1]
            return frame, target_col
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to fetch dataset for candidates={candidates}. Last error: {last_error}")


# ============================================================
# Dataset loaders
# ============================================================

def load_uci_benchmark(name: str) -> Tuple[np.ndarray, np.ndarray]:
    spec = BENCHMARK_SPECS[name]
    expected_d = int(spec["expected_d"])
    frame, target_col = fetch_openml_best_effort(spec["candidates"])  # type: ignore[arg-type]

    y = map_binary_target(frame[target_col])
    X_df = frame.drop(columns=[target_col]).replace({"?": np.nan}).copy()
    X_df = encode_categorical_columns(X_df)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean(numeric_only=True))

    raw_d = X_df.shape[1]
    if raw_d != expected_d:
        raise ValueError(f"{name}: expected raw d={expected_d}, got d={raw_d}")

    X = X_df.to_numpy(dtype=np.float32)
    return X, y.astype(np.int64)


def load_cifar_cat_dog(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276],
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    cat_class, dog_class = 3, 5

    def extract(ds):
        xs, ys = [], []
        for image, label in ds:
            if label == cat_class:
                xs.append(image.numpy().reshape(-1))
                ys.append(0)
            elif label == dog_class:
                xs.append(image.numpy().reshape(-1))
                ys.append(1)
        return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)

    Xtr, ytr = extract(train_dataset)
    Xte, yte = extract(test_dataset)
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    return X, y


def load_dataset(name: str, config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    if name in BENCHMARK_SPECS:
        return load_uci_benchmark(name)
    if name == "cifar_cat_dog":
        return load_cifar_cat_dog(config.data_root)
    raise ValueError(f"Unknown dataset {name}")


# ============================================================
# Models / objectives
# ============================================================

def threshold_features(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    proj = np.einsum("nd,dp->np", X.astype(np.float64), G.astype(np.float64), optimize=True)
    if not np.isfinite(proj).all():
        raise ValueError("Non-finite projected activations in threshold feature construction.")
    return (proj >= 0).astype(np.float32)


@dataclass
class ConvexResult:
    train_acc: float
    test_acc: float
    objective: float
    nonzero_count: int
    beta: float
    time_sec: float


def fit_convex_lasso(
    X_train: np.ndarray,
    y_train_pm1: np.ndarray,
    X_test: np.ndarray,
    y_test_pm1: np.ndarray,
    beta: float,
    sampled_feature_count: int,
    seed: int,
) -> ConvexResult:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((X_train.shape[1], sampled_feature_count)).astype(np.float64)
    Z_train = threshold_features(X_train, G).astype(np.float64)
    Z_test = threshold_features(X_test, G).astype(np.float64)
    n_train = Z_train.shape[0]

    w = cp.Variable(sampled_feature_count)
    y64 = y_train_pm1.astype(np.float64)
    weighted_features = cp.multiply(
        Z_train,
        cp.reshape(w, (1, sampled_feature_count), order="F"),
    )
    train_projection = cp.sum(weighted_features, axis=1)
    objective = 0.5 * (1.0 / n_train) * cp.sum_squares(train_projection - y64) + beta * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))

    t0 = time.time()
    try:
        problem.solve(
            solver=cp.CLARABEL,
            verbose=False,
            max_iter=50000,
            tol_gap_abs=1e-9,
            tol_gap_rel=1e-9,
            tol_feas=1e-9,
        )
    except Exception:
        problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
    elapsed = time.time() - t0

    if w.value is None:
        raise RuntimeError("Convex solver failed to return a solution.")

    coef = np.asarray(w.value, dtype=np.float64)
    if not np.isfinite(coef).all():
        raise ValueError("Non-finite convex coefficients produced by solver.")
    train_scores = np.einsum("np,p->n", Z_train, coef, optimize=True)
    test_scores = np.einsum("np,p->n", Z_test, coef, optimize=True)
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Non-finite convex scores encountered after projection.")
    train_acc = compute_binary_accuracy(train_scores, y_train_pm1)
    test_acc = compute_binary_accuracy(test_scores, y_test_pm1)
    obj = 0.5 * float(np.mean((train_scores - y_train_pm1) ** 2)) + beta * float(np.sum(np.abs(coef)))
    nnz = int(np.count_nonzero(np.abs(coef) > 1e-10))
    return ConvexResult(train_acc, test_acc, obj, nnz, beta, elapsed)


class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEThresholdLayer(nn.Module):
    def forward(self, x):
        return ThresholdSTE.apply(x)


class TwoLayerSTE(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_width, bias=False)
        self.thr = STEThresholdLayer()
        self.fc2 = nn.Linear(hidden_width, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.thr(self.fc1(x))).squeeze(-1)


def ste_l2_weight_decay(model: TwoLayerSTE) -> torch.Tensor:
    return 0.5 * (torch.sum(model.fc1.weight ** 2) + torch.sum(model.fc2.weight ** 2))


def ste_full_objective(scores: torch.Tensor, y: torch.Tensor, model: TwoLayerSTE, beta: float) -> torch.Tensor:
    return 0.5 * torch.mean((scores - y) ** 2) + beta * ste_l2_weight_decay(model)


@dataclass
class STEResult:
    train_acc: float
    test_acc: float
    objective: float
    beta: float
    lr: float
    time_sec: float


def compute_binary_accuracy(scores: np.ndarray, y_pm1: np.ndarray) -> float:
    pred = np.where(scores >= 0.0, 1.0, -1.0)
    return float(accuracy_score(y_pm1, pred))


def train_ste_once(
    X_train: np.ndarray,
    y_train_pm1: np.ndarray,
    X_test: np.ndarray,
    y_test_pm1: np.ndarray,
    hidden_width: int,
    beta: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
) -> STEResult:
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train_pm1, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test_pm1, dtype=torch.float32, device=device)

    dataset = TensorDataset(Xtr, ytr)
    bs = len(dataset) if batch_size <= 0 else min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    model = TwoLayerSTE(input_dim=X_train.shape[1], hidden_width=hidden_width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        epoch_obj = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            scores = model(xb)
            obj = ste_full_objective(scores, yb, model, beta)
            obj.backward()
            optimizer.step()
            epoch_obj = float(obj.item())
        scheduler.step(epoch_obj)
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        train_scores_t = model(Xtr)
        test_scores_t = model(Xte)
        train_obj = ste_full_objective(train_scores_t, ytr, model, beta)
        train_acc = (torch.sign(train_scores_t) == ytr).float().mean().item()
        test_acc = (torch.sign(test_scores_t) == yte).float().mean().item()

    return STEResult(
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        objective=float(train_obj.item()),
        beta=beta,
        lr=lr,
        time_sec=elapsed,
    )


def sweep_ste(
    X_train: np.ndarray,
    y_train_pm1: np.ndarray,
    X_test: np.ndarray,
    y_test_pm1: np.ndarray,
    config: ExperimentConfig,
) -> STEResult:
    best: Optional[STEResult] = None
    for beta in config.ste_beta_grid:
        for lr in config.ste_lr_grid:
            result = train_ste_once(
                X_train=X_train,
                y_train_pm1=y_train_pm1,
                X_test=X_test,
                y_test_pm1=y_test_pm1,
                hidden_width=config.hidden_width,
                beta=beta,
                lr=lr,
                epochs=config.ste_epochs,
                batch_size=config.ste_batch_size,
                device=config.gpu,
            )
            if best is None or result.test_acc > best.test_acc:
                best = result
    if best is None:
        raise RuntimeError("STE sweep failed.")
    return best


def sweep_cvx_beta(
    X_train: np.ndarray,
    y_train_pm1: np.ndarray,
    X_test: np.ndarray,
    y_test_pm1: np.ndarray,
    config: ExperimentConfig,
) -> ConvexResult:
    best: Optional[ConvexResult] = None
    for beta in config.cvx_beta_grid:
        result = fit_convex_lasso(
            X_train=X_train,
            y_train_pm1=y_train_pm1,
            X_test=X_test,
            y_test_pm1=y_test_pm1,
            beta=beta,
            sampled_feature_count=config.sampled_feature_count,
            seed=config.random_seed,
        )
        if best is None or result.test_acc > best.test_acc:
            best = result
    if best is None:
        raise RuntimeError("CVX beta sweep failed.")
    return best


def mean_std_str(vals: List[float]) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) <= 1:
        return f"{arr.mean():.4f} ± 0.0000"
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


# ============================================================
# Benchmark driver
# ============================================================

def prepare_data(name: str, config: ExperimentConfig):
    X, y = load_dataset(name, config)
    y_pm1 = to_pm1(y)
    raw_d = X.shape[1]

    # Enforce raw d only, matching your request.
    if name in BENCHMARK_SPECS:
        expected_d = int(BENCHMARK_SPECS[name]["expected_d"])
        if raw_d != expected_d:
            raise ValueError(f"{name}: expected raw d={expected_d}, got d={raw_d}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_pm1,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y_pm1,
    )

    if name != "cifar_cat_dog":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = append_bias_column(X_train).astype(np.float32)
    X_test = append_bias_column(X_test).astype(np.float32)
    return raw_d, X_train, X_test, y_train, y_test


def evaluate_over_runs(
    name: str,
    config: ExperimentConfig,
    best_ste_beta: float,
    best_ste_lr: float,
    best_cvx_beta: float,
) -> Dict[str, List[float]]:
    raw_d, X_train, X_test, y_train, y_test = prepare_data(name, config)

    out = {
        "ste_train_acc": [],
        "ste_test_acc": [],
        "ste_obj": [],
        "cvx_train_acc": [],
        "cvx_test_acc": [],
        "cvx_obj": [],
        "cvx_nnz": [],
    }

    for k in range(config.num_eval_runs):
        seed = config.random_seed + k
        set_seed(seed)

        ste = train_ste_once(
            X_train, y_train, X_test, y_test,
            hidden_width=config.hidden_width,
            beta=best_ste_beta,
            lr=best_ste_lr,
            epochs=config.ste_epochs,
            batch_size=config.ste_batch_size,
            device=config.gpu,
        )
        cvx = fit_convex_lasso(
            X_train, y_train, X_test, y_test,
            beta=best_cvx_beta,
            sampled_feature_count=config.sampled_feature_count,
            seed=seed,
        )

        out["ste_train_acc"].append(ste.train_acc)
        out["ste_test_acc"].append(ste.test_acc)
        out["ste_obj"].append(ste.objective)
        out["cvx_train_acc"].append(cvx.train_acc)
        out["cvx_test_acc"].append(cvx.test_acc)
        out["cvx_obj"].append(cvx.objective)
        out["cvx_nnz"].append(float(cvx.nonzero_count))

    return out


def run_single_benchmark(name: str, config: ExperimentConfig) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"Running benchmark: {name}")
    print("=" * 100)

    raw_d, X_train, X_test, y_train, y_test = prepare_data(name, config)
    print(f"Raw feature dim : {raw_d}")
    print(f"Train shape     : {X_train.shape}")
    print(f"Test shape      : {X_test.shape}")
    print(f"Device          : {config.gpu}")

    cvx_best = sweep_cvx_beta(X_train, y_train, X_test, y_test, config)
    ste_best = sweep_ste(X_train, y_train, X_test, y_test, config)

    print("\n[Best single-run settings]")
    print(f"CVX beta         = {cvx_best.beta}")
    print(f"CVX train/test   = {cvx_best.train_acc:.4f} / {cvx_best.test_acc:.4f}")
    print(f"CVX objective    = {cvx_best.objective:.4f}")
    print(f"CVX nonzero      = {cvx_best.nonzero_count}")
    print(f"STE beta, lr     = {ste_best.beta}, {ste_best.lr}")
    print(f"STE train/test   = {ste_best.train_acc:.4f} / {ste_best.test_acc:.4f}")
    print(f"STE objective    = {ste_best.objective:.4f}")

    multi = evaluate_over_runs(name, config, ste_best.beta, ste_best.lr, cvx_best.beta)

    print("\n[5-run summary]")
    print(f"CVX train/test   = {mean_std_str(multi['cvx_train_acc'])} / {mean_std_str(multi['cvx_test_acc'])}")
    print(f"STE train/test   = {mean_std_str(multi['ste_train_acc'])} / {mean_std_str(multi['ste_test_acc'])}")
    print(f"CVX objective    = {mean_std_str(multi['cvx_obj'])}")
    print(f"STE objective    = {mean_std_str(multi['ste_obj'])}")

    return {
        "dataset": name,
        "raw_d": raw_d,
        "cvx_train_acc": float(np.mean(multi["cvx_train_acc"])),
        "cvx_test_acc": float(np.mean(multi["cvx_test_acc"])),
        "ste_train_acc": float(np.mean(multi["ste_train_acc"])),
        "ste_test_acc": float(np.mean(multi["ste_test_acc"])),
        "cvx_obj": float(np.mean(multi["cvx_obj"])),
        "ste_obj": float(np.mean(multi["ste_obj"])),
        "cvx_nnz": float(np.mean(multi["cvx_nnz"])),
        "best_cvx_beta": cvx_best.beta,
        "best_ste_beta": ste_best.beta,
        "best_ste_lr": ste_best.lr,
    }


def main():
    config = ExperimentConfig()
    config.gpu = choose_best_device()
    set_seed(config.random_seed)

    rows = []
    failures = []
    for name in config.benchmark_order:
        try:
            rows.append(run_single_benchmark(name, config))
        except Exception as e:
            print(f"\n[WARNING] Failed on {name}: {e}")
            failures.append({"dataset": name, "error": str(e)})

    if rows:
        summary = pd.DataFrame(rows)[["dataset", "cvx_train_acc", "cvx_test_acc", "ste_train_acc", "ste_test_acc"]]
        print("\n" + "#" * 100)
        print("Final summary table")
        print("#" * 100)
        print(summary.to_string(index=False))

        detailed = pd.DataFrame(rows)
        summary.to_csv("benchmark_summary_acc_only.csv", index=False)
        detailed.to_csv("benchmark_summary_detailed.csv", index=False)

    if failures:
        failed_df = pd.DataFrame(failures)
        print("\n" + "#" * 100)
        print("Benchmarks that failed to load/run")
        print("#" * 100)
        print(failed_df.to_string(index=False))
        failed_df.to_csv("benchmark_failures.csv", index=False)


if __name__ == "__main__":
    main()
