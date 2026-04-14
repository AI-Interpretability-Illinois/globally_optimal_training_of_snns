from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FetchSpec = Tuple[str, Union[int, str]]

BENCHMARK_SPECS: Dict[str, Dict[str, object]] = {
    "bank": {"expected_d": 16, "candidates": [("data_id", 1558)]},
    "chess_krvkp": {"expected_d": 36, "candidates": [("data_id", 3), ("name", "kr-vs-kp")]},
    "mammographic": {"expected_d": 5, "candidates": [("data_id", 45557), ("name", "mammography")]},
    "ozone": {"expected_d": 72, "candidates": [("data_id", 1487), ("name", "ozone_level")]},
    "pima": {"expected_d": 8, "candidates": [("data_id", 37), ("name", "diabetes")]},
    "spambase": {"expected_d": 57, "candidates": [("data_id", 44), ("name", "spambase")]},
    "statlog_german": {"expected_d": 24, "candidates": [("data_id", 46918), ("name", "credit-g")]},
    "tic_tac_toe": {"expected_d": 9, "candidates": [("data_id", 50), ("name", "tic-tac-toe")]},
    "titanic": {"expected_d": 3, "candidates": [("name", "Titanic")]},
}


@dataclass
class UciDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str


def _encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).fillna("missing").astype("category").cat.codes
    return out


def _map_binary_target(series: pd.Series) -> np.ndarray:
    vals = series.astype(str).str.strip().to_numpy()
    uniq = np.unique(vals)
    if len(uniq) != 2:
        raise ValueError(f"Expected binary target, got classes {uniq}")
    return np.where(vals == uniq[-1], 1, 0).astype(np.int64)


def _fetch_openml_best_effort(candidates: List[FetchSpec]) -> Tuple[pd.DataFrame, str]:
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
        except Exception as exc:  # network/data-source errors only
            last_error = exc
            continue
    raise RuntimeError(f"Failed to fetch dataset for candidates={candidates}. Last error: {last_error}")


def load_uci_benchmark(name: str) -> Tuple[np.ndarray, np.ndarray]:
    spec = BENCHMARK_SPECS[name]
    expected_d = int(spec["expected_d"])
    frame, target_col = _fetch_openml_best_effort(spec["candidates"])  # type: ignore[arg-type]
    y = _map_binary_target(frame[target_col])
    x_df = frame.drop(columns=[target_col]).replace({"?": np.nan}).copy()
    x_df = _encode_categorical_columns(x_df)
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.fillna(x_df.mean(numeric_only=True))
    if x_df.shape[1] != expected_d:
        raise ValueError(f"{name}: expected raw d={expected_d}, got d={x_df.shape[1]}")
    return x_df.to_numpy(dtype=np.float32), y


def load_uci_dataset(
    *,
    name: str,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    standardize: bool = True,
) -> UciDataset:
    x, y = load_uci_benchmark(name=name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=seed + 1,
        stratify=y_train,
    )
    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
    return UciDataset(
        X_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        X_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        name=name,
    )
