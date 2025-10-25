from __future__ import annotations

from typing import Tuple

import numpy as np


def fill_missing_daily(dates: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if dates.size == 0:
        return dates, y
    start = dates[0]
    end = dates[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    date_to_qty = {int(dates[i].astype("datetime64[D]").astype(int)): y[i] for i in range(len(dates))}
    full_qty = np.zeros(full_dates.shape[0], dtype=float)
    for i, d in enumerate(full_dates):
        key = int(d.astype("datetime64[D]").astype(int))
        full_qty[i] = float(date_to_qty.get(key, 0.0))
    return full_dates, full_qty


def train_test_split_series(y: np.ndarray, test_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if test_size <= 0 or test_size >= y.size:
        return y, np.array([], dtype=float)
    return y[:-test_size], y[-test_size:]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
