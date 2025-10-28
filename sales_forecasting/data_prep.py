from __future__ import annotations

from typing import Tuple

import numpy as np


def create_temporal_features(dates: np.ndarray) -> np.ndarray:
    """Crea features temporales a partir de un array de fechas.
    
    Args:
        dates: Array de fechas en formato datetime64[D]
    
    Returns:
        Array (N, 6) con features temporales:
          0: day_of_week (0=Lunes, 6=Domingo)
          1: day_of_month (1-31)
          2: month (1-12)
          3: quarter (1-4)
          4: is_weekend (0/1)
          5: week_of_year (1-53)
    """
    if dates.size == 0:
        return np.array([], dtype=float).reshape(0, 6)
    
    # Convertir a datetime64[D] si no lo está
    dates = dates.astype('datetime64[D]')
    
    # Day of week: 0=Monday, 6=Sunday
    day_of_week = (dates.astype('datetime64[D]').view('int64') - 4) % 7
    
    # Extraer año, mes, día
    dates_dt = dates.astype('datetime64[M]')
    months = (dates_dt.view('int64') % 12) + 1
    years = dates_dt.view('int64') // 12 + 1970
    
    # Day of month
    days_from_epoch = dates.astype('datetime64[D]').view('int64')
    month_starts = dates_dt.astype('datetime64[D]').view('int64')
    day_of_month = days_from_epoch - month_starts + 1
    
    # Quarter
    quarter = ((months - 1) // 3) + 1
    
    # Is weekend (Saturday=5, Sunday=6)
    is_weekend = (day_of_week >= 5).astype(float)
    
    # Week of year (aproximado)
    year_starts = np.array([np.datetime64(f'{y}-01-01', 'D') for y in years])
    days_since_year_start = (dates - year_starts).astype('timedelta64[D]').astype(int)
    week_of_year = (days_since_year_start // 7) + 1
    
    # Combinar todas las features
    features = np.column_stack([
        day_of_week.astype(float),
        day_of_month.astype(float),
        months.astype(float),
        quarter.astype(float),
        is_weekend,
        week_of_year.astype(float)
    ])
    
    return features


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
