from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_lags(lags: Iterable[int]) -> list[int]:
    parsed = sorted({int(lag) for lag in lags if int(lag) > 0})
    return parsed


def _select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric_cols:
        return numeric_cols
    return numeric_cols


def _build_sliding_windows(
    values: np.ndarray,
    target: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size debe ser mayor a 0")
    if len(values) <= window_size:
        raise ValueError("No hay suficientes filas para construir ventanas.")

    x_list = []
    y_list = []
    for idx in range(window_size, len(values)):
        x_list.append(values[idx - window_size: idx])
        y_list.append(target[idx])
    return np.asarray(x_list), np.asarray(y_list)


def build_features(
    input_path: str | Path = "reports/eda/data/combined.jsonl",
    output_dir: str | Path = "reports/eda/features",
    lags: Iterable[int] = (1, 7, 30),
    target_col: str = "orders",
    window_size: int = 28,
) -> dict[str, Path]:
    """
    Construye features con lags y revenue_growth, y genera ventanas deslizantes.
    Retorna rutas de salida generadas.
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el dataset combinado: {in_path}")

    df = pd.read_json(in_path, lines=True, dtype=False)
    if "created" in df.columns:
        df["created"] = pd.to_datetime(df["created"], errors="coerce")
        df = df.dropna(subset=["created"]).sort_values("created")

    lag_list = _parse_lags(lags)
    for lag in lag_list:
        if "adSpend" in df.columns:
            df[f"adSpend_lag_{lag}"] = df["adSpend"].shift(lag)
        if "orders" in df.columns:
            df[f"orders_lag_{lag}"] = df["orders"].shift(lag)

    if "totalRevenue" in df.columns:
        df["revenue_growth"] = df["totalRevenue"].pct_change()

    output_base = Path(output_dir)
    _ensure_dir(output_base)

    features_path = output_base / "features.jsonl"
    df.to_json(features_path, orient="records", lines=True, date_format="iso")

    numeric_cols = _select_numeric_features(df, target_col=target_col)
    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target: {target_col}")

    df_model = df.dropna(subset=numeric_cols + [target_col]).copy()
    feature_values = df_model[numeric_cols].to_numpy()
    target_values = df_model[target_col].to_numpy()

    x_window, y_window = _build_sliding_windows(
        feature_values, target_values, window_size=window_size
    )

    windows_path = output_base / "windows.npz"
    np.savez_compressed(
        windows_path,
        X=x_window,
        y=y_window,
        feature_columns=np.array(numeric_cols),
    )

    return {
        "features": features_path,
        "windows": windows_path,
    }
