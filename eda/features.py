from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_existing(path: Path) -> None:
    if path.exists():
        path.unlink()


def _parse_lags(lags: Iterable[int]) -> list[int]:
    parsed = sorted({int(lag) for lag in lags if int(lag) > 0})
    return parsed


def _select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric_cols:
        return numeric_cols
    return numeric_cols


def impute_missing_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "median",
    add_missing_indicators: bool = False,
    exclude_target: bool = True,
) -> pd.DataFrame:
    """
    Imputa features numéricas antes de construir lags/ventanas.

    Por defecto el target queda excluido para no crear valores de entrenamiento
    artificiales. Para fechas insertadas se puede incluir explícitamente.
    """
    if strategy not in {"mean", "median"}:
        raise ValueError("imputation_strategy debe ser 'mean' o 'median'.")

    result = df.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if not exclude_target or col != target_col]

    for col in feature_cols:
        missing_mask = result[col].isna()
        if not missing_mask.any():
            continue

        if add_missing_indicators:
            result[f"{col}_was_missing"] = missing_mask.astype(int)

        fill_value = result[col].mean() if strategy == "mean" else result[col].median()
        
        if pd.isna(fill_value):
            fill_value = 0
        result[col] = result[col].fillna(fill_value)

    return result


def complete_daily_date_range(
    df: pd.DataFrame,
    date_col: str = "created",
    inserted_indicator_col: str = "date_was_missing",
) -> pd.DataFrame:
    """
    Inserta los días faltantes entre la primera y última fecha observada.
    """
    if date_col not in df.columns:
        return df.copy()

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.floor("D")
    result = result.dropna(subset=[date_col]).sort_values(date_col)
    if result.empty:
        return result

    result = result.drop_duplicates(subset=[date_col], keep="last")
    result["_original_row_present"] = 1

    full_dates = pd.date_range(result[date_col].min(), result[date_col].max(), freq="D")
    result = result.set_index(date_col).reindex(full_dates)
    result.index.name = date_col

    # result[inserted_indicator_col] = result["_original_row_present"].isna().astype(int)
    result = result.drop(columns=["_original_row_present"]).reset_index()
    return result


def add_cyclical_date_features(df: pd.DataFrame, date_col: str = "created") -> pd.DataFrame:
    """
    Agrega representaciones circulares para día de semana y mes.
    """
    if date_col not in df.columns:
        return df.copy()

    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    weekday = dates.dt.dayofweek
    month = dates.dt.month

    result["created_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["created_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["created_month_sin"] = np.sin(2 * np.pi * month / 12)
    result["created_month_cos"] = np.cos(2 * np.pi * month / 12)
    return result


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
    imputation_strategy: str = "median",
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
        df = complete_daily_date_range(df, date_col="created")
        df = add_cyclical_date_features(df, date_col="created")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = impute_missing_numeric_features(
        df,
        target_col=target_col,
        strategy=imputation_strategy,
        exclude_target=False,
    )

    lag_list = _parse_lags(lags)
    for lag in lag_list:
        if "adSpend" in df.columns:
            df[f"adSpend_lag_{lag}"] = df["adSpend"].shift(lag)
        if "orders" in df.columns:
            df[f"orders_lag_{lag}"] = df["orders"].shift(lag)

    if "totalRevenue" in df.columns:
        df["revenue_growth"] = df["totalRevenue"].pct_change()

    if "event_start" in df.columns:
        df["event_start"] = pd.to_numeric(df["event_start"], errors="coerce").fillna(0).astype(int)
        # La ventana usa días anteriores para predecir el siguiente día; este lead
        # permite que el modelo vea que mañana empieza un evento conocido.
        df["event_start_next_1"] = df["event_start"].shift(-1).fillna(0).astype(int)

    output_base = Path(output_dir)
    _ensure_dir(output_base)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = impute_missing_numeric_features(
        df,
        target_col=target_col,
        strategy=imputation_strategy,
    )

    features_path = output_base / "features.jsonl"
    _remove_existing(features_path)
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
    _remove_existing(windows_path)
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
