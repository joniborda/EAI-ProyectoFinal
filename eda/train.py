from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

_NEURALPROPHET_IMPORT_ERROR: str | None = None
try:
    from neuralprophet import NeuralProphet
except Exception as exc:  # pragma: no cover - optional dependency
    NeuralProphet = None
    _NEURALPROPHET_IMPORT_ERROR = repr(exc)

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


@contextlib.contextmanager
def _neuralprophet_torch_unsafe_weights() -> Iterator[None]:
    """PyTorch>=2.6 usa weights_only=True; NeuralProphet carga objetos NP vía torch.load."""
    import torch

    orig = torch.load

    def _load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = orig  # type: ignore[method-assign]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_windows(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    x_window = data["X"]
    y_window = data["y"]
    feature_columns = data.get("feature_columns")
    if feature_columns is None:
        features = []
    else:
        features = feature_columns.tolist()
    return x_window, y_window, features


def _flatten_windows(x_window: np.ndarray) -> np.ndarray:
    return x_window.reshape((x_window.shape[0], -1))


def _train_val_split(x: np.ndarray, y: np.ndarray, val_ratio: float) -> tuple[np.ndarray, ...]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1.")
    split_idx = int(len(x) * (1 - val_ratio))
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def _build_model_registry(random_state: int) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if xgb is not None:
        registry["xgboost"] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    else:
        registry["xgboost"] = {"error": "XGBoost no está instalado."}
    if CatBoostRegressor is not None:
        registry["catboost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    else:
        registry["catboost"] = {"error": "CatBoost no está instalado."}
    return registry


def _train_neuralprophet(
    df: pd.DataFrame,
    target_col: str,
    val_ratio: float,
    output_dir: Path,
) -> dict[str, float]:
    if NeuralProphet is None:
        raise RuntimeError("NeuralProphet no está disponible.")

    series_df = df[["created", target_col]].dropna().copy()
    series_df = series_df.rename(columns={"created": "ds", target_col: "y"})
    series_df["ds"] = pd.to_datetime(series_df["ds"], errors="coerce")
    series_df = series_df.dropna(subset=["ds"]).sort_values("ds")

    split_idx = int(len(series_df) * (1 - val_ratio))
    train_df = series_df.iloc[:split_idx]
    val_df = series_df.iloc[split_idx:]

    model = NeuralProphet()
    with _neuralprophet_torch_unsafe_weights():
        model.fit(train_df, freq="D")
        forecast = model.predict(val_df)
    metrics = _evaluate_predictions(val_df["y"].to_numpy(), forecast["yhat1"].to_numpy())

    joblib.dump(model, output_dir / "neuralprophet.joblib")
    return metrics


def compare_models(
    input_path: str | Path = "reports/eda/features/windows.npz",
    output_dir: str | Path = "reports/eda/models",
    series_path: str | Path = "reports/eda/features/features.jsonl",
    target_col: str = "orders",
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Entrena varios modelos, evalúa y guarda métricas + modelos.
    """
    npz_path = Path(input_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe el dataset de ventanas: {npz_path}")

    output_base = Path(output_dir)
    _ensure_dir(output_base)

    x_window, y_window, feature_columns = _load_windows(npz_path)
    x_flat = _flatten_windows(x_window)

    finite_mask = np.isfinite(x_flat).all(axis=1) & np.isfinite(y_window)
    x_flat = x_flat[finite_mask]
    y_window = y_window[finite_mask]

    x_train, x_val, y_train, y_val = _train_val_split(x_flat, y_window, val_ratio=val_ratio)

    registry = _build_model_registry(random_state=random_state)
    metrics: dict[str, Any] = {}

    for name, model in registry.items():
        if isinstance(model, dict):
            metrics[name] = model
            continue
        try:
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            metrics[name] = _evaluate_predictions(y_val, preds)

            model_path = output_base / f"{name}.joblib"
            if name == "xgboost" and xgb is not None:
                model.save_model(output_base / "xgboost.json")
            else:
                joblib.dump(model, model_path)
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    series_file = Path(series_path)
    if NeuralProphet is None:
        detail = f" Detalle: {_NEURALPROPHET_IMPORT_ERROR}" if _NEURALPROPHET_IMPORT_ERROR else ""
        metrics["neuralprophet"] = {"error": f"NeuralProphet no está instalado.{detail}"}
    elif not series_file.exists():
        metrics["neuralprophet"] = {"error": f"No existe series_path: {series_file}"}
    else:
        df_series = pd.read_json(series_file, lines=True, dtype=False)
        if "created" in df_series.columns and target_col in df_series.columns:
            try:
                metrics["neuralprophet"] = _train_neuralprophet(
                    df=df_series,
                    target_col=target_col,
                    val_ratio=val_ratio,
                    output_dir=output_base,
                )
            except Exception as exc:
                metrics["neuralprophet"] = {"error": str(exc)}
        else:
            metrics["neuralprophet"] = {"error": "Faltan columnas created/target_col en series."}

    metrics_path = output_base / "metrics.json"
    meta_path = output_base / "metadata.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "feature_columns": feature_columns,
        "window_shape": list(x_window.shape),
        "val_ratio": val_ratio,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "metrics": metrics_path,
        "metadata": meta_path,
        "models_dir": output_base,
    }
