from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda.train import (
    CatBoostRegressor,
    NeuralProphet,
    SARIMAX,
    TimeSeriesDataSet,
    Trainer,
    TrailingMeanWeekdayMedianBaseline,
    _atomic_joblib_dump,
    _atomic_write_json,
    _atomic_xgboost_save,
    _build_model_registry,
    _dependency_error,
    _evaluate_predictions,
    _fill_series_for_model,
    _flatten_windows,
    _load_windows,
    _prepare_target_series,
    _split_series,
    _train_exponential_smoothing,
    _train_lstm,
    _train_neuralprophet,
    _train_sarima,
    _train_temporal_fusion_transformer,
    _train_val_split,
    torch,
    xgb,
)

ParamGrid = dict[str, Sequence[Any] | Any]
Metrics = dict[str, float] | dict[str, str]
GridRunner = Callable[[dict[str, Any], Path], Metrics]

WINDOW_MODELS = {"linear_regression", "ridge", "random_forest", "xgboost", "catboost"}
SERIES_MODELS = {
    "baseline_tm7_sw8_blend",
    "sarima",
    "exponential_smoothing",
    "neuralprophet",
    "temporal_fusion_transformer",
}
SEQUENCE_MODELS = {"lstm"}
SUPPORTED_MODELS = WINDOW_MODELS | SERIES_MODELS | SEQUENCE_MODELS


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _grid_values(value: Sequence[Any] | Any) -> list[Any]:
    if isinstance(value, range):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        values = list(value)
        if not values:
            raise ValueError("Los rangos de hiperparametros no pueden estar vacios.")
        return values
    return [value]


def _expand_grid(param_grid: ParamGrid | None) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]

    keys = list(param_grid)
    values = [_grid_values(param_grid[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _metric_value(metrics: Metrics, metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return None


def _run_grid_search(
    *,
    model_name: str,
    param_grid: ParamGrid | None,
    output_dir: str | Path,
    selection_metric: str,
    runner: GridRunner,
) -> dict[str, Any]:
    combinations = _expand_grid(param_grid)
    output_base = Path(output_dir) / model_name
    _ensure_dir(output_base)

    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_value: float | None = None

    for run_idx, hyperparameters in enumerate(combinations):
        run_dir = output_base / f"run_{run_idx:03d}"
        _ensure_dir(run_dir)

        try:
            metrics = runner(hyperparameters, run_dir)
        except Exception as exc:
            metrics = {"error": str(exc)}

        metric_value = _metric_value(metrics, selection_metric)
        result = {
            "run_id": run_idx,
            "model": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "models_dir": str(run_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)
        _atomic_write_json(run_dir / "metadata.json", result)

        if metric_value is not None and (best_value is None or metric_value < best_value):
            best_value = metric_value
            best_result = result

    summary = {
        "model": model_name,
        "selection_metric": selection_metric,
        "total_runs": len(results),
        "best_run": best_result,
        "results": results,
    }
    summary_path = output_base / "grid_search_results.json"
    _atomic_write_json(summary_path, summary)

    return {
        "model": model_name,
        "results": summary_path,
        "models_dir": output_base,
        "best_run": best_result,
    }


def _load_flat_window_data(
    input_path: str | Path,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    npz_path = Path(input_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe el dataset de ventanas: {npz_path}")

    x_window, y_window, feature_columns = _load_windows(npz_path)
    finite_mask = np.isfinite(x_window).all(axis=(1, 2)) & np.isfinite(y_window)
    x_window = x_window[finite_mask]
    y_window = y_window[finite_mask]
    x_flat = _flatten_windows(x_window)
    x_train, x_val, y_train, y_val = _train_val_split(x_flat, y_window, val_ratio=val_ratio)
    return x_train, x_val, y_train, y_val, feature_columns, list(x_window.shape)


def _load_sequence_window_data(
    input_path: str | Path,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    npz_path = Path(input_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe el dataset de ventanas: {npz_path}")

    x_window, y_window, feature_columns = _load_windows(npz_path)
    finite_mask = np.isfinite(x_window).all(axis=(1, 2)) & np.isfinite(y_window)
    x_window = x_window[finite_mask]
    y_window = y_window[finite_mask]
    x_train, x_val, y_train, y_val = _train_val_split(x_window, y_window, val_ratio=val_ratio)
    return x_train, x_val, y_train, y_val, feature_columns, list(x_window.shape)


def _load_series_data(series_path: str | Path, target_col: str) -> pd.DataFrame:
    series_file = Path(series_path)
    if not series_file.exists():
        raise FileNotFoundError(f"No existe series_path: {series_file}")

    df_series = pd.read_json(series_file, lines=True, dtype=False)
    if "created" not in df_series.columns or target_col not in df_series.columns:
        raise ValueError("Faltan columnas created/target_col en series.")
    return df_series


def _grid_search_window_model(
    *,
    model_name: str,
    param_grid: ParamGrid | None,
    input_path: str | Path,
    output_dir: str | Path,
    val_ratio: float,
    random_state: int,
    selection_metric: str,
) -> dict[str, Any]:
    x_train, x_val, y_train, y_val, feature_columns, window_shape = _load_flat_window_data(
        input_path=input_path,
        val_ratio=val_ratio,
    )

    def runner(hyperparameters: dict[str, Any], run_dir: Path) -> Metrics:
        registry = _build_model_registry(random_state=random_state, hyperparameters=hyperparameters)
        model = registry[model_name]
        if isinstance(model, dict):
            return model

        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        metrics = _evaluate_predictions(y_val, preds)

        if model_name == "xgboost" and xgb is not None:
            _atomic_xgboost_save(model, run_dir / "xgboost.json")
        else:
            _atomic_joblib_dump(model, run_dir / f"{model_name}.joblib")

        _atomic_write_json(
            run_dir / "data_metadata.json",
            {"feature_columns": feature_columns, "window_shape": window_shape, "val_ratio": val_ratio},
        )
        return metrics

    return _run_grid_search(
        model_name=model_name,
        param_grid=param_grid,
        output_dir=output_dir,
        selection_metric=selection_metric,
        runner=runner,
    )


def _grid_search_lstm_model(
    *,
    param_grid: ParamGrid | None,
    input_path: str | Path,
    output_dir: str | Path,
    val_ratio: float,
    random_state: int,
    selection_metric: str,
) -> dict[str, Any]:
    x_train, x_val, y_train, y_val, feature_columns, window_shape = _load_sequence_window_data(
        input_path=input_path,
        val_ratio=val_ratio,
    )

    def runner(hyperparameters: dict[str, Any], run_dir: Path) -> Metrics:
        if torch is None:
            return _dependency_error("PyTorch")

        metrics = _train_lstm(
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            output_dir=run_dir,
            random_state=random_state,
            epochs=int(hyperparameters.get("lstm_epochs", 50)),
            batch_size=int(hyperparameters.get("lstm_batch_size", 128)),
            hidden_size=int(hyperparameters.get("lstm_hidden_size", 256)),
            num_layers=int(hyperparameters.get("lstm_num_layers", 3)),
            dropout=float(hyperparameters.get("lstm_dropout", 0.05)),
            learning_rate=float(hyperparameters.get("lstm_learning_rate", 1e-3)),
        )
        _atomic_write_json(
            run_dir / "data_metadata.json",
            {"feature_columns": feature_columns, "window_shape": window_shape, "val_ratio": val_ratio},
        )
        return metrics

    return _run_grid_search(
        model_name="lstm",
        param_grid=param_grid,
        output_dir=output_dir,
        selection_metric=selection_metric,
        runner=runner,
    )


def _grid_search_series_model(
    *,
    model_name: str,
    param_grid: ParamGrid | None,
    series_path: str | Path,
    output_dir: str | Path,
    target_col: str,
    val_ratio: float,
    random_state: int,
    selection_metric: str,
) -> dict[str, Any]:
    df_series = _load_series_data(series_path=series_path, target_col=target_col)

    def runner(hyperparameters: dict[str, Any], run_dir: Path) -> Metrics:
        if model_name == "baseline_tm7_sw8_blend":
            return _run_baseline_grid_candidate(
                df=df_series,
                target_col=target_col,
                val_ratio=val_ratio,
                output_dir=run_dir,
                hyperparameters=hyperparameters,
            )
        if model_name == "sarima":
            if SARIMAX is None:
                return _dependency_error("statsmodels")
            return _train_sarima(
                df=df_series,
                target_col=target_col,
                val_ratio=val_ratio,
                output_dir=run_dir,
                hyperparameters=hyperparameters,
            )
        if model_name == "exponential_smoothing":
            return _train_exponential_smoothing(
                df=df_series,
                target_col=target_col,
                val_ratio=val_ratio,
                output_dir=run_dir,
            )
        if model_name == "neuralprophet":
            if NeuralProphet is None:
                return _dependency_error("NeuralProphet")
            return _train_neuralprophet(
                df=df_series,
                target_col=target_col,
                val_ratio=val_ratio,
                output_dir=run_dir,
                hyperparameters=hyperparameters,
            )
        if model_name == "temporal_fusion_transformer":
            if torch is None or TimeSeriesDataSet is None or Trainer is None:
                return _dependency_error("TemporalFusionTransformer")
            return _train_temporal_fusion_transformer(
                df=df_series,
                target_col=target_col,
                val_ratio=val_ratio,
                output_dir=run_dir,
                random_state=random_state,
                max_encoder_length=int(hyperparameters.get("tft_max_encoder_length", 28)),
                max_epochs=int(hyperparameters.get("tft_max_epochs", 10)),
                batch_size=int(hyperparameters.get("tft_batch_size", 32)),
                learning_rate=float(hyperparameters.get("tft_learning_rate", 0.01)),
                hidden_size=int(hyperparameters.get("tft_hidden_size", 16)),
                attention_head_size=int(hyperparameters.get("tft_attention_head_size", 1)),
                dropout=float(hyperparameters.get("tft_dropout", 0.1)),
                hidden_continuous_size=int(hyperparameters.get("tft_hidden_continuous_size", 4)),
            )
        raise ValueError(f"Modelo no soportado para series: {model_name}")

    return _run_grid_search(
        model_name=model_name,
        param_grid=param_grid,
        output_dir=output_dir,
        selection_metric=selection_metric,
        runner=runner,
    )


def _run_baseline_grid_candidate(
    *,
    df: pd.DataFrame,
    target_col: str,
    val_ratio: float,
    output_dir: Path,
    hyperparameters: dict[str, Any],
) -> Metrics:
    baseline_series = _prepare_target_series(df, target_col=target_col)
    _, baseline_val_series = _split_series(baseline_series, val_ratio=val_ratio)
    baseline_val_series = _fill_series_for_model(baseline_val_series).dropna()

    model = TrailingMeanWeekdayMedianBaseline(
        trailing_days=int(hyperparameters.get("baseline_trailing_days", 7)),
        weekday_weeks=int(hyperparameters.get("baseline_weekday_weeks", 8)),
        trailing_weight=float(hyperparameters.get("baseline_trailing_weight", 0.45)),
        weekday_weight=float(hyperparameters.get("baseline_weekday_weight", 0.55)),
    ).fit(baseline_series)
    preds = model.predict(baseline_val_series.index)
    metrics = _evaluate_predictions(baseline_val_series.to_numpy(), preds)
    _atomic_joblib_dump(model, output_dir / "baseline_tm7_sw8_blend.joblib")
    return metrics


def grid_search_model(
    model_name: str,
    param_grid: ParamGrid | None = None,
    input_path: str | Path = "reports/eda/features/windows.npz",
    series_path: str | Path = "reports/eda/features/features.jsonl",
    output_dir: str | Path = "reports/eda/grid_search",
    target_col: str = "orders",
    val_ratio: float = 0.2,
    random_state: int = 42,
    selection_metric: str = "mae",
) -> dict[str, Any]:
    """
    Ejecuta grid search para un modelo especifico.

    param_grid acepta escalares o listas/rangos. Ejemplo:
    {"random_forest_n_estimators": [100, 300], "random_forest_max_depth": [None, 8]}
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo no soportado: {model_name}. Opciones: {sorted(SUPPORTED_MODELS)}")

    if model_name in WINDOW_MODELS:
        return _grid_search_window_model(
            model_name=model_name,
            param_grid=param_grid,
            input_path=input_path,
            output_dir=output_dir,
            val_ratio=val_ratio,
            random_state=random_state,
            selection_metric=selection_metric,
        )
    if model_name == "lstm":
        return _grid_search_lstm_model(
            param_grid=param_grid,
            input_path=input_path,
            output_dir=output_dir,
            val_ratio=val_ratio,
            random_state=random_state,
            selection_metric=selection_metric,
        )
    return _grid_search_series_model(
        model_name=model_name,
        param_grid=param_grid,
        series_path=series_path,
        output_dir=output_dir,
        target_col=target_col,
        val_ratio=val_ratio,
        random_state=random_state,
        selection_metric=selection_metric,
    )


def grid_search_linear_regression(**kwargs: Any) -> dict[str, Any]:
    return grid_search_model("linear_regression", **kwargs)


def grid_search_ridge(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("ridge", param_grid=param_grid, **kwargs)


def grid_search_random_forest(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("random_forest", param_grid=param_grid, **kwargs)


def grid_search_xgboost(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    if xgb is None:
        raise RuntimeError("XGBoost no esta instalado.")
    return grid_search_model("xgboost", param_grid=param_grid, **kwargs)


def grid_search_catboost(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost no esta instalado.")
    return grid_search_model("catboost", param_grid=param_grid, **kwargs)


def grid_search_lstm(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("lstm", param_grid=param_grid, **kwargs)


def grid_search_baseline_tm7_sw8_blend(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("baseline_tm7_sw8_blend", param_grid=param_grid, **kwargs)


def grid_search_sarima(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("sarima", param_grid=param_grid, **kwargs)


def grid_search_exponential_smoothing(**kwargs: Any) -> dict[str, Any]:
    return grid_search_model("exponential_smoothing", **kwargs)


def grid_search_neuralprophet(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("neuralprophet", param_grid=param_grid, **kwargs)


def grid_search_temporal_fusion_transformer(param_grid: ParamGrid | None = None, **kwargs: Any) -> dict[str, Any]:
    return grid_search_model("temporal_fusion_transformer", param_grid=param_grid, **kwargs)
