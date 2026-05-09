from __future__ import annotations

import contextlib
import json
import os
import warnings
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

_STATSMODELS_IMPORT_ERROR: str | None = None
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as exc:  # pragma: no cover - optional dependency
    ExponentialSmoothing = None
    SARIMAX = None
    _STATSMODELS_IMPORT_ERROR = repr(exc)

_TORCH_IMPORT_ERROR: str | None = None
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = repr(exc)

_TFT_IMPORT_ERROR: str | None = None
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # type: ignore[import-not-found]
    from pytorch_forecasting.metrics import QuantileLoss  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - optional dependency
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    QuantileLoss = None
    _TFT_IMPORT_ERROR = repr(exc)

_LIGHTNING_IMPORT_ERROR: str | None = None
try:
    from lightning.pytorch import Trainer  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - optional dependency
    try:
        from pytorch_lightning import Trainer  # type: ignore[import-not-found]
    except Exception as fallback_exc:  # pragma: no cover - optional dependency
        Trainer = None
        _LIGHTNING_IMPORT_ERROR = f"{repr(exc)}; fallback: {repr(fallback_exc)}"


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


def _remove_existing(path: Path) -> None:
    if path.exists():
        path.unlink()


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(path)


def _atomic_joblib_dump(payload: Any, path: Path) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_xgboost_save(model: Any, path: Path) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    model.save_model(tmp_path)
    tmp_path.replace(path)


def _atomic_lightning_checkpoint(trainer: Any, path: Path) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    trainer.save_checkpoint(tmp_path)
    tmp_path.replace(path)


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


def _split_series(series: pd.Series, val_ratio: float) -> tuple[pd.Series, pd.Series]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1.")
    split_idx = int(len(series) * (1 - val_ratio))
    return series.iloc[:split_idx], series.iloc[split_idx:]


def _dependency_error(package: str, detail: str | None = None) -> dict[str, str]:
    suffix = f" Detalle: {detail}" if detail else ""
    return {"error": f"{package} no está instalado o no está disponible.{suffix}"}


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def _extend_prediction_error_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.Index | None = None,
) -> None:
    actuals = np.asarray(y_true, dtype=float).reshape(-1)
    predictions = np.asarray(y_pred, dtype=float).reshape(-1)
    total = min(len(actuals), len(predictions))

    for sample_idx in range(total):
        actual = float(actuals[sample_idx])
        prediction = float(predictions[sample_idx])
        mape = abs((actual - prediction) / max(abs(actual), 1e-8))
        if not np.isfinite(mape):
            continue

        row: dict[str, Any] = {
            "model": model_name,
            "sample_idx": sample_idx,
            "y_true": actual,
            "y_pred": prediction,
            "mape": float(mape),
        }
        if index is not None and sample_idx < len(index):
            row["created"] = str(index[sample_idx])
        rows.append(row)


def _hp_int(hyperparameters: dict[str, Any], key: str, default: int) -> int:
    value = hyperparameters.get(key, default)
    if value is None:
        return default
    return int(value)


def _hp_float(hyperparameters: dict[str, Any], key: str, default: float) -> float:
    value = hyperparameters.get(key, default)
    if value is None:
        return default
    return float(value)


def _hp_optional_int(hyperparameters: dict[str, Any], key: str, default: int | None) -> int | None:
    value = hyperparameters.get(key, default)
    if value in (None, ""):
        return None
    return int(value)


def _log_to_mlflow(
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    output_base: Path,
    target_col: str,
    val_ratio: float,
    hyperparameters: dict[str, Any] | None = None,
) -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    import mlflow

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sales-forecasting")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"compare-models-{target_col}") as run:
        mlflow.log_params(
            {
                "target_col": target_col,
                "val_ratio": val_ratio,
                "window_shape": json.dumps(metadata["window_shape"]),
            }
        )
        if hyperparameters:
            mlflow.log_params({f"hp_{key}": str(value) for key, value in hyperparameters.items()})
        for model_name, model_metrics in metrics.items():
            if not isinstance(model_metrics, dict):
                continue
            for metric_name, metric_value in model_metrics.items():
                if isinstance(metric_value, int | float | np.floating):
                    mlflow.log_metric(f"{model_name}_{metric_name}", float(metric_value))
        mlflow.log_artifacts(str(output_base))
        return run.info.run_id


def _prepare_target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    series_df = df[["created", target_col]].dropna().copy()
    series_df["created"] = pd.to_datetime(series_df["created"], errors="coerce")
    series_df[target_col] = pd.to_numeric(series_df[target_col], errors="coerce")
    series_df = series_df.dropna(subset=["created", target_col]).sort_values("created")
    if series_df.empty:
        raise ValueError("No hay datos válidos para entrenar modelos de serie temporal.")

    series = (
        series_df.groupby(series_df["created"].dt.floor("D"))[target_col]
        .sum()
        .sort_index()
        .asfreq("D")
    )
    return series


def _fill_series_for_model(series: pd.Series) -> pd.Series:
    filled = series.astype(float).interpolate(method="time", limit_direction="both")
    return filled.ffill().bfill()


class TrailingMeanWeekdayMedianBaseline:
    """
    Modelo base que mezcla inercia reciente (TM7) y estacionalidad semanal (SW8).
    """

    def __init__(
        self,
        trailing_days: int = 7,
        weekday_weeks: int = 8,
        trailing_weight: float = 0.45,
        weekday_weight: float = 0.55,
    ) -> None:
        self.trailing_days = trailing_days
        self.weekday_weeks = weekday_weeks
        self.trailing_weight = trailing_weight
        self.weekday_weight = weekday_weight
        self.series_: pd.Series | None = None

    def fit(self, series: pd.Series) -> "TrailingMeanWeekdayMedianBaseline":
        if series.empty:
            raise ValueError("No hay datos para ajustar el baseline TM7/SW8.")

        fitted = series.copy()
        fitted.index = pd.to_datetime(fitted.index).floor("D")
        fitted = pd.to_numeric(fitted, errors="coerce").sort_index()
        self.series_ = fitted
        return self

    def _require_fitted(self) -> pd.Series:
        if self.series_ is None:
            raise RuntimeError("El baseline TM7/SW8 debe ajustarse con fit antes de predecir.")
        return self.series_

    def components_for(self, target_date: Any) -> dict[str, float | None]:
        series = self._require_fitted()
        target = pd.Timestamp(target_date).floor("D")

        trailing_start = target - pd.Timedelta(days=self.trailing_days)
        trailing_values = series[(series.index >= trailing_start) & (series.index < target)].dropna()

        weekday_start = target - pd.Timedelta(weeks=self.weekday_weeks)
        weekday_values = series[
            (series.index >= weekday_start)
            & (series.index < target)
            & (series.index.weekday == target.weekday())
        ].dropna()

        trailing_mean = None if trailing_values.empty else float(trailing_values.mean())
        weekday_median = None if weekday_values.empty else float(weekday_values.median())
        return {
            "trailing_mean_7": trailing_mean,
            "same_weekday_median_8w": weekday_median,
        }

    def predict_one(self, target_date: Any) -> float:
        components = self.components_for(target_date)
        trailing_mean = components["trailing_mean_7"]
        weekday_median = components["same_weekday_median_8w"]

        if trailing_mean is not None and weekday_median is not None:
            return self.trailing_weight * trailing_mean + self.weekday_weight * weekday_median
        if trailing_mean is not None:
            return trailing_mean
        if weekday_median is not None:
            return weekday_median
        raise ValueError(f"No hay historia suficiente para predecir {pd.Timestamp(target_date).date()}.")

    def predict(self, target_dates: Any) -> np.ndarray:
        dates = pd.Index(target_dates)
        return np.asarray([self.predict_one(target_date) for target_date in dates], dtype=float)


def _standardize_train_val(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0:
        y_std = 1.0

    x_train_scaled = (x_train - x_mean) / x_std
    x_val_scaled = (x_val - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std
    scalers: dict[str, np.ndarray | float] = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return x_train_scaled, x_val_scaled, y_train_scaled, scalers


def _build_model_registry(random_state: int, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
    hp = hyperparameters or {}
    registry: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=_hp_float(hp, "ridge_alpha", 5.0), random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=_hp_int(hp, "random_forest_n_estimators", 100),
            max_depth=_hp_optional_int(hp, "random_forest_max_depth", 8),
            min_samples_split=_hp_int(hp, "random_forest_min_samples_split", 2),
            min_samples_leaf=_hp_int(hp, "random_forest_min_samples_leaf", 3),
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if xgb is not None:
        registry["xgboost"] = xgb.XGBRegressor(
            n_estimators=_hp_int(hp, "xgboost_n_estimators", 400),
            max_depth=_hp_int(hp, "xgboost_max_depth", 4),
            learning_rate=_hp_float(hp, "xgboost_learning_rate", 0.05),
            subsample=_hp_float(hp, "xgboost_subsample", 0.9),
            colsample_bytree=_hp_float(hp, "xgboost_colsample_bytree", 0.8),
            objective="reg:squarederror",
            random_state=random_state,
        )
    else:
        registry["xgboost"] = {"error": "XGBoost no está instalado."}
    if CatBoostRegressor is not None:
        registry["catboost"] = CatBoostRegressor(
            iterations=_hp_int(hp, "catboost_iterations", 500),
            learning_rate=_hp_float(hp, "catboost_learning_rate", 0.05),
            depth=_hp_int(hp, "catboost_depth", 6),
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
    hyperparameters: dict[str, Any] | None = None,
    error_rows: list[dict[str, Any]] | None = None,
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

    hp = hyperparameters or {}
    model_kwargs: dict[str, Any] = {}
    if "neuralprophet_epochs" in hp:
        model_kwargs["epochs"] = _hp_int(hp, "neuralprophet_epochs", 40)
    if "neuralprophet_learning_rate" in hp:
        model_kwargs["learning_rate"] = _hp_float(hp, "neuralprophet_learning_rate", 0.05)
    model = NeuralProphet(**model_kwargs)
    with _neuralprophet_torch_unsafe_weights():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but StandardScaler was fitted with feature names",
                category=UserWarning,
            )
            model.fit(train_df, freq="D")
            forecast = model.predict(val_df)
    metrics = _evaluate_predictions(val_df["y"].to_numpy(), forecast["yhat1"].to_numpy())
    if error_rows is not None:
        _extend_prediction_error_rows(
            rows=error_rows,
            model_name="neuralprophet",
            y_true=val_df["y"].to_numpy(),
            y_pred=forecast["yhat1"].to_numpy(),
            index=pd.Index(val_df["ds"]),
        )

    model_path = output_dir / "neuralprophet.joblib"
    _atomic_joblib_dump(model, model_path)
    return metrics


def _train_sarima(
    df: pd.DataFrame,
    target_col: str,
    val_ratio: float,
    output_dir: Path,
    hyperparameters: dict[str, Any] | None = None,
    error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if SARIMAX is None:
        raise RuntimeError("statsmodels no está disponible.")

    series = _prepare_target_series(df, target_col=target_col)
    train_series, val_series = _split_series(series, val_ratio=val_ratio)
    train_series = _fill_series_for_model(train_series)
    val_series = _fill_series_for_model(val_series)

    hp = hyperparameters or {}
    order = (
        _hp_int(hp, "sarima_p", 3),
        _hp_int(hp, "sarima_d", 1),
        _hp_int(hp, "sarima_q", 1),
    )
    seasonal_order = (
        _hp_int(hp, "sarima_seasonal_p", 0),
        _hp_int(hp, "sarima_seasonal_d", 1),
        _hp_int(hp, "sarima_seasonal_q", 1),
        _hp_int(hp, "sarima_seasonal_period", 12),
    )

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=hp.get("enforce_stationarity", False),
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    preds = fitted.forecast(steps=len(val_series))
    metrics = _evaluate_predictions(val_series.to_numpy(), preds.to_numpy())
    if error_rows is not None:
        _extend_prediction_error_rows(
            rows=error_rows,
            model_name="sarima",
            y_true=val_series.to_numpy(),
            y_pred=preds.to_numpy(),
            index=val_series.index,
        )

    model_path = output_dir / "sarima.joblib"
    _atomic_joblib_dump(fitted, model_path)
    return metrics


def _train_exponential_smoothing(
    df: pd.DataFrame,
    target_col: str,
    val_ratio: float,
    output_dir: Path,
    error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels no está disponible.")

    series = _prepare_target_series(df, target_col=target_col)
    train_series, val_series = _split_series(series, val_ratio=val_ratio)
    train_series = _fill_series_for_model(train_series)
    val_series = _fill_series_for_model(val_series)

    use_seasonal = len(train_series) >= 14
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=7 if use_seasonal else None,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    preds = fitted.forecast(len(val_series))
    metrics = _evaluate_predictions(val_series.to_numpy(), preds.to_numpy())
    if error_rows is not None:
        _extend_prediction_error_rows(
            rows=error_rows,
            model_name="exponential_smoothing",
            y_true=val_series.to_numpy(),
            y_pred=preds.to_numpy(),
            index=val_series.index,
        )

    model_path = output_dir / "exponential_smoothing.joblib"
    _atomic_joblib_dump(fitted, model_path)
    return metrics


def _train_lstm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    random_state: int,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch no está disponible.")

    torch.manual_seed(random_state)
    x_train_scaled, x_val_scaled, y_train_scaled, scalers = _standardize_train_val(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
    )

    class _LSTMRegressor(nn.Module):  # type: ignore[misc]
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _LSTMRegressor(input_size=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(x_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()
    preds = preds_scaled * float(scalers["y_std"]) + float(scalers["y_mean"])
    metrics = _evaluate_predictions(y_val, preds)
    if error_rows is not None:
        _extend_prediction_error_rows(
            rows=error_rows,
            model_name="lstm",
            y_true=y_val,
            y_pred=preds,
        )

    model_path = output_dir / "lstm.pt"
    _atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": x_train.shape[-1],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "scalers": scalers,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        model_path,
    )
    return metrics

def _train_temporal_fusion_transformer(
    df: pd.DataFrame,
    target_col: str,
    val_ratio: float,
    output_dir: Path,
    random_state: int,
    max_encoder_length: int = 28,
    max_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    hidden_size: int = 16,
    attention_head_size: int = 1,
    dropout: float = 0.1,
    hidden_continuous_size: int = 4,
    error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    missing = []
    if torch is None:
        missing.append(f"torch: {_TORCH_IMPORT_ERROR}")
    if TimeSeriesDataSet is None or TemporalFusionTransformer is None or QuantileLoss is None:
        missing.append(f"pytorch-forecasting: {_TFT_IMPORT_ERROR}")
    if Trainer is None:
        missing.append(f"lightning: {_LIGHTNING_IMPORT_ERROR}")
    if missing:
        raise RuntimeError("; ".join(missing))

    torch.manual_seed(random_state)
    raw_series = _prepare_target_series(df, target_col=target_col)
    train_series, val_series = _split_series(raw_series, val_ratio=val_ratio)
    series = pd.concat(
        [
            _fill_series_for_model(train_series),
            _fill_series_for_model(val_series),
        ]
    )
    data = series.reset_index()
    data.columns = ["created", target_col]
    data["time_idx"] = np.arange(len(data), dtype=int)
    data["series_id"] = "main"
    data["month"] = data["created"].dt.month.astype(str).astype("category")
    data["weekday"] = data["created"].dt.weekday.astype(str).astype("category")

    split_idx = int(len(data) * (1 - val_ratio))
    if split_idx <= max_encoder_length:
        raise ValueError("No hay suficientes filas para entrenar TFT con max_encoder_length=28.")
    training_cutoff = split_idx - 1

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=["month", "weekday"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        optimizer="adam",
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    actuals = torch.cat([y[0] for _, y in iter(val_loader)]).detach().cpu().numpy().reshape(-1)
    predictions = model.predict(val_loader).detach().cpu().numpy().reshape(-1)
    metrics = _evaluate_predictions(actuals, predictions)
    if error_rows is not None:
        val_index = data.loc[data["time_idx"] > training_cutoff, "created"].reset_index(drop=True)
        _extend_prediction_error_rows(
            rows=error_rows,
            model_name="temporal_fusion_transformer",
            y_true=actuals,
            y_pred=predictions,
            index=pd.Index(val_index.iloc[: len(actuals)]),
        )

    checkpoint_path = output_dir / "temporal_fusion_transformer.ckpt"
    _atomic_lightning_checkpoint(trainer, checkpoint_path)
    return metrics


def compare_models(
    input_path: str | Path = "reports/eda/features/windows.npz",
    output_dir: str | Path = "reports/eda/models",
    series_path: str | Path = "reports/eda/features/features.jsonl",
    target_col: str = "orders",
    val_ratio: float = 0.2,
    random_state: int = 42,
    log_mlflow: bool = True,
    hyperparameters: dict[str, Any] | None = None,
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

    finite_mask = np.isfinite(x_window).all(axis=(1, 2)) & np.isfinite(y_window)
    x_window = x_window[finite_mask]
    y_window = y_window[finite_mask]
    x_flat = _flatten_windows(x_window)

    x_train, x_val, y_train, y_val = _train_val_split(x_flat, y_window, val_ratio=val_ratio)
    x_train_seq, x_val_seq, y_train_seq, y_val_seq = _train_val_split(
        x_window,
        y_window,
        val_ratio=val_ratio,
    )

    hp = hyperparameters or {}
    registry = _build_model_registry(random_state=random_state, hyperparameters=hp)
    metrics: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []

    for name, model in registry.items():
        if isinstance(model, dict):
            metrics[name] = model
            continue
        try:
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            metrics[name] = _evaluate_predictions(y_val, preds)
            _extend_prediction_error_rows(
                rows=error_rows,
                model_name=name,
                y_true=y_val,
                y_pred=preds,
            )

            model_path = output_base / f"{name}.joblib"
            if name == "xgboost" and xgb is not None:
                xgboost_path = output_base / "xgboost.json"
                _atomic_xgboost_save(model, xgboost_path)
            else:
                _atomic_joblib_dump(model, model_path)
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    if torch is None:
        metrics["lstm"] = _dependency_error("PyTorch", _TORCH_IMPORT_ERROR)
    else:
        try:
            metrics["lstm"] = _train_lstm(
                x_train=x_train_seq,
                x_val=x_val_seq,
                y_train=y_train_seq,
                y_val=y_val_seq,
                output_dir=output_base,
                random_state=random_state,
                epochs=_hp_int(hp, "lstm_epochs", 50),
                batch_size=_hp_int(hp, "lstm_batch_size", 32),
                hidden_size=_hp_int(hp, "lstm_hidden_size", 64),
                num_layers=_hp_int(hp, "lstm_num_layers", 2),
                dropout=_hp_float(hp, "lstm_dropout", 0.2),
                learning_rate=_hp_float(hp, "lstm_learning_rate", 1e-3),
                error_rows=error_rows,
            )
        except Exception as exc:
            metrics["lstm"] = {"error": str(exc)}

    series_file = Path(series_path)
    series_model_names = (
        "baseline_tm7_sw8_blend",
        "sarima",
        "exponential_smoothing",
        "neuralprophet",
        "temporal_fusion_transformer",
    )
    if not series_file.exists():
        for name in series_model_names:
            metrics[name] = {"error": f"No existe series_path: {series_file}"}
    else:
        df_series = pd.read_json(series_file, lines=True, dtype=False)
        if "created" in df_series.columns and target_col in df_series.columns:
            try:
                baseline_series = _prepare_target_series(df_series, target_col=target_col)
                _, baseline_val_series = _split_series(baseline_series, val_ratio=val_ratio)
                baseline_val_series = baseline_val_series.dropna()
                baseline = TrailingMeanWeekdayMedianBaseline().fit(baseline_series)
                baseline_preds = baseline.predict(baseline_val_series.index)
                metrics["baseline_tm7_sw8_blend"] = _evaluate_predictions(
                    baseline_val_series.to_numpy(),
                    baseline_preds,
                )
                _extend_prediction_error_rows(
                    rows=error_rows,
                    model_name="baseline_tm7_sw8_blend",
                    y_true=baseline_val_series.to_numpy(),
                    y_pred=baseline_preds,
                    index=baseline_val_series.index,
                )
                model_path = output_base / "baseline_tm7_sw8_blend.joblib"
                _atomic_joblib_dump(baseline, model_path)
            except Exception as exc:
                metrics["baseline_tm7_sw8_blend"] = {"error": str(exc)}

            if SARIMAX is None or ExponentialSmoothing is None:
                statsmodels_error = _dependency_error("statsmodels", _STATSMODELS_IMPORT_ERROR)
                metrics["sarima"] = statsmodels_error
                metrics["exponential_smoothing"] = statsmodels_error
            else:
                try:
                    metrics["sarima"] = _train_sarima(
                        df=df_series,
                        target_col=target_col,
                        val_ratio=val_ratio,
                        output_dir=output_base,
                        hyperparameters=hp,
                        error_rows=error_rows,
                    )
                except Exception as exc:
                    metrics["sarima"] = {"error": str(exc)}

                try:
                    metrics["exponential_smoothing"] = _train_exponential_smoothing(
                        df=df_series,
                        target_col=target_col,
                        val_ratio=val_ratio,
                        output_dir=output_base,
                        error_rows=error_rows,
                    )
                except Exception as exc:
                    metrics["exponential_smoothing"] = {"error": str(exc)}

            if NeuralProphet is None:
                metrics["neuralprophet"] = _dependency_error("NeuralProphet", _NEURALPROPHET_IMPORT_ERROR)
            else:
                try:
                    metrics["neuralprophet"] = _train_neuralprophet(
                        df=df_series,
                        target_col=target_col,
                        val_ratio=val_ratio,
                        output_dir=output_base,
                        error_rows=error_rows,
                    )
                except Exception as exc:
                    metrics["neuralprophet"] = {"error": str(exc)}

            try:
                metrics["temporal_fusion_transformer"] = _train_temporal_fusion_transformer(
                    df=df_series,
                    target_col=target_col,
                    val_ratio=val_ratio,
                    output_dir=output_base,
                    random_state=random_state,
                    max_encoder_length=_hp_int(hp, "tft_max_encoder_length", 28),
                    max_epochs=_hp_int(hp, "tft_max_epochs", 10),
                    batch_size=_hp_int(hp, "tft_batch_size", 32),
                    learning_rate=_hp_float(hp, "tft_learning_rate", 0.01),
                    hidden_size=_hp_int(hp, "tft_hidden_size", 16),
                    attention_head_size=_hp_int(hp, "tft_attention_head_size", 1),
                    dropout=_hp_float(hp, "tft_dropout", 0.1),
                    hidden_continuous_size=_hp_int(hp, "tft_hidden_continuous_size", 4),
                    error_rows=error_rows,
                )
            except Exception as exc:
                metrics["temporal_fusion_transformer"] = {"error": str(exc)}
        else:
            for name in series_model_names:
                metrics[name] = {"error": "Faltan columnas created/target_col en series."}

    metrics_path = output_base / "metrics.json"
    mape_distribution_path = output_base / "mape_distribution.jsonl"
    meta_path = output_base / "metadata.json"
    _atomic_write_json(metrics_path, metrics)
    _atomic_write_jsonl(mape_distribution_path, error_rows)

    metadata = {
        "feature_columns": feature_columns,
        "window_shape": list(x_window.shape),
        "val_ratio": val_ratio,
        "hyperparameters": hp,
        "mape_distribution_rows": len(error_rows),
    }
    _atomic_write_json(meta_path, metadata)

    result: dict[str, Any] = {
        "metrics": metrics_path,
        "mape_distribution": mape_distribution_path,
        "metadata": meta_path,
        "models_dir": output_base,
    }

    if log_mlflow:
        try:
            mlflow_run_id = _log_to_mlflow(
                metrics=metrics,
                metadata=metadata,
                output_base=output_base,
                target_col=target_col,
                val_ratio=val_ratio,
                hyperparameters=hp,
            )
            if mlflow_run_id is not None:
                result["mlflow_run_id"] = mlflow_run_id
        except Exception as exc:
            result["mlflow_error"] = str(exc)

    return result
