from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from eda.features import add_cyclical_date_features
from eda.train import (
    NeuralProphet,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    TrailingMeanWeekdayMedianBaseline,
    _hp_int,
    _prepare_target_series,
    _fill_series_for_model,
    torch,
    xgb,
    nn,
)


def load_training_artifacts(model_output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    best_path = model_output_dir / "best_model.json"
    meta_path = model_output_dir / "metadata.json"
    if not best_path.exists():
        raise FileNotFoundError(f"No existe {best_path}")
    with best_path.open(encoding="utf-8") as f:
        best = json.load(f)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    return best, meta


def _read_features_df(series_path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_json(series_path, lines=True, dtype=False)
    if "created" not in df.columns or target_col not in df.columns:
        raise ValueError(f"Faltan columnas 'created' o '{target_col}' en {series_path}")
    df = df.copy()
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df = df.dropna(subset=["created"]).sort_values("created")
    return df


def _predict_baseline(
    model: TrailingMeanWeekdayMedianBaseline,
    series: pd.Series,
    first_day: date,
    days: int,
) -> list[dict[str, Any]]:
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_day = first_day + timedelta(days=offset)
        components = model.components_for(target_day)
        prediction = model.predict_one(target_day)
        daily.append(
            {
                "date": target_day.isoformat(),
                "prediction": round(float(prediction), 2),
                "components": {
                    "trailing_mean_7": None
                    if components["trailing_mean_7"] is None
                    else round(float(components["trailing_mean_7"]), 2),
                    "same_weekday_median_8w": None
                    if components["same_weekday_median_8w"] is None
                    else round(float(components["same_weekday_median_8w"]), 2),
                },
            }
        )
    return daily


def _ensure_feature_frame(extended: pd.DataFrame, feature_columns: list[str], target_col: str) -> None:
    for col in feature_columns:
        if col not in extended.columns:
            extended[col] = 0.0 if col != target_col else np.nan


def _append_future_row_recursive(
    extended: pd.DataFrame,
    target_day: date,
    target_col: str,
    feature_columns: list[str],
) -> None:
    last = extended.iloc[-1]
    new_row = last.copy()
    new_row["created"] = pd.Timestamp(target_day)
    tmp = pd.DataFrame({"created": [new_row["created"]]})
    tmp = add_cyclical_date_features(tmp)
    for c in tmp.columns:
        if c != "created" and c in new_row.index:
            new_row[c] = tmp.iloc[0][c]
    if "created_weekday" in new_row.index:
        new_row["created_weekday"] = float(new_row["created"].dayofweek)
    if "created_month" in new_row.index:
        new_row["created_month"] = float(new_row["created"].month)
    for col in feature_columns:
        if col == target_col:
            continue
        if col.startswith("orders_lag_") or col.startswith("adSpend_lag_") or col == "revenue_growth":
            continue
        if col not in new_row.index:
            new_row[col] = last.get(col, 0.0)
    extended.loc[len(extended)] = new_row
    idx = extended.index[-1]
    for lag_col in feature_columns:
        if lag_col.startswith("orders_lag_"):
            k = int(lag_col.split("_")[-1])
            v = extended[target_col].shift(k).iloc[-1]
            extended.loc[idx, lag_col] = 0.0 if pd.isna(v) else float(v)
        elif lag_col.startswith("adSpend_lag_"):
            k = int(lag_col.split("_")[-1])
            if "adSpend" in extended.columns:
                v = extended["adSpend"].shift(k).iloc[-1]
                extended.loc[idx, lag_col] = 0.0 if pd.isna(v) else float(v)
    if "revenue_growth" in feature_columns and "totalRevenue" in extended.columns:
        prev = extended["totalRevenue"].shift(1).iloc[-1]
        cur = extended["totalRevenue"].iloc[-1]
        if pd.notna(prev) and prev != 0 and pd.notna(cur):
            extended.loc[idx, "revenue_growth"] = float((cur - prev) / prev)
    if "event_start_next_1" in feature_columns:
        extended.loc[idx, "event_start_next_1"] = 0.0


def _predict_window_sklearn(
    model: Any,
    extended: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_col: str,
    first_day: date,
    days: int,
) -> list[dict[str, Any]]:
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_d = first_day + timedelta(days=offset)
        _append_future_row_recursive(extended, target_d, target_col, feature_columns)
        wdf = extended[feature_columns].tail(window_size)
        if len(wdf) < window_size:
            raise ValueError("Historia insuficiente para la ventana del modelo.")
        x_flat = wdf.to_numpy(dtype=np.float64).reshape(1, -1)
        if np.isnan(x_flat).any():
            x_flat = np.nan_to_num(x_flat, nan=0.0)
        pred = model.predict(x_flat)[0]
        daily.append({"date": target_d.isoformat(), "prediction": round(float(pred), 2)})
        extended.loc[extended.index[-1], target_col] = float(pred)
    return daily


def _predict_xgboost(
    model_path: Path,
    extended: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_col: str,
    first_day: date,
    days: int,
) -> list[dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("XGBoost no está instalado.")
    booster = xgb.XGBRegressor()
    booster.load_model(str(model_path))
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_d = first_day + timedelta(days=offset)
        _append_future_row_recursive(extended, target_d, target_col, feature_columns)
        wdf = extended[feature_columns].tail(window_size)
        x_flat = np.nan_to_num(wdf.to_numpy(dtype=np.float64).reshape(1, -1), nan=0.0)
        pred = booster.predict(x_flat)[0]
        daily.append({"date": target_d.isoformat(), "prediction": round(float(pred), 2)})
        extended.loc[extended.index[-1], target_col] = float(pred)
    return daily


def _lstm_module_class(input_size: int, hidden_size: int, num_layers: int, dropout: float) -> Any:
    class _LSTMRegressor(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
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

        def forward(self, x: Any) -> Any:
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    return _LSTMRegressor()


def _predict_lstm(
    bundle: dict[str, Any],
    extended: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_col: str,
    first_day: date,
    days: int,
) -> list[dict[str, Any]]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch no está disponible.")
    input_size = int(bundle["input_size"])
    hidden_size = int(bundle["hidden_size"])
    num_layers = int(bundle["num_layers"])
    dropout = float(bundle["dropout"])
    scalers = bundle["scalers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _lstm_module_class(input_size, hidden_size, num_layers, dropout).to(device)
    net.load_state_dict(bundle["model_state_dict"])
    net.eval()
    daily: list[dict[str, Any]] = []
    x_mean = np.asarray(scalers["x_mean"], dtype=np.float64)
    x_std = np.asarray(scalers["x_std"], dtype=np.float64)
    y_mean = float(scalers["y_mean"])
    y_std = float(scalers["y_std"])
    for offset in range(days):
        target_d = first_day + timedelta(days=offset)
        _append_future_row_recursive(extended, target_d, target_col, feature_columns)
        w = extended[feature_columns].tail(window_size).to_numpy(dtype=np.float64).reshape(1, window_size, -1)
        w = np.nan_to_num(w, nan=0.0)
        w_scaled = (w - x_mean) / x_std
        with torch.no_grad():
            pred_scaled = net(torch.tensor(w_scaled, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)[0]
        pred = float(pred_scaled * y_std + y_mean)
        daily.append({"date": target_d.isoformat(), "prediction": round(pred, 2)})
        extended.loc[extended.index[-1], target_col] = pred
    return daily


def _statsmodels_last_date(fitted: Any) -> date:
    fv = getattr(fitted, "fittedvalues", None)
    if fv is None or len(fv) == 0:
        raise ValueError("No se pudo inferir la última fecha del modelo (statsmodels).")
    if isinstance(fv.index, pd.RangeIndex):
        raise ValueError(
            "El modelo fue ajustado sin calendario en el índice; no se puede alinear el forecast con fechas."
        )
    return pd.Timestamp(fv.index[-1]).normalize().date()


def _predict_sarima(fitted: Any, first_day: date, days: int) -> list[dict[str, Any]]:
    last = _statsmodels_last_date(fitted)
    horizon_end = first_day + timedelta(days=days - 1)
    total_steps = (horizon_end - last).days
    if total_steps < 1:
        raise ValueError("start_date debe ser posterior al último día visto por el modelo SARIMA.")
    fc = fitted.get_forecast(steps=total_steps)
    means = np.asarray(fc.predicted_mean, dtype=float)
    pred_dates = [last + timedelta(days=i) for i in range(1, total_steps + 1)]
    by_date = {d: float(means[i]) for i, d in enumerate(pred_dates)}
    daily = []
    for i in range(days):
        d = first_day + timedelta(days=i)
        daily.append({"date": d.isoformat(), "prediction": round(by_date[d], 2)})
    return daily


def _predict_exponential_smoothing(fitted: Any, first_day: date, days: int) -> list[dict[str, Any]]:
    last = _statsmodels_last_date(fitted)
    horizon_end = first_day + timedelta(days=days - 1)
    total_steps = (horizon_end - last).days
    if total_steps < 1:
        raise ValueError("start_date debe ser posterior al último día del suavizado exponencial.")
    preds = fitted.forecast(steps=total_steps)
    arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds, dtype=float)
    pred_dates = [last + timedelta(days=i) for i in range(1, total_steps + 1)]
    by_date = {d: float(arr[i]) for i, d in enumerate(pred_dates)}
    daily = []
    for i in range(days):
        d = first_day + timedelta(days=i)
        daily.append({"date": d.isoformat(), "prediction": round(by_date[d], 2)})
    return daily


def _predict_neuralprophet(model: Any, first_day: date, days: int) -> list[dict[str, Any]]:
    if NeuralProphet is None:
        raise RuntimeError("NeuralProphet no está disponible.")
    hist_df = model.history
    if hist_df is None or hist_df.empty:
        raise ValueError("El modelo NeuralProphet no tiene historial interno.")
    last_hist = pd.Timestamp(hist_df["ds"].max()).normalize()
    first_ts = pd.Timestamp(first_day).normalize()
    horizon_end = first_ts + pd.Timedelta(days=days - 1)
    total_periods = (horizon_end - last_hist).days
    if total_periods < 1:
        raise ValueError("start_date debe ser posterior al último día de entrenamiento de NeuralProphet.")
    future_df = model.make_future_dataframe(
        hist_df, periods=int(total_periods), n_historic_predictions=False
    )
    forecast = model.predict(future_df)
    ds_norm = pd.to_datetime(forecast["ds"]).dt.normalize()
    sel = forecast[ds_norm >= first_ts].head(int(days))
    if len(sel) < days:
        raise ValueError("NeuralProphet no devolvió suficientes filas para el horizonte pedido.")
    daily = []
    for _, row in sel.iterrows():
        ds = row["ds"]
        d = ds.date() if hasattr(ds, "date") else pd.Timestamp(ds).date()
        daily.append({"date": d.isoformat(), "prediction": round(float(row["yhat1"]), 2)})
    return daily


def _predict_tft(
    df_features: pd.DataFrame,
    target_col: str,
    ckpt_path: Path,
    hp: dict[str, Any],
    first_day: date,
    days: int,
) -> list[dict[str, Any]]:
    if torch is None or TemporalFusionTransformer is None or TimeSeriesDataSet is None:
        raise RuntimeError("PyTorch Forecasting / PyTorch no está disponible para TFT.")

    max_encoder_length = _hp_int(hp, "tft_max_encoder_length", 28)
    batch_size = max(1, _hp_int(hp, "tft_batch_size", 32))

    raw_series = _prepare_target_series(df_features, target_col=target_col)
    work = _fill_series_for_model(raw_series).copy()
    last_hist = pd.Timestamp(work.index.max()).normalize().date()
    if first_day <= last_hist:
        raise ValueError("start_date debe ser posterior al último día de la serie usada para TFT.")

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(ckpt_path),
        map_location=torch.device("cpu"),
    )
    model.eval()
    daily: list[dict[str, Any]] = []

    gap = (first_day - last_hist).days - 1
    for _ in range(gap):
        next_fill = pd.Timestamp(work.index.max()) + pd.Timedelta(days=1)
        work.loc[next_fill] = float(work.iloc[-1])
        work = work.sort_index()

    for _ in range(days):
        data = work.reset_index()
        data.columns = ["created", target_col]
        data["time_idx"] = np.arange(len(data), dtype=int)
        data["series_id"] = "main"
        data["month"] = data["created"].dt.month.astype(str)
        data["weekday"] = data["created"].dt.weekday.astype(str)

        next_date = pd.Timestamp(work.index.max()) + pd.Timedelta(days=1)
        # TFT / TimeSeriesDataSet no permiten NaN en el target; placeholder = último valor observado.
        y_placeholder = float(pd.to_numeric(data[target_col], errors="coerce").iloc[-1])
        if not np.isfinite(y_placeholder):
            y_placeholder = 0.0
        ext = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "created": [next_date],
                        target_col: [y_placeholder],
                    }
                ),
            ],
            ignore_index=True,
        )
        ext["time_idx"] = np.arange(len(ext), dtype=int)
        ext["series_id"] = "main"
        ext["month"] = ext["created"].dt.month.astype(str)
        ext["weekday"] = ext["created"].dt.weekday.astype(str)
        ext[target_col] = pd.to_numeric(ext[target_col], errors="coerce").ffill().bfill().fillna(0.0)

        training_cutoff = len(data) - 1
        sub = ext[lambda x: x.time_idx <= training_cutoff].copy()
        sub["month"] = sub["month"].astype(str).astype("category")
        sub["weekday"] = sub["weekday"].astype(str).astype("category")
        ext["month"] = ext["month"].astype(str).astype("category")
        ext["weekday"] = ext["weekday"].astype(str).astype("category")

        training_ds = TimeSeriesDataSet(
            sub,
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
        val_ds = TimeSeriesDataSet.from_dataset(
            training_ds,
            ext,
            min_prediction_idx=len(data),
            stop_randomization=True,
        )
        loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        with torch.no_grad():
            p = model.predict(loader).detach().cpu().numpy().reshape(-1)
        pred = float(p[0]) if len(p) else float("nan")
        daily.append({"date": next_date.date().isoformat(), "prediction": round(pred, 2)})
        work.loc[next_date] = pred
        work = work.sort_index()

    return daily


def predict_winner(
    *,
    model_output_dir: str | Path,
    series_path: str | Path,
    target_col: str,
    first_day: date | None,
    days: int,
) -> dict[str, Any]:
    """
    Predicción multi-día según best_model.json + artefactos en model_output_dir.
    """
    out_dir = Path(model_output_dir)
    best, meta = load_training_artifacts(out_dir)
    model_name = best["model_name"]
    hp = meta.get("hyperparameters") or {}

    df = _read_features_df(Path(series_path), target_col)
    series = _prepare_target_series(df, target_col=target_col)
    first = first_day or (pd.Timestamp(series.index.max()).date() + timedelta(days=1))

    model_path = out_dir / best.get("promoted_model_filename", "")
    if not model_path.exists():
        model_path = Path(best["promoted_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo promovido: {model_path}")

    daily: list[dict[str, Any]]

    if model_name == "baseline_tm7_sw8_blend":
        baseline = joblib.load(model_path)
        if not isinstance(baseline, TrailingMeanWeekdayMedianBaseline):
            raise TypeError("Artefacto baseline inválido.")
        daily = _predict_baseline(baseline, series, first, days)

    elif model_name in {"linear_regression", "ridge", "random_forest", "catboost"}:
        feature_columns = list(meta.get("feature_columns") or [])
        ws = meta.get("window_shape")
        if not feature_columns or not ws or len(ws) < 2:
            raise ValueError("metadata.json debe incluir feature_columns y window_shape para modelos de ventana.")
        window_size = int(ws[1])
        extended = df.copy()
        _ensure_feature_frame(extended, feature_columns, target_col)
        m = joblib.load(model_path)
        daily = _predict_window_sklearn(m, extended, feature_columns, window_size, target_col, first, days)

    elif model_name == "xgboost":
        feature_columns = list(meta.get("feature_columns") or [])
        ws = meta.get("window_shape")
        if not feature_columns or not ws or len(ws) < 2:
            raise ValueError("metadata.json debe incluir feature_columns y window_shape para XGBoost.")
        window_size = int(ws[1])
        extended = df.copy()
        _ensure_feature_frame(extended, feature_columns, target_col)
        daily = _predict_xgboost(model_path, extended, feature_columns, window_size, target_col, first, days)

    elif model_name == "lstm":
        feature_columns = list(meta.get("feature_columns") or [])
        ws = meta.get("window_shape")
        if not feature_columns or not ws or len(ws) < 2:
            raise ValueError("metadata.json incompleto para LSTM.")
        window_size = int(ws[1])
        extended = df.copy()
        _ensure_feature_frame(extended, feature_columns, target_col)
        try:
            bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(model_path, map_location="cpu")
        if not isinstance(bundle, dict):
            raise TypeError("Formato lstm.pt inesperado.")
        daily = _predict_lstm(bundle, extended, feature_columns, window_size, target_col, first, days)

    elif model_name == "sarima":
        fitted = joblib.load(model_path)
        daily = _predict_sarima(fitted, first, days)

    elif model_name == "exponential_smoothing":
        fitted = joblib.load(model_path)
        daily = _predict_exponential_smoothing(fitted, first, days)

    elif model_name == "neuralprophet":
        m = joblib.load(model_path)
        daily = _predict_neuralprophet(m, first, days)

    elif model_name == "temporal_fusion_transformer":
        daily = _predict_tft(df, target_col, model_path, hp, first, days)

    else:
        raise ValueError(f"Modelo ganador no soportado para inferencia: {model_name}")

    result: dict[str, Any] = {
        "model_name": model_name,
        "metric_name": best.get("metric_name"),
        "metric_value": best.get("metric_value"),
        "target_col": target_col,
        "from": first.isoformat(),
        "days": days,
        "daily": daily,
    }
    return result

