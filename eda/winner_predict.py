from __future__ import annotations

import json
from collections.abc import Mapping
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
    _apply_tft_adspend_scale,
    _finalize_tft_frame,
    _hp_int,
    _load_tft_adspend_scale,
    _prepare_target_series,
    _prepare_tft_dataframe,
    _tft_dataset_common_params,
    suppress_sklearn_feature_name_warning,
    tft_adspend_scale_path_for_ckpt,
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


_COMMERCIAL_COLS = ("adSpend", "rcRevenue", "totalRevenue")


def _commercial_fields_from_row(row: pd.Series) -> dict[str, Any | None]:
    """Valores comerciales de una fila del panel (para respuesta de /predict)."""
    out: dict[str, Any | None] = {}
    for col in _COMMERCIAL_COLS:
        if col not in row.index:
            out[col] = None
            continue
        v = row[col]
        out[col] = None if pd.isna(v) else round(float(v), 2)
    return out


def _fill_daily_commercial_from_features_df(
    daily: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    only_missing_keys: bool = False,
) -> None:
    """
    Completa adSpend, rcRevenue, totalRevenue desde el dataset de features por fecha.
    Si la fecha es posterior al último día observado, repite los valores del último día
    (misma lógica de “carry forward” que al armar filas futuras en modelos de ventana).
    Si only_missing_keys=True, no pisa claves cuyo valor ya es distinto de null.
    """
    if not daily:
        return
    have_cols = [c for c in _COMMERCIAL_COLS if c in df.columns]
    if not have_cols or "created" not in df.columns:
        for item in daily:
            for c in _COMMERCIAL_COLS:
                if only_missing_keys and item.get(c) is not None:
                    continue
                item[c] = None
        return

    work = df[["created"] + have_cols].copy()
    work["created"] = pd.to_datetime(work["created"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["created"])
    work["_d"] = work["created"].dt.date
    by_date = work.drop_duplicates(subset=["_d"], keep="last").set_index("_d")

    last_hist_d: date | None = None
    last_commercial: dict[str, Any] = {}
    if not by_date.empty:
        last_hist_d = by_date.index.max()
        tail = by_date.loc[last_hist_d]
        if isinstance(tail, pd.DataFrame):
            tail = tail.iloc[-1]
        for col in have_cols:
            last_commercial[col] = tail[col]

    for item in daily:
        d = date.fromisoformat(item["date"])
        for c in _COMMERCIAL_COLS:
            if only_missing_keys and item.get(c) is not None:
                continue
            if c not in have_cols:
                item[c] = None
                continue
            if d in by_date.index:
                v = by_date.loc[d, c]
                if isinstance(v, pd.Series):
                    v = v.iloc[-1]
                item[c] = None if pd.isna(v) else round(float(v), 2)
            elif last_hist_d is not None and d > last_hist_d:
                v = last_commercial.get(c)
                item[c] = None if v is None or pd.isna(v) else round(float(v), 2)
            else:
                item[c] = None


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
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
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
        if (
            col.startswith("orders_lag_")
            or col.startswith("adSpend_lag_")
            or col in ("revenue_growth", "investment_growth")
        ):
            continue
        if col not in new_row.index:
            new_row[col] = last.get(col, 0.0)
    extended.loc[len(extended)] = new_row
    idx = extended.index[-1]
    if future_ad_spend and "adSpend" in extended.columns and target_day in future_ad_spend:
        extended.loc[idx, "adSpend"] = float(future_ad_spend[target_day])
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
    if "investment_growth" in feature_columns and "adSpend" in extended.columns:
        prev = extended["adSpend"].shift(1).iloc[-1]
        cur = extended["adSpend"].iloc[-1]
        if pd.notna(prev) and prev != 0 and pd.notna(cur):
            extended.loc[idx, "investment_growth"] = float((cur - prev) / prev)
    if future_events is not None:
        ev_today = float(future_events.get(target_day, 0.0))
        ev_today_b = 1.0 if ev_today >= 0.5 else 0.0
        if "event_start" in extended.columns:
            extended.loc[idx, "event_start"] = ev_today_b
        if "event_start_next_1" in feature_columns:
            next_d = target_day + timedelta(days=1)
            ev_next = float(future_events.get(next_d, 0.0))
            extended.loc[idx, "event_start_next_1"] = 1.0 if ev_next >= 0.5 else 0.0
    elif "event_start_next_1" in feature_columns:
        extended.loc[idx, "event_start_next_1"] = 0.0


def _predict_window_sklearn(
    model: Any,
    extended: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_col: str,
    first_day: date,
    days: int,
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
) -> list[dict[str, Any]]:
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_d = first_day + timedelta(days=offset)
        _append_future_row_recursive(
            extended,
            target_d,
            target_col,
            feature_columns,
            future_ad_spend=future_ad_spend,
            future_events=future_events,
        )
        wdf = extended[feature_columns].tail(window_size)
        if len(wdf) < window_size:
            raise ValueError("Historia insuficiente para la ventana del modelo.")
        x_flat = wdf.to_numpy(dtype=np.float64).reshape(1, -1)
        if np.isnan(x_flat).any():
            x_flat = np.nan_to_num(x_flat, nan=0.0)
        pred = model.predict(x_flat)[0]
        extended.loc[extended.index[-1], target_col] = float(pred)
        row = extended.iloc[-1]
        day_out = {"date": target_d.isoformat(), "prediction": round(float(pred), 2)}
        day_out.update(_commercial_fields_from_row(row))
        daily.append(day_out)
    return daily


def _predict_xgboost(
    model_path: Path,
    extended: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_col: str,
    first_day: date,
    days: int,
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
) -> list[dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("XGBoost no está instalado.")
    booster = xgb.XGBRegressor()
    booster.load_model(str(model_path))
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_d = first_day + timedelta(days=offset)
        _append_future_row_recursive(
            extended,
            target_d,
            target_col,
            feature_columns,
            future_ad_spend=future_ad_spend,
            future_events=future_events,
        )
        wdf = extended[feature_columns].tail(window_size)
        x_flat = np.nan_to_num(wdf.to_numpy(dtype=np.float64).reshape(1, -1), nan=0.0)
        pred = booster.predict(x_flat)[0]
        extended.loc[extended.index[-1], target_col] = float(pred)
        row = extended.iloc[-1]
        day_out = {"date": target_d.isoformat(), "prediction": round(float(pred), 2)}
        day_out.update(_commercial_fields_from_row(row))
        daily.append(day_out)
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
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
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
        _append_future_row_recursive(
            extended,
            target_d,
            target_col,
            feature_columns,
            future_ad_spend=future_ad_spend,
            future_events=future_events,
        )
        w = extended[feature_columns].tail(window_size).to_numpy(dtype=np.float64).reshape(1, window_size, -1)
        w = np.nan_to_num(w, nan=0.0)
        w_scaled = (w - x_mean) / x_std
        with torch.no_grad():
            pred_scaled = net(torch.tensor(w_scaled, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)[0]
        pred = float(pred_scaled * y_std + y_mean)
        extended.loc[extended.index[-1], target_col] = pred
        row = extended.iloc[-1]
        day_out = {"date": target_d.isoformat(), "prediction": round(pred, 2)}
        day_out.update(_commercial_fields_from_row(row))
        daily.append(day_out)
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
    with suppress_sklearn_feature_name_warning():
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


def _tft_covariates_for_day(
    target_day: date,
    last_ad_spend: float,
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
) -> tuple[float, float]:
    if future_ad_spend and target_day in future_ad_spend:
        ad_spend = float(future_ad_spend[target_day])
    else:
        ad_spend = last_ad_spend
    if future_events is not None:
        ev = float(future_events.get(target_day, 0.0))
        event_start = 1.0 if ev >= 0.5 else 0.0
    else:
        event_start = 0.0
    return ad_spend, event_start


def _append_tft_future_row(
    data: pd.DataFrame,
    target_col: str,
    next_date: pd.Timestamp,
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
    y_placeholder: float | None = None,
) -> pd.DataFrame:
    target_day = next_date.date()
    last_ad = float(data["adSpend"].iloc[-1]) if "adSpend" in data.columns else 0.0
    ad_spend, event_start = _tft_covariates_for_day(
        target_day,
        last_ad,
        future_ad_spend=future_ad_spend,
        future_events=future_events,
    )
    if y_placeholder is None:
        y_placeholder = float(pd.to_numeric(data[target_col], errors="coerce").iloc[-1])
    if not np.isfinite(y_placeholder):
        y_placeholder = 0.0

    extended = pd.concat(
        [
            data,
            pd.DataFrame(
                {
                    "created": [next_date],
                    target_col: [y_placeholder],
                    "adSpend": [ad_spend],
                    "event_start": [event_start],
                }
            ),
        ],
        ignore_index=True,
    )
    return _finalize_tft_frame(extended, target_col)


def _predict_tft(
    df_features: pd.DataFrame,
    target_col: str,
    ckpt_path: Path,
    hp: dict[str, Any],
    first_day: date,
    days: int,
    *,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
) -> list[dict[str, Any]]:
    if torch is None or TemporalFusionTransformer is None or TimeSeriesDataSet is None:
        raise RuntimeError("PyTorch Forecasting / PyTorch no está disponible para TFT.")

    max_encoder_length = _hp_int(hp, "tft_max_encoder_length", 28)
    batch_size = max(1, _hp_int(hp, "tft_batch_size", 32))

    data = _prepare_tft_dataframe(df_features, target_col=target_col)
    adspend_scale = _load_tft_adspend_scale(tft_adspend_scale_path_for_ckpt(ckpt_path))
    last_hist = pd.Timestamp(data["created"].max()).normalize().date()
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
        next_date = pd.Timestamp(data["created"].max()) + pd.Timedelta(days=1)
        data = _append_tft_future_row(
            data,
            target_col,
            next_date,
            future_ad_spend=future_ad_spend,
            future_events=future_events,
        )

    for _ in range(days):
        next_date = pd.Timestamp(data["created"].max()) + pd.Timedelta(days=1)
        ext = _append_tft_future_row(
            data,
            target_col,
            next_date,
            future_ad_spend=future_ad_spend,
            future_events=future_events,
        )

        training_cutoff = len(data) - 1
        sub = ext[lambda x: x.time_idx <= training_cutoff].copy()
        sub_model = _apply_tft_adspend_scale(sub, adspend_scale) if adspend_scale else sub
        ext_model = _apply_tft_adspend_scale(ext, adspend_scale) if adspend_scale else ext
        training_ds = TimeSeriesDataSet(
            sub_model,
            **_tft_dataset_common_params(target_col, max_encoder_length),
        )
        val_ds = TimeSeriesDataSet.from_dataset(
            training_ds,
            ext_model,
            min_prediction_idx=len(data),
            stop_randomization=True,
        )
        loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        with torch.no_grad():
            p = model.predict(loader).detach().cpu().numpy().reshape(-1)
        pred = float(p[0]) if len(p) else float("nan")
        daily.append({"date": next_date.date().isoformat(), "prediction": round(pred, 2)})
        data = ext.copy()
        data.loc[data.index[-1], target_col] = pred
        data = _finalize_tft_frame(data, target_col)

    return daily


def predict_winner(
    *,
    model_output_dir: str | Path,
    series_path: str | Path,
    target_col: str,
    first_day: date | None,
    days: int,
    future_ad_spend: Mapping[date, float] | None = None,
    future_events: Mapping[date, float] | None = None,
) -> dict[str, Any]:
    """
    Predicción multi-día según best_model.json + artefactos en model_output_dir.

    future_ad_spend / future_events: aplican a modelos basados en ventana
    (linear_regression, ridge, random_forest, catboost, xgboost, lstm) y al TFT
    (adSpend, event_start, event_start_next_1 como covariables conocidas).
    """
    out_dir = Path(model_output_dir)
    best, meta = load_training_artifacts(out_dir)
    model_name = best["model_name"]
    hp = meta.get("hyperparameters") or {}

    ad_map: Mapping[date, float] | None = future_ad_spend if future_ad_spend else None
    ev_map: Mapping[date, float] | None = future_events if future_events else None

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
        daily = _predict_window_sklearn(
            m, extended, feature_columns, window_size, target_col, first, days,
            future_ad_spend=ad_map,
            future_events=ev_map,
        )

    elif model_name == "xgboost":
        feature_columns = list(meta.get("feature_columns") or [])
        ws = meta.get("window_shape")
        if not feature_columns or not ws or len(ws) < 2:
            raise ValueError("metadata.json debe incluir feature_columns y window_shape para XGBoost.")
        window_size = int(ws[1])
        extended = df.copy()
        _ensure_feature_frame(extended, feature_columns, target_col)
        daily = _predict_xgboost(
            model_path, extended, feature_columns, window_size, target_col, first, days,
            future_ad_spend=ad_map,
            future_events=ev_map,
        )

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
        daily = _predict_lstm(
            bundle, extended, feature_columns, window_size, target_col, first, days,
            future_ad_spend=ad_map,
            future_events=ev_map,
        )

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
        daily = _predict_tft(
            df,
            target_col,
            model_path,
            hp,
            first,
            days,
            future_ad_spend=ad_map,
            future_events=ev_map,
        )

    else:
        raise ValueError(f"Modelo ganador no soportado para inferencia: {model_name}")

    _fill_daily_commercial_from_features_df(daily, df, only_missing_keys=True)

    result: dict[str, Any] = {
        "model_name": model_name,
        "metric_name": best.get("metric_name"),
        "metric_value": best.get("metric_value"),
        "target_col": target_col,
        "from": first.isoformat(),
        "days": days,
        "daily": daily,
        "future_ad_spend_provided": bool(ad_map),
        "future_events_provided": bool(ev_map),
        "future_overrides_supported": (
            model_name
            in {
                "linear_regression",
                "ridge",
                "random_forest",
                "catboost",
                "xgboost",
                "lstm",
                "temporal_fusion_transformer",
            }
        ),
    }
    return result

