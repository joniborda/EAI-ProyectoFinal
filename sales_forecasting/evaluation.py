from __future__ import annotations

import numpy as np

from .db import fetch_sales_timeseries, fetch_daily_features
from .forecaster_sarimax import train_and_save as sarimax_train, forecast as sarimax_forecast
from .forecaster_rnn import train_and_save as rnn_train, forecast as rnn_forecast
from .data_prep import train_test_split_series, rmse, mae, mape


def evaluate(product_id: str, model: str = "sarimax", test_horizon: int = 14, target: str = "quantity") -> dict[str, float]:
    _, y = fetch_sales_timeseries(product_id, target=target)  # type: ignore[arg-type]
    _, X = fetch_daily_features(product_id)
    if y.size < test_horizon + 10:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

    y_train, y_test = train_test_split_series(y, test_horizon)

    if model == "sarimax":
        sarimax_train(y_train, product_id, target=target)  # type: ignore[arg-type]
        y_pred = sarimax_forecast(product_id, test_horizon, target=target)  # type: ignore[arg-type]
    elif model == "rnn":
        X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
        X_train_aligned = X_aligned[: y_train.shape[0], :] if X_aligned is not None else None
        rnn_train(y_train, product_id, target=target, Xexo=X_train_aligned)  # type: ignore[arg-type]
        y_pred = rnn_forecast(product_id, test_horizon, y_train, target=target, Xexo=X_train_aligned)  # type: ignore[arg-type]
    else:
        raise ValueError("Modelo no soportado")

    return {
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "mape": mape(y_test, y_pred),
    }
