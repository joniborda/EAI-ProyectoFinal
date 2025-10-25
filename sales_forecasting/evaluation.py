from __future__ import annotations

import numpy as np

from .db import fetch_sales_timeseries
from .forecaster_sarimax import train_and_save as sarimax_train, forecast as sarimax_forecast
from .forecaster_rnn import train_and_save as rnn_train, forecast as rnn_forecast
from .data_prep import train_test_split_series, rmse, mae, mape


def evaluate(product_id: str, model: str = "sarimax", test_horizon: int = 14) -> dict[str, float]:
    _, y = fetch_sales_timeseries(product_id)
    if y.size < test_horizon + 10:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

    y_train, y_test = train_test_split_series(y, test_horizon)

    if model == "sarimax":
        sarimax_train(y_train, product_id)
        y_pred = sarimax_forecast(product_id, test_horizon)
    elif model == "rnn":
        rnn_train(y_train, product_id)
        y_pred = rnn_forecast(product_id, test_horizon, y_train)
    else:
        raise ValueError("Modelo no soportado")

    return {
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "mape": mape(y_test, y_pred),
    }
