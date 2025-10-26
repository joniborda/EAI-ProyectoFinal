from __future__ import annotations

from datetime import date
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from .db import fetch_sales_timeseries, fetch_daily_features
from .forecaster_sarimax import forecast as sarimax_forecast, train_and_save as sarimax_train
from .forecaster_rnn import forecast as rnn_forecast, train_and_save as rnn_train

app = FastAPI(title="Sales Forecasting API", version="0.1.0")


ModelName = Literal["sarimax", "rnn"]
TargetName = Literal["quantity", "totalPrice"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(
    model: ModelName,
    product_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    target: TargetName = "quantity",
) -> dict[str, str]:
    dates, y = fetch_sales_timeseries(product_id, start_date, end_date, target=target)
    _, X = fetch_daily_features(product_id, start_date, end_date)
    if y.size == 0:
        raise HTTPException(status_code=404, detail="No hay datos para entrenar")

    if model == "sarimax":
        path = sarimax_train(y, product_id, target=target)
    elif model == "rnn":
        # Alinear shapes por si X incluye días fuera del rango exacto de y
        X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
        path = rnn_train(y, product_id, target=target, Xexo=X_aligned)
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    return {"saved": str(path)}


@app.get("/predict")
def predict(
    model: ModelName,
    product_id: str,
    horizon: int = 14,
    target: TargetName = "quantity",
) -> dict[str, list[float]]:
    if model == "sarimax":
        preds = sarimax_forecast(product_id, horizon, target=target)
    elif model == "rnn":
        dates, y = fetch_sales_timeseries(product_id, target=target)
        _, X = fetch_daily_features(product_id)
        if y.size == 0:
            raise HTTPException(status_code=404, detail="No hay datos para predecir")
        X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
        preds = rnn_forecast(product_id, horizon, y, target=target, Xexo=X_aligned)
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    return {"forecast": [float(max(p, 0.0)) for p in preds.tolist()]}
