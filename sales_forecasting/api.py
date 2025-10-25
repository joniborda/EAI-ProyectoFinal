from __future__ import annotations

from datetime import date
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from .db import fetch_sales_timeseries
from .forecaster_sarimax import forecast as sarimax_forecast, train_and_save as sarimax_train
from .forecaster_rnn import forecast as rnn_forecast, train_and_save as rnn_train

app = FastAPI(title="Sales Forecasting API", version="0.1.0")


ModelName = Literal["sarimax", "rnn"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(
    model: ModelName,
    product_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, str]:
    _, y = fetch_sales_timeseries(product_id, start_date, end_date)
    if y.size == 0:
        raise HTTPException(status_code=404, detail="No hay datos para entrenar")

    if model == "sarimax":
        path = sarimax_train(y, product_id)
    elif model == "rnn":
        path = rnn_train(y, product_id)
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    return {"saved": str(path)}


@app.get("/predict")
def predict(
    model: ModelName,
    product_id: str,
    horizon: int = 14,
) -> dict[str, list[float]]:
    if model == "sarimax":
        preds = sarimax_forecast(product_id, horizon)
    elif model == "rnn":
        _, y = fetch_sales_timeseries(product_id)
        if y.size == 0:
            raise HTTPException(status_code=404, detail="No hay datos para predecir")
        preds = rnn_forecast(product_id, horizon, y)
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    return {"forecast": [float(max(p, 0.0)) for p in preds.tolist()]}
