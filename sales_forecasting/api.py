from __future__ import annotations

from datetime import date
from typing import Literal, Optional, Union, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException

from .db import fetch_sales_timeseries, fetch_daily_features, fetch_sales_totals_timeseries
from .forecaster_sarimax import forecast as sarimax_forecast, train_and_save as sarimax_train
from .forecaster_rnn import forecast as rnn_forecast, train_and_save as rnn_train

app = FastAPI(title="Sales Forecasting API", version="0.1.0")


ModelName = Literal["sarimax", "rnn"]
TargetName = Literal["quantity", "totalPrice", "both"]


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
) -> Dict[str, str]:
    # Seleccionar origen de la serie: global vs por producto
    if product_id == "global":
        # Para target="both", entrenar ambas series
        targets: List[Literal["quantity", "totalPrice"]] = ["quantity", "totalPrice"] if target == "both" else [target]  # type: ignore[list-item]
        results: Dict[str, str] = {}
        for tgt in targets:
            dates, y = fetch_sales_totals_timeseries(start_date, end_date, target=tgt)
            # Excluir día actual si está presente (datos parciales)
            if dates.size and dates[-1] == np.datetime64("today", "D"):
                y = y[:-1]
            if y.size == 0:
                raise HTTPException(status_code=404, detail="No hay datos globales para entrenar")
            if model == "sarimax":
                path = sarimax_train(y, "global", target=tgt)
            elif model == "rnn":
                path = rnn_train(y, "global", target=tgt, Xexo=None)
                if path is None:
                    results[f"saved_{tgt}"] = ""
                    results[f"warning_{tgt}"] = "Serie demasiado corta para entrenar RNN"
                    continue
            else:
                raise HTTPException(status_code=400, detail="Modelo no soportado")
            results[f"saved_{tgt}"] = str(path)
        return results
    else:
        # Serie por producto
        dates, y = fetch_sales_timeseries(product_id, start_date, end_date, target=target if target != "both" else "quantity")
        _, X = fetch_daily_features(product_id, start_date, end_date)
        # Excluir día actual si está presente (datos parciales)
        if dates.size and dates[-1] == np.datetime64("today", "D"):
            y = y[:-1]
        if y.size == 0:
            raise HTTPException(status_code=404, detail="No hay datos para entrenar")

        if target == "both":
            results: Dict[str, str] = {}
            # Entrenar cantidad
            if model == "sarimax":
                p_qty = sarimax_train(y, product_id, target="quantity")
            else:
                X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
                p_qty = rnn_train(y, product_id, target="quantity", Xexo=X_aligned)
                if p_qty is None:
                    results["saved_quantity"] = ""
                    results["warning_quantity"] = "Serie demasiado corta para entrenar RNN"
                else:
                    results["saved_quantity"] = str(p_qty)

            # Releer y para totalPrice
            dates_tp, y_tp = fetch_sales_timeseries(product_id, start_date, end_date, target="totalPrice")
            if dates_tp.size and dates_tp[-1] == np.datetime64("today", "D"):
                y_tp = y_tp[:-1]
            if y_tp.size:
                if model == "sarimax":
                    p_tp = sarimax_train(y_tp, product_id, target="totalPrice")
                else:
                    X_aligned = X[-y_tp.shape[0] :, :] if X.size and X.shape[0] >= y_tp.shape[0] else None
                    p_tp = rnn_train(y_tp, product_id, target="totalPrice", Xexo=X_aligned)
                    if p_tp is None:
                        results["saved_totalPrice"] = ""
                        results["warning_totalPrice"] = "Serie demasiado corta para entrenar RNN"
                        return results
                results["saved_totalPrice"] = str(p_tp)
            return results

        # Caso normal con un solo target
        if model == "sarimax":
            path = sarimax_train(y, product_id, target=target)  # type: ignore[arg-type]
        elif model == "rnn":
            X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
            path = rnn_train(y, product_id, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
            if path is None:
                return {"saved": "", "warning": "Serie demasiado corta para entrenar RNN"}
        else:
            raise HTTPException(status_code=400, detail="Modelo no soportado")

        return {"saved": str(path)}


@app.get("/predict")
def predict(
    model: ModelName,
    product_id: str,
    horizon: int = 14,
    target: TargetName = "quantity",
) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    if target == "both":
        results: Dict[str, List[float]] = {}
        for tgt in ("quantity", "totalPrice"):
            if model == "sarimax":
                preds = sarimax_forecast("global" if product_id == "global" else product_id, horizon, target=tgt)  # type: ignore[arg-type]
            else:
                if product_id == "global":
                    _, y = fetch_sales_totals_timeseries(target=tgt)
                    X_aligned = None
                else:
                    _, y = fetch_sales_timeseries(product_id, target=tgt)  # type: ignore[arg-type]
                    _, X = fetch_daily_features(product_id)
                    if y.size == 0:
                        raise HTTPException(status_code=404, detail="No hay datos para predecir")
                    X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
                preds = rnn_forecast("global" if product_id == "global" else product_id, horizon, y, target=tgt, Xexo=X_aligned)  # type: ignore[arg-type]
            results[tgt] = [float(max(p, 0.0)) for p in preds.tolist()]
        return results

    if model == "sarimax":
        preds = sarimax_forecast("global" if product_id == "global" else product_id, horizon, target=target)  # type: ignore[arg-type]
    elif model == "rnn":
        if product_id == "global":
            _, y = fetch_sales_totals_timeseries(target=target)
            X_aligned = None
        else:
            _, y = fetch_sales_timeseries(product_id, target=target)  # type: ignore[arg-type]
            _, X = fetch_daily_features(product_id)
            if y.size == 0:
                raise HTTPException(status_code=404, detail="No hay datos para predecir")
            X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
        preds = rnn_forecast("global" if product_id == "global" else product_id, horizon, y, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    return {"forecast": [float(max(p, 0.0)) for p in preds.tolist()]}
