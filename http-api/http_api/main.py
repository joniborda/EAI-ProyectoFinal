from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from eda.train import (
    TrailingMeanWeekdayMedianBaseline,
    _prepare_target_series,
)
from eda.training_dag import run_training_dag
from eda.winner_predict import predict_winner


DEFAULT_WINDOWS_PATH = Path(os.getenv("WINDOWS_PATH", "/app/reports/eda/features/windows.npz"))
DEFAULT_FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "/app/reports/eda/features/features.jsonl"))
DEFAULT_MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "/app/reports/eda/models"))
DEFAULT_TARGET_COL = os.getenv("TARGET_COL", "orders")
DEFAULT_SELECTION_METRIC = os.getenv("SELECTION_METRIC", "mae")

app = FastAPI(title="Sales Forecasting API", version="1.0.0")


class TrainRequest(BaseModel):
    input_path: str = Field(default=str(DEFAULT_WINDOWS_PATH))
    output_dir: str = Field(default=str(DEFAULT_MODEL_OUTPUT_DIR))
    series_path: str = Field(default=str(DEFAULT_FEATURES_PATH))
    target_col: str = Field(default=DEFAULT_TARGET_COL)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    selection_metric: str = Field(default=DEFAULT_SELECTION_METRIC)


def _read_features(series_path: Path, target_col: str) -> pd.Series:
    if not series_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe series_path: {series_path}")

    df = pd.read_json(series_path, lines=True, dtype=False)
    if "created" not in df.columns or target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas 'created' o '{target_col}' en {series_path}",
        )
    return _prepare_target_series(df, target_col=target_col)


def _next_date_after(series: pd.Series) -> date:
    return pd.Timestamp(series.index.max()).date() + timedelta(days=1)


def _parse_json_date_value_list(raw: str | None, param_name: str) -> dict[date, float] | None:
    """
    Espera un JSON array de objetos {"date": "YYYY-MM-DD", "value": number}.
    Devuelve None si el parámetro viene vacío u omitido.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name}: JSON inválido ({exc})",
        ) from exc
    if data is None:
        return None
    if not isinstance(data, list):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name}: se esperaba un array JSON, por ejemplo "
            + '[{"date":"2026-06-01","value":100.5}]',
        )
    out: dict[date, float] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400,
                detail=f"{param_name}[{i}]: cada elemento debe ser un objeto con "
                '"date" y "value"',
            )
        if "date" not in item or "value" not in item:
            raise HTTPException(
                status_code=400,
                detail=f"{param_name}[{i}]: faltan claves date o value",
            )
        d_raw = item["date"]
        if isinstance(d_raw, datetime):
            d = d_raw.date()
        elif isinstance(d_raw, date):
            d = d_raw
        elif isinstance(d_raw, str):
            d = date.fromisoformat(d_raw.strip()[:10])
        else:
            raise HTTPException(
                status_code=400,
                detail=f"{param_name}[{i}].date: tipo no soportado",
            )
        try:
            v = float(item["value"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{param_name}[{i}].value: se esperaba un número",
            ) from exc
        if not math.isfinite(v):
            raise HTTPException(
                status_code=400,
                detail=f"{param_name}[{i}].value: debe ser un número finito",
            )
        out[d] = v
    return out or None


def _read_best_model_metadata(model_output_dir: Path) -> dict[str, Any]:
    metadata_path = model_output_dir / "best_model.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe best_model.json en {model_output_dir}. Ejecutá /train primero.",
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(
    model_output_dir: str = Query(default=str(DEFAULT_MODEL_OUTPUT_DIR)),
) -> dict[str, Any]:
    metrics_path = Path(model_output_dir) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe metrics.json en {model_output_dir}")

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/train")
def train(request: TrainRequest) -> dict[str, str]:
    try:
        result = run_training_dag(
            input_path=request.input_path,
            output_dir=request.output_dir,
            series_path=request.series_path,
            target_col=request.target_col,
            val_ratio=request.val_ratio,
            random_state=request.random_state,
            selection_metric=request.selection_metric,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {key: str(value) for key, value in result.items()}


@app.get("/best-model")
def best_model(
    model_output_dir: str = Query(default=str(DEFAULT_MODEL_OUTPUT_DIR)),
) -> dict[str, Any]:
    return _read_best_model_metadata(Path(model_output_dir))


@app.get("/predict")
def predict(
    start_date: date | None = Query(default=None),
    days: int = Query(default=1, ge=1, le=365),
    model_output_dir: str = Query(default=str(DEFAULT_MODEL_OUTPUT_DIR)),
    series_path: str = Query(default=str(DEFAULT_FEATURES_PATH)),
    target_col: str = Query(default=DEFAULT_TARGET_COL),
    future_ad_spend: str | None = Query(
        default=None,
        description='Opcional. JSON: [{"date":"YYYY-MM-DD","value":123.4}, ...] gasto publicitario por día.',
    ),
    future_events: str | None = Query(
        default=None,
        description='Opcional. JSON: [{"date":"YYYY-MM-DD","value":1}, ...] con value 1 si hay evento ese día, 0 si no.',
    ),
) -> dict[str, Any]:
    series = _read_features(Path(series_path), target_col=target_col)
    first_day = start_date or _next_date_after(series)
    ad_map = _parse_json_date_value_list(future_ad_spend, "future_ad_spend")
    ev_map = _parse_json_date_value_list(future_events, "future_events")
    try:
        return predict_winner(
            model_output_dir=Path(model_output_dir),
            series_path=Path(series_path),
            target_col=target_col,
            first_day=first_day,
            days=days,
            future_ad_spend=ad_map,
            future_events=ev_map,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predict/baseline")
def predict_baseline(
    start_date: date | None = Query(default=None),
    days: int = Query(default=1, ge=1, le=365),
    series_path: str = Query(default=str(DEFAULT_FEATURES_PATH)),
    target_col: str = Query(default=DEFAULT_TARGET_COL),
) -> dict[str, Any]:
    series = _read_features(Path(series_path), target_col=target_col)
    first_day = start_date or _next_date_after(series)

    model = TrailingMeanWeekdayMedianBaseline().fit(series)
    daily: list[dict[str, Any]] = []
    for offset in range(days):
        target_day = first_day + timedelta(days=offset)
        components = model.components_for(target_day)
        prediction = model.predict_one(target_day)
        daily.append(
            {
                "date": target_day.isoformat(),
                "prediction": round(prediction, 2),
                "components": {
                    "trailing_mean_7": None
                    if components["trailing_mean_7"] is None
                    else round(components["trailing_mean_7"], 2),
                    "same_weekday_median_8w": None
                    if components["same_weekday_median_8w"] is None
                    else round(components["same_weekday_median_8w"], 2),
                },
            }
        )

    return {
        "target_col": target_col,
        "from": first_day.isoformat(),
        "days": days,
        "method": "0.45 * trailing_mean_7 + 0.55 * same_weekday_median_8w",
        "daily": daily,
    }
