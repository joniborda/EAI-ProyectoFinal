from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import get_settings, safe_id_for_path


_settings = get_settings()


def _model_path(product_id: str, target: Literal["quantity", "totalPrice"]) -> Path:
    sid = safe_id_for_path(product_id)
    tgt = "qty" if target == "quantity" else "amt"
    return _settings.model_dir / f"sarimax_product_{sid}_{tgt}.joblib"


def train_and_save(
    y: np.ndarray,
    product_id: str,
    target: Literal["quantity", "totalPrice"] = "quantity",
    seasonal_periods: int = 7,
) -> Path:
    if y.size == 0:
        raise ValueError("No hay datos para entrenar SARIMAX.")

    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, int(seasonal_periods))

    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    path = _model_path(product_id, target)
    joblib.dump({"model": fitted, "order": order, "seasonal_order": seasonal_order}, path)
    return path


def load_model(product_id: str, target: Literal["quantity", "totalPrice"] = "quantity"):
    path = _model_path(product_id, target)
    if not path.exists():
        raise FileNotFoundError(f"Modelo SARIMAX no encontrado para product_id={product_id}: {path}")
    payload = joblib.load(path)
    return payload["model"]


def forecast(product_id: str, horizon_days: int, target: Literal["quantity", "totalPrice"] = "quantity") -> np.ndarray:
    fitted = load_model(product_id, target)
    fc = fitted.forecast(steps=int(horizon_days))
    fc = np.asarray(fc, dtype=float)
    fc[fc < 0.0] = 0.0
    return fc
