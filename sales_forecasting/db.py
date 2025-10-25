from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_settings


_settings = get_settings()
_engine: Engine = create_engine(_settings.database_url(), pool_pre_ping=True)


def get_engine() -> Engine:
    return _engine


def test_connection() -> None:
    with _engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def get_all_product_ids(limit: Optional[int] = None) -> list[str]:
    st = _settings
    sql = (
        f"SELECT DISTINCT li->> :pid_key AS product_id "
        f"FROM {st.orders_table}, LATERAL jsonb_array_elements({st.orders_line_items_col}) AS li "
        f"WHERE (li ? :pid_key) AND (li->>:pid_key IS NOT NULL) "
        f"ORDER BY product_id"
    )
    params = {"pid_key": st.item_product_id_key}
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with _engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [str(r[0]) for r in rows if r[0] is not None]


def fetch_sales_timeseries(
    product_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (dates_np, qty_np) para un product_id (texto) con frecuencia diaria y faltantes en 0."""
    st = _settings

    where_clauses = [f"(li->>:pid_key) = :product_id"]
    params: dict[str, object] = {"product_id": str(product_id), "pid_key": st.item_product_id_key}

    date_col = st.orders_date_col
    if start_date is not None:
        where_clauses.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where_clauses.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where_clauses)

    sql = text(
        f"""
        SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
               SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS qty
        FROM {st.orders_table}
        , LATERAL jsonb_array_elements({st.orders_line_items_col}) AS li
        WHERE {where_sql}
        GROUP BY dt
        ORDER BY dt
        """
    )

    params["qty_key"] = st.item_quantity_key

    with _engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float)

    dts = np.array([np.datetime64(r[0], "D") for r in rows], dtype="datetime64[D]")
    qty = np.array([float(r[1]) for r in rows], dtype=float)

    # Re-muestreo diario con relleno de 0
    start = dts[0]
    end = dts[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")

    date_to_qty = {int(dts[i].astype("datetime64[D]").astype(int)): qty[i] for i in range(len(dts))}
    full_qty = np.zeros(full_dates.shape[0], dtype=float)
    for i, d in enumerate(full_dates):
        key = int(d.astype("datetime64[D]").astype(int))
        full_qty[i] = date_to_qty.get(key, 0.0)

    return full_dates, full_qty


def fetch_last_date(product_id: str) -> Optional[np.datetime64]:
    st = _settings
    sql = text(
        f"""
        SELECT MAX((DATE_TRUNC('day', {st.orders_date_col}))::date) AS last_dt
        FROM {st.orders_table}
        , LATERAL jsonb_array_elements({st.orders_line_items_col}) AS li
        WHERE (li->>:pid_key) = :product_id
        """
    )
    params = {"product_id": str(product_id), "pid_key": st.item_product_id_key}
    with _engine.connect() as conn:
        row = conn.execute(sql, params).one_or_none()
    if not row or row[0] is None:
        return None
    return np.datetime64(row[0], "D")
