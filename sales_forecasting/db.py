from __future__ import annotations

from datetime import date
from typing import Literal, Optional

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
    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'
    table = _qi(st.orders_table)
    line_items_col = _qi(st.orders_line_items_col)

    print(_settings)

    sql = (
        f"SELECT DISTINCT li->> :pid_key AS product_id "
        f"FROM {table}, LATERAL jsonb_array_elements({line_items_col}) AS li "
        f"WHERE {line_items_col} IS NOT NULL AND (li ? :pid_key) AND (li->>:pid_key IS NOT NULL) "
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
    target: Literal["quantity", "totalPrice"] = "quantity",
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (dates_np, y_np) para un product_id (texto) con frecuencia diaria y faltantes en 0.

    target:
        - "quantity": suma de cantidades vendidas (line_items.quantity)
        - "totalPrice": suma del totalPrice de la orden, prorrateada por ítem del producto
    """
    st = _settings

    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    where_clauses = [f"(li->>:pid_key) = :product_id"]
    params: dict[str, object] = {"product_id": str(product_id), "pid_key": st.item_product_id_key}

    date_col = _qi(st.orders_date_col)
    table = _qi(st.orders_table)
    line_items_col = _qi(st.orders_line_items_col)
    total_price_col = _qi(st.orders_total_price_col)
    if start_date is not None:
        where_clauses.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where_clauses.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where_clauses)

    if target == "quantity":
        sql = text(
            f"""
            SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
                   SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS y
            FROM {table}
            , LATERAL jsonb_array_elements({line_items_col}) AS li
            WHERE {where_sql}
            GROUP BY dt
            ORDER BY dt
            """
        )
        params["qty_key"] = st.item_quantity_key
    else:
        # Para totalPrice (a nivel orden), prorrateamos por la fracción de cantidad del producto
        # respecto al total de items de la orden, y luego sumamos por día.
        sql = text(
            f"""
            WITH base AS (
                SELECT
                    (DATE_TRUNC('day', {date_col}))::date AS dt,
                    li AS li_json,
                    (
                        SELECT COALESCE(SUM((li2->>:qty_key)::float), 0)
                        FROM jsonb_array_elements(o.{line_items_col}) AS li2
                    ) AS order_total_qty,
                    (o.{total_price_col})::float AS order_total_price
                FROM {table} AS o,
                     LATERAL jsonb_array_elements(o.{line_items_col}) AS li
                WHERE {where_sql}
            )
            SELECT dt,
                   SUM(
                       CASE
                           WHEN (li_json ? :pid_key) AND (li_json->>:pid_key) = :product_id
                                AND (li_json ? :qty_key) AND order_total_qty > 0
                           THEN ( (li_json->>:qty_key)::float / order_total_qty ) * order_total_price
                           ELSE 0
                       END
                   ) AS y
            FROM base
            WHERE dt IS NOT NULL
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
    y = np.array([float(r[1]) for r in rows], dtype=float)

    # Re-muestreo diario con relleno de 0
    start = dts[0]
    end = dts[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")

    date_to_qty = {int(dts[i].astype("datetime64[D]").astype(int)): y[i] for i in range(len(dts))}
    full_qty = np.zeros(full_dates.shape[0], dtype=float)
    for i, d in enumerate(full_dates):
        key = int(d.astype("datetime64[D]").astype(int))
        full_qty[i] = date_to_qty.get(key, 0.0)

    return full_dates, full_qty


def fetch_last_date(product_id: str) -> Optional[np.datetime64]:
    st = _settings
    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'
    table = _qi(st.orders_table)
    date_col = _qi(st.orders_date_col)
    line_items_col = _qi(st.orders_line_items_col)
    sql = text(
        f"""
        SELECT MAX((DATE_TRUNC('day', {date_col}))::date) AS last_dt
        FROM {table}
        , LATERAL jsonb_array_elements({line_items_col}) AS li
        WHERE (li->>:pid_key) = :product_id
        """
    )
    params = {"product_id": str(product_id), "pid_key": st.item_product_id_key}
    with _engine.connect() as conn:
        row = conn.execute(sql, params).one_or_none()
    if not row or row[0] is None:
        return None
    return np.datetime64(row[0], "D")


def fetch_daily_features(
    product_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (dates_np, X_np) con agregados diarios por producto a nivel orden.

    Columnas de X:
      0: orders_count
      1: unique_customers
      2: avg_order_total_price
      3: num_channels
      4: num_sources
      5: avg_num_tags
    """
    st = _settings

    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    table = _qi(st.orders_table)
    date_col = _qi(st.orders_date_col)
    line_items_col = _qi(st.orders_line_items_col)
    customer_col = _qi(st.orders_customer_id_col)
    total_price_col = _qi(st.orders_total_price_col)
    channel_col = _qi(st.orders_channel_col)
    source_col = _qi(st.orders_source_name_col)
    tags_col = _qi(st.orders_tags_col)

    date_col = st.orders_date_col
    where: list[str] = []
    params: dict[str, object] = {"product_id": str(product_id), "pid_key": st.item_product_id_key}

    # Solo considerar órdenes que contienen el product_id en sus line_items
    where.append(
        f"EXISTS (SELECT 1 FROM jsonb_array_elements({line_items_col}) AS li WHERE (li->>:pid_key) = :product_id)"
    )

    if start_date is not None:
        where.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where)

    sql = text(
        f"""
        SELECT
            (DATE_TRUNC('day', {date_col}))::date AS dt,
            COUNT(*)::float AS orders_count,
            COUNT(DISTINCT {customer_col})::float AS unique_customers,
            AVG(({total_price_col})::float) AS avg_order_total_price,
            COUNT(DISTINCT {channel_col})::float AS num_channels,
            COUNT(DISTINCT {source_col})::float AS num_sources,
            AVG(COALESCE(jsonb_array_length({tags_col}), 0))::float AS avg_num_tags
        FROM {table}
        WHERE {where_sql}
        GROUP BY dt
        ORDER BY dt
        """
    )

    with _engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float).reshape(0, 6)

    dts = np.array([np.datetime64(r[0], "D") for r in rows], dtype="datetime64[D]")
    feats = np.array([[float(r[1]), float(r[2]), float(r[3] or 0.0), float(r[4]), float(r[5]), float(r[6] or 0.0)] for r in rows], dtype=float)

    # Relleno de días faltantes con ceros, manteniendo 6 columnas
    start = dts[0]
    end = dts[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    feats_full = np.zeros((full_dates.shape[0], feats.shape[1]), dtype=float)
    idx_map = {int(dts[i].astype("datetime64[D]").astype(int)): i for i in range(len(dts))}
    for i, d in enumerate(full_dates):
        key = int(d.astype("datetime64[D]").astype(int))
        j = idx_map.get(key)
        if j is not None:
            feats_full[i] = feats[j]
        else:
            feats_full[i] = 0.0

    return full_dates, feats_full
