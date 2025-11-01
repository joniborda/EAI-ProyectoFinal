from __future__ import annotations

from datetime import date
from typing import Literal, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_settings
from .data_prep import create_temporal_features


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


def fetch_sales_totals_timeseries(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Devuelve (dates_np, qty_np, totalPrice_np, qty_by_channel, qty_by_source, qty_by_product) agregados diarios a nivel global.

    Returns:
        dates_np: Array de fechas
        qty_np: Array de cantidades vendidas por día
        totalPrice_np: Array de precio total por día
        qty_by_channel: Dict[channel_name, quantity_array] - cantidad por channel alineada con dates_np
        qty_by_source: Dict[source_name, quantity_array] - cantidad por sourceName alineada con dates_np
        qty_by_product: Dict[product_id, quantity_array] - cantidad por productId alineada con dates_np
    """
    st = _settings

    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    date_col = _qi(st.orders_date_col)
    table = _qi(st.orders_table)
    line_items_col = _qi(st.orders_line_items_col)
    total_price_col = _qi(st.orders_total_price_col)
    channel_col = _qi(st.orders_channel_col)
    source_col = _qi(st.orders_source_name_col)

    where: list[str] = []
    params: dict[str, object] = {}
    if start_date is not None:
        where.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date
    where_sql = (" AND ".join(where)) if where else "TRUE"

    # Sumar cantidades de todos los line_items por día.
    sql = text(
        f"""
        SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
                SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS quantity,
                SUM((o.{total_price_col})::float) AS totalPrice
        FROM {table} AS o
        , LATERAL jsonb_array_elements(o.{line_items_col}) AS li
        WHERE o.{line_items_col} IS NOT NULL AND {where_sql}
        GROUP BY dt
        ORDER BY dt
        """
    )
    params["qty_key"] = st.item_quantity_key
    
    with _engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        empty_dates = np.array([], dtype="datetime64[D]")
        empty_values = np.array([], dtype=float)
        return empty_dates, empty_values, empty_values, {}, {}, {}

    dts = np.array([np.datetime64(r[0], "D") for r in rows], dtype="datetime64[D]")
    quantity = np.array([float(r[1]) for r in rows], dtype=float)
    totalPrice = np.array([float(r[2]) for r in rows], dtype=float)

    # Re-muestreo diario con relleno de 0
    start = dts[0]
    end = dts[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    
    date_to_qty = {int(dts[i].astype("datetime64[D]").astype(int)): quantity[i] for i in range(len(dts))}
    date_to_totalPrice = {int(dts[i].astype("datetime64[D]").astype(int)): totalPrice[i] for i in range(len(dts))}
    full_qty = np.zeros(full_dates.shape[0], dtype=float)
    full_totalPrice = np.zeros(full_dates.shape[0], dtype=float)

    for i, d in enumerate(full_dates):
        key = int(d.astype("datetime64[D]").astype(int))
        full_qty[i] = date_to_qty.get(key, 0.0)
        full_totalPrice[i] = date_to_totalPrice.get(key, 0.0)

    # Consulta por channel
    sql_by_channel = text(
        f"""
        SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
                o.{channel_col} AS channel,
                SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS quantity
        FROM {table} AS o
        , LATERAL jsonb_array_elements(o.{line_items_col}) AS li
        WHERE o.{line_items_col} IS NOT NULL AND {where_sql}
        GROUP BY dt, o.{channel_col}
        ORDER BY dt, o.{channel_col}
        """
    )
    
    with _engine.connect() as conn:
        rows_by_channel = conn.execute(sql_by_channel, params).fetchall()

    # Organizar por channel
    qty_by_channel: dict[str, dict[int, float]] = {}
    for r in rows_by_channel:
        dt_key = int(np.datetime64(r[0], "D").astype("datetime64[D]").astype(int))
        channel = str(r[1]) if r[1] else "unknown"
        qty = float(r[2])
        if channel not in qty_by_channel:
            qty_by_channel[channel] = {}
        qty_by_channel[channel][dt_key] = qty

    # Convertir a arrays alineados con full_dates
    qty_by_channel_arrays: dict[str, np.ndarray] = {}
    for channel, date_dict in qty_by_channel.items():
        arr = np.zeros(full_dates.shape[0], dtype=float)
        for i, d in enumerate(full_dates):
            key = int(d.astype("datetime64[D]").astype(int))
            arr[i] = date_dict.get(key, 0.0)
        qty_by_channel_arrays[channel] = arr

    # Consulta por sourceName
    sql_by_source = text(
        f"""
        SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
                o.{source_col} AS source_name,
                SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS quantity
        FROM {table} AS o
        , LATERAL jsonb_array_elements(o.{line_items_col}) AS li
        WHERE o.{line_items_col} IS NOT NULL AND {where_sql}
        GROUP BY dt, o.{source_col}
        ORDER BY dt, o.{source_col}
        """
    )
    
    with _engine.connect() as conn:
        rows_by_source = conn.execute(sql_by_source, params).fetchall()

    # Organizar por sourceName
    qty_by_source: dict[str, dict[int, float]] = {}
    for r in rows_by_source:
        dt_key = int(np.datetime64(r[0], "D").astype("datetime64[D]").astype(int))
        source = str(r[1]) if r[1] else "unknown"
        qty = float(r[2])
        if source not in qty_by_source:
            qty_by_source[source] = {}
        qty_by_source[source][dt_key] = qty

    # Convertir a arrays alineados con full_dates
    qty_by_source_arrays: dict[str, np.ndarray] = {}
    for source, date_dict in qty_by_source.items():
        arr = np.zeros(full_dates.shape[0], dtype=float)
        for i, d in enumerate(full_dates):
            key = int(d.astype("datetime64[D]").astype(int))
            arr[i] = date_dict.get(key, 0.0)
        qty_by_source_arrays[source] = arr

    # Consulta por productId
    sql_by_product = text(
        f"""
        SELECT (DATE_TRUNC('day', {date_col}))::date AS dt,
                li->>:pid_key AS product_id,
                SUM(CASE WHEN (li ? :qty_key) THEN (li->>:qty_key)::float ELSE 0 END) AS quantity
        FROM {table} AS o
        , LATERAL jsonb_array_elements(o.{line_items_col}) AS li
        WHERE o.{line_items_col} IS NOT NULL AND (li ? :pid_key) AND {where_sql}
        GROUP BY dt, li->>:pid_key
        ORDER BY dt, li->>:pid_key
        """
    )
    params["pid_key"] = st.item_product_id_key
    
    with _engine.connect() as conn:
        rows_by_product = conn.execute(sql_by_product, params).fetchall()

    # Organizar por productId
    qty_by_product: dict[str, dict[int, float]] = {}
    for r in rows_by_product:
        dt_key = int(np.datetime64(r[0], "D").astype("datetime64[D]").astype(int))
        product_id = str(r[1]) if r[1] else "unknown"
        qty = float(r[2])
        if product_id not in qty_by_product:
            qty_by_product[product_id] = {}
        qty_by_product[product_id][dt_key] = qty

    # Convertir a arrays alineados con full_dates
    qty_by_product_arrays: dict[str, np.ndarray] = {}
    for product_id, date_dict in qty_by_product.items():
        arr = np.zeros(full_dates.shape[0], dtype=float)
        for i, d in enumerate(full_dates):
            key = int(d.astype("datetime64[D]").astype(int))
            arr[i] = date_dict.get(key, 0.0)
        qty_by_product_arrays[product_id] = arr

    return full_dates, full_qty, full_totalPrice, qty_by_channel_arrays, qty_by_source_arrays, qty_by_product_arrays


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

    Columnas de X (12 features totales):
      Order features (0-5):
        0: orders_count
        1: unique_customers
        2: avg_order_total_price
        3: num_channels
        4: num_sources
        5: avg_num_tags
      Temporal features (6-11):
        6: day_of_week (0=Lunes, 6=Domingo)
        7: day_of_month (1-31)
        8: month (1-12)
        9: quarter (1-4)
        10: is_weekend (0/1)
        11: week_of_year (1-53)
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
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float).reshape(0, 12)

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

    # Agregar features temporales
    temporal_feats = create_temporal_features(full_dates)
    feats_with_temporal = np.concatenate([feats_full, temporal_feats], axis=1)

    return full_dates, feats_with_temporal


def fetch_global_daily_features(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (dates_np, X_np) con agregados diarios a nivel global.

    Columnas de X (12 features totales):
      Order features (0-5):
        0: orders_count
        1: unique_customers
        2: avg_order_total_price
        3: num_channels
        4: num_sources
        5: avg_num_tags
      Temporal features (6-11):
        6: day_of_week (0-6)
        7: day_of_month (1-31)
        8: month (1-12)
        9: quarter (1-4)
        10: is_weekend (0/1)
        11: week_of_year (1-53)
    """
    st = _settings

    def _qi(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    table = _qi(st.orders_table)
    date_col_q = _qi(st.orders_date_col)
    customer_col = _qi(st.orders_customer_id_col)
    total_price_col = _qi(st.orders_total_price_col)
    channel_col = _qi(st.orders_channel_col)
    source_col = _qi(st.orders_source_name_col)
    tags_col = _qi(st.orders_tags_col)

    where: list[str] = []
    params: dict[str, object] = {}
    if start_date is not None:
        where.append(f"{date_col_q} >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where.append(f"{date_col_q} <= :end_date")
        params["end_date"] = end_date
    where_sql = " AND ".join(where) if where else "TRUE"

    sql = text(
        f"""
        SELECT
            (DATE_TRUNC('day', {date_col_q}))::date AS dt,
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
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float).reshape(0, 12)

    dts = np.array([np.datetime64(r[0], 'D') for r in rows], dtype="datetime64[D]")
    feats = np.array(
        [
            [
                float(r[1]),
                float(r[2]),
                float(r[3] or 0.0),
                float(r[4]),
                float(r[5]),
                float(r[6] or 0.0),
            ]
            for r in rows
        ],
        dtype=float,
    )

    start = dts[0]
    end = dts[-1]
    full_dates = np.arange(start, end + np.timedelta64(1, 'D'), dtype="datetime64[D]")
    feats_full = np.zeros((full_dates.shape[0], feats.shape[1]), dtype=float)
    idx_map = {int(dts[i].astype('datetime64[D]').astype(int)): i for i in range(len(dts))}
    for i, d in enumerate(full_dates):
        key = int(d.astype('datetime64[D]').astype(int))
        j = idx_map.get(key)
        if j is not None:
            feats_full[i] = feats[j]
        else:
            feats_full[i] = 0.0

    temporal_feats = create_temporal_features(full_dates)
    feats_with_temporal = np.concatenate([feats_full, temporal_feats], axis=1)

    return full_dates, feats_with_temporal
