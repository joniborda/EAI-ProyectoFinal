from __future__ import annotations

from datetime import date
from typing import Literal, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_settings

_settings = get_settings()
_engine: Engine = create_engine(_settings.database_url(), pool_pre_ping=True)

DATE_FILTER = "2023-01-01"

def get_engine() -> Engine:
    return _engine

def test_connection() -> None:
    with _engine.connect() as conn:
        conn.execute(text("SELECT 1"))

def get_all_data() -> list[dict]:
    with _engine.connect() as conn:
        result = conn.execute(text(f"SELECT " +
        "id, created, \"customerId\", channel, \"sourceName\", \"tags\", " +
        "order_number_for_customer, " +
        f"jsonb_array_elements({_settings.orders_line_items_col}) as line_items FROM {_settings.orders_table} " +
        ### TODO: Quitar el filtro de fecha para obtener todos los datos
        f"where created >= '{DATE_FILTER}'"))
        return result.fetchall()

def get_all_ad_spends() -> list[dict]:
    with _engine.connect() as conn:
        result = conn.execute(text(f"SELECT "+
        " \"date\", \"adSpend\", \"rcRevenue\", \"totalRevenue\", \"totalMerchSold\", \"totalNewCustomerBarsSold\", " +
        " \"totalRecurringCustomerBarsSold\" " +
        f" FROM {_settings.ad_spends_table} " +
        ### TODO: Quitar el filtro de fecha para obtener todos los datos
        f"where date >= '{DATE_FILTER}'"))
        return result.fetchall()