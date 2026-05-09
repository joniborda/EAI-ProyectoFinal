from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from eda.db import get_all_ad_spends, get_all_data, get_all_event_starts


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_existing(path: Path) -> None:
    if path.exists():
        path.unlink()


def trim_combined_incomplete_last_day(
    df: pd.DataFrame, *, date_col: str = "created"
) -> pd.DataFrame:
    """
    Ordena por fecha y elimina la última fila (día en curso / datos incompletos típicos al exportar).
    """
    if df.empty or date_col not in df.columns:
        return df.copy()
    out = (
        df.assign(_sort_ts=pd.to_datetime(df[date_col], errors="coerce"))
        .sort_values("_sort_ts")
        .drop(columns=["_sort_ts"])
        .reset_index(drop=True)
    )
    return out.iloc[:-1].reset_index(drop=True)


def trim_existing_combined_jsonl(output_dir: str | Path) -> None:
    """Si existe combined.jsonl, lo reescribe sin la última fila por día."""
    path = Path(output_dir) / "combined.jsonl"
    if not path.exists():
        return
    df = pd.read_json(path, lines=True, dtype=False)
    trimmed = trim_combined_incomplete_last_day(df)
    _remove_existing(path)
    trimmed.to_json(path, orient="records", lines=True, date_format="iso")


def _save_dataframe(
    df: pd.DataFrame,
    base_path: Path,
    name: str,
    fmt: Literal["jsonl", "csv", "both"] = "jsonl",
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if fmt in ("jsonl", "both"):
        jsonl_path = base_path / f"{name}.jsonl"
        _remove_existing(jsonl_path)
        df.to_json(jsonl_path, orient="records", lines=True, date_format="iso")
        outputs["jsonl"] = jsonl_path
    if fmt in ("csv", "both"):
        csv_path = base_path / f"{name}.csv"
        _remove_existing(csv_path)
        df.to_csv(csv_path, index=False)
        outputs["csv"] = csv_path
    return outputs


def build_orders_dataset(
    output_dir: str | Path = "reports/eda/data",
    fmt: Literal["jsonl", "csv", "both"] = "jsonl",
) -> dict[str, Path]:
    """
    Lee órdenes desde la BD, deriva columnas de fecha y guarda el DataFrame en disco.
    Retorna las rutas de salida generadas.
    """
    data = get_all_data()
    df_data = pd.DataFrame(data)

    # Derivación de columnas temporales
    created_dt = pd.to_datetime(df_data["created"])
    df_data["created_weekday"] = created_dt.dt.dayofweek  # 0..6
    df_data["created_month"] = created_dt.dt.month        # 1..12
    df_data["created"] = created_dt.dt.date               # solo fecha (día)

    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    return _save_dataframe(df_data, out_dir, "orders", fmt)


def build_ad_spends_dataset(
    output_dir: str | Path = "reports/eda/data",
    fmt: Literal["jsonl", "csv", "both"] = "jsonl",
) -> dict[str, Path]:
    """
    Lee ad_spends desde la BD, normaliza tipos y guarda el DataFrame en disco.
    Retorna las rutas de salida generadas.
    """
    ad_spends = get_all_ad_spends()
    ad_spends_df = pd.DataFrame(ad_spends)
    ad_spends_df["date"] = pd.to_datetime(ad_spends_df["date"]).dt.date
    ad_spends_df["adSpend"] = pd.to_numeric(ad_spends_df["adSpend"], errors="coerce").fillna(0).astype(float)
    ad_spends_df["rcRevenue"] = pd.to_numeric(ad_spends_df["rcRevenue"], errors="coerce").fillna(0).astype(float)
    ad_spends_df["totalRevenue"] = pd.to_numeric(ad_spends_df["totalRevenue"], errors="coerce").fillna(0).astype(float)
    ad_spends_df["totalMerchSold"] = pd.to_numeric(ad_spends_df["totalMerchSold"], errors="coerce").fillna(0).astype(int)
    ad_spends_df["totalNewCustomerBarsSold"] = pd.to_numeric(
        ad_spends_df["totalNewCustomerBarsSold"], errors="coerce"
    ).fillna(0).astype(int)
    ad_spends_df["totalRecurringCustomerBarsSold"] = pd.to_numeric(
        ad_spends_df["totalRecurringCustomerBarsSold"], errors="coerce"
    ).fillna(0).astype(int)

    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    return _save_dataframe(ad_spends_df, out_dir, "ad_spends", fmt)


def build_events_dataset(
    output_dir: str | Path = "reports/eda/data",
    fmt: Literal["jsonl", "csv", "both"] = "jsonl",
) -> dict[str, Path]:
    """
    Lee eventos desde la BD y conserva solo la fecha de comienzo.
    """
    event_starts = get_all_event_starts()
    events_df = pd.DataFrame(event_starts)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["startDate"])
    else:
        events_df["startDate"] = pd.to_datetime(events_df["startDate"], errors="coerce").dt.date
        events_df = events_df.dropna(subset=["startDate"]).drop_duplicates().sort_values("startDate")

    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    return _save_dataframe(events_df, out_dir, "events", fmt)


def build_datasets(
    output_dir: str | Path = "reports/eda/data",
    fmt: Literal["jsonl", "csv", "both"] = "jsonl",
) -> dict[str, dict[str, Path]]:
    """
    Construye y guarda datasets base desde la BD.
    Si ya existe combined.jsonl (p. ej. tras un analyze previo), quita la última fila (día incompleto).
    """
    out: dict[str, dict[str, Path]] = {
        "orders": build_orders_dataset(output_dir=output_dir, fmt=fmt),
        "ad_spends": build_ad_spends_dataset(output_dir=output_dir, fmt=fmt),
        "events": build_events_dataset(output_dir=output_dir, fmt=fmt),
    }
    trim_existing_combined_jsonl(output_dir)
    return out



