#!/usr/bin/env python3
"""

Este archivo es una heurística de como se calculaba en Inventory Tracker la estimación.


Estimar totalRevenue cuando no tenés cohorts ni inputs manuales: solo historia en
`forecasts.totalRevenue` (heurística).

Métodos (--method):
  trailing_mean_7, same_weekday_median, blend (default 0.45*trail + 0.55*weekday).

Modos de uso:
  1) Un día:     --date 2026-04-15
  2) Ventana:    --from 2026-04-01 --days 14   (default --days con --from es 14)
  3) Equivalente: --date 2026-04-01 --days 14
     Predice cada día inclusive; día D usa solo filas con fecha < D.

Salida en modo ventana: lista `daily` + `predicted_total_sum` (suma de los días).

Requisitos: pip install psycopg2-binary

Ejemplos:
  python remix/scripts/estimate_total_revenue_from_history.py --date 2026-04-15
  python remix/scripts/estimate_total_revenue_from_history.py --from 2026-04-01 --days 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

try:
    from eda.train import TrailingMeanWeekdayMedianBaseline
except ImportError:  # pragma: no cover - permite ejecutar como `python eda/original_prediction.py`
    from train import TrailingMeanWeekdayMedianBaseline


def parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def fetch_series_strictly_before(conn: Any, before: date, limit: int = 500) -> list[tuple[date, float]]:
    """Todas las filas con date < before (útil para cargar de una vez hasta el final del rango)."""
    import psycopg2.extras

    sql = """
    SELECT d, tr FROM (
      SELECT date::text AS d, "totalRevenue"::float AS tr
      FROM forecasts
      WHERE date < %s::date
        AND "totalRevenue" IS NOT NULL
      ORDER BY date DESC
      LIMIT %s
    ) sub
    ORDER BY d ASC
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, (before.isoformat(), limit))
    rows = cur.fetchall()
    cur.close()
    out: list[tuple[date, float]] = []
    for r in rows:
        try:
            rd = r.get("d") if hasattr(r, "get") else r["d"]
            rt = r.get("tr") if hasattr(r, "get") else r["tr"]
            d = datetime.strptime(str(rd), "%Y-%m-%d").date()
            v = float(rt or 0)
        except (TypeError, ValueError, KeyError):
            continue
        out.append((d, v))
    return out


def trailing_mean_n(series: list[tuple[date, float]], target: date, n: int) -> float | None:
    model = _baseline_from_series(series, trailing_days=n)
    if model is None:
        return None
    return model.components_for(target)["trailing_mean_7"]


def same_weekday_median(
    series: list[tuple[date, float]], target: date, weeks_back: int = 8
) -> float | None:
    model = _baseline_from_series(series, weekday_weeks=weeks_back)
    if model is None:
        return None
    return model.components_for(target)["same_weekday_median_8w"]


def blend(a: float | None, b: float | None, wa: float = 0.45, wb: float = 0.55) -> float | None:
    if a is not None and b is not None:
        return wa * a + wb * b
    if a is not None:
        return a
    if b is not None:
        return b
    return None


def estimate_for_target(
    series: list[tuple[date, float]], target: date, method: str
) -> tuple[float | None, dict[str, Any]]:
    model = _baseline_from_series(series)
    if model is None:
        return None, {"trailing_mean_7": None, "same_weekday_median_8w": None}

    components = model.components_for(target)
    tm = components["trailing_mean_7"]
    sw = components["same_weekday_median_8w"]
    if method == "trailing_mean_7":
        est = tm
    elif method == "same_weekday_median":
        est = sw
    else:
        est = blend(tm, sw)
    comp = {
        "trailing_mean_7": None if tm is None else round(tm, 2),
        "same_weekday_median_8w": None if sw is None else round(sw, 2),
    }
    return est, comp


def _baseline_from_series(
    series: list[tuple[date, float]],
    trailing_days: int = 7,
    weekday_weeks: int = 8,
) -> TrailingMeanWeekdayMedianBaseline | None:
    if not series:
        return None

    dates = [d for d, _ in series]
    values = [v for _, v in series]
    pd_series = pd.Series(values, index=pd.to_datetime(dates), dtype=float)
    return TrailingMeanWeekdayMedianBaseline(
        trailing_days=trailing_days,
        weekday_weeks=weekday_weeks,
    ).fit(pd_series)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Heurística totalRevenue desde historia en forecasts."
    )
    p.add_argument("--date", help="Primer día YYYY-MM-DD (con --days>1 es inicio de rango)")
    p.add_argument("--from", dest="from_date", help="Sinónimo de inicio; si hay --date y --from, gana --from")
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Largo del rango (default: 1 solo con --date; 14 si solo pasás --from)",
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="Postgres (default: DATABASE_URL)",
    )
    p.add_argument(
        "--method",
        choices=("blend", "trailing_mean_7", "same_weekday_median"),
        default="blend",
        help="Método de estimación",
    )
    p.add_argument(
        "--history-limit",
        type=int,
        default=500,
        help="Máx. filas forecasts con date < límite (orden asc)",
    )
    args = p.parse_args()

    if not args.database_url:
        print("Falta DATABASE_URL o --database-url", file=sys.stderr)
        return 1

    if not args.date and not args.from_date:
        print("Indicá --date YYYY-MM-DD o --from YYYY-MM-DD", file=sys.stderr)
        return 1

    if args.from_date is not None:
        start = parse_date(args.from_date)
        n_days = args.days if args.days is not None else 14
    else:
        start = parse_date(args.date)  # type: ignore[arg-type]
        n_days = args.days if args.days is not None else 1

    if n_days < 1:
        print("--days debe ser >= 1", file=sys.stderr)
        return 1

    try:
        import psycopg2
        import psycopg2.extras  # noqa: F401
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        return 1

    # Último día a predecir (inclusive); para ese día hace falta historia date < last+1
    last_day = start + timedelta(days=n_days - 1)
    fetch_before = last_day + timedelta(days=1)

    conn = psycopg2.connect(args.database_url)
    try:
        series = fetch_series_strictly_before(conn, fetch_before, limit=args.history_limit)
    finally:
        conn.close()

    daily_out: list[dict[str, Any]] = []
    total_sum = 0.0
    all_ok = True

    for i in range(n_days):
        d = start + timedelta(days=i)
        est, comp = estimate_for_target(series, d, args.method)
        row: dict[str, Any] = {
            "date": d.isoformat(),
            "estimated_totalRevenue": None if est is None else round(est, 2),
            "components": comp,
        }
        daily_out.append(row)
        if est is not None:
            total_sum += est
        else:
            all_ok = False

    out: dict[str, Any] = {
        "from": start.isoformat(),
        "days": n_days,
        "through": last_day.isoformat(),
        "method": args.method,
        "history_points_loaded": len(series),
        "daily": daily_out,
        "predicted_total_sum": round(total_sum, 2),
        "all_days_estimated": all_ok,
        "notes": [
            "Cada día D usa solo forecasts con fecha estrictamente menor a D.",
            "predicted_total_sum suma solo días con estimación no nula.",
        ],
    }

    if not all_ok:
        out["warning"] = (
            "Al menos un día sin estimación (poca historia para ese weekday o <7 puntos)."
        )

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
