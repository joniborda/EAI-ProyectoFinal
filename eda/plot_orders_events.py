from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

def orders_daily_counts(df_orders: pd.DataFrame) -> pd.DataFrame:
    """Agrupa conteo de filas por día (misma lógica que group_orders_per_day_and_plot)."""
    work = df_orders.copy()
    if "created" not in work.columns:
        raise ValueError("Se espera columna 'created' en el DataFrame de órdenes.")
    created_dt = pd.to_datetime(work["created"], errors="coerce")
    work["created"] = created_dt.dt.normalize()
    out = (
        work.dropna(subset=["created"])
        .groupby("created", as_index=False)
        .size()
        .rename(columns={"size": "orders"})
        .sort_values("created")
    )
    return out


def drop_last_incomplete_day(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Quita el último día de la serie ordenada por fecha (día en curso / cierre parcial típico).
    Alineado con ``trim_combined_incomplete_last_day`` en ``eda.data_prep``.
    """
    if df_daily.empty:
        return df_daily
    sorted_df = df_daily.sort_values("created").reset_index(drop=True)
    if len(sorted_df) < 2:
        raise ValueError(
            "No hay al menos dos días de datos: no se puede excluir el último día incompleto y graficar."
        )
    return sorted_df.iloc[:-1].copy()


def load_orders_daily_from_db() -> pd.DataFrame:
    from eda.db import get_all_data

    data = get_all_data()
    df = pd.DataFrame(data)
    return orders_daily_counts(df)


def load_orders_daily_from_jsonl(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype=False)
    return orders_daily_counts(df)


def normalize_event_dates(raw: Iterable) -> list[pd.Timestamp]:
    """Lista de timestamps normalizados a medianoche (un solo valor por día)."""
    series = pd.to_datetime(pd.Series(list(raw)), errors="coerce").dropna()
    if series.empty:
        return []
    return sorted(series.dt.normalize().unique().tolist())


def load_event_dates_from_db() -> list[pd.Timestamp]:
    from eda.db import get_all_event_starts

    rows = get_all_event_starts()
    df = pd.DataFrame(rows)
    if df.empty or "startDate" not in df.columns:
        return []
    return normalize_event_dates(df["startDate"])


def load_event_dates_from_jsonl(path: str | Path) -> list[pd.Timestamp]:
    df = pd.read_json(path, lines=True, dtype=False)
    if df.empty or "startDate" not in df.columns:
        return []
    return normalize_event_dates(df["startDate"])


def _plot_orders_events_pillow(
    output_path: Path,
    df_plot: pd.DataFrame,
    event_dates_in_range: list[pd.Timestamp],
    *,
    title: str,
    y_label: str,
) -> None:
    """Fallback sin matplotlib (evita RecursionError en savefig en algunos entornos Windows)."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 1200, 720
    left, right, top, bottom = 100, 50, 95, 95
    pw, ph = w - left - right, h - top - bottom

    ts = pd.to_datetime(df_plot["created"])
    y = df_plot["orders"].astype(float).to_numpy()
    n = len(ts)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos para dibujar la serie.")

    t0 = ts.iloc[0]
    rel_x = (ts - t0).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    x_min, x_max = float(rel_x.min()), float(rel_x.max())
    span_x = x_max - x_min if x_max > x_min else 1.0
    y_min = float(y.min())
    y_max = float(y.max())
    span_y = y_max - y_min
    if span_y < 1e-9:
        y_min -= 1.0
        y_max += 1.0
        span_y = y_max - y_min

    def px_x(rx: float) -> int:
        return int(left + (rx - x_min) / span_x * pw)

    def px_y(yv: float) -> int:
        return int(top + ph - (yv - y_min) / span_y * ph)

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    draw.line((left, top + ph, left + pw, top + ph), fill="black", width=2)
    draw.line((left, top, left, top + ph), fill="black", width=2)

    pts = [(px_x(float(rel_x[i])), px_y(float(y[i]))) for i in range(n)]
    if len(pts) > 1:
        draw.line(pts, fill="#1f77b4", width=2)

    red = (220, 50, 50)
    for d in event_dates_in_range:
        rx = (pd.Timestamp(d).normalize() - pd.Timestamp(t0).normalize()).total_seconds() / 86400.0
        xv = px_x(float(rx))
        if left <= xv <= left + pw:
            draw.line((xv, top, xv, top + ph), fill=red, width=1)

    font = ImageFont.load_default()
    draw.text((20, 12), title, fill="black", font=font)
    draw.text((left, h - 70), "Fecha (días desde inicio)", fill="black", font=font)
    draw.text((8, top + ph // 2), y_label[:40], fill="black", font=font)

    for k in range(0, 6):
        yy = y_min + (k / 5) * span_y
        ypix = px_y(yy)
        draw.line((left - 4, ypix, left, ypix), fill="black")
        draw.text((left - 75, ypix - 6), f"{yy:.4g}", fill="black", font=font)

    draw.text((left - 5, top + ph + 8), f"{t0.strftime('%Y-%m-%d')}", fill="black", font=font)
    draw.text((left + pw - 120, top + ph + 8), f"{ts.iloc[-1].strftime('%Y-%m-%d')}", fill="black", font=font)

    leg_y = top + 10
    draw.line((left + pw - 140, leg_y + 4, left + pw - 120, leg_y + 4), fill="#1f77b4", width=3)
    draw.text((left + pw - 115, leg_y - 4), "Órdenes", fill="black", font=font)
    draw.line((left + pw - 140, leg_y + 24, left + pw - 120, leg_y + 24), fill=red, width=2)
    draw.text((left + pw - 115, leg_y + 16), "Evento", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def plot_orders_daily_with_event_markers(
    df_daily: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    *,
    group_months: int | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    event_line_color: str = "tab:red",
    event_linestyle: str = "--",
    event_alpha: float = 0.55,
    title: str | None = None,
) -> Path | None:
    """
    Serie **diaria** lineal de órdenes (sin agregar valores).

    ``group_months`` controla solo el **intervalo entre marcas principales del eje X**
    (cada N meses). La serie sigue siendo un punto por día.
    """
    if df_daily.empty:
        raise ValueError("No hay datos de órdenes para graficar.")

    import matplotlib

    if not show:
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    tick_months = int(group_months) if group_months is not None else 3
    tick_months = max(1, tick_months)

    df_plot = df_daily.copy()
    x = pd.to_datetime(df_plot["created"])
    y = df_plot["orders"].astype(float)
    x_min, x_max = x.min(), x.max()

    y_label_str = "Cantidad de órdenes (por día)"
    default_title = "Cantidad de órdenes por día e inicios de eventos"
    title_str = title if title else default_title
    in_range = [d for d in event_dates if x_min <= d <= x_max]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="tab:blue", linewidth=1.0, label="Órdenes por día")
    ax.set_ylabel(y_label_str)

    ax.set_title(title_str)

    for i, d in enumerate(in_range):
        ax.axvline(
            d,
            color=event_line_color,
            linestyle=event_linestyle,
            alpha=event_alpha,
            linewidth=1.2,
            label="Inicio de evento (startDate)" if i == 0 else None,
        )

    ax.set_xlabel(f"Fecha (marcas cada {tick_months} mes(es))")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.18, top=0.9)

    out: Path | None = None
    used_pillow = False
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(out, dpi=150)
        except RecursionError:
            plt.close(fig)
            used_pillow = True
            _plot_orders_events_pillow(
                out,
                df_plot,
                in_range,
                title=title_str,
                y_label=y_label_str,
            )

    if used_pillow:
        if show:
            from PIL import Image

            Image.open(out).show()
    elif show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)

    return out


def run_plot_orders_events(
    *,
    orders_path: str | Path | None,
    events_path: str | Path | None,
    use_db: bool,
    output_path: str | Path | None,
    show: bool,
    group_months: int | None = None,
) -> dict[str, object]:
    """
    Carga órdenes y eventos desde archivos (si existen y no use_db) o desde la BD.

    Siempre excluye el **último día** de la serie diaria de órdenes antes de graficar
    (ese día suele tener información incompleta).
    """
    orders_file = Path(orders_path) if orders_path else Path("reports/eda/data/orders.jsonl")
    events_file = Path(events_path) if events_path else Path("reports/eda/data/events.jsonl")

    if use_db or not orders_file.exists():
        df_daily = load_orders_daily_from_db()
        source_orders = "database"
    else:
        df_daily = load_orders_daily_from_jsonl(orders_file)
        source_orders = str(orders_file)

    n_days_before_trim = len(df_daily)
    df_daily = drop_last_incomplete_day(df_daily)

    if use_db or not events_file.exists():
        event_dates = load_event_dates_from_db()
        source_events = "database"
    else:
        event_dates = load_event_dates_from_jsonl(events_file)
        source_events = str(events_file)

    saved = plot_orders_daily_with_event_markers(
        df_daily,
        event_dates,
        group_months=group_months,
        output_path=output_path,
        show=show,
    )

    c0 = pd.Timestamp(df_daily["created"].min()).normalize()
    c1 = pd.Timestamp(df_daily["created"].max()).normalize()
    n_in_range = len([d for d in event_dates if c0 <= d <= c1])

    gm = int(group_months) if group_months is not None else 3
    gm = max(1, gm)

    return {
        "orders_source": source_orders,
        "events_source": source_events,
        "group_months": gm,
        "n_points_plotted": len(df_daily),
        "n_days_raw": len(df_daily),
        "n_days_before_trim": n_days_before_trim,
        "last_day_excluded": True,
        "n_events_in_range": n_in_range,
        "n_events_total": len(event_dates),
        "output_path": str(saved) if saved else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gráfico de órdenes diarias con líneas verticales en fechas de eventos.",
    )
    parser.add_argument(
        "--orders-path",
        default=None,
        help="orders.jsonl (por defecto reports/eda/data/orders.jsonl si existe, si no BD)",
    )
    parser.add_argument(
        "--events-path",
        default=None,
        help="events.jsonl (por defecto reports/eda/data/events.jsonl si existe, si no BD)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Leer órdenes y eventos desde la base de datos",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Ruta PNG de salida (opcional)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana interactiva",
    )
    parser.add_argument(
        "--group-months",
        type=int,
        default=3,
        metavar="N",
        help="Intervalo en meses entre marcas del eje X (la serie sigue siendo diaria). 1 = cada mes. Por defecto: 3",
    )
    args = parser.parse_args()
    gm = args.group_months if args.group_months >= 1 else 1
    info = run_plot_orders_events(
        orders_path=args.orders_path,
        events_path=args.events_path,
        use_db=args.from_db,
        output_path=args.output,
        show=not args.no_show,
        group_months=gm,
    )
    print(info)
