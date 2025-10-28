from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

import typer
import uvicorn

from .db import (
    fetch_sales_timeseries,
    fetch_daily_features,
    get_all_product_ids,
    test_connection,
    fetch_sales_totals_timeseries,
)
from .forecaster_sarimax import forecast as sarimax_forecast, train_and_save as sarimax_train
from .forecaster_rnn import forecast as rnn_forecast, train_and_save as rnn_train

app = typer.Typer(help="CLI para predicción de ventas por producto (orders.line_items)")


@app.command()
def test_db() -> None:
    """Prueba la conexión a la base de datos."""
    try:
        test_connection()
        typer.secho("Conexión OK", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error de conexión: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def train(
    model: str = typer.Option("sarimax", help="Modelo: sarimax|rnn"),
    product_id: str = typer.Option(..., help="ID del producto, 'all' o 'global'"),
    start_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
    end_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
    target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice|both"),
) -> None:
    """Entrena modelos por producto o global y guarda en models/."""
    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    if product_id == "global":
        # Entrenamiento global agregado
        targets = ["quantity", "totalPrice"] if target == "both" else [target]
        for tgt in targets:
            dates, y = fetch_sales_totals_timeseries(sd, ed, target=tgt)  # type: ignore[arg-type]
            # Excluir día actual con datos parciales
            if dates.size and dates[-1] == np.datetime64("today", "D"):
                y = y[:-1]
            if y.size == 0:
                typer.secho("Sin datos globales, se omite.", fg=typer.colors.YELLOW)
                continue
            if model == "sarimax":
                path = sarimax_train(y, "global", target=tgt)  # type: ignore[arg-type]
            elif model == "rnn":
                path = rnn_train(y, "global", target=tgt, Xexo=None)  # type: ignore[arg-type]
                if path is None:
                    typer.secho(
                        f"Serie global demasiado corta para entrenar RNN (len={y.size}). Se omite.",
                        fg=typer.colors.YELLOW,
                    )
                    continue
            else:
                typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
                raise typer.Exit(code=3)
            typer.secho(f"Modelo {model} guardado (global {tgt}): {path}", fg=typer.colors.GREEN)
        return
    elif product_id == "all":
        ids = get_all_product_ids()
        if not ids:
            typer.secho("No se encontraron productos.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
    else:
        ids = [product_id]

    for pid in ids:
        _, y = fetch_sales_timeseries(pid, sd, ed, target=target)  # type: ignore[arg-type]
        _, X = fetch_daily_features(pid, sd, ed)
        if y.size == 0:
            typer.secho(f"Sin datos para producto {pid}, se omite.", fg=typer.colors.YELLOW)
            continue
        if model == "sarimax":
            if target == "both":
                p1 = sarimax_train(y, pid, target="quantity")  # type: ignore[arg-type]
                dates_tp, y_tp = fetch_sales_timeseries(pid, sd, ed, target="totalPrice")  # type: ignore[arg-type]
                if dates_tp.size and dates_tp[-1] == np.datetime64("today", "D"):
                    y_tp = y_tp[:-1]
                if y_tp.size:
                    p2 = sarimax_train(y_tp, pid, target="totalPrice")  # type: ignore[arg-type]
                    typer.secho(f"Modelos {model} guardados: {p1}, {p2}", fg=typer.colors.GREEN)
                else:
                    typer.secho(f"Modelo {model} guardado: {p1}", fg=typer.colors.GREEN)
                continue
            else:
                path = sarimax_train(y, pid, target=target)  # type: ignore[arg-type]
                typer.secho(f"Modelo {model} guardado: {path}", fg=typer.colors.GREEN)
                continue
        elif model == "rnn":
            if target == "both":
                # quantity
                X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
                print("X_aligned shape: ", X_aligned.shape, "last 10 values: ", X_aligned[-10:])
                p1 = rnn_train(y, pid, target="quantity", Xexo=X_aligned)  # type: ignore[arg-type]
                if p1 is None:
                    typer.secho(
                        f"RNN corta para quantity (product_id={pid}, len={y.size}).", fg=typer.colors.YELLOW
                    )
                # totalPrice
                dates_tp, y_tp = fetch_sales_timeseries(pid, sd, ed, target="totalPrice")  # type: ignore[arg-type]
                if dates_tp.size and dates_tp[-1] == np.datetime64("today", "D"):
                    y_tp = y_tp[:-1]
                X_aligned_tp = X[-y_tp.shape[0] :, :] if X.size and y_tp.size and X.shape[0] >= y_tp.shape[0] else None
                p2 = None
                if y_tp.size:
                    p2 = rnn_train(y_tp, pid, target="totalPrice", Xexo=X_aligned_tp)  # type: ignore[arg-type]
                if p2 is None and y_tp.size:
                    typer.secho(
                        f"RNN corta para totalPrice (product_id={pid}, len={y_tp.size}).", fg=typer.colors.YELLOW
                    )
                if p1 or p2:
                    typer.secho(
                        f"Modelos {model} guardados: {p1 if p1 else ''} {p2 if p2 else ''}", fg=typer.colors.GREEN
                    )
                continue
            else:
                X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
                print("X_aligned shape: ", X_aligned.shape, "last 10 values: ", X_aligned[-10:])
                
                path = rnn_train(y, pid, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
                if path is None:
                    typer.secho(
                        f"Serie demasiado corta para entrenar RNN (product_id={pid}, len={y.size}). Se omite.",
                        fg=typer.colors.YELLOW,
                    )
                    continue
        else:
            typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
            raise typer.Exit(code=3)
        typer.secho(f"Modelo {model} guardado: {path}", fg=typer.colors.GREEN)


@app.command()
def predict(
    model: str = typer.Option("sarimax", help="Modelo: sarimax|rnn"),
    product_id: str = typer.Option(..., help="ID del producto, 'global'"),
    horizon: int = typer.Option(14, help="Días a predecir"),
    target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice|both"),
    out: Optional[Path] = typer.Option(None, help="Archivo CSV de salida"),
) -> None:
    """Genera pronóstico para un producto o global y opcionalmente guarda a CSV."""
    def _predict_single(tgt: str):
        nonlocal model, product_id, horizon
        if model == "sarimax":
            return sarimax_forecast("global" if product_id == "global" else product_id, horizon, target=tgt)  # type: ignore[arg-type]
        elif model == "rnn":
            if product_id == "global":
                _, y = fetch_sales_totals_timeseries(target=tgt)  # type: ignore[arg-type]
                X_aligned = None
            else:
                _, y = fetch_sales_timeseries(product_id, target=tgt)  # type: ignore[arg-type]
                _, X = fetch_daily_features(product_id)
                if y.size == 0:
                    typer.secho("No hay datos para predecir", fg=typer.colors.RED)
                    raise typer.Exit(code=4)
                X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
            return rnn_forecast("global" if product_id == "global" else product_id, horizon, y, target=tgt, Xexo=X_aligned)  # type: ignore[arg-type]
        else:
            typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
            raise typer.Exit(code=3)

    if target == "both":
        preds_qty = _predict_single("quantity")
        preds_amt = _predict_single("totalPrice")
        lines = [
            "forecast_quantity,forecast_totalPrice",
            *[f"{float(q):.6f},{float(a):.6f}" for q, a in zip(preds_qty, preds_amt)],
        ]
        typer.secho("\n".join(lines), fg=typer.colors.BLUE)
        if out:
            with out.open("w", encoding="utf-8") as f:
                f.write("forecast_quantity,forecast_totalPrice\n")
                for q, a in zip(preds_qty, preds_amt):
                    f.write(f"{float(q):.6f},{float(a):.6f}\n")
            typer.secho(f"Guardado en {out}", fg=typer.colors.GREEN)
        return

    preds = _predict_single(target)

    # Mostrar y/o guardar
    lines = [f"{float(p):.4f}" for p in preds]
    typer.secho("\n".join(lines), fg=typer.colors.BLUE)
    if out:
        with out.open("w", encoding="utf-8") as f:
            f.write("forecast\n")
            for p in preds:
                f.write(f"{float(p):.6f}\n")
        typer.secho(f"Guardado en {out}", fg=typer.colors.GREEN)


@app.command()
def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Inicia el servicio FastAPI (uvicorn)."""
    uvicorn.run("sales_forecasting.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
