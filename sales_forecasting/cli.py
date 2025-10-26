from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from .db import (
    fetch_sales_timeseries,
    fetch_daily_features,
    get_all_product_ids,
    test_connection,
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
    product_id: str = typer.Option(..., help="ID del producto o 'all' (texto)"),
    start_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
    end_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
    target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice"),
) -> None:
    """Entrena modelo(s) por producto y guarda en models/."""
    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    if product_id == "all":
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
            path = sarimax_train(y, pid, target=target)  # type: ignore[arg-type]
        elif model == "rnn":
            X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
            path = rnn_train(y, pid, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
        else:
            typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
            raise typer.Exit(code=3)
        typer.secho(f"Modelo {model} guardado: {path}", fg=typer.colors.GREEN)


@app.command()
def predict(
    model: str = typer.Option("sarimax", help="Modelo: sarimax|rnn"),
    product_id: str = typer.Option(..., help="ID del producto (texto)"),
    horizon: int = typer.Option(14, help="Días a predecir"),
    target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice"),
    out: Optional[Path] = typer.Option(None, help="Archivo CSV de salida"),
) -> None:
    """Genera pronóstico para un producto y opcionalmente guarda a CSV."""
    if model == "sarimax":
        preds = sarimax_forecast(product_id, horizon, target=target)  # type: ignore[arg-type]
    elif model == "rnn":
        _, y = fetch_sales_timeseries(product_id, target=target)  # type: ignore[arg-type]
        _, X = fetch_daily_features(product_id)
        if y.size == 0:
            typer.secho("No hay datos para predecir", fg=typer.colors.RED)
            raise typer.Exit(code=4)
        X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
        preds = rnn_forecast(product_id, horizon, y, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
    else:
        typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
        raise typer.Exit(code=3)

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
