from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .cli_handlers import (
	handle_test_db,
	handle_prepare_data,
	handle_train,
	handle_predict,
	handle_run_api,
)


app = typer.Typer(help="CLI para predicción de ventas por producto (orders.line_items)")


@app.command()
def test_db() -> None:
	"""Prueba la conexión a la base de datos."""
	handle_test_db()


@app.command()
def prepare_data(
	start_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
	end_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
	out_dir: Optional[Path] = typer.Option(None, help="Carpeta de salida para gráficos"),
) -> None:
	handle_prepare_data(start_date, end_date, out_dir)


@app.command()
def train(
	model: str = typer.Option("sarimax", help="Modelo: sarimax|rnn"),
	product_id: str = typer.Option(..., help="ID del producto, 'all' o 'global'"),
	start_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
	end_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (opcional)"),
	target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice|both"),
) -> None:
	handle_train(model, product_id, start_date, end_date, target)


@app.command()
def predict(
	model: str = typer.Option("sarimax", help="Modelo: sarimax|rnn"),
	product_id: str = typer.Option(..., help="ID del producto, 'global'"),
	horizon: int = typer.Option(14, help="Días a predecir"),
	target: str = typer.Option("quantity", help="Objetivo: quantity|totalPrice|both"),
	out: Optional[Path] = typer.Option(None, help="Archivo CSV de salida"),
) -> None:
	handle_predict(model, product_id, horizon, target, out)


@app.command()
def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
	"""Inicia el servicio FastAPI (uvicorn)."""
	handle_run_api(host, port)


if __name__ == "__main__":
	app()
