from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional

from eda.cli_handlers import handle_test_db
from eda.eda import handle_prepare_data

app = typer.Typer(help="CLI para EDA de ventas por producto (orders.line_items)")

@app.command()
def test_db() -> None:
	"""Prueba la conexión a la base de datos."""
	handle_test_db()

@app.command(
	
)
def prepare_data(no_with_plots: bool = typer.Option(True, help="Generar gráficos")) -> None:
	"""Prepara los datos para el EDA."""
	handle_prepare_data(with_plots=not no_with_plots)


if __name__ == "__main__":
	app()
