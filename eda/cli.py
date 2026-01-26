from __future__ import annotations
import typer

from eda.cli_handlers import handle_analyze, handle_build_datasets, handle_test_db

app = typer.Typer(help="CLI para EDA de ventas por producto (orders.line_items)")

@app.command()
def test_db() -> None:
	"""Prueba la conexión a la base de datos."""
	handle_test_db()

@app.command()
def build_datasets(
	output_dir: str = typer.Option("reports/eda/data", help="Directorio de salida"),
	fmt: str = typer.Option("jsonl", help="Formato de salida: jsonl|csv|both"),
) -> None:
	"""Construye y guarda datasets (orders, ad_spends) desde la BD."""
	handle_build_datasets(output_dir=output_dir, fmt=fmt)

@app.command()
def analyze(
	input_dir: str = typer.Option("reports/eda/data", help="Directorio con los datasets guardados"),
	plots_dir: str = typer.Option("reports/eda/plots", help="Directorio para guardar gráficos"),
	no_with_plots: bool = typer.Option(False, help="No generar gráficos"),
) -> None:
	"""Carga datasets guardados, muestra resúmenes y genera gráficos."""
	handle_analyze(input_dir=input_dir, with_plots=not no_with_plots, plots_dir=plots_dir)

@app.command()
def prepare_data(no_with_plots: bool = typer.Option(False, help="(Deprecado)")) -> None:
	"""[Deprecado] Construye datasets; usar 'build-datasets' y 'analyze' separados."""
	handle_build_datasets(output_dir="reports/eda/data", fmt="jsonl")


if __name__ == "__main__":
	app()
