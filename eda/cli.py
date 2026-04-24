from __future__ import annotations
import typer

from eda.cli_handlers import (
	handle_analyze,
	handle_build_datasets,
	handle_build_features,
	handle_compare_models,
	handle_test_db,
)

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
def build_features(
	input_path: str = typer.Option("reports/eda/data/combined.jsonl", help="Dataset combinado (jsonl)"),
	output_dir: str = typer.Option("reports/eda/features", help="Directorio de salida"),
	lags: str = typer.Option("1,7,30", help="Lags separados por coma"),
	target_col: str = typer.Option("orders", help="Columna target"),
	window_size: int = typer.Option(28, help="Tamaño de ventana deslizante"),
) -> None:
	"""Construye features (lags + crecimiento) y ventanas deslizantes."""
	handle_build_features(
		input_path=input_path,
		output_dir=output_dir,
		lags=lags,
		target_col=target_col,
		window_size=window_size,
	)


@app.command()
def compare_models(
	input_path: str = typer.Option("reports/eda/features/windows.npz", help="Ventanas de entrenamiento"),
	output_dir: str = typer.Option("reports/eda/models", help="Directorio de salida"),
	series_path: str = typer.Option("reports/eda/features/features.jsonl", help="Serie temporal para NeuralProphet"),
	target_col: str = typer.Option("orders", help="Columna target (serie)"),
	val_ratio: float = typer.Option(0.2, help="Porcentaje de validación"),
	random_state: int = typer.Option(42, help="Random seed"),
) -> None:
	"""Entrena varios modelos y reporta métricas para comparar."""
	handle_compare_models(
		input_path=input_path,
		output_dir=output_dir,
		series_path=series_path,
		target_col=target_col,
		val_ratio=val_ratio,
		random_state=random_state,
	)

@app.command()
def prepare_data(no_with_plots: bool = typer.Option(False, help="(Deprecado)")) -> None:
	"""[Deprecado] Construye datasets; usar 'build-datasets' y 'analyze' separados."""
	handle_build_datasets(output_dir="reports/eda/data", fmt="jsonl")


if __name__ == "__main__":
	app()
