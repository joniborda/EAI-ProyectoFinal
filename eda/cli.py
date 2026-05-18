from __future__ import annotations
import typer

from eda.metric_plots import _DIST_HIST_BINS
from eda.cli_handlers import (
	handle_analyze,
	handle_build_datasets,
	handle_build_features,
	handle_compare_models,
	handle_grid_search,
	handle_plot_mape_distribution,
	handle_plot_orders_events,
	handle_plot_orders_pct_change,
	handle_plot_orders_lag7_pct_change,
	handle_plot_true_vs_predictions,
	handle_test_db,
	handle_training_dag,
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
	"""Construye y guarda datasets (orders, ad_spends, events) desde la BD; recorta última fila de combined.jsonl si existe."""
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
	imputation_strategy: str = typer.Option("median", help="Imputación para features numéricas: mean|median"),
) -> None:
	"""Construye features (lags + crecimiento) y ventanas deslizantes."""
	handle_build_features(
		input_path=input_path,
		output_dir=output_dir,
		lags=lags,
		target_col=target_col,
		window_size=window_size,
		imputation_strategy=imputation_strategy,
	)


@app.command()
def compare_models(
	input_path: str = typer.Option("reports/eda/features/windows.npz", help="Ventanas de entrenamiento"),
	output_dir: str = typer.Option("reports/eda/models", help="Directorio de salida"),
	series_path: str = typer.Option("reports/eda/features/features.jsonl", help="Serie temporal para NeuralProphet"),
	target_col: str = typer.Option("orders", help="Columna target (serie)"),
	val_ratio: float = typer.Option(0.2, help="Porcentaje de validación"),
	random_state: int = typer.Option(42, help="Random seed"),
	hyperparameters_json: str | None = typer.Option(None, help="JSON o ruta a JSON con hiperparámetros"),
) -> None:
	"""Entrena varios modelos y reporta métricas para comparar."""
	handle_compare_models(
		input_path=input_path,
		output_dir=output_dir,
		series_path=series_path,
		target_col=target_col,
		val_ratio=val_ratio,
		random_state=random_state,
		hyperparameters_json=hyperparameters_json,
	)

@app.command()
def training_dag(
	input_path: str = typer.Option("reports/eda/features/windows.npz", help="Ventanas de entrenamiento"),
	output_dir: str = typer.Option("reports/eda/models", help="Directorio de salida"),
	series_path: str = typer.Option("reports/eda/features/features.jsonl", help="Serie temporal"),
	target_col: str = typer.Option("orders", help="Columna target (serie)"),
	val_ratio: float = typer.Option(0.2, help="Porcentaje de validación"),
	random_state: int = typer.Option(42, help="Random seed"),
	selection_metric: str = typer.Option("mae", help="Métrica para elegir ganador: mae|rmse|mape"),
	hyperparameters_json: str | None = typer.Option(None, help="JSON o ruta a JSON con hiperparámetros"),
) -> None:
	"""Ejecuta entrenamiento, selección del ganador y promoción para predict."""
	handle_training_dag(
		input_path=input_path,
		output_dir=output_dir,
		series_path=series_path,
		target_col=target_col,
		val_ratio=val_ratio,
		random_state=random_state,
		selection_metric=selection_metric,
		hyperparameters_json=hyperparameters_json,
	)


@app.command()
def grid_search(
	model_name: str = typer.Argument(..., help="Modelo a ejecutar"),
	param_grid_json: str | None = typer.Option(None, help="JSON o ruta a JSON con rangos de hiperparámetros"),
	input_path: str = typer.Option("reports/eda/features/windows.npz", help="Ventanas de entrenamiento"),
	series_path: str = typer.Option("reports/eda/features/features.jsonl", help="Serie temporal"),
	output_dir: str = typer.Option("reports/eda/grid_search", help="Directorio de salida"),
	target_col: str = typer.Option("orders", help="Columna target (serie)"),
	val_ratio: float = typer.Option(0.2, help="Porcentaje de validación"),
	random_state: int = typer.Option(42, help="Random seed"),
	selection_metric: str = typer.Option("mae", help="Métrica para elegir ganador: mae|rmse|mape"),
) -> None:
	"""Ejecuta grid search para un modelo específico."""
	handle_grid_search(
		model_name=model_name,
		param_grid_json=param_grid_json,
		input_path=input_path,
		series_path=series_path,
		output_dir=output_dir,
		target_col=target_col,
		val_ratio=val_ratio,
		random_state=random_state,
		selection_metric=selection_metric,
	)


@app.command()
def plot_mape_distribution(
	input_path: str = typer.Option("reports/eda/models/mape_distribution.jsonl", help="CSV/JSON/JSONL con valores de MAPE"),
	output_path: str = typer.Option("reports/eda/plots/mape_distribution.png", help="PNG de salida"),
	model_name: str | None = typer.Option("random_forest", help="Modelo a filtrar; usar vacío para no filtrar"),
	metric_col: str = typer.Option(
		"mape",
		help="mape | rmse | squared_error | abs_error (rmse/squared_error/abs_error usan y_true,y_pred del JSONL)",
	),
	model_col: str | None = typer.Option("model", help="Columna con nombre de modelo; usar vacío si no aplica"),
	bins: int = typer.Option(_DIST_HIST_BINS, help="Cantidad de bins del histograma"),
	no_show: bool = typer.Option(False, help="Guardar sin abrir ventana"),
) -> None:
	"""Histograma + KDE de MAPE, o de (y-ŷ)² / |y-ŷ| para ver errores alineados con RMSE / escala del target."""
	handle_plot_mape_distribution(
		input_path=input_path,
		output_path=output_path,
		model_name=model_name or None,
		metric_col=metric_col,
		model_col=model_col or None,
		bins=bins,
		show=not no_show,
	)


@app.command("plot-orders-events")
def plot_orders_events(
	orders_path: str | None = typer.Option(
		None,
		help="orders.jsonl; por defecto reports/eda/data/orders.jsonl si existe, si no BD",
	),
	events_path: str | None = typer.Option(
		None,
		help="events.jsonl; por defecto reports/eda/data/events.jsonl si existe, si no BD",
	),
	from_db: bool = typer.Option(False, help="Forzar lectura de órdenes y eventos desde la BD"),
	output_path: str | None = typer.Option(
		None,
		help="Guardar PNG (ej. reports/eda/plots/orders_with_events.png)",
	),
	no_show: bool = typer.Option(False, help="Guardar sin abrir ventana"),
	group_months: int = typer.Option(
		3,
		help="Marcas del eje X cada N meses (serie siempre diaria, sin sumar). 1 = cada mes",
	),
) -> None:
	"""Órdenes por día (línea diaria); eje X con marcas cada N meses; líneas verticales en eventos."""
	handle_plot_orders_events(
		orders_path=orders_path,
		events_path=events_path,
		from_db=from_db,
		output_path=output_path,
		show=not no_show,
		group_months=group_months,
	)


@app.command("plot-orders-pct-change")
def plot_orders_pct_change(
	orders_path: str | None = typer.Option(
		None,
		help="orders.jsonl; por defecto reports/eda/data/orders.jsonl si existe, si no BD",
	),
	events_path: str | None = typer.Option(
		None,
		help="events.jsonl; por defecto reports/eda/data/events.jsonl si existe, si no BD",
	),
	from_db: bool = typer.Option(False, help="Forzar lectura de órdenes y eventos desde la BD"),
	output_path: str | None = typer.Option(
		None,
		help="Guardar PNG (ej. reports/eda/plots/orders_pct_change_with_events.png)",
	),
	no_show: bool = typer.Option(False, help="Guardar sin abrir ventana"),
	group_months: int = typer.Option(
		3,
		help="Marcas del eje X cada N meses (serie siempre diaria, sin sumar). 1 = cada mes",
	),
) -> None:
	"""Variación %% día a día de órdenes; mismas marcas de eventos que plot-orders-events."""
	handle_plot_orders_pct_change(
		orders_path=orders_path,
		events_path=events_path,
		from_db=from_db,
		output_path=output_path,
		show=not no_show,
		group_months=group_months,
	)


@app.command("plot-orders-lag7-pct-change")
def plot_orders_lag7_pct_change(
	features_path: str | None = typer.Option(
		None,
		help="features.jsonl con orders_lag_7 (default: reports/eda/features/features.jsonl)",
	),
	events_path: str | None = typer.Option(
		None,
		help="events.jsonl; por defecto reports/eda/data/events.jsonl si existe, si no BD",
	),
	from_db: bool = typer.Option(False, help="Forzar lectura de eventos desde la BD"),
	output_path: str | None = typer.Option(
		None,
		help="Guardar PNG (ej. reports/eda/plots/orders_lag7_pct_change_with_events.png)",
	),
	no_show: bool = typer.Option(False, help="Guardar sin abrir ventana"),
	group_months: int = typer.Option(
		3,
		help="Marcas del eje X cada N meses (serie diaria). 1 = cada mes",
	),
) -> None:
	"""Variación %% diaria de órdenes suavizada con media móvil centrada de 7 días (features)."""
	handle_plot_orders_lag7_pct_change(
		features_path=features_path,
		events_path=events_path,
		from_db=from_db,
		output_path=output_path,
		show=not no_show,
		group_months=group_months,
	)


@app.command("plot-predictions")
def plot_predictions(
	model_name: str = typer.Argument(..., help="Nombre del modelo (ej. random_forest, temporal_fusion_transformer)"),
	input_path: str = typer.Option(
		"reports/eda/models/mape_distribution.jsonl",
		help="JSONL con y_true/y_pred por modelo (salida de compare-models / training-dag)",
	),
	output_path: str | None = typer.Option(
		None,
		help="PNG de salida (por defecto reports/eda/plots/true_vs_pred_<modelo>.png)",
	),
	target_col: str = typer.Option("orders", help="Texto en el título si no hay product_id"),
	product_id: str | None = typer.Option(
		None,
		help="Filtrar por columna product_id si existe en el JSONL",
	),
	no_show: bool = typer.Option(False, help="Guardar sin abrir ventana"),
) -> None:
	"""Gráfico True vs Prediction en validación para un modelo."""
	handle_plot_true_vs_predictions(
		model_name=model_name,
		input_path=input_path,
		output_path=output_path,
		target_col=target_col,
		product_id=product_id,
		show=not no_show,
	)


@app.command()
def prepare_data(no_with_plots: bool = typer.Option(False, help="(Deprecado)")) -> None:
	"""[Deprecado] Construye datasets; usar 'build-datasets' y 'analyze' separados."""
	handle_build_datasets(output_dir="reports/eda/data", fmt="jsonl")


if __name__ == "__main__":
	app()
