import json
from pathlib import Path

import typer


def _parse_hyperparameters_json(value: str | None) -> dict:
	if not value:
		return {}
	stripped = value.strip()
	if not stripped.startswith(("{", "[", "null")):
		candidate_path = Path(value)
		if candidate_path.exists():
			value = candidate_path.read_text(encoding="utf-8")
	parsed = json.loads(value)
	if parsed is None:
		return {}
	if not isinstance(parsed, dict):
		raise ValueError("hyperparameters_json debe ser un objeto JSON.")
	return parsed


def handle_test_db() -> None:
	try:
		from eda.db import test_connection

		test_connection()
		typer.secho("Conexión OK", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error de conexión: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_build_datasets(output_dir: str, fmt: str) -> None:
	try:
		from eda.data_prep import build_datasets

		result = build_datasets(output_dir=output_dir, fmt=fmt)  # type: ignore[arg-type]
		for ds_name, paths in result.items():
			for kind, path in paths.items():
				typer.secho(f"{ds_name} -> {kind}: {path}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error construyendo datasets: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_analyze(input_dir: str, with_plots: bool, plots_dir: str) -> None:
	try:
		from eda.analysis import run_analysis

		run_analysis(input_dir=input_dir, with_plots=with_plots, plots_dir=plots_dir)
	except Exception as e:
		typer.secho(f"Error en análisis: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_build_features(
	input_path: str,
	output_dir: str,
	lags: str,
	target_col: str,
	window_size: int,
	imputation_strategy: str,
) -> None:
	try:
		from eda.features import build_features

		lag_list = [int(part) for part in lags.split(",") if part.strip()]
		result = build_features(
			input_path=input_path,
			output_dir=output_dir,
			lags=lag_list,
			target_col=target_col,
			window_size=window_size,
			imputation_strategy=imputation_strategy,
		)
		for kind, path in result.items():
			typer.secho(f"{kind}: {path}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error construyendo features: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_compare_models(
	input_path: str,
	output_dir: str,
	series_path: str,
	target_col: str,
	val_ratio: float,
	random_state: int,
	hyperparameters_json: str | None = None,
) -> None:
	try:
		from eda.train import compare_models

		hyperparameters = _parse_hyperparameters_json(hyperparameters_json)
		result = compare_models(
			input_path=input_path,
			output_dir=output_dir,
			series_path=series_path,
			target_col=target_col,
			val_ratio=val_ratio,
			random_state=random_state,
			hyperparameters=hyperparameters,
		)
		typer.secho(f"Metrics: {result['metrics']}", fg=typer.colors.GREEN)
		typer.secho(f"MAPE distribution: {result['mape_distribution']}", fg=typer.colors.GREEN)
		typer.secho(f"Metadata: {result['metadata']}", fg=typer.colors.GREEN)
		typer.secho(f"Models dir: {result['models_dir']}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error entrenando modelos: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_training_dag(
	input_path: str,
	output_dir: str,
	series_path: str,
	target_col: str,
	val_ratio: float,
	random_state: int,
	selection_metric: str,
	hyperparameters_json: str | None = None,
) -> None:
	try:
		from eda.training_dag import run_training_dag

		hyperparameters = _parse_hyperparameters_json(hyperparameters_json)
		result = run_training_dag(
			input_path=input_path,
			output_dir=output_dir,
			series_path=series_path,
			target_col=target_col,
			val_ratio=val_ratio,
			random_state=random_state,
			selection_metric=selection_metric,
			hyperparameters=hyperparameters,
		)
		typer.secho(f"Metrics: {result['metrics']}", fg=typer.colors.GREEN)
		typer.secho(f"Metadata: {result['metadata']}", fg=typer.colors.GREEN)
		typer.secho(f"Models dir: {result['models_dir']}", fg=typer.colors.GREEN)
		typer.secho(
			f"Best model: {result['best_model']} ({result['metric_name']}={result['metric_value']:.4f})",
			fg=typer.colors.GREEN,
		)
		typer.secho(f"Best model artifact: {result['best_model_path']}", fg=typer.colors.GREEN)
		typer.secho(f"Best model metadata: {result['best_model_metadata']}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error ejecutando DAG de entrenamiento: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_grid_search(
	model_name: str,
	param_grid_json: str | None,
	input_path: str,
	series_path: str,
	output_dir: str,
	target_col: str,
	val_ratio: float,
	random_state: int,
	selection_metric: str,
) -> None:
	try:
		from eda.grid_search import grid_search_model

		param_grid = _parse_hyperparameters_json(param_grid_json)
		result = grid_search_model(
			model_name=model_name,
			param_grid=param_grid,
			input_path=input_path,
			series_path=series_path,
			output_dir=output_dir,
			target_col=target_col,
			val_ratio=val_ratio,
			random_state=random_state,
			selection_metric=selection_metric,
		)
		typer.secho(f"Grid search results: {result['results']}", fg=typer.colors.GREEN)
		typer.secho(f"Models dir: {result['models_dir']}", fg=typer.colors.GREEN)
		best_run = result.get("best_run")
		if best_run:
			metric_value = best_run["metrics"][selection_metric]
			typer.secho(
				f"Best run: {best_run['run_id']} ({selection_metric}={metric_value:.4f})",
				fg=typer.colors.GREEN,
			)
			typer.secho(f"Best hyperparameters: {best_run['hyperparameters']}", fg=typer.colors.GREEN)
		else:
			typer.secho("No hubo corridas validas para elegir ganador.", fg=typer.colors.YELLOW)
	except Exception as e:
		typer.secho(f"Error ejecutando grid search: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_plot_mape_distribution(
	input_path: str,
	output_path: str,
	model_name: str | None,
	metric_col: str,
	model_col: str | None,
	bins: int,
	show: bool,
) -> None:
	try:
		from eda.metric_plots import plot_mape_distribution

		result = plot_mape_distribution(
			input_path=input_path,
			output_path=output_path,
			model_name=model_name,
			metric_col=metric_col,
			model_col=model_col,
			bins=bins,
			show=show,
		)
		typer.secho(f"Plot: {result}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error generando gráfico de distribución: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def _echo_plot_orders_events_info(info: dict[str, object]) -> None:
	typer.secho(f"Órdenes: {info['orders_source']}", fg=typer.colors.GREEN)
	typer.secho(f"Eventos: {info['events_source']}", fg=typer.colors.GREEN)
	metric = info.get("metric", "orders")
	typer.secho(
		f"Métrica: {metric} | Eje X cada {info['group_months']} mes(es) | "
		f"puntos (días): {info['n_points_plotted']} "
		f"(excl. ultimo dia incompleto: {info['n_days_before_trim']} -> {info['n_days_raw']}) | "
		f"eventos (total / en rango): {info['n_events_total']} / {info['n_events_in_range']}",
		fg=typer.colors.GREEN,
	)
	if info.get("output_path"):
		typer.secho(f"Guardado: {info['output_path']}", fg=typer.colors.GREEN)


def handle_plot_orders_events(
	orders_path: str | None,
	events_path: str | None,
	from_db: bool,
	output_path: str | None,
	show: bool,
	group_months: int = 3,
) -> None:
	try:
		from eda.plot_orders_events import run_plot_orders_events

		gm = group_months if group_months >= 1 else 1
		info = run_plot_orders_events(
			orders_path=orders_path,
			events_path=events_path,
			use_db=from_db,
			output_path=output_path,
			show=show,
			group_months=gm,
		)
		_echo_plot_orders_events_info(info)
	except Exception as e:
		typer.secho(f"Error generando gráfico: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_plot_orders_pct_change(
	orders_path: str | None,
	events_path: str | None,
	from_db: bool,
	output_path: str | None,
	show: bool,
	group_months: int = 3,
) -> None:
	try:
		from eda.plot_orders_events import run_plot_orders_pct_variation_events

		gm = group_months if group_months >= 1 else 1
		info = run_plot_orders_pct_variation_events(
			orders_path=orders_path,
			events_path=events_path,
			use_db=from_db,
			output_path=output_path,
			show=show,
			group_months=gm,
		)
		_echo_plot_orders_events_info(info)
	except Exception as e:
		typer.secho(f"Error generando gráfico: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_plot_orders_lag7_pct_change(
	features_path: str | None,
	events_path: str | None,
	from_db: bool,
	output_path: str | None,
	show: bool,
	group_months: int = 3,
) -> None:
	try:
		from eda.plot_orders_events import run_plot_orders_lag7_pct_variation_events

		gm = group_months if group_months >= 1 else 1
		info = run_plot_orders_lag7_pct_variation_events(
			features_path=features_path,
			events_path=events_path,
			use_db=from_db,
			output_path=output_path,
			show=show,
			group_months=gm,
		)
		_echo_plot_orders_events_info(info)
	except Exception as e:
		typer.secho(f"Error generando gráfico: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_plot_true_vs_predictions(
	model_name: str,
	input_path: str,
	output_path: str | None,
	target_col: str,
	product_id: str | None,
	show: bool,
) -> None:
	try:
		from eda.metric_plots import plot_true_vs_predictions

		result = plot_true_vs_predictions(
			input_path=input_path,
			model_name=model_name,
			output_path=output_path,
			target_col=target_col,
			product_id=product_id,
			show=show,
		)
		typer.secho(f"Plot: {result}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error generando gráfico true vs pred: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)