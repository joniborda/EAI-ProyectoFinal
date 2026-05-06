import json
from pathlib import Path

import typer
from eda.db import test_connection
from eda.data_prep import build_datasets
from eda.analysis import run_analysis
from eda.features import build_features
from eda.train import compare_models
from eda.training_dag import run_training_dag


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
		test_connection()
		typer.secho("Conexión OK", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error de conexión: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_build_datasets(output_dir: str, fmt: str) -> None:
	try:
		result = build_datasets(output_dir=output_dir, fmt=fmt)  # type: ignore[arg-type]
		for ds_name, paths in result.items():
			for kind, path in paths.items():
				typer.secho(f"{ds_name} -> {kind}: {path}", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error construyendo datasets: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)


def handle_analyze(input_dir: str, with_plots: bool, plots_dir: str) -> None:
	try:
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
) -> None:
	try:
		lag_list = [int(part) for part in lags.split(",") if part.strip()]
		result = build_features(
			input_path=input_path,
			output_dir=output_dir,
			lags=lag_list,
			target_col=target_col,
			window_size=window_size,
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