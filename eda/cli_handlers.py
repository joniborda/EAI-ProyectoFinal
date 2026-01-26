import typer
from eda.db import test_connection
from eda.data_prep import build_datasets
from eda.analysis import run_analysis


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