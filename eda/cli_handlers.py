import typer
from eda.db import test_connection


def handle_test_db() -> None:
	try:
		test_connection()
		typer.secho("Conexión OK", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error de conexión: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)