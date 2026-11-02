"""Command-line entry point for the benchmark."""

import typer

app = typer.Typer(name="bench", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Warehouse SQL Bench: run and score text-to-SQL on Snowflake and Databricks."""
