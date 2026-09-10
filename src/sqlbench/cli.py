"""Command-line entry point for the benchmark."""

from typing import Annotated

import typer
from dotenv import load_dotenv

from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import Warehouse, WarehouseName
from sqlbench.warehouses.databricks_warehouse import DatabricksWarehouse
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse
from sqlbench.warehouses.snowflake_warehouse import SnowflakeWarehouse

PING_TIMEOUT_S = 60

app = typer.Typer(name="bench", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Warehouse SQL Bench: run and score text-to-SQL on Snowflake and Databricks."""
    load_dotenv()


def open_warehouse(name: WarehouseName) -> Warehouse:
    if name == WarehouseName.SNOWFLAKE:
        return SnowflakeWarehouse()
    if name == WarehouseName.DATABRICKS:
        return DatabricksWarehouse()
    return DuckDBWarehouse()


@app.command()
def ping(
    warehouse: Annotated[
        WarehouseName, typer.Option("--warehouse", "-w", help="Warehouse to ping.")
    ],
) -> None:
    """Run SELECT 1 and print how long the query took."""
    adapter = open_warehouse(warehouse)
    try:
        result = adapter.execute("SELECT 1", timeout_s=PING_TIMEOUT_S)
    except MissingEnvError as error:
        typer.echo(f"{warehouse}: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        adapter.close()

    if result.error is not None:
        typer.echo(f"{warehouse}: {result.error.category} error: {result.error.message}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{warehouse}: ok, SELECT 1 took {result.elapsed_ms:.0f} ms")
