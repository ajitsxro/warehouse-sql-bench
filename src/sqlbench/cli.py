"""Command-line entry point for the benchmark."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv

from sqlbench.schema.spec import load_schema
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import Warehouse, WarehouseName
from sqlbench.warehouses.databricks_warehouse import DatabricksWarehouse
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse
from sqlbench.warehouses.snowflake_warehouse import SnowflakeWarehouse

PING_TIMEOUT_S = 60
SCHEMAS_DIR = Path("schemas")

app = typer.Typer(name="bench", no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True, help="Check schema specs.")
app.add_typer(schema_app, name="schema")


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


def schema_paths(schemas_dir: Path, name: str | None) -> list[Path]:
    if name is not None:
        return [schemas_dir / f"{name}.yaml"]
    return sorted(schemas_dir.glob("*.yaml"))


@schema_app.command("validate")
def validate_schemas(
    schema: Annotated[str | None, typer.Option(help="One schema name; default is all.")] = None,
    schemas_dir: Annotated[Path, typer.Option("--dir", help="Directory of schema YAML.")] = (
        SCHEMAS_DIR
    ),
) -> None:
    """Validate schema YAML files and report every problem found."""
    paths = schema_paths(schemas_dir, schema)
    if not paths:
        typer.echo(f"no schema files in {schemas_dir}", err=True)
        raise typer.Exit(code=1)

    failed = False
    for path in paths:
        try:
            spec = load_schema(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            # pydantic's ValidationError is a ValueError and lists every field problem.
            typer.echo(f"{path.stem}: invalid\n{error}", err=True)
            failed = True
            continue
        main_rows = sum(table.rows for table in spec.tables)
        edge_rows = sum(table.edge_rows for table in spec.tables)
        typer.echo(
            f"{spec.name}: ok ({len(spec.tables)} tables, "
            f"{main_rows:,} main rows, {edge_rows:,} edge rows)"
        )
    if failed:
        raise typer.Exit(code=1)
