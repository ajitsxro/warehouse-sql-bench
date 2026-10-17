"""Command-line entry point for the benchmark."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv

from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.load.cost import load_cost_estimate
from sqlbench.load.databricks_loader import DatabricksLoader
from sqlbench.load.dataset import Loader, LoadError, load_dataset
from sqlbench.load.duckdb_loader import DuckDBLoader
from sqlbench.load.snowflake_loader import SnowflakeLoader
from sqlbench.schema.spec import load_schema
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import Warehouse, WarehouseName
from sqlbench.warehouses.databricks_warehouse import DatabricksWarehouse
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse
from sqlbench.warehouses.snowflake_warehouse import SnowflakeWarehouse

PING_TIMEOUT_S = 60
SCHEMAS_DIR = Path("schemas")
DATA_DIR = Path("data")
MANIFESTS_DIR = Path("manifests")
DEFAULT_SEED = 20261105
DUCKDB_FILE_NAME = "sqlbench.duckdb"

app = typer.Typer(name="bench", no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True, help="Check schema specs.")
app.add_typer(schema_app, name="schema")
data_app = typer.Typer(no_args_is_help=True, help="Generate and load benchmark data.")
app.add_typer(data_app, name="data")


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


class LoadProfile(StrEnum):
    MAIN = "main"
    EDGE = "edge"
    BOTH = "both"


def schema_paths(schemas_dir: Path, name: str | None) -> list[Path]:
    if name is not None:
        return [schemas_dir / f"{name}.yaml"]
    return sorted(schemas_dir.glob("*.yaml"))


def chosen_schema_paths(schemas_dir: Path, name: str | None, all_schemas: bool) -> list[Path]:
    if (name is None) == (not all_schemas):
        typer.echo("pass exactly one of --schema NAME or --all", err=True)
        raise typer.Exit(code=2)
    return schema_paths(schemas_dir, name)


def chosen_profiles(profile: LoadProfile) -> list[Profile]:
    if profile == LoadProfile.BOTH:
        return [Profile.MAIN, Profile.EDGE]
    return [Profile(profile.value)]


def open_loader(warehouse: WarehouseName, data_dir: Path) -> Loader:
    if warehouse == WarehouseName.DUCKDB:
        return DuckDBLoader(data_dir / DUCKDB_FILE_NAME)
    if warehouse == WarehouseName.SNOWFLAKE:
        return SnowflakeLoader()
    return DatabricksLoader(data_dir)


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


@data_app.command("generate")
def generate_data(
    profile: Annotated[
        Profile, typer.Option(help="main is realistic; edge is small and adversarial.")
    ],
    schema: Annotated[str | None, typer.Option(help="Schema to generate.")] = None,
    all_schemas: Annotated[bool, typer.Option("--all", help="Generate every schema.")] = False,
    seed: Annotated[int, typer.Option(help="Same seed, same files.")] = DEFAULT_SEED,
    schemas_dir: Annotated[Path, typer.Option("--dir")] = SCHEMAS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
) -> None:
    """Write Parquet files under data/ and a manifest under manifests/."""
    for path in chosen_schema_paths(schemas_dir, schema, all_schemas):
        manifest = generate_dataset(path, profile, seed, data_dir, manifest_dir)
        total_rows = sum(table.rows for table in manifest.tables.values())
        typer.echo(f"{manifest.schema_name} ({profile}): {total_rows:,} rows written")


@data_app.command("load")
def load_data(
    warehouse: Annotated[WarehouseName, typer.Option("--warehouse", "-w")],
    profile: Annotated[LoadProfile, typer.Option()] = LoadProfile.BOTH,
    schema: Annotated[str | None, typer.Option(help="Schema to load.")] = None,
    all_schemas: Annotated[bool, typer.Option("--all", help="Load every schema.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    schemas_dir: Annotated[Path, typer.Option("--dir")] = SCHEMAS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
) -> None:
    """Replace the warehouse copy of each dataset with the generated files. Safe to rerun."""
    paths = chosen_schema_paths(schemas_dir, schema, all_schemas)
    profiles = chosen_profiles(profile)
    if warehouse != WarehouseName.DUCKDB:
        table_count = len(profiles) * sum(len(load_schema(path).tables) for path in paths)
        typer.echo(f"Estimated cost: {load_cost_estimate(warehouse, table_count)}")
        if not yes and not typer.confirm("Load now?"):
            raise typer.Exit(code=1)

    try:
        loader = open_loader(warehouse, data_dir)
    except (LoadError, MissingEnvError) as error:
        typer.echo(f"{warehouse}: {error}", err=True)
        raise typer.Exit(code=1) from error
    try:
        for path in paths:
            for each_profile in profiles:
                load_dataset(loader, path, each_profile, data_dir, manifest_dir)
                typer.echo(f"{path.stem} ({each_profile}): loaded into {warehouse}")
    except LoadError as error:
        typer.echo(f"{warehouse}: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        loader.close()
