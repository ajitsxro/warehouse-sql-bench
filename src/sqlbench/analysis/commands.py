"""`bench analysis ...` commands: warehouse metadata the report needs, and hand labels."""

import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from sqlbench.analysis.functions import FUNCTIONS_DIR, fetch_and_write
from sqlbench.analysis.guard_check import check_blocks, guard_blocks, write_checks
from sqlbench.analysis.runs import load_run, run_parquet_path
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses import databricks_warehouse, snowflake_warehouse
from sqlbench.warehouses.base import SessionContext, WarehouseName

RESULTS_DIR = Path("results")

analysis_app = typer.Typer(no_args_is_help=True, help="Inputs for the error analysis.")


def connect_reader(warehouse: WarehouseName, context: SessionContext) -> Any:
    """A plain read-only connection (not an adapter), for metadata and EXPLAIN."""
    if warehouse == WarehouseName.SNOWFLAKE:
        return snowflake_warehouse.connect_from_env(context)
    return databricks_warehouse.connect_from_env(context)


def guard_checks_path(run_id: str, results_dir: Path) -> Path:
    return results_dir / f"guard-checks-{run_id}.csv"


@analysis_app.command("functions")
def fetch_functions(output_dir: Annotated[Path, typer.Option()] = FUNCTIONS_DIR) -> None:
    """Fetch each warehouse's built-in function names (metadata only, no table scans)."""

    def connect(warehouse: WarehouseName) -> Any:
        return connect_reader(warehouse, SessionContext())

    try:
        counts = fetch_and_write(connect, output_dir, datetime.date.today())
    except MissingEnvError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    for warehouse, count in counts.items():
        typer.echo(f"{warehouse}: {count} functions")


@analysis_app.command("guard-check")
def guard_check(
    run_id: Annotated[str, typer.Option()],
    results_dir: Annotated[Path, typer.Option()] = RESULTS_DIR,
) -> None:
    """EXPLAIN each guard-blocked prediction on its warehouse; writes guard-checks-<run>.csv."""
    blocked = guard_blocks(load_run(run_parquet_path(run_id, results_dir)))
    try:
        checks = check_blocks(blocked, connect_reader)
    except MissingEnvError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    path = guard_checks_path(run_id, results_dir)
    write_checks(checks, path)
    accepted = sum(1 for check in checks if check.accepted)
    typer.echo(f"{len(checks)} guard blocks, {accepted} accepted by the warehouse; wrote {path}")
