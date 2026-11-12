"""`bench analysis ...` commands: warehouse metadata the report needs, and hand labels."""

import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from sqlbench.analysis.functions import FUNCTIONS_DIR, fetch_and_write
from sqlbench.analysis.guard_check import check_blocks, guard_blocks, write_checks
from sqlbench.analysis.hand_labels import sample_failures, write_label_template, write_sheet
from sqlbench.analysis.runs import load_run, run_parquet_path
from sqlbench.questions.model import Question, load_question, question_paths
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses import databricks_warehouse, snowflake_warehouse
from sqlbench.warehouses.base import SessionContext, WarehouseName

RESULTS_DIR = Path("results")
QUESTIONS_DIR = Path("questions")
LABELS_FILE = RESULTS_DIR / "labels.csv"
HAND_SAMPLE_SIZE = 80

analysis_app = typer.Typer(no_args_is_help=True, help="Inputs for the error analysis.")


def connect_reader(warehouse: WarehouseName, context: SessionContext) -> Any:
    """A plain read-only connection (not an adapter), for metadata and EXPLAIN."""
    if warehouse == WarehouseName.SNOWFLAKE:
        return snowflake_warehouse.connect_from_env(context)
    return databricks_warehouse.connect_from_env(context)


def load_questions(questions_dir: Path) -> dict[str, Question]:
    questions = {}
    for path in question_paths(questions_dir):
        question = load_question(path)
        questions[question.id] = question
    return questions


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


@analysis_app.command("sample-failures")
def sample_failures_command(
    run_id: Annotated[str, typer.Option()],
    size: Annotated[int, typer.Option()] = HAND_SAMPLE_SIZE,
    labels_file: Annotated[Path, typer.Option()] = LABELS_FILE,
    sheet_file: Annotated[Path, typer.Option()] = RESULTS_DIR / "labels-sheet.md",
    results_dir: Annotated[Path, typer.Option()] = RESULTS_DIR,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
) -> None:
    """Pick failures to label by hand; writes a reading sheet and an empty labels.csv."""
    if labels_file.exists():
        typer.echo(f"{labels_file} exists; move it away to draw a new sample", err=True)
        raise typer.Exit(code=1)
    rows = load_run(run_parquet_path(run_id, results_dir))
    sample = sample_failures(rows, size)
    write_sheet(sample, load_questions(questions_dir), sheet_file)
    write_label_template(sample, labels_file)
    typer.echo(f"{len(sample)} failures: read {sheet_file}, fill in {labels_file}")
