"""`bench analysis ...` commands: warehouse metadata the report needs, and hand labels."""

import datetime
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from sqlbench.analysis.functions import FUNCTIONS_DIR, fetch_and_write, load_function_names
from sqlbench.analysis.guard_check import check_blocks, guard_blocks, read_checks, write_checks
from sqlbench.analysis.hand_labels import (
    read_labels,
    sample_failures,
    task_key,
    write_label_template,
    write_sheet,
)
from sqlbench.analysis.labels import SchemaNames, auto_labels
from sqlbench.analysis.report import ReportInputs, write_report
from sqlbench.analysis.runs import load_run, run_parquet_path
from sqlbench.analysis.tables import RunData
from sqlbench.harness.run import question_set_sha256
from sqlbench.questions.model import Question, load_question, question_paths
from sqlbench.schema.spec import load_schema
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses import databricks_warehouse, snowflake_warehouse
from sqlbench.warehouses.base import SessionContext, WarehouseName

RESULTS_DIR = Path("results")
QUESTIONS_DIR = Path("questions")
SCHEMAS_DIR = Path("schemas")
REPORTS_DIR = Path("reports")
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


def report_command(
    run: Annotated[str, typer.Option(help="The run id of an exported run.")],
    version: Annotated[str, typer.Option(help="Writes reports/<version>/.")] = "v1.0",
    results_dir: Annotated[Path, typer.Option()] = RESULTS_DIR,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
    labels_file: Annotated[Path, typer.Option()] = LABELS_FILE,
    output_root: Annotated[Path, typer.Option()] = REPORTS_DIR,
) -> None:
    """Write reports/<version>/README.md and charts from an exported run."""
    rows = load_run(run_parquet_path(run, results_dir))
    manifest = json.loads((results_dir / f"run-{run}.manifest.json").read_text())
    questions = load_questions(questions_dir)
    run_ids = sorted({row.question_id for row in rows})
    paths = [path for path in question_paths(questions_dir) if path.stem in run_ids]
    if question_set_sha256(paths) != manifest["question_set_sha256"]:
        typer.echo("question files changed since the run; the report would mislabel", err=True)
        raise typer.Exit(code=1)

    schemas = {}
    for name in sorted({questions[question_id].schema_name for question_id in run_ids}):
        schemas[name] = SchemaNames.from_spec(load_schema(schemas_dir / f"{name}.yaml"))
    functions = {
        WarehouseName.SNOWFLAKE: load_function_names(WarehouseName.SNOWFLAKE),
        WarehouseName.DATABRICKS: load_function_names(WarehouseName.DATABRICKS),
    }
    labels = {}
    for row in rows:
        question = questions[row.question_id]
        labels[task_key(row)] = auto_labels(row, question, schemas[question.schema_name], functions)

    guard_path = guard_checks_path(run, results_dir)
    guard_checks = read_checks(guard_path) if guard_path.exists() else None
    stored = set(labels)
    hand = []
    if labels_file.exists():
        hand = [label for label in read_labels(labels_file) if label.key in stored]

    data = RunData(rows, {question_id: questions[question_id] for question_id in run_ids}, labels)
    path = write_report(ReportInputs(data, manifest, guard_checks, hand), output_root / version)
    typer.echo(f"wrote {path}")
