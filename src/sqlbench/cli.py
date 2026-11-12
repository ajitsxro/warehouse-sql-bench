"""Command-line entry point for the benchmark."""

import datetime
import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import duckdb
import typer
import yaml
from dotenv import load_dotenv

from sqlbench.analysis.commands import analysis_app, report_command
from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.harness.dry_run import DRY_RUN_MODEL, DryRunWarehouse, GoldAnsweringModel
from sqlbench.harness.executor import Task, data_sha256
from sqlbench.harness.export import ExportError, export_run
from sqlbench.harness.freeze import build_manifest, write_manifest
from sqlbench.harness.run import (
    DEFAULT_WORKERS_PER_WAREHOUSE,
    RunError,
    build_tasks,
    data_manifest_sha256,
    git_sha,
    pending,
    question_set_sha256,
    run_tasks,
    sample_per_schema,
    start_or_resume,
)
from sqlbench.harness.serialize import result_sha256
from sqlbench.harness.store import ResultRow, ResultStore, RunInfo
from sqlbench.load.check import check_dataset
from sqlbench.load.cost import cost_estimate, query_cost_estimate
from sqlbench.load.databricks_loader import DatabricksLoader
from sqlbench.load.dataset import Loader, LoadError, load_dataset, read_manifest, session_context
from sqlbench.load.duckdb_loader import DuckDBLoader
from sqlbench.load.snowflake_loader import SnowflakeLoader
from sqlbench.models.cache import ResponseCache, cache_key
from sqlbench.models.clients import (
    ModelClient,
    PacedClient,
    ProviderError,
    RateLimiter,
    make_client,
)
from sqlbench.models.config import ModelConfig, find_model, load_models
from sqlbench.models.generate import build_prompt, estimate_cost, generate
from sqlbench.models.prompt import load_template
from sqlbench.questions.check import (
    GOLD_TIMEOUT_S,
    GOLD_WAREHOUSES,
    PROFILES,
    GoldTargets,
    check_question,
)
from sqlbench.questions.lint import lint_files
from sqlbench.questions.model import Question, Status, load_question, question_paths
from sqlbench.questions.verify import Choice, ReviewTools, inspect_question, review_drafts
from sqlbench.schema.spec import load_schema
from sqlbench.scoring import SCORER_VERSION
from sqlbench.scoring.outcomes import Outcome, score_dataset
from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import SessionContext, Warehouse, WarehouseName
from sqlbench.warehouses.databricks_warehouse import DatabricksWarehouse
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse
from sqlbench.warehouses.snowflake_warehouse import SnowflakeWarehouse

PING_TIMEOUT_S = 60
CHECK_TIMEOUT_S = 120
SCHEMAS_DIR = Path("schemas")
DATA_DIR = Path("data")
MANIFESTS_DIR = Path("manifests")
QUESTIONS_DIR = Path("questions")
MODELS_FILE = Path("models.yaml")
PROMPT_FILE = Path("prompts/v1.txt")
RESPONSE_CACHE = Path("results/llm_cache.duckdb")
RESULTS_FILE = Path("results/results.duckdb")
DRY_RUN_RESULTS_FILE = Path("results/dry_run.duckdb")
DEFAULT_SEED = 20261105
DUCKDB_FILE_NAME = "sqlbench.duckdb"

app = typer.Typer(name="bench", no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True, help="Check schema specs.")
app.add_typer(schema_app, name="schema")
data_app = typer.Typer(no_args_is_help=True, help="Generate and load benchmark data.")
app.add_typer(data_app, name="data")
questions_app = typer.Typer(no_args_is_help=True, help="Lint, check and verify questions.")
app.add_typer(questions_app, name="questions")
app.add_typer(analysis_app, name="analysis")
app.command("report")(report_command)


@app.callback()
def main() -> None:
    """Warehouse SQL Bench: run and score text-to-SQL on Snowflake and Databricks."""
    load_dotenv()


def open_warehouse(
    name: WarehouseName, context: SessionContext | None = None, data_dir: Path = DATA_DIR
) -> Warehouse:
    """A read-only query adapter. DuckDB with a context reads the loaded local file."""
    if name == WarehouseName.SNOWFLAKE:
        return SnowflakeWarehouse(context)
    if name == WarehouseName.DATABRICKS:
        return DatabricksWarehouse(context)
    if context is None:
        return DuckDBWarehouse()
    connection = duckdb.connect(str(data_dir / DUCKDB_FILE_NAME), read_only=True)
    return DuckDBWarehouse(connection, context)


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


def confirm_cost(
    warehouse: WarehouseName, paths: list[Path], profiles: list[Profile], yes: bool, prompt: str
) -> None:
    """Print a cost estimate for warehouse work and stop unless confirmed."""
    if warehouse == WarehouseName.DUCKDB:
        return
    table_count = len(profiles) * sum(len(load_schema(path).tables) for path in paths)
    typer.echo(f"Estimated cost: {cost_estimate(warehouse, table_count)}")
    if not yes and not typer.confirm(prompt):
        raise typer.Exit(code=1)


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
    confirm_cost(warehouse, paths, profiles, yes, "Load now?")

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


@data_app.command("check")
def check_data(
    warehouse: Annotated[WarehouseName, typer.Option("--warehouse", "-w")],
    profile: Annotated[LoadProfile, typer.Option()] = LoadProfile.BOTH,
    schema: Annotated[str | None, typer.Option(help="Schema to check.")] = None,
    all_schemas: Annotated[bool, typer.Option("--all", help="Check every schema.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    schemas_dir: Annotated[Path, typer.Option("--dir")] = SCHEMAS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
) -> None:
    """Compare loaded row counts and column aggregates with the manifests."""
    paths = chosen_schema_paths(schemas_dir, schema, all_schemas)
    profiles = chosen_profiles(profile)
    confirm_cost(warehouse, paths, profiles, yes, "Run the checks?")

    failed = False
    for path in paths:
        spec = load_schema(path)
        for each_profile in profiles:
            manifest = read_manifest(manifest_dir, spec.name, each_profile)
            adapter = open_warehouse(warehouse, session_context(spec.name, each_profile), data_dir)
            try:
                problems = check_dataset(adapter, spec, manifest, CHECK_TIMEOUT_S)
            except MissingEnvError as error:
                typer.echo(f"{warehouse}: {error}", err=True)
                raise typer.Exit(code=1) from error
            finally:
                adapter.close()
            if problems:
                failed = True
                typer.echo(f"{spec.name} ({each_profile}): {len(problems)} differences", err=True)
                for problem in problems:
                    typer.echo(f"  {problem}", err=True)
            else:
                typer.echo(f"{spec.name} ({each_profile}): matches the manifest")
    if failed:
        raise typer.Exit(code=1)


def lint_and_load(paths: list[Path], schemas_dir: Path) -> tuple[list[tuple[Path, Question]], int]:
    """Questions that pass lint, after printing the problems of those that don't."""
    clean: list[tuple[Path, Question]] = []
    failed_count = 0
    for path, found in lint_files(paths, schemas_dir).items():
        if not found:
            clean.append((path, load_question(path)))
            continue
        failed_count += 1
        typer.echo(f"{path}:", err=True)
        for problem in found:
            typer.echo(f"  {problem}", err=True)
    return clean, failed_count


def confirm_query_cost(question_count: int, yes: bool) -> None:
    """Each question runs its gold twice (main and edge) on each warehouse."""
    for warehouse in GOLD_WAREHOUSES:
        typer.echo(f"Estimated cost: {query_cost_estimate(warehouse, 2 * question_count)}")
    if not yes and not typer.confirm("Run the gold queries?"):
        raise typer.Exit(code=1)


@contextmanager
def gold_targets(schema_name: str) -> Iterator[GoldTargets]:
    targets: GoldTargets = {}
    try:
        for warehouse in GOLD_WAREHOUSES:
            for profile in PROFILES:
                context = session_context(schema_name, profile)
                targets[(warehouse, profile)] = open_warehouse(warehouse, context)
        yield targets
    finally:
        for target in targets.values():
            target.close()


@questions_app.command("lint")
def lint_questions(
    schema: Annotated[str | None, typer.Option(help="One schema's questions; default is all.")] = (
        None
    ),
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
) -> None:
    """Check question files without running anything on a warehouse."""
    paths = question_paths(questions_dir, schema)
    _, failed_count = lint_and_load(paths, schemas_dir)
    typer.echo(f"{len(paths)} questions, {failed_count} with problems")
    if failed_count:
        raise typer.Exit(code=1)


@questions_app.command("check")
def check_questions(
    schema: Annotated[str, typer.Option(help="Schema whose questions to check.")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
) -> None:
    """Run both golds on both warehouses and both datasets; their results must agree.

    Questions that fail lint or are rejected are not run.
    """
    clean, failed_count = lint_and_load(question_paths(questions_dir, schema), schemas_dir)
    to_run = [(path, question) for path, question in clean if question.status != Status.REJECTED]
    if not to_run:
        typer.echo("no questions to run")
        raise typer.Exit(code=1 if failed_count else 0)
    confirm_query_cost(len(to_run), yes)

    with gold_targets(schema) as targets:
        for _, question in to_run:
            try:
                check = check_question(question, targets)
            except MissingEnvError as error:
                typer.echo(str(error), err=True)
                raise typer.Exit(code=1) from error
            if check.passed:
                typer.echo(f"{question.id}: ok")
                continue
            failed_count += 1
            typer.echo(f"{question.id}: failed", err=True)
            for failure in check.failures:
                typer.echo(f"  {failure}", err=True)
    if failed_count:
        raise typer.Exit(code=1)


def git_user_name() -> str:
    completed = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def draft_paths(paths: list[Path]) -> list[Path]:
    """Drafts, plus files that don't load, so they can be fixed in the review."""
    drafts = []
    for path in paths:
        try:
            status = load_question(path).status
        except (OSError, ValueError, yaml.YAMLError):
            status = Status.DRAFT
        if status == Status.DRAFT:
            drafts.append(path)
    return drafts


def prompt_choice(allowed: list[Choice]) -> Choice:
    labels = {
        Choice.VERIFY: "[v]erify",
        Choice.EDIT: "[e]dit",
        Choice.REJECT: "[r]eject",
        Choice.SKIP: "[s]kip",
        Choice.QUIT: "[q]uit",
    }
    text = " ".join(labels[choice] for choice in allowed)
    while True:
        answer = typer.prompt(text).strip().lower()
        for choice in allowed:
            if answer == choice.value:
                return choice
        typer.echo(f"choose one of: {', '.join(choice.value for choice in allowed)}")


def open_in_editor(path: Path) -> None:
    """Open the file in $VISUAL or $EDITOR and wait for the editor to exit."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    subprocess.run([*shlex.split(editor), str(path)], check=False)


@questions_app.command("verify")
def verify_questions(
    schema: Annotated[str, typer.Option(help="Schema whose drafts to review.")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
) -> None:
    """Review draft questions one at a time: verify, edit, reject or skip."""
    verifier = git_user_name()
    if not verifier:
        typer.echo("set git config user.name; verify records it as verified_by", err=True)
        raise typer.Exit(code=1)
    drafts = draft_paths(question_paths(questions_dir, schema))
    if not drafts:
        typer.echo("no draft questions")
        return
    # Each edit runs the golds again, so this is the cost of one pass.
    confirm_query_cost(len(drafts), yes)

    with gold_targets(schema) as targets:
        tools = ReviewTools(
            inspect=lambda path: inspect_question(path, schemas_dir, targets),
            choose=prompt_choice,
            edit=open_in_editor,
            show=typer.echo,
        )
        try:
            counts = review_drafts(drafts, tools, verifier, datetime.date.today())
        except MissingEnvError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
    typer.echo(
        f"verified {counts[Choice.VERIFY]}, rejected {counts[Choice.REJECT]}, "
        f"skipped {counts[Choice.SKIP]}"
    )


@questions_app.command("freeze")
def freeze_questions(
    output: Annotated[Path, typer.Option()] = QUESTIONS_DIR / "MANIFEST.json",
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
    prompt_file: Annotated[Path, typer.Option()] = PROMPT_FILE,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
    check: Annotated[
        bool, typer.Option("--check", help="Compare with the existing file instead of writing.")
    ] = False,
) -> None:
    """Write the verified question ids and the input hashes a v1.0 run must match."""
    clean, failed_count = lint_and_load(question_paths(questions_dir), schemas_dir)
    if failed_count:
        typer.echo(f"{failed_count} questions fail lint; fix them before freezing", err=True)
        raise typer.Exit(code=1)
    verified = [(path, question) for path, question in clean if question.status == Status.VERIFIED]
    try:
        manifest = build_manifest(
            verified, prompt_file, load_template(prompt_file).sha256, manifest_dir, SCORER_VERSION
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    if check:
        frozen = json.loads(output.read_text())
        changed = sorted(
            key for key in manifest.keys() | frozen.keys() if manifest.get(key) != frozen.get(key)
        )
        if changed:
            typer.echo(f"inputs differ from {output}: {', '.join(changed)}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"inputs match {output} ({len(verified)} verified questions)")
        return
    write_manifest(manifest, output)
    typer.echo(f"froze {len(verified)} verified questions into {output}")


@app.command("generate")
def generate_sql(
    model: Annotated[str, typer.Option("--model", help="Model id from models.yaml.")],
    warehouse: Annotated[WarehouseName, typer.Option("--warehouse", "-w")],
    schema: Annotated[str | None, typer.Option(help="One schema's questions.")] = None,
    limit: Annotated[int | None, typer.Option(help="At most this many questions.")] = None,
    drafts: Annotated[
        bool, typer.Option("--drafts", help="Include draft questions (for spot checks).")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    models_file: Annotated[Path, typer.Option()] = MODELS_FILE,
    prompt_file: Annotated[Path, typer.Option()] = PROMPT_FILE,
    cache_file: Annotated[Path, typer.Option()] = RESPONSE_CACHE,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
) -> None:
    """Ask a model for each question's SQL and print what it returned. Runs no SQL."""
    if warehouse == WarehouseName.DUCKDB:
        typer.echo("questions have gold SQL for snowflake and databricks only", err=True)
        raise typer.Exit(code=2)
    config = find_model(load_models(models_file), model)
    template = load_template(prompt_file)
    allowed = {Status.VERIFIED, Status.DRAFT} if drafts else {Status.VERIFIED}
    clean, _ = lint_and_load(question_paths(questions_dir, schema), schemas_dir)
    chosen = [question for _, question in clean if question.status in allowed]
    if limit is not None:
        chosen = chosen[:limit]
    if not chosen:
        typer.echo("no questions to generate for (verified only, unless --drafts)")
        raise typer.Exit(code=1)

    schemas = {}
    prompts = []
    for question in chosen:
        if question.schema_name not in schemas:
            schemas[question.schema_name] = load_schema(
                schemas_dir / f"{question.schema_name}.yaml"
            )
        prompts.append(
            build_prompt(template, schemas[question.schema_name], warehouse, question, data_dir)
        )

    cache = ResponseCache(cache_file)
    try:
        uncached = [prompt for prompt in prompts if cache.get(cache_key(config, prompt)) is None]
        estimate = estimate_cost(config, uncached)
        typer.echo(
            f"{len(chosen)} questions, {len(uncached)} not cached: "
            f"at most ${estimate:.2f} for {config.id}"
        )
        if estimate > 0 and not yes and not typer.confirm("Call the model?"):
            raise typer.Exit(code=1)
        try:
            client = PacedClient(make_client(config), RateLimiter(config.requests_per_minute))
        except MissingEnvError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error

        total_cost = 0.0
        failures = 0
        for question, prompt in zip(chosen, prompts, strict=True):
            try:
                result = generate(question, prompt, warehouse, config, client, cache, template)
            except ProviderError as error:
                failures += 1
                typer.echo(f"{question.id}: provider error: {error}", err=True)
                continue
            total_cost += result.cost_usd
            source = "cached" if result.cached else f"${result.cost_usd:.4f}"
            typer.echo(
                f"{question.id}: {result.completion.input_tokens:,} in / "
                f"{result.completion.output_tokens:,} out tokens, {source}"
            )
            if result.sql is None:
                typer.echo("  no SQL in the response")
            else:
                for line in result.sql.splitlines():
                    typer.echo(f"  {line}")
    finally:
        cache.close()
    typer.echo(f"spent ${total_cost:.4f}; {failures} provider errors")
    if failures:
        raise typer.Exit(code=1)


class QuestionSet(StrEnum):
    VERIFIED = "verified"
    # Verified and draft questions: for dry runs and trying the pipeline.
    DRAFTS = "drafts"


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def chosen_models(models_file: Path, models: str) -> list[ModelConfig]:
    configured = load_models(models_file)
    if models == "all":
        return configured
    return [find_model(configured, model_id) for model_id in parse_list(models)]


def chosen_warehouses(warehouses: str) -> list[WarehouseName]:
    chosen = [WarehouseName(name) for name in parse_list(warehouses)]
    for warehouse in chosen:
        if warehouse not in GOLD_WAREHOUSES:
            typer.echo("questions have gold SQL for snowflake and databricks only", err=True)
            raise typer.Exit(code=2)
    return chosen


def adapter_factory(
    dry_run: bool, data_dir: Path
) -> Callable[[WarehouseName, str, Profile], Warehouse]:
    def open_adapter(warehouse: WarehouseName, schema_name: str, profile: Profile) -> Warehouse:
        context = session_context(schema_name, profile)
        if dry_run:
            local = open_warehouse(WarehouseName.DUCKDB, context, data_dir)
            return DryRunWarehouse(local, warehouse)
        return open_warehouse(warehouse, context, data_dir)

    return open_adapter


def describe_row(row: ResultRow) -> str:
    text = f"{row.model_id}/{row.warehouse}/{row.question_id}: {row.outcome}"
    if row.lucky_match:
        text += " (lucky: main correct, edge not)"
    return text


@app.command("run")
def run_benchmark(
    models: Annotated[str, typer.Option(help="all, or model ids separated by commas.")] = "all",
    warehouses: Annotated[str, typer.Option(help="Comma-separated.")] = "snowflake,databricks",
    questions: Annotated[QuestionSet, typer.Option()] = QuestionSet.VERIFIED,
    limit: Annotated[int | None, typer.Option(help="At most this many questions.")] = None,
    per_schema: Annotated[
        int | None, typer.Option(help="This many questions per schema, picked by id hash.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option(help="Resume this run instead of starting one.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Local DuckDB and a gold-answering fake model.")
    ] = False,
    workers: Annotated[int, typer.Option(help="Worker threads per warehouse.")] = (
        DEFAULT_WORKERS_PER_WAREHOUSE
    ),
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    results_file: Annotated[Path | None, typer.Option()] = None,
    models_file: Annotated[Path, typer.Option()] = MODELS_FILE,
    prompt_file: Annotated[Path, typer.Option()] = PROMPT_FILE,
    cache_file: Annotated[Path, typer.Option()] = RESPONSE_CACHE,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    schemas_dir: Annotated[Path, typer.Option("--schemas-dir")] = SCHEMAS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
) -> None:
    """Generate, execute and score every (model, warehouse, question); resumable by --run-id."""
    chosen_houses = chosen_warehouses(warehouses)
    allowed = (
        {Status.VERIFIED} if questions == QuestionSet.VERIFIED else {Status.VERIFIED, Status.DRAFT}
    )
    clean, _ = lint_and_load(question_paths(questions_dir), schemas_dir)
    picked = [(path, question) for path, question in clean if question.status in allowed]
    if per_schema is not None:
        sampled = {
            question.id for question in sample_per_schema([q for _, q in picked], per_schema)
        }
        picked = [(path, question) for path, question in picked if question.id in sampled]
    if limit is not None:
        picked = picked[:limit]
    if not picked:
        typer.echo(f"no {questions} questions to run")
        raise typer.Exit(code=1)
    asked = [question for _, question in picked]
    schemas = {}
    for question in asked:
        if question.schema_name not in schemas:
            schemas[question.schema_name] = load_schema(
                schemas_dir / f"{question.schema_name}.yaml"
            )

    template = load_template(prompt_file)
    run_models = [DRY_RUN_MODEL] if dry_run else chosen_models(models_file, models)
    tasks = build_tasks(run_models, chosen_houses, asked, schemas, template, data_dir)
    run = RunInfo(
        run_id=run_id or datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ"),
        git_sha=git_sha(Path.cwd()),
        prompt_sha256=template.sha256,
        question_set_sha256=question_set_sha256([path for path, _ in picked]),
        data_manifest_sha256=data_manifest_sha256(manifest_dir, list(schemas)),
        scorer_version=SCORER_VERSION,
        models=tuple(model.id for model in run_models),
        warehouses=tuple(warehouse.value for warehouse in chosen_houses),
    )

    store = ResultStore(results_file or (DRY_RUN_RESULTS_FILE if dry_run else RESULTS_FILE))
    cache = ResponseCache(":memory:" if dry_run else cache_file)
    try:
        try:
            start_or_resume(store, run)
        except RunError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
        todo = pending(store, run.run_id, tasks)
        typer.echo(f"run {run.run_id}: {len(todo)} of {len(tasks)} tasks to do")
        if not dry_run and todo:
            confirm_run_cost(todo, run_models, chosen_houses, cache, yes)

        clients: dict[str, ModelClient] = {}
        if dry_run:
            answers = {
                task.prompt: task.question.gold_sql.for_warehouse(task.warehouse) for task in todo
            }
            clients[DRY_RUN_MODEL.id] = GoldAnsweringModel(answers)
        else:
            try:
                for model in run_models:
                    limiter = RateLimiter(model.requests_per_minute)
                    clients[model.id] = PacedClient(make_client(model), limiter)
            except MissingEnvError as error:
                typer.echo(str(error), err=True)
                raise typer.Exit(code=1) from error

        summary = run_tasks(
            todo,
            run,
            store,
            cache,
            template,
            clients,
            adapter_factory(dry_run, data_dir),
            manifest_dir,
            workers_per_warehouse=workers,
            on_result=lambda row: typer.echo(describe_row(row)),
        )
        outcomes: dict[str, int] = {}
        for row in store.results(run.run_id):
            outcomes[row.outcome] = outcomes.get(row.outcome, 0) + 1
    finally:
        cache.close()
        store.close()

    typer.echo(f"stored {summary.stored} results, spent ${summary.cost_usd:.4f}")
    typer.echo(
        "outcomes so far: "
        + ", ".join(f"{name} {count}" for name, count in sorted(outcomes.items()))
    )
    if summary.failures:
        typer.echo(f"{len(summary.failures)} tasks failed and were not stored:", err=True)
        for failure in summary.failures:
            typer.echo(f"  {failure}", err=True)
        typer.echo(f"resume with: bench run --run-id {run.run_id} (same options)", err=True)
        raise typer.Exit(code=1)


def confirm_run_cost(
    todo: list[Task],
    models: list[ModelConfig],
    warehouses: list[WarehouseName],
    cache: ResponseCache,
    yes: bool,
) -> None:
    """Model calls not yet cached, warehouse queries, and what both cost at most."""
    total = 0.0
    for model in models:
        uncached = []
        for task in todo:
            if task.model.id == model.id and cache.get(cache_key(model, task.prompt)) is None:
                uncached.append(task.prompt)
        estimate = estimate_cost(model, uncached)
        total += estimate
        typer.echo(f"{model.id}: {len(uncached)} model calls, at most ${estimate:.2f}")
    typer.echo(f"models: at most ${total:.2f} in all")
    for warehouse in warehouses:
        # Two predictions per task; gold is at most two more per question, then cached.
        tasks_here = [task for task in todo if task.warehouse == warehouse]
        questions_here = {task.question.id for task in tasks_here}
        queries = 2 * len(tasks_here) + 2 * len(questions_here)
        typer.echo(f"{warehouse}: {query_cost_estimate(warehouse, queries)}")
    if not yes and not typer.confirm("Start the run?"):
        raise typer.Exit(code=1)


@app.command("export")
def export_results(
    run_id: Annotated[str, typer.Option()],
    results_file: Annotated[Path, typer.Option()] = RESULTS_FILE,
    models_file: Annotated[Path, typer.Option()] = MODELS_FILE,
    frozen_file: Annotated[Path, typer.Option()] = QUESTIONS_DIR / "MANIFEST.json",
    output_dir: Annotated[Path, typer.Option()] = RESULTS_FILE.parent,
) -> None:
    """Write results/run-<id>.parquet and its manifest; refuses an incomplete run."""
    store = ResultStore(results_file)
    try:
        exported = export_run(store, run_id, load_models(models_file), frozen_file, output_dir)
    except ExportError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finally:
        store.close()
    typer.echo(
        f"exported {exported.rows} rows (${exported.cost_usd:.2f} spent) to "
        f"{exported.parquet_path} and {exported.manifest_path}"
    )


@app.command("rerun-gold")
def rerun_gold(
    warehouses: Annotated[str, typer.Option(help="Comma-separated.")] = "snowflake,databricks",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Check the dry-run store.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation.")] = False,
    results_file: Annotated[Path | None, typer.Option()] = None,
    questions_dir: Annotated[Path, typer.Option("--dir")] = QUESTIONS_DIR,
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR,
    manifest_dir: Annotated[Path, typer.Option()] = MANIFESTS_DIR,
) -> None:
    """Run every cached gold query again and check its result hash still matches."""
    chosen_houses = {warehouse.value for warehouse in chosen_warehouses(warehouses)}
    by_id = {}
    for path in question_paths(questions_dir):
        loaded = load_question(path)
        by_id[loaded.id] = loaded
    store = ResultStore(results_file or (DRY_RUN_RESULTS_FILE if dry_run else RESULTS_FILE))
    try:
        golds = [gold for gold in store.all_gold() if gold.key.warehouse in chosen_houses]
        if not golds:
            typer.echo("no cached gold results")
            return
        if not dry_run:
            for warehouse in sorted(chosen_houses):
                count = sum(1 for gold in golds if gold.key.warehouse == warehouse)
                typer.echo(f"{warehouse}: {query_cost_estimate(WarehouseName(warehouse), count)}")
            if not yes and not typer.confirm("Run the gold queries again?"):
                raise typer.Exit(code=1)

        open_adapter = adapter_factory(dry_run, data_dir)
        counts = {"match": 0, "within tolerance": 0, "mismatch": 0, "stale": 0}
        for gold in golds:
            key = gold.key
            label = f"{key.question_id} {key.warehouse}/{key.profile}"
            asked = by_id.get(key.question_id)
            warehouse = WarehouseName(key.warehouse)
            profile = Profile(key.profile)
            gold_sql = asked.gold_sql.for_warehouse(warehouse) if asked else None
            if (
                asked is None
                or gold_sql is None
                or hashlib.sha256(gold_sql.encode()).hexdigest() != key.gold_sql_sha256
                or data_sha256(manifest_dir, asked.schema_name, profile) != key.data_sha256
            ):
                counts["stale"] += 1
                typer.echo(f"{label}: stale (gold SQL or data changed since it was cached)")
                continue
            adapter = open_adapter(warehouse, asked.schema_name, profile)
            try:
                result = adapter.execute(gold_sql, GOLD_TIMEOUT_S)
            finally:
                adapter.close()
            if (
                result.error is None
                and result_sha256(result, gold.order_matters) == gold.result_sha256
            ):
                counts["match"] += 1
                continue
            if result.error is None:
                scored = score_dataset(
                    gold.result, result, asked.order_matters, asked.float_tolerance
                )
                if scored.outcome == Outcome.CORRECT:
                    counts["within tolerance"] += 1
                    typer.echo(f"{label}: hash differs, equal within tolerance")
                    continue
            counts["mismatch"] += 1
            detail = result.error.message if result.error else "different result"
            typer.echo(f"{label}: MISMATCH ({detail})", err=True)
    finally:
        store.close()
    typer.echo(", ".join(f"{name} {count}" for name, count in counts.items()))
    if counts["mismatch"]:
        raise typer.Exit(code=1)
