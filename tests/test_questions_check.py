"""Gold checks: the stand-in comparison, the four-run check, and `bench questions check`."""

from pathlib import Path
from typing import Any

import pytest
from question_files import FakeWarehouse, Rows, as_targets, question_data, targets, write_question
from typer.testing import CliRunner

from sqlbench import cli
from sqlbench.datagen.tables import Profile
from sqlbench.questions.check import check_question
from sqlbench.questions.model import Question
from sqlbench.warehouses.base import (
    ErrorCategory,
    QueryError,
    SessionContext,
    Warehouse,
    WarehouseName,
)

FIXTURES = Path(__file__).parent / "fixtures"
SNOWFLAKE = WarehouseName.SNOWFLAKE
DATABRICKS = WarehouseName.DATABRICKS


def question(**changes: Any) -> Question:
    golds = {"snowflake": "SELECT 'sf'", "databricks": "SELECT 'db'"}
    data = question_data(
        category="dialect_sensitive",
        divergence="syntax",
        divergence_note="spelled differently",
        gold_sql=golds,
    )
    data.update(changes)
    return Question.model_validate(data)


def test_matching_golds_pass_and_each_runs_on_its_own_warehouse() -> None:
    fakes = targets([(1,)], [(1,)], [(2,)], [(2,)])

    check = check_question(question(), as_targets(fakes))

    assert check.passed
    assert len(check.runs) == 4
    assert fakes[(SNOWFLAKE, Profile.EDGE)].queries == ["SELECT 'sf'"]
    assert fakes[(DATABRICKS, Profile.MAIN)].queries == ["SELECT 'db'"]


def test_a_difference_on_edge_fails_even_if_main_matches() -> None:
    check = check_question(question(), as_targets(targets([(1,)], [(1,)], [(2,)], [(3,)])))

    assert check.failures == ["edge: snowflake and databricks differ: wrong_result (1 vs 1 rows)"]


def test_main_and_edge_are_not_compared_with_each_other() -> None:
    check = check_question(
        question(), as_targets(targets([(1,)], [(1,)], [(9,), (9,)], [(9,), (9,)]))
    )

    assert check.passed


def test_an_empty_edge_result_fails_when_main_has_rows() -> None:
    check = check_question(question(), as_targets(targets([(1,)], [(1,)], [], [])))

    assert check.failures == [
        "snowflake: edge result is empty but main is not; the question may be ill-posed",
        "databricks: edge result is empty but main is not; the question may be ill-posed",
    ]


def test_empty_on_both_datasets_is_allowed() -> None:
    assert check_question(question(), as_targets(targets([], [], [], []))).passed


def test_query_errors_fail_without_comparing() -> None:
    fakes = targets([(1,)], [(1,)], [(1,)], [(1,)])
    fakes[(DATABRICKS, Profile.MAIN)].error = QueryError(ErrorCategory.SYNTAX, "bad column")

    check = check_question(question(), as_targets(fakes))

    assert check.failures == ["databricks/main: syntax: bad column"]


def test_failures_show_near_miss_labels() -> None:
    fakes = targets([(1,), (2,)], [(1,)], [(1,)], [(1,)])

    check = check_question(question(), as_targets(fakes))

    assert check.failures == [
        "main: snowflake and databricks differ: wrong_result (2 vs 1 rows; row_subset)"
    ]


def test_order_matters_compares_rows_in_order() -> None:
    fakes = targets([(1,), (2,)], [(2,), (1,)], [(1,)], [(1,)])

    assert check_question(question(order_matters=False), as_targets(fakes)).passed
    assert not check_question(question(order_matters=True), as_targets(fakes)).passed


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[FakeWarehouse]:
    """Replaces the real adapters in the CLI; edge returns no rows from Databricks."""
    created: list[FakeWarehouse] = []

    def open_fake(name: WarehouseName, context: SessionContext | None = None) -> Warehouse:
        assert context is not None and context.schema is not None
        rows: Rows = [(1,)]
        if name == DATABRICKS and context.schema.endswith("__edge"):
            rows = []
        fake = FakeWarehouse(name, rows)
        created.append(fake)
        return fake

    monkeypatch.setattr(cli, "open_warehouse", open_fake)
    return created


def run_check(questions_dir: Path, *extra: str, answer: str = "") -> Any:
    arguments = ["questions", "check", "--schema", "mini_shop", "--dir", str(questions_dir)]
    arguments += ["--schemas-dir", str(FIXTURES), *extra]
    return CliRunner().invoke(cli.app, arguments, input=answer)


def test_cli_check_reports_each_question_and_closes_connections(
    tmp_path: Path, opened: list[FakeWarehouse]
) -> None:
    write_question(tmp_path)

    result = run_check(tmp_path, "--yes")

    assert result.exit_code == 1
    assert "Estimated cost: 2 queries" in result.output
    assert "mini_shop_001: failed" in result.output
    assert "edge: snowflake and databricks differ: empty_result (1 vs 0 rows)" in result.output
    assert len(opened) == 4
    assert all(fake.closed for fake in opened)


def test_cli_check_skips_rejected_and_lint_failures(
    tmp_path: Path, opened: list[FakeWarehouse]
) -> None:
    write_question(tmp_path, status="rejected")
    write_question(tmp_path, id="mini_shop_002", gold_sql={"snowflake": "x", "databricks": "y"})

    result = run_check(tmp_path, "--yes")

    assert result.exit_code == 1
    assert "no questions to run" in result.output
    assert "mini_shop_002.yaml" in result.output
    assert opened == []


def test_cli_check_stops_when_the_cost_is_declined(
    tmp_path: Path, opened: list[FakeWarehouse]
) -> None:
    write_question(tmp_path)

    result = run_check(tmp_path, answer="n\n")

    assert result.exit_code == 1
    assert "Run the gold queries?" in result.output
    assert opened == []
