"""Question model rules and `bench questions lint`."""

import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from question_files import both_golds, question_data, write_question
from typer.testing import CliRunner

from sqlbench.cli import app
from sqlbench.questions.lint import lint_files
from sqlbench.questions.model import Question, Status

FIXTURES = Path(__file__).parent / "fixtures"


def lint_one(tmp_path: Path, **changes: Any) -> list[str]:
    path = write_question(tmp_path, **changes)
    return lint_files([path], FIXTURES)[path]


def test_a_valid_portable_question_has_no_problems(tmp_path: Path) -> None:
    assert lint_one(tmp_path) == []


def test_new_questions_start_as_drafts() -> None:
    assert Question.model_validate(question_data()).status == Status.DRAFT


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"divergence": "syntax", "divergence_note": "x"}, "portable question has divergence"),
        ({"category": "dialect_sensitive"}, "needs divergence syntax or semantics"),
        (
            {"category": "dialect_sensitive", "divergence": "semantics"},
            "divergence_note is required",
        ),
        ({"status": "verified", "verified_by": "me"}, "needs verified_by and verified_at"),
        ({"features": ["telepathy"]}, "features.0"),
        ({"features": []}, "features"),
        ({"float_tolerance": -1}, "float_tolerance"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_model_rules(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Question.model_validate(question_data(**changes))


def test_a_verified_question_with_a_stamp_is_valid() -> None:
    data = question_data(status="verified", verified_by="Ajit", verified_at="2026-11-10")

    assert Question.model_validate(data).verified_at == datetime.date(2026, 11, 10)


def test_invalid_files_are_reported_not_raised(tmp_path: Path) -> None:
    problems = lint_one(tmp_path, category="dialect_sensitive")

    assert len(problems) == 1
    assert problems[0].startswith("invalid:")


def test_file_name_and_directory_must_match_the_question(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    moved = tmp_path / "other" / "renamed.yaml"
    moved.parent.mkdir()
    path.rename(moved)

    problems = lint_files([moved], FIXTURES)[moved]

    assert "file name must be mini_shop_001.yaml" in problems
    assert "file must be in questions/mini_shop/" in problems


def test_ids_must_be_unique_across_schemas(tmp_path: Path) -> None:
    first = write_question(tmp_path)
    second = tmp_path / "mini_org" / "mini_shop_001.yaml"
    second.parent.mkdir()
    second.write_text(first.read_text())

    problems = lint_files([first, second], FIXTURES)

    assert problems[first] == []
    assert f"id mini_shop_001 is also used by {first}" in problems[second]


def test_a_hyphen_split_by_yaml_folding_is_reported(tmp_path: Path) -> None:
    folded = "Label months as 'YYYY- MM', for example '2024-03'."

    assert lint_one(tmp_path, question=folded) == [
        "question text has 'Y- M': a hyphen split across lines"
    ]
    assert lint_one(tmp_path, question="A comma-separated list - in order.") == []


def test_unknown_schema(tmp_path: Path) -> None:
    assert lint_one(tmp_path, schema="nowhere") == ["unknown schema nowhere"]


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT * FROM shoppers", "unknown table shoppers"),
        ("SELECT * FROM mini_shop.customers", "use the bare table name"),
        ("SELECT nickname FROM customers", "column check failed"),
        ("DELETE FROM customers", "read-only guard"),
        ("SELECT 1; SELECT 2", "expected one statement"),
    ],
)
def test_gold_sql_problems_are_reported_per_dialect(tmp_path: Path, sql: str, message: str) -> None:
    problems = lint_one(tmp_path, gold_sql=both_golds(sql))

    assert len(problems) == 2
    assert problems[0].startswith("snowflake gold: ")
    assert problems[1].startswith("databricks gold: ")
    assert message.lower() in problems[0].lower()


def test_dialect_features_resolve_columns_in_each_dialect(tmp_path: Path) -> None:
    golds = {
        "snowflake": (
            "SELECT f.value::STRING AS tag, COUNT(*) AS customers "
            "FROM customers c, LATERAL FLATTEN(input => c.tags) f GROUP BY 1"
        ),
        "databricks": (
            "SELECT tag, COUNT(*) AS customers "
            "FROM customers LATERAL VIEW explode(tags) t AS tag GROUP BY tag"
        ),
    }
    changes = {
        "category": "dialect_sensitive",
        "divergence": "syntax",
        "divergence_note": "FLATTEN vs explode",
        "gold_sql": golds,
    }

    assert lint_one(tmp_path, **changes) == []


def test_ctes_count_as_tables(tmp_path: Path) -> None:
    sql = "WITH active AS (SELECT * FROM customers WHERE is_active) SELECT COUNT(*) FROM active"

    assert lint_one(tmp_path, gold_sql=both_golds(sql)) == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT customer_id FROM orders ORDER BY placed_at DESC, order_id LIMIT 5",
        "SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id "
        "ORDER BY n DESC, customer_id LIMIT 5",
        "SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 5",
        "SELECT customer_id AS who FROM customers ORDER BY who LIMIT 5",
    ],
)
def test_limit_with_an_order_ending_in_a_unique_key_passes(tmp_path: Path, sql: str) -> None:
    assert lint_one(tmp_path, gold_sql=both_golds(sql)) == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT customer_id FROM orders LIMIT 5",
        "SELECT customer_id FROM orders ORDER BY placed_at DESC LIMIT 5",
        "SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id ORDER BY n DESC "
        "LIMIT 5",
        "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment, is_active "
        "ORDER BY n, segment LIMIT 3",
        "SELECT * FROM (SELECT customer_id FROM orders ORDER BY total DESC LIMIT 3) AS top",
    ],
)
def test_limit_that_can_cut_through_ties_fails(tmp_path: Path, sql: str) -> None:
    problems = lint_one(tmp_path, gold_sql=both_golds(sql))

    assert len(problems) == 2
    assert "use RANK() in QUALIFY" in problems[0]


def test_order_matters_needs_an_outer_order_by(tmp_path: Path) -> None:
    ordered = "SELECT segment FROM customers GROUP BY segment ORDER BY segment"
    golds = {"snowflake": ordered, "databricks": "SELECT DISTINCT segment FROM customers"}

    problems = lint_one(tmp_path, order_matters=True, gold_sql=golds)

    assert problems == [
        "databricks gold: order_matters is true but the query has no outer ORDER BY"
    ]


def test_cli_lint_lists_problems_and_fails(tmp_path: Path) -> None:
    write_question(tmp_path)
    write_question(tmp_path, id="mini_shop_002", gold_sql=both_golds("SELECT * FROM shoppers"))

    result = CliRunner().invoke(
        app, ["questions", "lint", "--dir", str(tmp_path), "--schemas-dir", str(FIXTURES)]
    )

    assert result.exit_code == 1
    assert "mini_shop_002.yaml" in result.output
    assert "unknown table shoppers" in result.output
    assert "2 questions, 1 with problems" in result.output


def test_cli_lint_passes_on_clean_questions(tmp_path: Path) -> None:
    write_question(tmp_path)

    result = CliRunner().invoke(
        app,
        ["questions", "lint", "--schema", "mini_shop", "--dir", str(tmp_path)]
        + ["--schemas-dir", str(FIXTURES)],
    )

    assert result.exit_code == 0
    assert "1 questions, 0 with problems" in result.output
