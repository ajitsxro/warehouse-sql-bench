"""Automatic failure labels on hand-made rows."""

from dataclasses import replace

import pytest
from question_files import question_data
from test_harness_store import ROW

from sqlbench.analysis.labels import (
    Label,
    SchemaNames,
    Trap,
    auto_labels,
    contains_trap,
    question_traps,
    unknown_functions,
)
from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Question
from sqlbench.warehouses.base import WarehouseName

SNOWFLAKE = WarehouseName.SNOWFLAKE
DATABRICKS = WarehouseName.DATABRICKS
FUNCTIONS = {
    SNOWFLAKE: frozenset({"count", "iff", "object_agg", "flatten", "datediff", "greatest"}),
    DATABRICKS: frozenset({"count", "iff", "flatten", "explode", "datediff", "greatest"}),
}
SCHEMA = SchemaNames(
    tables=frozenset({"orders", "products"}),
    columns=frozenset({"order_id", "placed_at", "tags", "attributes", "discount"}),
    semi_structured=frozenset({"attributes"}),
)
PORTABLE = Question.model_validate(question_data())
SEMANTICS = Question.model_validate(
    question_data(
        category="dialect_sensitive",
        divergence="semantics",
        divergence_note="DATEDIFF counts boundaries on Snowflake (probe datediff_year).",
    )
)


def failed(warehouse: str, sql: str, error: str = "", outcome: str = "syntax_error") -> ResultRow:
    return replace(
        ROW,
        warehouse=warehouse,
        predicted_sql=sql,
        error_message=error or None,
        outcome=outcome,
        outcome_main=outcome,
        outcome_edge=outcome,
        lucky_match=False,
        near_miss=(),
    )


def labels(row: ResultRow, question: Question = PORTABLE) -> tuple[str, ...]:
    return auto_labels(row, question, SCHEMA, FUNCTIONS)


def test_a_correct_row_has_no_labels() -> None:
    row = replace(failed("snowflake", "SELECT 1"), outcome="correct")

    assert labels(row) == ()


def test_snowflake_table_function_on_databricks_is_a_leak() -> None:
    row = failed(
        "databricks",
        "SELECT f.value FROM orders o, LATERAL FLATTEN(input => o.attributes) f",
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'FLATTEN'",
    )

    assert Label.DIALECT_LEAK in labels(row)


def test_a_rejected_function_from_the_other_warehouse_is_a_leak() -> None:
    row = failed(
        "snowflake",
        "SELECT explode(tags) FROM products",
        "002139 (02000): SQL compilation error:\nUnknown function EXPLODE. (request_id=1)",
        outcome="runtime_error",
    )

    assert labels(row) == (Label.DIALECT_LEAK,)


def test_a_function_neither_warehouse_has_is_unknown_not_a_leak() -> None:
    row = failed(
        "databricks",
        "SELECT strftime('%Y', placed_at) FROM orders",
        "[UNRESOLVED_ROUTINE] Cannot resolve routine `strftime` on search path",
    )

    assert labels(row) == (Label.UNKNOWN_FUNCTION,)


def test_unknown_function_names_come_from_both_message_styles() -> None:
    snowflake = "Unknown functions STRFTIME, JULIANDAY. (request_id=x)"
    databricks = "Could not resolve FLATTEN to a table-valued function."

    assert unknown_functions(snowflake) == {"strftime", "julianday"}
    assert unknown_functions(databricks) == {"flatten"}


def test_an_invented_table_is_a_hallucination_even_if_it_is_a_column_name() -> None:
    row = failed(
        "databricks",
        "SELECT tag FROM products, tags",
        "[TABLE_OR_VIEW_NOT_FOUND] The table or view `tags` cannot be found.",
    )

    assert labels(row) == (Label.SCHEMA_HALLUCINATION,)


def test_an_invented_column_is_a_hallucination() -> None:
    row = failed(
        "snowflake",
        "SELECT ship_country FROM orders",
        "000904 (42000): SQL compilation error: invalid identifier 'O.SHIP_COUNTRY'",
    )

    assert labels(row) == (Label.SCHEMA_HALLUCINATION,)


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        # A real column in the wrong scope.
        ("SELECT o.placed_at FROM x", "invalid identifier 'O.PLACED_AT'"),
        # An alias the query defines but uses out of scope.
        ("SELECT 1 FROM UNNEST(tags) AS tag", "with name `tag` cannot be resolved"),
        # FLATTEN's own output column.
        ("SELECT seq FROM orders", "invalid identifier 'SEQ'"),
        # Nested access on a real column.
        ("SELECT attributes.items FROM orders", "with name `attributes`.`items` cannot"),
    ],
)
def test_names_that_exist_somewhere_are_not_hallucinations(sql: str, error: str) -> None:
    assert labels(failed("snowflake", sql, error)) == ()


def test_wrong_answer_containing_the_questions_trap_is_a_semantics_trap() -> None:
    row = failed(
        "snowflake",
        "SELECT COUNT(*) FROM orders WHERE DATEDIFF(year, placed_at, CURRENT_DATE) >= 1",
        outcome="wrong_result",
    )

    assert labels(row, SEMANTICS) == (Label.SEMANTICS_TRAP,)


def test_wrong_answer_without_the_trap_is_not_labelled() -> None:
    row = failed("snowflake", "SELECT COUNT(*) FROM orders", outcome="wrong_result")

    assert labels(row, SEMANTICS) == ()


def test_the_trap_label_needs_a_semantics_question() -> None:
    row = failed("snowflake", "SELECT DATEDIFF(year, placed_at, x) FROM orders", "", "wrong_result")

    assert labels(row, PORTABLE) == ()


def test_lucky_match_and_near_miss_tags_are_kept() -> None:
    row = replace(
        failed("snowflake", "SELECT 1", outcome="wrong_result"),
        lucky_match=True,
        near_miss=("edge:row_subset",),
    )

    assert labels(row) == (Label.LUCKY_MATCH, "edge:row_subset")


def test_traps_come_from_the_probe_names() -> None:
    question = Question.model_validate(
        question_data(
            category="dialect_sensitive",
            divergence="semantics",
            divergence_note="Both (probes datediff_month, datediff_month_full).",
        )
    )

    assert question_traps(question) == {Trap.DATEDIFF_BOUNDARY}


@pytest.mark.parametrize(
    ("trap", "with_trap", "without_trap"),
    [
        (
            Trap.NULL_ORDER,
            "SELECT 1 FROM orders ORDER BY discount DESC",
            "SELECT 1 FROM orders ORDER BY discount DESC NULLS LAST",
        ),
        (Trap.GREATEST_LEAST_NULL, "SELECT GREATEST(a, b) FROM t", "SELECT a FROM t"),
        (Trap.DAY_OF_WEEK, "SELECT DAYOFWEEK(placed_at) FROM orders", "SELECT 1"),
        (
            Trap.ADD_MONTHS_END,
            "SELECT DATEADD(month, 1, placed_at) FROM orders",
            "SELECT DATEADD(day, 30, placed_at) FROM orders",
        ),
        (Trap.COUNT_VARIANT, "SELECT COUNT(attributes:x) FROM orders", "SELECT COUNT(*) FROM t"),
        (Trap.CAST_TO_INT, "SELECT (a / 1000)::INT FROM t", "SELECT ROUND(a / 1000) FROM t"),
    ],
)
def test_trap_patterns(trap: Trap, with_trap: str, without_trap: str) -> None:
    assert contains_trap(with_trap, SNOWFLAKE, trap, SCHEMA)
    assert not contains_trap(without_trap, SNOWFLAKE, trap, SCHEMA)
