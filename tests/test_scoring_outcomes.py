"""Outcomes per dataset, and how main and edge combine."""

import datetime
from decimal import Decimal
from typing import Any

import pytest

from sqlbench.scoring import SCORER_VERSION
from sqlbench.scoring.outcomes import DatasetScore, GoldError, Outcome, combine, score_dataset
from sqlbench.warehouses.base import Column, ErrorCategory, QueryError, QueryResult


def result(rows: list[tuple[Any, ...]], types: list[str] | None = None) -> QueryResult:
    if types is None:
        width = len(rows[0]) if rows else 1
        types = ["TEXT"] * width
    columns = [Column(f"c{index}", column_type) for index, column_type in enumerate(types)]
    return QueryResult(elapsed_ms=1.0, columns=columns, rows=rows)


def failed(category: ErrorCategory) -> QueryResult:
    return QueryResult(elapsed_ms=1.0, error=QueryError(category, "boom"))


def score(
    gold: QueryResult,
    predicted: QueryResult | None,
    order_matters: bool = False,
    tolerance: float = 0.0,
) -> Outcome:
    return score_dataset(gold, predicted, order_matters, tolerance).outcome


GOLD = result([("a", 1), ("b", 2)], ["TEXT", "FIXED"])


def test_matching_rows_are_correct() -> None:
    assert score(GOLD, result([("b", 2), ("a", 1)], ["string", "int"])) == Outcome.CORRECT


def test_correct_records_the_column_order_used() -> None:
    scored = score_dataset(GOLD, result([(2, "b"), (1, "a")], ["int", "string"]), False, 0.0)

    assert scored == DatasetScore(Outcome.CORRECT, (1, 0))


def test_no_sql() -> None:
    assert score(GOLD, None) == Outcome.NO_SQL


@pytest.mark.parametrize(
    ("category", "outcome"),
    [
        (ErrorCategory.SYNTAX, Outcome.SYNTAX_ERROR),
        (ErrorCategory.RUNTIME, Outcome.RUNTIME_ERROR),
        (ErrorCategory.TIMEOUT, Outcome.TIMEOUT),
        (ErrorCategory.BLOCKED, Outcome.BLOCKED),
        (ErrorCategory.PERMISSION, Outcome.RUNTIME_ERROR),
        (ErrorCategory.OTHER, Outcome.RUNTIME_ERROR),
    ],
)
def test_errors_map_to_outcomes(category: ErrorCategory, outcome: Outcome) -> None:
    assert score(GOLD, failed(category)) == outcome


def test_failed_gold_raises() -> None:
    with pytest.raises(GoldError, match="gold failed: syntax: boom"):
        score(failed(ErrorCategory.SYNTAX), GOLD)


def test_extra_or_missing_columns_are_wrong_columns() -> None:
    assert score(GOLD, result([("a", 1, "x"), ("b", 2, "y")])) == Outcome.WRONG_COLUMNS
    assert score(GOLD, result([("a",), ("b",)])) == Outcome.WRONG_COLUMNS


def test_wrong_columns_wins_over_an_empty_result() -> None:
    assert score(GOLD, result([], ["TEXT"])) == Outcome.WRONG_COLUMNS


def test_no_rows_where_gold_has_rows_is_an_empty_result() -> None:
    assert score(GOLD, result([], ["TEXT", "FIXED"])) == Outcome.EMPTY_RESULT


def test_a_null_only_row_is_not_an_empty_result() -> None:
    gold_null_row = result([(None,)])

    assert score(gold_null_row, result([], ["TEXT"])) == Outcome.EMPTY_RESULT
    assert score(result([], ["TEXT"]), gold_null_row) == Outcome.WRONG_RESULT
    assert score(gold_null_row, result([(None,)])) == Outcome.CORRECT


def test_empty_gold_and_empty_prediction_are_correct() -> None:
    assert score(result([], ["TEXT"]), result([], ["string"])) == Outcome.CORRECT


def test_different_rows_are_a_wrong_result() -> None:
    assert score(GOLD, result([("a", 1), ("b", 3)], ["TEXT", "FIXED"])) == Outcome.WRONG_RESULT
    assert score(GOLD, result([("a", 1)], ["TEXT", "FIXED"])) == Outcome.WRONG_RESULT


def test_duplicate_rows_must_match_in_number() -> None:
    gold = result([("a",), ("a",)])

    assert score(gold, result([("a",), ("a",)])) == Outcome.CORRECT
    assert score(gold, result([("a",)])) == Outcome.WRONG_RESULT


def test_null_is_not_an_empty_string() -> None:
    assert score(result([(None,)]), result([("",)])) == Outcome.WRONG_RESULT


def test_order_matters_is_respected() -> None:
    swapped = result([("b", 2), ("a", 1)], ["TEXT", "FIXED"])

    assert score(GOLD, swapped, order_matters=True) == Outcome.WRONG_RESULT


def test_tolerance_is_applied() -> None:
    gold = result([(Decimal("3.496967"),)], ["FIXED"])
    predicted = result([(3.496966739855249,)], ["double"])

    assert score(gold, predicted, tolerance=0.000001) == Outcome.CORRECT
    assert score(gold, predicted) == Outcome.WRONG_RESULT


def test_column_types_decide_how_text_is_read() -> None:
    snowflake = result([('{\n  "size": "L"\n}',)], ["OBJECT"])
    databricks = result([([("size", "L")],)], ["map"])

    assert score(snowflake, databricks) == Outcome.CORRECT
    assert score(snowflake, result([('{"size": "L"}',)], ["string"])) == Outcome.WRONG_RESULT


def test_time_zones_are_compared_as_instants() -> None:
    utc = datetime.datetime(2024, 11, 3, 6, 0, tzinfo=datetime.UTC)
    gold = result([(utc,)], ["TIMESTAMP_TZ"])
    as_naive_utc = result([(datetime.datetime(2024, 11, 3, 6, 0),)], ["timestamp_ntz"])
    as_date = result([(datetime.date(2024, 11, 3),)], ["date"])

    assert score(gold, as_naive_utc) == Outcome.CORRECT
    assert score(gold, as_date) == Outcome.WRONG_RESULT


CORRECT = DatasetScore(Outcome.CORRECT, (0,))
WRONG = DatasetScore(Outcome.WRONG_RESULT)
EMPTY = DatasetScore(Outcome.EMPTY_RESULT)
SYNTAX = DatasetScore(Outcome.SYNTAX_ERROR)


@pytest.mark.parametrize(
    ("main", "edge", "outcome", "lucky"),
    [
        (CORRECT, CORRECT, Outcome.CORRECT, False),
        (CORRECT, WRONG, Outcome.WRONG_RESULT, True),
        (CORRECT, EMPTY, Outcome.EMPTY_RESULT, True),
        (WRONG, CORRECT, Outcome.WRONG_RESULT, False),
        (WRONG, EMPTY, Outcome.WRONG_RESULT, False),
        (SYNTAX, SYNTAX, Outcome.SYNTAX_ERROR, False),
    ],
)
def test_combining_main_and_edge(
    main: DatasetScore, edge: DatasetScore, outcome: Outcome, lucky: bool
) -> None:
    combined = combine(main, edge)

    assert combined.outcome == outcome
    assert combined.lucky_match is lucky
    assert combined.main == main
    assert combined.edge == edge


def test_every_score_records_the_scorer_version() -> None:
    assert combine(CORRECT, CORRECT).scorer_version == SCORER_VERSION
