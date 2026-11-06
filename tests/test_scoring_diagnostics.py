"""Near-miss diagnostics: each label, and that labels never change the outcome."""

from decimal import Decimal
from typing import Any

import pytest

from sqlbench.scoring.diagnostics import Diagnostic, decimal_places, equal_after_rounding
from sqlbench.scoring.outcomes import DatasetScore, Outcome, score_dataset
from sqlbench.warehouses.base import Column, ErrorCategory, QueryError, QueryResult


def result(rows: list[tuple[Any, ...]], types: list[str] | None = None) -> QueryResult:
    if types is None:
        types = ["TEXT"] * (len(rows[0]) if rows else 1)
    columns = [Column(f"c{index}", column_type) for index, column_type in enumerate(types)]
    return QueryResult(elapsed_ms=1.0, columns=columns, rows=rows)


def scored(
    gold: QueryResult, predicted: QueryResult, order_matters: bool = False, tolerance: float = 0.0
) -> DatasetScore:
    return score_dataset(gold, predicted, order_matters, tolerance)


GOLD = result([("a", 1), ("b", 2), ("c", 3)], ["TEXT", "FIXED"])


def test_extra_columns_containing_gold() -> None:
    predicted = result([(1, "x", "a"), (2, "y", "b"), (3, "z", "c")], ["FIXED", "TEXT", "TEXT"])

    assert scored(GOLD, predicted) == DatasetScore(
        Outcome.WRONG_COLUMNS, diagnostics=(Diagnostic.EXTRA_COLUMNS_SUPERSET,)
    )


def test_extra_columns_without_gold_inside() -> None:
    predicted = result([(9, "x", "a"), (2, "y", "b"), (3, "z", "c")], ["FIXED", "TEXT", "TEXT"])

    assert scored(GOLD, predicted).diagnostics == ()


def test_missing_columns_get_no_label() -> None:
    assert scored(GOLD, result([("a",), ("b",), ("c",)])) == DatasetScore(Outcome.WRONG_COLUMNS)


def test_row_subset_in_any_column_order() -> None:
    predicted = result([(3, "c"), (1, "a")], ["FIXED", "TEXT"])

    assert scored(GOLD, predicted) == DatasetScore(
        Outcome.WRONG_RESULT, diagnostics=(Diagnostic.ROW_SUBSET,)
    )


def test_row_subset_respects_duplicates() -> None:
    gold = result([("a",), ("b",)])

    assert scored(gold, result([("a",), ("a",)])).diagnostics == ()


def test_row_superset() -> None:
    predicted = result([("a", 1), ("b", 2), ("c", 3), ("c", 3)], ["TEXT", "FIXED"])

    assert scored(GOLD, predicted).diagnostics == (Diagnostic.ROW_SUPERSET,)


def test_no_superset_of_an_empty_gold() -> None:
    assert scored(result([], ["TEXT"]), result([("a",)])).diagnostics == ()


def test_subset_compares_numbers_approximately() -> None:
    gold = result([(1 / 3,), (2.0,)], ["double"])
    predicted = result([(Decimal("0.333333333"),)], ["FIXED"])

    assert scored(gold, predicted).diagnostics == (Diagnostic.ROW_SUBSET,)


def test_rounding_only() -> None:
    gold = result([("a", Decimal("3.4969"))], ["TEXT", "FIXED"])

    assert scored(gold, result([("a", 3.5)], ["TEXT", "double"])).diagnostics == (
        Diagnostic.ROUNDING_ONLY,
    )
    assert scored(gold, result([("a", 3.6)], ["TEXT", "double"])).diagnostics == ()


def test_rounding_to_whole_numbers() -> None:
    gold = result([(Decimal("35.4"),)], ["FIXED"])

    assert scored(gold, result([(35,)], ["FIXED"])).diagnostics == (Diagnostic.ROUNDING_ONLY,)


def test_order_only() -> None:
    predicted = result([("c", 3), ("a", 1), ("b", 2)], ["TEXT", "FIXED"])

    assert scored(GOLD, predicted, order_matters=True) == DatasetScore(
        Outcome.WRONG_RESULT, diagnostics=(Diagnostic.ORDER_ONLY,)
    )


def test_order_only_needs_order_to_matter() -> None:
    predicted = result([("c", 3), ("a", 1), ("b", 9)], ["TEXT", "FIXED"])

    assert Diagnostic.ORDER_ONLY not in scored(GOLD, predicted).diagnostics


@pytest.mark.parametrize(
    ("gold_value", "gold_type", "predicted_value", "predicted_type"),
    [
        (None, "TEXT", "", "string"),
        ("", "TEXT", None, "string"),
        (None, "VARIANT", "null", "VARIANT"),
        ("[]", "ARRAY", None, "array"),
        (None, "OBJECT", [], "map"),
    ],
)
def test_null_vs_empty(
    gold_value: Any, gold_type: str, predicted_value: Any, predicted_type: str
) -> None:
    gold = result([("a", gold_value)], ["TEXT", gold_type])
    predicted = result([("a", predicted_value)], ["TEXT", predicted_type])

    assert scored(gold, predicted) == DatasetScore(
        Outcome.WRONG_RESULT, diagnostics=(Diagnostic.NULL_VS_EMPTY,)
    )


def test_null_is_not_relabelled_as_zero() -> None:
    gold = result([(None,)], ["FIXED"])

    assert scored(gold, result([(0,)], ["FIXED"])).diagnostics == ()


def test_each_label_relaxes_one_rule_only() -> None:
    # Wrong order and rounded: neither order_only nor rounding_only alone explains it.
    gold = result([("a", Decimal("1.04")), ("b", Decimal("2.0"))], ["TEXT", "FIXED"])
    predicted = result([("b", Decimal("2.0")), ("a", Decimal("1.0"))], ["TEXT", "FIXED"])

    assert scored(gold, predicted, order_matters=True).diagnostics == ()


def test_correct_results_and_errors_carry_no_labels() -> None:
    assert scored(GOLD, GOLD).diagnostics == ()
    failed = QueryResult(elapsed_ms=1.0, error=QueryError(ErrorCategory.SYNTAX, "x"))
    assert scored(GOLD, failed).diagnostics == ()
    assert scored(GOLD, result([], ["TEXT", "FIXED"])).diagnostics == ()


@pytest.mark.parametrize(
    ("number", "places"),
    [(35, 0), (Decimal("2.50"), 2), (3.5, 1), (1e-05, 5), (Decimal("1E+2"), 0), (2.0, 1)],
)
def test_decimal_places_as_written(number: Any, places: int) -> None:
    assert decimal_places(number) == places


def test_rounding_boundary_is_half_a_unit() -> None:
    assert equal_after_rounding(Decimal("2.5"), 2)
    assert not equal_after_rounding(Decimal("2.51"), 2)
    assert not equal_after_rounding(float("nan"), 1.0)
    assert equal_after_rounding(float("nan"), float("nan"))
