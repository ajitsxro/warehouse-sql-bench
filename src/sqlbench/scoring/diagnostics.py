"""Near-miss labels for wrong answers. They feed error analysis and never change a score.

Each label re-runs the comparison under one relaxed rule. A label means:
had this one thing been different, the prediction would have matched.
"""

import math
from collections import Counter
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlbench.scoring.matching import MatchRules, Row, find_column_order, search_column_order
from sqlbench.scoring.normalize import NULL, Kind, Value, numbers_equal

# The extra-columns search tries subsets of at most this many predicted columns.
MAX_SUPERSET_COLUMNS = 12
# Row subset and superset checks compare numbers at this many significant digits.
SUBSET_SIGNIFICANT_DIGITS = 9
EMPTY_JSON = frozenset({"[]", "{}", "null"})


class Diagnostic(StrEnum):
    # Gold's columns are among the prediction's columns, plus some extra ones.
    EXTRA_COLUMNS_SUPERSET = "extra_columns_superset"
    # Every predicted row is a gold row; some gold rows are missing.
    ROW_SUBSET = "row_subset"
    # Every gold row is predicted, plus extra rows.
    ROW_SUPERSET = "row_superset"
    # Numbers differ only by rounding to the coarser value's decimal places.
    ROUNDING_ONLY = "rounding_only"
    # The right rows in the wrong order, when order matters.
    ORDER_ONLY = "order_only"
    # Matches if SQL NULL and empty values ('' , [], {}, JSON null) are the same.
    NULL_VS_EMPTY = "null_vs_empty"


def diagnose(
    gold: list[Row],
    predicted: list[Row],
    gold_width: int,
    predicted_width: int,
    order_matters: bool,
    tolerance: float,
) -> tuple[Diagnostic, ...]:
    """Labels for a prediction already scored wrong_result or wrong_columns."""

    def within_tolerance(left: Any, right: Any) -> bool:
        return numbers_equal(left, right, tolerance)

    strict = MatchRules(order_matters, within_tolerance)
    found: list[Diagnostic] = []

    if predicted_width > gold_width:
        if predicted_width <= MAX_SUPERSET_COLUMNS and find_column_order(
            gold, predicted, gold_width, predicted_width, strict
        ):
            found.append(Diagnostic.EXTRA_COLUMNS_SUPERSET)
        return tuple(found)
    if predicted_width != gold_width:
        return ()

    width = gold_width
    if 0 < len(predicted) < len(gold) and contains_rows(gold, predicted, width):
        found.append(Diagnostic.ROW_SUBSET)
    if gold and len(predicted) > len(gold) and contains_rows(predicted, gold, width):
        found.append(Diagnostic.ROW_SUPERSET)
    rounding = MatchRules(order_matters, equal_after_rounding)
    if find_column_order(gold, predicted, width, width, rounding) is not None:
        found.append(Diagnostic.ROUNDING_ONLY)
    if order_matters:
        unordered = MatchRules(False, within_tolerance)
        if find_column_order(gold, predicted, width, width, unordered) is not None:
            found.append(Diagnostic.ORDER_ONLY)
    gold_blanked = blank_empties(gold)
    predicted_blanked = blank_empties(predicted)
    if find_column_order(gold_blanked, predicted_blanked, width, width, strict) is not None:
        found.append(Diagnostic.NULL_VS_EMPTY)
    return tuple(found)


def decimal_places(number: Any) -> int:
    """Places after the decimal point as written: 2.50 has 2, 3.5 has 1, 35 has 0."""
    if isinstance(number, float):
        number = Decimal(repr(number))
    if isinstance(number, Decimal):
        exponent = number.as_tuple().exponent
        if isinstance(exponent, int):
            return max(-exponent, 0)
    return 0


def equal_after_rounding(left: Any, right: Any) -> bool:
    """Equal when they differ by at most half a unit in the coarser value's last place.

    3.5 and 3.4969 (1 place) differ by 0.0031 <= 0.05; 35 and 35.4 (0 places)
    differ by 0.4 <= 0.5.
    """
    for number in (left, right):
        if isinstance(number, float) and not math.isfinite(number):
            return numbers_equal(left, right, 0.0)
    places = min(decimal_places(left), decimal_places(right))
    half_unit = Decimal(5) * Decimal(10) ** (-places - 1)
    left_exact = Decimal(repr(left)) if isinstance(left, float) else Decimal(left)
    right_exact = Decimal(repr(right)) if isinstance(right, float) else Decimal(right)
    return abs(left_exact - right_exact) <= half_unit


def blank_empties(rows: list[Row]) -> list[Row]:
    """Rows with '', [], {} and JSON null replaced by SQL NULL."""
    blanked = []
    for row in rows:
        values = []
        for value in row:
            is_empty_string = value.kind == Kind.STRING and value.data == ""
            is_empty_json = value.kind == Kind.JSON and value.data in EMPTY_JSON
            values.append(NULL if is_empty_string or is_empty_json else value)
        blanked.append(tuple(values))
    return blanked


def approximate(value: Value) -> Value:
    """A hashable stand-in: numbers rounded to SUBSET_SIGNIFICANT_DIGITS digits."""
    if value.kind != Kind.NUMBER:
        return value
    as_float = float(value.data)
    if not math.isfinite(as_float):
        return Value(Kind.NUMBER, repr(as_float))
    return Value(Kind.NUMBER, f"{as_float:.{SUBSET_SIGNIFICANT_DIGITS}g}")


def approximate_rows(rows: list[Row], order: tuple[int, ...] | None = None) -> Counter[Row]:
    counted: Counter[Row] = Counter()
    for row in rows:
        picked = row if order is None else tuple(row[index] for index in order)
        counted[tuple(approximate(value) for value in picked)] += 1
    return counted


def counts_contain(larger: Counter[Row], smaller: Counter[Row]) -> bool:
    return all(larger[row] >= count for row, count in smaller.items())


def contains_rows(larger: list[Row], smaller: list[Row], width: int) -> bool:
    """Whether some column order makes every row of smaller appear in larger."""
    larger_counts = approximate_rows(larger)

    def column_ok(smaller_index: int, larger_index: int) -> bool:
        larger_column = approximate_rows(larger, (larger_index,))
        smaller_column = approximate_rows(smaller, (smaller_index,))
        return counts_contain(larger_column, smaller_column)

    def rows_ok(order: tuple[int, ...]) -> bool:
        # order maps each column of smaller to a column of larger; reorder smaller to match.
        inverse = [0] * width
        for smaller_index, larger_index in enumerate(order):
            inverse[larger_index] = smaller_index
        return counts_contain(larger_counts, approximate_rows(smaller, tuple(inverse)))

    return search_column_order(width, width, column_ok, rows_ok) is not None
