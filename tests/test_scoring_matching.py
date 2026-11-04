"""Row and column matching on hand-built result sets."""

from typing import Any

import pytest

from sqlbench.scoring.matching import (
    MAX_PERMUTED_COLUMNS,
    MatchRules,
    Row,
    find_column_order,
    rows_match,
)
from sqlbench.scoring.normalize import normalize_rows, numbers_equal


def rows(*values: tuple[Any, ...]) -> list[Row]:
    width = len(values[0]) if values else 0
    return normalize_rows(list(values), [""] * width)


def rules(order_matters: bool = False, tolerance: float = 0.0) -> MatchRules:
    return MatchRules(order_matters, lambda left, right: numbers_equal(left, right, tolerance))


def test_row_order_is_ignored_by_default() -> None:
    assert rows_match(rows((1, "a"), (2, "b")), rows((2, "b"), (1, "a")), rules())


def test_row_order_counts_when_order_matters() -> None:
    gold = rows((1,), (2,))

    assert rows_match(gold, rows((1,), (2,)), rules(order_matters=True))
    assert not rows_match(gold, rows((2,), (1,)), rules(order_matters=True))


def test_duplicates_count() -> None:
    gold = rows((1,), (1,), (2,))

    assert rows_match(gold, rows((2,), (1,), (1,)), rules())
    assert not rows_match(gold, rows((1,), (2,), (2,)), rules())
    assert not rows_match(gold, rows((1,), (2,)), rules())


def test_null_and_empty_string_rows_differ() -> None:
    assert not rows_match(rows((None,)), rows(("",)), rules())
    assert rows_match(rows((None,), ("",)), rows(("",), (None,)), rules())


def test_empty_results_match_only_empty_results() -> None:
    assert rows_match([], [], rules())
    assert not rows_match([], rows((None,)), rules())
    assert not rows_match(rows((None,)), [], rules())


def test_non_numeric_values_must_match_exactly_within_a_row() -> None:
    assert not rows_match(rows(("a", 1)), rows(("b", 1)), rules(tolerance=1.0))
    assert not rows_match(rows(("a", 1), ("b", 2)), rows(("a", 2), ("b", 1)), rules())


def test_numbers_match_within_tolerance() -> None:
    gold = rows(("a", 1.0), ("b", 2.0))

    assert rows_match(gold, rows(("b", 2.004), ("a", 0.996)), rules(tolerance=0.005))
    assert not rows_match(gold, rows(("b", 2.004), ("a", 0.99)), rules(tolerance=0.005))


def test_tolerance_pairing_that_sorting_misses() -> None:
    # Sorted by the first column, (1.00, 0) would be paired with (1.01, 10).
    gold = rows((1.00, 0.0), (1.03, 10.0))
    predicted = rows((1.04, 0.0), (1.01, 10.0))

    assert rows_match(gold, predicted, rules(tolerance=0.05))
    assert not rows_match(gold, predicted, rules(tolerance=0.01))


def test_nan_rows_match_each_other() -> None:
    assert rows_match(rows((float("nan"),), (1.0,)), rows((1.0,), (float("nan"),)), rules())


def test_columns_may_come_in_any_order() -> None:
    gold = rows((1, "a", None), (2, "b", "x"))
    predicted = rows(("a", None, 1), ("b", "x", 2))

    assert find_column_order(gold, predicted, 3, 3, rules()) == (2, 0, 1)


def test_columns_with_the_same_values_are_told_apart_by_rows() -> None:
    # Columns 0 and 1 hold the same values, so each is a candidate for the other.
    gold = rows((1, 2, "a"), (2, 1, "b"))
    predicted = rows((2, 1, "a"), (1, 2, "b"))

    assert find_column_order(gold, predicted, 3, 3, rules()) == (1, 0, 2)


def test_identical_columns_match_in_either_order() -> None:
    gold = rows((1, 1), (2, 2))

    assert find_column_order(gold, rows((2, 2), (1, 1)), 2, 2, rules()) == (0, 1)


def test_no_order_works_when_rows_differ() -> None:
    gold = rows((1, "a"), (2, "b"))

    assert find_column_order(gold, rows(("b", 1), ("a", 2)), 2, 2, rules()) is None


def test_permutation_with_ordered_rows() -> None:
    gold = rows((1, "a"), (2, "b"))
    predicted = rows(("a", 1), ("b", 2))

    assert find_column_order(gold, predicted, 2, 2, rules(order_matters=True)) == (1, 0)
    swapped_rows = rows(("b", 2), ("a", 1))
    assert find_column_order(gold, swapped_rows, 2, 2, rules(order_matters=True)) is None


def test_above_the_limit_columns_must_be_in_gold_order() -> None:
    width = MAX_PERMUTED_COLUMNS + 1
    gold = rows(tuple(range(width)))
    reversed_row = rows(tuple(reversed(range(width))))

    assert find_column_order(gold, gold, width, width, rules()) == tuple(range(width))
    assert find_column_order(gold, reversed_row, width, width, rules()) is None


def test_at_the_limit_any_order_is_searched() -> None:
    width = MAX_PERMUTED_COLUMNS
    gold = rows(tuple(range(width)))
    reversed_row = rows(tuple(reversed(range(width))))

    assert find_column_order(gold, reversed_row, width, width, rules()) == tuple(
        reversed(range(width))
    )


def test_empty_results_match_with_any_column_order() -> None:
    assert find_column_order([], [], 2, 2, rules()) == (0, 1)


def test_a_wider_prediction_can_contain_gold_columns() -> None:
    gold = rows((1,), (2,))
    predicted = rows(("x", 1), ("y", 2))

    assert find_column_order(gold, predicted, 1, 2, rules()) == (1,)


@pytest.mark.parametrize("tolerance", [0.0, 0.5])
def test_column_candidates_respect_tolerance(tolerance: float) -> None:
    gold = rows((1.0, "a"))
    predicted = rows(("a", 1.4))

    found = find_column_order(gold, predicted, 2, 2, rules(tolerance=tolerance))

    assert found == ((1, 0) if tolerance else None)
