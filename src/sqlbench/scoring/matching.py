"""Match a predicted result to gold: which columns line up, and whether the rows agree.

Column names are ignored. With at most MAX_PERMUTED_COLUMNS columns, any
order of the predicted columns may match; with more, columns must be in
gold's order. Rows are compared as a multiset (duplicates count) unless the
question says order matters.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlbench.scoring.normalize import Kind, Value

MAX_PERMUTED_COLUMNS = 6
# Above this many rows sharing the same non-numeric values, the bipartite
# fallback for near-equal numbers is skipped (see multiset_match).
MATCHING_LIMIT = 500

Row = tuple[Value, ...]
# Marks where a number sat in a row's non-numeric key.
NUMBER_SLOT = Value(Kind.NUMBER, None)


@dataclass(frozen=True)
class MatchRules:
    order_matters: bool
    # Decides whether two numbers are equal, e.g. within the question's tolerance.
    numbers_equal: Callable[[Any, Any], bool]


def values_match(left: Value, right: Value, rules: MatchRules) -> bool:
    if left.kind == Kind.NUMBER and right.kind == Kind.NUMBER:
        return rules.numbers_equal(left.data, right.data)
    return left.kind == right.kind and bool(left.data == right.data)


def row_matches(left: Row, right: Row, rules: MatchRules) -> bool:
    for left_value, right_value in zip(left, right, strict=True):
        if not values_match(left_value, right_value, rules):
            return False
    return True


def rows_match(gold: list[Row], predicted: list[Row], rules: MatchRules) -> bool:
    """Same rows, in order when order matters, else as multisets."""
    if len(gold) != len(predicted):
        return False
    if rules.order_matters:
        for gold_row, predicted_row in zip(gold, predicted, strict=True):
            if not row_matches(gold_row, predicted_row, rules):
                return False
        return True
    return multiset_match(gold, predicted, rules)


def multiset_match(gold: list[Row], predicted: list[Row], rules: MatchRules) -> bool:
    """Rows are grouped by their non-numeric values, which must match exactly.

    Within a group only numbers are left, and those may be equal within a
    tolerance, so they can't be hashed. Both sides are sorted and compared
    pairwise. If that fails, a bipartite matching looks for any pairing, as
    long as the group has at most MATCHING_LIMIT rows.
    """
    gold_groups = group_by_key(gold)
    predicted_groups = group_by_key(predicted)
    if gold_groups.keys() != predicted_groups.keys():
        return False
    for key, gold_numbers in gold_groups.items():
        predicted_numbers = predicted_groups[key]
        if len(gold_numbers) != len(predicted_numbers):
            return False
        if not numbers_match(gold_numbers, predicted_numbers, rules):
            return False
    return True


def group_by_key(rows: list[Row]) -> dict[Row, list[tuple[Any, ...]]]:
    groups: dict[Row, list[tuple[Any, ...]]] = {}
    for row in rows:
        key = []
        numbers = []
        for value in row:
            if value.kind == Kind.NUMBER:
                key.append(NUMBER_SLOT)
                numbers.append(value.data)
            else:
                key.append(value)
        groups.setdefault(tuple(key), []).append(tuple(numbers))
    return groups


def number_sort_key(numbers: tuple[Any, ...]) -> tuple[tuple[bool, float], ...]:
    keys = []
    for number in numbers:
        as_float = float(number)
        # NaN sorts last; it would otherwise break the ordering.
        keys.append((as_float != as_float, 0.0 if as_float != as_float else as_float))
    return tuple(keys)


def numbers_match(
    gold: list[tuple[Any, ...]], predicted: list[tuple[Any, ...]], rules: MatchRules
) -> bool:
    if not gold or not gold[0]:
        # No numeric columns: the group key already matched, with equal counts.
        return True
    sorted_gold = sorted(gold, key=number_sort_key)
    sorted_predicted = sorted(predicted, key=number_sort_key)
    if all(
        numbers_pair_match(left, right, rules)
        for left, right in zip(sorted_gold, sorted_predicted, strict=True)
    ):
        return True
    if len(gold) > MATCHING_LIMIT:
        return False
    return has_perfect_matching(sorted_gold, sorted_predicted, rules)


def numbers_pair_match(left: tuple[Any, ...], right: tuple[Any, ...], rules: MatchRules) -> bool:
    for left_number, right_number in zip(left, right, strict=True):
        if not rules.numbers_equal(left_number, right_number):
            return False
    return True


def has_perfect_matching(
    gold: list[tuple[Any, ...]], predicted: list[tuple[Any, ...]], rules: MatchRules
) -> bool:
    """Kuhn's augmenting-path algorithm: can every gold row pair with its own predicted row?"""
    candidates: list[list[int]] = []
    for gold_numbers in gold:
        matching = []
        for index, predicted_numbers in enumerate(predicted):
            if numbers_pair_match(gold_numbers, predicted_numbers, rules):
                matching.append(index)
        if not matching:
            return False
        candidates.append(matching)

    owner_of: list[int | None] = [None] * len(predicted)

    def assign(gold_index: int, visited: set[int]) -> bool:
        for predicted_index in candidates[gold_index]:
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            owner = owner_of[predicted_index]
            if owner is None or assign(owner, visited):
                owner_of[predicted_index] = gold_index
                return True
        return False

    return all(assign(gold_index, set()) for gold_index in range(len(gold)))


def column(rows: list[Row], index: int) -> list[Row]:
    return [(row[index],) for row in rows]


def project(rows: list[Row], order: tuple[int, ...]) -> list[Row]:
    """Rows with their columns taken in the given order."""
    return [tuple(row[index] for index in order) for row in rows]


def find_column_order(
    gold: list[Row],
    predicted: list[Row],
    gold_width: int,
    predicted_width: int,
    rules: MatchRules,
    max_permuted: int = MAX_PERMUTED_COLUMNS,
) -> tuple[int, ...] | None:
    """For each gold column, the predicted column that matches it, or None.

    predicted_width may exceed gold_width (the extra-columns diagnostic picks
    a subset). Above max_permuted gold columns only gold's own order is
    tried. Each gold column's candidates are the predicted columns whose
    values alone match it, which prunes the search before rows are compared.
    """
    if gold_width > max_permuted:
        identity = tuple(range(gold_width))
        if predicted_width == gold_width and rows_match(gold, predicted, rules):
            return identity
        return None

    candidates: list[list[int]] = []
    for gold_index in range(gold_width):
        gold_column = column(gold, gold_index)
        matching = []
        # The same position first, so an already-aligned result is found at once.
        order = [gold_index] + [i for i in range(predicted_width) if i != gold_index]
        for predicted_index in order:
            if predicted_index >= predicted_width:
                continue
            if rows_match(gold_column, column(predicted, predicted_index), rules):
                matching.append(predicted_index)
        if not matching:
            return None
        candidates.append(matching)

    chosen: list[int] = []

    def search(gold_index: int) -> tuple[int, ...] | None:
        if gold_index == gold_width:
            order = tuple(chosen)
            if rows_match(gold, project(predicted, order), rules):
                return order
            return None
        for predicted_index in candidates[gold_index]:
            if predicted_index in chosen:
                continue
            chosen.append(predicted_index)
            found = search(gold_index + 1)
            chosen.pop()
            if found is not None:
                return found
        return None

    return search(0)
