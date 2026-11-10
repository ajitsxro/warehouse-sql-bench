"""Properties that must hold for any result, checked with generated data."""

import datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from sqlbench.scoring.matching import MAX_PERMUTED_COLUMNS
from sqlbench.scoring.outcomes import Outcome, score_dataset
from sqlbench.warehouses.base import Column, QueryResult

# Each column holds one type, as in a real result: (type name, value strategy).
COLUMN_KINDS: list[tuple[str, st.SearchStrategy[Any]]] = [
    ("FIXED", st.integers(min_value=-(10**20), max_value=10**20)),
    ("FIXED", st.decimals(places=2, allow_nan=False, allow_infinity=False)),
    ("double", st.floats(allow_nan=True, allow_infinity=True)),
    ("TEXT", st.text(max_size=5)),
    ("boolean", st.booleans()),
    ("date", st.dates()),
    ("TIMESTAMP_NTZ", st.datetimes()),
    ("TIMESTAMP_TZ", st.datetimes(timezones=st.just(datetime.UTC))),
    ("array", st.lists(st.integers(min_value=-5, max_value=5), max_size=3)),
]

settings.register_profile("scoring", deadline=None, max_examples=150)
settings.load_profile("scoring")


@st.composite
def results(draw: st.DrawFn, max_width: int = MAX_PERMUTED_COLUMNS) -> QueryResult:
    width = draw(st.integers(min_value=1, max_value=max_width))
    kinds = draw(st.lists(st.sampled_from(COLUMN_KINDS), min_size=width, max_size=width))
    row_values = st.tuples(*[st.one_of(st.none(), strategy) for _, strategy in kinds])
    rows: list[tuple[Any, ...]] = draw(st.lists(row_values, max_size=8))
    # Duplicate rows on purpose: multiset matching must count them.
    duplicate: bool = draw(st.booleans())
    if rows and duplicate:
        rows.append(rows[0])
    columns = [Column(f"c{index}", type_name) for index, (type_name, _) in enumerate(kinds)]
    return QueryResult(elapsed_ms=1.0, columns=columns, rows=rows)


tolerances = st.sampled_from([0.0, 1e-9, 0.01])


@given(results(), st.booleans(), tolerances)
def test_a_result_always_matches_itself(
    result: QueryResult, order_matters: bool, tolerance: float
) -> None:
    assert score_dataset(result, result, order_matters, tolerance).outcome == Outcome.CORRECT


@given(results(), tolerances, st.randoms(use_true_random=False))
def test_row_order_never_matters_when_order_does_not_matter(
    result: QueryResult, tolerance: float, random: Any
) -> None:
    shuffled = list(result.rows)
    random.shuffle(shuffled)
    predicted = QueryResult(elapsed_ms=1.0, columns=result.columns, rows=shuffled)

    assert score_dataset(result, predicted, False, tolerance).outcome == Outcome.CORRECT


@given(results(), st.booleans(), tolerances, st.data())
def test_column_order_never_matters_up_to_the_limit(
    result: QueryResult, order_matters: bool, tolerance: float, data: st.DataObject
) -> None:
    width = len(result.columns)
    order = data.draw(st.permutations(list(range(width))))
    columns = [result.columns[index] for index in order]
    rows = [tuple(row[index] for index in order) for row in result.rows]
    predicted = QueryResult(elapsed_ms=1.0, columns=columns, rows=rows)

    assert score_dataset(result, predicted, order_matters, tolerance).outcome == Outcome.CORRECT
