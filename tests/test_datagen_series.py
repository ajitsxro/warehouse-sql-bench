import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sqlbench.datagen.generator import find_integrity_problems, generate_tables
from sqlbench.datagen.tables import Profile, TableData
from sqlbench.datagen.values import generate_series
from sqlbench.schema.spec import TableSpec, load_schema

FIXTURE = Path(__file__).parent / "fixtures" / "mini_org.yaml"
SCHEMA = load_schema(FIXTURE)
HISTORY = SCHEMA.table("salary_history")
SCANS = SCHEMA.table("badge_scans")


def rows_by_key(table: TableData, key: str) -> dict[Any, list[int]]:
    grouped: dict[Any, list[int]] = {}
    for index, value in enumerate(table.columns[key]):
        grouped.setdefault(value, []).append(index)
    return grouped


def series_table(**changes: Any) -> dict[str, Any]:
    raw = HISTORY.model_dump(exclude_unset=True)
    raw["series"].update(changes)
    return raw


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"key": "salary"}, "series key must be a non-nullable foreign key"),
        ({"time": "valid_to"}, "series time must be a non-nullable"),
        ({"step": "12h"}, "whole days"),
        ({"end": "salary"}, "series end must be nullable"),
        ({"step": "1w"}, "String should match pattern"),
        ({"end": "missing"}, "series column missing is not a column"),
    ],
)
def test_invalid_series_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TableSpec.model_validate(series_table(**changes))


def test_step_minutes() -> None:
    assert HISTORY.series is not None and HISTORY.series.step_minutes == 365 * 24 * 60
    assert SCANS.series is not None and SCANS.series.step_minutes == 60


def test_each_key_gets_one_run_of_increasing_stepped_times() -> None:
    keys = [3, 1, 2, 1, 3, 3, 1, 1]
    generated = generate_series(SCANS, keys, seed=1)

    assert generated["employee_id"] == [1, 1, 1, 1, 2, 3, 3, 3]
    times = generated["scanned_at"]
    hour = datetime.timedelta(hours=1)
    for first, second in [(0, 1), (1, 2), (2, 3), (5, 6), (6, 7)]:
        gap = times[second] - times[first]
        assert gap >= hour and gap % hour == datetime.timedelta(0)
        assert gap <= 6 * hour


def test_gaps_follow_the_gap_rate() -> None:
    keys = [1] * 5000
    times = generate_series(SCANS, keys, seed=1)["scanned_at"]

    steps = [(later - earlier) // datetime.timedelta(hours=1) for earlier, later in pairwise(times)]
    gap_share = sum(step > 1 for step in steps) / len(steps)
    assert 0.08 <= gap_share <= 0.12
    assert max(steps) == 6


def test_last_time_of_each_key_is_within_min_and_max() -> None:
    generated = generate_series(SCANS, list(range(1, 200)) * 3, seed=1)

    last_times = generated["scanned_at"][2::3]
    start = datetime.datetime(2024, 3, 1, tzinfo=datetime.UTC)
    end = datetime.datetime(2024, 3, 31, tzinfo=datetime.UTC)
    assert all(start <= value <= end for value in last_times)


def test_end_column_is_the_next_start_and_null_on_the_current_row() -> None:
    generated = generate_series(HISTORY, [7, 7, 7, 9], seed=1)

    starts = generated["valid_from"]
    assert generated["valid_to"] == [starts[1], starts[2], None, None]
    assert all(isinstance(value, datetime.date) for value in starts)


@pytest.mark.parametrize("profile", list(Profile))
def test_generated_series_stay_contiguous(profile: Profile) -> None:
    tables = generate_tables(SCHEMA, profile, seed=3)
    history = tables["salary_history"]

    assert find_integrity_problems(SCHEMA, tables) == []
    for rows in rows_by_key(history, "employee_id").values():
        starts = [history.columns["valid_from"][index] for index in rows]
        ends = [history.columns["valid_to"][index] for index in rows]
        assert starts == sorted(set(starts))
        assert ends == starts[1:] + [None]


def test_edge_keeps_series_times_on_their_step_grid() -> None:
    tables = generate_tables(SCHEMA, Profile.EDGE, seed=3)
    scans = tables["badge_scans"]

    assert "duplicates" in scans.injectors
    hour = datetime.timedelta(hours=1)
    for rows in rows_by_key(scans, "employee_id").values():
        times = sorted({scans.columns["scanned_at"][index] for index in rows})
        for earlier, later in pairwise(times):
            assert (later - earlier) % hour == datetime.timedelta(0)


def test_edge_drops_whole_series_rather_than_moving_them() -> None:
    tables = generate_tables(SCHEMA, Profile.EDGE, seed=3)
    history = tables["salary_history"]

    assert "empty_groups" in tables["employees"].injectors
    assert history.row_count < 400
    assert len(history.columns["history_id"]) == len(set(history.columns["history_id"]))
