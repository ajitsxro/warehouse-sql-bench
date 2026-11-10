"""Result serialization and the results store."""

import datetime
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest

from sqlbench.harness.serialize import (
    decode_value,
    encode_value,
    result_from_json,
    result_sha256,
    result_to_json,
)
from sqlbench.harness.store import GoldKey, ResultRow, ResultStore, RunInfo, StoredGold
from sqlbench.warehouses.base import Column, QueryResult


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        12345678901234567890,
        -0.5,
        'text with "quotes" and {braces}',
        Decimal("12.50"),
        Decimal("-0.00"),
        datetime.date(2024, 2, 29),
        datetime.datetime(2023, 12, 31, 23, 59, 59, 999999),
        datetime.datetime(2024, 3, 10, 3, 0, tzinfo=ZoneInfo("America/New_York")),
        datetime.datetime(2024, 3, 10, 7, 0, tzinfo=datetime.UTC),
        datetime.time(12, 30, 1, 5),
        datetime.timedelta(days=29, microseconds=7),
        b"\x00\xff",
        [1, "a", None],
        [("size", "L"), ("color", "red")],
        {"a": [1, 2], "b": {"c": None}},
        {1: "int key"},
    ],
)
def test_values_survive_a_round_trip(value: Any) -> None:
    decoded = decode_value(encode_value(value))

    assert decoded == value
    assert type(decoded) is type(value)


def test_aware_timestamps_keep_their_offset() -> None:
    value = datetime.datetime(2024, 3, 10, 3, 0, tzinfo=ZoneInfo("America/New_York"))

    decoded = decode_value(encode_value(value))

    assert decoded.utcoffset() == datetime.timedelta(hours=-4)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats(value: float) -> None:
    decoded = decode_value(encode_value(value))

    assert isinstance(decoded, float)
    assert repr(decoded) == repr(value)


def test_map_pairs_stay_tuples_inside_lists() -> None:
    decoded = decode_value(encode_value([("k", 1)]))

    assert decoded == [("k", 1)]
    assert isinstance(decoded[0], tuple)


def test_unknown_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="can't store a object"):
        encode_value(object())


RESULT = QueryResult(
    elapsed_ms=5.0,
    columns=[Column("c", "FIXED"), Column("v", "VARIANT")],
    rows=[(Decimal("1.5"), '{"a":1}'), (None, None)],
)


def test_results_round_trip_without_timing() -> None:
    restored = result_from_json(result_to_json(RESULT))

    assert restored.columns == RESULT.columns
    assert restored.rows == RESULT.rows
    assert restored.elapsed_ms == 0.0


def test_hash_ignores_row_order_unless_order_matters() -> None:
    reversed_rows = replace(RESULT, rows=list(reversed(RESULT.rows)))

    assert result_sha256(RESULT, False) == result_sha256(reversed_rows, False)
    assert result_sha256(RESULT, True) != result_sha256(reversed_rows, True)


def test_hash_covers_column_types_and_values() -> None:
    other_type = replace(RESULT, columns=[Column("c", "REAL"), Column("v", "VARIANT")])
    other_value = replace(RESULT, rows=[(Decimal("1.50"), '{"a":1}'), (None, None)])

    assert result_sha256(other_type, False) != result_sha256(RESULT, False)
    assert result_sha256(other_value, False) != result_sha256(RESULT, False)


RUN = RunInfo(
    run_id="run-1",
    git_sha="abc123",
    prompt_sha256="p" * 64,
    question_set_sha256="q" * 64,
    data_manifest_sha256="d" * 64,
    scorer_version=1,
    models=("model-a", "model-b"),
    warehouses=("snowflake",),
)

ROW = ResultRow(
    run_id="run-1",
    git_sha="abc123",
    prompt_sha256="p" * 64,
    question_set_sha256="q" * 64,
    data_manifest_sha256="d" * 64,
    scorer_version=1,
    model_id="model-a",
    model_version="model-a-1",
    served_model="model-a-1 via host",
    warehouse="snowflake",
    question_id="shop_001",
    category="portable",
    divergence="none",
    features=("join", "aggregate"),
    outcome_main="correct",
    outcome_edge="wrong_result",
    outcome="wrong_result",
    lucky_match=True,
    near_miss=("edge:row_subset",),
    predicted_sql="SELECT 1",
    error_category=None,
    error_message=None,
    latency_main_ms=120.5,
    latency_edge_ms=80.0,
    input_tokens=2000,
    output_tokens=50,
    cost_usd=0.0123,
    response_cached=False,
)


def test_runs_round_trip() -> None:
    store = ResultStore(":memory:")
    store.create_run(RUN)

    assert store.get_run("run-1") == RUN
    assert store.get_run("other") is None


def test_results_round_trip_and_report_completed_tasks() -> None:
    store = ResultStore(":memory:")
    store.add_result(ROW)
    store.add_result(replace(ROW, question_id="shop_002", near_miss=(), features=()))

    assert store.results("run-1")[0] == ROW
    assert store.completed("run-1") == {
        ("model-a", "snowflake", "shop_001"),
        ("model-a", "snowflake", "shop_002"),
    }
    assert store.completed("run-2") == set()


def test_a_task_is_stored_once_per_run() -> None:
    store = ResultStore(":memory:")
    store.add_result(ROW)

    with pytest.raises(duckdb.ConstraintException):
        store.add_result(ROW)


def test_gold_is_keyed_by_sql_and_data() -> None:
    store = ResultStore(":memory:")
    key = GoldKey("snowflake", "main", "shop_001", "s" * 64, "d" * 64)
    gold = StoredGold(key, False, RESULT, result_sha256(RESULT, False))
    store.put_gold(gold)

    stored = store.get_gold(key)

    assert stored is not None
    assert stored.result.rows == RESULT.rows
    assert stored.result_sha256 == gold.result_sha256
    assert store.get_gold(replace(key, data_sha256="e" * 64)) is None
    assert store.get_gold(replace(key, gold_sql_sha256="t" * 64)) is None


def test_the_first_gold_result_is_kept(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "nested" / "results.duckdb")
    key = GoldKey("snowflake", "edge", "shop_001", "s" * 64, "d" * 64)
    store.put_gold(StoredGold(key, False, RESULT, "first"))
    store.put_gold(StoredGold(key, False, replace(RESULT, rows=[]), "second"))

    assert [gold.result_sha256 for gold in store.all_gold()] == ["first"]
    store.close()
