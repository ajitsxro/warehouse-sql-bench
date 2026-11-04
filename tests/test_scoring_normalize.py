"""Normalization of single values, one type family at a time."""

import datetime
import math
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from sqlbench.scoring.normalize import (
    NULL,
    Kind,
    Value,
    normalize_rows,
    normalize_value,
    numbers_equal,
    values_equal,
)

UTC = datetime.UTC
NEW_YORK = ZoneInfo("America/New_York")


def same(left: Any, right: Any, tolerance: float = 0.0, types: tuple[str, str] = ("", "")) -> bool:
    left_value = normalize_value(left, types[0] or None)
    right_value = normalize_value(right, types[1] or None)
    return values_equal(left_value, right_value, tolerance)


# ---------------- numbers ----------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (3, Decimal("3")),
        (3, 3.0),
        (Decimal("2.50"), 2.5),
        (Decimal("2.50"), Decimal("2.5")),
        (-0.0, 0.0),
        (-0.0, 0),
        (Decimal("-0.00"), 0),
        (float("nan"), float("nan")),
        (float("inf"), float("inf")),
        (12345678901234567890, Decimal("12345678901234567890")),
    ],
)
def test_numbers_of_any_type_compare_by_value(left: Any, right: Any) -> None:
    assert same(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, 2),
        (Decimal("0.1"), Decimal("0.10000001")),
        (float("nan"), 0.0),
        (float("inf"), float("-inf")),
        (float("inf"), 1e308),
        (12345678901234567890, 12345678901234567891),
    ],
)
def test_different_numbers_are_not_equal_without_tolerance(left: Any, right: Any) -> None:
    assert not same(left, right)


def test_negative_zero_becomes_zero() -> None:
    assert normalize_value(-0.0) == Value(Kind.NUMBER, 0.0)
    assert math.copysign(1.0, normalize_value(-0.0).data) == 1.0


def test_absolute_tolerance_is_inclusive_at_the_boundary() -> None:
    assert numbers_equal(Decimal("1.00"), Decimal("1.01"), 0.01)
    assert not numbers_equal(Decimal("1.00"), Decimal("1.02"), 0.01)


def test_relative_tolerance_scales_with_the_larger_value() -> None:
    # 0.001 absolute is too tight, but 0.001 * 1000.5 = 1.0005 covers the gap of 0.5.
    assert numbers_equal(Decimal("1000.0"), Decimal("1000.5"), 0.001)
    assert not numbers_equal(Decimal("1000.0"), Decimal("1001.1"), 0.001)


def test_float_tolerance_boundary() -> None:
    assert numbers_equal(100.0, 100.01, 0.0001)
    assert not numbers_equal(100.0, 100.0102, 0.0001)


def test_tolerance_compares_an_exact_value_with_a_float() -> None:
    assert numbers_equal(Decimal("3.496967"), 3.496966739855249, 0.000001)
    assert not numbers_equal(Decimal("3.4969"), 3.496966739855249, 0.000001)


def test_zero_tolerance_still_matches_a_decimal_and_its_float() -> None:
    assert numbers_equal(Decimal("0.1"), 0.1, 0.0)


def test_nan_is_not_equal_to_a_number_even_with_tolerance() -> None:
    assert not numbers_equal(float("nan"), 1.0, 1e9)


# ---------------- timestamps and dates ----------------


def test_aware_timestamps_compare_as_utc_instants() -> None:
    utc = datetime.datetime(2024, 3, 10, 7, 0, tzinfo=UTC)
    new_york = datetime.datetime(2024, 3, 10, 3, 0, tzinfo=NEW_YORK)

    assert same(utc, new_york)
    assert normalize_value(new_york) == Value(Kind.TIMESTAMP, datetime.datetime(2024, 3, 10, 7, 0))


def test_a_naive_timestamp_is_read_as_utc_wall_time() -> None:
    naive = datetime.datetime(2024, 3, 10, 7, 0)

    assert same(naive, datetime.datetime(2024, 3, 10, 7, 0, tzinfo=UTC))
    assert not same(naive, datetime.datetime(2024, 3, 10, 7, 0, tzinfo=NEW_YORK))


def test_timestamps_compare_at_microsecond_precision() -> None:
    base = datetime.datetime(2023, 12, 31, 23, 59, 59, 999999)

    assert not same(base, base + datetime.timedelta(microseconds=1))


def test_timestamps_ignore_numeric_tolerance() -> None:
    base = datetime.datetime(2024, 1, 1)

    assert not same(base, base + datetime.timedelta(microseconds=1), tolerance=1.0)


def test_a_date_is_not_the_timestamp_at_its_midnight() -> None:
    assert not same(datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 1))
    assert same(datetime.date(2024, 2, 29), datetime.date(2024, 2, 29))


def test_times_and_intervals() -> None:
    assert same(datetime.time(12, 30), datetime.time(12, 30))
    assert same(datetime.timedelta(days=29), datetime.timedelta(days=29))
    assert not same(datetime.timedelta(days=29), 29)


# ---------------- strings, booleans, binary ----------------


@pytest.mark.parametrize(
    ("left", "right"),
    [("web", "web "), ("web", "Web"), ("web", " web"), ("", " ")],
)
def test_strings_are_exact(left: str, right: str) -> None:
    assert not same(left, right)


def test_booleans_are_not_numbers() -> None:
    assert same(True, True)
    assert not same(True, 1)
    assert not same(False, 0)
    assert not same(True, "true")


def test_binary_values_compare_by_bytes() -> None:
    assert same(b"\x00\x01", bytearray(b"\x00\x01"))
    assert not same(b"\x00", "00")


# ---------------- NULL ----------------


@pytest.mark.parametrize("other", ["", 0, 0.0, False, "null", "[]"])
def test_sql_null_equals_only_sql_null(other: Any) -> None:
    assert same(None, None)
    assert not same(None, other)


def test_sql_null_is_not_json_null() -> None:
    json_null = normalize_value("null", "VARIANT")

    assert json_null == Value(Kind.JSON, "null")
    assert normalize_value(None, "VARIANT") == NULL
    assert not values_equal(NULL, json_null, 0.0)


# ---------------- semi-structured ----------------


def test_snowflake_json_text_matches_databricks_containers() -> None:
    snowflake = '{\n  "b": 1,\n  "a": [\n    1.0,\n    2\n  ]\n}'

    assert same(snowflake, {"a": [1, 2], "b": 1}, types=("OBJECT", "struct"))


def test_databricks_map_pairs_match_a_snowflake_object() -> None:
    assert same(
        '{"size": "L", "color": "red"}', [("color", "red"), ("size", "L")], types=("OBJECT", "map")
    )


def test_map_keys_become_strings() -> None:
    assert normalize_value([(1, "a")], "map") == Value(Kind.JSON, '{"1":"a"}')


def test_arrays_keep_their_order() -> None:
    assert not same("[1, 2]", [2, 1], types=("ARRAY", "array"))
    assert same("[1, 2, 2]", [1, 2, 2], types=("ARRAY", "array"))


def test_json_numbers_are_normalized() -> None:
    assert same('{"x": 2.50}', {"x": Decimal("2.5")}, types=("VARIANT", "struct"))
    assert same("[1.0]", [1], types=("ARRAY", "array"))
    assert not same("[1.5]", [1], types=("ARRAY", "array"))


def test_json_null_inside_a_value_is_kept() -> None:
    assert same('{"gift": null}', {"gift": None}, types=("VARIANT", "struct"))
    assert not same('{"gift": null}', {}, types=("VARIANT", "struct"))


def test_a_json_string_is_not_a_plain_string() -> None:
    # Snowflake shows attributes:channel without a cast as "web", with quotes.
    assert not same('"web"', "web", types=("VARIANT", "TEXT"))


def test_json_looking_text_in_a_string_column_stays_text() -> None:
    assert normalize_value('{"a": 1}', "TEXT") == Value(Kind.STRING, '{"a": 1}')
    assert not same('{"a": 1}', '{"a":1}', types=("TEXT", "TEXT"))


def test_non_json_text_in_a_variant_column_stays_text() -> None:
    assert normalize_value("not json", "VARIANT") == Value(Kind.STRING, "not json")


def test_dates_inside_structs_use_iso_format() -> None:
    value = normalize_value({"day": datetime.date(2024, 2, 29)}, "struct")

    assert value == Value(Kind.JSON, '{"day":"2024-02-29"}')


def test_unknown_python_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="no normalization for object"):
        normalize_value(object())


def test_rows_use_each_column_type() -> None:
    rows = normalize_rows([("[1]", "[1]")], ["ARRAY", "TEXT"])

    assert rows == [(Value(Kind.JSON, "[1]"), Value(Kind.STRING, "[1]"))]
