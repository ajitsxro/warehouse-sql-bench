"""A simple result comparison for checking gold SQL.

This stands in until the scoring normalizer exists (Part 7), and the gold
check should switch to it then. It sorts rows unless order matters, ignores
column names, and compares numbers within the question's tolerance.
"""

import datetime
import json
import math
from decimal import Decimal
from typing import Any


def normalize_value(value: Any) -> Any:
    """Make equal values from the two warehouses compare equal.

    Snowflake returns NUMBER as Decimal, semi-structured values as JSON text
    and TIMESTAMP_TZ with an offset. Databricks returns Python ints and
    floats, nested values as lists, dicts and (key, value) tuples, and
    timestamps in the session time zone.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        return value.astimezone(datetime.UTC).replace(tzinfo=None)
    if isinstance(value, str) and value.startswith(("{", "[")):
        try:
            return canonical_json(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, (list, dict)):
        return canonical_json(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), default=str)


def plain(value: Any) -> Any:
    """Nested value as JSON-ready lists and dicts. A list of pairs is a map."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[str(key)] = plain(item)
        return result
    if isinstance(value, list):
        is_map = len(value) > 0
        for item in value:
            if not (isinstance(item, tuple) and len(item) == 2):
                is_map = False
        if is_map:
            return plain(dict(value))
        return [plain(item) for item in value]
    if isinstance(value, Decimal):
        return normalize_value(value)
    return value


def sort_key(row: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    """Orders rows of mixed types: NULL first, numbers by value, the rest by text."""
    parts: list[tuple[int, Any]] = []
    for value in row:
        if value is None:
            parts.append((0, ""))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append((1, float(value)))
        else:
            parts.append((2, str(value)))
    return tuple(parts)


def normalize_rows(rows: list[tuple[Any, ...]], order_matters: bool) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        normalized.append(tuple(normalize_value(value) for value in row))
    if not order_matters:
        normalized.sort(key=sort_key)
    return normalized


def same_value(left: Any, right: Any, tolerance: float) -> bool:
    numbers = (int, float)
    if (
        isinstance(left, numbers)
        and isinstance(right, numbers)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return math.isclose(left, right, rel_tol=0, abs_tol=tolerance)
    return bool(left == right)


def first_difference(
    left: list[tuple[Any, ...]], right: list[tuple[Any, ...]], tolerance: float
) -> str | None:
    """Describe where two normalized results differ, or None when they match."""
    if len(left) != len(right):
        return f"{len(left)} rows vs {len(right)} rows"
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if len(left_row) != len(right_row):
            return f"{len(left_row)} columns vs {len(right_row)} columns"
        for left_value, right_value in zip(left_row, right_row, strict=True):
            if not same_value(left_value, right_value, tolerance):
                return f"row {index + 1}: {left_row!r} vs {right_row!r}"
    return None
