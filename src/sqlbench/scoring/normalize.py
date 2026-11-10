"""Turn warehouse values into comparable values, one function per type family.

The adapters return plain Python values. The same answer can arrive in
different shapes: Snowflake returns NUMBER as int or Decimal and
VARIANT/OBJECT/ARRAY as JSON text; Databricks returns DOUBLE as float,
arrays as lists, structs as dicts and maps as (key, value) pairs. Each
value becomes a Value with a kind, and values of different kinds are never
equal, except that all numbers are compared numerically.
"""

import datetime
import json
import math
from decimal import Decimal
from enum import StrEnum
from typing import Any, NamedTuple


class Kind(StrEnum):
    NULL = "null"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    INTERVAL = "interval"
    BINARY = "binary"
    # Semi-structured: VARIANT, OBJECT, ARRAY, MAP, STRUCT, as canonical JSON text.
    JSON = "json"


class Value(NamedTuple):
    kind: Kind
    data: Any


NULL = Value(Kind.NULL, None)

# Column types whose string values are JSON text rather than plain strings.
# Snowflake reports VARIANT, OBJECT and ARRAY; Databricks reports variant
# (typed arrays, maps and structs already arrive as Python containers).
SEMI_STRUCTURED_TYPES = frozenset({"variant", "object", "array", "map", "struct"})

Number = int | Decimal | float


def normalize_number(value: Number) -> Value:
    """int and Decimal stay exact; float stays float. -0.0 becomes 0.0."""
    if isinstance(value, float) and value == 0.0:
        return Value(Kind.NUMBER, 0.0)
    if isinstance(value, Decimal) and value.is_zero():
        return Value(Kind.NUMBER, 0)
    return Value(Kind.NUMBER, value)


def normalize_timestamp(value: datetime.datetime) -> Value:
    """An aware timestamp becomes its UTC wall time; a naive one is kept as is.

    Every session runs in UTC, so a naive value (TIMESTAMP_NTZ) and an aware
    value with the same UTC wall time are the same answer.
    """
    if value.tzinfo is not None:
        value = value.astimezone(datetime.UTC).replace(tzinfo=None)
    return Value(Kind.TIMESTAMP, value)


def normalize_string(value: str) -> Value:
    """Exact: no trimming and no case folding."""
    return Value(Kind.STRING, value)


def normalize_semi_structured(value: Any) -> Value:
    """Canonical JSON: sorted keys, no whitespace, integral numbers as integers."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # A semi-structured column holding text that isn't JSON: keep it as text.
            return Value(Kind.STRING, value)
    return Value(Kind.JSON, json.dumps(plain_json(value), sort_keys=True, separators=(",", ":")))


def plain_json(value: Any) -> Any:
    """A nested value as JSON-ready dicts, lists and numbers.

    A list made only of (key, value) pairs is a Databricks map. Map keys
    become strings, as JSON requires.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[str(key)] = plain_json(item)
        return result
    if isinstance(value, list | tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return plain_json(dict(value))
        return [plain_json(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | Decimal | float):
        return json_number(value)
    if isinstance(value, datetime.datetime | datetime.date):
        return normalize_value(value).data.isoformat()
    raise TypeError(f"no JSON form for {type(value).__name__}")


def json_number(value: Number) -> int | float:
    """1, 1.0 and Decimal('1.00') are the same JSON number."""
    if isinstance(value, float) and not math.isfinite(value):
        return value
    if value == int(value):
        return int(value)
    return float(value)


def normalize_value(value: Any, column_type: str | None = None) -> Value:
    """One value, given its column's type name as the warehouse reported it."""
    if value is None:
        return NULL
    semi_structured = column_type is not None and column_type.lower() in SEMI_STRUCTURED_TYPES
    if semi_structured or isinstance(value, dict | list | tuple):
        return normalize_semi_structured(value)
    # bool before numbers: bool is a subclass of int.
    if isinstance(value, bool):
        return Value(Kind.BOOLEAN, value)
    if isinstance(value, int | Decimal | float):
        return normalize_number(value)
    if isinstance(value, str):
        return normalize_string(value)
    # datetime before date: datetime is a subclass of date.
    if isinstance(value, datetime.datetime):
        return normalize_timestamp(value)
    if isinstance(value, datetime.date):
        return Value(Kind.DATE, value)
    if isinstance(value, datetime.time):
        return Value(Kind.TIME, value)
    if isinstance(value, datetime.timedelta):
        return Value(Kind.INTERVAL, value)
    if isinstance(value, bytes | bytearray):
        return Value(Kind.BINARY, bytes(value).hex())
    raise TypeError(f"no normalization for {type(value).__name__}")


def normalize_rows(rows: list[tuple[Any, ...]], column_types: list[str]) -> list[tuple[Value, ...]]:
    normalized = []
    for row in rows:
        values = []
        for value, column_type in zip(row, column_types, strict=True):
            values.append(normalize_value(value, column_type))
        normalized.append(tuple(values))
    return normalized


def numbers_equal(left: Number, right: Number, tolerance: float) -> bool:
    """Equal if |left - right| <= tolerance (absolute) or <= tolerance * max(|left|, |right|).

    Exact numbers (int, Decimal) are compared exactly. If either is a float,
    both are compared as floats. NaN equals NaN; infinities equal only
    themselves.
    """
    if isinstance(left, float) or isinstance(right, float):
        left_float = float(left)
        right_float = float(right)
        if math.isnan(left_float) or math.isnan(right_float):
            return math.isnan(left_float) and math.isnan(right_float)
        if math.isinf(left_float) or math.isinf(right_float):
            return left_float == right_float
        difference = abs(left_float - right_float)
        largest = max(abs(left_float), abs(right_float))
        return difference <= tolerance or difference <= tolerance * largest
    exact_difference = abs(Decimal(left) - Decimal(right))
    if exact_difference == 0:
        return True
    limit = Decimal(repr(tolerance))
    largest_exact = max(abs(Decimal(left)), abs(Decimal(right)))
    return exact_difference <= limit or exact_difference <= limit * largest_exact


def values_equal(left: Value, right: Value, tolerance: float) -> bool:
    if left.kind == Kind.NUMBER and right.kind == Kind.NUMBER:
        return numbers_equal(left.data, right.data, tolerance)
    return left.kind == right.kind and bool(left.data == right.data)
