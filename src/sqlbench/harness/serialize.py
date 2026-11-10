"""Store query results as JSON without losing a type.

Gold results are kept in DuckDB and scored again later, so a Decimal must
come back as a Decimal, an aware timestamp with its offset, and a
Databricks map as (key, value) tuples. Plain JSON can't tell those apart,
so every non-JSON value is wrapped in a one-key object naming its type.
"""

import datetime
import hashlib
import json
import math
from decimal import Decimal
from typing import Any

from sqlbench.warehouses.base import Column, QueryResult


def encode_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"float": repr(value)}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    # datetime before date: datetime is a subclass of date.
    if isinstance(value, datetime.datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"date": value.isoformat()}
    if isinstance(value, datetime.time):
        return {"time": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"timedelta_us": value // datetime.timedelta(microseconds=1)}
    if isinstance(value, bytes | bytearray):
        return {"bytes": bytes(value).hex()}
    if isinstance(value, tuple):
        return {"tuple": [encode_value(item) for item in value]}
    if isinstance(value, list):
        return {"list": [encode_value(item) for item in value]}
    if isinstance(value, dict):
        return {"dict": [[encode_value(key), encode_value(item)] for key, item in value.items()]}
    raise TypeError(f"can't store a {type(value).__name__}")


def decode_value(encoded: Any) -> Any:
    if not isinstance(encoded, dict):
        return encoded
    ((tag, payload),) = encoded.items()
    if tag == "float":
        return float(payload)
    if tag == "decimal":
        return Decimal(payload)
    if tag == "datetime":
        return datetime.datetime.fromisoformat(payload)
    if tag == "date":
        return datetime.date.fromisoformat(payload)
    if tag == "time":
        return datetime.time.fromisoformat(payload)
    if tag == "timedelta_us":
        return datetime.timedelta(microseconds=payload)
    if tag == "bytes":
        return bytes.fromhex(payload)
    if tag == "tuple":
        return tuple(decode_value(item) for item in payload)
    if tag == "list":
        return [decode_value(item) for item in payload]
    if tag == "dict":
        return {decode_value(key): decode_value(item) for key, item in payload}
    raise ValueError(f"unknown stored type {tag}")


def result_to_json(result: QueryResult) -> str:
    """Columns and rows of a successful result."""
    columns = [[column.name, column.type] for column in result.columns]
    rows = [[encode_value(value) for value in row] for row in result.rows]
    return json.dumps({"columns": columns, "rows": rows}, separators=(",", ":"))


def result_from_json(text: str, elapsed_ms: float = 0.0) -> QueryResult:
    data = json.loads(text)
    columns = [Column(name, type_name) for name, type_name in data["columns"]]
    rows = [tuple(decode_value(value) for value in row) for row in data["rows"]]
    return QueryResult(elapsed_ms=elapsed_ms, columns=columns, rows=rows)


def result_sha256(result: QueryResult, order_matters: bool) -> str:
    """A fingerprint of the result. Row order counts only when the question says so.

    Without ORDER BY a warehouse may return rows in any order, so rows are
    sorted by their stored form before hashing.
    """
    rows = [json.dumps([encode_value(value) for value in row]) for row in result.rows]
    if not order_matters:
        rows.sort()
    types = [column.type for column in result.columns]
    payload = json.dumps({"types": types, "rows": rows})
    return hashlib.sha256(payload.encode()).hexdigest()
