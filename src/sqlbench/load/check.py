"""Compare loaded tables with the manifest: row counts and per-column aggregates.

The check runs through the read-only query adapter with the dataset's
session context, so it also proves the benchmark principal can read every
table by its unqualified name.
"""

import datetime
import math
from decimal import Decimal
from typing import Any

from sqlbench.datagen.generator import ColumnStats, Manifest
from sqlbench.schema.spec import SchemaSpec, TableSpec
from sqlbench.schema.types import TypeKind
from sqlbench.warehouses.base import Warehouse

# Doubles are summed in a different order by each engine.
DOUBLE_RELATIVE_TOLERANCE = 1e-9


def stats_query(table: TableSpec, expected: dict[str, ColumnStats]) -> tuple[str, list[str]]:
    """One SELECT for the whole table, and a label for each output column."""
    selections = ["COUNT(*)"]
    labels = ["rows"]
    for column in table.columns:
        name = column.name
        stats = expected[name]
        # IS NULL rather than COUNT(col): Snowflake's COUNT skips JSON nulls in
        # a VARIANT, but only SQL NULLs are counted in the manifest.
        selections.append(f"SUM(CASE WHEN {name} IS NULL THEN 1 ELSE 0 END)")
        labels.append(f"{name}.nulls")
        if stats.total is not None:
            selections.append(f"SUM({name})")
            labels.append(f"{name}.total")
        if stats.distinct is not None:
            selections.append(f"COUNT(DISTINCT {name})")
            labels.append(f"{name}.distinct")
        if stats.min is not None:
            selections.append(f"MIN({name})")
            selections.append(f"MAX({name})")
            labels.extend([f"{name}.min", f"{name}.max"])
    return f"SELECT {', '.join(selections)} FROM {table.name}", labels


def expected_values(table: TableSpec, rows: int, expected: dict[str, ColumnStats]) -> list[Any]:
    values: list[Any] = [rows]
    for column in table.columns:
        stats = expected[column.name]
        values.append(stats.nulls)
        for optional in (stats.total, stats.distinct):
            if optional is not None:
                values.append(optional)
        if stats.min is not None:
            values.extend([stats.min, stats.max])
    return values


def same_value(kind: TypeKind | None, expected: Any, actual: Any) -> bool:
    """Compare a manifest value with what the warehouse returned for it."""
    if actual is None:
        return False
    if kind == TypeKind.DOUBLE:
        return math.isclose(float(actual), float(expected), rel_tol=DOUBLE_RELATIVE_TOLERANCE)
    if isinstance(actual, datetime.datetime):
        timestamp: datetime.datetime = actual
        if kind == TypeKind.TIMESTAMP_TZ:
            # Sessions run in UTC, so a value without a zone is a UTC instant.
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=datetime.UTC)
        else:
            timestamp = timestamp.replace(tzinfo=None)
        return timestamp == datetime.datetime.fromisoformat(expected)
    if isinstance(actual, datetime.date):
        return actual == datetime.date.fromisoformat(expected)
    return Decimal(str(actual)) == Decimal(str(expected))


def check_dataset(
    warehouse: Warehouse, schema: SchemaSpec, manifest: Manifest, timeout_s: int
) -> list[str]:
    """Differences between the loaded tables and the manifest; empty when they match."""
    problems: list[str] = []
    for table in schema.tables:
        table_manifest = manifest.tables[table.name]
        sql, labels = stats_query(table, table_manifest.columns)
        result = warehouse.execute(sql, timeout_s)
        if result.error is not None:
            problems.append(f"{table.name}: {result.error.category}: {result.error.message}")
            continue
        expected = expected_values(table, table_manifest.rows, table_manifest.columns)
        for label, want, got in zip(labels, expected, result.rows[0], strict=True):
            kind = None
            if label.endswith((".total", ".min", ".max")):
                kind = table.column(label.split(".")[0]).logical_type.kind
            if not same_value(kind, want, got):
                problems.append(f"{table.name}.{label}: expected {want}, got {got}")
    return problems
