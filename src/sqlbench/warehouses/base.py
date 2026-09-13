"""The interface every warehouse adapter implements, and the result types it returns."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import pyarrow as pa


class WarehouseName(StrEnum):
    """Supported warehouses. The values double as sqlglot dialect names."""

    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    DUCKDB = "duckdb"


class ErrorCategory(StrEnum):
    """Coarse failure class for the error taxonomy.

    SYNTAX covers everything rejected before the query runs: parse errors and
    compile errors such as an unknown column or table. RUNTIME covers failures
    while the query runs, such as a bad cast or a scalar subquery returning
    several rows. BLOCKED means the read-only guard refused the SQL, so it never
    reached the warehouse; PERMISSION is only for errors the warehouse returns.
    """

    SYNTAX = "syntax"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    BLOCKED = "blocked"
    OTHER = "other"


@dataclass(frozen=True)
class QueryError:
    category: ErrorCategory
    message: str


@dataclass(frozen=True)
class Column:
    name: str
    # The warehouse's own type name, e.g. "FIXED" on Snowflake or "int" on Databricks.
    type: str


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one statement. On failure, error is set and columns and rows are empty."""

    elapsed_ms: float
    columns: list[Column] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: QueryError | None = None


class Warehouse(Protocol):
    name: WarehouseName

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        """Run one read-only statement. Query failures are returned, not raised."""
        ...

    def close(self) -> None: ...


def rows_from_arrow(table: pa.Table) -> list[tuple[Any, ...]]:
    """Convert an Arrow table to row tuples.

    Goes column by column instead of through Table.to_pylist(), which returns
    dicts and would silently drop a column when two share a name.
    """
    columns = [column.to_pylist() for column in table.columns]
    return list(zip(*columns, strict=True))
