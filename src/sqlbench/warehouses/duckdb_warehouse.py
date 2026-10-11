"""DuckDB adapter: a free, in-process stand-in for the real warehouses in tests and CI."""

import time

import duckdb

from sqlbench.warehouses.base import (
    SESSION_TIME_ZONE,
    Column,
    ErrorCategory,
    QueryError,
    QueryResult,
    SessionContext,
    WarehouseName,
    rows_from_arrow,
)
from sqlbench.warehouses.deadline import Deadline
from sqlbench.warehouses.guard import check_read_only

COMPILE_ERRORS = (
    duckdb.ParserException,
    duckdb.SyntaxException,
    duckdb.BinderException,
    duckdb.CatalogException,
)
# DataError covers conversion, out-of-range and type-mismatch failures.
RUNTIME_ERRORS = (duckdb.DataError, duckdb.InvalidInputException, duckdb.ConstraintException)


def categorize_duckdb_error(error: duckdb.Error) -> ErrorCategory:
    if isinstance(error, COMPILE_ERRORS):
        return ErrorCategory.SYNTAX
    if isinstance(error, duckdb.InterruptException):
        return ErrorCategory.TIMEOUT
    if isinstance(error, duckdb.PermissionException):
        return ErrorCategory.PERMISSION
    if isinstance(error, RUNTIME_ERRORS):
        return ErrorCategory.RUNTIME
    return ErrorCategory.OTHER


class DuckDBWarehouse:
    """Runs queries on a DuckDB connection. Pass one in to query tables you created.

    The context's database is an attached DuckDB database, such as the file
    stem of sqlbench.duckdb.
    """

    name = WarehouseName.DUCKDB

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        context: SessionContext | None = None,
    ) -> None:
        if connection is None:
            connection = duckdb.connect()
        self._connection = connection
        connection.execute(f"SET TimeZone = '{SESSION_TIME_ZONE}'")
        context = context or SessionContext()
        names = [name for name in (context.database, context.schema) if name is not None]
        if names:
            # Quoted, so names like ecommerce__edge are used exactly as written.
            connection.execute("USE " + ".".join(f'"{name}"' for name in names))

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        rejection = check_read_only(sql, self.name)
        if rejection is not None:
            return QueryResult(elapsed_ms=0.0, error=rejection)

        # DuckDB has no statement timeout setting, so interrupt from a timer thread.
        deadline = Deadline(timeout_s, self._connection.interrupt)
        start = time.perf_counter()
        try:
            with deadline:
                relation = self._connection.sql(sql)
                columns = [
                    Column(name, str(column_type))
                    for name, column_type in zip(relation.columns, relation.types, strict=True)
                ]
                rows = rows_from_arrow(relation.to_arrow_table())
        except duckdb.Error as error:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if deadline.expired:
                category = ErrorCategory.TIMEOUT
            else:
                category = categorize_duckdb_error(error)
            return QueryResult(elapsed_ms=elapsed_ms, error=QueryError(category, str(error)))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return QueryResult(elapsed_ms=elapsed_ms, columns=columns, rows=rows)

    def close(self) -> None:
        self._connection.close()
