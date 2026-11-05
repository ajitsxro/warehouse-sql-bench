"""Databricks adapter for a SQL warehouse, authenticated with a personal access token."""

import re
import time
from collections.abc import Callable

from databricks import sql as databricks_sql
from databricks.sql.client import Connection as DatabricksConnection
from databricks.sql.exc import Error as DatabricksError

from sqlbench.settings import require_env
from sqlbench.warehouses.base import (
    Column,
    ErrorCategory,
    QueryError,
    QueryResult,
    WarehouseName,
    rows_from_arrow,
)
from sqlbench.warehouses.deadline import Deadline
from sqlbench.warehouses.guard import check_read_only
from sqlbench.warehouses.sqlstate import category_from_sqlstate

ENV_VARS = ["DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"]
# The client cancels at timeout_s. The server timeout is a little longer, as a
# backstop in case the cancel request is lost, so the client normally fires
# first and the failure is reliably reported as a timeout.
SERVER_TIMEOUT_MARGIN_S = 10
# Databricks exposes no SQLSTATE attribute; it is only in the message text.
SQLSTATE_PATTERN = re.compile(r"SQLSTATE: ([0-9A-Z]{5})")


def categorize_databricks_error(error: DatabricksError) -> ErrorCategory:
    message = error.message or ""
    match = SQLSTATE_PATTERN.search(message)
    if match is None:
        return ErrorCategory.OTHER
    return category_from_sqlstate(match.group(1))


def connect_from_env() -> DatabricksConnection:
    env = require_env(ENV_VARS)
    return databricks_sql.connect(
        server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
        http_path=env["DATABRICKS_HTTP_PATH"],
        access_token=env["DATABRICKS_TOKEN"],
    )


class DatabricksWarehouse:
    """Runs queries on a Databricks SQL warehouse. The connection opens on the first execute."""

    name = WarehouseName.DATABRICKS

    def __init__(self, connect: Callable[[], DatabricksConnection] = connect_from_env) -> None:
        self._connect = connect
        self._connection: DatabricksConnection | None = None
        self._session_timeout_s: int | None = None

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        rejection = check_read_only(sql, self.name)
        if rejection is not None:
            return QueryResult(elapsed_ms=0.0, error=rejection)

        start = time.perf_counter()
        deadline: Deadline | None = None
        try:
            if self._connection is None:
                self._connection = self._connect()
            with self._connection.cursor() as cursor:
                server_timeout_s = int(timeout_s) + SERVER_TIMEOUT_MARGIN_S
                if self._session_timeout_s != server_timeout_s:
                    cursor.execute(f"SET STATEMENT_TIMEOUT = {server_timeout_s}")
                    self._session_timeout_s = server_timeout_s
                deadline = Deadline(timeout_s, cursor.cancel)
                # Time only the query, not connecting or session setup.
                start = time.perf_counter()
                with deadline:
                    cursor.execute(sql)
                    # Each description entry is (name, type_code, ...) per PEP 249.
                    columns = [
                        Column(description[0], str(description[1]))
                        for description in cursor.description or []
                    ]
                    rows = rows_from_arrow(cursor.fetchall_arrow())
        except DatabricksError as error:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if deadline is not None and deadline.expired:
                category = ErrorCategory.TIMEOUT
            else:
                category = categorize_databricks_error(error)
            return QueryResult(elapsed_ms=elapsed_ms, error=QueryError(category, str(error)))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return QueryResult(elapsed_ms=elapsed_ms, columns=columns, rows=rows)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
