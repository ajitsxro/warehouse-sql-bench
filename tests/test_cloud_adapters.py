"""Snowflake and Databricks adapters run against fake connections; no network access."""

import threading
from types import TracebackType
from typing import Any, cast

import pyarrow as pa
import pytest
from databricks.sql.client import Connection as DatabricksConnection
from databricks.sql.exc import ServerOperationError
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import ProgrammingError

from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import Column, ErrorCategory
from sqlbench.warehouses.databricks_warehouse import DatabricksWarehouse, connect_from_env
from sqlbench.warehouses.snowflake_warehouse import SnowflakeWarehouse


class FakeCursor:
    """Records statements and returns one fixed result, or raises for chosen SQL."""

    def __init__(self, description: list[Any], table: pa.Table) -> None:
        self.description = description
        self.table = table
        self.executed: list[str] = []
        self.errors: dict[str, Exception] = {}

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.executed.append(sql)
        if sql in self.errors:
            raise self.errors[sql]

    def fetch_arrow_all(self, **kwargs: Any) -> pa.Table:
        return self.table

    def fetchall_arrow(self) -> pa.Table:
        return self.table

    def cancel(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


class SnowflakeMeta:
    def __init__(self, name: str, type_code: int) -> None:
        self.name = name
        self.type_code = type_code


def make_snowflake(cursor: FakeCursor) -> tuple[SnowflakeWarehouse, list[FakeConnection]]:
    opened: list[FakeConnection] = []

    def connect() -> SnowflakeConnection:
        connection = FakeConnection(cursor)
        opened.append(connection)
        return cast(SnowflakeConnection, connection)

    return SnowflakeWarehouse(connect), opened


def test_snowflake_sets_timeout_once_and_maps_type_codes() -> None:
    table = pa.table({"N": [1, 2], "S": ["a", "b"]})
    # Type codes 0 and 2 are Snowflake's FIXED and TEXT.
    cursor = FakeCursor([SnowflakeMeta("N", 0), SnowflakeMeta("S", 2)], table)
    warehouse, opened = make_snowflake(cursor)

    first = warehouse.execute("SELECT n, s FROM t", timeout_s=30)
    warehouse.execute("SELECT n, s FROM t", timeout_s=30)
    warehouse.execute("SELECT n, s FROM t", timeout_s=60)

    assert first.error is None
    assert first.columns == [Column("N", "FIXED"), Column("S", "TEXT")]
    assert first.rows == [(1, "a"), (2, "b")]
    assert cursor.executed == [
        "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 30",
        "SELECT n, s FROM t",
        "SELECT n, s FROM t",
        "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 60",
        "SELECT n, s FROM t",
    ]
    assert len(opened) == 1


def test_snowflake_guard_rejects_before_connecting() -> None:
    cursor = FakeCursor([], pa.table({}))
    warehouse, opened = make_snowflake(cursor)

    result = warehouse.execute("DELETE FROM t", timeout_s=30)

    assert result.error is not None
    assert result.error.category == ErrorCategory.PERMISSION
    assert opened == []


def test_snowflake_query_error_is_returned_with_category_and_raw_message() -> None:
    cursor = FakeCursor([], pa.table({}))
    cursor.errors["SELECT nope FROM t"] = ProgrammingError(
        msg="invalid identifier 'NOPE'", errno=904, sqlstate="42000"
    )
    warehouse, _ = make_snowflake(cursor)

    result = warehouse.execute("SELECT nope FROM t", timeout_s=30)

    assert result.error is not None
    assert result.error.category == ErrorCategory.SYNTAX
    assert "invalid identifier 'NOPE'" in result.error.message


def test_snowflake_close_closes_the_open_connection() -> None:
    cursor = FakeCursor([SnowflakeMeta("N", 0)], pa.table({"N": [1]}))
    warehouse, opened = make_snowflake(cursor)
    warehouse.execute("SELECT 1 AS n", timeout_s=30)

    warehouse.close()

    assert opened[0].closed


class HangingDatabricksCursor(FakeCursor):
    """Blocks in execute until cancel() is called, like a stuck warehouse query."""

    def __init__(self) -> None:
        super().__init__([], pa.table({}))
        self.cancelled = threading.Event()

    def execute(self, sql: str) -> None:
        self.executed.append(sql)
        if sql.startswith("SET "):
            return
        # The 5s wait only matters if cancel never arrives; the test would then fail.
        self.cancelled.wait(timeout=5)
        raise ServerOperationError("Query was cancelled", {})

    def cancel(self) -> None:
        self.cancelled.set()


def make_databricks(cursor: FakeCursor) -> DatabricksWarehouse:
    def connect() -> DatabricksConnection:
        return cast(DatabricksConnection, FakeConnection(cursor))

    return DatabricksWarehouse(connect)


def test_databricks_returns_rows_and_sets_server_timeout_with_margin() -> None:
    cursor = FakeCursor([("n", "int", None, None, None, None, True)], pa.table({"n": [7]}))
    warehouse = make_databricks(cursor)

    result = warehouse.execute("SELECT 7 AS n", timeout_s=30)

    assert result.error is None
    assert result.columns == [Column("n", "int")]
    assert result.rows == [(7,)]
    assert cursor.executed == ["SET STATEMENT_TIMEOUT = 40", "SELECT 7 AS n"]


def test_databricks_client_side_timeout_cancels_and_reports_timeout() -> None:
    cursor = HangingDatabricksCursor()
    warehouse = make_databricks(cursor)

    result = warehouse.execute("SELECT * FROM huge", timeout_s=1)

    assert cursor.cancelled.is_set()
    assert result.error is not None
    assert result.error.category == ErrorCategory.TIMEOUT
    assert 900 <= result.elapsed_ms < 4000


def test_databricks_server_error_uses_sqlstate() -> None:
    cursor = FakeCursor([], pa.table({}))
    cursor.errors["SELECT * FROM missing"] = ServerOperationError(
        "[TABLE_OR_VIEW_NOT_FOUND] The table or view `missing` cannot be found. SQLSTATE: 42P01",
        {},
    )
    warehouse = make_databricks(cursor)

    result = warehouse.execute("SELECT * FROM missing", timeout_s=30)

    assert result.error is not None
    assert result.error.category == ErrorCategory.SYNTAX


def test_missing_credentials_raise_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ["SNOWFLAKE_ACCOUNT", "DATABRICKS_SERVER_HOSTNAME"]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(MissingEnvError, match="SNOWFLAKE_ACCOUNT"):
        SnowflakeWarehouse().execute("SELECT 1", timeout_s=30)
    with pytest.raises(MissingEnvError, match="DATABRICKS_SERVER_HOSTNAME"):
        DatabricksWarehouse().execute("SELECT 1", timeout_s=30)


def test_databricks_connect_caps_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_SERVER_HOSTNAME", "example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    monkeypatch.setenv("DATABRICKS_TOKEN", "not-a-real-token")
    captured: dict[str, Any] = {}

    def fake_connect(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("databricks.sql.connect", fake_connect)

    connect_from_env()

    assert captured["_retry_stop_after_attempts_count"] == 3
    assert captured["_retry_stop_after_attempts_duration"] == 30
