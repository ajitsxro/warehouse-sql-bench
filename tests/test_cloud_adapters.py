"""The Snowflake adapter runs against a fake connection; no network access."""

from types import TracebackType
from typing import Any, cast

import pyarrow as pa
import pytest
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import ProgrammingError

from sqlbench.settings import MissingEnvError
from sqlbench.warehouses.base import Column, ErrorCategory
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


def test_missing_credentials_raise_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)

    with pytest.raises(MissingEnvError, match="SNOWFLAKE_ACCOUNT"):
        SnowflakeWarehouse().execute("SELECT 1", timeout_s=30)
