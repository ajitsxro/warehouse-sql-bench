import datetime
from collections.abc import Iterator
from decimal import Decimal

import duckdb
import pytest

from sqlbench.warehouses.base import Column, ErrorCategory, SessionContext
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse


@pytest.fixture
def warehouse() -> Iterator[DuckDBWarehouse]:
    connection = duckdb.connect()
    # Setup DDL goes straight to the connection; the adapter only runs queries.
    connection.execute("CREATE TABLE orders (id INTEGER, amount DECIMAL(10, 2), placed DATE)")
    connection.execute(
        "INSERT INTO orders VALUES (1, 10.50, DATE '2024-01-31'), (2, 3.25, DATE '2024-02-29')"
    )
    adapter = DuckDBWarehouse(connection)
    yield adapter
    adapter.close()


def test_select_returns_columns_types_and_python_values(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT id, amount, placed FROM orders ORDER BY id", timeout_s=10)

    assert result.error is None
    assert result.columns == [
        Column("id", "INTEGER"),
        Column("amount", "DECIMAL(10,2)"),
        Column("placed", "DATE"),
    ]
    assert result.rows == [
        (1, Decimal("10.50"), datetime.date(2024, 1, 31)),
        (2, Decimal("3.25"), datetime.date(2024, 2, 29)),
    ]
    assert result.elapsed_ms >= 0


def test_duplicate_column_names_keep_both_values(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT 1 AS a, 2 AS a", timeout_s=10)

    assert [column.name for column in result.columns] == ["a", "a"]
    assert result.rows == [(1, 2)]


def test_empty_result_keeps_columns(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT id FROM orders WHERE id < 0", timeout_s=10)

    assert result.error is None
    assert result.columns == [Column("id", "INTEGER")]
    assert result.rows == []


def test_nulls_and_nested_types_round_trip(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT NULL::INTEGER AS n, [1, 2] AS l, {'k': 'v'} AS s", 10)

    assert result.rows == [(None, [1, 2], {"k": "v"})]


@pytest.mark.parametrize(
    ("sql", "category"),
    [
        ("SELECT missing_column FROM orders", ErrorCategory.SYNTAX),
        ("SELECT * FROM missing_table", ErrorCategory.SYNTAX),
        ("SELECT 'abc'::INTEGER", ErrorCategory.RUNTIME),
        ("SELECT (SELECT id FROM orders)", ErrorCategory.RUNTIME),
    ],
)
def test_database_errors_are_categorized(
    warehouse: DuckDBWarehouse, sql: str, category: ErrorCategory
) -> None:
    result = warehouse.execute(sql, timeout_s=10)

    assert result.error is not None
    assert result.error.category == category
    assert result.error.message
    assert result.rows == []


def test_long_query_times_out_and_connection_stays_usable(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT count(*) FROM range(100000000000)", timeout_s=1)

    assert result.error is not None
    assert result.error.category == ErrorCategory.TIMEOUT
    assert 900 <= result.elapsed_ms < 5000

    follow_up = warehouse.execute("SELECT 1", timeout_s=10)
    assert follow_up.rows == [(1,)]


def test_guard_blocks_writes_before_they_reach_duckdb(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("DROP TABLE orders", timeout_s=10)

    assert result.error is not None
    assert result.error.category == ErrorCategory.BLOCKED
    still_there = warehouse.execute("SELECT count(*) FROM orders", timeout_s=10)
    assert still_there.rows == [(2,)]


def test_context_makes_unqualified_names_resolve_in_that_schema() -> None:
    connection = duckdb.connect()
    connection.execute("ATTACH ':memory:' AS sqlbench")
    for schema in ("shop", "shop__edge"):
        connection.execute(f"CREATE SCHEMA sqlbench.{schema}")
    connection.execute("CREATE TABLE sqlbench.shop.orders AS SELECT 1 AS id")
    connection.execute("CREATE TABLE sqlbench.shop__edge.orders AS SELECT 2 AS id")

    edge = DuckDBWarehouse(connection, SessionContext("sqlbench", "shop__edge"))

    assert edge.execute("SELECT id FROM orders", timeout_s=10).rows == [(2,)]


def test_sessions_run_in_utc(warehouse: DuckDBWarehouse) -> None:
    result = warehouse.execute("SELECT TIMESTAMPTZ '2024-03-10 07:00:00+00' AS t", timeout_s=10)

    assert result.rows[0][0].utcoffset() == datetime.timedelta(0)
    assert str(result.rows[0][0].tzinfo) == "UTC"
