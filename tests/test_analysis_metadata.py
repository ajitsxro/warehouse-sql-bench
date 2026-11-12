"""Function lists and the guard check, with fake connections."""

import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

from test_harness_store import ROW

from sqlbench.analysis.functions import (
    fetch_and_write,
    load_function_names,
    names_from_listing,
)
from sqlbench.analysis.guard_check import check_blocks, guard_blocks, read_checks, write_checks
from sqlbench.warehouses.base import SessionContext, WarehouseName
from sqlbench.warehouses.guard import PARSE_FAILURE_PREFIX

TODAY = datetime.date(2026, 11, 12)


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def execute(self, sql: str) -> None:
        self.connection.statements.append(sql)
        if self.connection.error is not None:
            raise RuntimeError(self.connection.error)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.connection.rows

    def close(self) -> None:
        pass


class FakeConnection:
    def __init__(self, rows: list[tuple[Any, ...]], error: str | None = None) -> None:
        self.rows = rows
        self.error = error
        self.statements: list[str] = []
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def snowflake_row(name: str, builtin: str = "Y") -> tuple[Any, ...]:
    return (None, name, "", builtin)


def test_snowflake_listing_keeps_builtins_and_adds_table_functions() -> None:
    rows = [snowflake_row("IFF"), snowflake_row("IFF"), snowflake_row("MY_UDF", "N")]

    names = names_from_listing(WarehouseName.SNOWFLAKE, rows)

    assert "iff" in names
    assert "my_udf" not in names
    assert "flatten" in names
    assert names.count("iff") == 1


def test_databricks_listing_drops_operators() -> None:
    names = names_from_listing(WarehouseName.DATABRICKS, [("!=",), ("explode",), ("%",)])

    assert names == ["explode"]


def test_fetched_lists_round_trip(tmp_path: Path) -> None:
    connections = {
        WarehouseName.SNOWFLAKE: FakeConnection([snowflake_row("IFF")]),
        WarehouseName.DATABRICKS: FakeConnection([("explode",)]),
    }

    counts = fetch_and_write(lambda warehouse: connections[warehouse], tmp_path, TODAY)

    assert counts[WarehouseName.DATABRICKS] == 1
    assert load_function_names(WarehouseName.DATABRICKS, tmp_path) == {"explode"}
    assert "iff" in load_function_names(WarehouseName.SNOWFLAKE, tmp_path)
    assert all(connection.closed for connection in connections.values())


BLOCKED = replace(
    ROW,
    warehouse="databricks",
    question_id="hr_people_006",
    predicted_sql="WITH RECURSIVE t MAX RECURSION LEVEL 10 AS (SELECT 1) SELECT * FROM t",
    error_message=f"{PARSE_FAILURE_PREFIX}: Expecting (",
    outcome="syntax_error",
)


def test_only_guard_parse_failures_are_rechecked() -> None:
    warehouse_error = replace(ROW, error_message="000904 invalid identifier 'X'")

    assert guard_blocks([ROW, warehouse_error, BLOCKED]) == [BLOCKED]


def test_explain_decides_whether_the_warehouse_accepts(tmp_path: Path) -> None:
    contexts: list[SessionContext] = []
    accepting = FakeConnection([])

    def connect(warehouse: WarehouseName, context: SessionContext) -> Any:
        contexts.append(context)
        return accepting

    checks = check_blocks([BLOCKED], connect)

    assert checks[0].accepted
    assert accepting.statements == [f"EXPLAIN {BLOCKED.predicted_sql}"]
    assert contexts == [SessionContext("sqlbench", "hr_people")]
    assert accepting.closed

    rejecting = FakeConnection([], error="PARSE_SYNTAX_ERROR")
    rejected = check_blocks([BLOCKED], lambda warehouse, context: rejecting)
    path = tmp_path / "checks.csv"
    write_checks(checks + rejected, path)

    assert [check.accepted for check in read_checks(path)] == [True, False]
    assert read_checks(path)[1].warehouse_error == "PARSE_SYNTAX_ERROR"
