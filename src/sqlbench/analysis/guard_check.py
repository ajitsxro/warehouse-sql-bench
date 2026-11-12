"""Guard false positives: predictions the read-only guard couldn't parse, re-checked
with EXPLAIN on the warehouse they were meant for.

EXPLAIN compiles a query without running it. It goes through a plain
reader connection, not the adapter, because the adapter would apply the
same guard again. The reader principal has only SELECT grants, so this is
still read-only.
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench.datagen.tables import Profile
from sqlbench.harness.store import ResultRow
from sqlbench.load.dataset import session_context
from sqlbench.warehouses.base import SessionContext, WarehouseName
from sqlbench.warehouses.guard import PARSE_FAILURE_PREFIX

CSV_FIELDS = ["model_id", "warehouse", "question_id", "accepted", "warehouse_error"]
ERROR_CHARS = 300


@dataclass(frozen=True)
class GuardCheck:
    model_id: str
    warehouse: str
    question_id: str
    accepted: bool
    warehouse_error: str


def guard_blocks(rows: list[ResultRow]) -> list[ResultRow]:
    """Rows whose prediction never reached the warehouse because sqlglot couldn't parse it."""
    blocked = []
    for row in rows:
        if row.error_message and row.error_message.startswith(PARSE_FAILURE_PREFIX):
            blocked.append(row)
    return blocked


def explain(connection: Any, sql: str) -> str | None:
    """None if the warehouse compiles sql, else its error message."""
    cursor = connection.cursor()
    try:
        cursor.execute(f"EXPLAIN {sql}")
        cursor.fetchall()
    except Exception as error:  # noqa: BLE001 - any driver error means "not accepted"
        return str(error)
    finally:
        cursor.close()
    return None


def check_blocks(
    blocked: list[ResultRow],
    connect: Callable[[WarehouseName, SessionContext], Any],
) -> list[GuardCheck]:
    checks = []
    for row in blocked:
        warehouse = WarehouseName(row.warehouse)
        # Question ids are <schema>_<number>.
        schema_name = row.question_id.rsplit("_", 1)[0]
        context = session_context(schema_name, Profile.MAIN)
        connection = connect(warehouse, context)
        try:
            error = explain(connection, row.predicted_sql or "")
        finally:
            connection.close()
        checks.append(
            GuardCheck(
                row.model_id,
                row.warehouse,
                row.question_id,
                error is None,
                (error or "")[:ERROR_CHARS],
            )
        )
    return checks


def write_checks(checks: list[GuardCheck], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "model_id": check.model_id,
                    "warehouse": check.warehouse,
                    "question_id": check.question_id,
                    "accepted": "yes" if check.accepted else "no",
                    "warehouse_error": check.warehouse_error,
                }
            )


def read_checks(path: Path) -> list[GuardCheck]:
    checks = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            checks.append(
                GuardCheck(
                    record["model_id"],
                    record["warehouse"],
                    record["question_id"],
                    record["accepted"] == "yes",
                    record["warehouse_error"],
                )
            )
    return checks
