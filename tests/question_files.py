"""Question YAML files and fake warehouses for the question tests."""

from pathlib import Path
from typing import Any

import yaml

from sqlbench.datagen.tables import Profile
from sqlbench.questions.check import GoldTargets
from sqlbench.warehouses.base import QueryError, QueryResult, WarehouseName

Rows = list[tuple[Any, ...]]

PORTABLE_SQL = "SELECT segment, COUNT(*) AS customers FROM customers GROUP BY segment"


def question_data(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "mini_shop_001",
        "schema": "mini_shop",
        "question": "How many customers are in each segment?",
        "category": "portable",
        "divergence": "none",
        "features": ["aggregate"],
        "difficulty": "easy",
        "gold_sql": {"snowflake": PORTABLE_SQL, "databricks": PORTABLE_SQL},
        "order_matters": False,
        "float_tolerance": 0,
    }
    data.update(changes)
    return data


def write_question(questions_dir: Path, **changes: Any) -> Path:
    data = question_data(**changes)
    path: Path = questions_dir / data["schema"] / f"{data['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def both_golds(sql: str) -> dict[str, str]:
    return {"snowflake": sql, "databricks": sql}


class FakeWarehouse:
    """Returns fixed rows, or a fixed error, for every query."""

    def __init__(self, name: WarehouseName, rows: Rows, error: QueryError | None = None) -> None:
        self.name = name
        self.rows = rows
        self.error = error
        self.queries: list[str] = []
        self.closed = False

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        self.queries.append(sql)
        if self.error is not None:
            return QueryResult(elapsed_ms=1.0, error=self.error)
        return QueryResult(elapsed_ms=1.0, rows=self.rows)

    def close(self) -> None:
        self.closed = True


def targets(
    snowflake_main: Rows, databricks_main: Rows, snowflake_edge: Rows, databricks_edge: Rows
) -> dict[tuple[WarehouseName, Profile], FakeWarehouse]:
    return {
        (WarehouseName.SNOWFLAKE, Profile.MAIN): FakeWarehouse(
            WarehouseName.SNOWFLAKE, snowflake_main
        ),
        (WarehouseName.SNOWFLAKE, Profile.EDGE): FakeWarehouse(
            WarehouseName.SNOWFLAKE, snowflake_edge
        ),
        (WarehouseName.DATABRICKS, Profile.MAIN): FakeWarehouse(
            WarehouseName.DATABRICKS, databricks_main
        ),
        (WarehouseName.DATABRICKS, Profile.EDGE): FakeWarehouse(
            WarehouseName.DATABRICKS, databricks_edge
        ),
    }


def as_targets(fakes: dict[tuple[WarehouseName, Profile], FakeWarehouse]) -> GoldTargets:
    result: GoldTargets = {}
    for key, fake in fakes.items():
        result[key] = fake
    return result
