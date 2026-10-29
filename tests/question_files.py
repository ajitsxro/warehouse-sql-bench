"""Write question YAML files for tests."""

from pathlib import Path
from typing import Any

import yaml

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
