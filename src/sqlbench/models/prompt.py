"""Render the one prompt template every model gets.

The prompt holds the warehouse's DDL with bare table names, three sample
rows per table from the main dataset, and the question. Nothing in it
depends on the model.
"""

import datetime
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sqlbench.datagen.tables import Profile
from sqlbench.schema.ddl import create_table_statements
from sqlbench.schema.spec import SchemaSpec
from sqlbench.warehouses.base import WarehouseName

SAMPLE_ROW_COUNT = 3

# (name shown to the model, dialect name)
WAREHOUSE_LABELS = {
    WarehouseName.SNOWFLAKE: ("Snowflake", "Snowflake SQL"),
    WarehouseName.DATABRICKS: ("Databricks", "Databricks SQL"),
}


@dataclass(frozen=True)
class PromptTemplate:
    text: str
    # Recorded with every result, so a changed template never mixes with old results.
    sha256: str


def load_template(path: Path) -> PromptTemplate:
    text = path.read_text()
    return PromptTemplate(text, hashlib.sha256(text.encode()).hexdigest())


def render_prompt(
    template: PromptTemplate,
    schema: SchemaSpec,
    warehouse: WarehouseName,
    question: str,
    data_dir: Path,
) -> str:
    warehouse_name, dialect = WAREHOUSE_LABELS[warehouse]
    ddl = ";\n\n".join(create_table_statements(schema, warehouse, None)) + ";"
    return template.text.format(
        warehouse_name=warehouse_name,
        dialect=dialect,
        ddl=ddl,
        sample_rows=sample_rows(schema, data_dir),
        question=question,
    )


def sample_rows(schema: SchemaSpec, data_dir: Path) -> str:
    """The first rows of each table in the main dataset, one block per table."""
    blocks = []
    for table in schema.tables_in_dependency_order():
        path = data_dir / Profile.MAIN / schema.name / f"{table.name}.parquet"
        rows = pq.read_table(path).slice(0, SAMPLE_ROW_COUNT).to_pylist()
        lines = [table.name, " | ".join(column.name for column in table.columns)]
        for row in rows:
            values = [format_value(row[column.name]) for column in table.columns]
            lines.append(" | ".join(values))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_value(value: Any) -> str:
    """Values as a person would read them; nested values as compact JSON."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, datetime.date | Decimal):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, tuple) for item in value):
        # A Parquet map arrives as (key, value) pairs.
        return json.dumps(dict(value), separators=(", ", ": "))
    if isinstance(value, dict | list):
        return json.dumps(value, separators=(", ", ": "))
    return str(value)
