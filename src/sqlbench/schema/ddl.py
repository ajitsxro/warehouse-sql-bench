"""CREATE TABLE statements for Snowflake and Databricks from one schema spec.

Both sides use the same logical type per column. Where the warehouses have no
exact equivalent, the differences are listed in docs/type_mapping.md.
"""

from collections.abc import Callable

from sqlbench.schema.spec import SchemaSpec, TableSpec
from sqlbench.schema.types import LogicalType, TypeKind
from sqlbench.warehouses.base import WarehouseName

SNOWFLAKE_SIMPLE_TYPES = {
    TypeKind.INT: "INTEGER",
    TypeKind.BIGINT: "BIGINT",
    TypeKind.DOUBLE: "FLOAT",
    TypeKind.STRING: "VARCHAR",
    TypeKind.BOOL: "BOOLEAN",
    TypeKind.DATE: "DATE",
    TypeKind.TIMESTAMP_NTZ: "TIMESTAMP_NTZ",
    TypeKind.TIMESTAMP_TZ: "TIMESTAMP_TZ",
    TypeKind.VARIANT: "VARIANT",
    # Semi-structured, untyped: elements and values are VARIANT.
    TypeKind.ARRAY: "ARRAY",
    TypeKind.STRUCT: "OBJECT",
    TypeKind.MAP: "OBJECT",
}

DATABRICKS_SIMPLE_TYPES = {
    TypeKind.INT: "INT",
    TypeKind.BIGINT: "BIGINT",
    TypeKind.DOUBLE: "DOUBLE",
    TypeKind.STRING: "STRING",
    TypeKind.BOOL: "BOOLEAN",
    TypeKind.DATE: "DATE",
    TypeKind.TIMESTAMP_NTZ: "TIMESTAMP_NTZ",
    # Databricks TIMESTAMP is an instant shown in the session time zone.
    TypeKind.TIMESTAMP_TZ: "TIMESTAMP",
    TypeKind.VARIANT: "VARIANT",
}


def snowflake_type(logical_type: LogicalType) -> str:
    if logical_type.kind == TypeKind.DECIMAL:
        return f"NUMBER({logical_type.precision},{logical_type.scale})"
    return SNOWFLAKE_SIMPLE_TYPES[logical_type.kind]


def databricks_type(logical_type: LogicalType) -> str:
    kind = logical_type.kind
    if kind == TypeKind.DECIMAL:
        return f"DECIMAL({logical_type.precision},{logical_type.scale})"
    if kind == TypeKind.ARRAY:
        assert logical_type.element is not None
        return f"ARRAY<{databricks_type(logical_type.element)}>"
    if kind == TypeKind.MAP:
        assert logical_type.key is not None and logical_type.element is not None
        key_type = databricks_type(logical_type.key)
        return f"MAP<{key_type}, {databricks_type(logical_type.element)}>"
    if kind == TypeKind.STRUCT:
        parts = [f"{field.name}: {databricks_type(field.type)}" for field in logical_type.fields]
        return f"STRUCT<{', '.join(parts)}>"
    return DATABRICKS_SIMPLE_TYPES[kind]


TYPE_NAMERS: dict[WarehouseName, Callable[[LogicalType], str]] = {
    WarehouseName.SNOWFLAKE: snowflake_type,
    WarehouseName.DATABRICKS: databricks_type,
}


def create_table_statements(
    schema: SchemaSpec, dialect: WarehouseName, target_schema: str
) -> list[str]:
    """One CREATE OR REPLACE TABLE per table, parents before children.

    Keys become informational PRIMARY KEY / FOREIGN KEY constraints. Neither
    warehouse enforces them, but they document the join graph.
    """
    if dialect not in TYPE_NAMERS:
        raise ValueError(f"no DDL emitter for {dialect}")
    statements: list[str] = []
    for table in schema.tables_in_dependency_order():
        statements.append(create_table(table, TYPE_NAMERS[dialect], target_schema))
    return statements


def create_table(table: TableSpec, type_name: Callable[[LogicalType], str], target: str) -> str:
    lines: list[str] = []
    for column in table.columns:
        line = f"{column.name} {type_name(column.logical_type)}"
        if not column.nullable:
            line += " NOT NULL"
        lines.append(line)
    if table.primary_key is not None:
        lines.append(f"CONSTRAINT {table.name}_pk PRIMARY KEY ({table.primary_key})")
    for column in table.columns:
        if column.references is None:
            continue
        parent_table, parent_column = column.references.split(".")
        lines.append(
            f"CONSTRAINT {table.name}_{column.name}_fk FOREIGN KEY ({column.name}) "
            f"REFERENCES {target}.{parent_table} ({parent_column})"
        )
    body = ",\n    ".join(lines)
    return f"CREATE OR REPLACE TABLE {target}.{table.name} (\n    {body}\n)"
