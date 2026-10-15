"""Load into a local DuckDB file, for tests and for trying queries offline."""

from pathlib import Path

import duckdb

from sqlbench.schema.spec import TableSpec
from sqlbench.schema.types import LogicalType, TypeKind
from sqlbench.warehouses.base import SESSION_TIME_ZONE, WarehouseName

DUCKDB_SIMPLE_TYPES = {
    TypeKind.INT: "INTEGER",
    TypeKind.BIGINT: "BIGINT",
    TypeKind.DOUBLE: "DOUBLE",
    TypeKind.STRING: "VARCHAR",
    TypeKind.BOOL: "BOOLEAN",
    TypeKind.DATE: "DATE",
    TypeKind.TIMESTAMP_NTZ: "TIMESTAMP",
    TypeKind.TIMESTAMP_TZ: "TIMESTAMPTZ",
    # DuckDB has no VARIANT; its JSON type keeps JSON null apart from SQL NULL.
    TypeKind.VARIANT: "JSON",
}


def duckdb_type(logical_type: LogicalType) -> str:
    kind = logical_type.kind
    if kind == TypeKind.DECIMAL:
        return f"DECIMAL({logical_type.precision},{logical_type.scale})"
    if kind == TypeKind.ARRAY:
        assert logical_type.element is not None
        return f"{duckdb_type(logical_type.element)}[]"
    if kind == TypeKind.MAP:
        assert logical_type.key is not None and logical_type.element is not None
        return f"MAP({duckdb_type(logical_type.key)}, {duckdb_type(logical_type.element)})"
    if kind == TypeKind.STRUCT:
        parts = [f"{field.name} {duckdb_type(field.type)}" for field in logical_type.fields]
        return f"STRUCT({', '.join(parts)})"
    return DUCKDB_SIMPLE_TYPES[kind]


def quote_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


class DuckDBLoader:
    """Loads into the DuckDB file at path; its file stem becomes the database name."""

    name = WarehouseName.DUCKDB

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        self._connection.execute(f"SET TimeZone = '{SESSION_TIME_ZONE}'")

    def reset_schema(self, target: str) -> None:
        self._connection.execute(f'DROP SCHEMA IF EXISTS "{target}" CASCADE')
        self._connection.execute(f'CREATE SCHEMA "{target}"')

    def load_table(self, target: str, table: TableSpec, parquet_path: Path) -> None:
        # Constraints are left out: DuckDB enforces them, which the warehouses
        # don't, and the data was already checked when it was generated.
        selections = []
        for column in table.columns:
            column_type = duckdb_type(column.logical_type)
            selections.append(f"CAST({column.name} AS {column_type}) AS {column.name}")
        self._connection.execute(
            f'CREATE TABLE "{target}"."{table.name}" AS SELECT {", ".join(selections)} '
            f"FROM read_parquet({quote_literal(str(parquet_path))})"
        )

    def grant_read(self, target: str) -> None:
        # DuckDB has no roles. Queries open the file read-only instead.
        return None

    def close(self) -> None:
        self._connection.close()
