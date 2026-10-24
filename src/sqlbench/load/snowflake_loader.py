"""Load Parquet into Snowflake: PUT to an internal stage, then COPY INTO.

The stage lives in its own schema, sqlbench.loading, so replacing a data
schema never touches it, and the read-only role is never granted on it.
"""

from collections.abc import Callable
from pathlib import Path

from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import Error as SnowflakeError

from sqlbench.load.dataset import DATABASE, LoadError
from sqlbench.schema.ddl import create_table, self_reference_constraints, snowflake_type
from sqlbench.schema.spec import ColumnSpec, TableSpec
from sqlbench.schema.types import TypeKind
from sqlbench.settings import require_env
from sqlbench.warehouses.base import SessionContext, WarehouseName
from sqlbench.warehouses.snowflake_warehouse import connect_owner_from_env

STAGE = "loading.files"


def copy_expression(column: ColumnSpec) -> str:
    """How COPY reads one column from the staged Parquet row, which arrives as $1."""
    field = f"$1:{column.name}"
    kind = column.logical_type.kind
    if kind == TypeKind.VARIANT:
        # The Parquet column holds JSON text. Parsing turns the text 'null'
        # into JSON null, while SQL NULL stays NULL.
        return f"PARSE_JSON({field}::STRING)"
    if kind == TypeKind.MAP:
        # A Parquet map arrives as {"key_value": [{"key": ..., "value": ...}]},
        # and an empty map as JSON null, so empty and NULL maps look the same.
        # A NOT NULL map column gets an empty object for it.
        pairs = f"{field}:key_value"
        if not column.nullable:
            pairs = f"COALESCE({pairs}, [])"
        return (
            f"REDUCE({pairs}, {{}}::OBJECT, "
            "(acc, pair) -> OBJECT_INSERT(acc, pair:key::STRING, pair:value))"
        )
    return f"{field}::{snowflake_type(column.logical_type)}"


class SnowflakeLoader:
    """Loads as the owner role and grants SELECT to the read-only role."""

    name = WarehouseName.SNOWFLAKE

    def __init__(
        self,
        connect: Callable[[SessionContext], SnowflakeConnection] = connect_owner_from_env,
    ) -> None:
        self._reader_role = require_env(["SNOWFLAKE_READER_ROLE"])["SNOWFLAKE_READER_ROLE"]
        self._connection = connect(SessionContext())
        self.run(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
        self.run(f"USE DATABASE {DATABASE}")
        self.run("CREATE SCHEMA IF NOT EXISTS loading")
        self.run(
            f"CREATE STAGE IF NOT EXISTS {STAGE} "
            "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE)"
        )
        self.run(f"GRANT USAGE ON DATABASE {DATABASE} TO ROLE {self._reader_role}")

    def run(self, sql: str) -> None:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
        except SnowflakeError as error:
            raise LoadError(f"Snowflake rejected: {sql.splitlines()[0]}\n{error}") from error

    def reset_schema(self, target: str) -> None:
        # Replacing the whole schema also drops the tables' foreign keys, so the
        # tables can be recreated in any order.
        self.run(f"CREATE OR REPLACE SCHEMA {target}")

    def load_table(self, target: str, table: TableSpec, parquet_path: Path) -> None:
        self.run(create_table(table, snowflake_type, target))
        for statement in self_reference_constraints(table, target):
            self.run(statement)
        self.run(
            f"PUT 'file://{parquet_path.resolve()}' @{STAGE}/{target}/ "
            "OVERWRITE = TRUE AUTO_COMPRESS = FALSE"
        )
        names = []
        expressions = []
        for column in table.columns:
            names.append(column.name)
            expressions.append(copy_expression(column))
        # A transform rather than MATCH_BY_COLUMN_NAME, which can't reshape maps
        # or parse JSON text, and fails on NOT NULL map columns with empty maps.
        self.run(
            f"COPY INTO {target}.{table.name} ({', '.join(names)}) "
            f"FROM (SELECT {', '.join(expressions)} FROM @{STAGE}/{target}/) "
            f"FILES = ('{parquet_path.name}') FORCE = TRUE"
        )

    def grant_read(self, target: str) -> None:
        self.run(f"GRANT USAGE ON SCHEMA {target} TO ROLE {self._reader_role}")
        # Tables were just recreated, which drops earlier grants on them.
        self.run(f"GRANT SELECT ON ALL TABLES IN SCHEMA {target} TO ROLE {self._reader_role}")

    def close(self) -> None:
        self._connection.close()
