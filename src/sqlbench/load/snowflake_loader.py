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
from sqlbench.schema.spec import TableSpec
from sqlbench.schema.types import TypeKind
from sqlbench.settings import require_env
from sqlbench.warehouses.base import SessionContext, WarehouseName
from sqlbench.warehouses.snowflake_warehouse import connect_owner_from_env

STAGE = "loading.files"


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
        self.run(
            f"COPY INTO {target}.{table.name} FROM @{STAGE}/{target}/ "
            f"FILES = ('{parquet_path.name}') "
            "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE FORCE = TRUE"
        )
        # VARIANT columns hold JSON text in Parquet, and COPY stores it as a
        # VARIANT string. Parsing turns it into objects and arrays; the text
        # 'null' becomes JSON null and SQL NULL stays NULL.
        variant_columns = []
        for column in table.columns:
            if column.logical_type.kind == TypeKind.VARIANT:
                variant_columns.append(column.name)
        if variant_columns:
            assignments = [f"{name} = PARSE_JSON({name}::STRING)" for name in variant_columns]
            self.run(f"UPDATE {target}.{table.name} SET {', '.join(assignments)}")

    def grant_read(self, target: str) -> None:
        self.run(f"GRANT USAGE ON SCHEMA {target} TO ROLE {self._reader_role}")
        # Tables were just recreated, which drops earlier grants on them.
        self.run(f"GRANT SELECT ON ALL TABLES IN SCHEMA {target} TO ROLE {self._reader_role}")

    def close(self) -> None:
        self._connection.close()
