"""Load Parquet into Databricks: PUT to a Unity Catalog volume, then INSERT ... SELECT.

The volume lives in its own schema, sqlbench.loading, so dropping a data
schema never touches it, and the read-only principal is never granted on it.
"""

from collections.abc import Callable
from pathlib import Path

from databricks.sql.client import Connection as DatabricksConnection
from databricks.sql.exc import Error as DatabricksError

from sqlbench.load.dataset import DATABASE, LoadError
from sqlbench.schema.ddl import create_table, databricks_type, self_reference_constraints
from sqlbench.schema.spec import TableSpec
from sqlbench.schema.types import TypeKind
from sqlbench.settings import require_env
from sqlbench.warehouses.base import SessionContext, WarehouseName
from sqlbench.warehouses.databricks_warehouse import connect_owner_from_env

VOLUME = "loading.files"
VOLUME_ROOT = f"/Volumes/{DATABASE}/loading/files"


def connect_for_loading(allowed_local_path: str) -> DatabricksConnection:
    # The connector refuses to PUT files from outside staging_allowed_local_path.
    return connect_owner_from_env(SessionContext(), staging_allowed_local_path=allowed_local_path)


def select_expression(name: str, logical_kind: TypeKind, target_type: str) -> str:
    if logical_kind == TypeKind.VARIANT:
        # JSON text becomes VARIANT; the text 'null' is a VARIANT null, unlike SQL NULL.
        return f"parse_json({name})"
    return f"CAST({name} AS {target_type})"


class DatabricksLoader:
    """Loads with the owner's token and grants SELECT to the read-only principal."""

    name = WarehouseName.DATABRICKS

    def __init__(
        self,
        data_dir: Path,
        connect: Callable[[str], DatabricksConnection] = connect_for_loading,
    ) -> None:
        env = require_env(["DATABRICKS_READER_CLIENT_ID"])
        # Service principals are named by their application (client) id in grants.
        self._reader = env["DATABRICKS_READER_CLIENT_ID"]
        self._connection = connect(str(data_dir.resolve()))
        self.run(f"CREATE CATALOG IF NOT EXISTS {DATABASE}")
        self.run(f"USE CATALOG {DATABASE}")
        self.run("CREATE SCHEMA IF NOT EXISTS loading")
        self.run(f"CREATE VOLUME IF NOT EXISTS {VOLUME}")
        self.run(f"GRANT USE CATALOG ON CATALOG {DATABASE} TO `{self._reader}`")

    def run(self, sql: str) -> None:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
        except DatabricksError as error:
            raise LoadError(f"Databricks rejected: {sql.splitlines()[0]}\n{error}") from error

    def reset_schema(self, target: str) -> None:
        self.run(f"DROP SCHEMA IF EXISTS {target} CASCADE")
        self.run(f"CREATE SCHEMA {target}")

    def load_table(self, target: str, table: TableSpec, parquet_path: Path) -> None:
        volume_path = f"{VOLUME_ROOT}/{target}/{parquet_path.name}"
        self.run(f"PUT '{parquet_path.resolve()}' INTO '{volume_path}' OVERWRITE")
        # CREATE TABLE then INSERT, instead of CREATE TABLE AS SELECT, so the
        # NOT NULL and key constraints from the DDL emitter are kept.
        self.run(create_table(table, databricks_type, target))
        for statement in self_reference_constraints(table, target):
            self.run(statement)
        names = []
        expressions = []
        for column in table.columns:
            names.append(column.name)
            target_type = databricks_type(column.logical_type)
            expressions.append(
                select_expression(column.name, column.logical_type.kind, target_type)
            )
        self.run(
            f"INSERT INTO {target}.{table.name} ({', '.join(names)}) "
            f"SELECT {', '.join(expressions)} FROM parquet.`{volume_path}`"
        )

    def grant_read(self, target: str) -> None:
        # Granting on the schema covers its tables, including ones created later.
        self.run(f"GRANT USE SCHEMA, SELECT ON SCHEMA {target} TO `{self._reader}`")

    def close(self) -> None:
        self._connection.close()
