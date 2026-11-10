"""Snowflake and Databricks loaders against fake connections: the statements they send."""

from pathlib import Path
from types import TracebackType
from typing import Any, cast

import pytest
from databricks.sql.client import Connection as DatabricksConnection
from databricks.sql.exc import ServerOperationError
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import ProgrammingError

from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.load.databricks_loader import DatabricksLoader
from sqlbench.load.dataset import LoadError, load_dataset
from sqlbench.load.snowflake_loader import SnowflakeLoader
from sqlbench.warehouses.base import SessionContext

FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"


class RecordingCursor:
    def __init__(self, statements: list[str], failing: str | None, error: Exception) -> None:
        self.statements = statements
        self.failing = failing
        self.error = error

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if self.failing is not None and sql.startswith(self.failing):
            raise self.error


class RecordingConnection:
    """Records every statement; the one starting with failing raises error."""

    def __init__(self, failing: str | None = None, error: Exception | None = None) -> None:
        self.statements: list[str] = []
        self.failing = failing
        self.error = error or RuntimeError("unexpected")
        self.contexts: list[SessionContext] = []
        self.allowed_paths: list[str] = []
        self.closed = False

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self.statements, self.failing, self.error)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def generated(tmp_path: Path) -> Path:
    generate_dataset(FIXTURE, Profile.EDGE, 5, tmp_path / "data", tmp_path / "manifests")
    return tmp_path


@pytest.fixture(autouse=True)
def reader_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_READER_ROLE", "BENCH_READER")
    monkeypatch.setenv("DATABRICKS_READER_CLIENT_ID", "0000-reader")


def snowflake_loader(connection: RecordingConnection) -> SnowflakeLoader:
    def connect(context: SessionContext) -> SnowflakeConnection:
        connection.contexts.append(context)
        return cast(SnowflakeConnection, connection)

    return SnowflakeLoader(connect)


def load_edge(loader: Any, root: Path) -> None:
    load_dataset(loader, FIXTURE, Profile.EDGE, root / "data", root / "manifests")


def test_snowflake_sets_up_database_and_stage_once() -> None:
    connection = RecordingConnection()
    snowflake_loader(connection)

    assert connection.contexts == [SessionContext()]
    assert connection.statements == [
        "CREATE DATABASE IF NOT EXISTS sqlbench",
        "USE DATABASE sqlbench",
        "CREATE SCHEMA IF NOT EXISTS loading",
        "CREATE STAGE IF NOT EXISTS loading.files "
        "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE)",
        "GRANT USAGE ON DATABASE sqlbench TO ROLE BENCH_READER",
    ]


def test_snowflake_replaces_the_schema_then_loads_each_table(generated: Path) -> None:
    connection = RecordingConnection()
    loader = snowflake_loader(connection)
    setup_count = len(connection.statements)

    load_edge(loader, generated)

    statements = connection.statements[setup_count:]
    assert statements[0] == "CREATE OR REPLACE SCHEMA mini_shop__edge"
    creates = [sql for sql in statements if sql.startswith("CREATE OR REPLACE TABLE")]
    assert [sql.split()[4] for sql in creates] == [
        "mini_shop__edge.customers",
        "mini_shop__edge.orders",
        "mini_shop__edge.events",
    ]
    put = next(sql for sql in statements if sql.startswith("PUT"))
    parquet = (generated / "data" / "edge" / "mini_shop" / "customers.parquet").resolve()
    assert put == (
        f"PUT 'file://{parquet}' @loading.files/mini_shop__edge/ "
        "OVERWRITE = TRUE AUTO_COMPRESS = FALSE"
    )
    assert (
        "COPY INTO mini_shop__edge.orders FROM @loading.files/mini_shop__edge/ "
        "FILES = ('orders.parquet') MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE FORCE = TRUE"
    ) in statements
    assert statements[-2:] == [
        "GRANT USAGE ON SCHEMA mini_shop__edge TO ROLE BENCH_READER",
        "GRANT SELECT ON ALL TABLES IN SCHEMA mini_shop__edge TO ROLE BENCH_READER",
    ]


def test_snowflake_parses_json_text_only_in_variant_columns(generated: Path) -> None:
    connection = RecordingConnection()
    load_edge(snowflake_loader(connection), generated)

    updates = [sql for sql in connection.statements if sql.startswith("UPDATE")]
    assert updates == [
        "UPDATE mini_shop__edge.orders SET attributes = PARSE_JSON(attributes::STRING)",
        "UPDATE mini_shop__edge.events SET payload = PARSE_JSON(payload::STRING)",
    ]


def test_snowflake_reload_sends_the_same_statements(generated: Path) -> None:
    connection = RecordingConnection()
    loader = snowflake_loader(connection)
    setup_count = len(connection.statements)
    load_edge(loader, generated)
    first = connection.statements[setup_count:]
    load_edge(loader, generated)

    assert connection.statements[setup_count + len(first) :] == first


def test_snowflake_errors_become_load_errors_naming_the_statement(generated: Path) -> None:
    denied = ProgrammingError(msg="Insufficient privileges", errno=3001, sqlstate="42501")
    connection = RecordingConnection(failing="GRANT SELECT", error=denied)
    loader = snowflake_loader(connection)

    with pytest.raises(LoadError, match="GRANT SELECT ON ALL TABLES IN SCHEMA mini_shop__edge"):
        load_edge(loader, generated)


def databricks_loader(connection: RecordingConnection, data_dir: Path) -> DatabricksLoader:
    def connect(allowed_local_path: str) -> DatabricksConnection:
        connection.allowed_paths.append(allowed_local_path)
        return cast(DatabricksConnection, connection)

    return DatabricksLoader(data_dir, connect)


def test_databricks_sets_up_catalog_and_volume_and_allows_only_data_dir(tmp_path: Path) -> None:
    connection = RecordingConnection()
    databricks_loader(connection, tmp_path / "data")

    assert connection.allowed_paths == [str((tmp_path / "data").resolve())]
    assert connection.statements == [
        "CREATE CATALOG IF NOT EXISTS sqlbench",
        "USE CATALOG sqlbench",
        "CREATE SCHEMA IF NOT EXISTS loading",
        "CREATE VOLUME IF NOT EXISTS loading.files",
        "GRANT USE CATALOG ON CATALOG sqlbench TO `0000-reader`",
    ]


def test_databricks_uploads_then_inserts_with_typed_casts(generated: Path) -> None:
    connection = RecordingConnection()
    loader = databricks_loader(connection, generated / "data")
    setup_count = len(connection.statements)

    load_edge(loader, generated)

    statements = connection.statements[setup_count:]
    assert statements[:2] == [
        "DROP SCHEMA IF EXISTS mini_shop__edge CASCADE",
        "CREATE SCHEMA mini_shop__edge",
    ]
    parquet = (generated / "data" / "edge" / "mini_shop" / "orders.parquet").resolve()
    volume_path = "/Volumes/sqlbench/loading/files/mini_shop__edge/orders.parquet"
    assert f"PUT '{parquet}' INTO '{volume_path}' OVERWRITE" in statements
    insert = next(sql for sql in statements if sql.startswith("INSERT INTO mini_shop__edge.orders"))
    assert "parse_json(attributes)" in insert
    assert "CAST(specs AS MAP<STRING, STRING>)" in insert
    assert "CAST(total AS DECIMAL(10,2))" in insert
    assert insert.endswith(f"FROM parquet.`{volume_path}`")
    assert statements[-1] == "GRANT USE SCHEMA, SELECT ON SCHEMA mini_shop__edge TO `0000-reader`"


def test_databricks_creates_each_table_before_inserting(generated: Path) -> None:
    connection = RecordingConnection()
    load_edge(databricks_loader(connection, generated / "data"), generated)

    order: list[str] = []
    for sql in connection.statements:
        if sql.startswith(("CREATE OR REPLACE TABLE", "INSERT INTO")):
            order.append(" ".join(sql.split()[:5]))
    assert order == [
        "CREATE OR REPLACE TABLE mini_shop__edge.customers",
        "INSERT INTO mini_shop__edge.customers (customer_id, full_name,",
        "CREATE OR REPLACE TABLE mini_shop__edge.orders",
        "INSERT INTO mini_shop__edge.orders (order_id, customer_id,",
        "CREATE OR REPLACE TABLE mini_shop__edge.events",
        "INSERT INTO mini_shop__edge.events (customer_id, event_ts,",
    ]


def test_databricks_errors_become_load_errors(tmp_path: Path) -> None:
    denied = ServerOperationError("[PERMISSION_DENIED] no CREATE CATALOG. SQLSTATE: 42501", {})
    connection = RecordingConnection(failing="CREATE CATALOG", error=denied)

    with pytest.raises(LoadError, match="Databricks rejected: CREATE CATALOG"):
        databricks_loader(connection, tmp_path)
